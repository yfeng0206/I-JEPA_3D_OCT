#!/usr/bin/env python3
"""Stage 3 of the pipeline: train the cfg-7 adapter once, then freeze it.

    JEPA EMA target encoder  --L_rel-->  cfg-7 adapter  -->  changed guide

This is the JEPA -> MIRAGE direction.  The original MIRAGE stays frozen; a
small residual adapter sits on its decoder feature and the FROZEN
segmentation head reads the ADAPTED feature, so the head's own weights never
move but its output does.

    H     = H0 + alpha * tanh(A(H0))
    L     = FrozenSegHead(H)
    L_rel = MSE( Gram(pool(H)), sg(Gram(Z_ema)) )

Wiring note that cost a lot of time to establish: routing the adapter into a
SEPARATE residual logit head, while the frozen head still reads the
unadapted H0, is a structural dead end.  Without a labelled segmentation
loss that head receives exactly 0.000e+00 gradient, and both segmentation
agreement and mask Jaccard come out at exactly 1.000000.  The frozen head
must read the adapted feature.

Run ONCE after JEPA warmup.  Measurement showed the adapter saturates after
roughly 2,400 images against a fixed teacher (27.7% of the achievable L_rel
reduction at 2.4k, 30.7% at 4.8k, 32.5% at 19.2k), so there is no reason to
train it across 600,000 images or to refresh it every epoch.

Outputs a checkpoint containing the adapter weights plus the fingerprint of
the JEPA checkpoint that taught it, so the guide cache built afterwards can
record which teacher it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from jepa_to_mirage_probe import build_mirage, build_jepa    # noqa: E402

OUT = REPO / 'results/masking/adapter_stage'
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID, ANATOMY = 16, (1, 2)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
                               nn.Conv2d(c, c, 3, padding=1))
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.gelu(self.n(x + self.b(x)))


class Adapter(nn.Module):
    """cfg 7: depth 2, width 128, alpha 0.5.  Zero-init => identity at step 0."""

    def __init__(self, depth=2, width=128, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        layers = [nn.Conv2d(384, width, 1), nn.GELU()]
        layers += [ResBlock(width) for _ in range(depth)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h0):
        return h0 + self.alpha * torch.tanh(self.out(self.trunk(h0)))


def gram(x):
    x = F.normalize(x.float(), dim=-1)
    return x @ x.transpose(1, 2)


def sha(path, n=None):
    """SHA-256 of a file, streamed in full.

    This used to hash only the first 1 MiB, which is not a safe identity for
    multi-megabyte checkpoints: two adapters differing only after the first
    mebibyte collide, and a colliding digest silently reuses a stale 3.85 GiB
    guide cache. ``n`` is retained for callers that deliberately want a partial
    digest, but the default is now the whole file.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if n is not None:
            h.update(f.read(n))
        else:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
    return h.hexdigest()[:16]


def load_pairs(cache, n):
    im512 = np.load(cache / 'im512.npy', mmap_mode='r')
    im256 = np.load(cache / 'im256.npy', mmap_mode='r')
    n = min(n, len(im512))
    return im512, im256, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa-ckpt', required=True,
                    help='JEPA checkpoint whose EMA TARGET encoder teaches the adapter')
    ap.add_argument('--cache', default=r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
    ap.add_argument('--n-train', type=int, default=4800)
    ap.add_argument('--n-eval', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    cache = pathlib.Path(a.cache)
    im512, im256, _ = load_pairs(cache, a.n_train + a.n_eval)

    rng = np.random.default_rng(0)
    perm = rng.permutation(a.n_train + a.n_eval)
    idx_ev = np.sort(perm[:a.n_eval])
    idx_tr = np.sort(perm[a.n_eval:a.n_eval + a.n_train])

    mir = build_mirage(dev)
    assert not any(p.requires_grad for p in mir.parameters()), 'MIRAGE not frozen'
    grab = {}
    head = mir.output_adapters['semseg'].final_layer
    head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))
    for p in head.parameters():
        assert not p.requires_grad, 'seg head must stay frozen'

    enc = build_jepa(pathlib.Path(a.jepa_ckpt), dev)
    for p in enc.parameters():
        p.requires_grad_(False)

    def h0(idx):
        x = torch.from_numpy(np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x})
        return grab['H'].float()

    def rj(idx):
        b = np.asarray(im256[idx], np.float32) / 255.
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(dev)
        with torch.no_grad():
            z = enc(x)
            z = F.layer_norm(z, (z.size(-1),))
            return gram(z)                      # already detached: no_grad

    torch.manual_seed(0)
    mod = Adapter(a.depth, a.width, a.alpha).to(dev)
    nst = (len(idx_tr) + a.batch - 1) // a.batch
    opt = torch.optim.AdamW(mod.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=nst, pct_start=0.1)

    def evaluate():
        mod.eval()
        r0, r1, dr, ag, ch = [], [], [], [], []
        with torch.no_grad():
            for s in range(0, len(idx_ev), a.batch):
                i = np.sort(idx_ev[s:s + a.batch])
                H0 = h0(i); H = mod(H0)
                R = rj(i)
                U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
                U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
                r0.append(float(F.mse_loss(gram(U0), R)))
                r1.append(float(F.mse_loss(gram(U1), R)))
                dr.append(float((H - H0).norm() / H0.norm()))
                L0, Lf = head(H0), head(H)
                ag.append(float((Lf.argmax(1) == L0.argmax(1)).float().mean()))
                g0 = F.adaptive_avg_pool2d(L0.float().softmax(1)[:, ANATOMY], (GRID, GRID))
                g1 = F.adaptive_avg_pool2d(Lf.float().softmax(1)[:, ANATOMY], (GRID, GRID))
                ch.append(float((g1 - g0).abs().mean()))
        mod.train()
        b, c = float(np.mean(r0)), float(np.mean(r1))
        return {'rel_before': b, 'rel_after': c,
                'reduction_pct': 100 * (b - c) / b,
                'feature_drift': float(np.mean(dr)),
                'seg_agreement': float(np.mean(ag)),
                'score_change': float(np.mean(ch))}

    pre = evaluate()
    print('before training : L_rel %.6f  drift %.4f  seg-agree %.4f'
          % (pre['rel_before'], pre['feature_drift'], pre['seg_agreement']))
    assert pre['feature_drift'] < 1e-6, 'zero-init identity broken'
    assert abs(pre['seg_agreement'] - 1.0) < 1e-9, 'guide changed before training'
    print('  step-0 identity check PASSED (drift 0, seg agreement 1.0)')

    t0 = time.perf_counter()
    hist = []
    for s in range(0, len(idx_tr), a.batch):
        i = np.sort(idx_tr[s:s + a.batch])
        H = mod(h0(i))
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        loss = F.mse_loss(gram(U), rj(i))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append({'loss': float(loss), 'grad_norm': float(gn),
                     'lr': sched.get_last_lr()[0]})
    dt = time.perf_counter() - t0

    post = evaluate()
    print('after training  : L_rel %.6f -> %.6f   reduction %.2f%%'
          % (post['rel_before'], post['rel_after'], post['reduction_pct']))
    print('  feature drift  %.4f' % post['feature_drift'])
    print('  seg agreement  %.4f   (change metric, NOT accuracy)' % post['seg_agreement'])
    print('  score change   %.4f' % post['score_change'])
    print('  %d steps in %.0fs' % (len(hist), dt))

    out = pathlib.Path(a.out or (OUT / 'adapter_cfg7.pt'))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': mod.state_dict(),
                'cfg': {'depth': a.depth, 'width': a.width, 'alpha': a.alpha},
                'jepa_ckpt': str(a.jepa_ckpt),
                'jepa_sha': sha(a.jepa_ckpt),
                'n_train': int(a.n_train), 'lr': a.lr}, out)
    (OUT / 'adapter_stage.json').write_text(json.dumps(
        {'before': pre, 'after': post, 'steps': len(hist), 'seconds': dt,
         'jepa_ckpt': str(a.jepa_ckpt), 'jepa_sha': sha(a.jepa_ckpt),
         'history': hist[::10]}, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
