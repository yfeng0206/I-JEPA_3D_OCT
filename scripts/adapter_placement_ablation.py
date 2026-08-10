#!/usr/bin/env python
"""Where should the JEPA adapter be inserted into frozen MIRAGE?

The current adapter sits on H0, the last feature map before the segmentation
head, so only a 1,540-parameter 1x1 conv sees the perturbation.  Measurement
against GOALS ground truth shows it does not help and at alpha=0.5 it
significantly harms Dice.  Two hypotheses for why:

  (i)  tap point -- H0 is one linear map from the logits, so any perturbation
       lands directly on the class scores with nothing downstream to absorb it
  (ii) objective  -- MIRAGE (MultiMAE / MIM) and I-JEPA (joint-embedding
       predictive) simply encode different similarity structure, so no
       placement can reconcile them

This ablation separates them.  Three placements, everything else held fixed --
same teacher, same images, same steps, same alpha, same trunk shape:

  enc   pre proj_dec       (B, 256, 768)     shape-matched to JEPA
  mid   pre ConvNeXt blocks (B, 384, 64, 64)  whole decoder downstream
  h0    pre final_layer     (B, 384, 64, 64)  CURRENT -- 1 conv downstream

Training is identical for all three: capture the representation at the tap
under no_grad, then train the adapter on it.  L_rel needs no gradient through
the decoder because it is evaluated at the tap.

Evaluation differs: a forward pre-hook injects the adapter into MIRAGE's own
forward, so the frozen decoder actually processes the perturbed features and
the GOALS Dice reflects the true downstream cost.

Reports, per placement:
  gram_r_before   agreement with JEPA before training -- how far L_rel must move
  L_rel reduction how much of its objective the adapter achieved
  GOALS Dice      the only ground-truth metric; delta vs frozen MIRAGE
  amplification   (|dH0|/|H0|) / (|dZ|/|Z|) -- does the decoder magnify it?
  mask Jaccard    how much the masking guide actually relocates
"""
from __future__ import annotations

import argparse
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

from adapter_stage import ResBlock                                  # noqa: E402
from goals_eval import load_pairs, dice_iou, VOID, RES              # noqa: E402
from jepa_to_mirage_probe import (build_mirage, build_jepa,         # noqa: E402
                                  IMNET_MEAN, IMNET_STD)

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\adapter_sweep')
OUT = REPO / 'results/masking/placement'
GRID = 16
ANATOMY = [1, 2]
OCC_THRESHOLD = 0.25


class Adapter(nn.Module):
    """Zero-init residual: identity at step 0, bounded by alpha*tanh."""

    def __init__(self, ch, depth=2, width=128, alpha=0.5):
        super().__init__()
        layers = [nn.Conv2d(ch, width, 1), nn.GELU()]
        layers += [ResBlock(width) for _ in range(depth)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.alpha = alpha

    def forward(self, x):
        # Tokens (B,N,C) are reshaped to a square map so one trunk serves both
        # the encoder tap and the two decoder taps.
        tok = x.dim() == 3
        if tok:
            b, n, c = x.shape
            g = int(round(n ** 0.5))
            x = x.transpose(1, 2).reshape(b, c, g, g)
        y = x + self.alpha * torch.tanh(self.out(self.trunk(x)))
        if tok:
            y = y.flatten(2).transpose(1, 2)
        return y


def gram(x):
    x = F.normalize(x.float(), dim=-1)
    return x @ x.transpose(1, 2)


def to_tokens(x):
    """Any tap representation -> (B, 256, C) on the 16x16 grid."""
    if x.dim() == 3:
        return x
    return F.adaptive_avg_pool2d(x, (GRID, GRID)).flatten(2).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=4800)
    ap.add_argument('--n-eval', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--alphas', default=None,
                    help='comma-separated alphas; sweeps the transfer/damage '
                         'trade-off so taps can be compared at MATCHED L_rel '
                         'reduction rather than at matched alpha')
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--jepa-ckpt', default=str(
        r'D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar'))
    ap.add_argument('--tag', default='placement')
    ap.add_argument('--taps', default=None,
                    help='comma-separated subset of enc,mid,h0')
    ap.add_argument('--save-adapters', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'

    mir = build_mirage(dev)
    enc = build_jepa(pathlib.Path(a.jepa_ckpt), dev)
    for p in enc.parameters():
        p.requires_grad_(False)
    sem = mir.output_adapters['semseg']

    ALL_TAPS = {
        'enc': (sem.proj_dec, 768),
        'mid': (sem.blocks, 384),
        'h0': (sem.final_layer, 384),
    }
    # Hooks are always registered for every tap: predict() reads grab['h0'] to
    # measure drift even when h0 is not one of the taps being swept.
    taps = dict(ALL_TAPS)
    if a.taps:
        want = [t.strip() for t in a.taps.split(',')]
        taps = {k: v for k, v in ALL_TAPS.items() if k in want}

    grab = {}
    for nm, (mod, _) in ALL_TAPS.items():
        mod.register_forward_hook(
            lambda m, i, o, nm=nm: grab.update({nm: i[0].detach()}))

    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    perm = rng.permutation(a.n_train + a.n_eval)
    idx_ev = np.sort(perm[:a.n_eval])
    idx_tr = np.sort(perm[a.n_eval:])

    def mirage_taps(idx):
        x = torch.from_numpy(np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x})
        return {k: grab[k].float() for k in taps}

    def jepa_gram(idx):
        b = np.asarray(im256[idx], np.float32) / 255.
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2).astype(np.float32)).to(dev)
        with torch.no_grad():
            return gram(F.layer_norm(enc(x), (768,)))

    print('teacher   %s' % pathlib.Path(a.jepa_ckpt).name)
    print('adapter   depth %d width %d alpha %.2f lr %.0e'
          % (a.depth, a.width, a.alpha, a.lr))
    print('data      %d train / %d eval FairVision slices' % (len(idx_tr), len(idx_ev)))

    # JEPA Gram is the same target for every placement; compute it once.
    t0 = time.perf_counter()
    RJ = {}
    for split in (idx_tr, idx_ev):
        for s in range(0, len(split), a.batch):
            i = np.sort(split[s:s + a.batch])
            RJ[i.tobytes()] = jepa_gram(i).half().cpu()
    print('JEPA grams precomputed in %.0fs\n' % (time.perf_counter() - t0))

    g_imgs, g_gts, _ = load_pairs()

    def predict(adapter=None, tap=None):
        """Run MIRAGE's own forward, optionally injecting the adapter."""
        handle = None
        if adapter is not None:
            mod = taps[tap][0]

            def pre(m, args):
                x = args[0]
                y = adapter(x.float()).to(x.dtype)
                return (y,) + args[1:]

            handle = mod.register_forward_pre_hook(pre)
        out, drift, taprep = [], [], []
        try:
            for s in range(0, len(g_imgs), 8):
                x = torch.from_numpy(g_imgs[s:s + 8])[:, None].to(dev)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    o = mir({'bscan': x})
                lg = (o['semseg'] if isinstance(o, dict) else o).float()
                if lg.shape[-1] != RES:
                    lg = F.interpolate(lg, size=(RES, RES), mode='bilinear',
                                       align_corners=False)
                lg[:, VOID] = float('-inf')
                out.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
                drift.append(grab['h0'].float().clone())
                # Pre-hooks replace the module input before forward hooks fire,
                # so this is the ADAPTED representation when one is injected.
                if tap is not None:
                    taprep.append(grab[tap].float().clone())
        finally:
            if handle is not None:
                handle.remove()
        return (np.concatenate(out), torch.cat(drift),
                torch.cat(taprep) if taprep else None)

    def dice(pred):
        per = []
        for i in range(len(g_gts)):
            ds = [dice_iou(pred[i], g_gts[i], c)[0] for c in ANATOMY]
            per.append(np.mean([d for d in ds if d is not None]))
        return float(np.mean(per)), np.array(per)

    def guide(pred):
        """The 16x16 anatomy occupancy grid the masker actually consumes."""
        m = torch.from_numpy(np.isin(pred, ANATOMY).astype(np.float32))
        return (F.adaptive_avg_pool2d(m[:, None], (GRID, GRID))[:, 0]
                >= OCC_THRESHOLD)

    base_pred, base_h0, _ = predict()
    base_taps = {}
    for t in ALL_TAPS:
        base_taps[t] = predict(None, t)[2]
    base_dice, base_per = dice(base_pred)
    base_guide = guide(base_pred)
    print('frozen MIRAGE  GOALS anatomy Dice %.4f\n' % base_dice)

    results = {'frozen_dice': base_dice, 'teacher': str(a.jepa_ckpt),
               'cfg': {'depth': a.depth, 'width': a.width, 'alpha': a.alpha,
                       'lr': a.lr, 'n_train': int(a.n_train)},
               'placements': {}}

    hdr = ('%-6s %6s %8s %8s %10s %9s %9s %8s %8s' %
           ('tap', 'alpha', 'params', 'gram r0', 'L_rel red', 'Dice',
            'vs frozen', 'amplif', 'mask J'))
    print(hdr)
    print('-' * len(hdr))
    alphas = ([float(v) for v in a.alphas.split(',')] if a.alphas
              else [a.alpha])

    for tap, (_, ch) in taps.items():
        # MIRAGE's forward dominates runtime and does not depend on alpha, so
        # every alpha for a tap is trained in ONE pass over the data.
        mods = {}
        for al in alphas:
            torch.manual_seed(0)
            m = Adapter(ch, a.depth, a.width, al).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
            nst = (len(idx_tr) + a.batch - 1) // a.batch
            sch = (torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=a.lr, total_steps=nst, pct_start=0.1)
                if nst >= 20 else None)
            mods[al] = (m, opt, sch)
        nparam = sum(p.numel() for p in mods[alphas[0]][0].parameters())

        # Held-out L_rel and Gram agreement BEFORE training (alpha-independent).
        b0, r0 = [], []
        iu = np.triu_indices(GRID * GRID, k=1)
        with torch.no_grad():
            for s in range(0, len(idx_ev), a.batch):
                i = np.sort(idx_ev[s:s + a.batch])
                z = mirage_taps(i)[tap]
                r = RJ[i.tobytes()].to(dev).float()
                b0.append(float(F.mse_loss(gram(to_tokens(z)), r)))
                gm = gram(to_tokens(z)).cpu().numpy()
                rj = RJ[i.tobytes()].float().numpy()
                r0.extend([np.corrcoef(gm[k][iu], rj[k][iu])[0, 1]
                           for k in range(len(gm))])
        rel_before = float(np.mean(b0))
        gram_r0 = float(np.mean(r0))

        for s in range(0, len(idx_tr), a.batch):
            i = np.sort(idx_tr[s:s + a.batch])
            z = mirage_taps(i)[tap]
            r = RJ[i.tobytes()].to(dev).float()
            for al in alphas:
                m, opt, sch = mods[al]
                loss = F.mse_loss(gram(to_tokens(m(z))), r)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
                opt.zero_grad(set_to_none=True)

        for al in alphas:
            ad = mods[al][0]
            ad.eval()
            after = []
            with torch.no_grad():
                for s in range(0, len(idx_ev), a.batch):
                    i = np.sort(idx_ev[s:s + a.batch])
                    z = mirage_taps(i)[tap]
                    r = RJ[i.tobytes()].to(dev).float()
                    after.append(float(F.mse_loss(gram(to_tokens(ad(z))), r)))
            rel_after = float(np.mean(after))
            red = 100 * (rel_before - rel_after) / rel_before

            pred, h0, tz = predict(ad, tap)
            d, per = dice(pred)
            j = guide(pred)
            inter = (j & base_guide).flatten(1).sum(1).float()
            union = (j | base_guide).flatten(1).sum(1).float().clamp(min=1)
            jac = float((inter / union).mean())

            # How much does the frozen decoder magnify the perturbation?
            # Numerator and denominator must come from the SAME images, or the
            # h0 tap fails to return its sanity value of exactly 1.0.
            dh0 = float((h0 - base_h0).norm() / base_h0.norm())
            dz = float((tz - base_taps[tap]).norm() / base_taps[tap].norm())
            amp = dh0 / dz if dz > 1e-9 else float('nan')

            from scipy import stats
            p = float(stats.ttest_rel(per, base_per).pvalue)
            if a.save_adapters:
                torch.save({'state_dict': ad.state_dict(),
                            'cfg': {'ch': ch, 'depth': a.depth,
                                    'width': a.width, 'alpha': al},
                            'tap': tap, 'jepa_ckpt': str(a.jepa_ckpt)},
                           OUT / ('adapter_%s_a%.2f.pt' % (tap, al)))
            print('%-6s %6.2f %8s %8.3f %9.2f%% %9.4f %+9.5f %8.2f %8.4f'
                  % (tap, al, '{:,}'.format(nparam), gram_r0, red, d,
                     d - base_dice, amp, jac))
            results['placements']['%s_a%.2f' % (tap, al)] = {
                'tap': tap, 'alpha': al, 'channels': ch, 'params': nparam,
                'gram_r_before': gram_r0, 'rel_before': rel_before,
                'rel_after': rel_after, 'rel_reduction_pct': red,
                'goals_dice': d, 'dice_delta': d - base_dice,
                'dice_p_paired_t': p, 'better_on': int((per > base_per).sum()),
                'n_images': len(per), 'amplification': amp,
                'delta_h0_rel': dh0, 'delta_tap_rel': dz,
                'mask_jaccard_vs_frozen': jac,
            }

    (OUT / ('%s.json' % a.tag)).write_text(json.dumps(results, indent=2))
    print('\nwrote', OUT / ('%s.json' % a.tag))


if __name__ == '__main__':
    main()
