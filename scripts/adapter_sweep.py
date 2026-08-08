#!/usr/bin/env python3
"""Adapter hyperparameter sweep on a REAL single pass over N images.

Earlier probes ran 400 optimiser steps on the SAME 24 slices, which is
memorisation of a tiny set.  This does one pass over N fresh images, which is
the regime the real run will be in.

Stage A  --cache    load N slices, run JEPA once, store images (uint8) and the
                    JEPA Gram targets (fp16).  H0 is NOT cacheable: at N=6000 it
                    would be 35 GB, so MIRAGE is re-run every pass.
Stage B  --sweep    one pass per config, streaming MIRAGE.
Stage C  --figure   before/after visuals for the chosen config.

Wiring under test (FairVision-only, no labels):
    H = H0 + alpha * tanh(A(H0))          bounded residual on FEATURES
    L = FrozenSegHead(H)                  frozen head sees the ADAPTED H
    L_rel = MSE(Gram(pool(H)), sg(Gram(h_full)))
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

from jepa_to_mirage_probe import build_mirage, build_jepa   # noqa: E402

TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
RUNS = pathlib.Path(r'D:\jepa_phase0\runs\patch_mirage_envelope')
CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\adapter_sweep')
OUT = REPO / 'results/masking/adapter_sweep'
RES, GRID, ANATOMY = 512, 16, (1, 2)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------- adapters
def make_adapter(depth, width, alpha):
    return Adapter(depth=depth, width=width, alpha=alpha)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
                               nn.Conv2d(c, c, 3, padding=1))
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.gelu(self.n(x + self.b(x)))


class Adapter(nn.Module):
    """H -> H + alpha*tanh(f(H)).  Zero-init output => identity at step 0.

    `depth` counts 3x3 residual blocks in the trunk; depth=0 reproduces the
    original shallow adapter (1x1, GELU, 3x3, GELU, 1x1).
    """

    def __init__(self, depth=0, width=64, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        layers = [nn.Conv2d(384, width, 1), nn.GELU()]
        if depth == 0:
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        else:
            layers += [ResBlock(width) for _ in range(depth)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h0):
        return h0 + self.alpha * torch.tanh(self.out(self.trunk(h0)))


def gram(x):
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)


# ------------------------------------------------------------------- stage A
def cache(n_images, epoch, batch):
    import cv2
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    vols = sorted(TRAIN.glob('data_*.npz'))[:n_images]
    print('caching %d images ...' % len(vols), flush=True)

    im512 = np.zeros((len(vols), RES, RES), np.uint8)
    im256 = np.zeros((len(vols), 256, 256), np.uint8)
    names = []
    t0 = time.perf_counter()
    for i, p in enumerate(vols):
        with np.load(p, allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = len(vol) // 2
        raw = np.asarray(vol[d], np.float32)
        lo, hi = raw.min(), raw.max()
        u = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        im512[i] = (cv2.resize(u, (RES, RES), interpolation=cv2.INTER_LINEAR)
                    * 255).astype(np.uint8)
        im256[i] = (cv2.resize(u, (256, 256), interpolation=cv2.INTER_LINEAR)
                    * 255).astype(np.uint8)
        names.append('%s:%d' % (p.stem, d))
        if (i + 1) % 1000 == 0:
            print('   %d/%d  %.1fs' % (i + 1, len(vols),
                                       time.perf_counter() - t0), flush=True)

    enc = build_jepa(RUNS / ('jepa_patch_mirage-ep%d.pth.tar' % epoch), dev)
    RJ = np.zeros((len(vols), 256, 256), np.float16)
    for s in range(0, len(vols), batch):
        b = im256[s:s + batch].astype(np.float32) / 255.
        rgb = np.repeat(b[..., None], 3, -1)
        rgb = (rgb - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(dev)
        with torch.no_grad():
            h = enc(x)
            h = F.layer_norm(h, (h.size(-1),))
            RJ[s:s + batch] = gram(h).cpu().numpy().astype(np.float16)
    del enc
    torch.cuda.empty_cache()

    mir = build_mirage(dev)
    fl = mir.output_adapters['semseg'].final_layer
    np.savez(CACHE / 'head.npz', w=fl.weight.detach().cpu().numpy(),
             b=fl.bias.detach().cpu().numpy())
    del mir
    torch.cuda.empty_cache()

    np.save(CACHE / 'im512.npy', im512)
    np.save(CACHE / 'im256.npy', im256)
    np.save(CACHE / 'RJ.npy', RJ)
    (CACHE / 'names.json').write_text(json.dumps(names))
    print('cached %d images in %.1fs -> %s' %
          (len(vols), time.perf_counter() - t0, CACHE))


# ------------------------------------------------------------------- stage B
def one_pass(mir, grab, head, im512, RJ, cfg, dev, batch, eval_idx, log_every=0):
    """Train the adapter for ONE pass; return metrics + the eval-subset outputs."""
    torch.manual_seed(0)
    mod = make_adapter(cfg['depth'], cfg['width'], cfg['alpha']).to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    n = im512.shape[0]
    nsteps = (n + batch - 1) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg['lr'], total_steps=nsteps, pct_start=0.1)
    losses = []
    t0 = time.perf_counter()
    for si, s in enumerate(range(0, n, batch)):
        xb = torch.from_numpy(im512[s:s + batch].astype(np.float32) / 255.)
        x = xb[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
            H0 = grab['H'].float()
        H = mod(H0)
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        rj = torch.from_numpy(RJ[s:s + batch].astype(np.float32)).to(dev)
        loss = F.mse_loss(gram(U), rj)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if log_every and (si + 1) % log_every == 0:
            print('      step %4d/%d  L_rel %.5f' % (si + 1, nsteps,
                                                     np.mean(losses[-log_every:])),
                  flush=True)
    train_time = time.perf_counter() - t0

    # ---- evaluate on a fixed held-out-ish subset ------------------------
    mod.eval()
    outs = {'H0': [], 'H': [], 'L0': [], 'Lf': [], 'rel0': [], 'rel1': []}
    with torch.no_grad():
        for s in range(0, len(eval_idx), batch):
            idx = eval_idx[s:s + batch]
            x = torch.from_numpy(im512[idx].astype(np.float32) / 255.)[:, None].to(dev)
            mir({'bscan': x})
            H0 = grab['H'].float()
            H = mod(H0)
            rj = torch.from_numpy(RJ[idx].astype(np.float32)).to(dev)
            U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
            U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            outs['rel0'].append(float(F.mse_loss(gram(U0), rj)))
            outs['rel1'].append(float(F.mse_loss(gram(U1), rj)))
            outs['H0'].append(H0.cpu()); outs['H'].append(H.cpu())
            outs['L0'].append(head(H0).cpu()); outs['Lf'].append(head(H).cpu())
    for k in ('H0', 'H', 'L0', 'Lf'):
        outs[k] = torch.cat(outs[k])
    outs['train_time'] = train_time
    outs['loss_curve'] = losses
    outs['params'] = sum(p.numel() for p in mod.parameters())
    outs['state'] = {k: v.cpu() for k, v in mod.state_dict().items()}
    return outs


def evaluate(outs):
    import anatomy_target_sampler_v2 as A
    H0, H, L0, Lf = outs['H0'], outs['H'], outs['L0'], outs['Lf']

    def masks(L):
        P = L.float().softmax(1)
        g = F.adaptive_avg_pool2d(P[:, ANATOMY], (GRID, GRID)).numpy()
        return np.array([np.logical_or.reduce(
            A.build_targets([g[i, 0], g[i, 1]], 4)[0]) for i in range(g.shape[0])])

    m0, m1 = masks(L0), masks(Lf)
    inter = (m0 & m1).sum(axis=(1, 2))
    uni = (m0 | m1).sum(axis=(1, 2)).clip(min=1)
    g0 = F.adaptive_avg_pool2d(L0.float().softmax(1)[:, ANATOMY], (GRID, GRID))
    g1 = F.adaptive_avg_pool2d(Lf.float().softmax(1)[:, ANATOMY], (GRID, GRID))
    rel0, rel1 = float(np.mean(outs['rel0'])), float(np.mean(outs['rel1']))
    return {
        'params': outs['params'],
        'train_time_s': outs['train_time'],
        'L_rel_before': rel0, 'L_rel_after': rel1,
        'L_rel_reduction_pct': 100 * (rel0 - rel1) / rel0,
        'loss_first100': float(np.mean(outs['loss_curve'][:100])),
        'loss_last100': float(np.mean(outs['loss_curve'][-100:])),
        'feature_drift': float((H - H0).norm() / H0.norm()),
        'seg_agreement': float((Lf.argmax(1) == L0.argmax(1)).float().mean()),
        'mask_jaccard': float((inter / uni).mean()),
        'mask_cells_before': float(m0.sum(axis=(1, 2)).mean()),
        'mask_cells_after': float(m1.sum(axis=(1, 2)).mean()),
        'anatomy_mean_abs_change': float((g1 - g0).abs().mean()),
        'anatomy_max_abs_change': float((g1 - g0).abs().max()),
    }, m0, m1


def sweep(n_images, batch, n_eval, grid_json, out_dir):
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')[:n_images]
    RJ = np.load(CACHE / 'RJ.npy', mmap_mode='r')[:n_images]
    hd = np.load(CACHE / 'head.npz')
    head = nn.Conv2d(384, 4, 1).to(dev)
    head.weight.data = torch.tensor(hd['w']).to(dev)
    head.bias.data = torch.tensor(hd['b']).to(dev)
    for p in head.parameters():
        p.requires_grad_(False)

    mir = build_mirage(dev)
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach()))

    rng = np.random.default_rng(0)
    eval_idx = np.sort(rng.choice(n_images, size=min(n_eval, n_images),
                                  replace=False))

    configs = json.loads(pathlib.Path(grid_json).read_text()) if grid_json else [
        dict(depth=d, width=w, lr=lr, alpha=al)
        for d, w in ((0, 64), (2, 128), (4, 128))
        for lr in (1e-4, 1e-3)
        for al in (0.25, 0.5)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print('\n%d configs x %d images (batch %d, %d steps/pass)'
          % (len(configs), n_images, batch, (n_images + batch - 1)//batch),
          flush=True)
    hdr = ('%-5s%-7s%-7s%-8s%-9s%9s%9s%9s%10s%9s%8s'
           % ('cfg', 'depth', 'width', 'lr', 'alpha', 'params',
              'Lrel dn%', 'drift', 'segagree', 'maskJac', 'sec'))
    print('\n' + hdr); print('-' * len(hdr), flush=True)
    best = None
    for ci, cfg in enumerate(configs):
        outs = one_pass(mir, grab, head, im512, RJ, cfg, dev, batch, eval_idx)
        met, m0, m1 = evaluate(outs)
        row = dict(cfg=cfg, **met)
        results.append(row)
        print('%-5d%-7d%-7d%-8.0e%-9.2f%9d%9.1f%9.4f%10.4f%9.4f%8.0f'
              % (ci, cfg['depth'], cfg['width'], cfg['lr'], cfg['alpha'],
                 met['params'], met['L_rel_reduction_pct'], met['feature_drift'],
                 met['seg_agreement'], met['mask_jaccard'], met['train_time_s']),
              flush=True)
        if best is None or met['L_rel_reduction_pct'] > best[1]['L_rel_reduction_pct']:
            best = (ci, met, outs, m0, m1)
        torch.save(outs['state'], out_dir / ('adapter_cfg%d.pt' % ci))
    (out_dir / 'sweep.json').write_text(json.dumps(
        {'n_images': n_images, 'batch': batch, 'n_eval': len(eval_idx),
         'results': results}, indent=2))
    np.savez_compressed(out_dir / 'best_eval.npz',
                        ci=best[0], eval_idx=eval_idx,
                        L0=best[2]['L0'].numpy(), Lf=best[2]['Lf'].numpy(),
                        m0=best[3], m1=best[4],
                        loss=np.array(best[2]['loss_curve'], np.float32))
    print('\nwrote %s' % (out_dir / 'sweep.json'))


# ------------------------------------------------------------------- stage C
def figure(out_dir, n_show):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rep = json.loads((out_dir / 'sweep.json').read_text())
    z = np.load(out_dir / 'best_eval.npz')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')
    idx = z['eval_idx']
    L0 = torch.tensor(z['L0']); Lf = torch.tensor(z['Lf'])
    m0, m1 = z['m0'], z['m1']
    g0 = F.adaptive_avg_pool2d(L0.float().softmax(1)[:, ANATOMY], (GRID, GRID)).numpy()
    g1 = F.adaptive_avg_pool2d(Lf.float().softmax(1)[:, ANATOMY], (GRID, GRID)).numpy()
    R = rep['results']
    ci = int(z['ci'])

    fig = plt.figure(figsize=(23, 13.5))
    gs = fig.add_gridspec(4, n_show + 2, hspace=.33, wspace=.2,
                          left=.045, right=.985, top=.9, bottom=.05)

    ax = fig.add_subplot(gs[0, :2])
    L = z['loss']
    k = max(1, len(L) // 120)
    ax.plot(np.arange(len(L))[::k], L[::k], lw=1, alpha=.35, color='#2a9d8f')
    w = max(1, len(L) // 40)
    sm = np.convolve(L, np.ones(w) / w, 'valid')
    ax.plot(np.arange(len(sm)), sm, lw=2.2, color='#264653')
    ax.set_xlabel('step (one pass, fresh images)'); ax.set_ylabel(r'$L_{rel}$')
    ax.set_title('best config #%d : single-pass loss' % ci, fontsize=11)
    ax.grid(alpha=.25)

    for name, key, c in (('L_rel reduced (%)', 'L_rel_reduction_pct', 0),
                         ('mask Jaccard', 'mask_jaccard', 1),
                         ('seg agreement', 'seg_agreement', 2)):
        ax = fig.add_subplot(gs[0, 2 + c])
        lab = ['d%d/w%d\n%.0e/a%.2f' % (r['cfg']['depth'], r['cfg']['width'],
                                        r['cfg']['lr'], r['cfg']['alpha'])
               for r in R]
        v = [r[key] for r in R]
        col = ['#e76f51' if i == ci else '#8d99ae' for i in range(len(R))]
        ax.bar(range(len(R)), v, color=col)
        ax.set_xticks(range(len(R)))
        ax.set_xticklabels(lab, fontsize=6, rotation=90)
        ax.set_title(name, fontsize=11); ax.grid(axis='y', alpha=.25)

    ax = fig.add_subplot(gs[0, 5:])
    ax.scatter([r['L_rel_reduction_pct'] for r in R],
               [r['mask_jaccard'] for r in R],
               s=[18 + r['params'] / 8000 for r in R],
               c=[r['cfg']['alpha'] for r in R], cmap='viridis')
    ax.set_xlabel('L_rel reduced (%)'); ax.set_ylabel('mask Jaccard')
    ax.set_title('the trade-off\n(size = params, colour = alpha)', fontsize=11)
    ax.grid(alpha=.25)

    for j in range(n_show):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(im256[idx[j]], cmap='gray')
        ax.set_title('input  %d' % idx[j], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(im256[idx[j]], cmap='gray')
        ov = np.zeros((256, 256, 4))
        ov[np.kron(m0[j], np.ones((16, 16))).astype(bool)] = (.95, .15, .15, .5)
        ax.imshow(ov)
        ax.set_title('BEFORE  %d cells' % m0[j].sum(), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[3, j])
        ax.imshow(im256[idx[j]], cmap='gray')
        ov = np.zeros((256, 256, 4))
        ov[np.kron(m1[j], np.ones((16, 16))).astype(bool)] = (.15, .55, .95, .5)
        ax.imshow(ov)
        inter = (m0[j] & m1[j]).sum(); uni = max((m0[j] | m1[j]).sum(), 1)
        ax.set_title('AFTER  %d cells   J=%.2f' % (m1[j].sum(), inter / uni),
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for r, (g, t) in enumerate(((g0, 'anatomy BEFORE'), (g1, 'anatomy AFTER'))):
        ax = fig.add_subplot(gs[1 + r, n_show])
        ax.imshow(g[0].sum(0), cmap='magma', vmin=0, vmax=1)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[3, n_show])
    d = (g1[0].sum(0) - g0[0].sum(0))
    v = max(abs(d).max(), 1e-8)
    ax.imshow(d, cmap='bwr', vmin=-v, vmax=v)
    ax.set_title('change  max %.3f' % v, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1:, n_show + 1])
    ax.axis('off')
    b = R[ci]
    txt = ['BEST CONFIG  #%d' % ci, '',
           'depth   %d' % b['cfg']['depth'],
           'width   %d' % b['cfg']['width'],
           'lr      %.0e' % b['cfg']['lr'],
           'alpha   %.2f' % b['cfg']['alpha'],
           'params  %d' % b['params'], '',
           'L_rel  %.5f -> %.5f' % (b['L_rel_before'], b['L_rel_after']),
           '  reduced %.1f%%' % b['L_rel_reduction_pct'], '',
           'feature drift   %.4f' % b['feature_drift'],
           'seg agreement   %.4f' % b['seg_agreement'],
           'mask Jaccard    %.4f' % b['mask_jaccard'],
           'mask cells  %.1f -> %.1f' % (b['mask_cells_before'],
                                         b['mask_cells_after']),
           'anatomy |change| mean %.4f' % b['anatomy_mean_abs_change'],
           '                 max %.4f' % b['anatomy_max_abs_change'], '',
           'pass time  %.0f s' % b['train_time_s'],
           'images     %d' % rep['n_images'],
           'eval on    %d' % rep['n_eval']]
    ax.text(0, 1, '\n'.join(txt), va='top', family='monospace', fontsize=9.5,
            linespacing=1.55)

    fig.suptitle('Adapter sweep: ONE pass over %d FairVision images, batch %d   '
                 '(H = H0 + alpha*tanh(A(H0)),  frozen seg head on adapted H,  '
                 'L_rel only, no labels)' % (rep['n_images'], rep['batch']),
                 fontsize=13, y=.955)
    f = out_dir / 'adapter_sweep.png'
    fig.savefig(f, dpi=110, facecolor='white')
    print('wrote %s' % f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--figure', action='store_true')
    ap.add_argument('--n-images', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--n-eval', type=int, default=64)
    ap.add_argument('--n-show', type=int, default=5)
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--grid', type=str, default=None)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    if a.cache:
        return cache(a.n_images, a.epoch, a.batch)
    if a.sweep:
        return sweep(a.n_images, a.batch, a.n_eval, a.grid, a.out)
    if a.figure:
        return figure(a.out, a.n_show)
    ap.error('need --cache, --sweep or --figure')


if __name__ == '__main__':
    main()
