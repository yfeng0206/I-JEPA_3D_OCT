#!/usr/bin/env python3
"""Slice-position re-validation cache for the cfg-7 adapter.

Every adapter number we have (L_rel reduction 29.9%, drift 0.185, mask
Jaccard 0.745) came from `adapter_sweep.py`, which takes ONE slice per
volume:

    d = len(vol) // 2          # the middle B-scan

Real training does not do that.  `OCTSliceDataset` uses

    slice_indices = np.linspace(0, 199, num=num_slices)

so at num_slices=100 it spans the whole volume, positions 0..199.  Middle
B-scans cut through the optic nerve head and are the easiest, cleanest
scans in the volume; peripheral ones are thinner, noisier and have no ONH.
So cfg 7 is currently validated on 0.5% of what it will actually see, and
that 0.5% is the most favourable slice in each volume.

This builds a matched cache with one RANDOM linspace position per volume
so the two can be compared like for like.  Everything is written through
np.lib.format.open_memmap, so peak RAM stays at one volume (7.7 MB) rather
than the full 2.4 GB of arrays.

Stage --build    read volumes, write im512/im256/pos, run JEPA for the
                 Gram targets RJ.  H0 is deliberately NOT cached: at
                 (384,64,64) fp16 it is 3.00 MiB/sample, so 6000 samples
                 would be 17.6 GiB.  MIRAGE is streamed instead.
Stage --compare  train cfg 7 on middle slices and on stratified slices
                 under an identical budget and compare.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_sweep import Adapter, gram, IMNET_MEAN, IMNET_STD   # noqa: E402
from jepa_to_mirage_probe import build_mirage, build_jepa        # noqa: E402

TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
RUNS = pathlib.Path(r'D:\jepa_phase0\runs\patch_mirage_envelope')
MID = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\adapter_sweep')
OUT_CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
OUT = REPO / 'results/masking/slice_pos'
RES, GRID, ANATOMY = 512, 16, (1, 2)
NUM_SLICES = 100          # must match configs/patch_mirage_envelope.yaml


def slice_grid():
    """The exact depth indices OCTSliceDataset samples."""
    return np.linspace(0, 199, num=NUM_SLICES, dtype=np.int64)


# --------------------------------------------------------------- stage build
def build(n_images, epoch, batch, seed):
    import cv2
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    vols = sorted(TRAIN.glob('data_*.npz'))[:n_images]
    n = len(vols)
    grid = slice_grid()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(grid), size=n)
    pos = grid[pick]

    est = n * (RES * RES + 256 * 256 + 256 * 256 * 2) / 1024 ** 3
    print('volumes           %d' % n)
    print('slice positions   %d distinct of %d, range %d..%d, mean %.1f'
          % (len(np.unique(pos)), len(grid), pos.min(), pos.max(), pos.mean()))
    print('estimated on-disk %.2f GiB   (H0 deliberately not cached: '
          '%.1f GiB)' % (est, n * 3.0 / 1024))
    sys.stdout.flush()

    im512 = np.lib.format.open_memmap(OUT_CACHE / 'im512.npy', mode='w+',
                                      dtype=np.uint8, shape=(n, RES, RES))
    im256 = np.lib.format.open_memmap(OUT_CACHE / 'im256.npy', mode='w+',
                                      dtype=np.uint8, shape=(n, 256, 256))
    t0 = time.perf_counter()
    for i, p in enumerate(vols):
        with np.load(p, allow_pickle=True) as z:
            raw = np.asarray(z['oct_bscans'][int(pos[i])], np.float32)
        lo, hi = raw.min(), raw.max()
        u = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        im512[i] = (cv2.resize(u, (RES, RES), cv2.INTER_LINEAR) * 255).astype(np.uint8)
        im256[i] = (cv2.resize(u, (256, 256), cv2.INTER_LINEAR) * 255).astype(np.uint8)
        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print('   read %d/%d  %.0fs  eta %.0fs'
                  % (i + 1, n, el, el / (i + 1) * (n - i - 1)), flush=True)
    im512.flush(); im256.flush()

    enc = build_jepa(RUNS / ('jepa_patch_mirage-ep%d.pth.tar' % epoch), dev)
    RJ = np.lib.format.open_memmap(OUT_CACHE / 'RJ.npy', mode='w+',
                                   dtype=np.float16, shape=(n, 256, 256))
    for s in range(0, n, batch):
        b = np.asarray(im256[s:s + batch], np.float32) / 255.
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(dev)
        with torch.no_grad():
            h = enc(x)
            h = F.layer_norm(h, (h.size(-1),))
            RJ[s:s + batch] = gram(h).half().cpu().numpy()
        del x, h
    RJ.flush()
    np.save(OUT_CACHE / 'pos.npy', pos)
    (OUT_CACHE / 'meta.json').write_text(json.dumps({
        'n': int(n), 'seed': int(seed), 'jepa_epoch': int(epoch),
        'slice_positions': 'random choice of np.linspace(0,199,100)',
        'pos_min': int(pos.min()), 'pos_max': int(pos.max()),
        'pos_mean': float(pos.mean()), 'res': RES,
        'note': 'matched to adapter_sweep cache except slice position',
    }, indent=2))
    del enc
    torch.cuda.empty_cache(); gc.collect()
    print('build done in %.0fs' % (time.perf_counter() - t0))


# ------------------------------------------------------------- stage compare
def _pass(im512, RJ, idx_tr, idx_ev, mir, grab, batch, seed, lr, alpha):
    """One real pass of cfg 7; returns held-out L_rel reduction + drift."""
    dev = 'cuda'
    torch.manual_seed(seed)
    mod = Adapter(depth=2, width=128, alpha=alpha).to(dev)
    nst = (len(idx_tr) + batch - 1) // batch
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=nst, pct_start=0.1)

    def h0(idx):
        x = torch.from_numpy(np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                mir({'bscan': x})
        return grab['H'].float()

    for s in range(0, len(idx_tr), batch):
        idx = np.sort(idx_tr[s:s + batch])
        H = mod(h0(idx))
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        rj = torch.from_numpy(np.asarray(RJ[idx], np.float32)).to(dev)
        F.mse_loss(gram(U), rj).backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    mod.eval()
    r0, r1, dr = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx_ev), batch):
            idx = np.sort(idx_ev[s:s + batch])
            H0 = h0(idx); H = mod(H0)
            rj = torch.from_numpy(np.asarray(RJ[idx], np.float32)).to(dev)
            U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
            U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            r0.append(float(F.mse_loss(gram(U0), rj)))
            r1.append(float(F.mse_loss(gram(U1), rj)))
            dr.append(float((H - H0).norm() / H0.norm()))
    a, b = float(np.mean(r0)), float(np.mean(r1))
    del mod, opt
    torch.cuda.empty_cache(); gc.collect()
    return {'rel_before': a, 'rel_after': b,
            'reduction_pct': 100 * (a - b) / a, 'drift': float(np.mean(dr))}


def compare(batch, seed, lr, alpha):
    dev = 'cuda'
    OUT.mkdir(parents=True, exist_ok=True)
    mir = build_mirage(dev)
    assert not any(p.requires_grad for p in mir.parameters()), 'MIRAGE not frozen'
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach()))

    res = {}
    for tag, root in (('middle', MID), ('stratified', OUT_CACHE)):
        im512 = np.load(root / 'im512.npy', mmap_mode='r')
        RJ = np.load(root / 'RJ.npy', mmap_mode='r')
        n = len(im512)
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        idx_ev, idx_tr = np.sort(perm[:1200]), np.sort(perm[1200:])
        t0 = time.perf_counter()
        r = _pass(im512, RJ, idx_tr, idx_ev, mir, grab, batch, seed, lr, alpha)
        r['n_train'], r['n_eval'] = int(len(idx_tr)), int(len(idx_ev))
        r['seconds'] = round(time.perf_counter() - t0, 1)
        res[tag] = r
        print('%-11s heldout L_rel %.6f -> %.6f   reduction %5.2f%%   '
              'drift %.4f   (%.0fs)'
              % (tag, r['rel_before'], r['rel_after'], r['reduction_pct'],
                 r['drift'], r['seconds']), flush=True)
        del im512, RJ
        gc.collect()

    d = res['stratified']['reduction_pct'] - res['middle']['reduction_pct']
    res['delta_reduction_pp'] = d
    print()
    print('stratified minus middle: %+.2f pp' % d)
    (OUT / 'slice_pos.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'slice_pos.json')


def depth_profile(batch, seed, lr, alpha):
    """Where in the volume does the adapter do well or badly?

    -3.72 pp averaged over the volume could be a uniform tax or it could be
    the adapter failing on the peripheral B-scans and doing fine near the
    ONH.  Those imply very different things for a real run, so bin the
    held-out reduction by depth.
    """
    dev = 'cuda'
    OUT.mkdir(parents=True, exist_ok=True)
    mir = build_mirage(dev)
    assert not any(p.requires_grad for p in mir.parameters()), 'MIRAGE not frozen'
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach()))

    im512 = np.load(OUT_CACHE / 'im512.npy', mmap_mode='r')
    RJ = np.load(OUT_CACHE / 'RJ.npy', mmap_mode='r')
    pos = np.load(OUT_CACHE / 'pos.npy')
    n = len(im512)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    idx_ev, idx_tr = np.sort(perm[:1200]), np.sort(perm[1200:])

    torch.manual_seed(seed)
    mod = Adapter(depth=2, width=128, alpha=alpha).to(dev)
    nst = (len(idx_tr) + batch - 1) // batch
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=nst, pct_start=0.1)

    def h0(idx):
        x = torch.from_numpy(np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                mir({'bscan': x})
        return grab['H'].float()

    for s in range(0, len(idx_tr), batch):
        idx = np.sort(idx_tr[s:s + batch])
        H = mod(h0(idx))
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        rj = torch.from_numpy(np.asarray(RJ[idx], np.float32)).to(dev)
        F.mse_loss(gram(U), rj).backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    mod.eval()
    b0, b1 = [], []
    with torch.no_grad():
        for s in range(0, len(idx_ev), batch):
            idx = np.sort(idx_ev[s:s + batch])
            H0 = h0(idx); H = mod(H0)
            rj = torch.from_numpy(np.asarray(RJ[idx], np.float32)).to(dev)
            U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
            U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            b0.append(((gram(U0) - rj) ** 2).mean(dim=(1, 2)).cpu().numpy())
            b1.append(((gram(U1) - rj) ** 2).mean(dim=(1, 2)).cpu().numpy())
    r0 = np.concatenate(b0); r1 = np.concatenate(b1)
    dep = pos[idx_ev]

    edges = [0, 40, 80, 120, 160, 200]
    print('%14s %7s %11s %11s %11s' % ('depth band', 'n', 'before', 'after', 'reduction'))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dep >= lo) & (dep < hi)
        if not m.any():
            continue
        a, b = float(r0[m].mean()), float(r1[m].mean())
        red = 100 * (a - b) / a
        print('%6d-%-7d %7d %11.6f %11.6f %10.2f%%' % (lo, hi, m.sum(), a, b, red))
        rows.append({'lo': lo, 'hi': hi, 'n': int(m.sum()), 'before': a,
                     'after': b, 'reduction_pct': red})
    ov = 100 * float(r0.mean() - r1.mean()) / float(r0.mean())
    print('%14s %7d %11.6f %11.6f %10.2f%%'
          % ('ALL', len(r0), r0.mean(), r1.mean(), ov))
    span = max(x['reduction_pct'] for x in rows) - min(x['reduction_pct'] for x in rows)
    print()
    print('spread across depth bands: %.2f pp' % span)
    (OUT / 'depth_profile.json').write_text(json.dumps(
        {'bands': rows, 'overall_reduction_pct': float(ov),
         'spread_pp': float(span)}, indent=2))
    print('wrote', OUT / 'depth_profile.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--depth', action='store_true')
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.5)
    a = ap.parse_args()
    if a.build:
        build(a.n, a.epoch, a.batch, a.seed)
    if a.compare:
        compare(a.batch, a.seed, a.lr, a.alpha)
    if a.depth:
        depth_profile(a.batch, a.seed, a.lr, a.alpha)


if __name__ == '__main__':
    main()
