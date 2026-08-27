#!/usr/bin/env python3
"""Quantify the `global_min_pred` truncation and cost out the fixed-K fix.

src/masks/curriculum.py:1211-1218 does:

    global_min_pred = max(1, min(t.numel() for group in masks_pred for t in group))
    collated = [torch.stack([t[:global_min_pred] for t in group]) ...]

That is a min over EVERY target of EVERY sample in the microbatch, followed
by a front-slice.  One small target anywhere in the batch truncates all
4*B of them.  Checklist Step 14 forbids exactly this.

Of the three policies Step 14 offers, padding + validity mask is NOT
available to us: the predictor's attention has no padding mask, so pad
positions would be attended as if real.  That leaves fixed-K or bucketing.

This runs entirely on CPU against the cached 16x16 anatomy grids, so it can
run alongside a GPU job.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                      # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/variable_k'


def target_sizes():
    per = np.load(GRIDS)['per']            # (N,2,16,16)
    sizes, viable = [], 0
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        viable += 1
        parts, _ = A.build_targets(cs)
        sizes.append([int(p.sum()) for p in parts])
    return np.array(sizes, np.int64), viable, len(per)


def simulate(sizes, batch, trials, rng):
    """What global_min_pred does, per microbatch."""
    n = len(sizes)
    kept, mins, ones = [], [], 0
    for _ in range(trials):
        idx = rng.integers(0, n, size=batch)
        b = sizes[idx]                      # (batch,4)
        g = max(1, int(b.min()))
        mins.append(g)
        ones += int(g == 1)
        kept.append(batch * 4 * g / b.sum())
    return (np.array(mins), np.array(kept), 100.0 * ones / trials)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes, viable, total = target_sizes()
    flat = sizes.reshape(-1)
    print('slices           %d viable of %d' % (viable, total))
    print('targets          %d  (4 per slice)' % flat.size)
    print('cells per target min %d  p1 %d  p5 %d  median %d  mean %.1f  max %d'
          % (flat.min(), np.percentile(flat, 1), np.percentile(flat, 5),
             np.median(flat), flat.mean(), flat.max()))
    print('per-slice union  mean %.1f cells' % sizes.sum(1).mean())
    print()

    res = {'n_slices': int(viable), 'n_targets': int(flat.size),
           'cells_per_target': {
               'min': int(flat.min()), 'p1': float(np.percentile(flat, 1)),
               'p5': float(np.percentile(flat, 5)),
               'median': float(np.median(flat)), 'mean': float(flat.mean()),
               'max': int(flat.max())},
           'truncation': {}, 'fixed_k': {}}

    print('=== current behaviour: global_min_pred over a microbatch ===')
    print('%6s %10s %12s %14s' % ('batch', 'median K', 'K==1 rate', 'area kept'))
    for b in (8, 16, 32, 64):
        mins, kept, ones = simulate(sizes, b, 2000, rng)
        print('%6d %10d %11.1f%% %13.1f%%'
              % (b, int(np.median(mins)), ones, 100 * kept.mean()))
        res['truncation'][str(b)] = {
            'median_K': int(np.median(mins)), 'K_eq_1_pct': ones,
            'area_kept_pct': float(100 * kept.mean())}
    print()

    print('=== fixed-K policy: every target contributes exactly K cells ===')
    print('%5s %14s %14s %12s' % ('K', 'targets >= K', 'slices all>=K', 'area kept'))
    for k in (1, 2, 4, 6, 8, 10, 12, 16, 20):
        ge = 100.0 * (flat >= k).mean()
        allge = 100.0 * (sizes >= k).all(1).mean()
        kept = 100.0 * (np.minimum(sizes, k).sum() / sizes.sum())
        print('%5d %13.1f%% %13.1f%% %11.1f%%' % (k, ge, allge, kept))
        res['fixed_k'][str(k)] = {'targets_ge_pct': ge, 'slices_all_ge_pct': allge,
                                  'area_kept_pct': kept}
    print()
    b64 = res['truncation']['64']
    print('At the production microbatch of 64, global_min_pred keeps %.1f%% of '
          'target area\nand collapses to K=1 in %.1f%% of batches.'
          % (b64['area_kept_pct'], b64['K_eq_1_pct']))
    (OUT / 'variable_k.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'variable_k.json')


if __name__ == '__main__':
    main()
