#!/usr/bin/env python3
"""Cost out the three Step-14 policies for variable target size.

Policy 0  current      global_min_pred over the microbatch, front-slice
Policy 1  bucket-by-K  sort by the slice's smallest target, batch contiguous
                       runs, then truncate within the bucket.  No change to
                       the sampler or the method, only to batch assembly.
Policy 2  fixed-K      every target contributes exactly K indices; targets
                       with more are subsampled, slices that cannot reach K
                       fall back to random.  The full union is still hidden
                       from the context encoder, only the predicted index
                       list is capped, so the anatomical story is intact.

Padding + validity mask is excluded on purpose: the predictor's attention
has no padding mask, so pad positions would be attended as real tokens.

CPU only.
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


def sizes_table():
    per = np.load(GRIDS)['per']
    rows = []
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        parts, _ = A.build_targets(cs)
        rows.append([int(p.sum()) for p in parts])
    return np.array(rows, np.int64)


def p0_global(sizes, batch, trials, rng):
    n, keep = len(sizes), []
    for _ in range(trials):
        b = sizes[rng.integers(0, n, size=batch)]
        keep.append(batch * 4 * max(1, int(b.min())) / b.sum())
    return 100 * float(np.mean(keep))


def p1_bucket(sizes, batch, trials, rng):
    """Sort an epoch by smallest target, cut contiguous microbatches."""
    n, keep = len(sizes), []
    for _ in range(trials):
        order = rng.permutation(n)
        s = sizes[order]
        order2 = np.argsort(s.min(1), kind='stable')
        s = s[order2]
        nb = n // batch
        tot_kept = tot = 0
        for b in range(nb):
            blk = s[b * batch:(b + 1) * batch]
            g = max(1, int(blk.min()))
            tot_kept += batch * 4 * g
            tot += blk.sum()
        keep.append(tot_kept / tot)
    return 100 * float(np.mean(keep))


def p2_fixed(sizes, k):
    ok = (sizes >= k).all(1)
    kept_area = ok.sum() * 4 * k
    return {'K': k,
            'viable_slices_pct': 100 * float(ok.mean()),
            'fallback_pct': 100 * float(1 - ok.mean()),
            'area_kept_pct': 100 * float(kept_area / sizes.sum()),
            'union_cells': 4 * k}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes = sizes_table()
    print('slices %d   mean union %.1f cells   mean target %.1f'
          % (len(sizes), sizes.sum(1).mean(), sizes.mean()))
    print()
    res = {'n_slices': int(len(sizes)),
           'mean_union_cells': float(sizes.sum(1).mean())}

    print('=== policy 0 vs policy 1, area of the anatomy target actually predicted ===')
    print('%6s %14s %14s %10s' % ('batch', 'global-min', 'bucketed', 'gain'))
    res['compare'] = {}
    for b in (16, 32, 64):
        a = p0_global(sizes, b, 400, rng)
        c = p1_bucket(sizes, b, 40, rng)
        print('%6d %13.1f%% %13.1f%% %9.1fx' % (b, a, c, c / a))
        res['compare'][str(b)] = {'global_min_pct': a, 'bucketed_pct': c,
                                  'gain_x': c / a}
    print()

    print('=== policy 2, fixed K per target ===')
    print('%5s %14s %13s %13s' % ('K', 'viable slices', 'fallback', 'area kept'))
    res['fixed_k'] = {}
    for k in (4, 6, 8, 10, 12):
        r = p2_fixed(sizes, k)
        print('%5d %13.1f%% %12.1f%% %12.1f%%'
              % (k, r['viable_slices_pct'], r['fallback_pct'], r['area_kept_pct']))
        res['fixed_k'][str(k)] = r
    print()
    b64 = res['compare']['64']
    print('Bucketing alone lifts batch-64 retention %.1f%% -> %.1f%% with no '
          'change\nto the sampler, the mask or the method.'
          % (b64['global_min_pct'], b64['bucketed_pct']))
    (OUT / 'policies.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'policies.json')


if __name__ == '__main__':
    main()
