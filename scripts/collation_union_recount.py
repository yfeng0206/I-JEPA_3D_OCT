#!/usr/bin/env python3
"""Recount collation loss using TRUE distinct unions, not slot counts.

WHY THIS EXISTS
---------------
`scripts/collation_probe.py` reports a field named ``distinct_pred_cells``
computed as ``4 * global_min``.  That is the number of predictor SLOTS retained,
not the number of DISTINCT anatomy cells the predictor is asked about.  The four
target blocks overlap, so the four truncated prefixes can name the same cell more
than once.  Symmetrically, its ``fix.ideal`` (55.708) is the mean SUM of the four
target lengths, which double-counts overlap, while the true mean union is 55.604.

Numerator and denominator are therefore both slot-like, and neither is
"distinct cells".  Any sentence of the form "the predictor saw N distinct
anatomical cells" is unsupported by that script.

This script recomputes the same quantities while KEEPING the actual index lists,
so it can report, per batch size:

  slots_retained     4 * global_min                 (what the old script printed)
  distinct_retained  |union of the four prefixes|   (what we actually meant)
  ideal_distinct     |union of the four full lists| (the honest denominator)

It reproduces the original sampling exactly (same seeds, same collator, same
grid file) so the slot figures should match `collation.json` to within
Monte-Carlo noise, which doubles as a regression check on this rewrite.

CPU ONLY.  Never imports CUDA; the caller sets CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import anatomy_target_sampler_v2 as A                       # noqa: E402
from src.masks.multiblock import MaskCollator               # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = pathlib.Path('results/masking/collation')
NBATCH = 600


def build_all():
    """Per slice: the four FULL target index lists, plus context length.

    Identical sampling to collation_probe.build_all, but retains indices.
    """
    import torch
    per = np.load(GRIDS)['per']
    coll = MaskCollator(input_size=(256, 256), patch_size=16, npred=4)
    gen = torch.Generator().manual_seed(0)
    bh, bw = coll._sample_block_size(coll.enc_mask_scale, gen)
    rows = []
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        parts, _ = A.build_targets(cs)
        tg = [np.flatnonzero(p.ravel()).tolist() for p in parts]
        union = set().union(*[set(t) for t in tg])
        random.seed(i)
        kept = None
        for _ in range(50):
            top, left = coll._sample_block_location(bh, bw, coll.height, coll.width)
            idx = coll._block_to_indices(top, left, bh, bw)
            cand = [j for j in idx if j not in union]
            if len(cand) >= coll.min_keep:
                kept = cand
                break
        if kept is None:
            allp = list(range(coll.num_patches))
            kept = [j for j in allp if j not in union] or allp[:coll.min_keep]
        rows.append({'tg_idx': tg, 'union': len(union), 'ctx': len(kept)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_all()
    n = len(rows)
    # Per-slice target lengths, and the full-union ("ideal distinct") per slice.
    tg_len = np.array([[len(t) for t in r['tg_idx']] for r in rows], np.int64)
    ctx = np.array([r['ctx'] for r in rows], np.int64)
    uni = np.array([r['union'] for r in rows], np.int64)
    rng = np.random.default_rng(0)

    print('slices %d' % n)
    print('mean union (TRUE ideal distinct cells) : %.4f' % uni.mean())
    print('mean SUM of target lengths (old ideal) : %.4f' % tg_len.sum(1).mean())
    print('overlap double-count in old ideal      : %.4f cells (%.2f%%)'
          % (tg_len.sum(1).mean() - uni.mean(),
             100 * (tg_len.sum(1).mean() - uni.mean()) / uni.mean()))
    print()

    res = {'n': int(n),
           'mean_union_true_ideal': float(uni.mean()),
           'mean_sum_of_target_lengths_old_ideal': float(tg_len.sum(1).mean()),
           'mean_ctx': float(ctx.mean()),
           'batches': {}}

    hdr = '%6s %14s %18s %18s %14s' % (
        'batch', 'global_min', 'slots (4*gmin)', 'DISTINCT retained',
        'dup slots')
    print('=== slots vs distinct cells actually retained ===')
    print(hdr)
    for b in (16, 32, 64):
        slots, distinct, gmins = [], [], []
        for _ in range(NBATCH):
            s = rng.integers(0, n, size=b)
            g = max(1, int(tg_len[s].min()))
            gmins.append(g)
            slots.append(4 * g)
            # Global min-truncation keeps the first g indices of each target
            # list, in stored order, with no sorting (see multiblock.py:199-213).
            for j in s:
                pref = set()
                for t in rows[j]['tg_idx']:
                    pref.update(t[:g])
                distinct.append(len(pref))
        r = {'global_min': float(np.mean(gmins)),
             'slots_retained': float(np.mean(slots)),
             'distinct_retained': float(np.mean(distinct)),
             'duplicate_slots': float(np.mean(slots) - np.mean(distinct)),
             'distinct_pct_of_true_ideal':
                 float(100 * np.mean(distinct) / uni.mean()),
             'slots_pct_of_old_ideal':
                 float(100 * np.mean(slots) / tg_len.sum(1).mean())}
        res['batches'][str(b)] = r
        print('%6d %14.2f %18.2f %18.2f %14.2f'
              % (b, r['global_min'], r['slots_retained'],
                 r['distinct_retained'], r['duplicate_slots']))
    print()

    print('=== headline restated honestly (B=64) ===')
    b64 = res['batches']['64']
    print('old claim : %.2f "distinct cells" of an "ideal" %.3f  = %.1f%%'
          % (b64['slots_retained'], tg_len.sum(1).mean(),
             b64['slots_pct_of_old_ideal']))
    print('corrected : %.2f DISTINCT cells of a true ideal %.3f = %.1f%%'
          % (b64['distinct_retained'], uni.mean(),
             b64['distinct_pct_of_true_ideal']))
    print()

    print('=== fixed-K resample, distinct vs slots ===')
    print('%6s %16s %18s %14s' % ('K', 'slots (4K)', 'DISTINCT', 'dup slots'))
    res['fixed_k'] = {}
    for k in (8, 12, 16, 20, 24):
        d = []
        for r_ in rows:
            pref = set()
            for t in r_['tg_idx']:
                pref.update(t[:k])
            d.append(len(pref))
        dm = float(np.mean(d))
        res['fixed_k'][str(k)] = {
            'slots': 4 * k,
            'distinct_cells': dm,
            'pct_of_true_ideal': float(100 * dm / uni.mean()),
            'duplicate_slots': float(4 * k - dm)}
        print('%6d %16d %18.2f %14.2f' % (k, 4 * k, dm, 4 * k - dm))
    print()
    print('NOTE: the K=16 production setting is the one every ragged-target')
    print('config actually shipped (pred_target_k: 16), so the trained arms')
    print('sat at the K=16 row, NOT at the global-min row.')

    (OUT / 'collation_union.json').write_text(json.dumps(res, indent=2))
    print('\nwrote %s' % (OUT / 'collation_union.json'))


if __name__ == '__main__':
    main()
