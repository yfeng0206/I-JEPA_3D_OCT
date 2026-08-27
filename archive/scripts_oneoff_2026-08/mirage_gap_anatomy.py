#!/usr/bin/env python
"""What exactly is missing from MIRAGE's raw output, and what does repair add?

Observation from inspection: MIRAGE on FairVision often looks disconnected,
with a hole through the middle of the retina and scattered speckle.

There are two very different possible causes and they need different fixes:

  (a) SEGMENTATION ERROR -- the model is failing on this device, and the fix is
      more/better training data.
  (b) TAXONOMY -- GOALS only labels RNFL, GCIPL and choroid.  The layers
      between GCIPL and choroid (INL, OPL, ONL, photoreceptors, RPE) have NO
      label, so a hole through the middle is what a perfect model would also
      produce.  The fix is the envelope fill, which is already what we do.

This decides between them by measuring, per column, where the raw union is
present and where it is absent, split by whether the absence sits BETWEEN the
GCIPL and the choroid (taxonomy) or somewhere else (error).

    python scripts/mirage_gap_anatomy.py --volumes 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    CLASS_CHOROID,
    CLASS_GCIPL,
    CLASS_RNFL,
    build_union,
    unpack_guides,
)

NATIVE = 200


def column_report(hard, envelope):
    """Per-column accounting of raw union vs repaired envelope."""
    rnfl = hard == CLASS_RNFL
    gcipl = hard == CLASS_GCIPL
    chor = hard == CLASS_CHOROID
    raw = build_union(hard)

    out = {
        'cols_with_tissue': 0,
        'cols_raw_contiguous': 0,
        'cols_with_interior_gap': 0,
        'gap_between_gcipl_and_choroid': 0,
        'gap_elsewhere': 0,
        'gap_rows_taxonomy': 0,
        'gap_rows_other': 0,
        'added_rows_taxonomy': 0,
        'added_rows_other': 0,
        'raw_rows': 0,
        'env_rows': 0,
    }

    for c in range(hard.shape[1]):
        col_raw = raw[:, c]
        col_env = envelope[:, c]
        out['raw_rows'] += int(col_raw.sum())
        out['env_rows'] += int(col_env.sum())
        if not col_raw.any():
            continue
        out['cols_with_tissue'] += 1

        rows = np.where(col_raw)[0]
        top, bot = rows.min(), rows.max()
        interior = col_raw[top:bot + 1]
        holes = np.where(~interior)[0] + top
        if holes.size == 0:
            out['cols_raw_contiguous'] += 1
        else:
            out['cols_with_interior_gap'] += 1

        # Where is the GCIPL/choroid boundary region in this column?
        g = np.where(gcipl[:, c])[0]
        ch = np.where(chor[:, c])[0]
        if g.size and ch.size:
            lo, hi = g.max(), ch.min()   # between GCIPL bottom and choroid top
        else:
            lo, hi = -1, -1

        for h in holes:
            if lo >= 0 and lo < h < hi:
                out['gap_rows_taxonomy'] += 1
            else:
                out['gap_rows_other'] += 1
        if holes.size:
            in_band = [h for h in holes if lo >= 0 and lo < h < hi]
            if len(in_band) >= 0.5 * holes.size:
                out['gap_between_gcipl_and_choroid'] += 1
            else:
                out['gap_elsewhere'] += 1

        # What did the repair ADD in this column, and was it in that band?
        added = np.where(col_env & ~col_raw)[0]
        for a in added:
            if lo >= 0 and lo < a < hi:
                out['added_rows_taxonomy'] += 1
            else:
                out['added_rows_other'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--volumes', type=int, default=200)
    ap.add_argument('--slice-stride', type=int, default=20)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--root', default=r'D:\jepa_phase0\fairvision-glaucoma')
    args = ap.parse_args()

    mask_dir = os.path.join(args.root, 'mirage_masks', 'Training')
    guide_dir = os.path.join(args.root, 'mirage_guides', 'Training')
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.npz'))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(files), size=min(args.volumes, len(files)), replace=False)

    total = None
    n_slices = 0
    for vi in pick:
        name = files[vi]
        gp = os.path.join(guide_dir, name)
        if not os.path.isfile(gp):
            continue
        with np.load(os.path.join(mask_dir, name), allow_pickle=False) as z:
            hard_all = z['hard_masks']
        with np.load(gp, allow_pickle=False) as z:
            packed, valid = z['packed_envelopes'], z['valid']
        for sl in range(0, hard_all.shape[0], args.slice_stride):
            if not bool(valid[sl]):
                continue
            env = unpack_guides(packed[sl:sl + 1], (NATIVE, NATIVE))[0]
            r = column_report(hard_all[sl], env)
            if total is None:
                total = {k: 0 for k in r}
            for k, v in r.items():
                total[k] += v
            n_slices += 1

    t = total
    ct = t['cols_with_tissue']
    print('slices: %d   columns with any MIRAGE tissue: %d' % (n_slices, ct))
    print()
    print('RAW MIRAGE output, per column')
    print('  contiguous top-to-bottom      %6.1f%%' % (100 * t['cols_raw_contiguous'] / ct))
    print('  has an interior hole          %6.1f%%' % (100 * t['cols_with_interior_gap'] / ct))
    print()
    hole_cols = max(t['cols_with_interior_gap'], 1)
    print('Of the columns with a hole, where is it?')
    print('  mostly between GCIPL and choroid  %6.1f%%   <- unlabelled by GOALS' 
          % (100 * t['gap_between_gcipl_and_choroid'] / hole_cols))
    print('  mostly elsewhere                  %6.1f%%   <- genuine segmentation gap'
          % (100 * t['gap_elsewhere'] / hole_cols))
    print()
    gr = max(t['gap_rows_taxonomy'] + t['gap_rows_other'], 1)
    print('Hole PIXELS by location')
    print('  between GCIPL and choroid     %6.1f%%' % (100 * t['gap_rows_taxonomy'] / gr))
    print('  elsewhere                     %6.1f%%' % (100 * t['gap_rows_other'] / gr))
    print()
    ar = max(t['added_rows_taxonomy'] + t['added_rows_other'], 1)
    print('What the REPAIR added')
    print('  raw union area                %.4f of frame' % (t['raw_rows'] / (n_slices * NATIVE * NATIVE)))
    print('  repaired envelope area        %.4f of frame' % (t['env_rows'] / (n_slices * NATIVE * NATIVE)))
    print('  repair added                  %.1f%% more area' 
          % (100 * (t['env_rows'] - t['raw_rows']) / max(t['raw_rows'], 1)))
    print('  of the added pixels:')
    print('    between GCIPL and choroid   %6.1f%%   <- filling the unlabelled mid-retina'
          % (100 * t['added_rows_taxonomy'] / ar))
    print('    elsewhere                   %6.1f%%   <- extending/closing the envelope'
          % (100 * t['added_rows_other'] / ar))


if __name__ == '__main__':
    main()
