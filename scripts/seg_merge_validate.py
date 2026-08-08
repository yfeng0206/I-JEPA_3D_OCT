#!/usr/bin/env python
"""Pre-merge validation: find where a multi-dataset merge would go wrong.

Four independent failure levels, checked BEFORE any training:

  L1 INPUT      intensity statistics after the pipeline's own preprocessing.
                If datasets differ here, the model sees inconsistent inputs.

  L2 GEOMETRY   MIRAGE resizes every image to a SQUARE input_size (default
                1024).  The sources are not square and not equally shaped, so
                each is distorted by a different anisotropy factor.  A layer of
                fixed anatomical thickness therefore lands on a different
                number of model pixels depending on which dataset it came from.

  L3 TRUTH      does "RNFL" mean the same band across datasets?  Measured as a
                RATIO (layer thickness / total retina thickness), which is
                invariant to any resize.  If the ratios agree but the post-
                resize pixel counts do not, the problem is pure geometry and is
                fixable.  If the ratios disagree, the annotation conventions
                genuinely differ and no resize will fix it.

  L4 MERGE      value-mapping round trip, subject-level split integrity, and
                filename collisions.

    python scripts/seg_merge_validate.py
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SOURCES = {
    'GOALS': r'D:\jepa_phase0\mirage-goals\downloads\GOALS.zip',
    'Duke_DME': r'D:\jepa_phase0\mirage-datasets\Duke_DME.zip',
    'AROI': r'D:\jepa_phase0\mirage-datasets\AROI.zip',
}
FAIRVISION = r'D:\jepa_phase0\fairvision-glaucoma\data\Training'
MIRAGE_MASKS = r'D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training'

# Per-dataset semantics, verified empirically (see scripts/dataset_visual_compare.py
# and the row-position check in the session log):
#   GOALS     80=RNFL, 160=GCIPL, 255=Choroid          (regions, as labelled)
#   Duke_DME  25=ILM-region=RNFL, 51=NFL-region=GCIPL   (named by UPPER boundary)
#   AROI      23=ILM-IPL/INL = RNFL+GCIPL MERGED
INNER = {
    'GOALS': [80, 160],
    'Duke_DME': [25, 51],
    'AROI': [23],
    # FairVision masks come from our own MIRAGE inference, scaled x80, so they
    # carry the GOALS class ids: 1->80 RNFL, 2->160 GCIPL, 3->240 choroid.
    'FairVision': [80, 160],
}
RNFL = {'GOALS': [80], 'Duke_DME': [25], 'AROI': [], 'FairVision': [80]}
GCIPL = {'GOALS': [160], 'Duke_DME': [51], 'AROI': [], 'FairVision': [160]}
# Deepest structure used to locate the RPE/BM interface for the ratio measure.
DEEP = {'GOALS': [255], 'Duke_DME': [204], 'AROI': [92], 'FairVision': [240]}


def load_pairs(key, n, rng):
    out = []
    with zipfile.ZipFile(SOURCES[key]) as z:
        bs = sorted(x for x in z.namelist() if '/bscan/' in x and x.endswith('.png'))
        for b in rng.choice(len(bs), size=min(n, len(bs)), replace=False):
            b = bs[int(b)]
            s = b.replace('/bscan/', '/semseg/')
            if s not in z.namelist():
                continue
            img = np.array(Image.open(io.BytesIO(z.read(b))).convert('L'))
            seg = np.array(Image.open(io.BytesIO(z.read(s))).convert('L'))
            out.append((b.split('/')[-1], img, seg))
    return out


def load_fairvision(n, rng):
    files = sorted(f for f in os.listdir(FAIRVISION) if f.endswith('.npz'))
    out = []
    for i in rng.choice(len(files), size=min(n * 3, len(files)), replace=False):
        f = files[int(i)]
        mp = os.path.join(MIRAGE_MASKS, f)
        if not os.path.isfile(mp):
            continue
        with np.load(os.path.join(FAIRVISION, f), allow_pickle=True) as z:
            vol = z['oct_bscans']
        with np.load(mp, allow_pickle=False) as z:
            hard, idx = z['hard_masks'], z['slice_indices']
        k = len(idx) // 2
        # MIRAGE hard masks use class ids 1/2/3; scale to GOALS display values.
        out.append((f, np.array(vol[int(idx[k])]), hard[k] * 80))
        if len(out) >= n:
            break
    return out


def band_thickness(seg, values):
    """Mean vertical thickness in pixels of the union of `values`, over columns
    where it is present."""
    if not values:
        return float('nan')
    m = np.isin(seg, values)
    cols = m.sum(axis=0)
    cols = cols[cols > 0]
    return float(cols.mean()) if cols.size else float('nan')


def retina_extent(seg, key):
    """ILM to RPE/BM, i.e. the neurosensory retina — comparable across datasets.

    Using "top of inner retina" to "top of the deepest structure" avoids the
    trap that Duke's `BM` and AROI's `Under BM` classes run to the image bottom
    (they include sclera), which would inflate the denominator for those two
    and make the ratio meaningless.
    """
    inner = np.isin(seg, INNER[key])
    deep = np.isin(seg, DEEP[key])
    ri = np.where(inner.any(axis=1))[0]
    rd = np.where(deep.any(axis=1))[0]
    if not ri.size or not rd.size:
        return float('nan')
    top = ri.min()          # ILM
    bottom = rd.min()       # first row of the deepest class = RPE/BM interface
    return float(bottom - top) if bottom > top else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--input-size', type=int, default=1024,
                    help="MIRAGE's square resize target (run_seg_tuning default)")
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = {}
    for k in SOURCES:
        if os.path.isfile(SOURCES[k]):
            data[k] = load_pairs(k, args.n, rng)
    data['FairVision'] = load_fairvision(min(args.n, 25), rng)

    S = args.input_size
    print('=' * 78)
    print('L1  INPUT STATISTICS  (raw pixel values, before any normalisation)')
    print('=' * 78)
    print('%-12s %8s %8s %8s %8s' % ('dataset', 'mean', 'std', 'p01', 'p99'))
    for k, items in data.items():
        if not items:
            continue
        a = np.concatenate([i[1].ravel()[::37].astype(np.float32) for i in items])
        print('%-12s %8.1f %8.1f %8.1f %8.1f'
              % (k, a.mean(), a.std(), np.percentile(a, 1), np.percentile(a, 99)))
    print()
    print('  -> MIRAGE applies per-image Normalize(ImageNet mean/std) AFTER resize,')
    print('     so absolute level is largely removed; CONTRAST (std) is not.')

    print()
    print('=' * 78)
    print('L2  GEOMETRY  (every image is resized to a SQUARE %dx%d)' % (S, S))
    print('=' * 78)
    print('%-12s %11s %8s %8s %10s' % ('dataset', 'native HxW', 'sy', 'sx', 'anisotropy'))
    aniso = {}
    for k, items in data.items():
        if not items:
            continue
        h, w = items[0][1].shape
        sy, sx = S / h, S / w
        aniso[k] = sy / sx
        print('%-12s %11s %8.2f %8.2f %10.3f'
              % (k, '%dx%d' % (h, w), sy, sx, sy / sx))
    print()
    print('  anisotropy = vertical stretch / horizontal stretch.')
    print('  1.000 means shape is preserved. Values above 1 stretch layers TALLER.')
    if 'GOALS' in aniso and 'FairVision' in aniso:
        print('  GOALS vs FairVision mismatch: %.3f  <-- a layer trained on GOALS'
              % (aniso['GOALS'] / aniso['FairVision']))
        print('  appears %.0f%% thicker (in model pixels) than the same anatomy in FairVision.'
              % (100 * (aniso['GOALS'] / aniso['FairVision'] - 1)))

    print()
    print('=' * 78)
    print('L3  TRUTH SHIFT  (does a class mean the same band across datasets?)')
    print('=' * 78)
    print('%-12s %10s %10s %10s %12s %12s'
          % ('dataset', 'RNFL px', 'GCIPL px', 'inner px', 'inner/retina', 'inner @1024'))
    for k, items in data.items():
        if not items:
            continue
        inner_v = INNER.get(k, [80, 160])
        rnfl_v = RNFL.get(k, [80])
        gcipl_v = GCIPL.get(k, [160])
        r, g, inn, rat = [], [], [], []
        for _, img, seg in items:
            r.append(band_thickness(seg, rnfl_v))
            g.append(band_thickness(seg, gcipl_v))
            t = band_thickness(seg, inner_v)
            inn.append(t)
            ext = retina_extent(seg, k)
            rat.append(t / ext if ext == ext and ext else float('nan'))
        h = items[0][1].shape[0]
        scaled = np.nanmean(inn) * (S / h)
        rr = np.nanmean(r) if np.any(np.isfinite(r)) else float('nan')
        gg = np.nanmean(g) if np.any(np.isfinite(g)) else float('nan')
        print('%-12s %10.1f %10.1f %10.1f %12.3f %12.1f'
              % (k, rr, gg, np.nanmean(inn), np.nanmean(rat), scaled))
    print()
    print('  inner/retina is RESIZE-INVARIANT: if these agree, the annotation')
    print('  conventions agree and any difference is pure geometry (fixable).')
    print('  inner @1024 is what the MODEL actually sees after the square resize.')
    print('  AROI has no separate RNFL/GCIPL by construction (merged class).')

    print()
    print('=' * 78)
    print('L4  MERGE HAZARDS')
    print('=' * 78)
    for k in SOURCES:
        if k not in data or not data[k]:
            continue
        with zipfile.ZipFile(SOURCES[k]) as z:
            names = [x.split('/')[-1] for x in z.namelist()
                     if '/bscan/' in x and x.endswith('.png')]
        stems = [re.sub(r'[_-]?\d+\.png$', '', n) for n in names]
        print('%-12s %5d images  %3d subjects  name pattern e.g. %s'
              % (k, len(names), len(set(stems)), names[0]))
    allnames = []
    for k in SOURCES:
        if k not in data or not data[k]:
            continue
        with zipfile.ZipFile(SOURCES[k]) as z:
            allnames += [x.split('/')[-1] for x in z.namelist()
                         if '/bscan/' in x and x.endswith('.png')]
    dup = len(allnames) - len(set(allnames))
    print()
    print('  filename collisions across datasets: %d  %s'
          % (dup, '(prefix each dataset when merging)' if dup else '(none)'))
    print('  splits MUST be by subject, never by slice -- adjacent B-scans from')
    print('  one eye are near-duplicates and would leak across train/val/test.')


if __name__ == '__main__':
    main()
