#!/usr/bin/env python
"""Build a merged retinal-layer dataset with RNFL and GCIPL MERGED.

Why merge the two classes
-------------------------
Three independent reasons, in order of force:

1. The downstream consumer already discards the distinction.
   ``src/guides/mirage_envelope.py`` computes
   ``build_union = np.isin(labels, (RNFL, GCIPL, Choroid))`` and stores a
   bit-packed BOOLEAN envelope.  Since
   ``RNFL u GCIPL u Choroid == InnerRetina u Choroid`` when
   ``InnerRetina := RNFL u GCIPL``, this taxonomy produces a byte-identical
   guide while making the label space compatible across datasets.

2. It makes every source directly usable with no partial-label machinery.
   AROI annotates ``ILM-IPL/INL`` as one band that cannot be split; under the
   merged taxonomy that band IS the target class, so its 1,105 images become
   ordinary hard supervision instead of needing a superclass loss.

3. At FairVision's axial sampling the macular RNFL is only a few pixels thick,
   so an RNFL-vs-GCIPL split there sits near the resolution floor regardless of
   how much training data is thrown at it.

Output taxonomy (``INFO.json`` maps VALUE -> class index)::

    0   Elsewhere     index 0
    128 InnerRetina   index 1     RNFL u GCIPL
    255 Choroid       index 2
    1   Background    index 3     ignore (MIRAGE auto-detects "background")

Deliberate NON-interventions, both from review findings
-------------------------------------------------------
* No squaring.  A previous attempt resized to 512 before the pipeline's
  1024 resize; that is a mathematical no-op (``800x1100 -> 512 -> 1024`` has
  anisotropy 1.3750, identical to going direct).  Anisotropy is left as MIRAGE
  applies it, uniformly, and recorded as a known confound.
* No build-time min-max.  ``mutils/dataset_folder.py:157`` already applies
  ``normalize_to_0_1`` before the transform, so doing it here is redundant and
  quantises to uint8 for nothing.
* GOALS test images are copied BYTE-IDENTICAL, so GOALS test metrics stay
  comparable with the published baseline.

    python scripts/build_seg_merged_v2.py --out D:\\jepa_phase0\\mirage-datasets\\MergedV2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

V_ELSEWHERE, V_INNER, V_CHOROID, V_IGNORE = 0, 128, 255, 1

INFO = {
    "0": {"value": V_ELSEWHERE, "label": "Elsewhere"},
    "1": {"value": V_INNER, "label": "InnerRetina"},
    "2": {"value": V_CHOROID, "label": "Choroid"},
    "3": {"value": V_IGNORE, "label": "Background"},
}

SOURCES = {
    'GOALS': r'D:\jepa_phase0\mirage-goals\downloads\GOALS.zip',
    'Duke_DME': r'D:\jepa_phase0\mirage-datasets\Duke_DME.zip',
    'AROI': r'D:\jepa_phase0\mirage-datasets\AROI.zip',
}

# source value -> output value.  Every source value MUST appear here; anything
# missing is reported loudly rather than silently becoming Elsewhere.
REMAP = {
    'GOALS': {
        0: V_ELSEWHERE,
        80: V_INNER,       # RNFL  )
        160: V_INNER,      # GCIPL ) merged
        255: V_CHOROID,
    },
    'Duke_DME': {
        0: V_ELSEWHERE,
        25: V_INNER,       # region below ILM = RNFL   )
        51: V_INNER,       # region below NFL = GCL+IPL) merged
        76: V_ELSEWHERE,   # INL  ) mid-retina, which GOALS also leaves
        102: V_ELSEWHERE,  # OPL  ) unlabelled, so Elsewhere is correct
        127: V_ELSEWHERE,  # ONL  )
        153: V_ELSEWHERE,  # ISM
        178: V_ELSEWHERE,  # OS
        204: V_IGNORE,     # sub-BM: choroid + sclera, not GOALS' bounded choroid
        229: V_IGNORE,     # fluid
    },
    'AROI': {
        0: V_ELSEWHERE,    # above ILM
        23: V_INNER,       # ILM-IPL/INL == exactly the merged target class
        46: V_ELSEWHERE,   # IPL/INL-RPE
        69: V_ELSEWHERE,   # RPE-BM
        92: V_IGNORE,      # sub-BM: same sclera problem as Duke
        115: V_IGNORE,     # cyst
        138: V_IGNORE,     # PED
        161: V_IGNORE,     # SRF
    },
}

# Source class whose region starts exactly at Bruch's membrane. Everything
# below BM in these datasets is choroid THEN sclera with no boundary drawn
# between them, which is why the whole region is ignored by default.
SUB_BM_CLASS = {'Duke_DME': 204, 'AROI': 92}


def add_choroid_band(mapped, seg, ds, frac):
    """Label a bounded strip immediately below BM as choroid.

    Measured motivation: GOALS' choroid is a tight band spanning 41%-56% of
    image height that never reaches the bottom row, whereas the sub-BM region
    in Duke/AROI covers ~55% of the image and always reaches the bottom. So
    mapping all of sub-BM to choroid would teach gross over-prediction.

    Instead we take only the top `frac` of image height below BM. This
    UNDER-bounds the true choroid on purpose: tissue immediately beneath
    Bruch's membrane is choroid by definition, while the deeper part stays
    ignore. Under-bounding can only withhold supervision, never assert a wrong
    label -- the opposite error would be a truth-level defect.
    """
    src = SUB_BM_CLASS.get(ds)
    if src is None or frac <= 0:
        return mapped, 0
    H = mapped.shape[0]
    depth = max(1, int(round(frac * H)))
    region = seg == src
    cols = np.where(region.any(axis=0))[0]
    if cols.size == 0:
        return mapped, 0
    rows = np.arange(H)[:, None]
    top = np.argmax(region[:, cols], axis=0)[None, :]
    band = (rows >= top) & (rows < top + depth)
    sub = mapped[:, cols]
    # convert only pixels that are currently ignore, so a cyst/PED/fluid label
    # falling inside the band is never silently overwritten
    sub[band & (sub == V_IGNORE)] = V_CHOROID
    mapped[:, cols] = sub
    return mapped, int(band.sum())


def subject_of(dataset, filename):
    """Group key for leakage-free splitting; adjacent B-scans are near-duplicates."""
    stem = os.path.splitext(filename)[0]
    if dataset == 'Duke_DME':      # Subject_10_5 -> Subject_10
        return '_'.join(stem.split('_')[:2])
    if dataset == 'AROI':          # patient9_0057 -> patient9
        return stem.split('_')[0]
    return stem


def remap_mask(seg, table):
    out = np.full(seg.shape, V_IGNORE, dtype=np.uint8)
    for src, dst in table.items():
        out[seg == src] = dst
    unmapped = set(np.unique(seg).tolist()) - set(table)
    return out, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\jepa_phase0\mirage-datasets\MergedV2')
    ap.add_argument('--val-frac', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--choroid-band', type=float, default=0.0,
                    help='fraction of image height below BM to label choroid '
                         'in Duke/AROI; 0 disables (v2 behaviour)')
    ap.add_argument('--datasets', nargs='+',
                    default=['GOALS', 'Duke_DME', 'AROI'])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split in ('train', 'val', 'test'):
        for sub in ('bscan', 'semseg'):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))
    pixels = defaultdict(lambda: defaultdict(int))
    unmapped_all = defaultdict(set)
    assign = []
    sizes = defaultdict(set)

    for ds in args.datasets:
        path = SOURCES.get(ds)
        if not path or not os.path.isfile(path):
            print('SKIP %s (not found)' % ds)
            continue
        table = REMAP[ds]
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            bscans = sorted(x for x in names
                            if '/bscan/' in x and x.endswith('.png'))
            # Split policy:
            #   GOALS    keeps MIRAGE's OFFICIAL train/val/test so its test set
            #            stays the published benchmark.
            #   Duke     subject-split train/val (it has hard InnerRetina labels).
            #   AROI     100% train; it carries no choroid, so it cannot
            #            validate the full taxonomy.
            if ds == 'GOALS':
                split_of = {b: b.split('/')[1] for b in bscans}
            elif ds == 'AROI':
                split_of = {b: 'train' for b in bscans}
            else:
                subs = sorted({subject_of(ds, b.split('/')[-1]) for b in bscans})
                perm = rng.permutation(len(subs))
                n_val = max(1, int(round(len(subs) * args.val_frac)))
                pick = {subs[int(i)]: ('val' if j < n_val else 'train')
                        for j, i in enumerate(perm)}
                split_of = {b: pick[subject_of(ds, b.split('/')[-1])]
                            for b in bscans}
                assign += ['%-9s %-14s -> %s' % (ds, s, pick[s]) for s in subs]

            for b in bscans:
                s = b.replace('/bscan/', '/semseg/')
                if s not in names:
                    continue
                split = split_of[b]
                out_name = '%s__%s' % (ds, b.split('/')[-1])

                # Image bytes are copied VERBATIM: no resize, no renormalisation.
                # MIRAGE's own pipeline handles both, identically to how it
                # handled GOALS for the published baseline.
                with open(os.path.join(args.out, split, 'bscan', out_name),
                          'wb') as fh:
                    fh.write(z.read(b))

                seg = np.array(Image.open(io.BytesIO(z.read(s))).convert('L'))
                mapped, unmapped = remap_mask(seg, table)
                mapped, _ = add_choroid_band(mapped, seg, ds,
                                             args.choroid_band)
                unmapped_all[ds] |= unmapped
                Image.fromarray(mapped).save(
                    os.path.join(args.out, split, 'semseg', out_name))

                counts[ds][split] += 1
                sizes[ds].add(seg.shape)
                for v, c in zip(*np.unique(mapped, return_counts=True)):
                    pixels[ds][int(v)] += int(c)

    with open(os.path.join(args.out, 'INFO.json'), 'w') as f:
        json.dump(INFO, f, indent=4)

    print('=' * 72)
    print('MergedV2 -> %s' % args.out)
    print('  native resolution preserved, images copied verbatim')
    print('=' * 72)
    print('%-10s %8s %8s %8s %8s  %s'
          % ('dataset', 'train', 'val', 'test', 'total', 'native size(s)'))
    tot = defaultdict(int)
    for ds in counts:
        r = counts[ds]
        sz = ','.join('%dx%d' % s for s in sorted(sizes[ds])[:2])
        if len(sizes[ds]) > 2:
            sz += ',+%d' % (len(sizes[ds]) - 2)
        print('%-10s %8d %8d %8d %8d  %s'
              % (ds, r['train'], r['val'], r['test'], sum(r.values()), sz))
        for k in r:
            tot[k] += r[k]
    print('%-10s %8d %8d %8d %8d'
          % ('TOTAL', tot['train'], tot['val'], tot['test'], sum(tot.values())))

    print()
    print('Pixel composition per source')
    print('%-10s %12s %12s %12s %12s'
          % ('dataset', 'Elsewhere', 'InnerRetina', 'Choroid', 'ignored'))
    for ds in pixels:
        t = sum(pixels[ds].values())
        print('%-10s %12.4f %12.4f %12.4f %12.4f'
              % (ds,
                 pixels[ds].get(V_ELSEWHERE, 0) / t,
                 pixels[ds].get(V_INNER, 0) / t,
                 pixels[ds].get(V_CHOROID, 0) / t,
                 pixels[ds].get(V_IGNORE, 0) / t))

    print()
    bad = {k: v for k, v in unmapped_all.items() if v}
    if bad:
        for ds, u in bad.items():
            print('WARNING %s had UNMAPPED values %s (forced to ignore)'
                  % (ds, sorted(u)))
    else:
        print('Every source value was explicitly mapped. No silent fallbacks.')

    if assign:
        with open(os.path.join(args.out, 'split_assignment.txt'), 'w') as f:
            f.write('\n'.join(assign) + '\n')
        print('Subject-level split written to split_assignment.txt')


if __name__ == '__main__':
    main()
