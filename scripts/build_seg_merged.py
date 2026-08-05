#!/usr/bin/env python
"""Build a merged retinal-layer segmentation dataset in MIRAGE's format.

Combines GOALS, Duke DME and AROI into one dataset that trains a 4-class model
(Elsewhere / RNFL / GCIPL / Choroid) — the SAME taxonomy the current
FairVision guide is built from, so the guide definition does not change.

Why each source needs different handling
----------------------------------------
Verified empirically (see `scripts/seg_merge_validate.py` and the class
row-position check):

  GOALS     80=RNFL 160=GCIPL 255=Choroid.  Full supervision.
  Duke DME  classes are named by their UPPER boundary, so 25("ILM")=RNFL and
            51("NFL")=GCIPL.  Its 204("BM") runs to the image bottom and so
            includes sclera; GOALS choroid is a bounded band, therefore BM is
            NOT mapped to choroid — it becomes ignore.
  AROI      23("ILM-IPL/INL") is RNFL and GCIPL MERGED and cannot be split.
            It becomes a SUPERCLASS label: the loss only penalises predictions
            that fall outside {RNFL, GCIPL}.  92("Under BM") — same sclera
            problem as Duke — becomes ignore.  Lesions become ignore.

Two domain corrections applied at build time
--------------------------------------------
L2 GEOMETRY.  MIRAGE resizes everything to a square 1024x1024.  The sources are
not square, so each is distorted by a different amount (GOALS 1.375, AROI 1.032,
FairVision 1.000).  A layer of fixed anatomical thickness therefore lands on 38%
more model pixels when it comes from GOALS.  Every image is squared HERE so all
data matches the target domain's stored aspect.

L1 INPUT.  MIRAGE normalises with fixed ImageNet constants, not per-image
statistics, so contrast differences survive (GOALS std 41.4 vs FairVision 22.8).
Per-image min-max is applied here, matching the preprocessing the original
FairVision transfer run used.

Output label encoding (`INFO.json` maps VALUE -> class index)
-------------------------------------------------------------
    0   Elsewhere        index 0
    80  RNFL             index 1
    160 GCIPL            index 2
    255 Choroid          index 3
    200 InnerRetina      index 4  <- SUPERCLASS, never predicted by the model
    1   Background       index 5  <- ignore (MIRAGE auto-detects "background")

The model still emits 4 logits.  Indices 4 and 5 exist only in the label tensor
and are consumed by `src/losses/partial_label.py`.

    python scripts/build_seg_merged.py --out D:\\jepa_phase0\\mirage-datasets\\Merged_v1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --- output taxonomy -------------------------------------------------------
V_ELSEWHERE, V_RNFL, V_GCIPL, V_CHOROID = 0, 80, 160, 255
V_INNER = 200        # superclass: RNFL or GCIPL, unknown which
V_IGNORE = 1         # excluded from the loss

INFO = {
    "0": {"value": V_ELSEWHERE, "label": "Elsewhere"},
    "1": {"value": V_RNFL, "label": "RNFL"},
    "2": {"value": V_GCIPL, "label": "GCIPL"},
    "3": {"value": V_CHOROID, "label": "Choroid"},
    "4": {"value": V_INNER, "label": "InnerRetina"},
    "5": {"value": V_IGNORE, "label": "Background"},
}

SOURCES = {
    'GOALS': r'D:\jepa_phase0\mirage-goals\downloads\GOALS.zip',
    'Duke_DME': r'D:\jepa_phase0\mirage-datasets\Duke_DME.zip',
    'AROI': r'D:\jepa_phase0\mirage-datasets\AROI.zip',
}

# source value -> output value.  Anything absent from a table becomes V_IGNORE,
# which is deliberate: an unmapped class must never silently become Elsewhere.
REMAP = {
    'GOALS': {0: V_ELSEWHERE, 80: V_RNFL, 160: V_GCIPL, 255: V_CHOROID},
    'Duke_DME': {
        0: V_ELSEWHERE,
        25: V_RNFL,          # region below ILM   = RNFL
        51: V_GCIPL,         # region below NFL   = GCL+IPL
        76: V_ELSEWHERE,     # INL  ) the mid-retina GOALS leaves unlabelled,
        102: V_ELSEWHERE,    # OPL  ) so Elsewhere is the correct target here
        127: V_ELSEWHERE,    # ONL  )
        153: V_ELSEWHERE,    # ISM
        178: V_ELSEWHERE,    # OS
        204: V_IGNORE,       # sub-BM: includes sclera, not GOALS' choroid
        229: V_IGNORE,       # fluid
    },
    'AROI': {
        0: V_ELSEWHERE,      # above ILM
        23: V_INNER,         # ILM-IPL/INL = RNFL+GCIPL merged -> superclass
        46: V_ELSEWHERE,     # IPL/INL-RPE = mid/outer retina
        69: V_ELSEWHERE,     # RPE-BM
        92: V_IGNORE,        # sub-BM: includes sclera
        115: V_IGNORE,       # cyst
        138: V_IGNORE,       # PED
        161: V_IGNORE,       # SRF
    },
}


def subject_of(dataset, filename):
    """Group key for leakage-free splitting.

    Adjacent B-scans from one eye are near-duplicates, so splits must be by
    subject.  GOALS filenames carry no subject id, so GOALS keeps MIRAGE's own
    official split instead (handled by the caller).
    """
    stem = os.path.splitext(filename)[0]
    if dataset == 'Duke_DME':          # Subject_10_5 -> Subject_10
        return '_'.join(stem.split('_')[:2])
    if dataset == 'AROI':              # patient9_0057 -> patient9
        return stem.split('_')[0]
    return stem


def square_and_normalise(img, seg, size):
    """Apply the two build-time domain corrections.

    Image: per-image min-max to a full 0-255 range (removes the contrast gap),
    then a square resize so every source lands at the same anisotropy the
    target domain has.
    Mask: identical square resize with NEAREST so labels stay exact.
    """
    a = img.astype(np.float32)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
    im = Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    sm = Image.fromarray(seg.astype(np.uint8)).resize((size, size), Image.NEAREST)
    return np.array(im), np.array(sm)


def remap_mask(seg, table):
    out = np.full(seg.shape, V_IGNORE, dtype=np.uint8)
    seen = set(np.unique(seg).tolist())
    for src, dst in table.items():
        out[seg == src] = dst
    unmapped = seen - set(table)
    return out, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\jepa_phase0\mirage-datasets\Merged_v1')
    ap.add_argument('--size', type=int, default=512,
                    help='Square build size. The pipeline resizes to 1024 later; '
                         'building square is what removes the anisotropy.')
    ap.add_argument('--val-frac', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--datasets', nargs='+', default=['GOALS', 'Duke_DME', 'AROI'])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for split in ('train', 'val', 'test'):
        for sub in ('bscan', 'semseg'):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))
    pixel_hist = defaultdict(lambda: defaultdict(int))
    unmapped_all = defaultdict(set)
    assign_log = []

    for ds in args.datasets:
        path = SOURCES.get(ds)
        if not path or not os.path.isfile(path):
            print('SKIP %s (not found)' % ds)
            continue
        table = REMAP[ds]
        with zipfile.ZipFile(path) as z:
            bscans = sorted(x for x in z.namelist()
                            if '/bscan/' in x and x.endswith('.png'))
            # Split policy, chosen so the headline metric stays comparable to
            # the GOALS-only baseline (test Dice 0.9426):
            #
            #   GOALS     keeps MIRAGE's OFFICIAL train/val/test.  Its test set
            #             is therefore byte-identical to the baseline's, so the
            #             two runs can be compared directly.
            #   Duke DME  subject-split into train/val only.  It has hard
            #             RNFL/GCIPL labels so it can validate the model.
            #   AROI      100% TRAIN.  Its inner retina is a SUPERCLASS, which
            #             MIRAGE's mean_iou cannot score (it would treat label
            #             4 as a 5th class), and it carries no hard RNFL/GCIPL
            #             labels so it could not validate the split anyway.
            if ds == 'GOALS':
                split_of = {b: b.split('/')[1] for b in bscans}
            elif ds == 'AROI':
                split_of = {b: 'train' for b in bscans}
            else:
                subs = sorted({subject_of(ds, b.split('/')[-1]) for b in bscans})
                perm = rng.permutation(len(subs))
                n_val = max(1, int(round(len(subs) * args.val_frac)))
                pick = {}
                for j, i in enumerate(perm):
                    pick[subs[int(i)]] = 'val' if j < n_val else 'train'
                split_of = {b: pick[subject_of(ds, b.split('/')[-1])] for b in bscans}
                for s in subs:
                    assign_log.append('%-9s %-14s -> %s' % (ds, s, pick[s]))

            for b in bscans:
                s = b.replace('/bscan/', '/semseg/')
                if s not in z.namelist():
                    continue
                img = np.array(Image.open(io.BytesIO(z.read(b))).convert('L'))
                seg = np.array(Image.open(io.BytesIO(z.read(s))).convert('L'))
                mapped, unmapped = remap_mask(seg, table)
                unmapped_all[ds] |= unmapped
                img2, seg2 = square_and_normalise(img, mapped, args.size)
                split = split_of[b]
                name = '%s__%s' % (ds, b.split('/')[-1])
                Image.fromarray(img2).save(os.path.join(args.out, split, 'bscan', name))
                Image.fromarray(seg2).save(os.path.join(args.out, split, 'semseg', name))
                counts[ds][split] += 1
                for v, c in zip(*np.unique(seg2, return_counts=True)):
                    pixel_hist[ds][int(v)] += int(c)

    with open(os.path.join(args.out, 'INFO.json'), 'w') as f:
        json.dump(INFO, f, indent=4)

    label_of = {v['value']: v['label'] for v in INFO.values()}
    print('=' * 74)
    print('MERGED DATASET  ->  %s   (square %d, per-image min-max)'
          % (args.out, args.size))
    print('=' * 74)
    print('%-10s %8s %8s %8s %8s' % ('dataset', 'train', 'val', 'test', 'total'))
    tot = defaultdict(int)
    for ds in counts:
        r = counts[ds]
        print('%-10s %8d %8d %8d %8d'
              % (ds, r['train'], r['val'], r['test'], sum(r.values())))
        for k in r:
            tot[k] += r[k]
    print('%-10s %8d %8d %8d %8d'
          % ('TOTAL', tot['train'], tot['val'], tot['test'], sum(tot.values())))

    print()
    print('Label composition per source (fraction of pixels)')
    hdr = ['Elsewhere', 'RNFL', 'GCIPL', 'Choroid', 'InnerRetina', 'Background']
    print('%-10s ' % 'dataset' + ' '.join('%11s' % h for h in hdr))
    for ds in pixel_hist:
        total = sum(pixel_hist[ds].values())
        row = []
        for v in (V_ELSEWHERE, V_RNFL, V_GCIPL, V_CHOROID, V_INNER, V_IGNORE):
            row.append('%11.4f' % (pixel_hist[ds].get(v, 0) / total))
        print('%-10s ' % ds + ' '.join(row))

    print()
    for ds, u in unmapped_all.items():
        if u:
            print('WARNING %s had UNMAPPED source values %s -> forced to ignore'
                  % (ds, sorted(u)))
    if not any(unmapped_all.values()):
        print('All source values were explicitly mapped. No silent fallbacks.')

    if assign_log:
        with open(os.path.join(args.out, 'split_assignment.txt'), 'w') as f:
            f.write('\n'.join(assign_log) + '\n')
        print('Subject-level split assignment written to split_assignment.txt')


if __name__ == '__main__':
    main()
