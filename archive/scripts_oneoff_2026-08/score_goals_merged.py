"""Score segmentation predictions on the GOALS test set in the MERGED 3-class
taxonomy, so a 4-class GOALS-only baseline and a 3-class merged-data model can
be compared on identical ground truth.

Why this is needed
------------------
The published baseline number (all-class Dice 0.9426 / foreground 0.9251) is a
FOUR-class score: Elsewhere / RNFL / GCIPL / Choroid.  The merged model emits
THREE classes because RNFL and GCIPL were fused into InnerRetina.  Those two
numbers are not comparable -- fusing two classes generally RAISES Dice, because
every RNFL<->GCIPL boundary confusion silently becomes a correct pixel.

So we re-score the baseline's own saved predictions after applying the same
fusion.  Both arms are then measured with one metric, on one ground truth, on
the same 30 images.

Ground truth is taken from the merged build (already verified byte-identical to
the official GOALS test masks before remapping), and both prediction sets are
compared at 1024x1024, which is the resolution the baseline's predictions were
saved at and the resolution the model actually predicts.

Usage
-----
    python scripts/score_goals_merged.py \
        --baseline-preds <dir with 0071.png ... in 0/80/160/255> \
        --merged-preds   <dir with predictions in 0/128/255>
Either arm may be omitted.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image

ELSEWHERE, INNER, CHOROID = 0, 1, 2
NAMES = {ELSEWHERE: 'Elsewhere', INNER: 'InnerRetina', CHOROID: 'Choroid'}

# GOALS native palette -> merged index
GOALS_4CLASS = {0: ELSEWHERE, 80: INNER, 160: INNER, 255: CHOROID}
# merged palette -> merged index
MERGED_3CLASS = {0: ELSEWHERE, 128: INNER, 255: CHOROID, 1: 255}


def to_index(arr: np.ndarray, palette: dict) -> np.ndarray:
    out = np.full(arr.shape, 255, dtype=np.uint8)
    seen = np.zeros(arr.shape, dtype=bool)
    for value, idx in palette.items():
        m = arr == value
        out[m] = idx
        seen |= m
    if not seen.all():
        extra = sorted(np.unique(arr[~seen]).tolist())
        raise ValueError('unmapped pixel values %s for palette %s'
                         % (extra, sorted(palette)))
    return out


def load_indexed(path: pathlib.Path, palette: dict, size: int) -> np.ndarray:
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    idx = to_index(a, palette)
    if idx.shape != (size, size):
        idx = np.array(Image.fromarray(idx).resize((size, size), Image.NEAREST))
    return idx


def dice_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Per-class Dice and IoU over the valid region only."""
    out = {}
    for c in (ELSEWHERE, INNER, CHOROID):
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter = float(np.logical_and(p, g).sum())
        psum, gsum = float(p.sum()), float(g.sum())
        union = psum + gsum - inter
        out[c] = {
            'dice': (2 * inter / (psum + gsum)) if (psum + gsum) > 0 else np.nan,
            'iou': (inter / union) if union > 0 else np.nan,
            'gt_pixels': gsum,
        }
    return out


def score_arm(pred_dir: pathlib.Path, gt_dir: pathlib.Path, palette: dict,
              size: int, label: str):
    gts = sorted(gt_dir.glob('*.png'))
    if not gts:
        raise SystemExit('no ground-truth masks in %s' % gt_dir)

    per_class = {c: {'dice': [], 'iou': []} for c in (ELSEWHERE, INNER, CHOROID)}
    n = 0
    missing = []
    for gt_path in gts:
        stem = gt_path.name.split('__', 1)[-1]
        cand = pred_dir / stem
        if not cand.exists():
            alt = list(pred_dir.glob(pathlib.Path(stem).stem + '.*'))
            if not alt:
                missing.append(stem)
                continue
            cand = alt[0]
        gt = load_indexed(gt_path, MERGED_3CLASS, size)
        pred = load_indexed(cand, palette, size)
        valid = gt != 255
        r = dice_iou(pred, gt, valid)
        for c in per_class:
            if not np.isnan(r[c]['dice']):
                per_class[c]['dice'].append(r[c]['dice'])
                per_class[c]['iou'].append(r[c]['iou'])
        n += 1

    if missing:
        print('  WARNING: %d ground-truth masks had no prediction: %s'
              % (len(missing), missing[:5]))
    if n == 0:
        raise SystemExit('no matched prediction/ground-truth pairs for ' + label)

    summary = {'arm': label, 'n_images': n, 'per_class': {}}
    print('\n%s  (n=%d, scored at %dx%d)' % (label, n, size, size))
    print('  %-12s %8s %8s' % ('class', 'Dice', 'IoU'))
    for c in (ELSEWHERE, INNER, CHOROID):
        d = float(np.mean(per_class[c]['dice']))
        i = float(np.mean(per_class[c]['iou']))
        summary['per_class'][NAMES[c]] = {'dice': d, 'iou': i,
                                          'n': len(per_class[c]['dice'])}
        print('  %-12s %8.4f %8.4f' % (NAMES[c], d, i))

    all_d = float(np.mean([summary['per_class'][NAMES[c]]['dice']
                           for c in (ELSEWHERE, INNER, CHOROID)]))
    fg_d = float(np.mean([summary['per_class'][NAMES[c]]['dice']
                          for c in (INNER, CHOROID)]))
    all_i = float(np.mean([summary['per_class'][NAMES[c]]['iou']
                           for c in (ELSEWHERE, INNER, CHOROID)]))
    summary['all_class_mean_dice'] = all_d
    summary['foreground_mean_dice'] = fg_d
    summary['mean_iou'] = all_i
    print('  %-12s %8.4f  (all classes)' % ('mean Dice', all_d))
    print('  %-12s %8.4f  (InnerRetina + Choroid)' % ('fg Dice', fg_d))
    print('  %-12s %8.4f' % ('mean IoU', all_i))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=r'D:\jepa_phase0\mirage-datasets\MergedV2\test\semseg')
    ap.add_argument('--baseline-preds', default=None,
                    help='4-class GOALS-only predictions (0/80/160/255)')
    ap.add_argument('--merged-preds', default=None,
                    help='3-class merged-model predictions (0/128/255)')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    gt_dir = pathlib.Path(args.gt)
    results = []

    print('=' * 70)
    print('GOALS test, scored in the MERGED 3-class taxonomy')
    print('ground truth:', gt_dir)
    print('=' * 70)

    if args.baseline_preds:
        results.append(score_arm(pathlib.Path(args.baseline_preds), gt_dir,
                                 GOALS_4CLASS, args.size,
                                 'BASELINE (GOALS-only, RNFL+GCIPL fused)'))
    if args.merged_preds:
        results.append(score_arm(pathlib.Path(args.merged_preds), gt_dir,
                                 MERGED_3CLASS, args.size,
                                 'MERGED (GOALS+Duke+AROI)'))

    if len(results) == 2:
        b, m = results
        print('\n' + '=' * 70)
        print('DELTA  (merged - baseline), positive favours the merged model')
        print('=' * 70)
        for name in ('Elsewhere', 'InnerRetina', 'Choroid'):
            print('  %-12s Dice %+.4f   IoU %+.4f'
                  % (name,
                     m['per_class'][name]['dice'] - b['per_class'][name]['dice'],
                     m['per_class'][name]['iou'] - b['per_class'][name]['iou']))
        for k in ('all_class_mean_dice', 'foreground_mean_dice', 'mean_iou'):
            print('  %-22s %+.4f   (%.4f -> %.4f)'
                  % (k, m[k] - b[k], b[k], m[k]))
        print('\nNote: GOALS test is the SOURCE domain for the baseline and only')
        print('4.4% of the merged training set. It is a REGRESSION GUARDRAIL,')
        print('not the endpoint. The endpoint is FairVision transfer.')

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
        print('\nwrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
