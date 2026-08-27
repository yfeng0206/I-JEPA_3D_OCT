"""Measure the VERTICAL EXTENT of predicted choroid on FairVision.

Motivation
----------
The v3 dataset added a *bounded* synthetic choroid band (0.055 H below
Bruch's) to Duke/AROI so the class had evidence outside GOALS.  The whole
safety argument for that band was that it is BOUNDED: under-bounding can
only withhold supervision, it can never assert a wrong label.  The failure
mode it was designed to avoid is the naive sub-BM remap, whose band is
unbounded and runs to the image bottom (choroid + sclera + background).

On FairVision the v3 model predicts choroid_area 0.144 vs the baseline's
0.074 -- roughly 2x -- and that number is still growing with training
(0.129 at ep35 -> 0.144 at ep104).  Area alone cannot distinguish
"correctly thicker choroid" from "band leaking downward to the image
bottom", and choroid_above_rpe cannot see it either because it only looks
upward.  So measure the extent directly instead of reasoning about it.

Reported per arm (200x200 analysis grid, same as fairvision_model_compare):
  choroid_top_frac      mean first choroid row / H
  choroid_bot_frac      mean last choroid row / H
  choroid_thick_frac    mean (bottom - top + 1) / H
  bottom_touch_rate     fraction of choroid columns whose choroid reaches
                        the final image row -- THE failure signature
  bot_minus_rpe_frac    mean (choroid bottom - RPE row) / H
  top_minus_rpe_frac    mean (choroid top - RPE row) / H  (>0 == below RPE)

Reference values.  The GOALS row is measured HERE, on the 30 GOALS-test
ground-truth masks at this same 200x200 grid, because an earlier
hand-carried figure ("top 0.41, bottom 0.56") turned out to disagree with
the annotations and would have inverted the interpretation:
  GOALS choroid   top 0.457 H, bottom 0.502 H, thickness 0.050 H,
                  area 0.050, col coverage 1.00, reaches bottom NEVER
  AROI sub-BM     top 0.28 H, bottom 1.00 H, area 0.559, ALWAYS reaches bottom
  Duke sub-BM     top 0.32 H, bottom 1.00 H, area 0.552, ALWAYS reaches bottom

Two separate questions, and they need two separate metrics:
  * bottom_touch_rate is the FAILURE-MODE discriminator: ~0 means the model
    kept the bounded-band concept, ~1 means it regressed to the sub-BM blob.
  * choroid_thick_frac is the CALIBRATION metric: how the predicted band
    thickness compares with the GOALS annotation convention (0.050 H).
    A bounded band can still be badly over-thick, and area/above-RPE
    metrics are both blind to that -- area conflates thickness with
    coverage, and above-RPE only ever looks upward.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fairvision_model_compare import (  # noqa: E402
    ARMS, BASELINE_CKPT, DATA, MIRAGE_WS, build_model,
)


def column_stats(chor: np.ndarray, rpe_rows: np.ndarray):
    """Per-column vertical extent of the choroid mask.

    Returns (top_frac, bot_frac, thick_frac, touch, bot_rel_rpe, top_rel_rpe)
    as lists over columns that actually contain choroid.  Columns without
    choroid carry no extent information and are excluded rather than
    counted as zero, which would silently bias every mean downward.
    """
    h, w = chor.shape
    tops, bots, thicks, touches, botrel, toprel = [], [], [], [], [], []
    for c in range(w):
        rows = np.flatnonzero(chor[:, c])
        if rows.size == 0:
            continue
        top, bot = int(rows[0]), int(rows[-1])
        tops.append(top / h)
        bots.append(bot / h)
        thicks.append((bot - top + 1) / h)
        touches.append(1.0 if bot >= h - 1 else 0.0)
        botrel.append((bot - int(rpe_rows[c])) / h)
        toprel.append((top - int(rpe_rows[c])) / h)
    return tops, bots, thicks, touches, botrel, toprel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged-ckpt', required=True)
    ap.add_argument('--baseline-ckpt', default=str(BASELINE_CKPT))
    ap.add_argument('--volumes', type=int, default=8)
    ap.add_argument('--slices', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.chdir(MIRAGE_WS)
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = sorted(DATA.glob('*.npz'))
    if not files:
        raise SystemExit('no FairVision volumes at %s' % DATA)

    # Identical selection to fairvision_model_compare so the numbers line up
    # with the metrics already reported for these same slices.
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(files), size=min(args.volumes, len(files)),
                      replace=False)
    depths = np.linspace(20, 180, num=args.slices).astype(int)

    inputs = []
    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            inputs.append(cv2.resize(unit, (1024, 1024),
                                     interpolation=cv2.INTER_LINEAR))
    print('slices: %d  (%d volumes x %d depths)'
          % (len(inputs), len(pick), len(depths)))

    rpe_rows = []
    for a in inputs:
        s = cv2.resize(a, (200, 200), interpolation=cv2.INTER_LINEAR)
        prof = cv2.GaussianBlur(s, (1, 9), 0)
        rpe_rows.append(np.argmax(prof, axis=0))

    ckpts = {'baseline': pathlib.Path(args.baseline_ckpt),
             'merged': pathlib.Path(args.merged_ckpt)}
    results = {}

    for arm, (_inner_cls, chor_cls, n_logits, ignore_cls) in ARMS.items():
        ckpt = ckpts[arm]
        if not ckpt.exists():
            raise SystemExit('missing checkpoint for %s: %s' % (arm, ckpt))
        print('\n[%s] %s' % (arm, ckpt))
        model = build_model(n_logits, ckpt, device)

        tops, bots, thicks, touches, botrel, toprel = [], [], [], [], [], []
        areas = []
        for si, arr in enumerate(inputs):
            t = torch.from_numpy(arr)[None, None].to(device=device,
                                                     dtype=torch.float32)
            with torch.inference_mode(), torch.autocast(
                    device_type='cuda', dtype=torch.float16,
                    enabled=device == 'cuda'):
                out = model({'bscan': t})
            logits = out['semseg'] if isinstance(out, dict) else out
            logits = logits.float().clone()
            if ignore_cls is not None:
                logits[:, ignore_cls] = float('-inf')
            hard = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            hard = cv2.resize(hard, (200, 200),
                              interpolation=cv2.INTER_NEAREST)

            chor = np.isin(hard, chor_cls)
            areas.append(float(chor.mean()))
            a, b, c, d, e, f = column_stats(chor, rpe_rows[si])
            tops += a
            bots += b
            thicks += c
            touches += d
            botrel += e
            toprel += f

        del model
        torch.cuda.empty_cache()

        results[arm] = {
            'n_choroid_columns': len(tops),
            'choroid_area': float(np.mean(areas)),
            'choroid_top_frac': float(np.mean(tops)) if tops else None,
            'choroid_bot_frac': float(np.mean(bots)) if bots else None,
            'choroid_thick_frac': float(np.mean(thicks)) if thicks else None,
            'bottom_touch_rate': float(np.mean(touches)) if touches else None,
            'bot_minus_rpe_frac': float(np.mean(botrel)) if botrel else None,
            'top_minus_rpe_frac': float(np.mean(toprel)) if toprel else None,
        }

    print('\n' + '=' * 72)
    print('  CHOROID VERTICAL EXTENT  (200x200 grid, fractions of image H)')
    print('=' * 72)
    keys = ['choroid_area', 'choroid_top_frac', 'choroid_bot_frac',
            'choroid_thick_frac', 'bottom_touch_rate',
            'top_minus_rpe_frac', 'bot_minus_rpe_frac']
    print('  %-22s %10s %10s %10s' % ('metric', 'baseline', 'merged', 'delta'))
    for k in keys:
        bv, mv = results['baseline'][k], results['merged'][k]
        if bv is None or mv is None:
            continue
        print('  %-22s %10.4f %10.4f %+10.4f' % (k, bv, mv, mv - bv))

    print('\n  REFERENCE (GOALS row measured on the 30 GOALS-test GT masks):')
    print('    GOALS choroid : top 0.457  bottom 0.502  thick 0.050  touch 0.00')
    print('    AROI  sub-BM  : top 0.28   bottom 1.00                touch 1.00')
    print('    Duke  sub-BM  : top 0.32   bottom 1.00                touch 1.00')
    print('\n  bottom_touch_rate near 0 => bounded band concept preserved.')
    print('  bottom_touch_rate near 1 => regressed to the sub-BM blob.')
    print('  choroid_thick_frac vs 0.050 => calibration against the GOALS')
    print('  annotation convention (separate question from the failure mode).')

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            'n_slices': len(inputs), 'volumes': int(len(pick)),
            'depths': [int(d) for d in depths], 'seed': args.seed,
            'merged_ckpt': str(ckpts['merged']),
            'arms': results,
        }, indent=2))
        print('\nwrote %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
