#!/usr/bin/env python
"""Decisive test: does the model's INPUT SCALE explain the FairVision failure?

MIRAGE's segmentation fine-tuning hardcodes ``model_config.norm = 'minmax'``
(``run_seg_tuning.py:416``).  In ``simple_transform`` that string falls through
to ``else: pass`` — no normalisation op is appended — so the tensor the model
sees during training is whatever ``ToTensorV2`` makes of the PNG: **uint8 in
[0, 255]**.  Verified empirically: the transform returns
``dtype=torch.uint8, min=0, max=253``.

Our FairVision inference (``preprocess_mirage_masks.py:normalize_resize_batch``)
feeds **float32 in [0, 1]**.  That is a ~255x scale difference on the input.

If that mismatch is real and consequential, feeding [0, 255] instead should
change the segmentation materially and reduce topology violations.  If it
changes nothing, the scale is absorbed somewhere and the transfer failure has a
different cause — which is equally worth knowing before spending a training run.

Reported per scaling, on identical slices:
  * topology violation rate — columns whose layer order is anatomically
    impossible (RNFL must sit above GCIPL above choroid)
  * mean vertical runs per occupied column of the raw union (1.0 = unbroken)
  * raw union area as a fraction of the frame
  * per-pixel agreement between the two scalings

    python scripts/mirage_input_scale_probe.py --volumes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

MIRAGE_WS = Path(r'D:\jepa_phase0\mirage-goals')
DATA = Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')

CLASS_RNFL, CLASS_GCIPL, CLASS_CHOROID = 1, 2, 3
UNION = (CLASS_RNFL, CLASS_GCIPL, CLASS_CHOROID)


def topology_violation(hard):
    """Fraction of evaluable columns whose layer order is impossible."""
    bad = total = 0
    for c in range(hard.shape[1]):
        col = hard[:, c]
        pos = []
        for cls in (CLASS_RNFL, CLASS_GCIPL, CLASS_CHOROID):
            r = np.where(col == cls)[0]
            if r.size:
                pos.append(r.mean())
        if len(pos) < 2:
            continue
        total += 1
        if any(pos[i] >= pos[i + 1] for i in range(len(pos) - 1)):
            bad += 1
    return bad, total


def mean_runs(mask):
    vals = []
    for c in range(mask.shape[1]):
        col = mask[:, c]
        if not col.any():
            continue
        vals.append(int(np.sum(col[1:] & ~col[:-1])) + int(col[0]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--volumes', type=int, default=20)
    ap.add_argument('--slices', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, str(MIRAGE_WS))
    os.chdir(MIRAGE_WS)
    import cv2
    import orchestrate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = orchestrate.build_trained_model(device)
    model.eval()

    files = sorted(DATA.glob('*.npz'))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(files), size=min(args.volumes, len(files)), replace=False)
    depths = np.linspace(20, 180, num=args.slices).astype(int)

    stats = {s: {'bad': 0, 'tot': 0, 'runs': [], 'area': []}
             for s in ('unit', 'byte')}
    agree = []

    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            big = cv2.resize(unit, (1024, 1024), interpolation=cv2.INTER_LINEAR)

            hard_of = {}
            for scale, arr in (('unit', big), ('byte', big * 255.0)):
                t = torch.from_numpy(arr)[None, None].to(device=device,
                                                         dtype=torch.float32)
                with torch.inference_mode(), torch.autocast(
                        device_type='cuda', dtype=torch.float16,
                        enabled=device == 'cuda'):
                    out = model({'bscan': t})
                logits = out['semseg'] if isinstance(out, dict) else out
                hard = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
                hard = cv2.resize(hard, (200, 200), interpolation=cv2.INTER_NEAREST)
                hard_of[scale] = hard

                b, tt = topology_violation(hard)
                stats[scale]['bad'] += b
                stats[scale]['tot'] += tt
                union = np.isin(hard, UNION)
                stats[scale]['runs'] += mean_runs(union)
                stats[scale]['area'].append(float(union.mean()))
            agree.append(float((hard_of['unit'] == hard_of['byte']).mean()))

    n = len(pick) * len(depths)
    print('slices: %d   (%d volumes x %d depths)' % (n, len(pick), len(depths)))
    print()
    print('%-26s %14s %14s' % ('metric', 'input [0,1]', 'input [0,255]'))
    print('-' * 58)
    print('%-26s %14s %14s' % ('  (what we feed now)', 'CURRENT', 'TRAINING-MATCHED'))
    print('%-26s %14.4f %14.4f'
          % ('topology violation rate',
             stats['unit']['bad'] / max(stats['unit']['tot'], 1),
             stats['byte']['bad'] / max(stats['byte']['tot'], 1)))
    print('%-26s %14.3f %14.3f'
          % ('mean runs / column',
             float(np.mean(stats['unit']['runs'])),
             float(np.mean(stats['byte']['runs']))))
    print('%-26s %14.4f %14.4f'
          % ('raw union area',
             float(np.mean(stats['unit']['area'])),
             float(np.mean(stats['byte']['area']))))
    print()
    print('per-pixel agreement between the two scalings: %.4f' % float(np.mean(agree)))
    print()
    print('Reference: the stored guides were built with the [0,1] path and')
    print('measured 19% of slices above 5% topology violation, raw union area')
    print('0.1368, and mean runs/column 2.13 before repair.')


if __name__ == '__main__':
    main()
