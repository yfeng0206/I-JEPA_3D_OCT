"""Measure what the A' intersection does to target geometry.

A' (target = block INTERSECT anatomy) guarantees zero background target cells,
but it is not free: intersecting a rectangle with a thin, diagonal retinal band
can both shrink the target and BREAK IT INTO PIECES.  I-JEPA's own ablation is
explicit that this matters -- multi-block 54.2% vs scattered individual patches
17.6%, and shrinking target blocks cost 54.2% -> 19.2% -- so a design that
silently fragments or shrinks targets is a design that undoes the very prior it
is trying to exploit.

This script quantifies, over many random crops and slices:

  * target area, rectangles vs A'
  * connected components of the target set, rectangles vs A'
  * largest-component share
  * how much bigger the blocks would have to be for A' to recover the intended
    area (the compensation factor)

Everything runs against the production sampler and the production paired crop.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from demo_guided_masking import (  # noqa: E402
    CURRICULUM_CFG, GRID, cells_to_image, run_once,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402


def components(mask: np.ndarray):
    """4-connected component count and largest-component share."""
    lab = np.zeros_like(mask, dtype=np.int32)
    cur = 0
    sizes = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or lab[r, c]:
                continue
            cur += 1
            stack = [(r, c)]
            lab[r, c] = cur
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                            and mask[yy, xx] and not lab[yy, xx]):
                        lab[yy, xx] = cur
                        stack.append((yy, xx))
            sizes.append(n)
    if not sizes:
        return 0, 0.0
    return len(sizes), max(sizes) / float(sum(sizes))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', required=True)
    ap.add_argument('--threshold', type=float, default=0.0868)
    ap.add_argument('--draws', type=int, default=25,
                    help='random crops per slice')
    ap.add_argument('--scale', type=float, nargs=2, default=None,
                    help='override pred_mask_scale, e.g. --scale 0.30 0.40')
    a = ap.parse_args()

    z = np.load(a.guides, allow_pickle=True)
    images, probs = z['images'], z['probs']
    n = len(images)

    cfg = dict(CURRICULUM_CFG)
    scale = tuple(a.scale) if a.scale else (0.15, 0.2)
    gen = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=scale,
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False, curriculum_cfg=cfg)
    gen.set_epoch(99, 100)

    rec_area, ap_area = [], []
    rec_comp, ap_comp = [], []
    rec_share, ap_share = [], []
    accepted, feasible = [], []

    for i in range(n):
        for d in range(a.draws):
            seed = 1000 * i + d
            r = run_once(images[i], probs[i], gen, a.threshold, seed)
            rect = cells_to_image(r['union'])
            apm = cells_to_image(r['targets_ap'])
            rec_area.append(rect.sum())
            ap_area.append(apm.sum())
            c1, s1 = components(rect)
            c2, s2 = components(apm)
            rec_comp.append(c1); ap_comp.append(c2)
            rec_share.append(s1); ap_share.append(s2)
            accepted.append(float(r['stats']['accepted']))
            feasible.append(float(r['stats']['feasible']))

    f = lambda v: float(np.mean(v))  # noqa: E731
    keep = f(ap_area) / max(f(rec_area), 1e-9)
    print('slices %d x %d draws = %d masks   pred_mask_scale=%s  thr=%.4f'
          % (n, a.draws, len(rec_area), scale, a.threshold))
    print('\n%-28s %12s %12s' % ('metric', 'rectangles', "A' targets"))
    print('%-28s %12.2f %12.2f' % ('target cells (of 256)', f(rec_area), f(ap_area)))
    print('%-28s %12.4f %12.4f' % ('fraction of frame',
                                   f(rec_area) / 256.0, f(ap_area) / 256.0))
    print('%-28s %12.2f %12.2f' % ('connected components', f(rec_comp), f(ap_comp)))
    print('%-28s %12.4f %12.4f' % ('largest-component share',
                                   f(rec_share), f(ap_share)))
    print('\naccept rate %.4f   feasible rate %.4f' % (f(accepted), f(feasible)))
    print("A' keeps %.1f%% of the rectangle area" % (100 * keep))
    print('   -> to recover the intended target area, blocks must be %.2fx '
          'larger in area' % (1.0 / max(keep, 1e-9)))
    print('   -> i.e. pred_mask_scale approx (%.3f, %.3f)'
          % (scale[0] / max(keep, 1e-9), scale[1] / max(keep, 1e-9)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
