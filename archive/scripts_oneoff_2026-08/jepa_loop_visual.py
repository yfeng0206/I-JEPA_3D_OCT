#!/usr/bin/env python
"""The full I-JEPA masking loop on real B-scans, before and after rim growth.

Not plots -- the actual thing the encoder and predictor see:

  column 1  the B-scan
  column 2  MIRAGE anatomy region (what the guide says is tissue)
  column 3  BEFORE: context in grey, the 4 prediction targets in 4 colours
  column 4  AFTER:  same, with the guide's rim grown first

The change under test is deliberately minimal: the anatomy REGION is dilated
before the sampler runs, so target growth, geodesic partitioning and the
connectivity guarantee are all untouched. "Expand first, then patch."

Randomness is per (slice, epoch): how far the rim grows and in which
directions, so the same slice is not masked identically for 75 epochs.

Everything runs through the PRODUCTION CurriculumMaskGenerator, so what is
drawn is what training would actually use. Nothing here modifies src/.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.masks.curriculum import CurriculumMaskGenerator           # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID, PATCH, RES = 16, 16, 256
# 4 distinct target colours + grey context
TARGET_RGB = [(239, 71, 111), (255, 209, 102), (6, 214, 160), (17, 138, 178)]
CONTEXT_RGB = (150, 150, 150)
DROPPED_RGB = (35, 35, 40)
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def grow_rim(occ, radius, directions=None):
    """Grow the anatomy region outward. Never leaves the grid."""
    if radius <= 0:
        return occ.copy()
    m = occ > 0
    dirs = [DIRS[d] for d in (directions or DIRS)]
    out = m.copy()
    for _ in range(radius):
        g = out.copy()
        h, w = out.shape
        for dr, dc in dirs:
            s = np.zeros_like(out)
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            s[r0:r1, c0:c1] = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            g |= s
        out = g
    # New rim cells get a low but non-zero score so the sampler treats them as
    # admissible tissue edge rather than as core anatomy.
    grown = occ.copy()
    grown[out & (occ == 0)] = 0.30
    return grown


def draw(img, enc_idx, pred_idx_list):
    """Overlay context and the 4 targets onto the B-scan."""
    base = np.stack([img] * 3, -1)
    base = (base * 255).astype(np.uint8)
    canvas = (base * 0.45).astype(np.uint8)          # everything dimmed first
    def cells(idx):
        return [(int(i) // GRID, int(i) % GRID) for i in idx]
    for r, c in cells(enc_idx):                      # context: brightened
        y, x = r * PATCH, c * PATCH
        canvas[y:y + PATCH, x:x + PATCH] = base[y:y + PATCH, x:x + PATCH]
    for t, idx in enumerate(pred_idx_list):          # targets: tinted
        col = np.array(TARGET_RGB[t % 4], np.uint8)
        for r, c in cells(idx):
            y, x = r * PATCH, c * PATCH
            blk = base[y:y + PATCH, x:x + PATCH].astype(np.float32)
            canvas[y:y + PATCH, x:x + PATCH] = (0.45 * blk + 0.55 * col).astype(np.uint8)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--data', default=r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--radius', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    gfiles = sorted(pathlib.Path(a.guides).glob('*.npy'))
    picks = [gfiles[i] for i in rng.choice(len(gfiles), a.n, replace=False)]

    cfg = {'mode': 'mirage_anatomy', 'enabled': True, 'T_warm': 25, 'T_total': 30,
           'r_max': 1.0, 'ramp_shape': 'linear',
           'anatomy_mass_cap': 0.90, 'anatomy_tau': 0.10,
           'mirage_occupancy_threshold': 0.25}

    def masks_for(guide_grid, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen = CurriculumMaskGenerator(
            input_size=(RES, RES), patch_size=PATCH, npred=4, nenc=1,
            pred_mask_scale=(0.15, 0.2), pred_target_k=16, curriculum_cfg=cfg)
        gen.set_epoch(50, 100)
        occ = torch.from_numpy(guide_grid)[None].float()
        guides = torch.stack([occ, (occ >= 0.25).float()], 1)
        enc, pred = gen.generate(batch_size=1, guide_grids=guides,
                                 guide_valid=torch.ones(1, dtype=torch.bool))
        return enc[0][0].numpy(), [p[0].numpy() for p in pred]

    rows = []
    for gf in picks:
        vol = np.load(gf, mmap_mode='r')
        sl = int(rng.integers(25, 75))
        soft = np.asarray(vol[sl], np.float32) / 255.0
        occ_full = np.clip(soft[0] + soft[1], 0, 1)
        grid_occ = F.adaptive_avg_pool2d(
            torch.from_numpy(occ_full)[None, None], (GRID, GRID))[0, 0].numpy()
        if grid_occ.sum() < 1.0:
            continue
        img = np.asarray(F.interpolate(
            torch.from_numpy(occ_full)[None, None], size=(RES, RES),
            mode='bilinear', align_corners=False)[0, 0])
        npz = pathlib.Path(a.data) / (gf.stem + '.npz')
        if npz.exists():
            with np.load(npz, allow_pickle=True) as z:
                raw = np.asarray(z['oct_bscans'][sl * 2], np.float32)
            lo, hi = raw.min(), raw.max()
            raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0
            import cv2
            img = cv2.resize(raw, (RES, RES), interpolation=cv2.INTER_LINEAR)

        mode = rng.choice(['all', 'all', 'all', 'vertical', 'horizontal'])
        dirs = {'vertical': ['up', 'down'], 'horizontal': ['left', 'right']}.get(mode)
        grown = grow_rim(grid_occ, a.radius, dirs)

        e0, p0 = masks_for(grid_occ, 7)
        e1, p1 = masks_for(grown, 7)
        rows.append((gf.stem, sl, img, grid_occ, grown, e0, p0, e1, p1, mode))

    fig, axes = plt.subplots(len(rows), 4, figsize=(14.5, 3.9 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (nm, sl, img, occ, grown, e0, p0, e1, p1, mode) in enumerate(rows):
        axes[r, 0].imshow(img, cmap='gray')
        axes[r, 0].set_ylabel('%s\nslice %d' % (nm, sl), fontsize=8)
        up = np.kron((grown > 0).astype(float) + (occ > 0).astype(float),
                     np.ones((PATCH, PATCH)))
        axes[r, 1].imshow(img, cmap='gray')
        axes[r, 1].imshow(up, cmap='YlGn', alpha=0.45, vmin=0, vmax=2)
        axes[r, 2].imshow(draw(img, e0, p0))
        axes[r, 3].imshow(draw(img, e1, p1))
        n0 = len(e0); n1 = len(e1)
        h0 = len(set(np.concatenate(p0).tolist()))
        h1 = len(set(np.concatenate(p1).tolist()))
        axes[r, 2].set_xlabel('context %d tokens | hidden %d' % (n0, h0), fontsize=8)
        axes[r, 3].set_xlabel('context %d tokens | hidden %d   (rim +%d, %s)'
                              % (n1, h1, a.radius, mode), fontsize=8)
        if r == 0:
            axes[r, 0].set_title('B-scan', fontsize=10)
            axes[r, 1].set_title('MIRAGE anatomy (dark) + grown rim (light)', fontsize=10)
            axes[r, 2].set_title('BEFORE: context + 4 targets', fontsize=10)
            axes[r, 3].set_title('AFTER: rim grown first, then partitioned', fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(c) / 255, label='target %d' % (i + 1))
          for i, c in enumerate(TARGET_RGB)]
    hl.append(Patch(color=np.array(CONTEXT_RGB) / 255, label='context (encoder sees)'))
    fig.legend(handles=hl, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('I-JEPA masking loop on real B-scans: context and the 4 prediction '
                 'targets, before and after growing the anatomy rim', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUT / 'jepa_loop_before_after.png', dpi=140)
    print('wrote', OUT / 'jepa_loop_before_after.png')
    for nm, sl, _, _, _, e0, p0, e1, p1, mode in rows:
        print('  %-12s slice %2d  context %3d -> %3d   hidden %2d -> %2d   (%s)'
              % (nm, sl, len(e0), len(e1),
                 len(set(np.concatenate(p0).tolist())),
                 len(set(np.concatenate(p1).tolist())), mode))


if __name__ == '__main__':
    main()
