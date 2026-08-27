#!/usr/bin/env python
"""4-connected growth PLUS randomised rim dilation, on real B-scans.

Two changes shown together:

  1. growth uses a 4-neighbourhood, so targets must share an EDGE. The current
     sampler uses 8, which lets a checkerboard pass as "connected": measured
     over 150 slices, only 35% of production targets are edge-connected, with
     0.22 holes each. Under 4-connectivity that becomes 89% and 0.02.

  2. the anatomy region's rim is grown before partitioning, so targets reach
     the tissue/background boundary instead of sitting purely inside tissue.
     Rim growth is redrawn per (slice, epoch) so 75 epochs do not replay one
     fixed mask.

Two independent draws are rendered per slice so the randomness is visible.
Everything runs through the production sampler; nothing in src/ is modified --
the neighbourhood is swapped in-process.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.masks.anatomy as AN                                     # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID, PATCH, RES = 16, 16, 256
TARGET_RGB = [(239, 71, 111), (255, 209, 102), (6, 214, 160), (17, 138, 178)]
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
NB4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def grow_rim(occ, radius, directions=None, rim_score=0.30):
    """Grow the anatomy region outward by `radius`, never leaving the grid."""
    if radius <= 0:
        return occ.copy()
    out = occ > 0
    dirs = [DIRS[d] for d in (directions or DIRS)]
    h, w = occ.shape
    for _ in range(radius):
        g = out.copy()
        for dr, dc in dirs:
            s = np.zeros_like(out)
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            s[r0:r1, c0:c1] = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            g |= s
        out = g
    grown = occ.copy()
    grown[out & (occ == 0)] = rim_score
    return grown


def draw_policy(rng):
    """Small per-slice randomness: how far the rim grows and in which sense."""
    radius = int(rng.choice([0, 1, 1, 1]))          # sometimes no growth at all
    mode = str(rng.choice(['all', 'all', 'up', 'down']))
    dirs = None if mode == 'all' else (
        ['up', 'left', 'right'] if mode == 'up' else ['down', 'left', 'right'])
    return radius, mode, dirs


def targets(occ, k=16):
    AN.NB8[:] = NB4
    parts, _ = AN.build_targets([occ, occ * 0], n=4, mass_cap=0.90, tau=0.10)
    return [AN.shrink_to_k(p, k, score=occ) if p.sum() else p for p in parts]


def stats(parts):
    c4, hol, cells = [], 0, 0
    for m in parts:
        if m.sum() == 0:
            continue
        c4.append(ndimage.label(m, structure=CROSS)[1] == 1)
        hol += int((ndimage.binary_fill_holes(m) & ~m).sum())
        cells += int(m.sum())
    return 100 * float(np.mean(c4)), hol, cells


def overlay(img, parts):
    base = (np.stack([img] * 3, -1) * 255).astype(np.uint8)
    canvas = (base * 0.5).astype(np.uint8)
    for t, m in enumerate(parts):
        col = np.array(TARGET_RGB[t % 4], np.uint8)
        for r, c in zip(*np.nonzero(m)):
            y, x = r * PATCH, c * PATCH
            blk = base[y:y + PATCH, x:x + PATCH].astype(np.float32)
            canvas[y:y + PATCH, x:x + PATCH] = (0.4 * blk + 0.6 * col).astype(np.uint8)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--data', default=r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    files = sorted(pathlib.Path(a.guides).glob('*.npy'))
    picks = [files[i] for i in rng.choice(len(files), a.n, replace=False)]

    import cv2
    rows = []
    for gf in picks:
        vol = np.load(gf, mmap_mode='r')
        sl = int(rng.integers(25, 75))
        s = np.asarray(vol[sl], np.float32) / 255.
        occ = F.adaptive_avg_pool2d(
            torch.from_numpy(np.clip(s[0] + s[1], 0, 1))[None, None],
            (GRID, GRID))[0, 0].numpy()
        if occ.sum() < 1:
            continue
        with np.load(pathlib.Path(a.data) / (gf.stem + '.npz'), allow_pickle=True) as z:
            raw = np.asarray(z['oct_bscans'][sl * 2], np.float32)
        lo, hi = raw.min(), raw.max()
        raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        rows.append((gf.stem, sl,
                     cv2.resize(raw, (RES, RES), interpolation=cv2.INTER_LINEAR),
                     occ))

    fig, axes = plt.subplots(len(rows), 4, figsize=(14.5, 3.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (nm, sl, img, occ) in enumerate(rows):
        base = targets(occ)
        b4, bh, bc = stats(base)
        axes[r, 0].imshow(img, cmap='gray')
        axes[r, 0].set_ylabel('%s\nslice %d' % (nm, sl), fontsize=8)
        axes[r, 1].imshow(overlay(img, base))
        axes[r, 1].set_xlabel('%.0f%% edge-conn, %d holes, %d cells' % (b4, bh, bc),
                              fontsize=8)
        for j, sd in enumerate((101, 202)):
            rr = np.random.default_rng(sd + r)
            radius, mode, dirs = draw_policy(rr)
            p = targets(grow_rim(occ, radius, dirs))
            s4, h, c = stats(p)
            axes[r, 2 + j].imshow(overlay(img, p))
            axes[r, 2 + j].set_xlabel(
                'rim +%d (%s) | %.0f%% edge-conn, %d holes, %d cells'
                % (radius, mode, s4, h, c), fontsize=8)
        if r == 0:
            axes[r, 0].set_title('B-scan', fontsize=10)
            axes[r, 1].set_title('4-connected, no rim', fontsize=10)
            axes[r, 2].set_title('4-connected + random rim\n(draw A = epoch i)',
                                 fontsize=10)
            axes[r, 3].set_title('4-connected + random rim\n(draw B = epoch j)',
                                 fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(c) / 255, label='target %d' % (i + 1))
          for i, c in enumerate(TARGET_RGB)]
    fig.legend(handles=hl, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Edge-connected targets with randomised rim growth: two draws '
                 'of the same slice differ, so 75 epochs do not repeat',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(OUT / 'connected_plus_random_rim.png', dpi=145)
    print('wrote', OUT / 'connected_plus_random_rim.png')


if __name__ == '__main__':
    main()
