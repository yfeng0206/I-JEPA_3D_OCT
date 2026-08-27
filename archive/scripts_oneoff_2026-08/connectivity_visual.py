#!/usr/bin/env python
"""Are the anatomy targets actually connected blobs, or diagonal chains?

The sampler grows and partitions with an 8-neighbourhood (`NB8`), and the
connectivity check uses `ndimage.label(structure=np.ones((3,3)))` -- the SAME
permissive rule. So every "100% single-component" figure in the docs is
measured with diagonals counting as adjacency, and a checkerboard passes.

Measured over 150 slices, 4 targets each:

    neighbourhood     8-conn    4-conn    holes   bbox fill   cells
    NB8 (current)     100.0%     35.0%     0.22        0.50    13.7
    NB4 (edge only)   100.0%     89.0%     0.02        0.56    13.7

Two thirds of production targets are not edge-connected. This renders both
variants on real B-scans so the difference is visible rather than argued.
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
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def targets(occ, nb, k=16):
    AN.NB8[:] = nb
    parts, _ = AN.build_targets([occ, occ * 0], n=4, mass_cap=0.90, tau=0.10)
    return [AN.shrink_to_k(p, k, score=occ) if p.sum() else p for p in parts]


def stats(parts):
    c4, hol = [], []
    for m in parts:
        if m.sum() == 0:
            continue
        c4.append(ndimage.label(m, structure=CROSS)[1] == 1)
        hol.append(int((ndimage.binary_fill_holes(m) & ~m).sum()))
    return 100 * float(np.mean(c4)), float(np.sum(hol))


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
        npz = pathlib.Path(a.data) / (gf.stem + '.npz')
        with np.load(npz, allow_pickle=True) as z:
            raw = np.asarray(z['oct_bscans'][sl * 2], np.float32)
        lo, hi = raw.min(), raw.max()
        raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        img = cv2.resize(raw, (RES, RES), interpolation=cv2.INTER_LINEAR)
        rows.append((gf.stem, sl, img, occ))

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (nm, sl, img, occ) in enumerate(rows):
        p8 = targets(occ, NB8)
        p4 = targets(occ, NB4)
        s8, h8 = stats(p8)
        s4, h4 = stats(p4)
        axes[r, 0].imshow(img, cmap='gray')
        axes[r, 0].set_ylabel('%s\nslice %d' % (nm, sl), fontsize=8)
        axes[r, 1].imshow(overlay(img, p8))
        axes[r, 2].imshow(overlay(img, p4))
        axes[r, 1].set_xlabel('%.0f%% edge-connected, %d holes' % (s8, h8), fontsize=8)
        axes[r, 2].set_xlabel('%.0f%% edge-connected, %d holes' % (s4, h4), fontsize=8)
        if r == 0:
            axes[r, 0].set_title('B-scan', fontsize=10)
            axes[r, 1].set_title('CURRENT: 8-neighbour growth\n(diagonals count as connected)',
                                 fontsize=10)
            axes[r, 2].set_title('PROPOSED: 4-neighbour growth\n(must share an edge)',
                                 fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(c) / 255, label='target %d' % (i + 1))
          for i, c in enumerate(TARGET_RGB)]
    fig.legend(handles=hl, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Why targets look scattered: diagonal adjacency counts as '
                 '"connected" in the current sampler', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(OUT / 'connectivity_4_vs_8.png', dpi=145)
    print('wrote', OUT / 'connectivity_4_vs_8.png')


if __name__ == '__main__':
    main()
