#!/usr/bin/env python
"""NB8 k=16 vs NB8 k=16 + diagonal bridging, on real B-scans.

Column 2 is exactly what production emits today. Every place two target cells
meet at a corner only -- no shared edge -- is marked with a white x. Those are
the points where `n_components` reports "1 connected shape" (it checks with
np.ones((3,3)), which counts diagonals) while a 4-connected reading sees the
target fall apart into separate pieces.

Column 3 adds the bridge cells (white outline). For each diagonal step
(r,c)-(r+1,c+1) either (r+1,c) or (r,c+1) closes it; the one with higher
anatomy occupancy wins, so bridging spends its cells on tissue.

Targets come from the production CurriculumMaskGenerator, not a reimplementation
-- bridging is applied to its output, which is how it would be integrated.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bridge_diagonals_sweep import bridge_diagonals, CROSS  # noqa: E402
from scripts.bridge_exact_k import trim_to_k_nb4                    # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator            # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID, PATCH, RES = 16, 16, 256
TARGET_RGB = [(239, 71, 111), (255, 209, 102), (6, 214, 160), (17, 138, 178)]


def outline(ax, m, color, lw=1.6):
    """Trace the boundary of every cell run in `m`."""
    for r in range(GRID):
        for c in range(GRID):
            if not m[r, c]:
                continue
            y, x = r * PATCH, c * PATCH
            if r == 0 or not m[r - 1, c]:
                ax.plot([x, x + PATCH], [y, y], color=color, lw=lw)
            if r == GRID - 1 or not m[r + 1, c]:
                ax.plot([x, x + PATCH], [y + PATCH, y + PATCH], color=color, lw=lw)
            if c == 0 or not m[r, c - 1]:
                ax.plot([x, x], [y, y + PATCH], color=color, lw=lw)
            if c == GRID - 1 or not m[r, c + 1]:
                ax.plot([x + PATCH, x + PATCH], [y, y + PATCH], color=color, lw=lw)


def diagonal_junctions(m):
    """Corner points where two cells touch diagonally with no edge path."""
    pts = []
    for r in range(GRID - 1):
        for c in range(GRID - 1):
            a, b = m[r, c], m[r + 1, c + 1]
            x_, y_ = m[r + 1, c], m[r, c + 1]
            if a and b and not x_ and not y_:
                pts.append(((c + 1) * PATCH, (r + 1) * PATCH))
            if x_ and y_ and not a and not b:
                pts.append(((c + 1) * PATCH, (r + 1) * PATCH))
    return pts


def paint(base, masks, extra=None):
    img = (base * 0.35).astype(np.uint8)
    for ti, m in enumerate(masks):
        col = np.array(TARGET_RGB[ti % 4], np.uint8)
        for r, c in zip(*np.nonzero(m)):
            y, x = r * PATCH, c * PATCH
            blk = base[y:y + PATCH, x:x + PATCH].astype(np.float32)
            img[y:y + PATCH, x:x + PATCH] = (0.4 * blk + 0.6 * col).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--data', default=r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    files = sorted(pathlib.Path(a.guides).glob('*.npy'))
    picks = [files[i] for i in rng.choice(len(files), a.n * 3, replace=False)]

    cfg = {'mode': 'mirage_anatomy', 'enabled': True, 'T_warm': 25, 'T_total': 30,
           'r_max': 1.0, 'ramp_shape': 'linear', 'anatomy_mass_cap': 0.90,
           'anatomy_tau': 0.10, 'mirage_occupancy_threshold': 0.25}

    rows = []
    for k, gf in enumerate(picks):
        if len(rows) == a.n:
            break
        vol = np.load(gf, mmap_mode='r')
        sl = int(rng.integers(25, 75))
        soft = np.asarray(vol[sl], np.float32) / 255.0
        with np.load(pathlib.Path(a.data) / (gf.stem + '.npz'), allow_pickle=True) as z:
            raw = np.asarray(z['oct_bscans'][sl * 2], np.float32)
        lo, hi = raw.min(), raw.max()
        raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0

        img_pil = Image.fromarray((raw * 255).astype(np.uint8))
        occ_full = np.clip(soft[0] + soft[1], 0, 1)
        gp = Image.fromarray((occ_full * 255).astype(np.uint8)).resize(
            img_pil.size, Image.NEAREST)

        torch.manual_seed(1000 + k)
        top, left, h, w = T.RandomResizedCrop.get_params(
            img_pil, [0.3, 1.0], [3.0 / 4.0, 4.0 / 3.0])
        crop_img = np.asarray(TF.resized_crop(
            img_pil, top, left, h, w, [RES, RES],
            T.InterpolationMode.BICUBIC), np.float32) / 255.
        crop_g = np.asarray(TF.resized_crop(
            gp, top, left, h, w, [RES, RES],
            T.InterpolationMode.NEAREST), np.float32) / 255.
        occ = F.adaptive_avg_pool2d(
            torch.from_numpy(crop_g)[None, None], (GRID, GRID))[0, 0].numpy()
        if occ.sum() < 0.5:
            continue

        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        gen = CurriculumMaskGenerator(
            input_size=(RES, RES), patch_size=PATCH, npred=4, nenc=1,
            pred_mask_scale=(0.15, 0.2), pred_target_k=16, curriculum_cfg=cfg)
        gen.set_epoch(50, 100)
        t = torch.from_numpy(occ)[None].float()
        enc, pred = gen.generate(
            batch_size=1, guide_grids=torch.stack([t, (t >= 0.25).float()], 1),
            guide_valid=torch.ones(1, dtype=torch.bool))

        before, after, nadd, ntrim = [], [], 0, 0
        for p in pred:
            m = np.zeros(GRID * GRID, bool)
            m[p[0].numpy()] = True
            m = m.reshape(GRID, GRID)
            before.append(m)
            k = int(m.sum())
            b, n = bridge_diagonals(m, occ)
            grown = int(b.sum())
            if grown > k:                      # exact-K: give the cells back
                b = trim_to_k_nb4(b, k, occ)
            after.append(b); nadd += n; ntrim += grown - int(b.sum())
        rows.append((gf.stem, sl, crop_img, occ, before, after, (nadd, ntrim),
                     enc[0][0].numpy()))

    fig, axes = plt.subplots(len(rows), 4, figsize=(15.5, 3.9 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    tot = dict(b_ok=0, a_ok=0, n=0, b_cells=0, a_cells=0, b_anat=0.0, a_anat=0.0)
    for r, (nm, sl, cimg, occ, before, after, (nadd, ntrim), eidx) in enumerate(rows):
        base = (np.stack([cimg] * 3, -1) * 255).astype(np.uint8)

        axes[r, 0].imshow(cimg, cmap='gray')
        axes[r, 0].imshow(np.kron(occ, np.ones((PATCH, PATCH))), cmap='YlGn',
                          alpha=0.45, vmin=0, vmax=1)
        axes[r, 0].set_ylabel('%s\nslice %d' % (nm, sl), fontsize=8)
        axes[r, 0].set_xlabel('anatomy occupancy, 16x16', fontsize=8)

        for col, masks, tag in ((1, before, 'before'), (2, after, 'after')):
            ax = axes[r, col]
            ax.imshow(paint(base, masks))
            nc = [ndimage.label(m, structure=CROSS)[1] for m in masks]
            for ti, m in enumerate(masks):
                outline(ax, m, np.array(TARGET_RGB[ti % 4]) / 255, 1.4)
            if tag == 'before':
                for m in masks:
                    for x, y in diagonal_junctions(m):
                        ax.plot(x, y, marker='x', color='white', ms=9, mew=2.4)
                ok = sum(c == 1 for c in nc)
                cells = sum(int(m.sum()) for m in masks)
                anat = sum(float(occ[m].sum()) for m in masks)
                ax.set_xlabel('4-conn %d/4  pieces %s  cells %d  anat %.1f'
                              % (ok, '+'.join(map(str, nc)), cells, anat),
                              fontsize=8, color='#c0392b' if ok < 4 else 'k')
                tot['b_ok'] += ok; tot['n'] += 4
                tot['b_cells'] += cells; tot['b_anat'] += anat
            else:
                for mb, ma in zip(before, after):
                    for r_, c_ in zip(*np.nonzero(ma & ~mb)):
                        ax.plot([c_ * PATCH, (c_ + 1) * PATCH, (c_ + 1) * PATCH,
                                 c_ * PATCH, c_ * PATCH],
                                [r_ * PATCH, r_ * PATCH, (r_ + 1) * PATCH,
                                 (r_ + 1) * PATCH, r_ * PATCH],
                                color='white', lw=2.2)
                    for r_, c_ in zip(*np.nonzero(mb & ~ma)):
                        ax.plot([c_ * PATCH, (c_ + 1) * PATCH],
                                [r_ * PATCH, (r_ + 1) * PATCH],
                                color='#888888', lw=1.8, ls=':')
                        ax.plot([c_ * PATCH, (c_ + 1) * PATCH],
                                [(r_ + 1) * PATCH, r_ * PATCH],
                                color='#888888', lw=1.8, ls=':')
                ok = sum(c == 1 for c in nc)
                cells = sum(int(m.sum()) for m in masks)
                anat = sum(float(occ[m].sum()) for m in masks)
                ax.set_xlabel('4-conn %d/4  +%d bridged  -%d trimmed  cells %d '
                              '(same)  anat %.1f'
                              % (ok, nadd, ntrim, cells, anat), fontsize=8,
                              color='#1e8449' if ok == 4 else '#c0392b')
                tot['a_ok'] += ok
                tot['a_cells'] += cells; tot['a_anat'] += anat

        ctx = np.zeros_like(base)
        union = np.zeros((GRID, GRID), bool)
        for m in after:
            union |= m
        keep = [i for i in eidx if not union.ravel()[int(i)]]
        for i in keep:
            rr, cc = int(i) // GRID, int(i) % GRID
            y, x = rr * PATCH, cc * PATCH
            ctx[y:y + PATCH, x:x + PATCH] = base[y:y + PATCH, x:x + PATCH]
        axes[r, 3].imshow(ctx)
        axes[r, 3].set_xlabel('context encoder input: %d of 256 tokens' % len(keep),
                              fontsize=8)

        if r == 0:
            for c, t_ in enumerate([
                    '1. anatomy guide (after crop)',
                    '2. BEFORE - NB8 k=16 (production today)',
                    '3. AFTER - bridge, then trim back to same K',
                    '4. context the encoder sees']):
                axes[r, c].set_title(t_, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(c) / 255, label='target %d' % (i + 1))
          for i, c in enumerate(TARGET_RGB)]
    hl += [plt.Line2D([], [], marker='x', color='white', mew=2.4, ls='',
                      markersize=9, label='diagonal-only join (breaks 4-conn)'),
           Patch(facecolor='none', edgecolor='white', lw=2.2, label='bridge cell added'),
           plt.Line2D([], [], color='#888888', lw=1.8, ls=':',
                      label='cell given back (keeps budget exact)')]
    fig.legend(handles=hl, loc='lower center', ncol=6, fontsize=8.5,
               facecolor='#dddddd')
    fig.suptitle('Diagonal bridging: same anatomy-hugging growth, but every '
                 'target becomes one solid edge-connected region', fontsize=12)
    plt.tight_layout(rect=[0, 0.055, 1, 0.95])
    plt.savefig(OUT / 'bridge_before_after.png', dpi=140)
    print('wrote', OUT / 'bridge_before_after.png')
    print('  4-connected targets  before %d/%d   after %d/%d'
          % (tot['b_ok'], tot['n'], tot['a_ok'], tot['n']))
    print('  hidden cells         before %d      after %d  (+%d)'
          % (tot['b_cells'], tot['a_cells'], tot['a_cells'] - tot['b_cells']))
    print('  anatomy hidden       before %.1f    after %.1f  (%+.1f)'
          % (tot['b_anat'], tot['a_anat'], tot['a_anat'] - tot['b_anat']))


if __name__ == '__main__':
    main()
