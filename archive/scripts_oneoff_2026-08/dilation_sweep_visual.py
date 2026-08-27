#!/usr/bin/env python
"""Show what boundary dilation actually does, across its parameter range.

The first prototype drew random settings per slice, which made the effect
invisible on half the panels. This sweeps the parameters explicitly on fixed
slices so the mechanism is legible, and then measures the effect over many
slices so the choice can be made on numbers rather than impressions.

Finding that motivated the rewrite: current targets are NOT purely interior.
`grow_components` admits any cell with occupancy above tau=0.10, so targets
already straddle the tissue edge -- measured mean occupancy of a target is
about 0.60-0.70, not the ~0.93 that the "on-region" statistic reports (that
one thresholds at 0.25 and asks only whether a cell is on-region at all).
Dilation therefore pushes further out from an edge the targets already touch,
rather than discovering the edge for the first time.
"""
from __future__ import annotations

import argparse
import json
import pathlib
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
sys.path.insert(0, str(REPO / 'scripts'))

from src.masks.anatomy import build_targets, shrink_to_k           # noqa: E402
from dilation_prototype import dilate, dilated_target              # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID = 16


def grid_for(path, sl):
    vol = np.load(path, mmap_mode='r')
    soft = np.asarray(vol[sl], np.float32) / 255.0
    t = torch.from_numpy(soft)[None]
    return F.adaptive_avg_pool2d(t, (GRID, GRID))[0].numpy()


def targets_for(g, k, rng, radius, ring, dirs):
    occ = np.clip(g[0] + g[1], 0, 1)
    parts, _ = build_targets([g[0], g[1]], n=4, mass_cap=0.90, tau=0.10)
    out = []
    for p in parts:
        if p.sum() == 0:
            out.append(p)
        elif radius == 0 or ring == 0:
            out.append(shrink_to_k(p, k, score=occ))
        else:
            out.append(dilated_target(p, k, occ, rng, radius, ring, dirs, None))
    return out, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--stats-slices', type=int, default=150)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(a.guides).glob('*.npy'))

    # ---- figure 1: parameter sweep on two fixed slices --------------------
    settings = [(0, 0.00), (1, 0.15), (1, 0.30), (2, 0.30), (2, 0.50)]
    picks = [(files[7], 50), (files[41], 38)]
    fig, axes = plt.subplots(len(picks), len(settings) + 1,
                             figsize=(2.3 * (len(settings) + 1), 2.5 * len(picks)))
    for r, (f, sl) in enumerate(picks):
        g = grid_for(f, sl)
        occ = np.clip(g[0] + g[1], 0, 1)
        axes[r, 0].imshow(occ, cmap='bone', vmin=0, vmax=1, interpolation='nearest')
        axes[r, 0].set_ylabel('%s\nslice %d' % (f.stem, sl), fontsize=7)
        if r == 0:
            axes[r, 0].set_title('anatomy occupancy', fontsize=8)
        base = None
        for c, (radius, ring) in enumerate(settings):
            rng = np.random.default_rng(0)
            parts, _ = targets_for(g, a.k, rng, radius, ring, None)
            u = np.zeros_like(occ, bool)
            for p in parts:
                u |= p
            if base is None:
                base = u.copy()
            im = np.zeros(occ.shape + (3,), np.uint8)
            im[occ >= 0.25] = (55, 80, 55)
            im[u & base] = (70, 200, 120)
            im[u & ~base] = (250, 220, 60)
            axes[r, c + 1].imshow(im, interpolation='nearest')
            lab = 'r=%d ring=%d%%' % (radius, 100 * ring)
            pur = 100 * float(occ[u].mean()) if u.any() else 0
            if r == 0:
                axes[r, c + 1].set_title(lab, fontsize=8)
            axes[r, c + 1].set_xlabel('%d cells, %.0f%% tissue' % (u.sum(), pur),
                                      fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=(70 / 255, 200 / 255, 120 / 255), label='target cells'),
          Patch(color=(250 / 255, 220 / 255, 60 / 255), label='added vs r=0'),
          Patch(color=(55 / 255, 80 / 255, 55 / 255), label='anatomy region')]
    fig.legend(handles=hl, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Boundary dilation parameter sweep (target size fixed at %d cells)'
                 % a.k, fontsize=11)
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    plt.savefig(OUT / 'dilation_sweep.png', dpi=150)
    print('wrote', OUT / 'dilation_sweep.png')

    # ---- figure 2 + numbers: effect over many slices ----------------------
    rng0 = np.random.default_rng(0)
    idx = rng0.choice(len(files), a.stats_slices, replace=False)
    rows = {}
    for radius, ring in settings:
        pur, bg, moved = [], [], []
        rng = np.random.default_rng(1)
        for i in idx:
            g = grid_for(files[i], int(rng.integers(20, 80)))
            if g.sum() < 1.0:
                continue
            parts, occ = targets_for(g, a.k, rng, radius, ring, None)
            u = np.zeros_like(occ, bool)
            for p in parts:
                u |= p
            if not u.any():
                continue
            pur.append(float(occ[u].mean()))
            bg.append(float((occ[u] < 0.25).mean()))
            b, _ = targets_for(g, a.k, np.random.default_rng(1), 0, 0.0, None)
            ub = np.zeros_like(occ, bool)
            for p in b:
                ub |= p
            inter = (u & ub).sum(); union = max((u | ub).sum(), 1)
            moved.append(inter / union)
        rows[(radius, ring)] = (float(np.mean(pur)), float(np.mean(bg)),
                                float(np.mean(moved)))

    print()
    print('%-16s %12s %14s %14s' %
          ('setting', 'tissue mass', 'background %', 'Jaccard vs r=0'))
    print('-' * 60)
    for (radius, ring), (p, b, j) in rows.items():
        print('r=%d ring=%3d%%     %11.3f %13.1f%% %14.3f'
              % (radius, 100 * ring, p, 100 * b, j))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    labs = ['r=%d\n%d%%' % (r, 100 * g) for (r, g) in rows]
    ax[0].bar(labs, [100 * rows[k][1] for k in rows], color='#fadc3c')
    ax[0].set_ylabel('% of target cells that are background')
    ax[0].set_title('How much background enters the target')
    ax[0].grid(alpha=.3, axis='y')
    ax[1].bar(labs, [rows[k][2] for k in rows], color='#46c878')
    ax[1].set_ylabel('Jaccard vs undilated')
    ax[1].set_ylim(0, 1)
    ax[1].set_title('How much the target actually moves')
    ax[1].grid(alpha=.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUT / 'dilation_stats.png', dpi=150)
    print('wrote', OUT / 'dilation_stats.png')
    (OUT / 'dilation_stats.json').write_text(json.dumps(
        {'r%d_ring%d' % (r, 100 * g): {'tissue_mass': v[0], 'background_frac': v[1],
                                       'jaccard_vs_undilated': v[2]}
         for (r, g), v in rows.items()}, indent=2))


if __name__ == '__main__':
    main()
