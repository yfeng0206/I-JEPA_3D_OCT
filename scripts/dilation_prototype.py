#!/usr/bin/env python
"""Prototype: dilate anatomy targets to include the tissue/background boundary.

Motivation. Targets currently sit almost entirely INSIDE the retina (93% of
predicted cells are tissue at full ramp). The retina/background boundary --
arguably the most information-dense part of a B-scan, and the part the
envelope baseline spends most of its budget on -- is barely predicted at all.

This grows each target outward so a controlled slice of the boundary ring
enters the target, and adds run-to-run variation so 75 epochs do not see the
identical mask for a given slice.

Design constraints taken from the existing pipeline:

* targets must stay EXACTLY `pred_target_k` cells (the predictor has no
  padding mask), so dilation enlarges the CANDIDATE POOL and selection then
  draws k from it -- it never changes the target size
* targets must stay CONNECTED (`shrink_to_k` preserves connectivity)
* dilated cells may overlap another target; a cell belonging to two classes is
  acceptable and is not deduplicated
* dilation may never leave the grid

Randomness is deliberately small and is applied per (slice, epoch):
  ring_frac   how much of the target is drawn from the dilated ring
  direction   which side the growth favours (up/down/left/right/all)
  radius      how far the ring extends

This script only VISUALISES. Nothing here is wired into training.
"""
from __future__ import annotations

import argparse
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

from src.masks.anatomy import build_targets, shrink_to_k           # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID = 16

# 4-connected neighbour offsets, keyed by the direction they extend toward.
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def dilate(mask, radius=1, directions=None, bounds=None):
    """Grow `mask` by `radius` cells, optionally only toward `directions`.

    `bounds` is an optional boolean grid of admissible cells -- pass the valid
    (uncropped) region so growth cannot escape the image or the crop.
    """
    m = np.asarray(mask, bool)
    dirs = [DIRS[d] for d in (directions or DIRS)]
    out = m.copy()
    for _ in range(int(radius)):
        grown = out.copy()
        for dr, dc in dirs:
            shifted = np.zeros_like(out)
            h, w = out.shape
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            shifted[r0:r1, c0:c1] = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            grown |= shifted
        out = grown
    if bounds is not None:
        out &= np.asarray(bounds, bool)
    return out


def dilated_target(part, k, occ, rng, radius=1, ring_frac=0.25, directions=None,
                   bounds=None):
    """Return exactly k connected cells, `ring_frac` of them from the ring.

    The core is the original anatomy target; the ring is what dilation adds.
    Cells are taken from the core first so the target stays anchored on tissue
    and connected, then ring cells are appended -- each chosen adjacent to what
    is already selected, which is what keeps the result a single component.
    """
    core = np.asarray(part, bool)
    if core.sum() == 0:
        return core
    ring = dilate(core, radius, directions, bounds) & ~core
    n_ring = int(round(k * float(ring_frac)))
    n_ring = min(n_ring, int(ring.sum()))
    n_core = k - n_ring

    sel = shrink_to_k(core, min(n_core, int(core.sum())), score=occ)
    if n_ring <= 0:
        return sel

    cand = list(zip(*np.nonzero(ring)))
    rng.shuffle(cand)
    taken = 0
    for r, c in cand:
        if taken >= n_ring:
            break
        touches = False
        for dr, dc in DIRS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < sel.shape[0] and 0 <= cc < sel.shape[1] and sel[rr, cc]:
                touches = True
                break
        if touches:
            sel[r, c] = True
            taken += 1
    # Any shortfall (ring exhausted) is returned to the core so the target
    # still holds exactly k cells.
    if int(sel.sum()) < k:
        sel = shrink_to_k(core | sel, k, score=occ)
    return sel


def sample_policy(rng, radius_choices=(0, 1, 1, 2), ring_choices=(0.0, 0.15, 0.25, 0.35)):
    """Draw the per-slice randomness. Small on purpose."""
    radius = int(rng.choice(radius_choices))
    ring = float(rng.choice(ring_choices))
    mode = rng.choice(['all', 'all', 'vertical', 'horizontal'])
    if mode == 'vertical':
        dirs = ['up', 'down']
    elif mode == 'horizontal':
        dirs = ['left', 'right']
    else:
        dirs = None
    return radius, ring, dirs, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    files = sorted(pathlib.Path(a.guides).glob('*.npy'))[:200]
    picks = [files[i] for i in rng.choice(len(files), a.n, replace=False)]

    rows = []
    for f in picks:
        vol = np.load(f, mmap_mode='r')
        sl = int(rng.integers(20, 80))
        soft = np.asarray(vol[sl], np.float32) / 255.0      # (2, 200, 200)
        t = torch.from_numpy(soft)[None]
        g = F.adaptive_avg_pool2d(t, (GRID, GRID))[0].numpy()
        if g.sum() < 1.0:
            continue
        occ = np.clip(g[0] + g[1], 0, 1)
        parts, _ = build_targets([g[0], g[1]], n=4, mass_cap=0.90, tau=0.10)
        base = [shrink_to_k(p, a.k, score=occ) if p.sum() else p for p in parts]
        radius, ring_frac, dirs, mode = sample_policy(rng)
        bounds = np.ones_like(occ, bool)          # full grid; crop handled upstream
        dil = [dilated_target(p, a.k, occ, rng, radius, ring_frac, dirs, bounds)
               if p.sum() else p for p in parts]
        rows.append((f.stem, sl, occ, base, dil, radius, ring_frac, mode))

    if not rows:
        raise SystemExit('no usable slices sampled')

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (name, sl, occ, base, dil, radius, ring_frac, mode) in enumerate(rows):
        ub = np.zeros_like(occ, bool)
        ud = np.zeros_like(occ, bool)
        for p in base:
            ub |= p
        for p in dil:
            ud |= p

        axes[r, 0].imshow(occ, cmap='bone', vmin=0, vmax=1, interpolation='nearest')
        axes[r, 0].set_ylabel('%s\nslice %d' % (name, sl), fontsize=7)
        if r == 0:
            axes[r, 0].set_title('MIRAGE anatomy occupancy', fontsize=9)

        im = np.zeros(occ.shape + (3,), np.uint8)
        im[occ >= 0.25] = (55, 80, 55)
        im[ub] = (70, 200, 120)
        axes[r, 1].imshow(im, interpolation='nearest')
        if r == 0:
            axes[r, 1].set_title('BEFORE: targets (%d cells)' % ub.sum(), fontsize=9)
        else:
            axes[r, 1].set_xlabel('%d cells' % ub.sum(), fontsize=7)

        im2 = np.zeros(occ.shape + (3,), np.uint8)
        im2[occ >= 0.25] = (55, 80, 55)
        im2[ud & ub] = (70, 200, 120)
        im2[ud & ~ub] = (250, 220, 60)
        axes[r, 2].imshow(im2, interpolation='nearest')
        ttl = ('AFTER: r=%d ring=%.0f%% %s' % (radius, 100 * ring_frac, mode))
        if r == 0:
            axes[r, 2].set_title(ttl, fontsize=9)
        else:
            axes[r, 2].set_xlabel(ttl, fontsize=7)

        on_b = float(occ[ub].mean()) if ub.any() else 0.0
        on_d = float(occ[ud].mean()) if ud.any() else 0.0
        axes[r, 3].bar(['before', 'after'], [100 * on_b, 100 * on_d],
                       color=['#46c878', '#fadc3c'])
        axes[r, 3].set_ylim(0, 100)
        axes[r, 3].set_ylabel('% on anatomy', fontsize=7)
        axes[r, 3].tick_params(labelsize=7)
        if r == 0:
            axes[r, 3].set_title('tissue purity of the target', fontsize=9)

    for ax in axes[:, :3].ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=(70 / 255, 200 / 255, 120 / 255), label='original target cells'),
          Patch(color=(250 / 255, 220 / 255, 60 / 255), label='added by dilation (boundary)'),
          Patch(color=(55 / 255, 80 / 255, 55 / 255), label='anatomy region')]
    fig.legend(handles=hl, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Anatomy targets before and after boundary dilation '
                 '(target size stays exactly %d cells)' % a.k, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUT / 'dilation_before_after.png', dpi=150)
    print('wrote', OUT / 'dilation_before_after.png')


if __name__ == '__main__':
    main()
