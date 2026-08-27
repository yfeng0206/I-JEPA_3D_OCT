#!/usr/bin/env python3
"""Before/after for the single-component growth bug, on real split-support slices.

BEFORE  grow_region      seeds once at the global max -> fills ONE component
AFTER   grow_components  grows every component, budget split by mass share
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                    # noqa: E402
from jepa_to_mirage_probe import build_mirage            # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\adapter_sweep')
OUT = REPO / 'results/masking/split_fix'
GRID, TAU, ANATOMY = 16, 0.10, (1, 2)
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.15, 0.70, 0.30), (0.95, 0.65, 0.10)]


def old_build(class_scores, n=4, mass_cap=0.80, tau=TAU, max_hole=2,
              max_ratio=1.25):
    """The previous single-component pipeline, for the BEFORE panel."""
    shape = class_scores[0].shape
    grown, masses, caps = [], [], []
    for score in class_scores:
        if not (score > tau).any():
            grown.append(None); masses.append(0.0); caps.append(0); continue
        U = A.grow_region(score, mass_cap, tau, None)
        U = A.fill_small_holes(U, score, max_hole)
        grown.append(U); masses.append(float(score.sum())); caps.append(int(U.sum()))
    ks = A.allocate(masses, n, caps)
    parts, regions = [], []
    for U, k in zip(grown, ks):
        if k == 0 or U is None:
            continue
        regions.append(U)
        parts.extend(A.rebalance(A.geodesic_partition(U, k), max_ratio))
    while len(parts) < n:
        parts.append(np.zeros(shape, bool))
    return parts[:n], regions


def scan(n_probe, cap):
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    mir = build_mirage(dev)
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(L0=o.detach()))
    rows = []
    B = 8
    for s in range(0, n_probe, B):
        x = torch.from_numpy(im512[s:s + B].astype(np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
            g = F.adaptive_avg_pool2d(grab['L0'].float().softmax(1)[:, ANATOMY],
                                      (GRID, GRID)).cpu().numpy()
        for j in range(g.shape[0]):
            Pi, Pc = g[j, 0], g[j, 1]
            a = Pi + Pc
            stranded = 0.0
            ncomp = 0
            for sc in (Pi, Pc):
                S = sc > TAU
                if not S.any():
                    continue
                lab, k = ndimage.label(S, structure=np.ones((3, 3)))
                ncomp = max(ncomp, k)
                if k > 1:
                    big = max(range(1, k + 1), key=lambda q: (lab == q).sum())
                    stranded += float(sc[S].sum() - sc[lab == big].sum()) / max(a.sum(), 1e-9)
            if ncomp > 1 and A.is_viable([Pi, Pc], 4, min_cells=4, mass_cap=cap):
                rows.append((s + j, stranded, ncomp, Pi, Pc))
    rows.sort(key=lambda r: -r[1])
    del mir
    torch.cuda.empty_cache()
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument('--n-probe', type=int, default=240)
    ap.add_argument('--n-show', type=int, default=4)
    ap.add_argument('--cap', type=float, default=0.80)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = scan(a.n_probe, a.cap)
    print('found %d split-support slices in %d probed' % (len(rows), a.n_probe))
    rows = rows[:a.n_show]
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')

    n = len(rows)
    fig = plt.figure(figsize=(20, 4.1 * n))
    gs = fig.add_gridspec(n, 5, hspace=.28, wspace=.14,
                          left=.03, right=.99, top=.93, bottom=.02)
    summary = []
    for r, (idx, stranded, ncomp, Pi, Pc) in enumerate(rows):
        aa = Pi + Pc
        pb, rb = old_build([Pi, Pc], 4, mass_cap=a.cap)
        pa, ra = A.build_targets([Pi, Pc], 4, mass_cap=a.cap)
        Ub = np.logical_or.reduce(pb)
        Ua = np.logical_or.reduce(pa)
        mb = aa[Ub].sum() / aa.sum()
        ma = aa[Ua].sum() / aa.sum()
        summary.append(dict(idx=int(idx), components=int(ncomp),
                            mass_before=float(mb), mass_after=float(ma),
                            cells_before=int(Ub.sum()), cells_after=int(Ua.sum()),
                            sizes_before=[int(p.sum()) for p in pb],
                            sizes_after=[int(p.sum()) for p in pa]))

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(im256[idx], cmap='gray')
        ax.set_ylabel('slice %d' % idx, fontsize=10)
        ax.set_title('B-scan', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[r, 1])
        lab_img = np.zeros((GRID, GRID))
        for ci, sc in enumerate((Pi, Pc)):
            S = sc > TAU
            if not S.any():
                continue
            lb, k = ndimage.label(S, structure=np.ones((3, 3)))
            for q in range(1, k + 1):
                lab_img[lb == q] = ci * 3 + q
        ax.imshow(lab_img, cmap='tab20', interpolation='nearest')
        ax.set_title('support components  (%d)' % ncomp, fontsize=10)
        ax.set_xlabel('%.0f%% of mass stranded' % (100 * stranded), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        for c, (parts, U, m, t) in enumerate((
                (pb, Ub, mb, 'BEFORE  single-component'),
                (pa, Ua, ma, 'AFTER  multi-component'))):
            ax = fig.add_subplot(gs[r, 2 + c])
            ax.imshow(im256[idx], cmap='gray')
            ov = np.zeros((256, 256, 4))
            for k2, p in enumerate(parts):
                if p.sum():
                    ov[np.kron(p, np.ones((16, 16))).astype(bool)] = (*COLORS[k2], .55)
            ax.imshow(ov)
            ax.set_title('%s\n%d cells, %.0f%% of anatomy mass'
                         % (t, int(U.sum()), 100 * m), fontsize=10)
            ax.set_xlabel('target sizes %s' % [int(p.sum()) for p in parts], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[r, 4])
        d = np.zeros((GRID, GRID))
        d[Ua & ~Ub] = 1
        d[Ub & ~Ua] = -1
        ax.imshow(d, cmap='bwr', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title('recovered (red) / dropped (blue)', fontsize=10)
        ax.set_xlabel('+%d cells   +%.0f pp mass'
                      % (int(Ua.sum()) - int(Ub.sum()), 100 * (ma - mb)), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    (a.out / 'split_fix.json').write_text(json.dumps(summary, indent=2))
    fig.suptitle('Single-component growth bug: the sampler could only ever fill ONE '
                 'component per class   (mass_cap %.2f)' % a.cap, fontsize=14, y=.985)
    f = a.out / 'split_fix.png'
    fig.savefig(f, dpi=112, facecolor='white')
    print('wrote %s' % f)
    for s in summary:
        print('slice %-6d comps %d   mass %.3f -> %.3f   cells %2d -> %2d   %s -> %s'
              % (s['idx'], s['components'], s['mass_before'], s['mass_after'],
                 s['cells_before'], s['cells_after'],
                 s['sizes_before'], s['sizes_after']))


if __name__ == '__main__':
    main()
