"""Side-by-side: independent-regions sampler vs union-first-then-split.

Same slices, same MIRAGE anatomy, same anatomy budget -- only the target
CONSTRUCTOR differs, so any visual difference is attributable to it.

    OLD  grow4       four regions grown independently, one seeded per band.
                     The union is whatever they happen to cover, so it
                     fragments into 2-5 islands, and before a diversity
                     penalty was added the four regions collapsed onto the
                     same ridge (72.5% pairwise overlap vs I-JEPA's 23.9%).

    NEW  union-first  grow ONE connected union to the anatomy budget, order it
                      geodesically, then partition it into four connected,
                      near-equal targets and dilate them to I-JEPA-like overlap.

Reads the demo npz written by scripts/demo_anatomy_masking.py --dump.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

GRID, PATCH = 16, 16
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.20, 0.75, 0.35), (0.98, 0.65, 0.10)]


def paint(parts, shape=(GRID, GRID)):
    canvas = np.ones(shape + (3,)) * 0.12
    for k, p in enumerate(parts):
        canvas[p] = 0.5 * canvas[p] + 0.5 * np.array(COLORS[k])
    return canvas


def stats_of(parts, a):
    from anatomy_target_sampler import topology_of
    union = np.zeros_like(parts[0])
    for p in parts:
        union |= p
    sizes = [int(p.sum()) for p in parts]
    A = float(a.sum())
    ov = sum(int((parts[i] & parts[j]).sum())
             for i in range(4) for j in range(i + 1, 4))
    t = topology_of(union)
    return {
        'sizes': sizes, 'spread': max(sizes) - min(sizes),
        'union_cells': int(union.sum()), 'union_components': t['components'],
        'union_holes': t['holes'],
        'targets_connected': all(topology_of(p)['components'] == 1
                                 for p in parts if p.sum()),
        'visible': float(1 - (a * union).sum() / A),
        'overlap_pct': 100.0 * ov / 6 / max(np.mean(sizes), 1),
        'union': union,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\maskdemo\demo.npz'))
    ap.add_argument('--rho', type=float, default=0.70)
    ap.add_argument('--out', default='results/masking/demo/sampler_comparison.png')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from anatomy_budget_sweep import sel_grow4
    from anatomy_target_sampler import build_targets

    z = np.load(a.npz, allow_pickle=True)
    imgs, grids, names = z['imgs'], z['grids'], [str(s) for s in z['names']]
    n = len(imgs)

    # Three constructors, in the order they were developed.  v1 is the ORIGINAL
    # grow4 with no diversity penalty (claim_penalty=1.0 disables it) -- the
    # version whose four regions collapsed onto the same ridge.  v2 is that same
    # sampler after the penalty fix.  v3 is union-first-then-split.
    variants = [
        ('v1 ORIGINAL\n4 independent, no penalty',
         lambda g, A: sel_grow4(g, a.rho * A, claim_penalty=1.0)),
        ('v2 + diversity penalty\n4 independent, penalty 0.5',
         lambda g, A: sel_grow4(g, a.rho * A, claim_penalty=0.5)),
        ('v3 NEW union-first\n1 union split into 4',
         lambda g, A: build_targets(g, rho=a.rho, overlap=0.24)[0]),
    ]

    fig, ax = plt.subplots(n, 5, figsize=(20, 4.0 * n))
    if n == 1:
        ax = ax[None]
    rows = []

    for r in range(n):
        img, g = imgs[r], grids[r]
        A = float(g.sum())
        res = [(lbl, fn(g, A)) for lbl, fn in variants]
        sts = [stats_of(p, g) for _, p in res]
        rows.append({'slice': names[r], 'anatomy_mass': A,
                     **{('v%d' % (i + 1)): {k: v for k, v in s.items() if k != 'union'}
                        for i, s in enumerate(sts)}})

        ax[r, 0].imshow(img, cmap='gray')
        ax[r, 0].set_ylabel(names[r], fontsize=9)
        if r == 0:
            ax[r, 0].set_title('OCT crop', fontsize=11)
        ax[r, 1].imshow(g, cmap='viridis', vmin=0, vmax=1)
        ax[r, 1].set_title('anatomy %.1f cells' % A, fontsize=10)

        for i, ((lbl, parts), s) in enumerate(zip(res, sts)):
            axis = ax[r, 2 + i]
            axis.imshow(paint(parts))
            bad = s['union_components'] > 1
            axis.set_title('%s\n%s  spread %d\nunion %d, %d comp%s  ov %.0f%%'
                           % (lbl if r == 0 else '',
                              '/'.join(map(str, s['sizes'])), s['spread'],
                              s['union_cells'], s['union_components'],
                              ' !!' if bad else '', s['overlap_pct']),
                           fontsize=9, color='darkred' if bad else 'darkgreen')

        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    handles = [Patch(facecolor=COLORS[k], label='target %d' % (k + 1)) for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11)
    fig.suptitle('Target constructor evolution   $\\rho$=%.2f   '
                 '(red title = union fragmented into several islands)' % a.rho,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(rows, indent=2))

    print('%-16s | %-24s | %-24s | %-24s' %
          ('', 'v1 ORIGINAL', 'v2 +penalty', 'v3 union-first'))
    print('%-16s | %5s %5s %5s %5s | %5s %5s %5s %5s | %5s %5s %5s %5s' %
          ('slice', 'union', 'comp', 'sprd', 'ov%', 'union', 'comp', 'sprd', 'ov%',
           'union', 'comp', 'sprd', 'ov%'))
    for x in rows:
        cells = []
        for k in ('v1', 'v2', 'v3'):
            s = x[k]
            cells += [s['union_cells'], s['union_components'], s['spread'],
                      round(s['overlap_pct'])]
        print('%-16s | %5d %5d %5d %5d | %5d %5d %5d %5d | %5d %5d %5d %5d'
              % (x['slice'], *cells))
    print()
    for k, lbl in (('v1', 'v1 ORIGINAL   '), ('v2', 'v2 +penalty   '),
                   ('v3', 'v3 union-first')):
        u = [x[k]['union_components'] for x in rows]
        print('%s union components %s   frag %d/%d   mean overlap %.0f%%'
              % (lbl, u, sum(c > 1 for c in u), len(u),
                 np.mean([x[k]['overlap_pct'] for x in rows])))
    print('\nreference: I-JEPA rectangles overlap 23.9%% of a block')
    print('wrote %s' % a.out)


if __name__ == '__main__':
    main()
