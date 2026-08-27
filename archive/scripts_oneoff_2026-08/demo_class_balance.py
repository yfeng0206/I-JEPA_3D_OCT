#!/usr/bin/env python3
"""Which anatomy class does each masking method actually hide?

The concern this answers: earlier arms were suspected of hiding more choroid
than inner retina.  Measured over 1000 slices for three methods that all use the
SAME MIRAGE anatomy map, so the only difference is mask geometry.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                       # noqa: E402
from src.masks.multiblock import MaskCollator                # noqa: E402
from demo_pipeline_trace import oracle_band                  # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/pipeline'
GRID = 16


def measure(per):
    coll = MaskCollator()
    g = torch.Generator()
    rows = {k: {'inner': [], 'chor': [], 'cells': []}
            for k in ('random', 'oracle', 'anatomy')}
    for i in range(len(per)):
        Pi, Pc = per[i, 0], per[i, 1]
        si, sc = float(Pi.sum()), float(Pc.sum())
        if si <= 1e-6 or sc <= 1e-6:
            continue
        seed = 10000 + i
        g.manual_seed(seed)
        ps = [coll._sample_block_size(coll.pred_mask_scale, g)
              for _ in range(coll.npred)]
        random.seed(seed)
        R = np.zeros((GRID, GRID), bool)
        for bh, bw in ps:
            t, l = coll._sample_block_location(bh, bw, GRID, GRID)
            for idx in coll._block_to_indices(t, l, bh, bw):
                R[idx // GRID, idx % GRID] = True
        O = oracle_band(Pi + Pc)
        if A.is_viable([Pi, Pc], 4, min_cells=4):
            M = np.logical_or.reduce(A.build_targets([Pi, Pc], 4)[0])
        else:
            M = R
        for k, m in (('random', R), ('oracle', O), ('anatomy', M)):
            rows[k]['inner'].append(Pi[m].sum() / si)
            rows[k]['chor'].append(Pc[m].sum() / sc)
            rows[k]['cells'].append(int(m.sum()))
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per = np.load(GRIDS)['per']
    rows = measure(per)
    OUT.mkdir(parents=True, exist_ok=True)

    meth = ['random', 'oracle', 'anatomy']
    lab = ['RANDOM\nrect', 'ORACLE\nribbon', 'ANATOMY\n(ours)']
    col = ['#8d99ae', '#e07a5f', '#2a9d8f']
    inn = [float(np.mean(rows[k]['inner'])) for k in meth]
    ch = [float(np.mean(rows[k]['chor'])) for k in meth]
    ce = [float(np.mean(rows[k]['cells'])) for k in meth]

    rep = {k: {'inner_masked': float(np.mean(rows[k]['inner'])),
               'choroid_masked': float(np.mean(rows[k]['chor'])),
               'cells': float(np.mean(rows[k]['cells'])),
               'inner_per_cell': float(np.mean(rows[k]['inner'])
                                       / np.mean(rows[k]['cells'])),
               'choroid_per_cell': float(np.mean(rows[k]['chor'])
                                         / np.mean(rows[k]['cells'])),
               'inner_over_choroid': float(np.mean(rows[k]['inner'])
                                           / np.mean(rows[k]['chor']))}
           for k in meth}
    (OUT / 'class_balance.json').write_text(json.dumps(rep, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
    x = np.arange(3)
    w = 0.36

    ax = axes[0]
    ax.bar(x - w/2, inn, w, label='InnerRetina', color='#4c6ef5')
    ax.bar(x + w/2, ch, w, label='Choroid', color='#f08c00')
    for j in range(3):
        ax.text(x[j]-w/2, inn[j]+.012, '%.3f' % inn[j], ha='center', fontsize=9.5)
        ax.text(x[j]+w/2, ch[j]+.012, '%.3f' % ch[j], ha='center', fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel('fraction of that class hidden')
    ax.set_title('How much of each class is masked\n(higher = more hidden)', fontsize=11)
    ax.set_ylim(0, 0.92); ax.legend(fontsize=9); ax.grid(axis='y', alpha=.25)

    ax = axes[1]
    ax.bar(x, ce, 0.5, color=col)
    for j in range(3):
        ax.text(x[j], ce[j]+2, '%.1f' % ce[j], ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel('grid cells spent (of 256)')
    ax.set_title('Mask budget spent', fontsize=11)
    ax.grid(axis='y', alpha=.25)

    ax = axes[2]
    ipc = [rep[k]['inner_per_cell']*100 for k in meth]
    cpc = [rep[k]['choroid_per_cell']*100 for k in meth]
    ax.bar(x - w/2, ipc, w, label='InnerRetina', color='#4c6ef5')
    ax.bar(x + w/2, cpc, w, label='Choroid', color='#f08c00')
    for j in range(3):
        ax.text(x[j]-w/2, ipc[j]+.03, '%.2f' % ipc[j], ha='center', fontsize=9.5)
        ax.text(x[j]+w/2, cpc[j]+.03, '%.2f' % cpc[j], ha='center', fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel('% of class mass hidden per cell spent')
    ax.set_title('Efficiency: anatomy hidden per cell\n(higher = less budget wasted)',
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=.25)

    ax = axes[3]
    ratio = [rep[k]['inner_over_choroid'] for k in meth]
    bars = ax.bar(x, ratio, 0.5, color=col)
    ax.axhline(1.0, color='k', ls='--', lw=1.2)
    ax.text(2.48, 1.005, 'balanced', fontsize=9, ha='right', va='bottom')
    for j, b in enumerate(bars):
        ax.text(x[j], ratio[j]+.006, '%.3f' % ratio[j], ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel('inner masked / choroid masked')
    ax.set_title('Class bias\n(<1 = choroid-heavy, >1 = inner-heavy)', fontsize=11)
    ax.set_ylim(0.85, 1.09); ax.grid(axis='y', alpha=.25)

    fig.suptitle('Which anatomy does each masking method hide?   '
                 '1000 FairVision slices, identical MIRAGE anatomy map',
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    f = OUT / 'class_balance.png'
    fig.savefig(f, dpi=125, bbox_inches='tight', facecolor='white')
    print('wrote %s' % f)

    print('\n%-10s%8s%15s%16s%12s' % ('method', 'cells', 'INNER masked',
                                      'CHOROID masked', 'inner/chor'))
    for k in meth:
        r = rep[k]
        print('%-10s%8.1f%15.3f%16.3f%12.3f'
              % (k, r['cells'], r['inner_masked'], r['choroid_masked'],
                 r['inner_over_choroid']))


if __name__ == '__main__':
    main()
