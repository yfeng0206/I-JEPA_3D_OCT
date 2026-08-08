#!/usr/bin/env python3
"""Visualise what the collation fix changes.

Row per slice, columns:

  1  the anatomy guide (MIRAGE soft score, 16x16)
  2  the 4 targets the sampler produced
  3  what the CONTEXT encoder hides   -- identical in both policies
  4  cells the predictor is asked to reconstruct, OLD global-min policy
  5  cells the predictor is asked to reconstruct, NEW fixed-K policy

The point of column 3 is that it does not change: the context mask is built
by subtracting the FULL untruncated union, so this fix cannot alter what the
encoder sees.  Columns 4 and 5 are where the loss lives.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402
from matplotlib.colors import ListedColormap                 # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import anatomy_target_sampler_v2 as A                        # noqa: E402
from src.masks.utils import resample_to_k                    # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = pathlib.Path('results/masking/collation')
K = 16
NROW = 5
G = 16


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    torch.manual_seed(0)

    rows = []
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        parts, _ = A.build_targets(cs)
        rows.append((i, cs, parts))
        if len(rows) >= 400:
            break

    # Pick slices spanning small -> large union so the effect is visible.
    rows.sort(key=lambda r: sum(int(p.sum()) for p in r[2]))
    picks = [rows[int(x)] for x in np.linspace(0, len(rows) - 1, NROW)]

    # A realistic batch minimum: one tiny target collapses everything.
    gmin = 1

    tgt_cmap = ListedColormap(['#0d1117', '#e6194b', '#3cb44b', '#4363d8', '#f58231'])
    fig, ax = plt.subplots(NROW, 5, figsize=(15.5, 3.1 * NROW))
    for r, (idx, cs, parts) in enumerate(picks):
        score = np.asarray(cs[0]) + np.asarray(cs[1])
        union = np.logical_or.reduce(parts)

        lab = np.zeros((G, G), np.int32)
        for t, p in enumerate(parts):
            lab[p] = t + 1

        old = np.zeros((G, G), bool)
        new = np.zeros((G, G), bool)
        for p in parts:
            fl = torch.from_numpy(np.flatnonzero(p.ravel())).long()
            old.ravel()[fl[:gmin].numpy()] = True
            new.ravel()[np.unique(resample_to_k(fl, K).numpy())] = True
        # Count DISTINCT cells off the grid, not per target: inner and choroid
        # regions can share a cell when both pass tau (6.8% of slices, mean
        # 0.10 cells), so summing per target would overcount.
        n_old = int(old.sum())
        n_new = int(new.sum())

        ax[r, 0].imshow(score, cmap='magma', vmin=0, vmax=1)
        ax[r, 0].set_ylabel('slice %d' % idx, fontsize=9)
        ax[r, 1].imshow(lab, cmap=tgt_cmap, vmin=0, vmax=4)
        ax[r, 2].imshow(~union, cmap='gray', vmin=0, vmax=1)
        ax[r, 3].imshow(old, cmap='gray', vmin=0, vmax=1)
        ax[r, 4].imshow(new, cmap='gray', vmin=0, vmax=1)

        tot = int(union.sum())
        if r == 0:
            for c, t in enumerate([
                '1. anatomy guide\nMIRAGE score',
                '2. four targets\nT1 T2 T3 T4',
                '3. context encoder sees\n(white = visible)',
                '4. OLD: predicted cells\nglobal-min K=1',
                '5. NEW: predicted cells\nfixed-K resample K=%d' % K,
            ]):
                ax[r, c].set_title(t, fontsize=10)
        ax[r, 2].set_xlabel('%d cells hidden' % tot, fontsize=8)
        ax[r, 3].set_xlabel('%d of %d predicted  (%.0f%%)'
                            % (n_old, tot, 100 * n_old / tot),
                            fontsize=8, color='#b00020')
        ax[r, 4].set_xlabel('%d of %d predicted  (%.0f%%)'
                            % (n_new, tot, 100 * n_new / tot),
                            fontsize=8, color='#0a7d00')
        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle('Collation fix: the context mask is unchanged, the predicted '
                 'target is not\ncolumn 3 is identical under both policies; '
                 'columns 4 vs 5 are where 92.8% of the signal was lost',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = OUT / 'collation_fix.png'
    fig.savefig(p, dpi=120)
    print('wrote', p)


if __name__ == '__main__':
    main()
