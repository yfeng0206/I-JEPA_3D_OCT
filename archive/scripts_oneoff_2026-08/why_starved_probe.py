#!/usr/bin/env python3
"""Why the predictor was starved, and why it was also biased.

Two separate causes.

CAUSE 1 -- the guard was written for rectangles and never fired.
Original I-JEPA draws ONE block size per group per batch
(`pred_sizes[p]`), then `_block_to_indices(top, left, bh, bw)` gives every
sample in that group exactly bh*bw indices.  All equal, so
`min(t.numel() ...)` returns that same number and `t[:min]` is a no-op.  The
min-truncate was a defensive line that could not bite.  Anatomy targets are
grown from the data, so their sizes are ragged and it bites hard.

CAUSE 2 -- the survivors are not a random sample.
`t[:K]` keeps the FIRST K entries.  Anatomy indices are produced in raster
order, so the surviving cell is the topmost-leftmost cell of each target,
every time.  The predictor was therefore trained almost entirely on the
upper boundary of the retinal band rather than on the tissue.

This quantifies both and renders the bias.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import anatomy_target_sampler_v2 as A                        # noqa: E402
from src.masks.utils import resample_to_k                    # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = pathlib.Path('results/masking/collation')
G = 16
K = 16


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    torch.manual_seed(0)

    sorted_ok = 0
    n = 0
    smallest = []
    heat_all = np.zeros((G, G))
    heat_old = np.zeros((G, G))
    heat_new = np.zeros((G, G))
    rows_all, rows_old, rows_new = [], [], []

    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        parts, _ = A.build_targets(cs)
        n += 1
        sizes = [int(p.sum()) for p in parts]
        smallest.append(min(sizes))
        for p in parts:
            fl = np.flatnonzero(p.ravel())
            sorted_ok += int(np.all(np.diff(fl) > 0))
            heat_all.ravel()[fl] += 1
            rows_all.extend((fl // G).tolist())
            # OLD: front-slice to the batch minimum, which is 1 in 99.8% of
            # microbatches at batch 64.
            o = fl[:1]
            heat_old.ravel()[o] += 1
            rows_old.extend((o // G).tolist())
            nw = np.unique(resample_to_k(torch.from_numpy(fl).long(), K).numpy())
            heat_new.ravel()[nw] += 1
            rows_new.extend((nw // G).tolist())

    smallest = np.array(smallest)
    print('slices                          %d' % n)
    print('targets in raster order         %d / %d' % (sorted_ok, n * 4))
    print()
    print('smallest target in a slice: mean %.1f  median %d  p10 %d  min %d'
          % (smallest.mean(), np.median(smallest), np.percentile(smallest, 10),
             smallest.min()))
    for t in (1, 2, 3, 5):
        print('   slices whose smallest target is <= %d cells   %5.1f%%'
              % (t, 100 * (smallest <= t).mean()))
    print()
    for b in (16, 32, 64):
        p_any = 1 - (1 - (smallest <= 1).mean()) ** b
        print('P(a 1-cell target appears in a microbatch of %2d) = %5.1f%%'
              % (b, 100 * p_any))
    print()
    ra, ro, rn = np.array(rows_all), np.array(rows_old), np.array(rows_new)
    print('mean grid ROW of the cells the predictor actually sees')
    print('   all anatomy cells          %.2f' % ra.mean())
    print('   OLD front-slice survivors  %.2f   <- pulled to the top edge' % ro.mean())
    print('   NEW resampled survivors    %.2f' % rn.mean())
    print()
    print('fraction of survivors in the TOP THIRD of each target band')
    print('   all anatomy cells          %.1f%%' % (100 * (ra < np.percentile(ra, 33)).mean()))
    print('   OLD front-slice survivors  %.1f%%' % (100 * (ro < np.percentile(ra, 33)).mean()))
    print('   NEW resampled survivors    %.1f%%' % (100 * (rn < np.percentile(ra, 33)).mean()))

    res = {'n_slices': int(n),
           'targets_raster_ordered': int(sorted_ok), 'targets_total': int(n * 4),
           'smallest_target': {'mean': float(smallest.mean()),
                               'median': float(np.median(smallest)),
                               'min': int(smallest.min()),
                               'pct_le_1': float(100 * (smallest <= 1).mean())},
           'mean_row': {'all': float(ra.mean()), 'old': float(ro.mean()),
                        'new': float(rn.mean())}}
    (OUT / 'why_starved.json').write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    for a, h, t in zip(
            ax[:3], [heat_all, heat_old, heat_new],
            ['all anatomy cells\n(what the sampler chose)',
             'OLD: front-slice survivors\nt[:1], raster order',
             'NEW: fixed-K resample\nK=%d' % K]):
        a.imshow(h / max(h.max(), 1), cmap='inferno', vmin=0, vmax=1)
        a.set_title(t, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    ax[3].plot(np.bincount(ra, minlength=G) / len(ra), label='all anatomy', lw=2)
    ax[3].plot(np.bincount(ro, minlength=G) / len(ro), label='OLD front-slice', lw=2)
    ax[3].plot(np.bincount(rn, minlength=G) / len(rn), label='NEW resample', lw=2)
    ax[3].set_xlabel('grid row (0 = top of B-scan)')
    ax[3].set_ylabel('fraction of predicted cells')
    ax[3].set_title('row distribution', fontsize=11)
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=.3)
    fig.suptitle('Why the predictor was starved: truncation kept the first cell in '
                 'raster order,\nso it trained on the TOP EDGE of the retina, not the '
                 'tissue', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'why_starved.png', dpi=120)
    print()
    print('wrote', OUT / 'why_starved.png')


if __name__ == '__main__':
    main()
