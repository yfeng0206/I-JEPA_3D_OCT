#!/usr/bin/env python3
"""Per-epoch spatial coverage: what does each masking method never look at?

Within one step the arms obviously differ -- that is the point.  Over a
whole epoch they should even out, because every slice is seen once and the
mask is redrawn each time.  Random masking wanders over the whole B-scan, so
its per-epoch coverage should be close to uniform.  Anatomy masking is
deliberately not uniform, and the question is how much, and where.

That matters for two reasons:

  1. any cell that is never in a target is never predicted, so the predictor
     gets no gradient there for the entire epoch
  2. if coverage is extremely concentrated, the effective number of distinct
     prediction problems per epoch collapses even when the per-step cell
     count looks healthy

Accumulates, per 16x16 grid cell and per arm:
    hidden    times the cell was removed from the context encoder
    predicted times the cell actually reached the loss (post fixed-K)

CPU only.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                        # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator     # noqa: E402
from src.masks.utils import resample_to_k                    # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/coverage'
G, K, NPRED = 16, 16, 4


def rect_arm(per, scale, guided, seed=0, epoch=50, total=100):
    B = len(per)
    guides = torch.from_numpy(((per[:, 0] + per[:, 1]) > 0.5).astype(np.float32))
    cfg = {'mode': 'mirage_envelope', 'enabled': True,
           'T_warm': 25, 'T_total': 30, 'r_max': 1.0, 'ramp_shape': 'linear'}
    gen = CurriculumMaskGenerator(input_size=(256, 256), patch_size=16,
                                  npred=NPRED, nenc=1, pred_mask_scale=scale,
                                  curriculum_cfg=cfg)
    # WITHOUT this the ramp sits at r_t=0 and the guide is never consulted, so
    # a "guided" arm silently measures pure random masking.  T_warm=25 means
    # the production run masked randomly for its first 25 epochs by design.
    gen.set_epoch(epoch if guided else 0, total)
    assert (gen.r_t > 0) == bool(guided), 'r_t=%.3f but guided=%s' % (gen.r_t, guided)
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    hid = np.zeros(G * G); prd = np.zeros(G * G)
    step = 64
    for s in range(0, B, step):
        n = min(step, B - s)
        _, pred = gen.generate(batch_size=n, guide_grids=guides[s:s + n],
                               guide_valid=torch.ones(n, dtype=torch.bool))
        for b in range(n):
            u = set()
            for p in pred:
                idx = p[b].numpy()
                prd[idx] += 1
                u.update(idx.tolist())
            hid[sorted(u)] += 1
    return hid, prd


def anatomy_arm(per, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    hid = np.zeros(G * G); prd = np.zeros(G * G); nfb = 0
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if A.is_viable(cs):
            parts, _ = A.build_targets(cs)
        else:
            nfb += 1
            parts = []
            for _ in range(NPRED):
                m = np.zeros((G, G), bool)
                r0, c0 = rng.integers(0, G - 4), rng.integers(0, G - 4)
                m[r0:r0 + 4, c0:c0 + 4] = True
                parts.append(m)
        u = np.logical_or.reduce(parts)
        hid[np.flatnonzero(u.ravel())] += 1
        for p in parts:
            fl = np.flatnonzero(p.ravel())
            prd[resample_to_k(torch.from_numpy(fl).long(), K).numpy()] += 1
    return hid, prd, nfb


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    N = len(per)
    anat = (per[:, 0] + per[:, 1]) > 0.5
    anat_freq = anat.reshape(N, -1).mean(0)

    arms = {}
    arms['random_default'] = rect_arm(per, (0.15, 0.2), guided=False) + (0,)
    arms['random_matched'] = rect_arm(per, (0.055, 0.075), guided=False) + (0,)
    arms['envelope_default'] = rect_arm(per, (0.15, 0.2), guided=True) + (0,)
    arms['anatomy'] = anatomy_arm(per)

    print('per-epoch coverage over %d slices (one pass, each slice once)' % N)
    print()
    hdr = ('%-18s %9s %9s %11s %11s %10s %9s'
           % ('arm', 'hid/slice', 'prd/slice', 'cells never',
              'cells never', 'Gini', 'corr with'))
    print(hdr)
    print('%-18s %9s %9s %11s %11s %10s %9s'
          % ('', '', '', 'hidden', 'predicted', '(predicted)', 'anatomy'))
    print('-' * len(hdr))
    res = {}
    for tag, (hid, prd, nfb) in arms.items():
        never_h = 100 * float((hid == 0).mean())
        never_p = 100 * float((prd == 0).mean())
        c = float(np.corrcoef(prd, anat_freq)[0, 1])
        r = {'hidden_per_slice': float(hid.sum() / N),
             'predicted_per_slice': float(prd.sum() / N),
             'cells_never_hidden_pct': never_h,
             'cells_never_predicted_pct': never_p,
             'gini_predicted': gini(prd),
             'corr_with_anatomy': c,
             'fallback_pct': 100.0 * nfb / N}
        res[tag] = r
        print('%-18s %9.1f %9.1f %10.1f%% %10.1f%% %10.3f %9.3f'
              % (tag, r['hidden_per_slice'], r['predicted_per_slice'],
                 never_h, never_p, r['gini_predicted'], c))

    print()
    print('coverage ratio, anatomy vs random_matched (predicted cells)')
    pa = arms['anatomy'][1]; pr = arms['random_matched'][1]
    ratio = pa / np.maximum(pr, 1e-9)
    top = np.argsort(-ratio)[:3]; bot = np.argsort(ratio)[:3]
    print('   most over-covered cells (row,col): %s'
          % [(int(i // G), int(i % G), round(float(ratio[i]), 2)) for i in top])
    print('   most under-covered cells         : %s'
          % [(int(i // G), int(i % G), round(float(ratio[i]), 3)) for i in bot])
    print('   ratio range %.3f .. %.2f' % (ratio.min(), ratio.max()))
    res['ratio_min'] = float(ratio.min()); res['ratio_max'] = float(ratio.max())

    # ---------------------------------------------------------------- figure
    order = ['random_default', 'random_matched', 'envelope_default', 'anatomy']
    fig, ax = plt.subplots(2, 5, figsize=(20, 7.6))
    for c, tag in enumerate(order):
        hid, prd = arms[tag][0], arms[tag][1]
        for r, (m, nm) in enumerate([(hid, 'hidden'), (prd, 'predicted')]):
            im = ax[r, c].imshow((m / N).reshape(G, G), cmap='viridis',
                                 vmin=0, vmax=(m / N).max())
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(tag, fontsize=11)
            if c == 0:
                ax[r, c].set_ylabel('%s\nper slice' % nm, fontsize=10)
            plt.colorbar(im, ax=ax[r, c], fraction=.046)
    ax[0, 4].imshow(anat_freq.reshape(G, G), cmap='magma')
    ax[0, 4].set_title('anatomy frequency\n(reference)', fontsize=11)
    ax[0, 4].set_xticks([]); ax[0, 4].set_yticks([])
    for tag in order:
        p = arms[tag][1].reshape(G, G).sum(1)
        ax[1, 4].plot(p / p.sum(), label=tag, lw=2)
    ax[1, 4].plot(anat_freq.reshape(G, G).sum(1) / anat_freq.sum(),
                  'k--', label='anatomy', lw=1.5)
    ax[1, 4].set_xlabel('grid row (0 = top)'); ax[1, 4].legend(fontsize=7)
    ax[1, 4].set_title('predicted-cell row profile', fontsize=11)
    ax[1, 4].grid(alpha=.3)
    fig.suptitle('Per-epoch spatial coverage: over a full pass, random masking is '
                 'nearly uniform while anatomy masking concentrates on the retina\n'
                 'bottom row is what actually reaches the loss', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / 'coverage.png', dpi=110)
    (OUT / 'coverage.json').write_text(json.dumps(res, indent=2))
    print()
    print('wrote', OUT / 'coverage.png')


if __name__ == '__main__':
    main()
