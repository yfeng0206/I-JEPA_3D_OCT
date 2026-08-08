"""Is the residual adapter actually able to learn, and where?

Two questions the logit-scale sweep left open.

  Q1  "60.9% dead cells at alpha=1" -- benign or fatal?
      Benign reading: the dead cells are confident vitreous/sclera, exactly
      where MIRAGE should be preserved, and the live cells are the anatomy
      boundary where refinement matters.
      Fatal reading: boundary cells are dead too, so the adapter can only
      edit places that do not matter.
      Resolved by splitting authority by cell type.

  Q2  How much gradient does the adapter get at initialisation?
      With dL = 0 (zero-init), shifting anatomy logits by +a and non-anatomy
      by -a gives M(a) = A e^a / (A e^a + B e^-a) with A = M, B = 1 - M, so

          dM/da |_{a=0} = 2 M (1 - M)

      the logistic derivative: maximal at M = 0.5, vanishing at M in {0,1}.
      Since 73.9% of pixels sit at M < 0.01, most of the image contributes
      almost no gradient regardless of alpha.  Verified numerically here.

      Important counterweight: the adapter is convolutional, so its weights
      are SHARED across space.  Zero gradient at a location does not mean
      that location can never change -- only that it does not itself teach.

Input: the L0.npz written by scripts/mirage_logit_scale.py --dump
Pure numpy, no GPU, no model load.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CLASS_NAMES = ('Elsewhere', 'InnerRetina', 'Choroid', 'Background')
ANATOMY = (1, 2)
NON_ANATOMY = (0, 3)
POOL = 8
ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0)

# pooled-soft-score bins defining cell type
BG_HI = 0.05      # below this: interior background
AN_LO = 0.95      # above this: interior anatomy


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def pool(m):
    n, h, w = m.shape
    return m.reshape(n, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/logit_scale'))
    a = ap.parse_args()

    L0 = np.load(a.npz, allow_pickle=True)['logits']
    P = softmax(L0, axis=1)
    M = P[:, ANATOMY].sum(axis=1)            # (N,128,128)
    soft = pool(M)                           # (N,16,16)
    rep = {'n_slices': int(L0.shape[0])}

    # ---- cell typing ----------------------------------------------------
    kind = np.full(soft.shape, 1, dtype=np.int8)   # 1 = boundary/mixed
    kind[soft < BG_HI] = 0                         # 0 = interior background
    kind[soft > AN_LO] = 2                         # 2 = interior anatomy
    names = {0: 'interior-background', 1: 'boundary/mixed', 2: 'interior-anatomy'}
    rep['cell_type_frac'] = {names[k]: float((kind == k).mean()) for k in (0, 1, 2)}

    # ---- Q1: authority split by cell type -------------------------------
    d = np.zeros((1, 4, 1, 1), dtype=np.float32)
    d[0, list(ANATOMY)] = 1.0
    d[0, list(NON_ANATOMY)] = -1.0

    rep['authority_by_cell_type'] = {}
    for al in ALPHAS:
        up = pool(softmax(L0 + al * d, axis=1)[:, ANATOMY].sum(axis=1))
        dn = pool(softmax(L0 - al * d, axis=1)[:, ANATOMY].sum(axis=1))
        auth = up - dn
        rep['authority_by_cell_type'][str(al)] = {
            names[k]: {'mean': float(auth[kind == k].mean()),
                       'dead_frac': float((auth[kind == k] < 0.01).mean())}
            for k in (0, 1, 2)
        }
        if al == 2.0:
            rep['authority_vs_soft_corr'] = float(
                np.corrcoef(auth.ravel(), np.minimum(soft, 1 - soft).ravel())[0, 1])

    # ---- Q2: gradient at initialisation ---------------------------------
    grad_px = 2.0 * M * (1.0 - M)            # analytic dM/da at a=0
    # numeric check by central difference
    h = 1e-3
    num = (softmax(L0 + h * d, axis=1)[:, ANATOMY].sum(axis=1)
           - softmax(L0 - h * d, axis=1)[:, ANATOMY].sum(axis=1)) / (2 * h)
    rep['grad_analytic_vs_numeric_maxabs'] = float(np.abs(grad_px - num).max())

    rep['grad_pixel'] = {'mean': float(grad_px.mean()),
                         **{f'p{q}': float(np.percentile(grad_px, q))
                            for q in (50, 90, 99)}}
    # a cell can only teach if it contains at least one non-saturated pixel
    is_soft = (M > 0.01) & (M < 0.99)
    cell_has_soft = is_soft.reshape(is_soft.shape[0], 16, POOL, 16, POOL).any(axis=(2, 4))
    rep['cells_with_any_soft_pixel_frac'] = float(cell_has_soft.mean())
    rep['soft_pixel_frac'] = float(is_soft.mean())
    gcell = pool(grad_px)
    rep['grad_cell_by_type'] = {names[k]: float(gcell[kind == k].mean())
                                for k in (0, 1, 2)}
    rep['grad_share_from_boundary_cells'] = float(
        gcell[kind == 1].sum() / gcell.sum())

    # ---- Q3: can the adapter ever fix the known mid-retina gap? ---------
    # MIRAGE inherits a taxonomy hole: GOALS does not label INL/OPL/ONL/
    # photoreceptors/RPE, so the middle retina falls into Elsewhere and splits
    # the band in two.  Since gradient at init is 2M(1-M), gap pixels that sit
    # at M ~ 0 are unreachable: the adapter cannot be taught to include them.
    anat = M > 0.5
    gap_M, gap_g, gap_h = [], [], []
    for i in range(M.shape[0]):
        for c in range(M.shape[2]):
            rows = np.flatnonzero(anat[i, :, c])
            if rows.size < 2:
                continue
            interior = slice(rows[0] + 1, rows[-1])
            seg = anat[i, interior, c]
            if seg.size == 0 or seg.all():
                continue
            hole = ~seg
            gap_M.append(M[i, interior, c][hole])
            gap_g.append(grad_px[i, interior, c][hole])
            gap_h.append(hole.sum() / M.shape[1])
    if gap_M:
        gm = np.concatenate(gap_M)
        gg = np.concatenate(gap_g)
        rep['midretina_gap'] = {
            'columns_with_gap_frac': float(len(gap_h) / (M.shape[0] * M.shape[2])),
            'mean_gap_height_frac': float(np.mean(gap_h)),
            'gap_M_mean': float(gm.mean()),
            'gap_M_p50': float(np.percentile(gm, 50)),
            'gap_M_p90': float(np.percentile(gm, 90)),
            'gap_saturated_frac': float((gm < 0.01).mean()),
            'gap_grad_mean': float(gg.mean()),
            'gap_grad_p50': float(np.percentile(gg, 50)),
            'gap_grad_vs_boundary_ratio': float(gg.mean() / grad_px[(M > 0.01) & (M < 0.99)].mean()),
        }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'adapter_feasibility.json').write_text(json.dumps(rep, indent=2))

    # ---- console --------------------------------------------------------
    print('slices %d' % rep['n_slices'])
    print('\n--- cell types (16x16 grid, pooled soft score) ---')
    for k in (0, 1, 2):
        print('  %-20s %6.1f%%' % (names[k], 100 * rep['cell_type_frac'][names[k]]))

    print('\n--- Q1: adapter authority by cell type ---')
    print('  %6s | %-28s | %-28s | %-28s' %
          ('alpha', 'interior-background', 'boundary/mixed', 'interior-anatomy'))
    for al in ALPHAS:
        r = rep['authority_by_cell_type'][str(al)]
        cells = []
        for k in (0, 1, 2):
            v = r[names[k]]
            cells.append('mean %.3f  dead %5.1f%%' % (v['mean'], 100 * v['dead_frac']))
        print('  %6.1f | %-28s | %-28s | %-28s' % (al, *cells))
    print('  corr(authority, min(soft,1-soft)) at alpha=2 : %.4f'
          % rep['authority_vs_soft_corr'])

    print('\n--- Q2: gradient at init  dM/da = 2M(1-M) ---')
    print('  analytic vs numeric max|diff| : %.2e' % rep['grad_analytic_vs_numeric_maxabs'])
    g = rep['grad_pixel']
    print('  per-pixel  mean %.4f   p50 %.4f   p90 %.4f   p99 %.4f'
          % (g['mean'], g['p50'], g['p90'], g['p99']))
    print('  non-saturated pixels                : %.1f%%' % (100 * rep['soft_pixel_frac']))
    print('  cells containing >=1 soft pixel     : %.1f%%'
          % (100 * rep['cells_with_any_soft_pixel_frac']))
    for k in (0, 1, 2):
        print('  mean cell gradient %-20s %.4f' % (names[k], rep['grad_cell_by_type'][names[k]]))
    print('  share of total gradient from boundary cells: %.1f%%'
          % (100 * rep['grad_share_from_boundary_cells']))

    if 'midretina_gap' in rep:
        g2 = rep['midretina_gap']
        print('\n--- Q3: is the known mid-retina gap reachable by the adapter? ---')
        print('  columns with an interior gap   : %.1f%%'
              % (100 * g2['columns_with_gap_frac']))
        print('  mean gap height (frac of image): %.4f' % g2['mean_gap_height_frac'])
        print('  gap M    mean %.4f  p50 %.4f  p90 %.4f'
              % (g2['gap_M_mean'], g2['gap_M_p50'], g2['gap_M_p90']))
        print('  gap pixels saturated (M<0.01)  : %.1f%%' % (100 * g2['gap_saturated_frac']))
        print('  gap gradient mean %.5f  (%.3fx the mean soft-pixel gradient)'
              % (g2['gap_grad_mean'], g2['gap_grad_vs_boundary_ratio']))

    _figure(a.out, soft, kind, gcell, rep, names)
    print('\nwrote %s' % (a.out / 'adapter_feasibility.json'))


def _figure(out, soft, kind, gcell, rep, names):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    al = [float(k) for k in rep['authority_by_cell_type']]
    for k, c in zip((0, 1, 2), ('tab:gray', 'tab:red', 'tab:green')):
        ax[0].plot(al, [rep['authority_by_cell_type'][str(x)][names[k]]['mean'] for x in al],
                   'o-', color=c, label=names[k])
    ax[0].set_xscale('log'); ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel('mean authority'); ax[0].legend(fontsize=8)
    ax[0].set_title('Q1: authority by cell type'); ax[0].grid(alpha=0.3)

    ax[1].scatter(soft.ravel(), gcell.ravel(), s=3, alpha=0.2)
    ax[1].set_xlabel('pooled soft score'); ax[1].set_ylabel(r'cell gradient $2M(1-M)$')
    ax[1].set_title('Q2: gradient concentrates at boundary')

    ax[2].imshow(kind[0], cmap='coolwarm', vmin=0, vmax=2)
    ax[2].set_title('cell types (slice 0)\ngrey=bg  red=boundary  green=anat')
    ax[2].set_xticks([]); ax[2].set_yticks([])

    im = ax[3].imshow(gcell[0], cmap='magma')
    ax[3].set_title('per-cell gradient at init (slice 0)')
    ax[3].set_xticks([]); ax[3].set_yticks([])
    plt.colorbar(im, ax=ax[3], fraction=0.046)

    fig.suptitle('Residual-adapter feasibility: where can it act, where can it learn', fontsize=13)
    fig.tight_layout()
    p = out / 'adapter_feasibility.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % p)


if __name__ == '__main__':
    main()
