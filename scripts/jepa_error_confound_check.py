"""Corrected scorer-direction measurement, on the TRUE 4-block target distribution.

An earlier probe (scripts/jepa_error_vs_anatomy.py) scored all 256 patches from a
single large context block and reported corr(error, anatomy) = -0.31.  An
adversarial review showed that number is confounded:

    corr(error, distance-to-context-centroid) = +0.563   <- probe artifact
    corr(error, mean patch intensity)         = -0.435
    corr(anatomy, mean intensity)             = +0.664
    partial corr(error, anatomy | intensity)  = -0.036
    partial corr(error, anatomy | dist, var, intensity) = +0.034

The predictor was trained to fill in ~4 blocks; asking it to fill in all 256 at
once produces error that grows radially away from the context, which is not a
property of the data.  This script removes that artifact by using the PRODUCTION
collator's own context+4-target draw, scoring only the real target patches, and
accumulating over many draws until every patch has been a target many times.

It then reports the correlation of error with anatomy both raw and partialled
against the two nuisance variables, so the design question -- "if a scorer
follows error, does it walk on or off anatomy?" -- is answered on the
distribution the scorer would actually see.

Repo venv.  Reads the MIRAGE anatomy from the L0.npz produced by
scripts/mirage_logit_scale.py --dump so MIRAGE is not re-run.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.helper import init_patch_model                      # noqa: E402
from src.masks.multiblock import MaskCollator                # noqa: E402
from src.masks.utils import apply_masks                      # noqa: E402
from scripts.jepa_error_vs_anatomy import (                  # noqa: E402
    anatomy_grid, load_slice, load_jepa, CROP, PATCH, GRID)


def partial_corr(x, y, Z):
    """corr(x, y) after linearly removing every column of Z from both."""
    A = np.column_stack([np.ones(len(x))] + [z.ravel() for z in Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def patch_intensity_and_var(img):
    """Mean and variance of each 16x16 patch of the (already normalised) image."""
    g = img[0]                                               # channels identical
    p = g.unfold(0, PATCH, PATCH).unfold(1, PATCH, PATCH)    # (16,16,16,16)
    return (p.mean(dim=(-1, -2)).numpy().reshape(-1),
            p.var(dim=(-1, -2)).numpy().reshape(-1))


@torch.no_grad()
def block_errors(encoder, predictor, target_encoder, img, collator, k, device):
    """Per-patch error using the production context + 4-target-block draw.

    Returns the per-patch mean error, the number of times each patch was a
    target, and the mean distance from that patch to the context centroid over
    the draws in which it was a target (the nuisance variable to control).
    """
    x = img[None].to(device)
    h = target_encoder(x)
    h = F.layer_norm(h, (h.size(-1),))

    n = GRID * GRID
    err_sum = np.zeros(n)
    err_cnt = np.zeros(n)
    dist_sum = np.zeros(n)
    mind_sum = np.zeros(n)
    rows, cols = np.divmod(np.arange(n), GRID)

    for _ in range(k):
        _, masks_enc, masks_pred = collator([img])
        venc = [m.to(device) for m in masks_enc]
        vpred = [m.to(device) for m in masks_pred]
        z = encoder(x, venc)
        z = predictor(z, venc, vpred)                        # (npred, K_pred, D)
        h_t = apply_masks(h, vpred)
        e = F.smooth_l1_loss(z, h_t, reduction='none').mean(-1).cpu().numpy()

        ci = masks_enc[0][0].numpy()
        cr, cc = rows[ci].mean(), cols[ci].mean()
        # distance to the NEAREST VISIBLE context patch, not to the centroid.
        # The context block can be an annulus whose centroid lies in its own
        # hole, so centroid distance is a poor proxy for "how far is the
        # nearest available information".
        d_all = np.hypot(rows[:, None] - rows[ci][None, :],
                         cols[:, None] - cols[ci][None, :]).min(axis=1)
        for b, m in enumerate(masks_pred):
            idx = m[0].numpy()
            err_sum[idx] += e[b]
            err_cnt[idx] += 1
            dist_sum[idx] += np.hypot(rows[idx] - cr, cols[idx] - cc)
            mind_sum[idx] += d_all[idx]

    seen = err_cnt > 0
    err = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    mind = np.full(n, np.nan)
    err[seen] = err_sum[seen] / err_cnt[seen]
    dist[seen] = dist_sum[seen] / err_cnt[seen]
    mind[seen] = mind_sum[seen] / err_cnt[seen]
    return err, err_cnt, dist, mind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=pathlib.Path, required=True)
    ap.add_argument('--ckpt', type=pathlib.Path, required=True)
    ap.add_argument('--k', type=int, default=400, help='4-block draws per slice')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/error_vs_anatomy'))
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = np.load(a.npz, allow_pickle=True)
    names = [str(s) for s in z['names']]
    anat = anatomy_grid(z['logits']).reshape(len(names), -1)

    encoder, predictor, target_encoder, epoch = load_jepa(a.ckpt, device)
    collator = MaskCollator(input_size=(CROP, CROP), patch_size=PATCH)
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)

    E, D, V, I, A, C, MD = [], [], [], [], [], [], []
    for nm in names:
        img = load_slice(nm)
        err, cnt, dist, mind = block_errors(encoder, predictor, target_encoder, img,
                                            collator, a.k, device)
        inten, var = patch_intensity_and_var(img)
        E.append(err); D.append(dist); V.append(var); I.append(inten); C.append(cnt)
        MD.append(mind)
    E, D, V, I, C, MD = map(np.array, (E, D, V, I, C, MD))
    A = anat

    ok = np.isfinite(E) & np.isfinite(D) & np.isfinite(MD)
    e, d, v, i, an, md = E[ok], D[ok], V[ok], I[ok], A[ok], MD[ok]

    rep = {
        'n_slices': len(names), 'k_draws': a.k, 'epoch': epoch,
        'min_times_a_target': int(np.nanmin(C)),
        'mean_times_a_target': float(np.nanmean(C)),
        'patches_never_targeted': int((C == 0).sum()),
        'corr_err_anat': float(np.corrcoef(e, an)[0, 1]),
        'corr_err_dist_centroid': float(np.corrcoef(e, d)[0, 1]),
        'corr_err_dist_nearest_visible': float(np.corrcoef(e, md)[0, 1]),
        'corr_err_intensity': float(np.corrcoef(e, i)[0, 1]),
        'corr_err_variance': float(np.corrcoef(e, v)[0, 1]),
        'corr_anat_intensity': float(np.corrcoef(an, i)[0, 1]),
        'corr_anat_dist_centroid': float(np.corrcoef(an, d)[0, 1]),
        'corr_anat_dist_nearest_visible': float(np.corrcoef(an, md)[0, 1]),
        'partial_err_anat_given_intensity': partial_corr(e, an, [i]),
        'partial_err_anat_given_dist_centroid': partial_corr(e, an, [d]),
        'partial_err_anat_given_dist_nearest': partial_corr(e, an, [md]),
        'partial_err_anat_given_all_centroid': partial_corr(e, an, [d, v, i]),
        'partial_err_anat_given_all_nearest': partial_corr(e, an, [md, v, i]),
        'partial_err_anat_given_both_dists_all': partial_corr(e, an, [d, md, v, i]),
        'err_on_anatomy': float(e[an > 0.5].mean()),
        'err_off_anatomy': float(e[an <= 0.5].mean()),
    }
    per = []
    for s in range(len(names)):
        m = ok[s]
        if m.sum() > 10:
            per.append(float(np.corrcoef(E[s][m], A[s][m])[0, 1]))
    rep['corr_per_slice_mean'] = float(np.mean(per))
    rep['n_negative_slices'] = int(sum(p < 0 for p in per))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'error_confound_check.json').write_text(json.dumps(rep, indent=2))

    print('--- CORRECTED: true 4-block target distribution (ep%d, K=%d, n=%d) ---'
          % (epoch, a.k, len(names)))
    print('  coverage: every patch was a target >= %d times (mean %.1f), never-targeted %d'
          % (rep['min_times_a_target'], rep['mean_times_a_target'],
             rep['patches_never_targeted']))
    print('\n  raw correlations with error:')
    print('    anatomy   %+.4f' % rep['corr_err_anat'])
    print('    dist to context CENTROID        %+.4f' % rep['corr_err_dist_centroid'])
    print('    dist to NEAREST VISIBLE context %+.4f' % rep['corr_err_dist_nearest_visible'])
    print('    intensity %+.4f      patch variance  %+.4f' %
          (rep['corr_err_intensity'], rep['corr_err_variance']))
    print('  nuisance structure: corr(anat,intensity) %+.4f'
          '   corr(anat,dist_centroid) %+.4f   corr(anat,dist_nearest) %+.4f'
          % (rep['corr_anat_intensity'], rep['corr_anat_dist_centroid'],
             rep['corr_anat_dist_nearest_visible']))
    print('\n  partial corr(error, anatomy | ...):')
    print('    | intensity                    %+.4f' % rep['partial_err_anat_given_intensity'])
    print('    | dist CENTROID                %+.4f' % rep['partial_err_anat_given_dist_centroid'])
    print('    | dist NEAREST VISIBLE         %+.4f' % rep['partial_err_anat_given_dist_nearest'])
    print('    | centroid + var + intensity   %+.4f' % rep['partial_err_anat_given_all_centroid'])
    print('    | nearest  + var + intensity   %+.4f' % rep['partial_err_anat_given_all_nearest'])
    print('    | both dists + var + intensity %+.4f' % rep['partial_err_anat_given_both_dists_all'])
    print('\n  err on anatomy %.5f   off anatomy %.5f' %
          (rep['err_on_anatomy'], rep['err_off_anatomy']))
    print('  per-slice corr mean %+.4f   negative in %d/%d'
          % (rep['corr_per_slice_mean'], rep['n_negative_slices'], len(per)))
    print('\nwrote %s' % (a.out / 'error_confound_check.json'))


if __name__ == '__main__':
    main()
