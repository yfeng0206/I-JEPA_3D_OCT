#!/usr/bin/env python
"""Does JEPA's latent know anatomy, and does it agree with MIRAGE?

L_rel rests on an assumption nobody has tested: that JEPA and MIRAGE should
agree about which patches resemble which other patches.  If that assumption is
false, forcing MIRAGE toward JEPA's similarity structure necessarily drags
MIRAGE away from its own (correct) tissue organisation -- which is exactly the
monotone Dice degradation measured in the GOALS sweep.

This script tests the assumption directly using human-annotated GOALS labels.
Nothing is trained.  Three questions:

  Q1 separability -- given a patch's latent, can you tell inner retina from
     choroid?  Measured per model with a cross-validated linear probe and a
     silhouette score.  If JEPA scores below MIRAGE, JEPA is the weaker teacher
     on the axis we care about.

  Q2 agreement -- correlate the two 256x256 patch-similarity (Gram) matrices.
     This is the quantity L_rel drives to zero.  Low agreement means L_rel is
     asking MIRAGE to move a long way.

  Q3 tap point -- run Q1/Q2 at both MIRAGE tap points: the encoder output
     (256x768, shape-matched to JEPA) and the decoder feature H0 (384x64x64,
     one 1x1 conv from the logits) where the adapter currently sits.

Both models see the same image and both emit 256 tokens on a 16x16 grid, so
tokens are spatially aligned and each one inherits a ground-truth class by
majority vote over its pixels.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from goals_eval import load_pairs                      # noqa: E402
from jepa_to_mirage_probe import (build_mirage, build_jepa,     # noqa: E402
                                  IMNET_MEAN, IMNET_STD)

GRID = 16
OUT = REPO / 'results/masking/latent_probe'
INNER, CHOROID = 1, 2


def tokens_from_labels(gt, grid=GRID):
    """Majority class per grid cell, plus that cell's purity."""
    h = gt.shape[0] // grid
    cells = gt.reshape(grid, h, grid, h).transpose(0, 2, 1, 3).reshape(grid * grid, -1)
    lab = np.zeros(grid * grid, np.int64)
    pur = np.zeros(grid * grid, np.float32)
    for i, c in enumerate(cells):
        cnt = np.bincount(c, minlength=4)
        lab[i] = cnt.argmax()
        pur[i] = cnt.max() / cnt.sum()
    return lab, pur


def gram(x):
    x = F.normalize(x.float(), dim=-1)
    return x @ x.transpose(-2, -1)


def separability(feats, labels, groups, name):
    """Cross-validated linear probe AUC + silhouette for inner vs choroid.

    Folds are grouped BY IMAGE. Splitting flattened tokens at random puts
    patches from the same B-scan in both train and test, so the probe can
    memorise an image rather than learn the tissue distinction -- which is what
    produced a saturated AUC of 1.0000 for every representation. Scaling is
    fitted inside each fold for the same reason.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = np.isin(labels, (INNER, CHOROID))
    X, y, g = feats[m], (labels[m] == CHOROID).astype(int), groups[m]
    if len(np.unique(y)) < 2:
        return None
    aucs = []
    for tr, te in StratifiedGroupKFold(5).split(X, y, groups=g):
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    sil = silhouette_score(StandardScaler().fit_transform(X), y, metric='cosine')
    print('  %-34s probe AUC %.4f +/- %.4f   silhouette %+.4f   (n=%d, %d images)'
          % (name, np.mean(aucs), np.std(aucs), sil, len(y), len(np.unique(g))))
    return {'probe_auc': float(np.mean(aucs)), 'probe_auc_sd': float(np.std(aucs)),
            'silhouette': float(sil), 'n_tokens': int(len(y)),
            'n_images': int(len(np.unique(g))), 'grouped_cv': True}


def cka(X, Y):
    """Linear CKA between two representations of the same tokens."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = REPO.parent  # placeholder, overridden below
    jepa_ckpt = (sys.argv[1] if len(sys.argv) > 1 else
                 r'D:\jepa_phase0\runs\patch_mirage_anatomy\jepa_patch_mirage-ep30.pth.tar')

    mir = build_mirage(dev)
    enc = build_jepa(pathlib.Path(jepa_ckpt), dev)
    sem = mir.output_adapters['semseg']

    grab = {}
    sem.proj_dec.register_forward_hook(
        lambda m, i, o: grab.update(enc_out=i[0].detach()))
    sem.final_layer.register_forward_hook(
        lambda m, i, o: grab.update(h0=i[0].detach()))

    imgs, gts, names = load_pairs()
    print('GOALS held-out images : %d' % len(imgs))
    print('JEPA teacher          : %s' % pathlib.Path(jepa_ckpt).name)
    print()

    M_enc, M_h0, J, L, P = [], [], [], [], []
    for i in range(len(imgs)):
        x512 = torch.from_numpy(imgs[i])[None, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x512})
        e = grab['enc_out'].float()                    # (1, 256, 768)
        # The encoder emits patch tokens plus global tokens; keep the patches.
        e = e[:, :GRID * GRID]
        h = grab['h0'].float()                         # (1, 384, 64, 64)
        h = F.adaptive_avg_pool2d(h, (GRID, GRID)).flatten(2).transpose(1, 2)

        import cv2
        b = cv2.resize(imgs[i], (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        xj = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(dev)
        with torch.no_grad():
            z = F.layer_norm(enc(xj), (768,))          # (1, 256, 768)

        lab, pur = tokens_from_labels(gts[i])
        M_enc.append(e[0].cpu().numpy()); M_h0.append(h[0].cpu().numpy())
        J.append(z[0].cpu().numpy()); L.append(lab); P.append(pur)

    M_enc = np.stack(M_enc); M_h0 = np.stack(M_h0); J = np.stack(J)
    L = np.stack(L); P = np.stack(P)

    # Only trust cells that are mostly one tissue.
    keep = P.reshape(-1) >= 0.7
    # Image id per token, so cross-validation can be grouped by B-scan.
    img_id = np.repeat(np.arange(len(L)), L.shape[1]).reshape(-1)[keep]
    fe, fh, fj, fl = (M_enc.reshape(-1, M_enc.shape[-1])[keep],
                      M_h0.reshape(-1, M_h0.shape[-1])[keep],
                      J.reshape(-1, J.shape[-1])[keep],
                      L.reshape(-1)[keep])

    print('Q1  Can the latent tell INNER RETINA from CHOROID?')
    print('    (linear probe, 5-fold CV GROUPED BY IMAGE, cells >=70%% one tissue)')
    res = {'jepa_ckpt': str(jepa_ckpt), 'n_images': len(imgs),
           'n_tokens_kept': int(keep.sum()), 'separability': {}}
    res['separability']['mirage_encoder'] = separability(fe, fl, img_id, 'MIRAGE encoder  (256x768)')
    res['separability']['mirage_h0'] = separability(fh, fl, img_id, 'MIRAGE H0 decoder (384) <-adapter')
    res['separability']['jepa'] = separability(fj, fl, img_id, 'JEPA encoder     (256x768)')

    print()
    print('Q2  Do the two models AGREE about patch-to-patch similarity?')
    print('    (this is exactly what L_rel drives to zero)')
    gj = gram(torch.from_numpy(J)).numpy()
    ge = gram(torch.from_numpy(M_enc)).numpy()
    gh = gram(torch.from_numpy(M_h0)).numpy()
    iu = np.triu_indices(GRID * GRID, k=1)
    agr = {}
    for nm, g in (('MIRAGE encoder vs JEPA', ge), ('MIRAGE H0 vs JEPA', gh)):
        r = np.mean([np.corrcoef(g[i][iu], gj[i][iu])[0, 1] for i in range(len(g))])
        mse = float(np.mean((g - gj) ** 2))
        print('  %-30s Pearson r %+.4f   MSE %.4f' % (nm, r, mse))
        agr[nm] = {'pearson_r': float(r), 'gram_mse': mse}
    res['gram_agreement'] = agr

    print()
    print('Q3  Linear CKA (representation-level similarity)')
    res['cka'] = {
        'mirage_encoder_vs_jepa': cka(fe, fj),
        'mirage_h0_vs_jepa': cka(fh, fj),
        'mirage_encoder_vs_h0': cka(fe, fh),
    }
    for k, v in res['cka'].items():
        print('  %-28s %.4f' % (k, v))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'latent_anatomy.json').write_text(json.dumps(res, indent=2))
    print('\nwrote', OUT / 'latent_anatomy.json')


if __name__ == '__main__':
    main()

