"""Does JEPA's EMA target representation have anything to teach MIRAGE?

The bidirectional proposal replaces the (measurably degenerate) gradient-through-
mask path with a detached feature-distillation path:

    L_semantic = 1 - cos( P_psi(h_i^MIRAGE), stopgrad(z_i^JEPA) )

That is safe from the self-grading failure -- MIRAGE can no longer make its own
test easier.  But safety is not usefulness.  The loss is only worth adding if
JEPA's target features actually carry anatomical information.  Three outcomes:

  (a) JEPA features predict anatomy WELL  -> teaching is consistent with
      segmentation; probably redundant, but harmless.
  (b) JEPA features predict anatomy POORLY -> L_semantic pulls MIRAGE's decoder
      toward non-anatomical structure while L_seg pulls it back.  Actively
      harmful, and the damage grows with lambda.
  (c) JEPA carries anatomy PLUS something MIRAGE lacks -> genuinely useful.

Telling (a)/(b)/(c) apart needs three numbers, all measured here:

  1. linear probe  JEPA features  -> anatomy occupancy      (can JEPA see it?)
  2. linear probe  MIRAGE features -> anatomy occupancy     (reference ceiling)
  3. CCA / cross-prediction between the two feature sets    (already shared?)

plus the quantity that decides whether distillation injects signal or noise:
the fraction of JEPA feature variance that is NOT linearly explainable by
MIRAGE features, i.e. what would actually be new to MIRAGE.

Stage 1 (MIRAGE venv, GPU): --dump   both feature sets for the same slices
Stage 2 (repo venv, CPU):   --analyze-from   probes and CCA
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
MIRAGE_CKPT = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
               r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
JEPA_CKPT = r'D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar'

MIRAGE_RES, JEPA_RES, PATCH, GRID, POOL = 1024, 256, 16, 16, 8
ANATOMY = (1, 2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dump(out_path, n_slices):
    import os
    import cv2
    from compare_512_vs_1024 import build
    from fairvision_model_compare import MIRAGE_WS
    from src.helper import init_patch_model

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- JEPA EMA target encoder (the teacher in the proposal) ----------
    encoder, _ = init_patch_model(device, patch_size=PATCH, crop_size=JEPA_RES,
                                  model_name='vit_base')
    sd = torch.load(JEPA_CKPT, map_location='cpu', weights_only=False)
    encoder.load_state_dict({k.replace('module.', ''): v
                             for k, v in sd['target_encoder'].items()}, strict=True)
    encoder.eval()
    del sd

    # ---- MIRAGE, hooked at the decoder feature H the proposal pools -----
    mirage = build(MIRAGE_RES, MIRAGE_CKPT, device)
    grab = {}
    mirage.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(H=i[0].detach(), L0=o.detach()))

    vols = sorted(p.stem for p in TEST.glob('data_*.npz'))[:n_slices]
    ZJ, HM, AN, names = [], [], [], []
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for vi, vol_id in enumerate(vols):
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = int((vi + 0.5) / len(vols) * (len(vol) - 1))
        raw = np.asarray(vol[d], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

        # JEPA branch: 256, ImageNet-normalised, exactly as training sees it
        j = cv2.resize(unit, (JEPA_RES, JEPA_RES), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(j)[None].repeat(3, 1, 1)
        t = ((t - mean) / std)[None].to(device)
        with torch.no_grad():
            z_j = encoder(t)                                  # (1,256,768) full image
            z_j = F.layer_norm(z_j, (z_j.size(-1),))          # as in train_patch.py

        # MIRAGE branch: 1024, same field of view
        m_img = cv2.resize(unit, (MIRAGE_RES, MIRAGE_RES), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(m_img)[None, None].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            mirage({'bscan': x})
        H = grab['H']                                         # (1,384,128,128)
        h_pool = F.adaptive_avg_pool2d(H, (GRID, GRID))        # (1,384,16,16)
        P = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)   # (1,128,128)
        anat = F.adaptive_avg_pool2d(P[:, None], (GRID, GRID))[0, 0]

        ZJ.append(z_j[0].cpu().numpy())                                    # (256,768)
        HM.append(h_pool[0].permute(1, 2, 0).reshape(-1, H.shape[1]).cpu().numpy())
        AN.append(anat.cpu().numpy().reshape(-1))                          # (256,)
        names.append('%s:%d' % (vol_id, d))
        if (vi + 1) % 10 == 0:
            print('  %d/%d' % (vi + 1, len(vols)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, zj=np.stack(ZJ), hm=np.stack(HM),
                        anat=np.stack(AN), names=np.array(names))
    print('wrote %s   zj %s   hm %s' %
          (out_path, np.stack(ZJ).shape, np.stack(HM).shape))


def cv_probe(X, y, n_slices, n_patch, folds=5, alpha=1.0):
    """Ridge probe with slice-level splits (never split patches of one slice)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    idx = np.arange(n_slices)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)
    r2, preds = [], np.zeros_like(y)
    for f in range(folds):
        te = np.isin(np.repeat(np.arange(n_slices), n_patch), parts[f])
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        preds[te] = p
        ss_res = ((y[te] - p) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        r2.append(1 - ss_res / ss_tot)
    return float(np.mean(r2)), float(np.std(r2)), preds


def analyze(npz_path, out_dir):
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler

    z = np.load(npz_path, allow_pickle=True)
    ZJ, HM, AN = z['zj'], z['hm'], z['anat']
    n_slices, n_patch = ZJ.shape[0], ZJ.shape[1]
    Xj = ZJ.reshape(-1, ZJ.shape[-1])
    Xm = HM.reshape(-1, HM.shape[-1])
    y = AN.reshape(-1)
    rep = {'n_slices': int(n_slices), 'n_patches': int(len(y)),
           'jepa_dim': int(Xj.shape[1]), 'mirage_dim': int(Xm.shape[1])}

    # ---- 1/2. can each feature set linearly predict anatomy? -------------
    for tag, X in (('jepa', Xj), ('mirage', Xm)):
        r2, sd, _ = cv_probe(X, y, n_slices, n_patch)
        rep['probe_%s_to_anatomy_r2' % tag] = r2
        rep['probe_%s_to_anatomy_r2_std' % tag] = sd

    # ---- 3. how much of JEPA is ALREADY linearly in MIRAGE? --------------
    # This is the decisive quantity: distillation can only inject the part of
    # JEPA that MIRAGE cannot already reconstruct.
    sm, sj = StandardScaler().fit(Xm), StandardScaler().fit(Xj)
    Xms, Xjs = sm.transform(Xm), sj.transform(Xj)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xms)), Xms]), Xjs,
                               rcond=None)
    resid = Xjs - np.column_stack([np.ones(len(Xms)), Xms]) @ coef
    rep['jepa_variance_explained_by_mirage'] = float(
        1 - resid.var(axis=0).sum() / Xjs.var(axis=0).sum())
    rep['jepa_variance_NEW_to_mirage'] = float(
        resid.var(axis=0).sum() / Xjs.var(axis=0).sum())

    # ---- does the NEW part carry anatomy, or is it off-task? -------------
    r2_new, sd_new, _ = cv_probe(resid, y, n_slices, n_patch)
    rep['probe_jepaResidual_to_anatomy_r2'] = r2_new
    rep['probe_jepaResidual_to_anatomy_r2_std'] = sd_new

    # ---- CCA: shared subspace dimensionality -----------------------------
    k = 16
    cca = CCA(n_components=k, max_iter=500).fit(Xms, Xjs)
    U, V = cca.transform(Xms, Xjs)
    corrs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(k)]
    rep['cca_correlations'] = corrs
    rep['cca_components_above_0.5'] = int(sum(c > 0.5 for c in corrs))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'bidirectional_probe.json').write_text(json.dumps(rep, indent=2))

    print('=== Does JEPA have anatomy knowledge to teach MIRAGE? ===')
    print('  %d slices x %d patches = %d samples' % (n_slices, n_patch, len(y)))
    print('\n  linear probe -> anatomy occupancy (5-fold, slice-level splits):')
    print('    MIRAGE decoder feats (384d) : R2 = %.4f +- %.4f   <- reference'
          % (rep['probe_mirage_to_anatomy_r2'], rep['probe_mirage_to_anatomy_r2_std']))
    print('    JEPA target feats    (768d) : R2 = %.4f +- %.4f'
          % (rep['probe_jepa_to_anatomy_r2'], rep['probe_jepa_to_anatomy_r2_std']))
    print('\n  overlap between the two feature spaces:')
    print('    JEPA variance already linearly in MIRAGE : %.1f%%'
          % (100 * rep['jepa_variance_explained_by_mirage']))
    print('    JEPA variance NEW to MIRAGE              : %.1f%%'
          % (100 * rep['jepa_variance_NEW_to_mirage']))
    print('    CCA components with r > 0.5              : %d / %d'
          % (rep['cca_components_above_0.5'], len(corrs)))
    print('    top CCA correlations: ' + ' '.join('%.3f' % c for c in corrs[:6]))
    print('\n  does the NEW part carry anatomy?')
    print('    JEPA-residual -> anatomy    : R2 = %.4f +- %.4f'
          % (r2_new, sd_new))
    print('\nwrote %s' % (out_dir / 'bidirectional_probe.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--analyze-from', type=pathlib.Path)
    ap.add_argument('--n-slices', type=int, default=40)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/bidirectional')
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_slices)
    if a.analyze_from:
        return analyze(a.analyze_from, a.out)
    raise SystemExit('need --dump or --analyze-from')


if __name__ == '__main__':
    main()
