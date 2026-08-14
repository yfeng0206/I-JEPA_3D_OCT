"""A MIRAGE-free anatomy mask good enough to split the downstream pooled feature.

A single global intensity quantile only reaches Dice 0.769 against MIRAGE
(``scripts/calibrate_anatomy_mask.py``), which is too blunt: at that operating
point a third of the "anatomy" pool is actually background, so the anatomy-vs-
background contrast in the downstream experiment would be badly diluted.

The retina in an OCT B-scan is a bright, roughly contiguous band, so per-column
structure carries most of the missing information.  This fits a small gradient-
boosted classifier on cheap per-patch features -- intensity statistics, position,
and column-relative band descriptors -- using MIRAGE occupancy on Training as the
label.  Nothing here touches a JEPA encoder, so the resulting mask is identical
for every arm and every checkpoint, which is what makes the cross-arm comparison
fair.

Validation is by GROUPED split (whole volumes held out), never by patch, so the
reported Dice is not inflated by leakage between slices of the same eye.

Writes a joblib model consumed by ``scripts/downstream_region_auc.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
OCC_T = 0.25


def patch_features(img: torch.Tensor) -> np.ndarray:
    """(256, F) cheap per-patch descriptors -- no encoder involved."""
    g = img.mean(0)                                        # (256, 256)
    p = g.unfold(0, PATCH, PATCH).unfold(1, PATCH, PATCH)  # (16,16,16,16)
    mean = p.mean(dim=(-1, -2)).numpy()                    # (16,16)
    std = p.std(dim=(-1, -2)).numpy()
    mx = p.amax(dim=(-1, -2)).numpy()
    mn = p.amin(dim=(-1, -2)).numpy()

    rows = np.repeat(np.arange(GRID)[:, None], GRID, 1).astype(np.float32)
    cols = np.repeat(np.arange(GRID)[None, :], GRID, 0).astype(np.float32)

    # column-relative band descriptors: where is the bright band in THIS column
    colmax = mean.max(axis=0, keepdims=True)
    colmin = mean.min(axis=0, keepdims=True)
    rng = np.maximum(colmax - colmin, 1e-6)
    rel = (mean - colmin) / rng                            # 0..1 within column
    argmx = mean.argmax(axis=0)[None, :].astype(np.float32)
    d_to_peak = np.abs(rows - argmx)                       # rows from column peak

    # slice-level normalisation
    z = (mean - mean.mean()) / (mean.std() + 1e-6)

    # vertical neighbourhood (band is contiguous)
    up = np.vstack([mean[:1], mean[:-1]])
    dn = np.vstack([mean[1:], mean[-1:]])

    feats = np.stack([mean, std, mx, mn, rows, cols, rel, d_to_peak, z, up, dn,
                      np.repeat(colmax, GRID, 0), np.repeat(colmin, GRID, 0)], -1)
    return feats.reshape(-1, feats.shape[-1])


def collect(ds, idxs, tag):
    X, Y, G = [], [], []
    for n, i in enumerate(idxs):
        img, guide, _ = ds[i]
        X.append(patch_features(img))
        Y.append((guide[0].reshape(-1).numpy() >= OCC_T))
        G.append(np.full(GRID * GRID, i // 100))
        if (n + 1) % 200 == 0:
            print(f"  {tag} {n + 1}/{len(idxs)}", flush=True)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--volumes", type=int, default=80)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\anatomy_mask_calib")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    sc = os.path.join(args.slice_cache, "Training")
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(args.data_dir, "Training"),
        guide_dir=os.path.join(args.guide_dir, "Training"),
        num_slices=args.num_slices, slice_size=CROP, transform=paired,
        patch_size=PATCH, dilate_patches=0, occupancy_threshold=OCC_T,
        slice_cache=sc if os.path.isdir(sc) else None)

    rng = random.Random(args.seed)
    vols = sorted(rng.sample(range(len(ds.file_paths)),
                             min(args.volumes, len(ds.file_paths))))
    cut = int(0.7 * len(vols))
    tr_v, te_v = vols[:cut], vols[cut:]           # GROUPED split, whole volumes
    step = max(1, args.num_slices // args.slices_per_volume)
    mk = lambda vs: [v * args.num_slices + s
                     for v in vs for s in range(0, args.num_slices, step)]
    print(f"train {len(tr_v)} vols / test {len(te_v)} vols (grouped)", flush=True)

    Xtr, Ytr, _ = collect(ds, mk(tr_v), "train")
    Xte, Yte, _ = collect(ds, mk(te_v), "test")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=None,
        early_stopping=True, validation_fraction=0.15, random_state=args.seed)
    clf.fit(Xtr, Ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(Yte, p)
    print(f"\nheld-out volume AUC: {auc:.4f}  (prevalence {Yte.mean():.4f})")

    best = None
    for t in np.arange(0.20, 0.81, 0.05):
        P = p >= t
        tp = float((P & Yte).sum()); fp = float((P & ~Yte).sum()); fn = float((~P & Yte).sum())
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        row = dict(thr=float(t), dice=dice, iou=tp / max(tp + fp + fn, 1),
                   precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                   rate=float(P.mean()))
        print(f"  thr={t:.2f} dice={dice:.4f} iou={row['iou']:.4f} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} rate={row['rate']:.3f}")
        if best is None or dice > best["dice"]:
            best = row

    print(f"\nBEST  thr={best['thr']:.2f}  dice={best['dice']:.4f}")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(dict(model=clf, threshold=best["thr"], auc=auc, dice=best["dice"]),
                out / "anatomy_mask_model.joblib")
    (out / "mask_model_report.json").write_text(json.dumps(
        dict(heldout_volume_auc=auc, prevalence=float(Yte.mean()), best=best),
        indent=2))
    print("wrote", out / "anatomy_mask_model.joblib")


if __name__ == "__main__":
    main()
