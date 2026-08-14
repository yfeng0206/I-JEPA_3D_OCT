"""Can a MIRAGE-free intensity rule stand in for the anatomy mask?

The downstream frozen eval runs over Training, Validation and Test, but the
MIRAGE soft-guide cache was only ever built for Training (12,000 files, no
Validation/Test).  To split the downstream pooled feature into an anatomy part
and a background part on every split, an anatomy mask is needed that does not
depend on MIRAGE.

This calibrates the obvious candidate -- a per-slice quantile threshold on patch
mean intensity -- against the MIRAGE occupancy labels on Training, where both
exist.  It sweeps the quantile and reports patch-level agreement, so the
substitute mask is only used downstream if it is demonstrably faithful.

Reports ROC-AUC of raw intensity against the MIRAGE label (threshold-free), plus
Dice/IoU/accuracy at each candidate quantile.
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


def patch_means(img: torch.Tensor) -> np.ndarray:
    """Mean intensity of each 16x16 patch, channel-averaged."""
    g = img.mean(0)
    p = g.unfold(0, PATCH, PATCH).unfold(1, PATCH, PATCH)
    return p.mean(dim=(-1, -2)).numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--volumes", type=int, default=40)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\anatomy_mask_calib")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    sc = os.path.join(args.slice_cache, args.split)
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(args.data_dir, args.split),
        guide_dir=os.path.join(args.guide_dir, args.split),
        num_slices=args.num_slices, slice_size=CROP, transform=paired,
        patch_size=PATCH, dilate_patches=0, occupancy_threshold=OCC_T,
        slice_cache=sc if os.path.isdir(sc) else None)

    rng = random.Random(args.seed)
    vols = sorted(rng.sample(range(len(ds.file_paths)),
                             min(args.volumes, len(ds.file_paths))))
    step = max(1, args.num_slices // args.slices_per_volume)
    idxs = [v * args.num_slices + s
            for v in vols for s in range(0, args.num_slices, step)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    I, Y, Q = [], [], {}
    quants = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.735, 0.75, 0.80]
    for q in quants:
        Q[q] = []

    for n, i in enumerate(idxs):
        img, guide, _ = ds[i]
        pm = patch_means(img)
        y = (guide[0].reshape(-1).numpy() >= OCC_T)
        I.append(pm); Y.append(y)
        for q in quants:
            Q[q].append(pm >= np.quantile(pm, q))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(idxs)}", flush=True)

    I = np.concatenate(I); Y = np.concatenate(Y)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(Y, I)
    print(f"\nMIRAGE anatomy prevalence: {Y.mean():.4f}")
    print(f"ROC-AUC of raw patch intensity vs MIRAGE label: {auc:.4f}\n")

    rows = []
    for q in quants:
        P = np.concatenate(Q[q])
        tp = float((P & Y).sum()); fp = float((P & ~Y).sum()); fn = float((~P & Y).sum())
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        acc = float((P == Y).mean())
        rows.append(dict(quantile=q, pred_rate=float(P.mean()), dice=dice,
                         iou=iou, acc=acc,
                         precision=tp / max(tp + fp, 1),
                         recall=tp / max(tp + fn, 1)))
        print(f"  q={q:.3f}  rate={P.mean():.3f}  dice={dice:.4f}  "
              f"iou={iou:.4f}  acc={acc:.4f}  P={rows[-1]['precision']:.3f}  "
              f"R={rows[-1]['recall']:.3f}")

    best = max(rows, key=lambda r: r["dice"])
    print(f"\nbest quantile by Dice: {best['quantile']} (dice {best['dice']:.4f})")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(
        dict(auc_intensity_vs_mirage=auc, prevalence=float(Y.mean()),
             sweep=rows, best=best), indent=2))
    print("wrote", out / "calibration.json")


if __name__ == "__main__":
    main()
