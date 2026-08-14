"""Encoder context under the `window` crop, across visibility floors.

Two independent dials control how much anatomy the encoder actually receives:

  cover_min_visible_frac  how much anatomy COVER refuses to hide with targets
  enc_truncate            which context tokens survive the batch crop

The floor alone is not enough -- under the stock `prefix` crop the fifth
percentile of anatomy-in-context stays pinned at 0 even at a 0.30 floor, because
the crop structurally deletes the bottom of the image.  This renders the same
slices across floors under `window`, with the stock `prefix` column alongside
for reference, so the two effects can be told apart by eye.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH, GRID = 256, 16, 16
NP = GRID * GRID
BASE = dict(input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
            allow_overlap=False)


def up(m):
    return np.kron(m.reshape(GRID, GRID), np.ones((PATCH, PATCH), dtype=bool))


def gen_for(floor, trunc):
    g = CurriculumMaskGenerator(**BASE, curriculum_cfg=dict(
        mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
        ramp_shape="linear", mirage_occupancy_threshold=0.25,
        anatomy_tau=0.10, cover_leave_frac=floor,
        cover_min_visible_frac=floor, cover_min_visible_cells=4,
        cover_fill="random_legal", enc_truncate=trunc))
    g.set_epoch(50, 100)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", nargs="+", type=float,
                    default=[0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\context_loss")
    args = ap.parse_args()

    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    ds = GuidedOCTSliceDataset(
        data_dir=r"D:\jepa_phase0\fairvision-glaucoma\data\Training",
        guide_dir=(r"C:\jepa_data\mirage_soft_guides"
                   r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training"),
        num_slices=100, slice_size=CROP, transform=paired, patch_size=PATCH,
        dilate_patches=0, occupancy_threshold=0.25,
        slice_cache=r"C:\jepa_data\slice_cache\Training")

    cols = [("prefix", args.floors[0])] + [("window", f) for f in args.floors]
    gens = {(t, f): gen_for(f, t) for t, f in cols}

    picks = []
    for start in range(0, args.batches * 64, 64):
        items = [ds[i] for i in range(start, start + 64)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = imgs.size(0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)
        out = {}
        for key, g in gens.items():
            random.seed(3 + start); np.random.seed(3 + start)
            torch.manual_seed(3 + start)
            out[key] = g.generate(batch_size=B, guide_grids=guides,
                                  guide_valid=valid)
        for b in range(B):
            a = anat[b]
            if a.sum() < 30:
                continue
            rec = {}
            for key in gens:
                me, mp = out[key]
                c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
                t = np.zeros(NP, bool)
                for m in mp:
                    t[m[b].numpy()] = True
                rec[key] = (c, t, int((a & c).sum()), 100 * (a & t).sum() / a.sum())
            if rec[("prefix", args.floors[0])][2] == 0 and len(picks) < args.show:
                picks.append((start + b, imgs[b].numpy(), a, rec))

    ncol = 1 + len(cols)
    fig, ax = plt.subplots(len(picks), ncol, figsize=(3.5 * ncol, 3.4 * len(picks)))
    if len(picks) == 1:
        ax = ax[None, :]
    for r, (idx, img, a, rec) in enumerate(picks):
        g = img.mean(0); g = (g - g.min()) / (np.ptp(g) + 1e-6)
        A = up(a)
        ax[r, 0].imshow(g, cmap="gray")
        ax[r, 0].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.2)
        ax[r, 0].set_title(f"slice {idx} — anatomy {int(a.sum())}", fontsize=9)
        for c, key in enumerate(cols, start=1):
            ctx, tgt, n_a, hid = rec[key]
            v = np.zeros_like(g); v[up(ctx)] = g[up(ctx)]
            ax[r, c].imshow(v, cmap="gray")
            ax[r, c].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
            t, f = key
            col = "#d62728" if n_a == 0 else "#2ca02c"
            head = "prefix (STOCK)" if t == "prefix" else f"window  floor {f:.2f}"
            ax[r, c].set_title(f"{head}\n{n_a} anatomy in ctx · {hid:.0f}% hidden",
                               fontsize=9, color=col, fontweight="bold")
        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle(
        "Encoder context: stock `prefix` crop vs `window` crop across visibility floors\n"
        "green = anatomy.  Raising the floor hides less anatomy AND leaves more of it "
        "in the context; the crop policy decides whether any survives at all.",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / "window_vs_floor.png"
    fig.savefig(dst, dpi=125)
    print("wrote", dst)


if __name__ == "__main__":
    main()
