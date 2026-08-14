"""Compare encoder-context crop policies side by side, on identical slices.

The crop exists only to make the batch rectangular, but WHICH tokens it drops
turns out to matter a lot:

  prefix  stock behaviour -- a row-major prefix of the sorted context indices,
          so it always keeps the TOP of the image and deletes the bottom.
          Coherent, but on a horizontal retina it can remove all anatomy.
  window  a CONTIGUOUS run of the sorted indices, slid to the offset retaining
          the most anatomy.  Still one coherent band, no systematic bias.
  guard   uniformly random tokens with a few anatomy cells reserved.  Unbiased,
          but scatters the context into a speckle that no longer resembles the
          coherent region stock I-JEPA feeds the encoder.

Same slices, same targets, same token count in every column.
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
MODES = ["prefix", "window", "guard"]
LABEL = {"prefix": "prefix  (STOCK JEPA)",
         "window": "window  (contiguous, anatomy-aware)",
         "guard": "guard  (scattered — NOT JEPA-like)"}


def up(m):
    return np.kron(m.reshape(GRID, GRID), np.ones((PATCH, PATCH), dtype=bool))


def cfg_for(mode):
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=0.15,
                cover_min_visible_frac=0.15, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate=mode,
                enc_guard_cells=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=8)
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

    gens = {}
    for m in MODES:
        g = CurriculumMaskGenerator(**BASE, curriculum_cfg=cfg_for(m))
        g.set_epoch(50, 100)
        gens[m] = g

    picks, agg = [], {m: [0, 0, 0] for m in MODES}   # [anat_sum, zero, n]
    for start in range(0, args.batches * 64, 64):
        items = [ds[i] for i in range(start, start + 64)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = imgs.size(0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)
        outs = {}
        for m in MODES:
            random.seed(11 + start); np.random.seed(11 + start)
            torch.manual_seed(11 + start)
            outs[m] = gens[m].generate(batch_size=B, guide_grids=guides,
                                       guide_valid=valid)
        for b in range(B):
            a = anat[b]
            if a.sum() == 0:
                continue
            ctx, cnt = {}, {}
            for m in MODES:
                c = np.zeros(NP, bool); c[outs[m][0][0][b].numpy()] = True
                ctx[m] = c; cnt[m] = int((a & c).sum())
                agg[m][0] += 100 * cnt[m] / a.sum()
                agg[m][1] += (cnt[m] == 0); agg[m][2] += 1
            tgt = np.zeros(NP, bool)
            for mm in outs["prefix"][1]:
                tgt[mm[b].numpy()] = True
            if cnt["prefix"] == 0 and len(picks) < args.show:
                picks.append((start + b, imgs[b].numpy(), a, tgt, ctx, cnt))

    print(f"{'mode':10s} {'anat_ctx%':>10s} {'zero%':>8s}")
    for m in MODES:
        s, z, n = agg[m]
        print(f"{m:10s} {s/n:10.1f} {100*z/n:7.2f}%")

    ncol = 2 + len(MODES)
    fig, ax = plt.subplots(len(picks), ncol, figsize=(4.0 * ncol, 3.5 * len(picks)))
    if len(picks) == 1:
        ax = ax[None, :]
    for r, (idx, img, a, tgt, ctx, cnt) in enumerate(picks):
        g = img.mean(0); g = (g - g.min()) / (np.ptp(g) + 1e-6)
        A = up(a)
        ax[r, 0].imshow(g, cmap="gray")
        ax[r, 0].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.2)
        ax[r, 0].set_title(f"slice {idx} — anatomy {int(a.sum())}", fontsize=9)

        v = g.copy(); v[up(tgt)] = 0
        ax[r, 1].imshow(v, cmap="gray")
        ax[r, 1].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
        ax[r, 1].set_title("after TARGETS (identical in all)", fontsize=9)

        for c, m in enumerate(MODES, start=2):
            v = np.zeros_like(g); v[up(ctx[m])] = g[up(ctx[m])]
            ax[r, c].imshow(v, cmap="gray")
            ax[r, c].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
            col = "#d62728" if cnt[m] == 0 else "#2ca02c"
            ax[r, c].set_title(f"{LABEL[m]}\n{cnt[m]} anatomy cells",
                               fontsize=9, color=col, fontweight="bold")
        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle(
        "Encoder context crop policies — identical slices, identical targets, "
        "identical token count\n"
        "green = anatomy.  Only WHICH context tokens survive the batch crop differs.",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / "truncation_modes_comparison.png"
    fig.savefig(dst, dpi=125)
    print("wrote", dst)


if __name__ == "__main__":
    main()
