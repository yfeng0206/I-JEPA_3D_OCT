"""Render the slices where COVER + stock `prefix` leaves ZERO anatomy in context.

Columns per failing slice:
  1. image with the anatomy contour
  2. what the TARGETS remove
  3. encoder context under `prefix`   <- the failure: no anatomy reaches the encoder
  4. encoder context under the ORACLE fallback for that same slice

Default floor is 0.20. These are the slices the fallback rule has to rescue.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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


def cover_cfg(floor):
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=floor,
                cover_min_visible_frac=floor, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate="prefix")


def oracle_cfg():
    return dict(mode="anatomical_prior", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, enc_truncate="prefix")


def sets(me, mp, b):
    c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
    t = np.zeros(NP, bool)
    for m in mp:
        t[m[b].numpy()] = True
    return c, t


# one colour per predictor block; green is reserved for the anatomy contour
BLOCK_COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3",
                "#a65628", "#f781bf"]


def block_sets(mp, b):
    """The npred target blocks kept SEPARATE, so overlaps stay visible."""
    out = []
    for m in mp:
        v = np.zeros(NP, bool); v[m[b].numpy()] = True
        out.append(v)
    return out


def tint(gray, blocks, alpha=0.62):
    """Grayscale -> RGB with each target block painted its own colour."""
    rgb = np.stack([gray] * 3, axis=-1)
    for j, bm in enumerate(blocks):
        M = up(bm)
        col = np.array(mcolors.to_rgb(BLOCK_COLORS[j % len(BLOCK_COLORS)]))
        rgb[M] = (1.0 - alpha) * rgb[M] + alpha * col
    return np.clip(rgb, 0.0, 1.0)


WITHHELD = "#00e5ff"   # not masked by any target, yet never reaches the encoder


def budget_map(gray, ctx, tgt):
    """Three-way map of every patch:
       grey  = the encoder actually receives it
       cyan  = NOT masked by a target, but still withheld (crop + block sampling)
       black = removed by a target block
    """
    rgb = np.stack([gray] * 3, axis=-1)
    held = (~tgt) & (~ctx)
    rgb[up(tgt)] = 0.0
    col = np.array(mcolors.to_rgb(WITHHELD))
    M = up(held)
    rgb[M] = 0.30 * rgb[M] + 0.70 * col
    return np.clip(rgb, 0.0, 1.0), held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.20)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\arm_stats")
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

    cov = CurriculumMaskGenerator(**BASE, curriculum_cfg=cover_cfg(args.floor))
    cov.set_epoch(50, 100)
    orc = CurriculumMaskGenerator(**BASE, curriculum_cfg=oracle_cfg())
    orc.set_epoch(50, 100)

    B, picks, seen, zero = 64, [], 0, 0
    for bi in range(args.batches):
        start = bi * B
        items = [ds[i] for i in range(start, start + B)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)

        random.seed(11 + start); np.random.seed(11 + start)
        torch.manual_seed(11 + start)
        c_me, c_mp = cov.generate(batch_size=B, imgs_cpu=imgs,
                                  guide_grids=guides, guide_valid=valid)
        random.seed(11 + start); np.random.seed(11 + start)
        torch.manual_seed(11 + start)
        o_me, o_mp = orc.generate(batch_size=B, imgs_cpu=imgs,
                                  guide_grids=guides, guide_valid=valid)

        for b in range(B):
            a = anat[b]
            if a.sum() == 0:
                continue
            seen += 1
            c, t = sets(c_me, c_mp, b)
            if int((a & c).sum()) != 0:
                continue
            zero += 1
            if len(picks) < args.show:
                oc, ot = sets(o_me, o_mp, b)
                blocks = block_sets(c_mp, b)
                picks.append((start + b, imgs[b].numpy(), a, t, c, oc,
                              int((a & oc).sum()), blocks))
        if len(picks) >= args.show and bi >= 2:
            break

    print(f"floor {args.floor}: {zero}/{seen} slices with ZERO anatomy in "
          f"context ({100.0 * zero / max(seen, 1):.2f}%)")
    if not picks:
        print("no failing slices found"); return

    fig, ax = plt.subplots(len(picks), 5, figsize=(19.5, 3.7 * len(picks)))
    if len(picks) == 1:
        ax = ax[None, :]
    for r, (idx, img, a, t, c, oc, n_resc, blocks) in enumerate(picks):
        g = img.mean(0); g = (g - g.min()) / (np.ptp(g) + 1e-6)
        A = up(a)

        ax[r, 0].imshow(g, cmap="gray")
        ax[r, 0].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.1)
        ax[r, 0].set_title(f"slice {idx}\nanatomy = {int(a.sum())} cells",
                           fontsize=9)

        ax[r, 1].imshow(tint(g, blocks))
        ax[r, 1].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.4)
        per = " ".join(f"{int((a & bm).sum())}" for bm in blocks)
        ax[r, 1].set_title(
            f"1) {len(blocks)} TARGET blocks\n"
            f"{int((a & t).sum())}/{int(a.sum())} anatomy masked  ·  per block: {per}",
            fontsize=9)

        bm_img, held = budget_map(g, c, t)
        ax[r, 2].imshow(bm_img)
        ax[r, 2].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.4)
        ax[r, 2].set_title(
            f"2) WHERE THE REST WENT\n"
            f"cyan = withheld by CROP: {int(held.sum())} cells "
            f"({int((a & held).sum())} anatomy)",
            fontsize=9, color="#0077aa", fontweight="bold")

        v = np.zeros_like(g); v[up(c)] = g[up(c)]
        ax[r, 3].imshow(v, cmap="gray")
        ax[r, 3].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.1)
        ax[r, 3].set_title(
            f"3) ENCODER GETS — prefix\n{int(c.sum())} cells, "
            f"0 anatomy  (FAILURE)",
            fontsize=9, color="#d62728", fontweight="bold")

        v = np.zeros_like(g); v[up(oc)] = g[up(oc)]
        ax[r, 4].imshow(v, cmap="gray")
        ax[r, 4].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.1)
        ok = n_resc > 0
        ax[r, 4].set_title(
            f"ORACLE FALLBACK\n{n_resc} anatomy cells  ({'RESCUED' if ok else 'STILL BLANK'})",
            fontsize=9, color="#2ca02c" if ok else "#d62728", fontweight="bold")

        for cc in range(5):
            ax[r, cc].set_xticks([]); ax[r, cc].set_yticks([])

    handles = [mpatches.Patch(color=BLOCK_COLORS[j], label=f"target block {j + 1}")
               for j in range(4)]
    handles.append(mpatches.Patch(color=WITHHELD,
                                  label="NOT masked, but withheld from encoder by the crop"))
    handles.append(mpatches.Patch(color="#00ff00", label="anatomy (MIRAGE guide)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.002))

    fig.suptitle(
        f"COVER floor {args.floor:.2f}, stock `prefix` crop — why the encoder ends up with NO anatomy\n"
        "The targets remove little. The CROP (cyan) removes the rest, including every anatomy cell the floor protected.",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"zero_anatomy_floor{int(args.floor * 100)}.png"
    fig.savefig(dst, dpi=120)
    print("wrote", dst)


if __name__ == "__main__":
    main()
