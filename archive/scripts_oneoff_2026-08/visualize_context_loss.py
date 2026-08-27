"""Where does the encoder's anatomy context actually go?

COVER enforces its anatomy-visibility floor on the TARGET union.  What the
encoder receives is a different thing entirely:

    enc block (85-100% of the grid)  ->  minus the target union  ->  truncated
    to the batch-wide minimum length by ``t[:min_len]`` on SORTED indices

That last step is a ROW-MAJOR PREFIX, so it discards the highest indices, i.e.
the BOTTOM of the image.  The retina is a roughly horizontal band, so a slice
whose retina sits low can have its remaining anatomy deleted from the context by
truncation alone -- after COVER has already spent its targets hiding ~79% of it.

This renders, per slice, the four stages side by side so the loss is visible
rather than inferred, and picks out the slices where the encoder ends up seeing
ZERO anatomy.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
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
COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear",
              mirage_occupancy_threshold=0.25)


def up(m):
    return np.kron(m.reshape(GRID, GRID), np.ones((PATCH, PATCH), dtype=bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="cover_random",
                    choices=["cover_random", "envelope"])
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\context_loss")
    ap.add_argument("--compare_truncate", action="store_true",
                    help="render prefix vs guard on the SAME slices")
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

    if args.arm == "cover_random":
        cfg = dict(mode="mirage_cover", anatomy_tau=0.10, cover_leave_frac=0.15,
                   cover_min_visible_frac=0.15, cover_min_visible_cells=4,
                   cover_fill="random_legal", **COMMON)
    else:
        cfg = dict(mode="mirage_envelope", mirage_min_block_fill=0.4,
                   mirage_min_retina_visible=0.25, mirage_max_attempts=30,
                   mirage_spread=True, mirage_overlap_tolerance=0.25, **COMMON)
    gen = CurriculumMaskGenerator(**BASE, curriculum_cfg=cfg)
    gen.set_epoch(50, 100)

    if args.compare_truncate:
        cfg_g = dict(cfg); cfg_g["enc_truncate"] = "guard"; cfg_g["enc_guard_cells"] = 6
        gen_g = CurriculumMaskGenerator(**BASE, curriculum_cfg=cfg_g)
        gen_g.set_epoch(50, 100)
        rows = []
        for start in range(0, args.batches * 64, 64):
            items = [ds[i] for i in range(start, start + 64)]
            imgs = torch.stack([it[0] for it in items], 0)
            guides = torch.stack([it[1] for it in items], 0)
            valid = torch.stack([it[2] for it in items], 0)
            B = imgs.size(0)
            anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)
            outs = {}
            for tag, gg in (("prefix", gen), ("guard", gen_g)):
                random.seed(7 + start); np.random.seed(7 + start)
                torch.manual_seed(7 + start)
                outs[tag] = gg.generate(batch_size=B, guide_grids=guides,
                                        guide_valid=valid)
            for b in range(B):
                a = anat[b]
                if a.sum() == 0:
                    continue
                cp = np.zeros(NP, bool); cp[outs["prefix"][0][0][b].numpy()] = True
                cg = np.zeros(NP, bool); cg[outs["guard"][0][0][b].numpy()] = True
                np_, ng = int((a & cp).sum()), int((a & cg).sum())
                if np_ == 0 and ng > 0:
                    rows.append((start + b, imgs[b].numpy(), a, cp, cg, np_, ng))
        print(f"slices rescued by 'guard' (prefix=0 anatomy -> guard>0): {len(rows)}")
        picks = rows[:args.show]
        fig, ax = plt.subplots(len(picks), 3, figsize=(11.5, 3.5 * len(picks)))
        if len(picks) == 1:
            ax = ax[None, :]
        for r, (idx, img, a, cp, cg, n_p, n_g) in enumerate(picks):
            g = img.mean(0); g = (g - g.min()) / (np.ptp(g) + 1e-6)
            A = up(a)
            ax[r, 0].imshow(g, cmap="gray")
            ax[r, 0].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.2)
            ax[r, 0].set_title(f"slice {idx} — anatomy {int(a.sum())} cells", fontsize=9)
            for c, (m, n, tag, col) in enumerate(
                    ((cp, n_p, "BEFORE (prefix crop)", "#d62728"),
                     (cg, n_g, "AFTER (guard crop)", "#2ca02c")), start=1):
                v = np.zeros_like(g); v[up(m)] = g[up(m)]
                ax[r, c].imshow(v, cmap="gray")
                ax[r, c].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
                ax[r, c].set_title(f"{tag} — {n} anatomy cells", fontsize=9,
                                   color=col, fontweight="bold")
            for c in range(3):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        fig.suptitle(
            "Encoder context after the batch crop: BEFORE the crop deletes all "
            "anatomy, AFTER it reserves a few anatomy tokens\n"
            "same slices, same targets, same token count — green outline = anatomy",
            fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{args.arm}_truncation_before_after.png"
        fig.savefig(dst, dpi=130)
        print("wrote", dst)
        return

    worst = []
    stats = []
    for start in range(0, args.batches * 64, 64):
        items = [ds[i] for i in range(start, start + 64)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = imgs.size(0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)
        random.seed(7 + start); np.random.seed(7 + start); torch.manual_seed(7 + start)
        me, mp = gen.generate(batch_size=B, guide_grids=guides, guide_valid=valid)
        for b in range(B):
            a = anat[b]
            if a.sum() == 0:
                continue
            tgt = np.zeros(NP, bool)
            for m in mp:
                tgt[m[b].numpy()] = True
            pre = ~tgt
            post = np.zeros(NP, bool); post[me[0][b].numpy()] = True
            n_pre = int((a & pre).sum()); n_post = int((a & post).sum())
            stats.append((int(a.sum()), n_pre, n_post))
            worst.append((n_post, n_pre, int(a.sum()), start + b,
                          imgs[b].numpy(), a, tgt, pre, post))

    st = np.array(stats)
    frac_pre = 100 * st[:, 1] / st[:, 0]
    frac_post = 100 * st[:, 2] / st[:, 0]
    print(f"{args.arm}: n={len(st)}")
    print(f"  anatomy in context PRE-truncation  {frac_pre.mean():5.1f}%  "
          f"zero: {100*(st[:,1]==0).mean():.1f}%")
    print(f"  anatomy in context POST-truncation {frac_post.mean():5.1f}%  "
          f"zero: {100*(st[:,2]==0).mean():.1f}%")
    print(f"  lost to truncation alone: {frac_pre.mean()-frac_post.mean():.1f} pp")

    worst.sort(key=lambda r: (r[0], -r[1]))
    picks = worst[:args.show]

    fig, ax = plt.subplots(len(picks), 4, figsize=(15, 3.35 * len(picks)))
    if len(picks) == 1:
        ax = ax[None, :]
    for r, (n_post, n_pre, n_a, idx, img, a, tgt, pre, post) in enumerate(picks):
        g = img.mean(0)
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        A = up(a)

        ax[r, 0].imshow(g, cmap="gray")
        ax[r, 0].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.2)
        ax[r, 0].set_title(f"slice {idx} — anatomy {n_a} cells", fontsize=9)

        v = g.copy(); v[up(tgt)] = 0
        ax[r, 1].imshow(v, cmap="gray")
        ax[r, 1].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
        ax[r, 1].set_title(f"after TARGETS — {n_pre} anatomy cells left", fontsize=9)

        v = np.zeros_like(g); v[up(pre)] = g[up(pre)]
        ax[r, 2].imshow(v, cmap="gray")
        ax[r, 2].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
        ax[r, 2].set_title(f"context PRE-crop — {n_pre} anatomy", fontsize=9)

        v = np.zeros_like(g); v[up(post)] = g[up(post)]
        ax[r, 3].imshow(v, cmap="gray")
        ax[r, 3].contour(A, levels=[0.5], colors="#00ff00", linewidths=1.0)
        col = "#d62728" if n_post == 0 else "#000000"
        ax[r, 3].set_title(
            f"context POST-crop — {n_post} anatomy"
            + ("   ← ENCODER SEES NO ANATOMY" if n_post == 0 else ""),
            fontsize=9, color=col,
            fontweight="bold" if n_post == 0 else "normal")
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle(
        f"{args.arm}: the anatomy floor is enforced on TARGETS, but the encoder "
        f"context is cropped afterwards\n"
        f"row-major prefix truncation removes the BOTTOM of the image — "
        f"green outline = anatomy",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.arm}_context_loss.png"
    fig.savefig(dst, dpi=130)
    print("wrote", dst)


if __name__ == "__main__":
    main()
