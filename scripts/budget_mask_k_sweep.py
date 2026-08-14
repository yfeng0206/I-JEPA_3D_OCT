#!/usr/bin/env python3
"""Same slices, same crop, swept across k — so k is the only thing that moves.

budget_mask_visualize.py re-reads the dataset per k, and RandomResizedCrop
draws a new crop each read, so those panels are not comparable across k.
Here each slice is read ONCE and reused for every k.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from budget_mask_prototype import build_budget_targets, GRID, NPATCH  # noqa: E402
from budget_mask_visualize import COLORS, CROP, OCC_T, PATCH, load_slices, to_img, upsample  # noqa: E402


def panel(ax, img, parts, k, hit_note=""):
    ax.imshow(img, cmap="gray")
    ov = np.zeros((CROP, CROP, 4))
    for j, p in enumerate(parts):
        ov[upsample(p) > 0] = matplotlib.colors.to_rgba(COLORS[j % len(COLORS)])
    ov[..., 3] *= 0.55
    ax.imshow(ov)
    union = int(np.logical_or.reduce(parts).sum()) if parts else 0
    ctx = NPATCH - union
    sizes = "/".join(str(int(p.sum())) for p in parts)
    ax.set_title(f"k={k}  hidden {union}/256 ({100*union/NPATCH:.0f}%)\n"
                 f"sizes {sizes}{hit_note}", fontsize=7.5)
    ax.axis("off")
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 30, 40, 57])
    ap.add_argument("--slices", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\budget_masks")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds, idxs = load_slices(args.slices, args.seed, args.slices)
    # Read ONCE; every k below sees this exact crop.
    cache = []
    for i in idxs:
        img_t, guide, _ = ds[i]
        occ = guide[0].numpy()
        cs = ([guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4
              else [occ, np.zeros_like(occ)])
        cache.append((i, to_img(img_t), occ, cs))

    ncol = 1 + len(args.ks)
    fig, axes = plt.subplots(len(cache), ncol,
                             figsize=(2.55 * ncol, 2.9 * len(cache)))
    if len(cache) == 1:
        axes = axes[None, :]

    for r, (i, img, occ, cs) in enumerate(cache):
        ax = axes[r, 0]
        ax.imshow(img)
        ax.imshow(upsample(occ >= OCC_T), cmap="autumn", alpha=0.30)
        ax.set_title(f"slice {i}\nanatomy {int((occ >= OCC_T).sum())}/256",
                     fontsize=7.5)
        ax.axis("off")
        for c, k in enumerate(args.ks):
            rng = random.Random(1234 + r)      # same RNG per row across k
            parts, info = build_budget_targets(cs, n=4, k=k, rng=rng)
            note = "  FALLBACK" if info["fallback"] else ""
            panel(axes[r, c + 1], img, parts, k, note)

    handles = [mpatches.Patch(color=COLORS[j], label=f"target {j+1}")
               for j in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("BUDGET masking swept over k — identical crop per row.  "
                 "k=16 is the current anatomy arm; k~40 is the stock I-JEPA "
                 "block size; k=30 matches the envelope baseline's hidden union.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    out = outdir / "budget_mask_k_sweep.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("wrote", out)

    # --- second figure: what the encoder still sees, same slices -----------
    fig, axes = plt.subplots(len(cache), ncol,
                             figsize=(2.55 * ncol, 2.9 * len(cache)))
    if len(cache) == 1:
        axes = axes[None, :]
    for r, (i, img, occ, cs) in enumerate(cache):
        ax = axes[r, 0]
        ax.imshow(img)
        ax.set_title(f"slice {i}\nfull image", fontsize=7.5)
        ax.axis("off")
        for c, k in enumerate(args.ks):
            rng = random.Random(1234 + r)
            parts, _ = build_budget_targets(cs, n=4, k=k, rng=rng)
            union = np.logical_or.reduce(parts)
            vis = img.copy()
            vis[upsample(union) > 0] = 0.0
            ctx = NPATCH - int(union.sum())
            ax = axes[r, c + 1]
            ax.imshow(vis)
            ax.set_title(f"k={k}  context {ctx}/256 ({100*ctx/NPATCH:.0f}%)",
                         fontsize=7.5)
            ax.axis("off")
    fig.suptitle("What the ENCODER still sees after removing the target union",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    out2 = outdir / "budget_mask_k_sweep_context.png"
    fig.savefig(out2, dpi=125)
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
