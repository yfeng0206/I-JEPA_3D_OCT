#!/usr/bin/env python3
"""Render BUDGET masking on real FairVision B-scans, and audit its failures.

Produces
  budget_mask_samples_k{K}.png   grid of random slices: image, anatomy guide,
                                 the 4 target blocks, and the surviving context
  budget_mask_audit_fairvision.json   failure rates per k over many slices
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from budget_mask_prototype import build_budget_targets, GRID, NPATCH  # noqa: E402
from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

CROP, PATCH = 256, 16
OCC_T = 0.25
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]   # 4 target blocks


def load_slices(n_slices, seed, volumes,
                data_dir=r"D:\jepa_phase0\fairvision-glaucoma\data",
                guide_dir=(r"C:\jepa_data\mirage_soft_guides"
                           r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"),
                cache=r"C:\jepa_data\slice_cache", split="Training"):
    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    sc = os.path.join(cache, split)
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(data_dir, split),
        guide_dir=os.path.join(guide_dir, split),
        num_slices=100, slice_size=CROP, transform=paired, patch_size=PATCH,
        dilate_patches=0, occupancy_threshold=OCC_T,
        slice_cache=sc if os.path.isdir(sc) else None)
    rng = random.Random(seed)
    n_vol = len(ds.file_paths)
    vols = rng.sample(range(n_vol), min(volumes, n_vol))
    idxs = [v * 100 + rng.randrange(100) for v in vols][:n_slices]
    return ds, idxs


def to_img(t):
    a = t.permute(1, 2, 0).numpy()
    a = (a - a.min()) / max(a.max() - a.min(), 1e-8)
    return a


def upsample(mask_grid):
    return np.kron(mask_grid.astype(float), np.ones((PATCH, PATCH)))


def render(ds, idxs, k, out_png, n_pred=4, tau=0.10, seed=0):
    rng = random.Random(seed)
    rows = len(idxs)
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.4 * rows))
    if rows == 1:
        axes = axes[None, :]

    for r, i in enumerate(idxs):
        img_t, guide, valid = ds[i]
        img = to_img(img_t)
        occ = guide[0].numpy()
        cs = [guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4 \
            else [occ, np.zeros_like(occ)]

        parts, info = build_budget_targets(cs, n=n_pred, k=k, tau=tau, rng=rng)
        union = np.logical_or.reduce(parts) if parts else np.zeros((GRID, GRID), bool)

        # 1: image + anatomy guide outline
        ax = axes[r, 0]
        ax.imshow(img)
        ax.imshow(upsample(occ >= OCC_T), cmap="autumn", alpha=0.28)
        ax.set_title(f"slice {i} · anatomy {int((occ>=OCC_T).sum())}/256",
                     fontsize=8)
        ax.axis("off")

        # 2: the 4 target blocks
        ax = axes[r, 1]
        ax.imshow(img, cmap="gray")
        overlay = np.zeros((CROP, CROP, 4))
        for j, p in enumerate(parts):
            c = matplotlib.colors.to_rgba(COLORS[j % len(COLORS)])
            m = upsample(p) > 0
            overlay[m] = c
        overlay[..., 3] *= 0.55
        ax.imshow(overlay)
        sizes = "/".join(str(int(p.sum())) for p in parts)
        ax.set_title(f"4 targets k={k} · sizes {sizes}", fontsize=8)
        ax.axis("off")

        # 3: what the encoder still sees
        ax = axes[r, 2]
        vis = img.copy()
        hide = upsample(union) > 0
        vis[hide] = 0.0
        ax.imshow(vis)
        ctx = NPATCH - int(union.sum())
        ax.set_title(f"context left {ctx}/256 ({100*ctx/NPATCH:.0f}%)"
                     + ("  FALLBACK" if info["fallback"] else ""), fontsize=8)
        ax.axis("off")

    handles = [mpatches.Patch(color=COLORS[j], label=f"target {j+1}")
               for j in range(n_pred)]
    fig.legend(handles=handles, loc="lower center", ncol=n_pred, fontsize=8)
    fig.suptitle(f"BUDGET masking — hide all anatomy, grow into background "
                 f"(k={k} cells/block, {n_pred} blocks)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=115)
    plt.close(fig)
    print("wrote", out_png)


def audit(ds, idxs, ks, n_pred=4, tau=0.10, seed=42):
    out = {}
    for k in ks:
        rng = random.Random(seed)
        rows = []
        for i in idxs:
            _, guide, _ = ds[i]
            occ = guide[0].numpy()
            cs = [guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4 \
                else [occ, np.zeros_like(occ)]
            parts, info = build_budget_targets(cs, n=n_pred, k=k, tau=tau, rng=rng)
            sizes = [int(p.sum()) for p in parts]
            info["sizes"] = sizes
            info["hit_budget"] = all(s == k for s in sizes)
            info["deficit"] = int(sum(max(0, k - s) for s in sizes))
            rows.append(info)
        N = len(rows)
        union = np.array([r["union"] for r in rows], float)
        out[str(k)] = dict(
            k=k, slices=N,
            pct_fallback=100 * sum(r["fallback"] for r in rows) / N,
            pct_seeded_bg=100 * sum("seeded_bg" in r["reason"] for r in rows) / N,
            pct_short=100 * sum(r["n_short"] > 0 for r in rows) / N,
            pct_hit_budget=100 * sum(r["hit_budget"] for r in rows) / N,
            pct_not_4conn=100 * sum(not r["conn_ok"] for r in rows) / N,
            mean_deficit_cells=float(np.mean([r["deficit"] for r in rows])),
            union_mean=float(union.mean()),
            union_frac=float(union.mean() / NPATCH),
            context_mean=float(NPATCH - union.mean()),
            context_frac=float((NPATCH - union.mean()) / NPATCH),
            anat_cells_mean=float(np.mean([r["anat_cells"] for r in rows])),
            grown_into_bg_mean=float(np.mean([r["grew_into_bg"] for r in rows])),
            target_union=k * n_pred,
        )
        r = out[str(k)]
        print(f"k={k:3d} | union {r['union_mean']:6.1f}/{r['target_union']:3d} "
              f"({r['union_frac']*100:4.1f}%) | ctx {r['context_mean']:6.1f} "
              f"({r['context_frac']*100:4.1f}%) | hit {r['pct_hit_budget']:5.1f}% "
              f"| short {r['pct_short']:5.1f}% | deficit {r['mean_deficit_cells']:4.1f} "
              f"| fallback {r['pct_fallback']:4.1f}% | not4conn {r['pct_not_4conn']:4.1f}%",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[16, 30, 40, 50, 57])
    ap.add_argument("--viz_k", type=int, nargs="+", default=[30, 40, 57])
    ap.add_argument("--viz_slices", type=int, default=6)
    ap.add_argument("--audit_slices", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\budget_masks")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("loading audit slices ...", flush=True)
    ds, aidx = load_slices(args.audit_slices, args.seed, args.audit_slices)
    print(f"audit on {len(aidx)} slices from {len(aidx)} volumes", flush=True)
    res = audit(ds, aidx, args.k, seed=args.seed)
    (outdir / "budget_mask_audit_fairvision.json").write_text(
        json.dumps(res, indent=2))
    print("wrote", outdir / "budget_mask_audit_fairvision.json")

    _, vidx = load_slices(args.viz_slices, args.seed + 7, args.viz_slices)
    for k in args.viz_k:
        render(ds, vidx, k, str(outdir / f"budget_mask_samples_k{k}.png"),
               seed=args.seed)


if __name__ == "__main__":
    main()
