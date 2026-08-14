#!/usr/bin/env python3
"""Visualise + audit COVER masking on real FairVision B-scans.

Columns per row (identical crop across all columns):
    image + anatomy guide
    current arm            irregular blobs, pred_target_k = 16
    COVER                  4 stock rectangles greedily placed over the anatomy
    context after COVER    what the encoder still sees
"""
from __future__ import annotations

import argparse
import json
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
from budget_mask_visualize import COLORS, CROP, OCC_T, load_slices, to_img, upsample  # noqa: E402
from cover_mask_prototype import cover_targets  # noqa: E402


def draw_masks(ax, img, masks, title):
    ax.imshow(img, cmap="gray")
    ov = np.zeros((CROP, CROP, 4))
    for j, m in enumerate(masks):
        ov[upsample(m) > 0] = matplotlib.colors.to_rgba(COLORS[j % len(COLORS)])
    ov[..., 3] *= 0.55
    ax.imshow(ov)
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--audit_slices", type=int, default=600)
    ap.add_argument("--leave", type=float, default=0.15)
    ap.add_argument("--min_visible", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\budget_masks")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- audit on FairVision guides ---------------------------
    print("audit ...", flush=True)
    ds_a, aidx = load_slices(args.audit_slices, 42, args.audit_slices)
    res = {}
    for lf in (0.10, 0.15, 0.20):
        rng = random.Random(42)
        gen = torch.Generator(); gen.manual_seed(42)
        rows = []
        for i in aidx:
            _, guide, _ = ds_a[i]
            occ = guide[0].numpy()
            cs = ([guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4
                  else [occ, np.zeros_like(occ)])
            _, info = cover_targets(cs, n=4, leave_frac=lf,
                                    min_visible_frac=lf, gen=gen, rng=rng)
            rows.append(info)
        cf = np.array([r["covered_frac"] for r in rows])
        vf = np.array([r["visible_frac"] for r in rows])
        un = np.array([r["union"] for r in rows], float)
        sl = np.array([r["slots"] for r in rows], float)
        res[str(lf)] = dict(
            leave_frac=lf, min_visible_frac=lf, slices=len(rows),
            pct_hit=100 * float(np.mean([r["hit_target"] for r in rows])),
            pct_floor_ok=100 * float(np.mean([r["floor_ok"] for r in rows])),
            pct_zero_visible=100 * float(np.mean(vf <= 1e-9)),
            pct_fallback=100 * float(np.mean([r["fallback"] for r in rows])),
            anatomy_hidden_mean=float(cf.mean()),
            anatomy_hidden_max=float(cf.max()),
            visible_frac_mean=float(vf.mean()),
            visible_frac_min=float(vf.min()),
            visible_cells_mean=float(np.mean([r["visible_cells"] for r in rows])),
            visible_cells_min=float(np.min([r["visible_cells"] for r in rows])),
            union_mean=float(un.mean()), union_frac=float(un.mean() / NPATCH),
            slots_mean=float(sl.mean()), overlap_mean=float((sl - un).mean()),
            n_cover=float(np.mean([r["n_cover"] for r in rows])),
            n_transition=float(np.mean([r["n_transition"] for r in rows])),
            n_random=float(np.mean([r["n_random"] for r in rows])),
            anat_cells_mean=float(np.mean([r["anat_cells"] for r in rows])),
        )
        r = res[str(lf)]
        print(f"floor={lf:.2f} | hidden {r['anatomy_hidden_mean']*100:5.1f}% "
              f"(max {r['anatomy_hidden_max']*100:5.1f}%) | visible "
              f"{r['visible_frac_mean']*100:5.1f}% (min {r['visible_frac_min']*100:4.1f}%, "
              f"{r['visible_cells_mean']:4.1f} cells, min {r['visible_cells_min']:.0f}) "
              f"| zero-visible {r['pct_zero_visible']:4.1f}% "
              f"| union {r['union_mean']:6.1f} ({r['union_frac']*100:4.1f}%) "
              f"| fallback {r['pct_fallback']:4.1f}% "
              f"| c/t/r {r['n_cover']:.2f}/{r['n_transition']:.2f}/{r['n_random']:.2f}",
              flush=True)
    (outdir / "cover_mask_audit_fairvision.json").write_text(json.dumps(res, indent=2))
    print("wrote", outdir / "cover_mask_audit_fairvision.json")

    # ---------------- visuals ----------------------------------------------
    ds, idxs = load_slices(args.slices, args.seed, args.slices)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(11.0, 2.85 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    for r, i in enumerate(idxs):
        img_t, guide, _ = ds[i]
        img = to_img(img_t)
        occ = guide[0].numpy()
        cs = ([guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4
              else [occ, np.zeros_like(occ)])
        anat = occ >= OCC_T
        n_anat = int(anat.sum())

        ax = axes[r, 0]
        ax.imshow(img)
        ax.imshow(upsample(anat), cmap="autumn", alpha=0.30)
        ax.set_title(f"slice {i} · anatomy {n_anat}/256", fontsize=7.5)
        ax.axis("off")

        rng = random.Random(1000 + r)
        blobs, _ = build_budget_targets(cs, n=4, k=16, rng=rng)
        bu = np.logical_or.reduce(blobs)
        hid_b = int((anat & bu).sum())
        draw_masks(axes[r, 1], img, blobs,
                   f"CURRENT blobs k=16\nhidden {int(bu.sum())}/256 · "
                   f"anatomy hidden {100*hid_b/max(n_anat,1):.0f}%")

        rng = random.Random(1000 + r)
        gen = torch.Generator(); gen.manual_seed(1000 + r)
        rects, info = cover_targets(cs, n=4, leave_frac=args.leave,
                                    min_visible_frac=args.min_visible,
                                    gen=gen, rng=rng)
        ru = np.logical_or.reduce(rects)
        hid_r = int((anat & ru).sum())
        vis_r = n_anat - hid_r
        draw_masks(axes[r, 2], img, rects,
                   f"COVER rectangles\nhidden {int(ru.sum())}/256 · "
                   f"anatomy hidden {100*hid_r/max(n_anat,1):.0f}% "
                   f"({vis_r} cells still visible)")

        ax = axes[r, 3]
        vis = img.copy()
        vis[upsample(ru) > 0] = 0.0
        ax.imshow(vis)
        # mark the anatomy that survives into the context
        keep = anat & ~ru
        ax.imshow(upsample(keep), cmap="cool", alpha=0.45 * (upsample(keep) > 0))
        ctx = NPATCH - int(ru.sum())
        ax.set_title(f"context after COVER · {ctx}/256\n"
                     f"cyan = {vis_r} anatomy cells kept visible", fontsize=7.5)
        ax.axis("off")

    handles = [mpatches.Patch(color=COLORS[j], label=f"block {j+1}")
               for j in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("COVER masking — stock I-JEPA rectangles (4 blocks, "
                 "pred_mask_scale 0.15-0.2, overlap allowed) placed greedily "
                 f"to hide {int((1-args.leave)*100)}% of the anatomy",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    out = outdir / "cover_mask_samples.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
