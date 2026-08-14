#!/usr/bin/env python3
"""Full five-arm masking audit: distributions, duplicate-slot accounting, Excel.

Answers three questions the mean-only reports could not:

  1. How much anatomy does each arm ACTUALLY hide, post-collation, under both
     the cell definition (occupancy >= 0.25) and the mass definition
     (soft class score > tau)?
  2. Does the encoder still see anatomy -- and how often does it see NONE?
  3. **Duplicate slots.**  ``pred_target_k`` forces every target to exactly K
     indices and pads short ones WITH REPLACEMENT (src/masks/utils.py:
     ``resample_to_k``).  The blob arm runs K=16 on ragged targets, so some of
     its 16 "supervised tokens" are the same cell repeated.  Slot counts
     therefore overstate real supervision for that arm and only that arm.

Writes per-image rows (CSV), a multi-sheet Excel workbook and PNG figures.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NPATCH = GRID * GRID
OCC_T, TAU = 0.25, 0.10

BASE_KW = dict(
    input_size=(CROP, CROP), patch_size=PATCH,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False,
)
COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear",
              mirage_occupancy_threshold=OCC_T)
ARM_ORDER = ["random", "oracle", "envelope", "anatomy", "cover"]


def build_arms():
    arms = {}
    arms["random"] = MaskCollator(**BASE_KW)
    arms["oracle"] = CurriculumMaskGenerator(
        **BASE_KW, curriculum_cfg=dict(mode="anatomical_prior", **COMMON))
    arms["envelope"] = CurriculumMaskGenerator(**BASE_KW, curriculum_cfg=dict(
        mode="mirage_envelope", mirage_min_block_fill=0.4,
        mirage_min_retina_visible=0.25, mirage_max_attempts=30,
        mirage_spread=True, mirage_overlap_tolerance=0.25, **COMMON))
    arms["anatomy"] = CurriculumMaskGenerator(
        **BASE_KW, pred_target_k=16, curriculum_cfg=dict(
            mode="mirage_anatomy", anatomy_mass_cap=0.9, anatomy_tau=TAU,
            anatomy_bridge_diagonals=True, **COMMON))
    arms["cover"] = CurriculumMaskGenerator(**BASE_KW, curriculum_cfg=dict(
        mode="mirage_cover", anatomy_tau=TAU, cover_leave_frac=0.15,
        cover_min_visible_frac=0.15, cover_min_visible_cells=4,
        cover_transition=True, **COMMON))
    for a in arms.values():
        if hasattr(a, "set_epoch"):
            a.set_epoch(50, 100)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--volumes", type=int, default=20)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\five_arm_audit")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

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
    n_vol = len(ds.file_paths)
    vols = sorted(random.sample(range(n_vol), min(args.volumes, n_vol)))
    idxs = [v * args.num_slices + s for v in vols for s in range(args.num_slices)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    arms = build_arms()
    rows = []

    for start in range(0, len(idxs), args.batch_size):
        items = [ds[i] for i in idxs[start:start + args.batch_size]]
        images = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = images.size(0)

        occ = guides[:, 0].reshape(B, -1).numpy()
        soft = ((guides[:, 2] + guides[:, 3]).reshape(B, -1).numpy()
                if guides.shape[1] >= 4 else occ.copy())
        cell_anat = occ >= OCC_T
        mass_val = np.where(soft > TAU, soft, 0.0)

        for name in ARM_ORDER:
            arm = arms[name]
            random.seed(args.seed + start)
            np.random.seed(args.seed + start)
            torch.manual_seed(args.seed + start)
            if name == "random":
                _, m_enc, m_pred = arm([images[i] for i in range(B)])
            elif name == "oracle":
                m_enc, m_pred = arm.generate(batch_size=B, imgs_cpu=images)
            else:
                m_enc, m_pred = arm.generate(batch_size=B, guide_grids=guides,
                                             guide_valid=valid)
            for b in range(B):
                slots = uniq = 0
                tgt = np.zeros(NPATCH, bool)
                for g in m_pred:
                    idx = g[b].numpy()
                    slots += idx.size
                    uniq += np.unique(idx).size
                    tgt[idx] = True
                ctx = np.zeros(NPATCH, bool)
                ctx[m_enc[0][b].numpy()] = True

                nc = int(cell_anat[b].sum())
                tm = float(mass_val[b].sum())
                ctx_anat = int((cell_anat[b] & ctx).sum())
                rows.append(dict(
                    arm=name, gidx=int(idxs[start + b]),
                    slots_per_img=slots, uniq_per_img=uniq,
                    dup_slots=slots - uniq,
                    dup_pct=100.0 * (slots - uniq) / slots if slots else 0.0,
                    slots_per_mask=slots / len(m_pred),
                    uniq_per_mask=uniq / len(m_pred),
                    union=int(tgt.sum()), ctx_tokens=int(ctx.sum()),
                    anat_cells=nc, anat_mass=tm,
                    hidden_cells_pct=100.0 * (cell_anat[b] & tgt).sum() / nc if nc else np.nan,
                    hidden_mass_pct=100.0 * mass_val[b][tgt].sum() / tm if tm > 0 else np.nan,
                    ctx_cells_pct=100.0 * ctx_anat / nc if nc else np.nan,
                    ctx_mass_pct=100.0 * mass_val[b][ctx].sum() / tm if tm > 0 else np.nan,
                    ctx_anat_tokens=ctx_anat,
                    zero_anat_ctx=int(ctx_anat == 0),
                ))
        if (start + B) % 320 == 0:
            print(f"  {start + B}/{len(idxs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_rows.csv", index=False)

    agg = df.groupby("arm").agg(
        n=("gidx", "count"),
        slots_per_mask=("slots_per_mask", "mean"),
        uniq_per_mask=("uniq_per_mask", "mean"),
        dup_slots_per_img=("dup_slots", "mean"),
        dup_pct=("dup_pct", "mean"),
        union=("union", "mean"),
        ctx_tokens=("ctx_tokens", "mean"),
        hidden_cells_pct=("hidden_cells_pct", "mean"),
        hidden_mass_pct=("hidden_mass_pct", "mean"),
        ctx_cells_pct=("ctx_cells_pct", "mean"),
        ctx_anat_tokens=("ctx_anat_tokens", "mean"),
        zero_anat_ctx_pct=("zero_anat_ctx", lambda s: 100.0 * s.mean()),
        anat_cells=("anat_cells", "mean"),
    ).reindex(ARM_ORDER).reset_index()

    pcts = df.groupby("arm")[["hidden_cells_pct", "hidden_mass_pct",
                              "ctx_cells_pct", "ctx_anat_tokens",
                              "uniq_per_mask"]].describe(
        percentiles=[.05, .25, .5, .75, .95]).reindex(ARM_ORDER)

    with pd.ExcelWriter(out / "five_arm_audit.xlsx", engine="openpyxl") as xw:
        agg.to_excel(xw, sheet_name="summary", index=False)
        pcts.to_excel(xw, sheet_name="distributions")
        for name in ARM_ORDER:
            sub = df[df.arm == name].drop(columns=["arm"])
            sub.to_excel(xw, sheet_name=f"raw_{name}"[:31], index=False)
    print("\nwrote", out / "five_arm_audit.xlsx")

    # ---------------- figures ----------------
    colors = dict(random="#888888", oracle="#f58231", envelope="#4363d8",
                  anatomy="#e6194b", cover="#3cb44b")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for name in ARM_ORDER:
        s = df[df.arm == name]
        ax[0, 0].hist(s.hidden_mass_pct.dropna(), bins=40, histtype="step",
                      lw=2, label=name, color=colors[name], density=True)
        ax[0, 1].hist(s.ctx_cells_pct.dropna(), bins=40, histtype="step",
                      lw=2, label=name, color=colors[name], density=True)
        ax[1, 0].hist(s.uniq_per_mask, bins=40, histtype="step", lw=2,
                      label=name, color=colors[name], density=True)
    ax[0, 0].set_title("Anatomy MASS hidden by targets (%)"); ax[0, 0].legend()
    ax[0, 1].set_title("Anatomy left in encoder context (% of all anatomy)")
    ax[0, 1].legend()
    ax[1, 0].set_title("UNIQUE cells supervised per target (duplicates removed)")
    ax[1, 0].legend()
    z = agg.set_index("arm")["zero_anat_ctx_pct"]
    ax[1, 1].bar(z.index, z.values, color=[colors[a] for a in z.index])
    ax[1, 1].set_title("Slices where the encoder sees ZERO anatomy (%)")
    for i, v in enumerate(z.values):
        ax[1, 1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=130)

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(ARM_ORDER)); w = 0.38
    a = agg.set_index("arm").reindex(ARM_ORDER)
    ax2[0].bar(x - w/2, a.slots_per_mask, w, label="slots/mask (counted)",
               color="#bbbbbb", edgecolor="k")
    ax2[0].bar(x + w/2, a.uniq_per_mask, w, label="UNIQUE cells/mask (real)",
               color="#3cb44b", edgecolor="k")
    ax2[0].set_xticks(x); ax2[0].set_xticklabels(ARM_ORDER)
    ax2[0].set_title("Supervision per target: counted vs real"); ax2[0].legend()
    for i, (s_, u_) in enumerate(zip(a.slots_per_mask, a.uniq_per_mask)):
        if s_ - u_ > 0.05:
            ax2[0].text(i, s_ + .4, f"-{100*(s_-u_)/s_:.0f}%", ha="center",
                        color="crimson", fontweight="bold")
    ax2[1].bar(x - w/2, a.hidden_mass_pct, w, label="anatomy hidden (mass %)",
               color="#4363d8", edgecolor="k")
    ax2[1].bar(x + w/2, a.ctx_cells_pct, w, label="anatomy in context (%)",
               color="#f58231", edgecolor="k")
    ax2[1].set_xticks(x); ax2[1].set_xticklabels(ARM_ORDER)
    ax2[1].set_title("Anatomy coverage vs context retained"); ax2[1].legend()
    fig2.tight_layout()
    fig2.savefig(out / "supervision_and_coverage.png", dpi=130)

    pd.set_option("display.width", 200)
    print("\n", agg.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    (out / "summary.json").write_text(json.dumps(
        agg.to_dict(orient="records"), indent=2))
    print("\nfigures ->", out / "distributions.png", "|",
          out / "supervision_and_coverage.png")


if __name__ == "__main__":
    main()
