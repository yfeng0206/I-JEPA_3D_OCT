"""What each arm actually covers: anatomy vs background, as distributions.

Summary means hide the interesting part.  Two arms can hide the same average
fraction of anatomy while one does it consistently and the other swings between
total occlusion and nearly none, and that difference decides whether the
predictor ever sees a usable anatomy context.  This measures, per image and per
arm, four quantities and plots their full distributions:

  * fraction of the ANATOMY that the target union hides,
  * fraction of the BACKGROUND that the target union hides,
  * fraction of the predicted SLOTS that sit on background -- which, because the
    loss is ``reduction='mean'`` and therefore content-blind, is literally the
    share of the gradient budget spent predicting background,
  * fraction of the anatomy still visible to the encoder as context.

The oracle arm is driven with ``imgs_cpu`` rather than guides: its
``anatomical_prior`` band comes from the row-intensity profile, and passing only
``guide_grids`` silently leaves ``bias_active=False`` so it degrades to uniform
random without raising.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.masks.multiblock import MaskCollator                     # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NPATCH = GRID * GRID
OCC_T, TAU = 0.25, 0.30

BASE = dict(input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
            allow_overlap=False)
COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear",
              mirage_occupancy_threshold=OCC_T)

ORDER = ["random", "oracle", "envelope", "blob", "cover_transition", "cover_random"]
COLORS = {"random": "#4363d8", "oracle": "#e6194b", "envelope": "#f58231",
          "blob": "#3cb44b", "cover_transition": "#911eb4",
          "cover_random": "#000000"}


def build_arms():
    a = {"random": MaskCollator(**BASE)}
    a["oracle"] = CurriculumMaskGenerator(
        **BASE, curriculum_cfg=dict(mode="anatomical_prior", **COMMON))
    a["envelope"] = CurriculumMaskGenerator(**BASE, curriculum_cfg=dict(
        mode="mirage_envelope", mirage_min_block_fill=0.4,
        mirage_min_retina_visible=0.25, mirage_max_attempts=30,
        mirage_spread=True, mirage_overlap_tolerance=0.25, **COMMON))
    a["blob"] = CurriculumMaskGenerator(
        **BASE, pred_target_k=16, curriculum_cfg=dict(
            mode="mirage_anatomy", anatomy_mass_cap=0.9, anatomy_tau=TAU,
            anatomy_bridge_diagonals=True, **COMMON))
    for tag, fill in (("cover_transition", "transition"),
                      ("cover_random", "random_legal")):
        a[tag] = CurriculumMaskGenerator(**BASE, curriculum_cfg=dict(
            mode="mirage_cover", anatomy_tau=0.10, cover_leave_frac=0.15,
            cover_min_visible_frac=0.15, cover_min_visible_cells=4,
            cover_fill=fill, **COMMON))
    for g in a.values():
        if hasattr(g, "set_epoch"):
            g.set_epoch(50, 100)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--volumes", type=int, default=40)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\arm_coverage")
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

    rng0 = random.Random(args.seed)
    vols = sorted(rng0.sample(range(len(ds.file_paths)),
                              min(args.volumes, len(ds.file_paths))))
    step = max(1, args.num_slices // args.slices_per_volume)
    idxs = [v * args.num_slices + s
            for v in vols for s in range(0, args.num_slices, step)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    arms = build_arms()
    rows = []
    for start in range(0, len(idxs), args.batch_size):
        items = [ds[i] for i in idxs[start:start + args.batch_size]]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = imgs.size(0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= OCC_T)

        for name in ORDER:
            arm = arms[name]
            random.seed(args.seed + start)
            np.random.seed(args.seed + start)
            torch.manual_seed(args.seed + start)
            if name == "random":
                _, m_enc, m_pred = arm([im for im in imgs])
            elif name == "oracle":
                m_enc, m_pred = arm.generate(batch_size=B, imgs_cpu=imgs)
            else:
                m_enc, m_pred = arm.generate(batch_size=B, guide_grids=guides,
                                             guide_valid=valid)
            for b in range(B):
                a = anat[b]
                if a.sum() == 0:
                    continue
                slots = np.concatenate([m[b].numpy() for m in m_pred])
                u = np.zeros(NPATCH, bool)
                u[slots] = True
                ctx = m_enc[0][b].numpy()
                bg = ~a
                rows.append(dict(
                    arm=name, image=start + b,
                    anat_hidden=100 * (a & u).sum() / a.sum(),
                    bg_hidden=100 * (bg & u).sum() / max(bg.sum(), 1),
                    slots_bg=100 * (~a[slots]).mean(),
                    ctx_anat=100 * (a[ctx]).sum() / a.sum(),
                    union=int(u.sum()), n_slots=int(slots.size)))
        if start % (args.batch_size * 10) == 0:
            print(f"  {start + B}/{len(idxs)}", flush=True)

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)
    agg = df.groupby("arm").agg(
        anat_hidden=("anat_hidden", "mean"), anat_hidden_sd=("anat_hidden", "std"),
        bg_hidden=("bg_hidden", "mean"), bg_hidden_sd=("bg_hidden", "std"),
        slots_bg=("slots_bg", "mean"), ctx_anat=("ctx_anat", "mean"),
        union=("union", "mean"), n_slots=("n_slots", "mean"),
    ).reindex(ORDER).reset_index()
    pd.set_option("display.width", 250)
    print("\n=== coverage by arm (%) ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    agg.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(agg.to_dict("records"), indent=2))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    panels = [
        ("anat_hidden", "% of ANATOMY hidden by the targets",
         "higher = predictor must reconstruct more retina"),
        ("bg_hidden", "% of BACKGROUND hidden by the targets",
         "higher = more of the loss spent on empty space"),
        ("slots_bg", "% of predicted SLOTS on background",
         "the loss is content-blind, so this IS the gradient budget"),
        ("ctx_anat", "% of ANATOMY left visible as context",
         "what the encoder still gets to see"),
    ]
    for k, (col, title, sub) in enumerate(panels):
        a_ = ax[k // 3, k % 3]
        for name in ORDER:
            s = df[df.arm == name][col]
            a_.hist(s, bins=45, histtype="step", lw=2.1, density=True,
                    label=f"{name} ({s.mean():.1f}%)", color=COLORS[name])
        a_.set_title(title, fontsize=11, fontweight="bold")
        a_.set_xlabel(sub, fontsize=9)
        a_.legend(fontsize=8)
        a_.grid(alpha=.3)

    # anatomy vs background coverage, side by side
    a_ = ax[1, 1]
    x = np.arange(len(ORDER)); w = 0.38
    a_.bar(x - w / 2, agg.anat_hidden, w, yerr=agg.anat_hidden_sd, capsize=3,
           label="anatomy hidden", color="#e6194b", edgecolor="k")
    a_.bar(x + w / 2, agg.bg_hidden, w, yerr=agg.bg_hidden_sd, capsize=3,
           label="background hidden", color="#4363d8", edgecolor="k")
    for xi, v in zip(x - w / 2, agg.anat_hidden):
        a_.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, agg.bg_hidden):
        a_.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    a_.set_xticks(x); a_.set_xticklabels(ORDER, rotation=20, fontsize=8)
    a_.set_ylabel("% hidden"); a_.set_ylim(0, 105)
    a_.set_title("What each arm covers", fontsize=11, fontweight="bold")
    a_.legend(fontsize=8); a_.grid(alpha=.3, axis="y")

    # anatomy-hidden vs background-slot share
    a_ = ax[1, 2]
    for name in ORDER:
        r = agg[agg.arm == name].iloc[0]
        a_.scatter(r.slots_bg, r.anat_hidden, s=210, color=COLORS[name],
                   edgecolor="k", zorder=3)
        a_.annotate(name, (r.slots_bg, r.anat_hidden), fontsize=9,
                    textcoords="offset points", xytext=(9, 5))
    a_.set_xlabel("% of predicted slots on background\n(gradient budget spent on background)",
                  fontsize=9)
    a_.set_ylabel("% of anatomy hidden", fontsize=9)
    a_.set_title("The two axes that separate the arms", fontsize=11,
                 fontweight="bold")
    a_.grid(alpha=.3)

    fig.suptitle("What each masking arm actually covers — same slices, same block sizes, "
                 "full ramp (r_t = 1.0)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "arm_coverage.png", dpi=135)
    print(f"\nwrote {out / 'arm_coverage.png'}")


if __name__ == "__main__":
    main()
