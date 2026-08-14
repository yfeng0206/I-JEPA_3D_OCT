"""How much of each arm's gradient budget is spent predicting background?

``train_patch.py:698`` computes ``F.smooth_l1_loss(z, h_rep)`` with the default
``reduction='mean'``.  That average is taken over every element of the target
tensor, so **every predicted slot carries identical weight regardless of what is
underneath it**.  The loss is content-blind.

The consequence is arithmetic, needs no encoder, and is the L0 half of the
background question: an arm whose targets are 65% background spends 65% of its
per-slot gradient budget on background, and an arm whose targets are 3%
background spends almost none.  This script measures that split exactly, on the
real mask distributions, including the duplicate slots that ``pred_target_k``
introduces (``src/masks/utils.py::resample_to_k`` pads short targets WITH
REPLACEMENT, which only the blob arm triggers).

It also records the same split for the CONTEXT set, because the encoder's input
composition is the other half of the story.

CPU only -- no model is loaded.
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
    arms = {"random": MaskCollator(**BASE_KW)}
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
    ap.add_argument("--slices_per_volume", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\target_composition")
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

    arms = build_arms()
    rows = []
    for start in range(0, len(idxs), args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        items = [ds[i] for i in chunk]
        images = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = images.size(0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= OCC_T)

        for name in ARM_ORDER:
            arm = arms[name]
            random.seed(args.seed + start); torch.manual_seed(args.seed + start)
            if name == "random":
                _, m_enc, m_pred = arm([im for im in images])
            elif name == "oracle":
                # anatomical_prior derives its band from the row-intensity
                # profile, so it needs the IMAGE.  Passing only guide_grids
                # leaves bias_active False and silently degrades to uniform.
                m_enc, m_pred = arm.generate(batch_size=B, imgs_cpu=images)
            else:
                m_enc, m_pred = arm.generate(batch_size=B, guide_grids=guides,
                                             guide_valid=valid)
            for b in range(B):
                slots = np.concatenate([g[b].numpy() for g in m_pred])
                uniq = np.unique(slots)
                ctx = m_enc[0][b].numpy()
                a = anat[b]
                rows.append(dict(
                    arm=name, image=start + b,
                    anat_cells=int(a.sum()),
                    n_slots=int(slots.size),
                    n_unique=int(uniq.size),
                    slots_bg_frac=float((~a[slots]).mean()),
                    unique_bg_frac=float((~a[uniq]).mean()),
                    ctx_tokens=int(ctx.size),
                    ctx_bg_frac=float((~a[ctx]).mean()),
                ))
        if start % (args.batch_size * 8) == 0:
            print(f"  {start + B}/{len(idxs)}", flush=True)

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)

    agg = df.groupby("arm").agg(
        anat_cells=("anat_cells", "mean"),
        slots=("n_slots", "mean"), unique=("n_unique", "mean"),
        slots_bg_pct=("slots_bg_frac", lambda s: 100 * s.mean()),
        unique_bg_pct=("unique_bg_frac", lambda s: 100 * s.mean()),
        ctx_tokens=("ctx_tokens", "mean"),
        ctx_bg_pct=("ctx_bg_frac", lambda s: 100 * s.mean()),
    ).reindex(ARM_ORDER).reset_index()
    agg["dup_pct"] = 100 * (1 - agg["unique"] / agg["slots"])

    print("\n=== gradient budget spent on BACKGROUND (loss is content-blind) ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    agg.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        agg.to_dict(orient="records"), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
