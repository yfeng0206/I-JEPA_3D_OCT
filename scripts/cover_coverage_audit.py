#!/usr/bin/env python3
"""Paired anatomy-coverage audit: does COVER actually hide more than envelope?

The composition probe reported COVER 79.3% vs envelope 78.0% "hidden share of
all anatomy", i.e. a contrast of only +1.3 points -- which would mean COVER
barely moves the variable it exists to move.  Two things could produce that
reading, and they have opposite implications:

  1. It is real.  Envelope already covers most of the retina, so there is very
     little headroom and COVER is not worth running.
  2. It is a definitional artefact.  The probe scores anatomy as
     ``occupancy >= 0.25`` CELLS, whereas COVER optimises soft class-score
     MASS above ``tau=0.10``.  A thin band scored by cells saturates long
     before it saturates by mass.

This script settles it by scoring BOTH arms with BOTH definitions on the SAME
slices and the SAME batch-shared block sizes (identical seed per arm), and by
measuring the encoder context as actually returned -- which is the only thing
that answers "is anatomy still visible to the model".
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
OCC_T = 0.25
TAU = 0.10

BASE_KW = dict(
    input_size=(CROP, CROP), patch_size=PATCH,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False,
)
COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear",
              mirage_occupancy_threshold=OCC_T)


def build_arms():
    arms = {}
    arms["random"] = MaskCollator(**BASE_KW)
    arms["envelope"] = CurriculumMaskGenerator(**BASE_KW, curriculum_cfg=dict(
        mode="mirage_envelope", mirage_min_block_fill=0.4,
        mirage_min_retina_visible=0.25, mirage_max_attempts=30,
        mirage_spread=True, mirage_overlap_tolerance=0.25, **COMMON))
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
    ap.add_argument("--volumes", type=int, default=10)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\cover_coverage_audit.json")
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
    n_vol = len(ds.file_paths)
    vols = sorted(random.sample(range(n_vol), min(args.volumes, n_vol)))
    idxs = [v * args.num_slices + s for v in vols for s in range(args.num_slices)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    arms = build_arms()
    acc = {k: [] for k in arms}

    for start in range(0, len(idxs), args.batch_size):
        items = [ds[i] for i in idxs[start:start + args.batch_size]]
        images = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        B = images.size(0)

        occ = guides[:, 0].reshape(B, -1).numpy()                 # occupancy
        if guides.shape[1] >= 4:
            soft = (guides[:, 2] + guides[:, 3]).reshape(B, -1).numpy()
        else:
            soft = occ.copy()
        cell_anat = occ >= OCC_T          # composition-probe definition
        mass_sup = soft > TAU             # COVER definition
        mass_val = np.where(mass_sup, soft, 0.0)

        for name, arm in arms.items():
            # Identical seed per arm => identical batch-shared block sizes.
            random.seed(args.seed + start)
            np.random.seed(args.seed + start)
            torch.manual_seed(args.seed + start)
            if name == "random":
                _, m_enc, m_pred = arm([images[i] for i in range(B)])
            else:
                m_enc, m_pred = arm.generate(batch_size=B, guide_grids=guides,
                                             guide_valid=valid)
            for b in range(B):
                tgt = np.zeros(GRID * GRID, bool)
                for g in m_pred:
                    tgt[g[b].numpy()] = True
                ctx = np.zeros(GRID * GRID, bool)
                ctx[m_enc[0][b].numpy()] = True

                nc = int(cell_anat[b].sum())
                tm = float(mass_val[b].sum())
                acc[name].append(dict(
                    tok_per_mask=float(np.mean([len(g[b]) for g in m_pred])),
                    union=int(tgt.sum()),
                    ctx_tokens=int(ctx.sum()),
                    hid_cell=(float((cell_anat[b] & tgt).sum()) / nc) if nc else np.nan,
                    hid_mass=(float(mass_val[b][tgt].sum()) / tm) if tm > 0 else np.nan,
                    ctx_cell=(float((cell_anat[b] & ctx).sum()) / nc) if nc else np.nan,
                    ctx_mass=(float(mass_val[b][ctx].sum()) / tm) if tm > 0 else np.nan,
                    ctx_anat_tokens=int((cell_anat[b] & ctx).sum()),
                    ctx_zero_anat=bool((cell_anat[b] & ctx).sum() == 0),
                    anat_cells=nc,
                ))
        if (start + B) % 320 == 0:
            print(f"  {start + B}/{len(idxs)}", flush=True)

    res = {}
    for name, rows in acc.items():
        a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        res[name] = {
            "n": len(rows),
            "tok_per_mask": float(a["tok_per_mask"].mean()),
            "union": float(a["union"].mean()),
            "ctx_tokens": float(a["ctx_tokens"].mean()),
            "hidden_cells_pct": float(np.nanmean(a["hid_cell"]) * 100),
            "hidden_mass_pct": float(np.nanmean(a["hid_mass"]) * 100),
            "ctx_cells_pct": float(np.nanmean(a["ctx_cell"]) * 100),
            "ctx_mass_pct": float(np.nanmean(a["ctx_mass"]) * 100),
            "ctx_anat_tokens": float(a["ctx_anat_tokens"].mean()),
            "ctx_zero_anat_pct": float(a["ctx_zero_anat"].mean() * 100),
            "anat_cells": float(a["anat_cells"].mean()),
        }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))

    print("\nANATOMY HIDDEN BY TARGETS (post-truncation, as the model sees it)")
    print(f"{'arm':10s} {'tok/mask':>8s} {'cells def':>10s} {'MASS def':>9s} "
          f"{'ctx cells':>10s} {'ctx MASS':>9s} {'ctx anat tok':>12s} {'ZERO anat ctx':>13s}")
    print("-" * 92)
    for name in ("random", "envelope", "cover"):
        r = res[name]
        print(f"{name:10s} {r['tok_per_mask']:8.1f} {r['hidden_cells_pct']:9.1f}% "
              f"{r['hidden_mass_pct']:8.1f}% {r['ctx_cells_pct']:9.1f}% "
              f"{r['ctx_mass_pct']:8.1f}% {r['ctx_anat_tokens']:12.1f} "
              f"{r['ctx_zero_anat_pct']:12.1f}%")
    e, c = res["envelope"], res["cover"]
    print(f"\ncover - envelope:  cells {c['hidden_cells_pct']-e['hidden_cells_pct']:+.1f} pts"
          f"   MASS {c['hidden_mass_pct']-e['hidden_mass_pct']:+.1f} pts")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
