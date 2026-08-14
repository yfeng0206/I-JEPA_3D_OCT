#!/usr/bin/env python3
"""Measure what each masking policy actually shows and hides, on real data.

For every arm we report, per image:

    context tokens         how many patches reach the encoder
      - on anatomy         ... of those, how many sit on retinal tissue
      - background         ... how many are off-tissue (black)
    hidden tokens (union)  how many distinct patches are removed as targets
      - on anatomy         ... of those, how many sit on tissue
      - background         ... how many are background
    predictor slots        target cells actually fed to the loss, duplicates
                           included (pred_target_k pads short targets WITH
                           REPLACEMENT, so slots > unique hidden cells)

Arms use the production classes verbatim:
    random    src.masks.multiblock.MaskCollator
    oracle    CurriculumMaskGenerator(mode='anatomical_prior')
    envelope  CurriculumMaskGenerator(mode='mirage_envelope')
    anatomy   CurriculumMaskGenerator(mode='mirage_anatomy', pred_target_k=16,
                                      anatomy_bridge_diagonals=True)

All arms are scored against the SAME anatomy reference so the comparison is
about masking policy alone.  On FairVision that reference is the MIRAGE guide
occupancy (fraction of the patch covered by retina, thresholded at the
production 0.25).  On GOALS it is the ground-truth segmentation.
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
sys.path.insert(0, str(HERE.parent))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

GRID = 16
NPATCH = GRID * GRID
OCC_THRESHOLD = 0.25          # production mirage_occupancy_threshold
CROP = 256
PATCH = 16

BASE_KW = dict(
    input_size=(CROP, CROP),
    patch_size=PATCH,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    nenc=1,
    npred=4,
    min_keep=10,
    allow_overlap=False,
)

CURR_COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear")


def build_arms(guide_dir):
    """Instantiate one mask generator per arm, all ramped to r_t = 1.0."""
    arms = {}
    arms["random"] = MaskCollator(**BASE_KW)

    arms["oracle"] = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(mode="anatomical_prior", **CURR_COMMON),
    )
    arms["envelope"] = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(
            mode="mirage_envelope", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25, **CURR_COMMON,
        ),
    )
    arms["anatomy"] = CurriculumMaskGenerator(
        **BASE_KW,
        pred_target_k=16,
        curriculum_cfg=dict(
            mode="mirage_anatomy", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25,
            anatomy_mass_cap=0.9, anatomy_tau=0.1,
            anatomy_bridge_diagonals=True, **CURR_COMMON,
        ),
    )
    arms["cover"] = CurriculumMaskGenerator(
        **BASE_KW,
        # pred_target_k deliberately UNSET, exactly as the envelope arm: COVER
        # targets are rectangles of the batch-shared sizes, so the stock
        # global-min truncation is harmless and keeps the arms comparable.
        curriculum_cfg=dict(
            mode="mirage_cover", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25,
            anatomy_tau=0.1,
            cover_leave_frac=0.15, cover_min_visible_frac=0.15,
            cover_min_visible_cells=4, cover_transition=True, **CURR_COMMON,
        ),
    )
    for name, a in arms.items():
        if hasattr(a, "set_epoch"):
            a.set_epoch(50, 100)          # well past T_total -> r_t = r_max = 1
    return arms


def run_arm(name, arm, images, guides, valid):
    """Return (list_of_context_index_sets, list_of_target_index_lists)."""
    B = images.size(0)
    if name == "random":
        _, m_enc, m_pred = arm([images[i] for i in range(B)])
    elif name == "oracle":
        m_enc, m_pred = arm.generate(batch_size=B, imgs_cpu=images)
    else:
        m_enc, m_pred = arm.generate(
            batch_size=B, guide_grids=guides, guide_valid=valid
        )
    ctx = [set(m_enc[0][b].tolist()) for b in range(B)]
    blocks = []
    for b in range(B):
        blocks.append([g[b].tolist() for g in m_pred])
    return ctx, blocks


def score_image(ctx, blocks, on_anat_flat):
    """Per-mask and per-image splits of context / masked tokens."""
    ci = np.fromiter(ctx, dtype=int, count=len(ctx))
    c_on = int(on_anat_flat[ci].sum()) if ci.size else 0

    # ---- per individual target mask (npred of them) ----
    per_tok, per_on, per_uniq, per_uniq_on = [], [], [], []
    for blk in blocks:
        a = np.asarray(blk, dtype=int)
        u = np.unique(a)
        per_tok.append(a.size)
        per_on.append(int(on_anat_flat[a].sum()))
        per_uniq.append(u.size)
        per_uniq_on.append(int(on_anat_flat[u].sum()))

    slots = np.concatenate([np.asarray(b, dtype=int) for b in blocks])
    hid = np.unique(slots)
    h_on = int(on_anat_flat[hid].sum()) if hid.size else 0

    return dict(
        # context
        n_ctx=int(ci.size),
        ctx_on_anat=c_on,
        ctx_bg=int(ci.size) - c_on,
        # PER MASK (one target block)
        mask_tokens=float(np.mean(per_tok)),
        mask_on_anat=float(np.mean(per_on)),
        mask_bg=float(np.mean(per_tok) - np.mean(per_on)),
        mask_uniq_cells=float(np.mean(per_uniq)),
        mask_uniq_on_anat=float(np.mean(per_uniq_on)),
        mask_uniq_bg=float(np.mean(per_uniq) - np.mean(per_uniq_on)),
        # per image, union over the npred masks
        n_hidden=int(hid.size),
        hid_on_anat=h_on,
        hid_bg=int(hid.size) - h_on,
        n_slots=int(slots.size),
        slot_dupes=int(slots.size - hid.size),
        anat_cells=int(on_anat_flat.sum()),
    )


def aggregate(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    out = {"n_images": len(rows)}
    for k, v in arr.items():
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sd"] = float(v.std())
    # Derived rates, computed from the means so they read as percentages.
    ctx, hid = arr["n_ctx"].mean(), arr["n_hidden"].mean()
    out["ctx_frac_of_grid"] = float(ctx / NPATCH)
    out["hidden_frac_of_grid"] = float(hid / NPATCH)
    out["ctx_pct_on_anat"] = float(arr["ctx_on_anat"].mean() / ctx * 100) if ctx else 0.0
    out["hidden_pct_on_anat"] = float(arr["hid_on_anat"].mean() / hid * 100) if hid else 0.0
    # What share of all anatomy cells in the image does each set cover?
    anat = arr["anat_cells"].mean()
    out["anat_cells_mean"] = float(anat)
    out["ctx_share_of_all_anat"] = float(arr["ctx_on_anat"].mean() / anat * 100) if anat else 0.0
    out["hidden_share_of_all_anat"] = float(arr["hid_on_anat"].mean() / anat * 100) if anat else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--volumes", type=int, default=100)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\mask_stats_fairvision.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0,
    )
    sc = os.path.join(args.slice_cache, args.split)
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(args.data_dir, args.split),
        guide_dir=os.path.join(args.guide_dir, args.split),
        num_slices=args.num_slices,
        slice_size=CROP,
        transform=paired,
        patch_size=PATCH,
        dilate_patches=0,
        occupancy_threshold=OCC_THRESHOLD,
        slice_cache=sc if os.path.isdir(sc) else None,
    )
    n_vol = len(ds.file_paths)
    print(f"dataset: {len(ds)} slices from {n_vol} volumes", flush=True)

    vols = sorted(random.sample(range(n_vol), min(args.volumes, n_vol)))
    idxs = [v * args.num_slices + s
            for v in vols for s in range(args.num_slices)]
    print(f"sampling {len(vols)} volumes -> {len(idxs)} slices", flush=True)

    arms = build_arms(args.guide_dir)
    rows = {k: [] for k in arms}

    done = 0
    for start in range(0, len(idxs), args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        items = [ds[i] for i in chunk]
        images = torch.stack([it[0] for it in items], dim=0)
        guides = torch.stack([it[1] for it in items], dim=0)
        valid = torch.stack([it[2] for it in items], dim=0)

        # Anatomy reference: channel 0 is the per-patch retina occupancy.
        occ = guides[:, 0].reshape(guides.size(0), -1).numpy()
        on_anat = (occ >= OCC_THRESHOLD)

        for name, arm in arms.items():
            try:
                ctx, tgt = run_arm(name, arm, images, guides, valid)
            except Exception as exc:  # noqa: BLE001
                print(f"  ARM {name} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                raise
            for b in range(images.size(0)):
                rows[name].append(score_image(ctx[b], tgt[b], on_anat[b]))

        done += len(chunk)
        if done % (args.batch_size * 10) == 0 or done >= len(idxs):
            print(f"  {done}/{len(idxs)} slices", flush=True)

    res = {name: aggregate(r) for name, r in rows.items()}
    res["_meta"] = dict(
        dataset="FairVision-glaucoma", split=args.split,
        volumes=len(vols), slices=len(idxs), seed=args.seed,
        occupancy_threshold=OCC_THRESHOLD, guide_dir=args.guide_dir,
        anatomy_reference="MIRAGE guide occupancy channel 0 >= 0.25",
        grid=f"{GRID}x{GRID}", total_patches=NPATCH,
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)

    hdr = (f"{'arm':10s} | {'tok/mask':>8s} {'on_anat':>8s} {'bg':>8s} | "
           f"{'ctx':>7s} {'on_anat':>8s} {'bg':>8s} | {'mask%an':>8s} {'ctx%an':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("random", "oracle", "envelope", "anatomy", "cover"):
        r = res.get(name) or {}
        if not r:
            continue
        mt, mo = r["mask_tokens_mean"], r["mask_on_anat_mean"]
        ct, co = r["n_ctx_mean"], r["ctx_on_anat_mean"]
        print(f"{name:10s} | {mt:8.1f} {mo:8.1f} {r['mask_bg_mean']:8.1f} | "
              f"{ct:7.1f} {co:8.1f} {r['ctx_bg_mean']:8.1f} | "
              f"{mo / mt * 100:7.1f}% {co / ct * 100:6.1f}%")


if __name__ == "__main__":
    main()
