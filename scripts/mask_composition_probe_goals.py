#!/usr/bin/env python3
"""Same mask-composition measurement, but on GOALS with GROUND-TRUTH anatomy.

FairVision has no anatomy labels, so there the reference is MIRAGE's predicted
guide and the measurement inherits MIRAGE's segmentation error.  GOALS is
labelled, so here we can do two things FairVision cannot:

  1. score every arm against the TRUE anatomy mask, and
  2. drive the guide-consuming arms (envelope, anatomy) from the ground truth
     itself, which removes the segmentation model entirely and isolates the
     masking POLICY.

Labels come from MergedV3 (palette 0=Elsewhere, 128=InnerRetina, 255=Choroid);
on-anatomy means InnerRetina or Choroid.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from mask_composition_probe import (  # noqa: E402
    CROP, GRID, NPATCH, OCC_THRESHOLD, PATCH,
    aggregate, build_arms, run_arm, score_image,
)

PALETTE = {0: 0, 128: 1, 255: 2}          # raw pixel value -> class id
NAMES = {0: "Elsewhere", 1: "InnerRetina", 2: "Choroid"}


def patch_frac(mask_2d, patch=PATCH):
    """Fraction of each patch covered by a boolean pixel mask -> (16,16)."""
    h, w = mask_2d.shape
    gh, gw = h // patch, w // patch
    return mask_2d.reshape(gh, patch, gw, patch).mean(axis=(1, 3))


def load_goals(root, splits, limit=0):
    """Yield (image_tensor_3xHxW, guide_4x16x16, on_anat_flat) per B-scan."""
    items = []
    for sp in splits:
        d = pathlib.Path(root) / sp
        bs = sorted((d / "bscan").glob("GOALS__*.png"))
        ms = sorted((d / "semseg").glob("GOALS__*.png"))
        assert [b.name for b in bs] == [m.name for m in ms], f"{sp} mismatch"
        items.extend(zip(bs, ms))
    if limit:
        items = items[:limit]

    for bp, mp in items:
        img = Image.open(bp).convert("L").resize((CROP, CROP), Image.BILINEAR)
        arr = np.asarray(img, np.float32) / 255.0
        ten = torch.from_numpy(arr)[None].repeat(3, 1, 1)

        raw = np.asarray(Image.open(mp).resize((CROP, CROP), Image.NEAREST))
        raw = raw[..., 0] if raw.ndim == 3 else raw
        gt = np.zeros(raw.shape, np.uint8)
        for v, c in PALETTE.items():
            gt[raw == v] = c

        inner = patch_frac((gt == 1).astype(np.float32))
        chor = patch_frac((gt == 2).astype(np.float32))
        occ = patch_frac((gt > 0).astype(np.float32))
        placement = (occ >= OCC_THRESHOLD).astype(np.float32)
        guide = torch.from_numpy(
            np.stack([occ, placement, inner, chor]).astype(np.float32)
        )
        on_anat = (occ.reshape(-1) >= OCC_THRESHOLD)
        yield ten, guide, on_anat, bp.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\jepa_phase0\mirage-datasets\MergedV3")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--repeats", type=int, default=10,
                    help="mask draws per image (masks are stochastic)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\mask_stats_goals.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = list(load_goals(args.root, args.splits, args.limit))
    print(f"GOALS B-scans: {len(data)} "
          f"(x{args.repeats} mask draws = {len(data) * args.repeats} samples)",
          flush=True)
    if not data:
        raise SystemExit("no GOALS images found")

    arms = build_arms(guide_dir="<ground-truth>")
    rows = {k: [] for k in arms}

    done = 0
    for rep in range(args.repeats):
        for start in range(0, len(data), args.batch_size):
            chunk = data[start:start + args.batch_size]
            images = torch.stack([c[0] for c in chunk], dim=0)
            guides = torch.stack([c[1] for c in chunk], dim=0)
            valid = torch.ones(len(chunk), dtype=torch.bool)
            on_anat = np.stack([c[2] for c in chunk])

            for name, arm in arms.items():
                ctx, tgt = run_arm(name, arm, images, guides, valid)
                for b in range(len(chunk)):
                    rows[name].append(score_image(ctx[b], tgt[b], on_anat[b]))
            done += len(chunk)
        print(f"  repeat {rep + 1}/{args.repeats} ({done} samples)", flush=True)

    res = {name: aggregate(r) for name, r in rows.items()}
    res["_meta"] = dict(
        dataset="GOALS (MergedV3)", splits=args.splits,
        bscans=len(data), repeats=args.repeats,
        samples=len(data) * args.repeats, seed=args.seed,
        occupancy_threshold=OCC_THRESHOLD,
        anatomy_reference="GROUND-TRUTH semseg, InnerRetina|Choroid, "
                          "patch coverage >= 0.25",
        guide_source="GROUND-TRUTH (no MIRAGE) - isolates masking policy",
        note="no random-resized-crop here; images resized directly to 256 so "
             "the anatomy reference is exact",
        grid=f"{GRID}x{GRID}", total_patches=NPATCH,
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)

    hdr = (f"{'arm':10s} | {'tok/mask':>8s} {'on_anat':>8s} {'bg':>8s} | "
           f"{'ctx':>7s} {'on_anat':>8s} {'bg':>8s} | {'mask%an':>8s} {'ctx%an':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("random", "oracle", "envelope", "anatomy"):
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
