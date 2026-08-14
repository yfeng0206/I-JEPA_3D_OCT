"""COVER-then-RANDOM: hide anatomy up to a target, then place plain JEPA blocks.

The variant under test spends blocks greedily on hiding anatomy until
``1 - leave_frac`` of the anatomy mass is covered while keeping at least
``min_visible_frac`` visible, and then stops being clever: every block left over
is a plain uniform rectangle exactly as the stock I-JEPA sampler would draw it
(``transition=False``).

The three things worth knowing before running it are measured here:

  1. how much anatomy actually ends up hidden,
  2. how many of the 4 blocks end up doing coverage work versus being random,
  3. how often it fails -- either breaching the hard visibility floor
     (``floor_violation``) or degenerating entirely (``fallback``).

For contrast it also runs the shipped ``transition=True`` variant on the exact
same slices and block sizes, so the cost of the change is visible rather than
assumed.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.cover import build_targets, anatomy_support        # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
OCC_T, TAU = 0.25, 0.30


def sample_block_sizes(rng, n=4, scale=(0.15, 0.2), ar=(0.75, 1.5)):
    """Mirror the production block-size draw (one draw per batch, npred=4)."""
    s = rng.uniform(*scale)
    a = rng.uniform(*ar)
    n_cells = int(GRID * GRID * s)
    h = max(1, int(round(np.sqrt(n_cells / a))))
    w = max(1, int(round(np.sqrt(n_cells * a))))
    h, w = min(h, GRID), min(w, GRID)
    return [(h, w)] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--volumes", type=int, default=30)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=20)
    ap.add_argument("--leave_frac", type=float, default=0.15)
    ap.add_argument("--min_visible_frac", type=float, default=0.15)
    ap.add_argument("--min_visible_cells", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\cover_random")
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

    rows = []
    examples = []
    for n, i in enumerate(idxs):
        img, guide, _ = ds[i]
        g = guide.numpy()
        sizes = sample_block_sizes(random.Random(args.seed + n))
        for tag, kw in (("cover_random_legal", dict(fill="random_legal")),
                        ("cover_random_free", dict(fill="random")),
                        ("cover_transition", dict(fill="transition"))):
            r = random.Random(args.seed * 7919 + n)
            masks, info = build_targets(
                class_scores=[g[2], g[3]] if g.shape[0] >= 4 else [g[0]],
                block_sizes=sizes, rng=r,
                leave_frac=args.leave_frac,
                min_visible_frac=args.min_visible_frac,
                min_visible_cells=args.min_visible_cells,
                tau=TAU, guided=[True] * 4, fixed=[None] * 4, **kw)
            rows.append(dict(
                variant=tag, idx=i,
                hidden_pct=100 * info["covered_frac"],
                visible_pct=100 * info["visible_frac"],
                n_cover=info["n_cover"], n_random=info["n_random"],
                n_transition=info["n_transition"], n_unguided=info["n_unguided"],
                hit_target=bool(info["hit_target"]), floor_ok=bool(info["floor_ok"]),
                floor_violation=bool(info["floor_violation"]),
                fallback=bool(info["fallback"]), ok=bool(info["ok"]),
                anat_cells=info["anat_cells"], union=info["union"]))
            if tag == "cover_random_legal" and len(examples) < 8 and n % 7 == 0:
                examples.append((i, img.numpy(), g, [m.copy() for m in masks],
                                 list(info["slot_kind"]), dict(
                                     hidden=100 * info["covered_frac"],
                                     ok=bool(info["ok"]))))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(idxs)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_slice.csv", index=False)

    print("\n=== COVER-then-RANDOM vs shipped COVER (same slices, same blocks) ===")
    agg = df.groupby("variant").agg(
        anatomy_hidden_pct=("hidden_pct", "mean"),
        anatomy_visible_pct=("visible_pct", "mean"),
        blocks_cover=("n_cover", "mean"),
        blocks_random=("n_random", "mean"),
        blocks_transition=("n_transition", "mean"),
        hit_target_pct=("hit_target", lambda s: 100 * s.mean()),
        floor_ok_pct=("floor_ok", lambda s: 100 * s.mean()),
        FAIL_floor_violation_pct=("floor_violation", lambda s: 100 * s.mean()),
        FAIL_fallback_pct=("fallback", lambda s: 100 * s.mean()),
        usable_ok_pct=("ok", lambda s: 100 * s.mean()),
    ).reset_index()
    pd.set_option("display.width", 250)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    agg.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(agg.to_dict("records"), indent=2))

    cr = df[df.variant == "cover_random_legal"]
    print("\n--- COVER-then-RANDOM(legal), distribution of blocks per image (of 4) ---")
    print(cr.n_cover.value_counts().sort_index().rename("n_cover").to_string())
    print(cr.n_random.value_counts().sort_index().rename("n_random").to_string())
    print(f"\nanatomy hidden: mean {cr.hidden_pct.mean():.1f}%  "
          f"p10 {cr.hidden_pct.quantile(.1):.1f}%  p90 {cr.hidden_pct.quantile(.9):.1f}%")
    print(f"TOTAL FAILURE RATE (unusable, caller must fall back): "
          f"{100 * (~cr.ok).mean():.2f}%")

    np.save(out / "examples.npy", np.array(examples, dtype=object),
            allow_pickle=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
