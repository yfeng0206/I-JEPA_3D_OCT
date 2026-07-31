#!/usr/bin/env python
"""Constrained MIRAGE masking: original block sizes plus acceptance criteria.

Target blocks keep the current I-JEPA geometry (``pred_mask_scale`` unchanged),
but a sample is only accepted when

  * every target rectangle is at least ``min_block_fill`` covered by the MIRAGE
    retinal region, and
  * at least ``min_retina_visible`` of the retina survives in the encoder's
    context, so the encoder always has anatomy left to reason from.

Sweeps the two thresholds to show which combinations are feasible, then renders
the chosen configuration.

Example:
    python scripts/mirage_constrained_masking.py --sweep
    python scripts/mirage_constrained_masking.py --render --fill 0.4 --visible 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    build_union,
    expand_envelope,
    patch_occupancy,
    repair_union,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_constrained")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
EXPANSION = 0.05
MIN_OCCUPANCY = 0.5
MAX_ATTEMPTS = 30
TILE, HEADER, ROW_LABEL = 256, 20, 112


def make_generator(pred_scale=(0.15, 0.20)) -> CurriculumMaskGenerator:
    return CurriculumMaskGenerator(
        input_size=(CROP, CROP),
        patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=pred_scale,
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg={"mode": "anatomical_prior", "T_warm": 25, "T_total": 30,
                        "r_max": 1.0, "oracle_region_frac": 0.28,
                        "oracle_lateral_frac": 0.6},
    )


def window_sums(grid: np.ndarray, block_h: int, block_w: int):
    height, width = grid.shape
    n_top, n_left = height - block_h + 1, width - block_w + 1
    if n_top <= 0 or n_left <= 0:
        return None
    padded = np.zeros((height + 1, width + 1), dtype=np.float64)
    padded[1:, 1:] = grid
    sat = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        sat[block_h:, block_w:]
        - sat[:n_top, block_w:]
        - sat[block_h:, :n_left]
        + sat[:n_top, :n_left]
    )


def sample_constrained(
    generator,
    occupancy,
    seed,
    min_block_fill,
    min_retina_visible,
    max_attempts=MAX_ATTEMPTS,
    spread=True,
    placement_region=None,
):
    """Original block sizes, accepted only if fill and retention thresholds hold.

    ``placement_region`` optionally supplies a dilated patch grid used *only* to
    judge how well a block covers retina.  Retina visibility is always measured
    against the true MIRAGE region, so adding placement slack can never inflate
    the apparent amount of retina left for the encoder.
    """
    in_region = (
        np.asarray(placement_region, dtype=np.float64)
        if placement_region is not None
        else (occupancy >= MIN_OCCUPANCY).astype(np.float64)
    )
    retina = set(np.flatnonzero(occupancy.reshape(-1) >= MIN_OCCUPANCY).tolist())
    best = None

    for attempt in range(max_attempts):
        attempt_seed = seed * 131 + attempt
        generator._size_gen.manual_seed(attempt_seed)
        sizes = [
            generator._sample_block_size(
                generator.pred_mask_scale, generator._size_gen
            )
            for _ in range(generator.npred)
        ]
        enc_h, enc_w = generator._sample_block_size(
            generator.enc_mask_scale, generator._size_gen
        )
        rng = np.random.default_rng(attempt_seed)
        torch.manual_seed(attempt_seed)
        random.seed(attempt_seed)

        blocks, union, fills = [], set(), []
        feasible = True
        claimed = np.zeros_like(in_region)
        occupied_cols = np.flatnonzero(in_region.any(axis=0))
        segments: list = []
        if spread and occupied_cols.size:
            edges = np.linspace(
                occupied_cols.min(), occupied_cols.max() + 1, len(sizes) + 1
            )
            segments = [
                (int(np.floor(edges[i])), int(np.ceil(edges[i + 1])))
                for i in range(len(sizes))
            ]
            segments = [segments[i] for i in rng.permutation(len(segments))]
        for block_index, (block_h, block_w) in enumerate(sizes):
            counts = window_sums(in_region, block_h, block_w)
            if counts is None:
                feasible = False
                break
            fill = counts / float(block_h * block_w)
            candidates = fill >= min_block_fill
            if not candidates.any():
                feasible = False
                break
            # Spread the blocks along the retina: restrict this block to its own
            # lateral segment, and prefer windows that do not reuse patches an
            # earlier block already claimed.  Both relax rather than fail.
            if segments:
                low, high = segments[block_index]
                centre_cols = np.arange(candidates.shape[1]) + block_w / 2.0
                banded = np.zeros_like(candidates)
                inside = (centre_cols >= low) & (centre_cols < high)
                banded[:, inside] = candidates[:, inside]
                if banded.any():
                    candidates = banded
            overlap = window_sums(claimed, block_h, block_w)
            free = candidates & (overlap <= 0.25 * block_h * block_w)
            if free.any():
                candidates = free
            rows, cols = np.nonzero(candidates)
            picked = int(rng.choice(rows.size))
            top, left = int(rows[picked]), int(cols[picked])
            blocks.append(
                {
                    "top": top,
                    "left": left,
                    "h": block_h,
                    "w": block_w,
                    "fill": float(fill[top, left]),
                }
            )
            fills.append(float(fill[top, left]))
            union.update(generator._block_to_indices(top, left, block_h, block_w))
            claimed[top : top + block_h, left : left + block_w] = 1.0
        if not feasible:
            continue

        context, enc_block = place_context(generator, union, enc_h, enc_w, rng)
        visible = (
            len(retina & set(context)) / len(retina) if retina else 1.0
        )
        record = {
            "blocks": blocks,
            "union": union,
            "context": context,
            "enc_block": enc_block,
            "retina_visible": visible,
            "min_fill": min(fills) if fills else 0.0,
            "mean_fill": float(np.mean(fills)) if fills else 0.0,
            "attempts": attempt + 1,
            "accepted": visible >= min_retina_visible,
        }
        if best is None or record["retina_visible"] > best["retina_visible"]:
            best = record
        if record["accepted"]:
            return record
    if best is not None:
        return best

    # No placement satisfied the fill threshold on this slice (a badly
    # fragmented guide).  Fall back to plain uniform blocks, exactly as the
    # training run must when a guide is unusable, and flag it for logging.
    fallback_seed = seed * 131 + max_attempts
    generator._size_gen.manual_seed(fallback_seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    enc_h, enc_w = generator._sample_block_size(
        generator.enc_mask_scale, generator._size_gen
    )
    rng = np.random.default_rng(fallback_seed)
    blocks, union = [], set()
    for block_h, block_w in sizes:
        top = int(rng.integers(0, generator.height - block_h + 1))
        left = int(rng.integers(0, generator.width - block_w + 1))
        fill = float(in_region[top : top + block_h, left : left + block_w].mean())
        blocks.append(
            {"top": top, "left": left, "h": block_h, "w": block_w, "fill": fill}
        )
        union.update(generator._block_to_indices(top, left, block_h, block_w))
    context, enc_block = place_context(generator, union, enc_h, enc_w, rng)
    visible = len(retina & set(context)) / len(retina) if retina else 1.0
    return {
        "blocks": blocks,
        "union": union,
        "context": context,
        "enc_block": enc_block,
        "retina_visible": visible,
        "min_fill": min(b["fill"] for b in blocks),
        "mean_fill": float(np.mean([b["fill"] for b in blocks])),
        "attempts": max_attempts,
        "accepted": False,
        "fallback": True,
    }


def sample_cascaded(
    generator,
    occupancy,
    seed,
    fill_ladder=(0.5, 0.4, 0.3),
    min_retina_visible=0.25,
    max_attempts=MAX_ATTEMPTS,
    spread=True,
):
    """Try progressively looser fill thresholds before giving up on the guide.

    A single strict threshold is brittle: on steep or poorly segmented slices no
    full-size block can reach 50% retina, and the sampler collapses straight to
    uniform random placement -- strictly worse than a slightly looser but still
    guided mask.  The ladder keeps as much guidance as the slice can support.
    """
    for fill in fill_ladder:
        sample = sample_constrained(
            generator,
            occupancy,
            seed,
            fill,
            min_retina_visible,
            max_attempts=max_attempts,
            spread=spread,
        )
        if sample is None or sample.get("fallback"):
            continue
        if sample["accepted"]:
            sample["fill_threshold_used"] = fill
            return sample
        best_guided = sample
        best_guided["fill_threshold_used"] = fill
    # Nothing satisfied the retention rule; return the loosest guided attempt if
    # one exists, else the random fallback from the final rung.
    final = sample_constrained(
        generator,
        occupancy,
        seed,
        fill_ladder[-1],
        min_retina_visible,
        max_attempts=max_attempts,
        spread=spread,
    )
    if final is not None:
        final.setdefault("fill_threshold_used", fill_ladder[-1])
    return final


def place_context(generator, union, enc_h, enc_w, rng):
    for _attempt in range(50):
        top = int(rng.integers(0, generator.height - enc_h + 1))
        left = int(rng.integers(0, generator.width - enc_w + 1))
        indices = generator._block_to_indices(top, left, enc_h, enc_w)
        kept = [i for i in indices if i not in union]
        if len(kept) >= generator.min_keep:
            return kept, {"top": top, "left": left, "h": enc_h, "w": enc_w}
    kept = [i for i in range(generator.num_patches) if i not in union]
    return kept, {"top": 0, "left": 0, "h": generator.height, "w": generator.width}


def load_grids(volumes, slices):
    grids = []
    for path in sorted(MASK_DIR.glob("data_*.npz"))[:volumes]:
        with np.load(path, allow_pickle=False) as cache:
            masks = cache["hard_masks"]
            slice_indices = cache["slice_indices"].astype(int)
            label = int(cache["glaucoma"])
        for cache_index in slices:
            envelope, _v, _s = repair_union(
                build_union(masks[cache_index]), params=DEFAULT_REPAIR
            )
            grown, _achieved = expand_envelope(envelope, EXPANSION)
            grown_256 = (
                np.asarray(
                    Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                        (CROP, CROP), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
            grids.append(
                {
                    "volume": path.stem,
                    "label": label,
                    "cache_index": int(cache_index),
                    "slice": int(slice_indices[cache_index]),
                    "occupancy": patch_occupancy(grown_256, patch_size=PATCH),
                }
            )
    return grids


def evaluate(grids, fill, visible, generator):
    accepted, retina_vis, block_fill, area, attempts, in_region = [], [], [], [], [], []
    fallbacks: list = []
    for item in grids:
        occupancy = item["occupancy"]
        sample = sample_constrained(
            generator, occupancy, 23 + item["slice"], fill, visible
        )
        if sample is None:
            accepted.append(0.0)
            continue
        accepted.append(1.0 if sample["accepted"] else 0.0)
        fallbacks.append(1.0 if sample.get("fallback") else 0.0)
        retina_vis.append(sample["retina_visible"])
        block_fill.append(sample["mean_fill"])
        area.append(len(sample["union"]) / 256.0)
        attempts.append(sample["attempts"])
        flat = occupancy.reshape(-1)
        target = flat[sorted(sample["union"])]
        in_region.append(float((target >= MIN_OCCUPANCY).mean()))
    return {
        "acceptance_rate": round(float(np.mean(accepted)), 3),
        "retina_visible": round(float(np.mean(retina_vis)), 3) if retina_vis else 0.0,
        "mean_block_fill": round(float(np.mean(block_fill)), 3) if block_fill else 0.0,
        "target_in_region": round(float(np.mean(in_region)), 3) if in_region else 0.0,
        "masked_area": round(float(np.mean(area)), 3) if area else 0.0,
        "mean_attempts": round(float(np.mean(attempts)), 1) if attempts else 0.0,
        "fallback_rate": round(float(np.mean(fallbacks)), 3) if fallbacks else 0.0,
    }


def render(slice_256, occupancy, sample):
    base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
    region_pixels = np.kron(
        occupancy >= MIN_OCCUPANCY, np.ones((PATCH, PATCH), dtype=bool)
    )
    tint = np.array([40.0, 220.0, 90.0], dtype=np.float32)
    alpha = 0.28 * region_pixels[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha

    enc = sample["enc_block"]
    visible = np.zeros((GRID, GRID), dtype=bool)
    visible[enc["top"] : enc["top"] + enc["h"], enc["left"] : enc["left"] + enc["w"]] = True
    blended[~np.kron(visible, np.ones((PATCH, PATCH), dtype=bool))] *= 0.5

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(
        [enc["left"] * PATCH, enc["top"] * PATCH,
         (enc["left"] + enc["w"]) * PATCH - 1, (enc["top"] + enc["h"]) * PATCH - 1],
        outline=(0, 220, 255),
        width=2,
    )
    for block in sample["blocks"]:
        x0, y0 = block["left"] * PATCH, block["top"] * PATCH
        draw.rectangle(
            [x0, y0, x0 + block["w"] * PATCH - 1, y0 + block["h"] * PATCH - 1],
            fill=(255, 60, 60, 85),
            outline=(255, 60, 60),
        )
        draw.text((x0 + 4, y0 + 4), f"{block['fill'] * 100:.0f}%", fill=(255, 240, 120))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fill", type=float, default=0.4)
    parser.add_argument("--visible", type=float, default=0.3)
    parser.add_argument("--volumes", type=int, default=10)
    parser.add_argument("--slices", type=int, nargs="+", default=[0, 25, 50, 75, 99])
    parser.add_argument("--compare", type=float, nargs="+")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    generator = make_generator()
    if args.compare:
        args.output.mkdir(parents=True, exist_ok=True)
        grids = load_grids(args.volumes, args.slices)
        stats = {
            f"fill>={fill:.2f}": evaluate(grids, fill, args.visible, generator)
            for fill in args.compare
        }
        print(json.dumps(stats, indent=2))
        (args.output / "compare.json").write_text(
            json.dumps({"visible": args.visible, "stats": stats}, indent=2) + "\n",
            encoding="utf-8",
        )
        titles = ["Original OCT"] + [f"fill >= {f:.0%}" for f in args.compare]
        for path in sorted(MASK_DIR.glob("data_*.npz"))[:3]:
            with np.load(path, allow_pickle=False) as cache:
                masks = cache["hard_masks"]
                slice_indices = cache["slice_indices"].astype(int)
                label = int(cache["glaucoma"])
            with np.load(DATA_DIR / path.name, allow_pickle=False) as data:
                volume = data["oct_bscans"]
            rows = []
            for cache_index in (0, 50, 99):
                slice_index = int(slice_indices[cache_index])
                slice_256 = np.asarray(
                    Image.fromarray(volume[slice_index], mode="L").resize(
                        (CROP, CROP), Image.BILINEAR
                    )
                )
                envelope, _v, _s = repair_union(
                    build_union(masks[cache_index]), params=DEFAULT_REPAIR
                )
                grown, _a = expand_envelope(envelope, EXPANSION)
                grown_256 = (
                    np.asarray(
                        Image.fromarray(
                            grown.astype(np.uint8) * 255, mode="L"
                        ).resize((CROP, CROP), Image.Resampling.NEAREST)
                    )
                    > 127
                )
                occupancy = patch_occupancy(grown_256, patch_size=PATCH)
                columns = [Image.fromarray(slice_256, mode="L").convert("RGB")]
                samples = []
                for fill in args.compare:
                    sample = sample_constrained(
                        generator, occupancy, 23 + slice_index, fill, args.visible
                    )
                    columns.append(render(slice_256, occupancy, sample))
                    samples.append(sample)
                rows.append(
                    {"slice": slice_index, "columns": columns, "samples": samples}
                )
            panel = Image.new(
                "RGB",
                (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
                "white",
            )
            draw = ImageDraw.Draw(panel)
            draw.text(
                (6, 5),
                f"{path.stem} glaucoma={label}   retina visible >= {args.visible:.0%}; "
                "yellow % = share of each block that is retina",
                fill="black",
            )
            for column, name in enumerate(titles):
                draw.text(
                    (ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black"
                )
            for index, row in enumerate(rows):
                top = HEADER * 2 + index * TILE
                draw.text((4, top + 6), f"slice {row['slice']}", fill="black")
                for column, image in enumerate(row["columns"]):
                    panel.paste(image, (ROW_LABEL + column * TILE, top))
                for column, sample in enumerate(row["samples"]):
                    x = ROW_LABEL + (column + 1) * TILE + 4
                    draw.text(
                        (x, top + TILE - 34),
                        "accepted" if sample["accepted"] else "RELAXED",
                        fill=(120, 255, 160) if sample["accepted"] else (255, 110, 110),
                    )
                    draw.text(
                        (x, top + TILE - 20),
                        f"retina vis {sample['retina_visible'] * 100:.0f}%  "
                        f"fill {sample['mean_fill'] * 100:.0f}%",
                        fill=(255, 240, 120),
                    )
            panel.save(args.output / f"{path.stem}_fillcompare.png", optimize=True)
        return

    if args.sweep:
        grids = load_grids(args.volumes, args.slices)
        results = {}
        for fill in (0.3, 0.4, 0.5, 0.6, 0.7):
            for visible in (0.2, 0.3, 0.4):
                results[f"fill>={fill:.1f}_visible>={visible:.1f}"] = evaluate(
                    grids, fill, visible, generator
                )
        print(json.dumps(results, indent=2))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "sweep.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        return

    if args.render:
        args.output.mkdir(parents=True, exist_ok=True)
        for path in sorted(MASK_DIR.glob("data_*.npz"))[:3]:
            with np.load(path, allow_pickle=False) as cache:
                masks = cache["hard_masks"]
                slice_indices = cache["slice_indices"].astype(int)
                label = int(cache["glaucoma"])
            with np.load(DATA_DIR / path.name, allow_pickle=False) as data:
                volume = data["oct_bscans"]
            rows = []
            for cache_index in (0, 50, 99):
                slice_index = int(slice_indices[cache_index])
                slice_256 = np.asarray(
                    Image.fromarray(volume[slice_index], mode="L").resize(
                        (CROP, CROP), Image.BILINEAR
                    )
                )
                envelope, _v, _s = repair_union(
                    build_union(masks[cache_index]), params=DEFAULT_REPAIR
                )
                grown, _a = expand_envelope(envelope, EXPANSION)
                grown_256 = (
                    np.asarray(
                        Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                            (CROP, CROP), Image.Resampling.NEAREST
                        )
                    )
                    > 127
                )
                occupancy = patch_occupancy(grown_256, patch_size=PATCH)
                sample = sample_constrained(
                    generator, occupancy, 23 + slice_index, args.fill, args.visible
                )
                rows.append(
                    {
                        "slice": slice_index,
                        "image": render(slice_256, occupancy, sample),
                        "original": Image.fromarray(slice_256, mode="L").convert("RGB"),
                        "sample": sample,
                    }
                )
            panel = Image.new(
                "RGB", (ROW_LABEL + TILE * 2, HEADER * 2 + TILE * len(rows)), "white"
            )
            draw = ImageDraw.Draw(panel)
            draw.text(
                (6, 5),
                f"{path.stem} glaucoma={label}   block fill >= {args.fill:.0%}, "
                f"retina visible >= {args.visible:.0%}   "
                "(yellow % = how much of each block is retina)",
                fill="black",
            )
            for column, name in enumerate(["Original OCT", "constrained masking"]):
                draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
            for index, row in enumerate(rows):
                top = HEADER * 2 + index * TILE
                sample = row["sample"]
                draw.text((4, top + 6), f"slice {row['slice']}", fill="black")
                draw.text(
                    (4, top + 24),
                    "OK" if sample["accepted"] else "relaxed",
                    fill="green" if sample["accepted"] else "red",
                )
                draw.text(
                    (4, top + 40),
                    f"vis {sample['retina_visible'] * 100:.0f}%",
                    fill="black",
                )
                draw.text((4, top + 56), f"tries {sample['attempts']}", fill="black")
                panel.paste(row["original"], (ROW_LABEL, top))
                panel.paste(row["image"], (ROW_LABEL + TILE, top))
            panel.save(args.output / f"{path.stem}_constrained.png", optimize=True)
        print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
