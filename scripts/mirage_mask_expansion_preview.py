#!/usr/bin/env python
"""Preview MIRAGE-guided target-block masking at several envelope expansions.

Runs the *actual* training-time sampler -- ``CurriculumMaskGenerator``'s block
size draw plus its summed-area-table biased location sampler -- against the
repaired retinal envelope grown by 0%, 5%, 10% and 15%, so the effect of border
expansion on real target placement can be judged directly.

Block sizes are drawn from one shared seed per slice, so across the expansion
columns only the block *locations* change.

Example:
    python scripts/mirage_mask_expansion_preview.py --volumes 3
"""

from __future__ import annotations

import argparse
import json
import os
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
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_mask_expansion")

CROP = 256
PATCH = 16
GRID = CROP // PATCH
EXPANSIONS = (0.0, 0.05, 0.10, 0.15)

TILE = 256
HEADER = 20
ROW_LABEL = 104


def make_generator() -> CurriculumMaskGenerator:
    """Generator configured exactly as the MIRAGE run will be."""
    return CurriculumMaskGenerator(
        input_size=(CROP, CROP),
        patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg={
            "mode": "anatomical_prior",
            "T_warm": 25,
            "T_total": 30,
            "r_max": 1.0,
            "ramp_shape": "linear",
        },
    )


def sample_masks(
    generator: CurriculumMaskGenerator,
    weight_grid: np.ndarray,
    seed: int,
) -> dict:
    """Draw four biased target blocks plus one context block, as in training."""
    generator._size_gen.manual_seed(seed)
    pred_sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    enc_sizes = [
        generator._sample_block_size(generator.enc_mask_scale, generator._size_gen)
        for _ in range(generator.nenc)
    ]

    torch.manual_seed(seed)
    import random

    random.seed(seed)

    grid = torch.from_numpy(np.asarray(weight_grid, dtype=np.float32))
    blocks = []
    pred_union: set = set()
    for block_h, block_w in pred_sizes:
        top, left = generator._sample_biased_location(block_h, block_w, grid)
        indices = generator._block_to_indices(top, left, block_h, block_w)
        blocks.append(
            {"top": top, "left": left, "h": block_h, "w": block_w, "indices": indices}
        )
        pred_union.update(indices)

    context: list = []
    for block_h, block_w in enc_sizes:
        chosen = None
        for _attempt in range(50):
            top, left = generator._sample_uniform_location(
                block_h, block_w, generator.height, generator.width
            )
            indices = generator._block_to_indices(top, left, block_h, block_w)
            kept = [i for i in indices if i not in pred_union]
            if len(kept) >= generator.min_keep:
                chosen = kept
                break
        if chosen is None:
            chosen = [i for i in range(generator.num_patches) if i not in pred_union]
        context.extend(chosen)
    return {"pred": blocks, "pred_union": pred_union, "context": sorted(set(context))}


def block_statistics(sample: dict, weight_grid: np.ndarray) -> dict:
    """The six per-batch statistics the training run will log."""
    flat = np.asarray(weight_grid, dtype=np.float32).reshape(-1)
    per_block = [len(block["indices"]) for block in sample["pred"]]
    target_indices = [i for block in sample["pred"] for i in block["indices"]]
    occupancy = flat[target_indices]
    return {
        "patches_per_block": per_block,
        "unique_target_patches": len(sample["pred_union"]),
        "context_patches": len(sample["context"]),
        "target_on_envelope_frac": float((occupancy > 0.0).mean()),
        "target_background_frac": float((occupancy == 0.0).mean()),
        "mean_target_occupancy": float(occupancy.mean()),
    }


def render(
    slice_256: np.ndarray,
    envelope_256: np.ndarray,
    sample: dict,
) -> Image.Image:
    """OCT with the envelope tinted green and the four target blocks in red."""
    base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
    tint = np.array([40.0, 220.0, 90.0], dtype=np.float32)
    alpha = 0.30 * envelope_256[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha

    visible = np.zeros((GRID, GRID), dtype=bool)
    for index in sample["context"]:
        visible[index // GRID, index % GRID] = True
    visible_pixels = np.kron(visible, np.ones((PATCH, PATCH), dtype=bool))
    blended[~visible_pixels] *= 0.45

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for block in sample["pred"]:
        x0, y0 = block["left"] * PATCH, block["top"] * PATCH
        x1 = x0 + block["w"] * PATCH - 1
        y1 = y0 + block["h"] * PATCH - 1
        draw.rectangle([x0, y0, x1, y1], fill=(255, 60, 60, 90), outline=(255, 60, 60))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=3)
    parser.add_argument("--slices", type=int, nargs="+", default=[0, 50, 99])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    generator = make_generator()
    mask_paths = sorted(MASK_DIR.glob("data_*.npz"))[: args.volumes]
    if not mask_paths:
        raise FileNotFoundError(f"No MIRAGE caches under {MASK_DIR}")

    records = []
    for mask_path in mask_paths:
        with np.load(mask_path, allow_pickle=False) as cache:
            masks = cache["hard_masks"]
            slice_indices = cache["slice_indices"].astype(int)
            label = int(cache["glaucoma"])
        with np.load(DATA_DIR / mask_path.name, allow_pickle=False) as data:
            volume = data["oct_bscans"]

        rows = []
        for cache_index in args.slices:
            slice_index = int(slice_indices[cache_index])
            slice_256 = np.asarray(
                Image.fromarray(volume[slice_index], mode="L").resize(
                    (CROP, CROP), Image.BILINEAR
                )
            )
            envelope, _valid, _stats = repair_union(
                build_union(masks[cache_index]), params=DEFAULT_REPAIR
            )
            seed = args.seed + slice_index

            columns = [
                Image.fromarray(slice_256, mode="L").convert("RGB")
            ]
            labels = ["Original OCT"]
            stats_per_column = []
            for expansion in EXPANSIONS:
                grown, achieved = expand_envelope(envelope, expansion)
                grown_256 = (
                    np.asarray(
                        Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                            (CROP, CROP), Image.Resampling.NEAREST
                        )
                    )
                    > 127
                )
                weight_grid = patch_occupancy(grown_256, patch_size=PATCH)
                sample = sample_masks(generator, weight_grid, seed)
                stats = block_statistics(sample, weight_grid)
                stats.update(
                    {
                        "expansion": expansion,
                        "achieved_expansion": round(achieved, 4),
                        "envelope_area_frac": round(float(grown.mean()), 4),
                        "grid_coverage": round(float((weight_grid > 0).mean()), 4),
                    }
                )
                stats_per_column.append(stats)
                columns.append(render(slice_256, grown_256, sample))
                labels.append(f"+{int(expansion * 100)}% expand")
                records.append(
                    {
                        "volume": mask_path.stem,
                        "glaucoma": label,
                        "slice": slice_index,
                        **stats,
                    }
                )
            rows.append(
                {
                    "slice": slice_index,
                    "columns": columns,
                    "labels": labels,
                    "stats": stats_per_column,
                }
            )

        width = ROW_LABEL + TILE * len(rows[0]["columns"])
        height = HEADER * 2 + TILE * len(rows)
        panel = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (6, 5),
            f"{mask_path.stem}  glaucoma={label}   "
            "green=repaired envelope, red=4 target blocks, dark=hidden from encoder",
            fill="black",
        )
        for column, name in enumerate(rows[0]["labels"]):
            draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
        for row_index, row in enumerate(rows):
            top = HEADER * 2 + row_index * TILE
            draw.text((4, top + 6), f"slice {row['slice']}", fill="black")
            for column, image in enumerate(row["columns"]):
                panel.paste(image, (ROW_LABEL + column * TILE, top))
            for column, stats in enumerate(row["stats"]):
                x = ROW_LABEL + (column + 1) * TILE + 4
                draw.text(
                    (x, top + TILE - 34),
                    f"on-env {stats['target_on_envelope_frac'] * 100:.0f}%",
                    fill=(120, 255, 160),
                )
                draw.text(
                    (x, top + TILE - 20),
                    f"area {stats['envelope_area_frac'] * 100:.0f}%  "
                    f"uniq {stats['unique_target_patches']}",
                    fill=(120, 255, 160),
                )
        panel.save(args.output / f"{mask_path.stem}_expansion.png", optimize=True)

    def mean_by(expansion: float, key: str) -> float:
        values = [r[key] for r in records if r["expansion"] == expansion]
        return float(np.mean(values))

    summary = {
        str(expansion): {
            "achieved_expansion": round(mean_by(expansion, "achieved_expansion"), 4),
            "envelope_area_frac": round(mean_by(expansion, "envelope_area_frac"), 4),
            "grid_coverage": round(mean_by(expansion, "grid_coverage"), 4),
            "target_on_envelope_frac": round(
                mean_by(expansion, "target_on_envelope_frac"), 4
            ),
            "target_background_frac": round(
                mean_by(expansion, "target_background_frac"), 4
            ),
            "mean_target_occupancy": round(
                mean_by(expansion, "mean_target_occupancy"), 4
            ),
            "unique_target_patches": round(
                mean_by(expansion, "unique_target_patches"), 2
            ),
            "context_patches": round(mean_by(expansion, "context_patches"), 2),
        }
        for expansion in EXPANSIONS
    }
    (args.output / "summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
