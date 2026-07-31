#!/usr/bin/env python
"""Compare oracle-band and MIRAGE-envelope target placement side by side.

Both arms use identical block geometry, so this shows whether MIRAGE's target
blocks spill onto background any more than the anatomical oracle's already do.

Example:
    python scripts/mirage_vs_oracle_masking.py --volumes 3
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
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_vs_oracle")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE, HEADER, ROW_LABEL = 256, 20, 104


def make_generator() -> CurriculumMaskGenerator:
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
            "oracle_region_frac": 0.28,
            "oracle_lateral_frac": 0.6,
        },
    )


def normalized_tensor(slice_256: np.ndarray) -> torch.Tensor:
    """Replicate the training transform's tensor + ImageNet normalisation."""
    rgb = np.repeat(slice_256[..., None].astype(np.float32) / 255.0, 3, axis=2)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


def sample_blocks(generator, weight_grid, seed, biased=True):
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    torch.manual_seed(seed)
    random.seed(seed)
    grid = torch.from_numpy(np.asarray(weight_grid, dtype=np.float32))
    blocks, union = [], set()
    for block_h, block_w in sizes:
        if biased:
            top, left = generator._sample_biased_location(block_h, block_w, grid)
        else:
            top, left = generator._sample_uniform_location(
                block_h, block_w, generator.height, generator.width
            )
        indices = generator._block_to_indices(top, left, block_h, block_w)
        blocks.append({"top": top, "left": left, "h": block_h, "w": block_w})
        union.update(indices)
    flat = np.asarray(weight_grid, dtype=np.float32).reshape(-1)
    target = flat[sorted(union)]
    return {
        "blocks": blocks,
        "unique": len(union),
        "on_region": float((target > 0).mean()),
        "background": float((target == 0).mean()),
    }


def render(slice_256, region_grid, sample, colour=(40, 220, 90)):
    base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
    region_pixels = np.kron(region_grid > 0, np.ones((PATCH, PATCH), dtype=bool))
    tint = np.array(colour, dtype=np.float32)
    alpha = 0.30 * region_pixels[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha
    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for block in sample["blocks"]:
        x0, y0 = block["left"] * PATCH, block["top"] * PATCH
        draw.rectangle(
            [x0, y0, x0 + block["w"] * PATCH - 1, y0 + block["h"] * PATCH - 1],
            fill=(255, 60, 60, 80),
            outline=(255, 60, 60),
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=3)
    parser.add_argument("--slices", type=int, nargs="+", default=[0, 50, 99])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    generator = make_generator()
    titles = ["Original OCT", "RANDOM (no guide)", "ORACLE band", "MIRAGE +0%", "MIRAGE +5%"]
    records = []

    for mask_path in sorted(MASK_DIR.glob("data_*.npz"))[: args.volumes]:
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
            seed = args.seed + slice_index

            oracle_grid = (
                generator._anatomical_prior_weight_grid_for_image(
                    normalized_tensor(slice_256)
                )
                .cpu()
                .numpy()
            )
            envelope, _v, _s = repair_union(
                build_union(masks[cache_index]), params=DEFAULT_REPAIR
            )
            grids = {"MIRAGE +0%": None, "MIRAGE +5%": None}
            for name, frac in (("MIRAGE +0%", 0.0), ("MIRAGE +5%", 0.05)):
                grown, _ = expand_envelope(envelope, frac)
                grown_256 = (
                    np.asarray(
                        Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                            (CROP, CROP), Image.Resampling.NEAREST
                        )
                    )
                    > 127
                )
                grids[name] = patch_occupancy(grown_256, patch_size=PATCH)

            random_sample = sample_blocks(
                generator, grids["MIRAGE +0%"], seed, biased=False
            )
            oracle_sample = sample_blocks(generator, oracle_grid, seed)
            mirage0 = sample_blocks(generator, grids["MIRAGE +0%"], seed)
            mirage5 = sample_blocks(generator, grids["MIRAGE +5%"], seed)

            columns = [
                Image.fromarray(slice_256, mode="L").convert("RGB"),
                render(slice_256, grids["MIRAGE +0%"], random_sample, (90, 90, 90)),
                render(slice_256, oracle_grid, oracle_sample, (80, 140, 255)),
                render(slice_256, grids["MIRAGE +0%"], mirage0),
                render(slice_256, grids["MIRAGE +5%"], mirage5),
            ]
            stats = [
                {"name": "random", **random_sample, "region": float((grids["MIRAGE +0%"] > 0).mean())},
                {"name": "oracle", **oracle_sample, "region": float((oracle_grid > 0).mean())},
                {"name": "mirage0", **mirage0, "region": float((grids["MIRAGE +0%"] > 0).mean())},
                {"name": "mirage5", **mirage5, "region": float((grids["MIRAGE +5%"] > 0).mean())},
            ]
            for entry in stats:
                records.append(
                    {
                        "volume": mask_path.stem,
                        "slice": slice_index,
                        "glaucoma": label,
                        "arm": entry["name"],
                        "on_region": round(entry["on_region"], 4),
                        "background": round(entry["background"], 4),
                        "region_coverage": round(entry["region"], 4),
                        "unique_target_patches": entry["unique"],
                    }
                )
            rows.append({"slice": slice_index, "columns": columns, "stats": stats})

        panel = Image.new(
            "RGB",
            (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.text(
            (6, 5),
            f"{mask_path.stem} glaucoma={label}   red=4 target blocks; "
            "blue=oracle band, green=MIRAGE envelope, grey=unguided",
            fill="black",
        )
        for column, name in enumerate(titles):
            draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
        for index, row in enumerate(rows):
            top = HEADER * 2 + index * TILE
            draw.text((4, top + 6), f"slice {row['slice']}", fill="black")
            for column, image in enumerate(row["columns"]):
                panel.paste(image, (ROW_LABEL + column * TILE, top))
            for column, entry in enumerate(row["stats"]):
                x = ROW_LABEL + (column + 1) * TILE + 4
                draw.text(
                    (x, top + TILE - 32),
                    f"on-region {entry['on_region'] * 100:.0f}%",
                    fill=(255, 240, 120),
                )
                draw.text(
                    (x, top + TILE - 18),
                    f"region {entry['region'] * 100:.0f}% of grid",
                    fill=(255, 240, 120),
                )
        panel.save(args.output / f"{mask_path.stem}_arms.png", optimize=True)

    summary = {}
    for arm in ("random", "oracle", "mirage0", "mirage5"):
        subset = [r for r in records if r["arm"] == arm]
        summary[arm] = {
            "mean_on_region": round(float(np.mean([r["on_region"] for r in subset])), 4),
            "mean_background": round(float(np.mean([r["background"] for r in subset])), 4),
            "mean_region_coverage": round(
                float(np.mean([r["region_coverage"] for r in subset])), 4
            ),
        }
    (args.output / "summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
