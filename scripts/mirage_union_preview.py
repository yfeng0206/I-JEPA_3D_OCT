#!/usr/bin/env python
"""Render raw vs repaired MIRAGE retinal-union guides for visual inspection.

Produces one bounded PNG panel per volume showing the first, middle and last
cached B-scan with: original OCT, raw union, repaired envelope (repair-added
pixels highlighted), overlay, and the 16x16 fractional-occupancy grid.

Example:
    python scripts/mirage_union_preview.py --volumes 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    RepairParams,
    build_union,
    patch_occupancy,
    repair_union,
)

DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_union_preview")

TILE = 200
HEADER = 18
ROW_LABEL = 74
COLUMN_TITLES = (
    "Original OCT",
    "Raw MIRAGE union",
    "Repaired envelope",
    "Overlay",
    "16x16 occupancy",
)


def to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")


def binary_image(mask: np.ndarray) -> Image.Image:
    return to_rgb(np.where(mask, 255, 0))


def repair_image(raw: np.ndarray, repaired: np.ndarray) -> Image.Image:
    """White = original union, green = pixels added by the repair."""
    rgb = np.zeros(raw.shape + (3,), dtype=np.uint8)
    rgb[raw] = (255, 255, 255)
    rgb[repaired & ~raw] = (0, 220, 90)
    return Image.fromarray(rgb, mode="RGB")


def overlay_image(slice_2d: np.ndarray, repaired: np.ndarray) -> Image.Image:
    base = np.repeat(slice_2d[..., None].astype(np.float32), 3, axis=2)
    tint = np.array([255.0, 70.0, 70.0], dtype=np.float32)
    alpha = 0.42 * repaired[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def resize_guide(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize, matching the paired training transform."""
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def occupancy_image(grid: np.ndarray) -> Image.Image:
    scaled = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L").resize(
        (TILE, TILE), Image.Resampling.NEAREST
    )
    return image.convert("RGB")


def build_panel(
    volume_id: str,
    label: int,
    rows: list,
    params: RepairParams,
) -> Image.Image:
    width = ROW_LABEL + TILE * len(COLUMN_TITLES)
    height = HEADER * 2 + TILE * len(rows)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 4),
        f"{volume_id}  glaucoma={label}  "
        f"gap<={params.max_horizontal_gap}px  jump<={params.max_boundary_jump}px",
        fill="black",
    )
    for column, title in enumerate(COLUMN_TITLES):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 2), title, fill="black")

    for row_index, row in enumerate(rows):
        top = HEADER * 2 + row_index * TILE
        draw.text((4, top + TILE // 2 - 16), row["position"], fill="black")
        draw.text((4, top + TILE // 2 - 2), f"slice {row['slice']}", fill="black")
        draw.text(
            (4, top + TILE // 2 + 12),
            f"+{row['stats']['added_area_frac'] * 100:.0f}%",
            fill="green" if row["stats"]["valid"] else "red",
        )
        for column, image in enumerate(row["images"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    args = parser.parse_args()

    params = DEFAULT_REPAIR
    args.output.mkdir(parents=True, exist_ok=True)
    mask_paths = sorted(args.mask_dir.glob("data_*.npz"))[: args.volumes]
    if not mask_paths:
        raise FileNotFoundError(f"No MIRAGE caches under {args.mask_dir}")

    summary = []
    for mask_path in mask_paths:
        source_path = args.data_dir / mask_path.name
        with np.load(mask_path, allow_pickle=False) as cache:
            masks = cache["hard_masks"]
            slice_indices = cache["slice_indices"].astype(int)
            label = int(cache["glaucoma"])
        with np.load(source_path, allow_pickle=False) as data:
            volume = data["oct_bscans"]

        positions = (
            ("first", 0),
            ("middle", masks.shape[0] // 2),
            ("last", masks.shape[0] - 1),
        )
        rows = []
        for position, cache_index in positions:
            slice_index = int(slice_indices[cache_index])
            slice_2d = volume[slice_index]
            raw = build_union(masks[cache_index])
            repaired, valid, stats = repair_union(raw, params=params)
            grid = patch_occupancy(resize_guide(repaired, 256), patch_size=16)
            rows.append(
                {
                    "position": position,
                    "slice": slice_index,
                    "stats": stats,
                    "images": [
                        to_rgb(slice_2d),
                        binary_image(raw),
                        repair_image(raw, repaired),
                        overlay_image(slice_2d, repaired),
                        occupancy_image(grid),
                    ],
                }
            )
            summary.append(
                {
                    "volume": mask_path.stem,
                    "glaucoma": label,
                    "position": position,
                    "slice": slice_index,
                    "valid": bool(valid),
                    "grid_min": float(grid.min()),
                    "grid_max": float(grid.max()),
                    "grid_mean": float(grid.mean()),
                    **{key: round(float(value), 5) for key, value in stats.items()},
                }
            )

        panel = build_panel(mask_path.stem, label, rows, params)
        panel.save(args.output / f"{mask_path.stem}_union.png", optimize=True)

    report = {
        "repair_params": params.to_dict(),
        "volumes": len(mask_paths),
        "rows": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    raw_mean = float(np.mean([row["raw_area_frac"] for row in summary]))
    repaired_mean = float(np.mean([row["repaired_area_frac"] for row in summary]))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "panels": len(mask_paths),
                "raw_area_frac_mean": round(raw_mean, 4),
                "repaired_area_frac_mean": round(repaired_mean, 4),
                "all_valid": all(row["valid"] for row in summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
