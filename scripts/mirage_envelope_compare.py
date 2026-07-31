#!/usr/bin/env python
"""Compare retinal-envelope repair variants on the worst-connected slices.

Scans a sample of cached MIRAGE slices, ranks them by how poorly the raw union
holds together, then renders the hardest cases under both envelope modes so the
fill strategy can be chosen from evidence rather than intuition.

Columns per row: original OCT | raw union | column_gap envelope |
section envelope | section overlay | 16x16 occupancy (section).

Example:
    python scripts/mirage_envelope_compare.py --volumes 40 --worst 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    build_union,
    patch_occupancy,
    repair_union,
)

DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_envelope_compare")

TILE = 200
HEADER = 18
ROW_LABEL = 96
COLUMN_TITLES = (
    "Original OCT",
    "Raw union",
    "A: no spike guard",
    "B: spike guard (default)",
    "B overlay",
    "16x16 occupancy",
)

COLUMN_GAP = replace(DEFAULT_REPAIR, min_fill_width=1)
SECTION = DEFAULT_REPAIR


def to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")


def repair_image(raw: np.ndarray, repaired: np.ndarray) -> Image.Image:
    rgb = np.zeros(raw.shape + (3,), dtype=np.uint8)
    rgb[raw & repaired] = (255, 255, 255)
    rgb[repaired & ~raw] = (0, 220, 90)
    rgb[raw & ~repaired] = (235, 60, 60)
    return Image.fromarray(rgb, mode="RGB")


def overlay_image(slice_2d: np.ndarray, repaired: np.ndarray) -> Image.Image:
    base = np.repeat(slice_2d[..., None].astype(np.float32), 3, axis=2)
    tint = np.array([255.0, 70.0, 70.0], dtype=np.float32)
    alpha = 0.42 * repaired[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def resize_guide(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize((size, size), Image.Resampling.NEAREST)) > 127


def occupancy_image(grid: np.ndarray) -> Image.Image:
    scaled = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    return (
        Image.fromarray(scaled, mode="L")
        .resize((TILE, TILE), Image.Resampling.NEAREST)
        .convert("RGB")
    )


def horizontal_sections(mask: np.ndarray) -> int:
    """Number of horizontally separated retinal sections."""
    occupied = mask.any(axis=0)
    return int(np.count_nonzero(occupied[1:] & ~occupied[:-1]) + occupied[0])


def vertical_runs(mask: np.ndarray) -> float:
    """Mean number of separate foreground runs per occupied column.

    A clean envelope has ~1.0; higher means the column is still broken up.
    """
    occupied_cols = np.flatnonzero(mask.any(axis=0))
    if occupied_cols.size == 0:
        return 0.0
    runs = []
    for col in occupied_cols:
        column = mask[:, col]
        runs.append(int(np.count_nonzero(column[1:] & ~column[:-1]) + column[0]))
    return float(np.mean(runs))


def evaluate(labels: np.ndarray) -> dict:
    raw = build_union(labels)
    gap_mask, gap_valid, gap_stats = repair_union(raw, params=COLUMN_GAP)
    sec_mask, sec_valid, sec_stats = repair_union(raw, params=SECTION)
    grid = patch_occupancy(resize_guide(sec_mask, 256), patch_size=16)
    return {
        "raw": raw,
        "gap_mask": gap_mask,
        "sec_mask": sec_mask,
        "grid": grid,
        "raw_runs": vertical_runs(raw),
        "gap_runs": vertical_runs(gap_mask),
        "sec_runs": vertical_runs(sec_mask),
        "raw_area": float(raw.mean()),
        "gap_area": float(gap_mask.mean()),
        "sec_area": float(sec_mask.mean()),
        "gap_sections": horizontal_sections(gap_mask),
        "sec_sections": horizontal_sections(sec_mask),
        "gap_valid": bool(gap_valid),
        "sec_valid": bool(sec_valid),
        "gap_dropped": gap_stats["components_dropped"],
        "sec_dropped": sec_stats["components_dropped"],
    }


def build_panel(title: str, rows: list) -> Image.Image:
    width = ROW_LABEL + TILE * len(COLUMN_TITLES)
    height = HEADER * 2 + TILE * len(rows)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), title, fill="black")
    for column, name in enumerate(COLUMN_TITLES):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 2), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["label"], fill="black")
        draw.text(
            (4, top + 22),
            f"runs raw {row['raw_runs']:.2f}",
            fill="black",
        )
        draw.text((4, top + 36), f"gap {row['gap_runs']:.2f}", fill="black")
        draw.text((4, top + 50), f"sec {row['sec_runs']:.2f}", fill="black")
        draw.text(
            (4, top + 68),
            f"area {row['gap_area'] * 100:.0f}/{row['sec_area'] * 100:.0f}%",
            fill="black",
        )
        draw.text(
            (4, top + 86),
            f"sections {row['gap_sections']}/{row['sec_sections']}",
            fill="blue",
        )
        for column, image in enumerate(row["images"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=40)
    parser.add_argument("--worst", type=int, default=6)
    parser.add_argument("--slices", type=int, nargs="+", default=[0, 25, 50, 75, 99])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    mask_paths = sorted(MASK_DIR.glob("data_*.npz"))[: args.volumes]
    if not mask_paths:
        raise FileNotFoundError(f"No MIRAGE caches under {MASK_DIR}")

    records = []
    for mask_path in mask_paths:
        with np.load(mask_path, allow_pickle=False) as cache:
            masks = cache["hard_masks"]
            slice_indices = cache["slice_indices"].astype(int)
            label = int(cache["glaucoma"])
        for cache_index in args.slices:
            result = evaluate(masks[cache_index])
            records.append(
                {
                    "volume": mask_path.stem,
                    "glaucoma": label,
                    "cache_index": int(cache_index),
                    "slice": int(slice_indices[cache_index]),
                    **{
                        key: value
                        for key, value in result.items()
                        if not isinstance(value, np.ndarray)
                    },
                }
            )

    def aggregate(key: str) -> float:
        return float(np.mean([record[key] for record in records]))

    summary = {
        "scanned_volumes": len(mask_paths),
        "scanned_slices": len(records),
        "mean_raw_runs": round(aggregate("raw_runs"), 4),
        "mean_column_gap_runs": round(aggregate("gap_runs"), 4),
        "mean_section_runs": round(aggregate("sec_runs"), 4),
        "mean_raw_area": round(aggregate("raw_area"), 4),
        "mean_column_gap_area": round(aggregate("gap_area"), 4),
        "mean_section_area": round(aggregate("sec_area"), 4),
        "identical_masks_frac": round(
            float(
                np.mean(
                    [
                        1.0 if record["gap_area"] == record["sec_area"] else 0.0
                        for record in records
                    ]
                )
            ),
            4,
        ),
        "column_gap_invalid": int(sum(1 for r in records if not r["gap_valid"])),
        "section_invalid": int(sum(1 for r in records if not r["sec_valid"])),
        "multi_section_slices": int(
            sum(1 for r in records if r["sec_sections"] > 1)
        ),
    }

    worst = sorted(records, key=lambda r: -r["sec_runs"])[: args.worst]
    rows = []
    for record in worst:
        mask_path = MASK_DIR / f"{record['volume']}.npz"
        with np.load(mask_path, allow_pickle=False) as cache:
            labels = cache["hard_masks"][record["cache_index"]]
        with np.load(DATA_DIR / f"{record['volume']}.npz", allow_pickle=False) as data:
            slice_2d = data["oct_bscans"][record["slice"]]
        result = evaluate(labels)
        rows.append(
            {
                "label": f"{record['volume']} s{record['slice']} g={record['glaucoma']}",
                "raw_runs": result["raw_runs"],
                "gap_runs": result["gap_runs"],
                "sec_runs": result["sec_runs"],
                "gap_area": result["gap_area"],
                "sec_area": result["sec_area"],
                "gap_sections": result["gap_sections"],
                "sec_sections": result["sec_sections"],
                "images": [
                    to_rgb(slice_2d),
                    to_rgb(np.where(result["raw"], 255, 0)),
                    repair_image(result["raw"], result["gap_mask"]),
                    repair_image(result["raw"], result["sec_mask"]),
                    overlay_image(slice_2d, result["sec_mask"]),
                    occupancy_image(result["grid"]),
                ],
            }
        )

    build_panel(
        "Worst-connected slices: white=raw kept, GREEN=repair added, "
        "RED=raw signal deleted by the filter (lower is better)",
        rows,
    ).save(args.output / "worst_connected.png", optimize=True)

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "column_gap_params": COLUMN_GAP.to_dict(),
                "section_params": SECTION.to_dict(),
                "worst": [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("images",)
                    }
                    for record in worst
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
