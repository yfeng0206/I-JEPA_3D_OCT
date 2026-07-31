#!/usr/bin/env python
"""Show how patch dilation pushes target blocks off the true retina.

Dilation was meant to absorb MIRAGE's boundary error.  Once it actually took
effect it interacted badly with the retina-visibility rule: the rule rewards
leaving true retina uncovered, and a dilated region offers plenty of off-retina
places to sit, so the blocks drift onto the ring.  This renders the true region,
the dilated ring and the resulting blocks side by side so the drift is visible.

Example:
    python scripts/mirage_dilation_impact.py --dilations 0 1 2
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
    dilate_patch_grid,
    patch_occupancy,
    unpack_guides,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402

GUIDE_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training")
DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_dilation_impact")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
TILE, HEADER, ROW_LABEL = 256, 20, 104

MASK_KWARGS = dict(
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


def render(slice_256, truth, dilated, blocks, on_region):
    """Green = true MIRAGE retina, amber = dilation-only ring, red = targets."""
    base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
    truth_px = np.kron(truth, np.ones((PATCH, PATCH), dtype=bool))
    ring_px = np.kron(dilated & ~truth, np.ones((PATCH, PATCH), dtype=bool))
    green = np.array([40.0, 220.0, 90.0], dtype=np.float32)
    amber = np.array([250.0, 190.0, 40.0], dtype=np.float32)
    base = base * (1 - 0.34 * truth_px[..., None]) + green * (0.34 * truth_px[..., None])
    base = base * (1 - 0.30 * ring_px[..., None]) + amber * (0.30 * ring_px[..., None])
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for top, left, height, width in blocks:
        draw.rectangle(
            [left * PATCH, top * PATCH,
             (left + width) * PATCH - 1, (top + height) * PATCH - 1],
            fill=(255, 60, 60, 80),
            outline=(255, 60, 60),
        )
    draw.text((5, 5), f"on true retina {on_region * 100:.0f}%", fill=(255, 240, 120))
    return image


def blocks_from(masks_pred, index, grid):
    out = []
    for group in masks_pred:
        indices = group[index].tolist()
        rows = [i // grid for i in indices]
        cols = [i % grid for i in indices]
        out.append((min(rows), min(cols), max(rows) - min(rows) + 1,
                    max(cols) - min(cols) + 1))
    return out


def on_region_fraction(masks_pred, index, truth):
    union = set()
    for group in masks_pred:
        union.update(group[index].tolist())
    flat = truth.reshape(-1)
    return float(np.mean([flat[i] for i in sorted(union)])) if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dilations", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Sweep the patch occupancy threshold at dilation 0 instead.",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(GUIDE_DIR.glob("data_*.npz"))[: args.rows]
    sweep_thresholds = args.thresholds is not None
    if sweep_thresholds:
        configs = [(t, 0) for t in args.thresholds]
        titles = ["Original OCT", "RANDOM (no guide)"] + [
            f"threshold {t:.2f}" for t in args.thresholds
        ]
    else:
        configs = [(0.5, d) for d in args.dilations]
        titles = ["Original OCT", "RANDOM (no guide)"] + [
            ("no dilation" if d == 0 else f"+{d} patch dilation")
            for d in args.dilations
        ]
    collator = MaskCollator(**MASK_KWARGS)
    rows, records = [], []

    for row_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as cache:
            slot = 50
            envelope = unpack_guides(
                cache["packed_envelopes"][slot : slot + 1], (200, 200)
            )[0]
            slice_index = int(cache["slice_indices"][slot])
        with np.load(DATA_DIR / path.name, allow_pickle=False) as data:
            slice_2d = data["oct_bscans"][slice_index]
        slice_256 = np.asarray(
            Image.fromarray(slice_2d, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )
        grown = (
            np.asarray(
                Image.fromarray(envelope.astype(np.uint8) * 255, mode="L").resize(
                    (CROP, CROP), Image.Resampling.NEAREST
                )
            )
            > 127
        )
        occupancy = patch_occupancy(grown, patch_size=PATCH)
        reference = occupancy >= 0.5  # fixed reference for the green overlay

        seed = args.seed + row_index
        images = [torch.zeros(3, CROP, CROP) for _ in range(1)]
        torch.manual_seed(seed)
        random.seed(seed)
        _i, _enc_r, pred_r = collator(images)
        random_on = float(
            np.mean([occupancy.reshape(-1)[i]
                     for i in sorted({j for g in pred_r for j in g[0].tolist()})])
        )
        columns = [
            Image.fromarray(slice_256, mode="L").convert("RGB"),
            render(slice_256, reference, reference,
                   blocks_from(pred_r, 0, GRID), random_on),
        ]
        stats = [{"config": "random", "on_region": random_on,
                  "region_cells": int(reference.sum())}]

        for threshold, dilation in configs:
            region = occupancy >= threshold
            placement = dilate_patch_grid(region, dilation)
            guide = torch.from_numpy(
                np.stack([occupancy.astype(np.float32),
                          placement.astype(np.float32)])
            )[None]
            generator = CurriculumMaskGenerator(
                curriculum_cfg={
                    "mode": "mirage_envelope", "T_warm": 25, "T_total": 30,
                    "r_max": 1.0, "mirage_min_block_fill": 0.40,
                    "mirage_min_retina_visible": 0.25,
                    "mirage_occupancy_threshold": threshold,
                    "mirage_spread": False,
                },
                **MASK_KWARGS,
            )
            generator.set_epoch(30)
            torch.manual_seed(seed)
            random.seed(seed)
            _enc, pred = generator.generate(
                batch_size=1, guide_grids=guide,
                guide_valid=torch.ones(1, dtype=torch.bool),
            )
            union = sorted({j for g in pred for j in g[0].tolist()})
            # Threshold-free score: mean true retinal coverage of the targets.
            on_region = float(np.mean([occupancy.reshape(-1)[i] for i in union]))
            columns.append(
                render(slice_256, reference, placement,
                       blocks_from(pred, 0, GRID), on_region)
            )
            stats.append({
                "config": f"thr{threshold}_dil{dilation}",
                "on_region": on_region,
                "region_cells": int(region.sum()),
                "placement_cells": int(placement.sum()),
            })
            records.append({"volume": path.stem, "slice": slice_index, **stats[-1]})
        rows.append({"name": f"{path.stem} s{slice_index}", "columns": columns,
                     "stats": stats})

    panel = Image.new(
        "RGB",
        (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        "GREEN = MIRAGE retina at threshold 0.50   AMBER = admitted only by this "
        "config   RED = target blocks.  Number = mean true retinal coverage of "
        "the target patches (higher is better).",
        fill="black",
    )
    for column, name in enumerate(titles):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["name"], fill="black")
        draw.text((4, top + 24),
                  f"retina {row['stats'][0]['region_cells']}/256", fill="black")
        for column, image in enumerate(row["columns"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
        for column, entry in enumerate(row["stats"]):
            if "placement_cells" in entry:
                x = ROW_LABEL + (column + 1) * TILE + 4
                draw.text((x, top + TILE - 18),
                          f"region {entry['region_cells']}/256",
                          fill=(255, 210, 120))
    name = "threshold_impact.png" if sweep_thresholds else "dilation_impact.png"
    panel.save(args.output / name, optimize=True)
    (args.output / "summary.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output / name)}, indent=2))


if __name__ == "__main__":
    main()
