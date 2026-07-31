#!/usr/bin/env python
"""Show what the occupancy threshold actually does to the sampler's region.

The threshold is the least intuitive knob in the sweep, so this renders the
same slice at several thresholds: the continuous occupancy grid, then the
binary region each threshold produces.

Usage:
    python scripts/threshold_explainer.py --rows 3
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mirage_method_sweep",
    Path(_PROJECT_ROOT) / "scripts" / "mirage_method_sweep.py",
)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["mirage_method_sweep"] = sweep
_spec.loader.exec_module(sweep)

from src.guides.mirage_envelope import patch_occupancy, unpack_guides  # noqa: E402

CROP, PATCH, GRID, NATIVE = sweep.CROP, sweep.PATCH, sweep.GRID, sweep.NATIVE
TILE, HEADER, LABEL = 256, 56, 130
THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 1.00)
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\threshold_explainer")


def heat(occupancy: np.ndarray) -> Image.Image:
    """Continuous occupancy as a blue->yellow ramp."""
    grid = np.kron(occupancy, np.ones((PATCH, PATCH), dtype=np.float32))
    rgb = np.zeros(grid.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 255 * grid
    rgb[..., 1] = 210 * grid
    rgb[..., 2] = 90 * (1 - grid) + 40 * grid
    rgb[grid <= 0.001] = 22
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def binary(image: np.ndarray, region: np.ndarray) -> Image.Image:
    """The region the sampler is allowed to aim at, over the OCT."""
    mask = np.kron(region, np.ones((PATCH, PATCH), dtype=bool))
    rgb = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    tint = np.array([255.0, 210.0, 60.0], dtype=np.float32)
    rgb[mask] = rgb[mask] * 0.45 + tint * 0.55
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    picks = random.Random(args.seed).sample(guides, args.rows)

    titles = ["Original OCT", "Patch occupancy\n(0=none, 1=all retina)"] + [
        f"threshold {t:.2f}" for t in THRESHOLDS
    ]
    canvas = Image.new(
        "RGB",
        (LABEL + TILE * len(titles), HEADER + TILE * args.rows + 10),
        (18, 18, 18),
    )
    pen = ImageDraw.Draw(canvas)
    pen.text(
        (8, 6),
        "What the occupancy threshold does.  MIRAGE gives a pixel map; on the "
        "16x16 patch grid each patch gets an occupancy = fraction of it that is "
        "retina.",
        fill=(240, 240, 240),
    )
    pen.text(
        (8, 22),
        "The threshold picks which patches the sampler may aim at.  LOW = "
        "generous, big region, sloppy edges.  HIGH = strict, small region, only "
        "solid retina.  Yellow = allowed region.",
        fill=(200, 200, 200),
    )
    for column, name in enumerate(titles):
        for line, text in enumerate(name.split("\n")):
            pen.text((LABEL + TILE * column + 6, HEADER - 26 + 12 * line),
                     text, fill=(220, 220, 220))

    for row, guide_path in enumerate(picks):
        with np.load(guide_path, allow_pickle=False) as cache:
            slot = random.Random(args.seed + row).randrange(
                len(cache["slice_indices"])
            )
            envelope = unpack_guides(
                cache["packed_envelopes"][slot:slot + 1], (NATIVE, NATIVE)
            )[0]
            slice_index = int(cache["slice_indices"][slot])
        with np.load(sweep.DATA_DIR / guide_path.name, allow_pickle=False) as data:
            image = data["oct_bscans"][slice_index]

        image_crop, (guide_crop,), _ = sweep.paired_crop(
            image, [envelope], np.random.default_rng(args.seed + row)
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)

        top = HEADER + TILE * row
        pen.text((8, top + 6), f"{guide_path.stem}\ns{slice_index}",
                 fill=(235, 235, 235))
        canvas.paste(Image.fromarray(image_crop).convert("RGB"), (LABEL, top))
        canvas.paste(heat(occupancy), (LABEL + TILE, top))
        for index, threshold in enumerate(THRESHOLDS):
            region = occupancy >= threshold
            canvas.paste(
                binary(image_crop, region), (LABEL + TILE * (index + 2), top)
            )
            pen.text(
                (LABEL + TILE * (index + 2) + 6, top + TILE - 16),
                f"{int(region.sum())}/256 patches allowed",
                fill=(255, 230, 120),
            )

    path = args.output / "threshold_explainer.png"
    canvas.save(path)
    print(f"wrote {path}  {canvas.size}")


if __name__ == "__main__":
    main()
