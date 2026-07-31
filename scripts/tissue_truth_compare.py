#!/usr/bin/env python
"""Render competing tissue-truth definitions side by side for manual judging.

The masking sweep is only as good as its notion of "real tissue".  The first
reference thresholded patch means with Otsu and visibly missed tissue, so this
script draws every candidate reference on identical crops and lets a human
decide which one matches what the eye sees.

Columns:
    Original OCT
    patch-mean Otsu      -- the original, expected to under-count
    pixel Otsu           -- fixes edge patches
    noise-band pixels    -- pixel-level segmentation, before patch rounding
    noise-band patches   -- the same after 50% patch coverage
    + MIRAGE raw union   -- noise-band unioned with raw MIRAGE (no dilation)

Usage:
    python scripts/tissue_truth_compare.py --rows 12 --seed 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import build_union, unpack_guides  # noqa: E402
from src.guides.tissue_truth import (  # noqa: E402
    patch_coverage,
    tissue_pixels_noise_band,
    truth_patchmean_otsu,
    truth_pixel_otsu,
)

GUIDE_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\tissue_truth_compare")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NATIVE = 200
TILE = 268
HEADER = 34
ROW_LABEL = 150


def paired_crop(image: np.ndarray, masks: list[np.ndarray], rng):
    """One RandomResizedCrop draw applied to the image and every mask."""
    height, width = image.shape
    for _ in range(10):
        area = height * width * rng.uniform(0.3, 1.0)
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        crop_w = int(round(np.sqrt(area * ratio)))
        crop_h = int(round(np.sqrt(area / ratio)))
        if crop_w <= width and crop_h <= height:
            top = int(rng.integers(0, height - crop_h + 1))
            left = int(rng.integers(0, width - crop_w + 1))
            break
    else:
        top, left, crop_h, crop_w = 0, 0, height, width
    image_crop = np.asarray(
        Image.fromarray(image[top:top + crop_h, left:left + crop_w], mode="L")
        .resize((CROP, CROP), Image.BICUBIC)
    )
    out = []
    for mask in masks:
        window = mask[top:top + crop_h, left:left + crop_w].astype(np.uint8) * 255
        out.append(
            np.asarray(
                Image.fromarray(window, mode="L").resize(
                    (CROP, CROP), Image.NEAREST
                )
            )
            > 127
        )
    return image_crop, out


def tint(image: np.ndarray, mask: np.ndarray, colour, strength: float = 0.45):
    """Overlay a mask on the greyscale crop; mask may be patch- or pixel-sized."""
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    if mask.shape[0] != image.shape[0]:
        mask = np.kron(mask, np.ones((PATCH, PATCH), dtype=bool))
    hit = mask.astype(bool)
    for channel, value in enumerate(colour):
        rgb[..., channel][hit] = (
            rgb[..., channel][hit] * (1 - strength) + value * strength
        )
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--per-panel", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="sweep the noise-floor multiplier instead of comparing families",
    )
    parser.add_argument("--prefix", default="truth")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    guides = sorted(GUIDE_DIR.glob("data_*.npz"))
    picks = rng.sample(guides, min(args.rows, len(guides)))

    k_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    if args.k_sweep:
        titles = ["Original OCT"] + [
            f"noise-band k={k:g}" for k in k_values
        ]
        colours = [None] + [(60, 230, 120)] * len(k_values)
    else:
        titles = [
            "Original OCT",
            "A. patch-mean Otsu (old)",
            "B. pixel Otsu",
            "C. noise-band pixels",
            "D. noise-band patches",
            "E. D + MIRAGE raw union",
        ]
        colours = [None, (255, 60, 60), (255, 170, 40), (60, 230, 120),
                   (60, 230, 120), (90, 170, 255)]

    records = []
    rows = []
    for guide_path in picks:
        with np.load(guide_path, allow_pickle=False) as cache:
            slice_indices = cache["slice_indices"].astype(int)
            packed = cache["packed_envelopes"]
        with np.load(MASK_DIR / guide_path.name, allow_pickle=False) as cache:
            hard = cache["hard_masks"]
        with np.load(DATA_DIR / guide_path.name, allow_pickle=False) as data:
            volume = data["oct_bscans"]

        slot = rng.randrange(len(slice_indices))
        envelope = unpack_guides(packed[slot:slot + 1], (NATIVE, NATIVE))[0]
        raw_union = build_union(hard[slot])
        image = volume[int(slice_indices[slot])]
        draw_rng = np.random.default_rng(args.seed * 977 + slot)
        image_crop, (env_crop, raw_crop) = paired_crop(
            image, [envelope, raw_union], draw_rng
        )

        truth_a = truth_patchmean_otsu(image_crop, patch=PATCH)
        truth_b = truth_pixel_otsu(image_crop, patch=PATCH)
        band_pixels = tissue_pixels_noise_band(image_crop)
        truth_c = band_pixels
        truth_d = patch_coverage(band_pixels, PATCH) >= 0.5
        truth_e = patch_coverage(band_pixels | raw_crop, PATCH) >= 0.5

        if args.k_sweep:
            masks = [None]
            counts = []
            for k in k_values:
                grid = patch_coverage(
                    tissue_pixels_noise_band(image_crop, k=k), PATCH
                ) >= 0.5
                masks.append(grid)
                counts.append(int(grid.sum()))
            record = {
                "volume": guide_path.stem,
                "slice": int(slice_indices[slot]),
                **{f"k{k:g}": c for k, c in zip(k_values, counts)},
            }
        else:
            masks = [None, truth_a, truth_b, truth_c, truth_d, truth_e]
            counts = [
                int(truth_a.sum()), int(truth_b.sum()),
                int(truth_d.sum()), int(truth_e.sum()),
            ]
            record = {
                "volume": guide_path.stem,
                "slice": int(slice_indices[slot]),
                "patchmean_otsu": int(truth_a.sum()),
                "pixel_otsu": int(truth_b.sum()),
                "noise_band": int(truth_d.sum()),
                "union_mirage": int(truth_e.sum()),
                "envelope_patches": int(
                    (patch_coverage(env_crop, PATCH) >= 0.5).sum()
                ),
            }

        rows.append(
            {
                "label": f"{guide_path.stem} s{int(slice_indices[slot])}",
                "image": image_crop,
                "masks": masks,
                "counts": counts,
            }
        )
        records.append(record)

    for index in range(0, len(rows), args.per_panel):
        block = rows[index:index + args.per_panel]
        canvas = Image.new(
            "RGB",
            (ROW_LABEL + TILE * len(titles), HEADER + 16 + TILE * len(block)),
            (18, 18, 18),
        )
        pen = ImageDraw.Draw(canvas)
        pen.text(
            (8, 6),
            "Which reference matches the tissue you actually see? "
            "A is the old one. C shows the pixel segmentation behind D. "
            "E adds raw MIRAGE (partly circular).",
            fill=(235, 235, 235),
        )
        for column, name in enumerate(titles):
            pen.text((ROW_LABEL + TILE * column + 6, HEADER - 12), name,
                     fill=(215, 215, 215))
        for row_index, row in enumerate(block):
            top = HEADER + 16 + TILE * row_index
            pen.text((8, top + 6), row["label"], fill=(235, 235, 235))
            if args.k_sweep:
                for line, (k, count) in enumerate(zip(k_values, row["counts"])):
                    pen.text((8, top + 24 + 16 * line),
                             f"k={k:g}  {count}/256", fill=(130, 240, 170))
            else:
                a, b, d, e = row["counts"]
                pen.text((8, top + 24), f"A {a}/256", fill=(255, 120, 120))
                pen.text((8, top + 40), f"B {b}/256", fill=(255, 190, 110))
                pen.text((8, top + 56), f"D {d}/256", fill=(130, 240, 170))
                pen.text((8, top + 72), f"E {e}/256", fill=(150, 200, 255))
                pen.text((8, top + 92), f"A misses {d - a} patches",
                         fill=(255, 120, 120))
            for column, mask in enumerate(row["masks"]):
                tile = (
                    Image.fromarray(row["image"]).convert("RGB")
                    if mask is None
                    else tint(row["image"], mask, colours[column])
                )
                canvas.paste(tile, (ROW_LABEL + TILE * column, top))
        path = args.output / f"{args.prefix}_{index // args.per_panel + 1:02d}.png"
        canvas.save(path)
        print(f"wrote {path}")

    (args.output / f"{args.prefix}_counts.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    keys = [k for k in records[0] if k not in ("volume", "slice")]
    summary = {
        f"mean_{key}": round(float(np.mean([r[key] for r in records])), 1)
        for key in keys
    }
    if not args.k_sweep:
        summary["mean_patches_missed_by_old"] = round(
            float(np.mean([r["noise_band"] - r["patchmean_otsu"]
                           for r in records])), 1
        )
        summary["mean_added_by_mirage"] = round(
            float(np.mean([r["union_mirage"] - r["noise_band"]
                           for r in records])), 2
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
