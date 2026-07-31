#!/usr/bin/env python
"""Render every masking method on the same randomly chosen slices.

Each row is one randomly selected (volume, slice) pair drawn from across the
training set; each column is a method applied to that identical crop, so the
comparison is like-for-like and can be judged by eye.

Truth shown is the noise-band reference: a pixel-level threshold calibrated
against the vitreous noise floor, reduced to patch occupancy.  It is derived
from the image alone, NOT from MIRAGE, so it independently answers whether a
method masks real tissue.  The earlier patch-mean Otsu reference under-counted
tissue by roughly 45% and has been retired -- see src/guides/tissue_truth.py.

Example:
    python scripts/mirage_method_panels.py --rows 12
"""

from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "mirage_method_sweep",
    Path(_PROJECT_ROOT) / "scripts" / "mirage_method_sweep.py",
)
sweep = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves its module via sys.modules.
sys.modules["mirage_method_sweep"] = sweep
_spec.loader.exec_module(sweep)

from src.guides.mirage_envelope import (  # noqa: E402
    build_union,
    patch_occupancy,
    unpack_guides,
)
from src.guides.tissue_truth import (  # noqa: E402
    patch_coverage,
    tissue_pixels_noise_band,
)

OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_method_panels")
CROP, PATCH, GRID, NATIVE = sweep.CROP, sweep.PATCH, sweep.GRID, sweep.NATIVE
TILE, HEADER, ROW_LABEL = 256, 34, 150


def overlay(image, tissue, targets=None, context=None, envelope=None):
    """Green = tissue (noise-band truth), cyan = MIRAGE envelope, red = targets."""
    base = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    if envelope is not None:
        env_px = np.kron(envelope, np.ones((PATCH, PATCH), dtype=bool))
        cyan = np.array([60.0, 190.0, 235.0], dtype=np.float32)
        base = base * (1 - 0.34 * env_px[..., None]) + cyan * (
            0.34 * env_px[..., None]
        )
    else:
        tissue_px = np.kron(tissue, np.ones((PATCH, PATCH), dtype=bool))
        green = np.array([40.0, 210.0, 90.0], dtype=np.float32)
        base = base * (1 - 0.30 * tissue_px[..., None]) + green * (
            0.30 * tissue_px[..., None]
        )
    if context is not None:
        visible = np.zeros((GRID, GRID), dtype=bool)
        for index in context:
            visible[index // GRID, index % GRID] = True
        seen = np.kron(visible, np.ones((PATCH, PATCH), dtype=bool))
        target_px = np.zeros((GRID, GRID), dtype=bool)
        for index in targets or ():
            target_px[index // GRID, index % GRID] = True
        target_px = np.kron(target_px, np.ones((PATCH, PATCH), dtype=bool))
        base[~(seen | target_px)] *= 0.55
    image_out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    if targets:
        draw = ImageDraw.Draw(image_out, "RGBA")
        grid_mask = np.zeros((GRID, GRID), dtype=bool)
        for index in targets:
            grid_mask[index // GRID, index % GRID] = True
        for row in range(GRID):
            for col in range(GRID):
                if grid_mask[row, col]:
                    draw.rectangle(
                        [col * PATCH, row * PATCH,
                         (col + 1) * PATCH - 1, (row + 1) * PATCH - 1],
                        fill=(255, 60, 60, 95),
                    )
        draw.rectangle([0, 0, CROP - 1, CROP - 1], outline=(255, 60, 60), width=1)
    return image_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-panel", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    rng = np.random.default_rng(args.seed)
    picks = []
    seen_volumes = set()
    while len(picks) < args.rows:
        index = int(rng.integers(len(guides)))
        if index in seen_volumes:      # one slice per volume: spread the sample
            continue
        seen_volumes.add(index)
        picks.append((guides[index], int(rng.integers(100))))

    generators = {m.key: sweep.make_generator(m) for m in sweep.METHODS}
    for generator in generators.values():
        if hasattr(generator, "set_epoch"):
            generator.set_epoch(30)

    titles = ["Original OCT", "Tissue truth (noise-band)",
              "MIRAGE envelope (repaired)"] + [
        m.label for m in sweep.METHODS
    ]
    records, rows = [], []
    for guide_path, slot in picks:
        with np.load(guide_path, allow_pickle=False) as cache:
            envelope = unpack_guides(
                cache["packed_envelopes"][slot:slot + 1], (NATIVE, NATIVE)
            )[0]
            slice_index = int(cache["slice_indices"][slot])
            guide_valid = bool(cache["valid"][slot])
        with np.load(sweep.DATA_DIR / guide_path.name, allow_pickle=False) as data:
            image = data["oct_bscans"][slice_index]

        seed = args.seed + slot
        crop_rng = np.random.default_rng(seed)
        image_crop, (guide_crop,), _box = sweep.paired_crop(
            image, [envelope], crop_rng
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)
        tissue = patch_coverage(
            tissue_pixels_noise_band(image_crop), PATCH
        ) >= 0.5
        tissue_flat = tissue.reshape(-1)
        tissue_count = max(int(tissue.sum()), 1)

        columns = [
            Image.fromarray(image_crop, mode="L").convert("RGB"),
            overlay(image_crop, tissue),
            overlay(image_crop, tissue, envelope=(occupancy >= 0.5)),
        ]
        cells = []
        for method in sweep.METHODS:
            targets, context, _stats = sweep.sample_method(
                method, generators[method.key], occupancy, seed,
                image_crop=image_crop,
            )
            target_list = sorted(targets)
            covered = int(tissue_flat[target_list].sum()) if target_list else 0
            retained = int(tissue_flat[sorted(context)].sum()) if context else 0
            cells.append({
                "method": method.label,
                "purity": covered / max(len(target_list), 1),
                "tissue_covered": covered / tissue_count,
                "context_retention": retained / tissue_count,
                "targets": len(target_list),
            })
            columns.append(overlay(image_crop, tissue, targets, context))
            records.append({
                "volume": guide_path.stem, "slice": slice_index,
                "guide_valid": guide_valid, **cells[-1],
            })
        rows.append({
            "name": f"{guide_path.stem} s{slice_index}",
            "valid": guide_valid,
            "tissue": int(tissue.sum()),
            "columns": columns,
            "cells": cells,
        })

    chunk = args.per_panel
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk]
        panel = Image.new(
            "RGB",
            (ROW_LABEL + TILE * len(titles), HEADER + 16 + TILE * len(block)),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.text(
            (6, 4),
            "Masking methods on identical crops.  GREEN = real tissue by the "
            "noise-band reference (pixel-level, independent of MIRAGE).  CYAN = "
            "the repaired MIRAGE envelope actually used as the guide.  RED = "
            "masked target patches.  DIMMED = hidden from the encoder.",
            fill="black",
        )
        draw.text(
            (6, 18),
            "Truth is a pixel threshold calibrated to the vitreous noise floor, "
            "then 50% patch coverage -- it keeps dim choroid and boundary "
            "patches that the retired patch-mean Otsu dropped.  "
            "purity = masked patches that are tissue | cov = tissue masked | "
            "keep = tissue still visible to encoder",
            fill="black",
        )
        for column, name in enumerate(titles):
            draw.text((ROW_LABEL + column * TILE + 4, HEADER + 2), name, fill="black")
        for index, row in enumerate(block):
            top = HEADER + 16 + index * TILE
            draw.text((4, top + 6), row["name"], fill="black")
            draw.text((4, top + 22), f"tissue {row['tissue']}/256", fill="black")
            draw.text((4, top + 38), "guide OK" if row["valid"] else "guide INVALID",
                      fill="green" if row["valid"] else "red")
            for column, image_col in enumerate(row["columns"]):
                panel.paste(image_col, (ROW_LABEL + column * TILE, top))
            for column, cell in enumerate(row["cells"]):
                x = ROW_LABEL + (column + 3) * TILE + 4
                draw.text((x, top + TILE - 32),
                          f"purity {cell['purity']:.2f}  cov {cell['tissue_covered']:.2f}",
                          fill=(255, 240, 120))
                keep = cell["context_retention"]
                draw.text((x, top + TILE - 18),
                          f"keep {keep:.2f}   targets {cell['targets']}",
                          fill=(140, 255, 170) if keep >= 0.25 else (255, 120, 120))
        name = f"methods_{start // chunk + 1:02d}.png"
        panel.save(args.output / name, optimize=True)
        print(f"wrote {args.output / name}")

    (args.output / "panels.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    summary = {}
    for method in sweep.METHODS:
        subset = [r for r in records if r["method"] == method.label]
        summary[method.label] = {
            k: round(float(np.mean([r[k] for r in subset])), 4)
            for k in ("purity", "tissue_covered", "context_retention")
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
