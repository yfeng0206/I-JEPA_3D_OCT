#!/usr/bin/env python
"""Show centre-anchored masking with a hard floor on retinal context.

Plain centre-anchoring puts every target-block centre on retina.  That gives
the best purity of any full-size policy but masks ~90% of the retina, leaving
the encoder almost nothing anatomical to predict from.  This renders the
constrained variant, which enforces a floor in two stages:

    1. retry  -- redraw the placement (sizes held fixed) up to N times and take
                 the first that leaves enough retina visible;
    2. trim   -- if every retry fails, peel outer rows/columns off the blocks
                 until the floor is met.

Columns show the placement before and after the constraint so the cost of the
constraint is visible, not just asserted.

Slices are chosen to be hard on purpose: near-empty volume edges, invalid
guides, and mid-volume optic-nerve-head slices.

Usage:
    python scripts/center_constrained_preview.py --min-visible 0.20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    "mirage_method_sweep",
    Path(_PROJECT_ROOT) / "scripts" / "mirage_method_sweep.py",
)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["mirage_method_sweep"] = sweep
_spec.loader.exec_module(sweep)

from src.guides.mirage_envelope import patch_occupancy, unpack_guides  # noqa: E402
from src.guides.tissue_truth import (  # noqa: E402
    patch_coverage,
    tissue_pixels_noise_band,
)

CROP, PATCH, GRID, NATIVE = sweep.CROP, sweep.PATCH, sweep.GRID, sweep.NATIVE
TILE, HEADER, LABEL = 256, 74, 210
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\center_constrained_preview")

# Deliberately awkward cases.  data_00289 has 15 slices whose guide failed
# validation outright; data_00231 s199 is a volume-edge slice with too little
# tissue to guide.  The s95-s105 entries are mid-volume, where the optic nerve
# head splits the retina into two disconnected sections.
#
# Note: data_08569 (the near-black slice from the MIRAGE transfer folder) is a
# *Test* volume; Training runs to data_06000 only, so equivalents are used.
HARD_SLICES = [
    ("data_00289", 0, "guide INVALID, near-empty"),
    ("data_00289", 2, "guide INVALID, near-empty"),
    ("data_00289", 100, "mid-volume, same eye for contrast"),
    ("data_00231", 199, "volume edge, guide INVALID"),
    ("data_00231", 100, "mid-volume, same eye for contrast"),
    ("data_02093", 100, "mid-volume, ONH split"),
    ("data_02093", 104, "mid-volume, ONH split"),
    ("data_05103", 199, "volume edge"),
    ("data_05103", 100, "mid-volume"),
    ("data_03235", 100, "mid-volume"),
]


def overlay(image, retina, targets=(), context=None, colour=(235, 60, 60)):
    """Green = MIRAGE retina, red = masked targets, dimmed = hidden."""
    base = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    green = np.kron(retina, np.ones((PATCH, PATCH), dtype=bool))
    base[green] = base[green] * 0.68 + np.array([40.0, 210.0, 90.0]) * 0.32

    target_grid = np.zeros((GRID, GRID), dtype=bool)
    for index in targets:
        target_grid[index // GRID, index % GRID] = True
    target_px = np.kron(target_grid, np.ones((PATCH, PATCH), dtype=bool))

    if context is not None:
        visible = np.zeros((GRID, GRID), dtype=bool)
        for index in context:
            visible[index // GRID, index % GRID] = True
        seen = np.kron(visible, np.ones((PATCH, PATCH), dtype=bool))
        base[~(seen | target_px)] *= 0.55

    base[target_px] = base[target_px] * 0.55 + np.asarray(colour, float) * 0.45
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-visible", type=float, default=0.20)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    module = sweep.preview_module()
    method = sweep.Method(
        "preview", "centre constrained", "center_constrained", 0.50, 0,
        min_retina_visible=args.min_visible,
    )
    generator = sweep.make_generator(method)
    generator.set_epoch(30)

    titles = [
        "Original OCT",
        "MIRAGE retina (guide)",
        "A. centre-anchored (no floor)",
        f"B. + floor {args.min_visible:.0%} (retry, then trim)",
        "what the encoder still sees (B)",
    ]

    rows, records = [], []
    for volume, slice_index, note in HARD_SLICES:
        guide_path = sweep.GUIDE_DIR / f"{volume}.npz"
        if not guide_path.exists():
            print(f"skip {volume}: no guide")
            continue
        with np.load(guide_path, allow_pickle=False) as cache:
            slice_indices = cache["slice_indices"].astype(int)
            hits = np.flatnonzero(slice_indices == slice_index)
            if hits.size == 0:
                print(f"skip {volume} s{slice_index}: not in guide")
                continue
            slot = int(hits[0])
            envelope = unpack_guides(
                cache["packed_envelopes"][slot:slot + 1], (NATIVE, NATIVE)
            )[0]
            guide_valid = bool(cache["valid"][slot])
        with np.load(sweep.DATA_DIR / f"{volume}.npz", allow_pickle=False) as data:
            image = data["oct_bscans"][slice_index]

        seed = args.seed + slot
        image_crop, (guide_crop,), _box = sweep.paired_crop(
            image, [envelope], np.random.default_rng(seed)
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)
        retina = occupancy >= 0.5
        retina_total = max(int(retina.sum()), 1)
        tissue = patch_coverage(
            tissue_pixels_noise_band(image_crop), PATCH
        ) >= 0.5

        plain = module.sample_center_anchored(generator, occupancy, seed)
        constrained = module.sample_center_anchored_constrained(
            generator, occupancy, seed,
            min_retina_visible=args.min_visible,
            max_attempts=args.max_attempts,
        )
        context, _ = module.context_patches(
            generator, constrained["union"], seed
        )

        def visible_fraction(union):
            covered = np.zeros(GRID * GRID, dtype=bool)
            covered[sorted(union)] = True
            return float((retina.reshape(-1) & ~covered).sum()) / retina_total

        record = {
            "volume": volume,
            "slice": slice_index,
            "note": note,
            "guide_valid": guide_valid,
            "retina_patches": int(retina.sum()),
            "tissue_patches": int(tissue.sum()),
            "plain_visible": round(visible_fraction(plain["union"]), 4),
            "plain_targets": len(plain["union"]),
            "constrained_visible": round(
                visible_fraction(constrained["union"]), 4
            ),
            "constrained_targets": len(constrained["union"]),
            "attempts": constrained["attempts"],
            "trim_steps": constrained["trim_steps"],
            "trimmed": constrained["trimmed"],
            "floor_met": constrained["floor_met"],
            "smallest_block": constrained["smallest_block"],
        }
        records.append(record)
        rows.append((record, image_crop, retina, plain, constrained, context))

    canvas = Image.new(
        "RGB",
        (LABEL + TILE * len(titles), HEADER + TILE * len(rows) + 10),
        (18, 18, 18),
    )
    pen = ImageDraw.Draw(canvas)
    pen.text(
        (8, 6),
        "Centre-anchored masking with a hard floor on retinal context.  "
        "GREEN = MIRAGE retina.  RED = masked target blocks.  "
        "DIMMED = hidden from the encoder.",
        fill=(240, 240, 240),
    )
    pen.text(
        (8, 22),
        f"Column A anchors every block centre on retina and masks nearly all of "
        f"it.  Column B enforces >= {args.min_visible:.0%} of retina left "
        f"visible: first by redrawing the placement (block sizes held fixed), "
        f"and if that fails,",
        fill=(205, 205, 205),
    )
    pen.text(
        (8, 38),
        "by trimming outer rows/columns off the blocks.  'trim' in the label "
        "means the retries were exhausted and the forced trim ran.",
        fill=(205, 205, 205),
    )
    for column, name in enumerate(titles):
        pen.text((LABEL + TILE * column + 6, HEADER - 16), name,
                 fill=(225, 225, 225))

    for index, (record, image_crop, retina, plain, constrained, context) in \
            enumerate(rows):
        top = HEADER + TILE * index
        pen.text((8, top + 6), f"{record['volume']} s{record['slice']}",
                 fill=(240, 240, 240))
        pen.text((8, top + 22), record["note"], fill=(170, 170, 170))
        pen.text((8, top + 40),
                 "guide OK" if record["guide_valid"] else "guide INVALID",
                 fill=(120, 220, 140) if record["guide_valid"] else (255, 120, 120))
        pen.text((8, top + 58), f"retina {record['retina_patches']}/256",
                 fill=(120, 220, 140))
        pen.text((8, top + 76),
                 f"A leaves {record['plain_visible']:.0%}",
                 fill=(255, 150, 150))
        pen.text((8, top + 94),
                 f"B leaves {record['constrained_visible']:.0%}",
                 fill=(150, 230, 255) if record["floor_met"] else (255, 180, 90))
        pen.text((8, top + 112), f"attempts {record['attempts']}",
                 fill=(200, 200, 200))
        pen.text(
            (8, top + 130),
            f"TRIM {record['trim_steps']} lines" if record["trimmed"]
            else "no trim needed",
            fill=(255, 200, 90) if record["trimmed"] else (150, 150, 150),
        )
        pen.text(
            (8, top + 148),
            "floor MET" if record["floor_met"] else "floor NOT met (block size floor)",
            fill=(120, 220, 140) if record["floor_met"] else (255, 150, 150),
        )
        pen.text((8, top + 166),
                 f"targets {record['plain_targets']} -> "
                 f"{record['constrained_targets']}",
                 fill=(200, 200, 200))
        pen.text((8, top + 184),
                 f"smallest block {record['smallest_block']}p",
                 fill=(180, 180, 180))

        canvas.paste(Image.fromarray(image_crop).convert("RGB"), (LABEL, top))
        canvas.paste(overlay(image_crop, retina), (LABEL + TILE, top))
        canvas.paste(
            overlay(image_crop, retina, plain["union"]), (LABEL + TILE * 2, top)
        )
        canvas.paste(
            overlay(image_crop, retina, constrained["union"],
                    colour=(90, 170, 255)),
            (LABEL + TILE * 3, top),
        )
        canvas.paste(
            overlay(image_crop, retina, constrained["union"], context,
                    colour=(90, 170, 255)),
            (LABEL + TILE * 4, top),
        )

    path = args.output / f"center_floor_{int(args.min_visible * 100):02d}.png"
    canvas.save(path)
    (args.output / f"center_floor_{int(args.min_visible * 100):02d}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"wrote {path}  {canvas.size}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
