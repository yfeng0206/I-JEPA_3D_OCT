#!/usr/bin/env python
"""Render each target block in its own colour, before and after the swap.

The constant-budget swap proposal keeps four target blocks and a fixed token
count, but it does so by punching holes in the rectangles and scattering the
displaced tokens elsewhere.  Colouring each block separately is the quickest way
to see whether "four blocks" still means anything after the swap, or whether the
targets have become four clouds of confetti.

Columns:
    Original OCT
    MIRAGE retina (the guide)
    A. centre-anchored, four solid rectangles
    B. after constant-budget swap to a 20% visible-retina floor
    C. what moved: removed patches vs added patches

Usage:
    python scripts/swap_block_identity.py --rows 5
"""

from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "swap_budget_eval",
    Path(_PROJECT_ROOT) / "scripts" / "swap_budget_eval.py",
)
swapmod = importlib.util.module_from_spec(_spec)
sys.modules["swap_budget_eval"] = swapmod
_spec.loader.exec_module(swapmod)

sweep = swapmod.sweep
from src.guides.mirage_envelope import patch_occupancy, unpack_guides  # noqa: E402

GRID, PATCH, CROP, NATIVE = sweep.GRID, sweep.PATCH, sweep.CROP, sweep.NATIVE
TILE, HEADER, LABEL = 256, 92, 200
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\swap_block_identity")

# One colour per target block, chosen to stay distinct over grey OCT.
BLOCK_COLOURS = [
    (235, 70, 70),      # red
    (90, 175, 255),     # blue
    (250, 205, 70),     # amber
    (140, 235, 150),    # green
]
RETINA_COLOUR = (60, 200, 110)


def paint(image, groups, colours, retina=None, grid_lines=True):
    canvas = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    if retina is not None:
        pixels = np.kron(retina.reshape(GRID, GRID),
                         np.ones((PATCH, PATCH), dtype=bool))
        canvas[pixels] = (
            canvas[pixels] * 0.72
            + np.asarray(RETINA_COLOUR, dtype=np.float32) * 0.28
        )
    for indices, colour in zip(groups, colours):
        if not len(indices):
            continue
        grid = np.zeros(GRID * GRID, dtype=bool)
        grid[list(indices)] = True
        pixels = np.kron(grid.reshape(GRID, GRID),
                         np.ones((PATCH, PATCH), dtype=bool))
        canvas[pixels] = (
            canvas[pixels] * 0.35 + np.asarray(colour, dtype=np.float32) * 0.65
        )
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    if grid_lines:
        pen = ImageDraw.Draw(out)
        for line in range(0, CROP + 1, PATCH):
            pen.line([(line, 0), (line, CROP)], fill=(64, 64, 64))
            pen.line([(0, line), (CROP, line)], fill=(64, 64, 64))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--min-keep", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    module = sweep.preview_module()
    generator = sweep.make_generator(sweep.Method("m", "m", "mirage", 0.25, 0))
    generator.set_epoch(30)

    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    picks = random.Random(args.seed).sample(guides, args.rows)

    titles = [
        "Original OCT",
        "MIRAGE retina (guide)",
        "A. 4 solid blocks (centre-anchored)",
        "B. after constant-budget swap",
        "C. removed (cyan) vs added (magenta)",
    ]
    canvas = Image.new(
        "RGB",
        (LABEL + TILE * len(titles), HEADER + TILE * args.rows + 10),
        (18, 18, 18),
    )
    pen = ImageDraw.Draw(canvas)
    pen.text((8, 6),
             "Does 'four target blocks' survive the constant-budget swap?  "
             "Each block has its own colour: red, blue, amber, green.",
             fill=(245, 245, 245))
    pen.text((8, 24),
             "A shows the four rectangles as sampled.  B shows them after "
             "uncovering retinal patches and re-covering the same number "
             "elsewhere to force 20% visible retina.",
             fill=(205, 205, 205))
    pen.text((8, 42),
             "The token count is preserved, but the shapes are not: holes open "
             "inside the retina and the displaced tokens attach wherever room "
             "exists, often on a different block's border.",
             fill=(205, 205, 205))
    for column, title in enumerate(titles):
        pen.text((LABEL + TILE * column + 6, HEADER - 16), title,
                 fill=(228, 228, 228))

    records = []
    for row, guide_path in enumerate(picks):
        with np.load(guide_path, allow_pickle=False) as cache:
            slice_indices = cache["slice_indices"].astype(int)
            slot = random.Random(args.seed + row).randrange(len(slice_indices))
            envelope = unpack_guides(
                cache["packed_envelopes"][slot:slot + 1], (NATIVE, NATIVE)
            )[0]
        with np.load(sweep.DATA_DIR / guide_path.name, allow_pickle=False) as data:
            image = data["oct_bscans"][int(slice_indices[slot])]

        seed = args.seed + slot
        image_crop, (guide_crop,), _ = sweep.paired_crop(
            image, [envelope], np.random.default_rng(seed)
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)
        retina_flat = (occupancy >= 0.5).reshape(-1)
        retina_total = max(int(retina_flat.sum()), 1)

        centre = module.sample_center_anchored(generator, occupancy, seed)
        context_list, _ = module.context_patches(
            generator, centre["union"], seed
        )
        context_all = set(context_list) | set(centre["union"])
        blocks = [
            generator._block_to_indices(b["top"], b["left"], b["h"], b["w"])
            for b in centre["blocks"]
        ]
        union_before = set().union(*blocks)
        keep_before = sum(
            1 for i in (context_all - union_before) if retina_flat[i]
        ) / retina_total

        after, swaps, ok = swapmod.swap_to_floor(
            blocks, retina_flat, context_all, args.min_keep, random.Random(seed)
        )
        union_after = set().union(*after)
        keep_after = sum(
            1 for i in (context_all - union_after) if retina_flat[i]
        ) / retina_total
        removed = sorted(union_before - union_after)
        added = sorted(union_after - union_before)
        solid = [swapmod.is_solid(sorted(b)) for b in after]

        record = {
            "volume": guide_path.stem,
            "slice": int(slice_indices[slot]),
            "retina_patches": retina_total,
            "keep_before": round(keep_before, 4),
            "keep_after": round(keep_after, 4),
            "swaps": swaps,
            "floor_met": bool(ok and keep_after >= args.min_keep),
            "blocks_before": len(blocks),
            "blocks_after": len(after),
            "sizes_before": [len(b) for b in blocks],
            "sizes_after": [len(b) for b in after],
            "solid_after": int(sum(solid)),
        }
        records.append(record)

        top = HEADER + TILE * row
        pen.text((8, top + 6), f"{record['volume']} s{record['slice']}",
                 fill=(245, 245, 245))
        pen.text((8, top + 26), f"retina {retina_total}/256",
                 fill=(120, 220, 140))
        pen.text((8, top + 44), f"A keeps {keep_before:.0%}",
                 fill=(255, 150, 150))
        pen.text((8, top + 62), f"B keeps {keep_after:.0%}",
                 fill=(150, 230, 255))
        pen.text((8, top + 80), f"{swaps} tokens swapped",
                 fill=(230, 180, 255))
        pen.text((8, top + 98),
                 "block sizes " + "/".join(str(s) for s in record["sizes_after"]),
                 fill=(200, 200, 200))
        pen.text((8, top + 116),
                 f"still rectangles: {record['solid_after']}/4",
                 fill=(140, 220, 150) if record["solid_after"] == 4
                 else (255, 140, 140))

        canvas.paste(Image.fromarray(image_crop).convert("RGB"), (LABEL, top))
        canvas.paste(
            paint(image_crop, [], [], retina=retina_flat),
            (LABEL + TILE, top),
        )
        canvas.paste(
            paint(image_crop, blocks, BLOCK_COLOURS, retina=retina_flat),
            (LABEL + TILE * 2, top),
        )
        canvas.paste(
            paint(image_crop, [sorted(b) for b in after], BLOCK_COLOURS,
                  retina=retina_flat),
            (LABEL + TILE * 3, top),
        )
        canvas.paste(
            paint(image_crop, [removed, added],
                  [(80, 230, 240), (240, 90, 230)], retina=retina_flat),
            (LABEL + TILE * 4, top),
        )

    path = args.output / "swap_block_identity.png"
    canvas.save(path)
    (args.output / "swap_block_identity.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"wrote {path}  {canvas.size}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
