#!/usr/bin/env python
"""Show why rendered target masks are not four clean rectangles.

The sampler picks four rectangular target blocks.  The collator then has to
stack them into a fixed-size tensor, so it truncates every block to the length
of the shortest one:

    curriculum.py:1196   global_min_pred = min(t.numel() for group ... )
    curriculum.py:1202   torch.stack([t[:global_min_pred] for t in group])

Block indices are row-major sorted (``curriculum.py:697``), so the truncation
removes patches from the *bottom row* of the larger blocks.  A 6x6 block cut to
35 patches loses one corner cell and stops being a rectangle.

This is why the panels show staircase edges that four rectangles could not
produce.  It affects the random, oracle and MIRAGE arms identically -- both
``multiblock.py:129`` and ``curriculum.py:965`` draw four *independent* block
sizes per batch -- so arm comparisons stay valid, but the nominal mask is not
the delivered mask.

Usage:
    python scripts/collator_truncation_demo.py
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
sys.modules["mirage_method_sweep"] = sweep
_spec.loader.exec_module(sweep)

from src.guides.mirage_envelope import (  # noqa: E402
    dilate_patch_grid,
    patch_occupancy,
    unpack_guides,
)

CROP, PATCH, GRID, NATIVE = sweep.CROP, sweep.PATCH, sweep.GRID, sweep.NATIVE
TILE, HEADER, LABEL = 256, 76, 190
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\collator_truncation")
PALETTE = [(235, 70, 70), (90, 180, 255), (250, 200, 70), (140, 230, 140)]


def draw_blocks(image, blocks, index_sets):
    """Colour each target block separately so overlaps and clipping show."""
    canvas = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    for order, indices in enumerate(index_sets):
        grid = np.zeros(GRID * GRID, dtype=bool)
        if len(indices):
            grid[list(indices)] = True
        pixels = np.kron(grid.reshape(GRID, GRID),
                         np.ones((PATCH, PATCH), dtype=bool))
        colour = np.asarray(PALETTE[order % len(PALETTE)], dtype=np.float32)
        canvas[pixels] = canvas[pixels] * 0.45 + colour * 0.55
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    pen = ImageDraw.Draw(out)
    for line in range(0, CROP + 1, PATCH):
        pen.line([(line, 0), (line, CROP)], fill=(70, 70, 70), width=1)
        pen.line([(0, line), (CROP, line)], fill=(70, 70, 70), width=1)
    return out


def is_rectangle(indices):
    if not len(indices):
        return True
    rows = sorted({i // GRID for i in indices})
    cols = sorted({i % GRID for i in indices})
    contiguous = rows == list(range(rows[0], rows[-1] + 1)) and \
        cols == list(range(cols[0], cols[-1] + 1))
    return contiguous and len(rows) * len(cols) == len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    method = sweep.Method("t", "MIRAGE thr 0.25 +1 dilate", "mirage", 0.25, 1)
    generator = sweep.make_generator(method)
    generator.set_epoch(30)

    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    picks = random.Random(args.seed).sample(guides, args.rows)

    titles = ["Original OCT", "A. the 4 blocks the sampler chose",
              "B. what the collator delivers", "difference (patches dropped)"]
    canvas = Image.new(
        "RGB",
        (LABEL + TILE * len(titles), HEADER + TILE * args.rows + 10),
        (18, 18, 18),
    )
    pen = ImageDraw.Draw(canvas)
    pen.text((8, 6),
             "Why the masks are not four clean rectangles.  Each target block "
             "has its own colour.  The sampler picks four rectangles (A); the "
             "collator truncates them all to the",
             fill=(240, 240, 240))
    pen.text((8, 22),
             "length of the shortest, removing patches from the BOTTOM ROW of "
             "the larger ones (B), which produces the staircase edges.  "
             "Indices are row-major sorted, so the cut is always from below.",
             fill=(205, 205, 205))
    pen.text((8, 38),
             "This is pre-existing behaviour shared by the random, oracle and "
             "MIRAGE arms, so the comparison is unaffected -- but the nominal "
             "mask is not the delivered mask.",
             fill=(205, 205, 205))
    for column, title in enumerate(titles):
        pen.text((LABEL + TILE * column + 6, HEADER - 16), title,
                 fill=(225, 225, 225))

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
        region = occupancy >= method.threshold
        placement = dilate_patch_grid(region, method.dilate)
        guide = torch.from_numpy(
            np.stack([occupancy.astype(np.float32), placement.astype(np.float32)])
        )[None]

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 31))
        _enc, pred = generator.generate(
            batch_size=1, guide_grids=guide,
            guide_valid=torch.ones(1, dtype=torch.bool),
        )
        delivered = [sorted(int(i) for i in group[0].tolist()) for group in pred]

        # Recover the nominal rectangles: each delivered block is the row-major
        # prefix of its rectangle, so the bounding box gives the original.
        nominal = []
        for indices in delivered:
            rows_used = sorted({i // GRID for i in indices})
            cols_used = sorted({i % GRID for i in indices})
            block = [
                r * GRID + c
                for r in range(rows_used[0], rows_used[-1] + 1)
                for c in range(cols_used[0], cols_used[-1] + 1)
            ]
            nominal.append(block)

        dropped = [sorted(set(n) - set(d)) for n, d in zip(nominal, delivered)]
        record = {
            "volume": guide_path.stem,
            "slice": int(slice_indices[slot]),
            "nominal_sizes": [len(b) for b in nominal],
            "delivered_sizes": [len(b) for b in delivered],
            "dropped": int(sum(len(d) for d in dropped)),
            "delivered_rectangles": sum(1 for d in delivered if is_rectangle(d)),
        }
        records.append(record)

        top = HEADER + TILE * row
        pen.text((8, top + 6), f"{record['volume']} s{record['slice']}",
                 fill=(240, 240, 240))
        pen.text((8, top + 26),
                 "chose:    " + " ".join(f"{n:2d}" for n in record["nominal_sizes"]),
                 fill=(200, 200, 200))
        pen.text((8, top + 44),
                 "delivered:" + " ".join(f"{n:2d}" for n in record["delivered_sizes"]),
                 fill=(200, 200, 200))
        pen.text((8, top + 64), f"dropped {record['dropped']} patches",
                 fill=(255, 190, 90))
        pen.text((8, top + 84),
                 f"still rectangles: {record['delivered_rectangles']}/4",
                 fill=(255, 140, 140)
                 if record["delivered_rectangles"] < 4 else (140, 220, 150))

        canvas.paste(Image.fromarray(image_crop).convert("RGB"), (LABEL, top))
        canvas.paste(draw_blocks(image_crop, None, nominal),
                     (LABEL + TILE, top))
        canvas.paste(draw_blocks(image_crop, None, delivered),
                     (LABEL + TILE * 2, top))
        canvas.paste(draw_blocks(image_crop, None, dropped),
                     (LABEL + TILE * 3, top))

    path = args.output / "collator_truncation.png"
    canvas.save(path)
    (args.output / "collator_truncation.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"wrote {path}  {canvas.size}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
