#!/usr/bin/env python
"""Test the constant-budget swap proposal for forcing visible retinal context.

The proposal: centre four original-size rectangles on the MIRAGE region; if less
than ``min_keep`` of the retina survives into the encoder context, uncover the
minimum number of retinal target patches and add the same number of target
patches elsewhere, so the predictor's target budget is exactly preserved.

The claim under test is that this is a fair swap.  It is not obviously so: the
uncovered patches sit *inside* the retina, immediately adjacent to the target
patches that remain, so they enter the context as neighbours of what the
predictor must reconstruct.  I-JEPA's difficulty comes from predicting a
contiguous block from *distant* context; a block with holes in it is a
materially easier problem even at identical token count.

Measured here:

* ``keep_true``   -- retina inside the context block and unmasked;
* ``adjacency``   -- fraction of target patches with a 4-neighbour in context,
                     the leak the swap introduces;
* ``solid_frac``  -- fraction of target blocks still contiguous rectangles;
* ``budget``      -- target token count, which the proposal aims to hold fixed.

Usage:
    python scripts/swap_budget_eval.py --volumes 300 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

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

GRID, PATCH, NATIVE = sweep.GRID, sweep.PATCH, sweep.NATIVE
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\swap_budget_eval")
MIN_KEEP = 0.20


def neighbours(index):
    row, col = divmod(index, GRID)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = row + dr, col + dc
        if 0 <= r < GRID and 0 <= c < GRID:
            yield r * GRID + c


def adjacency_leak(union, context):
    """Fraction of target patches touching a context patch."""
    if not union:
        return float("nan")
    context = set(context)
    touching = sum(
        1 for i in union if any(n in context for n in neighbours(i))
    )
    return touching / len(union)


def is_solid(indices):
    if not indices:
        return True
    rows = sorted({i // GRID for i in indices})
    cols = sorted({i % GRID for i in indices})
    return (
        rows == list(range(rows[0], rows[-1] + 1))
        and cols == list(range(cols[0], cols[-1] + 1))
        and len(rows) * len(cols) == len(indices)
    )


def swap_to_floor(blocks, retina_flat, context_all, min_keep, rng):
    """Uncover retinal targets and re-cover elsewhere, holding the budget fixed.

    Uncovered patches are chosen from the retinal interior of the blocks;
    replacements are drawn from unused non-retinal patches, preferring those
    adjacent to an existing target so the additions are not isolated specks.
    """
    per_block = [set(b) for b in blocks]
    union = set().union(*per_block) if per_block else set()
    retina_total = int(retina_flat.sum())
    if retina_total == 0:
        return per_block, 0, True

    def visible():
        return sum(
            1 for i in (context_all - union) if retina_flat[i]
        ) / retina_total

    required = int(np.ceil(min_keep * retina_total))
    swapped = 0
    while visible() < min_keep:
        # Uncover a retinal target patch that is inside the context block --
        # uncovering one outside it buys nothing, since it stays invisible.
        candidates = [
            i for i in union if retina_flat[i] and i in context_all
        ]
        if not candidates:
            return per_block, swapped, False
        victim = rng.choice(sorted(candidates))

        pool = [
            i for i in range(GRID * GRID)
            if i not in union and not retina_flat[i]
        ]
        if not pool:
            return per_block, swapped, False
        adjacent = [i for i in pool if any(n in union for n in neighbours(i))]
        replacement = rng.choice(sorted(adjacent) if adjacent else sorted(pool))

        for block in per_block:
            if victim in block:
                block.discard(victim)
                block.add(replacement)
                break
        union = set().union(*per_block)
        swapped += 1
        if swapped > 4 * retina_total:      # safety valve
            return per_block, swapped, False
    return per_block, swapped, True


def evaluate_volume(task):
    guide_path, stride, seed = task
    guide_path = Path(guide_path)
    module = sweep.preview_module()
    try:
        with np.load(guide_path, allow_pickle=False) as cache:
            packed = cache["packed_envelopes"]
            slice_indices = cache["slice_indices"].astype(int)
        with np.load(sweep.DATA_DIR / guide_path.name, allow_pickle=False) as data:
            volume = data["oct_bscans"]
    except Exception as error:  # noqa: BLE001
        return {"error": f"{guide_path.name}: {error}"}

    generator = sweep.make_generator(sweep.Method("m", "m", "mirage", 0.25, 0))
    generator.set_epoch(30)
    rows = {"plain": [], "swap": [], "thr25": []}

    for slot in range(0, len(slice_indices), stride):
        envelope = unpack_guides(packed[slot:slot + 1], (NATIVE, NATIVE))[0]
        image = volume[int(slice_indices[slot])]
        image_crop, (guide_crop,), _ = sweep.paired_crop(
            image, [envelope], np.random.default_rng(seed + slot)
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)
        retina_flat = (occupancy >= 0.5).reshape(-1)
        if not retina_flat.any():
            continue
        retina_total = int(retina_flat.sum())
        local = seed + slot

        centre = module.sample_center_anchored(generator, occupancy, local)
        context_list, _ = module.context_patches(
            generator, centre["union"], local
        )
        context_all = set(context_list) | set(centre["union"])

        blocks = [
            generator._block_to_indices(b["top"], b["left"], b["h"], b["w"])
            for b in centre["blocks"]
        ]
        union = set().union(*blocks)
        keep_plain = sum(
            1 for i in (context_all - union) if retina_flat[i]
        ) / retina_total
        rows["plain"].append({
            "keep_true": keep_plain,
            "adjacency": adjacency_leak(union, context_all - union),
            "solid": 1.0,
            "budget": sum(len(b) for b in blocks),
            "unique": len(union),
            "swapped": 0.0,
            "floor_met": float(keep_plain >= MIN_KEEP),
        })

        swapped_blocks, swaps, ok = swap_to_floor(
            blocks, retina_flat, context_all, MIN_KEEP, random.Random(local)
        )
        union2 = set().union(*swapped_blocks)
        keep_swap = sum(
            1 for i in (context_all - union2) if retina_flat[i]
        ) / retina_total
        rows["swap"].append({
            "keep_true": keep_swap,
            "adjacency": adjacency_leak(union2, context_all - union2),
            "solid": float(np.mean([is_solid(sorted(b)) for b in swapped_blocks])),
            "budget": sum(len(b) for b in swapped_blocks),
            "unique": len(union2),
            "swapped": float(swaps),
            "floor_met": float(ok and keep_swap >= MIN_KEEP),
        })

        targets, context, _ = sweep.sample_method(
            sweep.Method("m", "m", "mirage", 0.25, 0), generator, occupancy,
            local, image_crop=image_crop,
        )
        keep_thr = sum(
            1 for i in set(context) if retina_flat[i]
        ) / retina_total
        rows["thr25"].append({
            "keep_true": keep_thr,
            "adjacency": adjacency_leak(set(targets), set(context)),
            "solid": 1.0,
            "budget": len(targets),
            "unique": len(targets),
            "swapped": 0.0,
            "floor_met": float(keep_thr >= MIN_KEEP),
        })

    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", type=int, default=300)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    picks = sorted(random.Random(args.seed).sample(
        guides, min(args.volumes, len(guides))
    ))
    tasks = [(str(p), args.stride, args.seed + i) for i, p in enumerate(picks)]
    print(json.dumps({"volumes": len(tasks), "min_keep": MIN_KEEP}), flush=True)

    started = time.monotonic()
    pooled = {"plain": [], "swap": [], "thr25": []}
    with Pool(processes=args.workers) as pool:
        for index, result in enumerate(
            pool.imap_unordered(evaluate_volume, tasks, chunksize=4), start=1
        ):
            if "error" in result:
                continue
            for key, rows in result["rows"].items():
                pooled[key].extend(rows)
            if index % 100 == 0 or index == len(tasks):
                rate = index / max(time.monotonic() - started, 1e-9)
                print(f"{index}/{len(tasks)}; {rate:.1f} vol/s", flush=True)

    labels = {
        "thr25": "MIRAGE thr 0.25 (solid blocks)",
        "plain": "centre-anchored, no floor",
        "swap": "centre + constant-budget swap to 20%",
    }
    summary = {}
    header = (
        f"{'method':<38}{'keep_TRUE':>11}{'floor%':>9}{'adjacency':>11}"
        f"{'solid%':>9}{'budget':>9}{'uniq':>7}{'swaps':>8}"
    )
    print(f"\nslices {len(pooled['thr25'])}\n")
    print(header)
    print("-" * len(header))
    for key in ("thr25", "plain", "swap"):
        rows = pooled[key]
        if not rows:
            continue
        agg = {
            field: float(np.nanmean([r[field] for r in rows]))
            for field in rows[0]
        }
        summary[key] = {"label": labels[key], "n": len(rows), **agg}
        print(
            f"{labels[key]:<38}{agg['keep_true']:>11.4f}"
            f"{agg['floor_met'] * 100:>8.0f}%{agg['adjacency']:>11.4f}"
            f"{agg['solid'] * 100:>8.0f}%{agg['budget']:>9.1f}"
            f"{agg['unique']:>7.1f}{agg['swapped']:>8.1f}"
        )
    print(
        "\nkeep_TRUE = retina inside the context block and unmasked"
        "\nadjacency = fraction of target patches touching a context patch "
        "(higher = easier to predict)"
        "\nsolid%    = target blocks still contiguous rectangles"
        "\nbudget    = total target patches (sum over 4 blocks, before dedup)"
    )

    (args.output / "swap_budget.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {args.output / 'swap_budget.json'}")


if __name__ == "__main__":
    main()
