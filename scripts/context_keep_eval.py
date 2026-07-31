#!/usr/bin/env python
"""Compare target-location policies on the metric that actually matters.

Two corrections over the earlier centre-anchored work, both raised in review:

**The visible-retina metric was wrong.**  It measured retina not covered by a
target block, ``|G \\ T| / |G|``.  But the encoder does not see the whole frame:
the context is itself a sampled block covering 85-100% of it, so retina outside
that block is invisible regardless of where targets land.  The metric that
matches what the predictor is actually conditioned on is

    K = |G and C_raw minus T| / |G|

Both are reported here so the size of the error is visible, not assumed.

**Forced trimming is not location-only.**  Peeling rows off blocks changes block
size, aspect ratio and -- through the collator's global-min truncation -- the
target length of every image in the batch.  It is included as a measured arm so
the cost is on record, but the recommended policy does not trim.

Usage:
    python scripts/context_keep_eval.py --volumes 400 --workers 8
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
from src.guides.tissue_truth import (  # noqa: E402
    patch_coverage,
    tissue_pixels_noise_band,
)

CROP, PATCH, GRID, NATIVE = sweep.CROP, sweep.PATCH, sweep.GRID, sweep.NATIVE
OUTPUT = Path(r"D:\jepa_phase0\fairvision-glaucoma\context_keep_eval")

ARMS = (
    ("random", "RANDOM baseline"),
    ("oracle", "ORACLE anatomical"),
    ("mirage_thr25", "MIRAGE thr 0.25 (biased-location)"),
    ("center_plain", "centre-anchored, no floor"),
    ("center_trim20", "centre-anchored + trim to 20%"),
    ("mixed1_20", "1 guided + 3 uniform, floor 20%"),
    ("mixed2_20", "2 guided + 2 uniform, floor 20%"),
    ("mixed3_20", "3 guided + 1 uniform, floor 20%"),
    ("mixed2_25", "2 guided + 2 uniform, floor 25%"),
    ("mixed2_30", "2 guided + 2 uniform, floor 30%"),
)


def blank_row():
    return {
        "keep_true": np.nan, "keep_naive": np.nan, "floor_met": np.nan,
        "purity": np.nan, "tissue_cov": np.nan, "unique": np.nan,
        "context": np.nan, "smallest_block": np.nan, "overlap": np.nan,
        "trim": np.nan,
    }


def score(union, context, retina_flat, tissue_flat, blocks, extra=None):
    retina_total = max(int(retina_flat.sum()), 1)
    tissue_total = max(int(tissue_flat.sum()), 1)
    union_list = sorted(union)
    context_set = set(context)
    visible_retina = sum(1 for i in context_set if retina_flat[i])
    unmasked_retina = retina_total - sum(1 for i in union_list if retina_flat[i])
    block_area = sum(b["h"] * b["w"] for b in blocks) if blocks else 0
    row = {
        "keep_true": visible_retina / retina_total,
        "keep_naive": unmasked_retina / retina_total,
        "purity": (
            sum(1 for i in union_list if tissue_flat[i]) / max(len(union_list), 1)
        ),
        "tissue_cov": sum(1 for i in union_list if tissue_flat[i]) / tissue_total,
        "unique": len(union_list),
        "context": len(context_set),
        "smallest_block": min((b["h"] * b["w"] for b in blocks), default=np.nan),
        # 1.0 means the four blocks are disjoint; lower means they pile up.
        "overlap": (len(union_list) / block_area) if block_area else np.nan,
        "floor_met": np.nan,
        "trim": 0.0,
    }
    if extra:
        row.update(extra)
    return row


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

    method = sweep.Method("m", "m", "mirage", 0.25, 0)
    generator = sweep.make_generator(method)
    generator.set_epoch(30)
    plain_gen = sweep.make_generator(sweep.Method("r", "r", "random"))
    oracle_gen = sweep.make_generator(sweep.Method("o", "o", "oracle"))
    oracle_gen.set_epoch(30)

    rows = {key: [] for key, _ in ARMS}
    for slot in range(0, len(slice_indices), stride):
        envelope = unpack_guides(packed[slot:slot + 1], (NATIVE, NATIVE))[0]
        image = volume[int(slice_indices[slot])]
        image_crop, (guide_crop,), _ = sweep.paired_crop(
            image, [envelope], np.random.default_rng(seed + slot)
        )
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)
        retina_flat = (occupancy >= 0.5).reshape(-1)
        tissue_flat = (
            patch_coverage(tissue_pixels_noise_band(image_crop), PATCH) >= 0.5
        ).reshape(-1)
        if not tissue_flat.any() or not retina_flat.any():
            continue
        local = seed + slot

        for key, _label in ARMS:
            if key == "random":
                targets, context, _ = sweep.sample_method(
                    sweep.Method("r", "r", "random"), plain_gen, occupancy, local
                )
                rows[key].append(
                    score(targets, context, retina_flat, tissue_flat, [])
                )
                continue
            if key == "oracle":
                targets, context, _ = sweep.sample_method(
                    sweep.Method("o", "o", "oracle"), oracle_gen, occupancy,
                    local, image_crop=image_crop,
                )
                rows[key].append(
                    score(targets, context, retina_flat, tissue_flat, [])
                )
                continue
            if key == "mirage_thr25":
                targets, context, _ = sweep.sample_method(
                    method, generator, occupancy, local, image_crop=image_crop
                )
                rows[key].append(
                    score(targets, context, retina_flat, tissue_flat, [])
                )
                continue
            if key == "center_plain":
                result = module.sample_center_anchored(generator, occupancy, local)
                context, _ = module.context_patches(
                    generator, result["union"], local
                )
                rows[key].append(
                    score(result["union"], context, retina_flat, tissue_flat,
                          result["blocks"])
                )
                continue
            if key == "center_trim20":
                result = module.sample_center_anchored_constrained(
                    generator, occupancy, local, min_retina_visible=0.20
                )
                context, _ = module.context_patches(
                    generator, result["union"], local
                )
                rows[key].append(
                    score(result["union"], context, retina_flat, tissue_flat,
                          result["blocks"],
                          {"floor_met": float(result["floor_met"]),
                           "trim": float(result["trim_steps"])})
                )
                continue

            guided, floor = {
                "mixed1_20": (1, 0.20), "mixed2_20": (2, 0.20),
                "mixed3_20": (3, 0.20), "mixed2_25": (2, 0.25),
                "mixed2_30": (2, 0.30),
            }[key]
            result = module.sample_mixed_guided(
                generator, occupancy, local, n_guided=guided, min_keep_frac=floor
            )
            if result is None:
                rows[key].append(blank_row())
                continue
            rows[key].append(
                score(result["union"], result["context"], retina_flat,
                      tissue_flat, result["blocks"],
                      {"floor_met": float(result["floor_met"])})
            )

    return {"volume": guide_path.stem, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", type=int, default=400)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    guides = sorted(sweep.GUIDE_DIR.glob("data_*.npz"))
    picks = sorted(random.Random(args.seed).sample(
        guides, min(args.volumes, len(guides))
    ))
    tasks = [(str(p), args.stride, args.seed + i) for i, p in enumerate(picks)]
    print(json.dumps({"volumes": len(tasks), "stride": args.stride,
                      "arms": [k for k, _ in ARMS]}), flush=True)

    started = time.monotonic()
    results = []
    with Pool(processes=args.workers) as pool:
        for index, result in enumerate(
            pool.imap_unordered(evaluate_volume, tasks, chunksize=4), start=1
        ):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                rate = index / max(time.monotonic() - started, 1e-9)
                print(f"{index}/{len(tasks)}; {rate:.1f} vol/s; "
                      f"ETA={(len(tasks) - index) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    pooled = {key: [] for key, _ in ARMS}
    for result in results:
        if "error" in result:
            continue
        for key, rows in result["rows"].items():
            pooled[key].extend(rows)

    summary, slices = {}, 0
    for key, label in ARMS:
        rows = pooled[key]
        if not rows:
            continue
        slices = max(slices, len(rows))
        summary[key] = {"label": label, "n": len(rows)}
        for field in rows[0]:
            values = np.array([r[field] for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[key][field] = float(values.mean()) if values.size else float("nan")

    header = (
        f"{'arm':<38}{'keep_TRUE':>11}{'keep_naive':>12}{'floor%':>9}"
        f"{'purity':>9}{'uniq':>7}{'ctx':>7}{'minblk':>8}{'disjoint':>10}"
    )
    print(f"\nvolumes {len(results)}   slices {slices}\n")
    print(header)
    print("-" * len(header))
    for key, _label in ARMS:
        if key not in summary:
            continue
        row = summary[key]
        floor = row["floor_met"]
        print(
            f"{row['label']:<38}{row['keep_true']:>11.4f}{row['keep_naive']:>12.4f}"
            f"{'' if np.isnan(floor) else f'{floor * 100:8.0f}%'}"
            f"{'' if not np.isnan(floor) else '        -'}"
            f"{row['purity']:>9.4f}{row['unique']:>7.1f}{row['context']:>7.1f}"
            f"{row['smallest_block']:>8.1f}{row['overlap']:>10.3f}"
        )
    print(
        "\nkeep_TRUE  = retina inside the context block and not masked "
        "(what the encoder really sees)"
    )
    print("keep_naive = retina not masked, ignoring the context crop (overstates)")
    print("disjoint   = unique targets / total block area; 1.0 = no block overlap")

    (args.output / "context_keep.json").write_text(
        json.dumps({"summary": summary, "volumes": len(results),
                    "slices": slices}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {args.output / 'context_keep.json'}")


if __name__ == "__main__":
    main()
