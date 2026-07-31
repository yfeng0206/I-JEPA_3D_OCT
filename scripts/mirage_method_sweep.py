#!/usr/bin/env python
"""Evaluate MIRAGE masking variants across the whole training set.

Truth is deliberately NOT MIRAGE.  Scoring masking policies against the same
segmentation that produced them would be circular, so "real tissue" is defined
independently by image brightness: OCT B-scans are strongly bimodal (dark
vitreous versus bright tissue), so a per-slice Otsu threshold over patch-mean
intensity gives a MIRAGE-free reference.

Two metrics carry the trade-off that matters:

* **target purity** -- of the patches we mask, how much is real tissue.
* **context retention** -- of all tissue patches, how much the encoder still
  sees.  Masking everything informative would starve the predictor.

Chance level is reported analytically as the slice's tissue fraction: uniform
block placement has an expected purity equal to it, so no baseline sampler has
to be run.

Examples:
    python scripts/mirage_method_sweep.py --limit 50 --workers 4   # smoke
    python scripts/mirage_method_sweep.py --workers 8              # full run
    python scripts/mirage_method_sweep.py --panels 12              # visuals
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    build_union,
    dilate_patch_grid,
    patch_occupancy,
    unpack_guides,
)
from src.guides.tissue_truth import (  # noqa: E402
    DEFAULT_K,
    patch_coverage,
    tissue_pixels_noise_band,
    truth_patchmean_otsu,
    truth_pixel_otsu,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

GUIDE_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_method_sweep")

TRUTH_NAMES = ("patchmean_otsu", "pixel_otsu", "noise_band", "union_mirage")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NATIVE = 200

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


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    kind: str          # "mirage" | "center" | "region"
    threshold: float = 0.5
    dilate: int = 0
    pred_scale: tuple = (0.15, 0.20)
    spread: bool = False


METHODS = (
    Method("random", "RANDOM (baseline)", "random"),
    Method("oracle", "ORACLE anatomical", "oracle"),
    Method("thr50", "MIRAGE thr 0.50", "mirage", 0.50, 0),
    Method("thr25", "MIRAGE thr 0.25", "mirage", 0.25, 0),
    Method("thr10", "MIRAGE thr 0.10", "mirage", 0.10, 0),
    Method("thr75", "MIRAGE thr 0.75", "mirage", 0.75, 0),
    Method("thr50_dil1", "MIRAGE thr 0.50 +1 dilate", "mirage", 0.50, 1),
    Method("thr75_dil1", "MIRAGE thr 0.75 +1 dilate", "mirage", 0.75, 1),
    Method("thr90_dil1", "MIRAGE thr 0.90 +1 dilate", "mirage", 0.90, 1),
    Method("thr95_dil1", "MIRAGE thr 0.95 +1 dilate", "mirage", 0.95, 1),
    Method("thr99_dil1", "MIRAGE thr 0.99 +1 dilate", "mirage", 0.99, 1),
    Method("thr100_dil1", "MIRAGE thr 1.00 +1 dilate", "mirage", 1.00, 1),
    Method("thr25_dil1", "MIRAGE thr 0.25 +1 dilate", "mirage", 0.25, 1),
    Method("thr10_dil1", "MIRAGE thr 0.10 +1 dilate", "mirage", 0.10, 1),
    Method("thr75_dil2", "MIRAGE thr 0.75 +2 dilate", "mirage", 0.75, 2),
    Method("thr50_dil2", "MIRAGE thr 0.50 +2 dilate", "mirage", 0.50, 2),
    Method("center", "centre-anchored", "center", 0.50, 0),
    Method("region_only", "region-only small blocks", "region", 0.50, 0,
           pred_scale=(0.04, 0.05), spread=True),
    # Removed after the sweep concluded, results in
    # docs/experiments/mirage_guided_masking.md:
    #   * COMBINED (MIRAGE union salience) -- salience alone covers 37.0% of the
    #     frame and the union 37.1%, so MIRAGE contributed 0.1% and the guide
    #     stopped being MIRAGE-guided.  Tightening salience to k=3/4/5 did not
    #     rescue it.
    #   * center_constrained (retry then trim to a retina-visible floor) --
    #     reached the floor on only 37% of slices, and trimming changes block
    #     size and aspect ratio, so it is no longer a location-only change.
    # Both are recoverable from git history (see commit that added this file).
)

ALL_METHODS = METHODS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PREVIEW = None


def preview_module():
    """Lazily import the centre-anchored / region-only samplers."""
    global _PREVIEW
    if _PREVIEW is None:
        spec = importlib.util.spec_from_file_location(
            "mirage_region_only_preview",
            Path(_PROJECT_ROOT) / "scripts" / "mirage_region_only_preview.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PREVIEW = module
    return _PREVIEW


def otsu(values: np.ndarray) -> float:
    """Otsu threshold over a small 1-D sample (the 256 patch means)."""
    hist, edges = np.histogram(values, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centres)
    total_mean = mean_bg[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_bg = mean_bg / weight_bg
        mu_fg = (total_mean - mean_bg) / weight_fg
        between = weight_bg * weight_fg * (mu_bg - mu_fg) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centres[int(np.argmax(between))])


def paired_crop(image: np.ndarray, masks, rng):
    """Apply one RandomResizedCrop draw to image and every mask."""
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
    if isinstance(masks, np.ndarray):
        masks = [masks]
    cropped = [
        np.asarray(
            Image.fromarray(
                mask[top:top + crop_h, left:left + crop_w].astype(np.uint8) * 255,
                mode="L",
            ).resize((CROP, CROP), Image.NEAREST)
        ) > 127
        for mask in masks
    ]
    return image_crop, cropped, (top, left, crop_h, crop_w)


def make_generator(method: Method):
    """Build the sampler for a method.

    RANDOM uses the stock multiblock collator, which is the exact baseline the
    original run used.  ORACLE uses the existing intensity-derived anatomical
    prior, unchanged, so both reference arms are measured with the same code
    that produced them.
    """
    kwargs = dict(MASK_KWARGS)
    kwargs["pred_mask_scale"] = method.pred_scale
    if method.kind == "random":
        from src.masks.multiblock import MaskCollator

        return MaskCollator(**kwargs)
    if method.kind == "oracle":
        return CurriculumMaskGenerator(
            curriculum_cfg={
                "mode": "anatomical_prior",
                "T_warm": 25,
                "T_total": 30,
                "r_max": 1.0,
                "oracle_region_frac": 0.28,
                "oracle_lateral_frac": 0.6,
                "oracle_row_offset": 0.0,
                "oracle_min_band_rows": 3,
            },
            **kwargs,
        )
    return CurriculumMaskGenerator(
        curriculum_cfg={
            "mode": "mirage_envelope",
            "T_warm": 25,
            "T_total": 30,
            "r_max": 1.0,
            "mirage_min_block_fill": 0.40,
            "mirage_min_retina_visible": 0.25,
            "mirage_occupancy_threshold": method.threshold,
            "mirage_spread": method.spread,
        },
        **kwargs,
    )


def normalized_tensor(image_crop: np.ndarray) -> torch.Tensor:
    """Replicate the training transform so the oracle sees what it expects."""
    rgb = np.repeat(image_crop[..., None].astype(np.float32) / 255.0, 3, axis=2)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


def sample_method(method: Method, generator, occupancy, seed, image_crop=None):
    """Return (target index set, context index set, stats dict)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))

    if method.kind == "random":
        _imgs, masks_enc, masks_pred = generator(
            [torch.zeros(3, CROP, CROP)]
        )
        targets = {int(i) for group in masks_pred for i in group[0].tolist()}
        context = {int(i) for i in masks_enc[0][0].tolist()}
        return targets, context, {"accept_rate": float("nan"), "fallbacks": 0}

    if method.kind == "oracle":
        tensor = normalized_tensor(image_crop)[None]
        masks_enc, masks_pred = generator.generate(
            batch_size=1, imgs_cpu=tensor
        )
        targets = {int(i) for group in masks_pred for i in group[0].tolist()}
        context = {int(i) for i in masks_enc[0][0].tolist()}
        return targets, context, {"accept_rate": float("nan"), "fallbacks": 0}

    region = occupancy >= method.threshold
    placement = dilate_patch_grid(region, method.dilate)
    guide = torch.from_numpy(
        np.stack([occupancy.astype(np.float32), placement.astype(np.float32)])
    )[None]

    if method.kind == "mirage":
        masks_enc, masks_pred = generator.generate(
            batch_size=1, guide_grids=guide,
            guide_valid=torch.ones(1, dtype=torch.bool),
        )
        targets = {int(i) for group in masks_pred for i in group[0].tolist()}
        context = {int(i) for i in masks_enc[0][0].tolist()}
        stats = dict(generator.mirage_stats)
        return targets, context, stats

    module = preview_module()
    sampler = (
        module.sample_center_anchored if method.kind == "center"
        else module.sample_region_only
    )
    result = sampler(generator, occupancy, seed)
    kept, _block = module.context_patches(generator, result["union"], seed)
    return set(result["union"]), set(kept), {
        "accept_rate": float("nan"),
        "fallbacks": 0,
        "trim_steps": result.get("trim_steps", 0),
        "trimmed": float(result.get("trimmed", False)),
        "attempts": result.get("attempts", float("nan")),
        "mean_block_fill": result.get("mean_occupancy", float("nan")),
    }


def _select_methods(keys):
    """Restrict METHODS in this process.  Used as a Pool initialiser.

    Windows spawns workers by re-importing the module, so a filter applied in
    the parent does not reach them; each worker must be told explicitly.
    """
    global METHODS
    METHODS = tuple(m for m in ALL_METHODS if m.key in keys)


def evaluate_volume(task):
    """Sample every method on every slice and cache the resulting masks.

    Nothing here depends on the tissue-truth definition.  Target placement is
    a function of the guide occupancy and the RNG seed only, so the masks are
    cached as packed bitmaps and scored separately.  Changing the truth then
    costs a re-score (seconds) instead of a re-sample (tens of minutes).
    """
    guide_path, slice_stride, seed = task
    guide_path = Path(guide_path)
    try:
        with np.load(guide_path, allow_pickle=False) as cache:
            packed = cache["packed_envelopes"]
            slice_indices = cache["slice_indices"].astype(int)
            valid_flags = cache["valid"].astype(bool)
        with np.load(DATA_DIR / guide_path.name, allow_pickle=False) as data:
            volume = data["oct_bscans"]
        with np.load(MASK_DIR / guide_path.name, allow_pickle=False) as cache:
            hard_masks = cache["hard_masks"]
    except Exception as error:  # noqa: BLE001
        return {"error": f"{guide_path.name}: {error}"}

    generators = {m.key: make_generator(m) for m in METHODS}
    for generator in generators.values():
        if hasattr(generator, "set_epoch"):
            generator.set_epoch(30)

    slots = list(range(0, len(slice_indices), slice_stride))
    cells = GRID * GRID
    targets_out = np.zeros((len(slots), len(METHODS), cells), dtype=bool)
    context_out = np.zeros((len(slots), len(METHODS), cells), dtype=bool)
    accepts = np.full((len(slots), len(METHODS)), np.nan, dtype=np.float32)
    crop_boxes = np.zeros((len(slots), 4), dtype=np.int16)
    truth_grids = {name: np.zeros((len(slots), cells), dtype=bool)
                   for name in TRUTH_NAMES}
    guide_grids = np.zeros((len(slots), cells), dtype=bool)
    keep = np.zeros(len(slots), dtype=bool)

    for row, slot in enumerate(slots):
        envelope = unpack_guides(packed[slot:slot + 1], (NATIVE, NATIVE))[0]
        raw_union = build_union(hard_masks[slot])
        image = volume[int(slice_indices[slot])]

        rng = np.random.default_rng(seed + slot)
        image_crop, cropped, box = paired_crop(
            image, [envelope, raw_union], rng
        )
        guide_crop, raw_crop = cropped[0], cropped[1]
        occupancy = patch_occupancy(guide_crop, patch_size=PATCH)

        band = tissue_pixels_noise_band(image_crop)
        grids = {
            "patchmean_otsu": truth_patchmean_otsu(image_crop, patch=PATCH),
            "pixel_otsu": truth_pixel_otsu(image_crop, patch=PATCH),
            "noise_band": patch_coverage(band, PATCH) >= 0.5,
            "union_mirage": patch_coverage(band | raw_crop, PATCH) >= 0.5,
        }
        if not grids["noise_band"].any():
            continue
        keep[row] = True
        crop_boxes[row] = box
        guide_grids[row] = (occupancy >= 0.5).reshape(-1)
        for name, grid in grids.items():
            truth_grids[name][row] = grid.reshape(-1)

        for index, method in enumerate(METHODS):
            target_set, context_set, stats = sample_method(
                method, generators[method.key], occupancy,
                seed + slot, image_crop=image_crop,
            )
            if target_set:
                targets_out[row, index, sorted(target_set)] = True
            if context_set:
                context_out[row, index, sorted(context_set)] = True
            accepts[row, index] = stats.get("accept_rate", np.nan)

    return {
        "volume": guide_path.stem,
        "slots": np.asarray(slots, dtype=np.int32)[keep],
        "slice_indices": slice_indices[np.asarray(slots)][keep],
        "guide_valid": valid_flags[np.asarray(slots)][keep],
        "crop_boxes": crop_boxes[keep],
        "targets": np.packbits(targets_out[keep], axis=-1),
        "context": np.packbits(context_out[keep], axis=-1),
        "accepts": accepts[keep],
        "guide": np.packbits(guide_grids[keep], axis=-1),
        "truth": {name: np.packbits(grid[keep], axis=-1)
                  for name, grid in truth_grids.items()},
    }


def score_cache(cache, truth_name):
    """Derive every metric from cached masks under one truth definition."""
    cells = GRID * GRID
    truth = np.unpackbits(cache["truth"][truth_name], axis=-1,
                          count=cells).astype(bool)
    targets = np.unpackbits(cache["targets"], axis=-1, count=cells).astype(bool)
    context = np.unpackbits(cache["context"], axis=-1, count=cells).astype(bool)
    guide = np.unpackbits(cache["guide"], axis=-1, count=cells).astype(bool)

    tissue_count = truth.sum(axis=-1).astype(np.float32)
    tissue_expand = truth[:, None, :]
    target_count = targets.sum(axis=-1).astype(np.float32)
    covered = (targets & tissue_expand).sum(axis=-1).astype(np.float32)
    retained = (context & tissue_expand).sum(axis=-1).astype(np.float32)

    intersection = (guide & truth).sum(axis=-1).astype(np.float32)
    union = (guide | truth).sum(axis=-1).astype(np.float32)

    return {
        "tissue_fraction": tissue_count / cells,
        "iou": intersection / np.maximum(union, 1.0),
        "purity": covered / np.maximum(target_count, 1.0),
        "tissue_covered": covered / tissue_count[:, None],
        "context_retention": retained / tissue_count[:, None],
        "masked_area": target_count / cells,
        "unique_targets": target_count,
        "context_patches": context.sum(axis=-1).astype(np.float32),
        "accept": cache["accepts"],
    }


def aggregate(results, truth_name):
    """Aggregate cached masks under one truth definition."""
    fields = ("purity", "tissue_covered", "context_retention", "masked_area",
              "unique_targets", "context_patches", "accept")
    pooled = {f: [] for f in fields}
    tissue, agreement = [], []
    volumes = 0
    for result in results:
        if "error" in result or len(result["slots"]) == 0:
            continue
        volumes += 1
        scored = score_cache(result, truth_name)
        tissue.append(scored["tissue_fraction"])
        agreement.append(scored["iou"])
        for field in fields:
            pooled[field].append(np.atleast_2d(scored[field]))

    if not volumes:
        return {"volumes": 0, "slices": 0, "methods": {}}

    tissue = np.concatenate(tissue)
    agreement = np.concatenate(agreement)
    stacked = {f: np.concatenate(pooled[f], axis=0) for f in fields}

    summary = {
        "truth": truth_name,
        "volumes": volumes,
        "slices": int(tissue.size),
        "chance_purity": float(tissue.mean()),
        "mirage_vs_brightness_iou": float(agreement.mean()),
        "methods": {},
    }
    for index, method in enumerate(METHODS):
        summary["methods"][method.key] = {
            "label": method.label,
            "target_purity": float(stacked["purity"][:, index].mean()),
            "tissue_covered": float(stacked["tissue_covered"][:, index].mean()),
            "context_retention": float(
                stacked["context_retention"][:, index].mean()
            ),
            "masked_area": float(stacked["masked_area"][:, index].mean()),
            "unique_targets": float(stacked["unique_targets"][:, index].mean()),
            "context_patches": float(stacked["context_patches"][:, index].mean()),
            "accept_rate": float(np.nanmean(stacked["accept"][:, index]))
            if np.isfinite(stacked["accept"][:, index]).any() else float("nan"),
        }
    return summary


def print_summary(summary):
    chance = summary["chance_purity"]
    print(f"\n=== truth: {summary['truth']} ===")
    print(f"Volumes {summary['volumes']}   slices {summary['slices']}")
    print(f"Chance purity (tissue fraction of image): {chance:.4f}")
    print(
        "MIRAGE-guide-vs-truth IoU: "
        f"{summary['mirage_vs_brightness_iou']:.4f}"
    )
    print()
    header = (
        f"{'method':<28}{'purity':>9}{'lift':>7}{'tissue_cov':>12}"
        f"{'ctx_retain':>12}{'masked':>9}{'uniq':>7}{'ctx':>7}{'accept':>8}"
    )
    print(header)
    print("-" * len(header))
    for key, row in summary["methods"].items():
        print(
            f"{row['label']:<28}{row['target_purity']:>9.4f}"
            f"{row['target_purity'] / chance:>7.2f}x"
            f"{row['tissue_covered']:>12.4f}{row['context_retention']:>12.4f}"
            f"{row['masked_area']:>9.3f}{row['unique_targets']:>7.1f}"
            f"{row['context_patches']:>7.1f}{row['accept_rate']:>8.2f}"
        )
    print(
        "\npurity      = fraction of masked patches that are real tissue "
        "(chance = tissue fraction)"
    )
    print("tissue_cov  = fraction of all tissue that got masked")
    print("ctx_retain  = fraction of all tissue still visible to the encoder")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="volumes (default: all)")
    parser.add_argument(
        "--sample", type=int,
        help="randomly sample N volumes instead of taking the first N",
    )
    parser.add_argument("--slice-stride", type=int, default=1)
    parser.add_argument(
        "--methods",
        help="comma-separated method keys to run (default: all)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.methods:
        wanted = [k.strip() for k in args.methods.split(",") if k.strip()]
        known = {m.key for m in ALL_METHODS}
        missing = [k for k in wanted if k not in known]
        if missing:
            parser.error(f"unknown method keys: {missing}; known: {sorted(known)}")
        _select_methods(set(wanted))

    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(GUIDE_DIR.glob("data_*.npz"))
    if args.sample and args.sample < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.sample))
    elif args.limit:
        paths = paths[: args.limit]
    tasks = [(str(p), args.slice_stride, args.seed + i) for i, p in enumerate(paths)]

    print(
        json.dumps(
            {
                "volumes": len(tasks),
                "slice_stride": args.slice_stride,
                "methods": [m.key for m in METHODS],
                "workers": args.workers,
            }
        ),
        flush=True,
    )

    started = time.monotonic()
    results = []
    if args.workers > 1:
        keys = {m.key for m in METHODS}
        with Pool(
            processes=args.workers, initializer=_select_methods, initargs=(keys,)
        ) as pool:
            for index, result in enumerate(
                pool.imap_unordered(evaluate_volume, tasks, chunksize=4), start=1
            ):
                results.append(result)
                if index % 250 == 0 or index == len(tasks):
                    elapsed = time.monotonic() - started
                    rate = index / max(elapsed, 1e-9)
                    print(
                        f"{index}/{len(tasks)} volumes; {rate:.1f} vol/s; "
                        f"ETA={(len(tasks) - index) / max(rate, 1e-9) / 60:.1f} min",
                        flush=True,
                    )
    else:
        for index, task in enumerate(tasks, start=1):
            results.append(evaluate_volume(task))
            if index % 25 == 0:
                print(f"{index}/{len(tasks)}", flush=True)

    errors = [r["error"] for r in results if "error" in r]
    good = [r for r in results if "error" not in r and len(r["slots"])]

    cache_path = args.output / "mask_cache.npz"
    np.savez_compressed(
        cache_path,
        volumes=np.array([r["volume"] for r in good]),
        slice_counts=np.array([len(r["slots"]) for r in good], dtype=np.int32),
        slice_indices=np.concatenate([r["slice_indices"] for r in good]),
        guide_valid=np.concatenate([r["guide_valid"] for r in good]),
        crop_boxes=np.concatenate([r["crop_boxes"] for r in good]),
        targets=np.concatenate([r["targets"] for r in good]),
        context=np.concatenate([r["context"] for r in good]),
        accepts=np.concatenate([r["accepts"] for r in good]),
        guide=np.concatenate([r["guide"] for r in good]),
        methods=np.array([m.key for m in METHODS]),
        **{
            f"truth_{name}": np.concatenate([r["truth"][name] for r in good])
            for name in TRUTH_NAMES
        },
    )
    print(
        f"\nCached masks for {len(good)} volumes -> {cache_path} "
        f"({cache_path.stat().st_size / 1e6:.1f} MB). "
        "Re-scoring a new truth needs no re-sampling.",
        flush=True,
    )

    summaries = {}
    for truth_name in TRUTH_NAMES:
        summary = aggregate(results, truth_name)
        summaries[truth_name] = summary
        print_summary(summary)
    payload = {
        "truths": summaries,
        "primary_truth": "noise_band",
        "errors": errors[:20],
        "error_count": len(errors),
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
    }
    (args.output / "method_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        print(f"\n{len(errors)} volumes failed; first: {errors[0]}")
    print(f"\nSaved {args.output / 'method_sweep.json'}")


if __name__ == "__main__":
    main()
