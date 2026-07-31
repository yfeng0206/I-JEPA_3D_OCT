#!/usr/bin/env python
"""Compare MIRAGE masking against the random and oracle arms.

Two checks, because the two arms must agree on different things:

**Geometry parity** -- the experimental contract says the ONLY variable is
where target blocks land.  Target-block count, patches per block, unique target
coverage and context-token count must therefore match the random and oracle
arms.  The historical runs never logged these, so they are recomputed here for
all three samplers on identical slices with identical seeds.

**Placement divergence** -- the one quantity that SHOULD differ.  Targets are
expected to fall on retina far more often than random, comparable to the oracle.

Also parses a live training log and compares its loss trajectory with the
archived oracle/random epoch summaries.

Examples:
    python scripts/compare_mirage_vs_oracle.py --volumes 20
    python scripts/compare_mirage_vs_oracle.py --train-log D:/.../mirage.log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
ORACLE_SUMMARY = Path("logs/pretraining/run3_epoch_summary.csv")
RANDOM_SUMMARY = Path("logs/pretraining/run1_epoch_summary.csv")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

# Tolerances for "these arms agree".  Block geometry is driven by the same
# size sampler in every arm, so only sampling noise should separate them.
GEOMETRY_TOLERANCE = {
    "patches_per_block": 0.06,
    "unique_target_patches": 0.06,
    "context_patches": 0.10,
}


def make_generator(mode: str, **extra) -> CurriculumMaskGenerator:
    cfg = {"mode": mode, "T_warm": 25, "T_total": 30, "r_max": 1.0,
           "ramp_shape": "linear"}
    cfg.update(extra)
    return CurriculumMaskGenerator(curriculum_cfg=cfg, **MASK_KWARGS)


def summarise(masks_enc, masks_pred, occupancy):
    """The statistics both arms must agree on, plus the one that must differ."""
    batch = masks_pred[0].shape[0]
    per_block, unique, context, on_region = [], [], [], []
    for index in range(batch):
        blocks = [group[index].tolist() for group in masks_pred]
        union = set()
        for block in blocks:
            per_block.append(len(block))
            union.update(block)
        unique.append(len(union))
        context.append(int(masks_enc[0][index].numel()))
        if occupancy is not None:
            flat = occupancy[index].reshape(-1)
            on_region.append(
                float(np.mean([flat[i] >= 0.5 for i in sorted(union)]))
            )
    return {
        "patches_per_block": float(np.mean(per_block)),
        "unique_target_patches": float(np.mean(unique)),
        "context_patches": float(np.mean(context)),
        "target_on_region": float(np.mean(on_region)) if on_region else float("nan"),
    }


def load_batch(paths, rng, batch_size):
    """One batch of images plus their aligned guide grids.

    Volumes are sampled with replacement (a different slice is drawn each time),
    so the batch size is independent of how many volumes are scanned.
    """
    images, occupancy, placement = [], [], []
    for choice in rng.integers(len(paths), size=batch_size):
        path = Path(paths[int(choice)])
        with np.load(path, allow_pickle=False) as cache:
            index = int(rng.integers(cache["packed_envelopes"].shape[0]))
            envelope = unpack_guides(
                cache["packed_envelopes"][index : index + 1], (200, 200)
            )[0]
            slice_index = int(cache["slice_indices"][index])
        with np.load(DATA_DIR / path.name, allow_pickle=False) as data:
            slice_2d = data["oct_bscans"][slice_index]
        image = np.asarray(
            Image.fromarray(slice_2d, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )
        rgb = np.repeat(image[..., None].astype(np.float32) / 255.0, 3, axis=2)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        grown = (
            np.asarray(
                Image.fromarray(envelope.astype(np.uint8) * 255, mode="L").resize(
                    (CROP, CROP), Image.Resampling.NEAREST
                )
            )
            > 127
        )
        grid = patch_occupancy(grown, patch_size=PATCH)
        occupancy.append(grid)
        placement.append(dilate_patch_grid(grid >= 0.5, 1).astype(np.float32))
    guides = torch.from_numpy(
        np.stack([np.stack([o, p], axis=0) for o, p in zip(occupancy, placement)])
    ).float()
    return torch.stack(images), guides, np.stack(occupancy)


def geometry_check(volumes, batches, batch_size, seed):
    paths = sorted(GUIDE_DIR.glob("data_*.npz"))[:volumes]
    if not paths:
        raise FileNotFoundError(f"No guides under {GUIDE_DIR}")
    collator = MaskCollator(**MASK_KWARGS)
    oracle = make_generator("anatomical_prior", oracle_region_frac=0.28,
                            oracle_lateral_frac=0.6)
    mirage = make_generator("mirage_envelope", mirage_min_block_fill=0.40,
                            mirage_min_retina_visible=0.25)
    oracle.set_epoch(30)
    mirage.set_epoch(30)

    accumulated = {"random": [], "oracle": [], "mirage": []}
    for batch_index in range(batches):
        rng = np.random.default_rng(seed + batch_index)
        images, guides, occupancy = load_batch(paths, rng, batch_size)

        # Identical seeds so all three draw the same block SIZES.
        torch.manual_seed(seed + batch_index)
        random.seed(seed + batch_index)
        _imgs, enc_r, pred_r = collator([img for img in images])
        accumulated["random"].append(summarise(enc_r, pred_r, occupancy))

        torch.manual_seed(seed + batch_index)
        random.seed(seed + batch_index)
        enc_o, pred_o = oracle.generate(batch_size=images.size(0), imgs_cpu=images)
        accumulated["oracle"].append(summarise(enc_o, pred_o, occupancy))

        torch.manual_seed(seed + batch_index)
        random.seed(seed + batch_index)
        enc_m, pred_m = mirage.generate(
            batch_size=images.size(0),
            guide_grids=guides,
            guide_valid=torch.ones(images.size(0), dtype=torch.bool),
        )
        accumulated["mirage"].append(summarise(enc_m, pred_m, occupancy))

    return {
        arm: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        for arm, rows in accumulated.items()
    }


def parse_training_log(path: Path):
    """Pull loss and MIRAGE mask stats out of a live training log."""
    iteration = re.compile(
        r"\[Epoch (\d+)/\d+ \| Iter (\d+)/\d+\] loss=([\d.]+)"
    )
    mirage = re.compile(
        r"\[MIRAGE\] patches/block=([\d.]+)\s+unique_targets=([\d.]+)\s+"
        r"context=([\d.]+)\s+on_region=([\d.]+)\s+background=([\d.]+)\s+"
        r"fallbacks=(\d+)\s+infeasible=(\d+)\s+unbiased=(\d+)\s+"
        r"accept=([\d.]+)\s+fill=([\d.]+)\s+retina_visible=([\d.]+)"
    )
    losses, stats = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = iteration.search(line)
        if match:
            losses.append(
                {"epoch": int(match.group(1)), "iter": int(match.group(2)),
                 "loss": float(match.group(3))}
            )
        match = mirage.search(line)
        if match:
            stats.append(
                {
                    "patches_per_block": float(match.group(1)),
                    "unique_target_patches": float(match.group(2)),
                    "context_patches": float(match.group(3)),
                    "target_on_region": float(match.group(4)),
                    "fallbacks": int(match.group(6)),
                    "infeasible": int(match.group(7)),
                    "unbiased": int(match.group(8)),
                    "accept_rate": float(match.group(9)),
                    "mean_block_fill": float(match.group(10)),
                    "retina_visible": float(match.group(11)),
                }
            )
    return losses, stats


def read_summary(path: Path):
    if not path.is_file():
        return {}
    rows = {}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        record = dict(zip(header, values))
        rows[int(record["epoch"])] = {
            k: float(v) for k, v in record.items() if k != "epoch"
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=20)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--skip-geometry", action="store_true")
    args = parser.parse_args()

    report = {}
    failures = []

    if not args.skip_geometry:
        geometry = geometry_check(
            args.volumes, args.batches, args.batch_size, args.seed
        )
        report["geometry"] = geometry
        print("Mask geometry (must match across arms)")
        print(f"{'metric':<24}{'random':>10}{'oracle':>10}{'mirage':>10}{'status':>10}")
        for key, tolerance in GEOMETRY_TOLERANCE.items():
            values = [geometry[arm][key] for arm in ("random", "oracle", "mirage")]
            spread = (max(values) - min(values)) / max(np.mean(values), 1e-9)
            ok = spread <= tolerance
            if not ok:
                failures.append(f"{key} spread {spread:.1%} > {tolerance:.0%}")
            print(
                f"{key:<24}{values[0]:>10.1f}{values[1]:>10.1f}{values[2]:>10.1f}"
                f"{('OK' if ok else 'DIFFERS'):>10}"
            )
        print()
        print("Target placement (SHOULD differ - this is the experiment)")
        print(f"{'target_on_region':<24}"
              f"{geometry['random']['target_on_region']:>10.3f}"
              f"{geometry['oracle']['target_on_region']:>10.3f}"
              f"{geometry['mirage']['target_on_region']:>10.3f}")
        lift = geometry["mirage"]["target_on_region"] / max(
            geometry["random"]["target_on_region"], 1e-9
        )
        print(f"{'mirage / random lift':<24}{lift:>30.2f}x")
        report["lift_vs_random"] = lift
        if lift < 1.15:
            failures.append(f"MIRAGE lift over random is only {lift:.2f}x")

    if args.train_log and args.train_log.is_file():
        losses, stats = parse_training_log(args.train_log)
        print(f"\nLive log: {args.train_log}")
        if losses:
            recent = losses[-20:]
            epochs = sorted({r["epoch"] for r in recent})
            print(f"  iterations parsed: {len(losses)}  epochs seen: {epochs}")
            print(f"  recent loss mean: {np.mean([r['loss'] for r in recent]):.4f}")
            report["recent_loss"] = float(np.mean([r["loss"] for r in recent]))
            oracle_rows = read_summary(ORACLE_SUMMARY)
            random_rows = read_summary(RANDOM_SUMMARY)
            for epoch in epochs:
                reference = oracle_rows.get(epoch) or random_rows.get(epoch)
                if reference:
                    print(
                        f"  epoch {epoch}: reference train_loss="
                        f"{reference.get('train_loss')}, val_loss="
                        f"{reference.get('val_loss')}"
                    )
        if stats:
            recent = stats[-10:]
            print("  MIRAGE mask stats (recent mean):")
            for key in (
                "patches_per_block", "unique_target_patches", "context_patches",
                "target_on_region", "accept_rate", "mean_block_fill",
                "retina_visible",
            ):
                print(f"    {key:<24}{np.mean([r[key] for r in recent]):>8.3f}")
            fallbacks = sum(r["fallbacks"] for r in recent)
            infeasible = sum(r["infeasible"] for r in recent)
            print(f"    {'fallbacks (recent)':<24}{fallbacks:>8d}")
            print(f"    {'infeasible (recent)':<24}{infeasible:>8d}")
            report["live_stats"] = {
                k: float(np.mean([r[k] for r in recent])) for k in recent[0]
            }
            if not args.skip_geometry:
                # The live run must agree with the offline geometry prediction.
                for key in ("patches_per_block", "unique_target_patches"):
                    predicted = report["geometry"]["mirage"][key]
                    observed = np.mean([r[key] for r in recent])
                    drift = abs(observed - predicted) / max(predicted, 1e-9)
                    if drift > 0.15:
                        failures.append(
                            f"live {key} {observed:.1f} drifts {drift:.0%} from "
                            f"offline {predicted:.1f}"
                        )

    print()
    if failures:
        print("ATTENTION:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("All parity checks passed.")
    report["failures"] = failures
    Path("logs").mkdir(exist_ok=True)
    Path("logs/mirage_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
