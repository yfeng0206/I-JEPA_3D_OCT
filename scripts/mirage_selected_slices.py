#!/usr/bin/env python
"""Run constrained MIRAGE masking on specific OCT PNG slices.

Segments each image with the trained MIRAGE-Large GOALS head, repairs the
retinal envelope, expands it, then renders the constrained target masking at
several block-fill thresholds so they can be compared on the same slice.

Must run with the MIRAGE environment:
    D:\\jepa_phase0\\mirage-goals\\.venv\\Scripts\\python.exe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
MIRAGE_WORKSPACE = Path(r"D:\jepa_phase0\mirage-goals")
if str(MIRAGE_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(MIRAGE_WORKSPACE))

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    build_union,
    dilate_patch_grid,
    expand_envelope,
    patch_occupancy,
    repair_union,
)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mirage_constrained_masking",
    Path(_PROJECT_ROOT) / "scripts" / "mirage_constrained_masking.py",
)
constrained = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(constrained)

OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_selected_slices")
CROP, PATCH = 256, 16
GRID = CROP // PATCH
TILE, HEADER, ROW_LABEL = 256, 20, 150


@torch.inference_mode()
def segment(model, image_200: np.ndarray, device: str) -> np.ndarray:
    """Native min-max -> bilinear 1024 -> MIRAGE argmax -> nearest 200."""
    values = image_200.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / (high - low) if high > low else np.zeros_like(values)
    resized = cv2.resize(normalized, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized)[None, None].to(device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        logits = model({"bscan": tensor})["semseg"]
    hard = torch.softmax(logits.float(), dim=1).argmax(dim=1)[:, None].float()
    native = torch.nn.functional.interpolate(hard, size=(200, 200), mode="nearest-exact")
    return native[0, 0].to(torch.uint8).cpu().numpy()


def run_threshold_sweep(args, model, device) -> None:
    """Render the proposed masking at several patch-occupancy thresholds."""
    from src.guides.mirage_envelope import dilate_patch_grid  # noqa: F401
    from src.masks.curriculum import CurriculumMaskGenerator

    thresholds = args.thresholds
    titles = ["Original OCT", "MIRAGE envelope"] + [
        f"threshold {t:.2f}" for t in thresholds
    ]
    rows, records = [], []
    for index, image_path in enumerate(args.images):
        image_200 = np.asarray(Image.open(image_path).convert("L"))
        labels = segment(model, image_200, device)
        envelope, valid, _stats = repair_union(
            build_union(labels), params=DEFAULT_REPAIR
        )
        envelope_256 = (
            np.asarray(
                Image.fromarray(envelope.astype(np.uint8) * 255, mode="L").resize(
                    (CROP, CROP), Image.Resampling.NEAREST
                )
            )
            > 127
        )
        occupancy = patch_occupancy(envelope_256, patch_size=PATCH)
        slice_256 = np.asarray(
            Image.fromarray(image_200, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )

        base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
        region_px = np.kron(occupancy >= 0.5, np.ones((PATCH, PATCH), dtype=bool))
        tint = np.array([40.0, 220.0, 90.0], dtype=np.float32)
        alpha = 0.35 * region_px[..., None].astype(np.float32)
        columns = [
            Image.fromarray(slice_256, mode="L").convert("RGB"),
            Image.fromarray(
                np.clip(base * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8),
                mode="RGB",
            ),
        ]
        samples = []
        for threshold in thresholds:
            generator = CurriculumMaskGenerator(
                input_size=(CROP, CROP),
                patch_size=PATCH,
                enc_mask_scale=(0.85, 1.0),
                pred_mask_scale=(0.15, 0.2),
                aspect_ratio=(0.75, 1.5),
                nenc=1,
                npred=4,
                min_keep=10,
                allow_overlap=False,
                curriculum_cfg={
                    "mode": "mirage_envelope", "T_warm": 25, "T_total": 30,
                    "r_max": 1.0, "mirage_min_block_fill": args.fill,
                    "mirage_min_retina_visible": args.visible,
                    "mirage_occupancy_threshold": threshold,
                    "mirage_spread": False,
                },
            )
            generator.set_epoch(30)
            region = (occupancy >= threshold).astype(np.float32)
            guide = torch.from_numpy(
                np.stack([occupancy.astype(np.float32), region])
            )[None]
            torch.manual_seed(args.seed + index)
            import random as _random

            _random.seed(args.seed + index)
            _enc, pred = generator.generate(
                batch_size=1, guide_grids=guide,
                guide_valid=torch.ones(1, dtype=torch.bool),
            )
            union = sorted({j for g in pred for j in g[0].tolist()})
            coverage = float(np.mean([occupancy.reshape(-1)[i] for i in union]))
            stats = generator.mirage_stats

            view = base.copy()
            extra = np.kron(
                (occupancy >= threshold) & (occupancy < 0.5),
                np.ones((PATCH, PATCH), dtype=bool),
            )
            amber = np.array([250.0, 190.0, 40.0], dtype=np.float32)
            view = view * (1 - 0.35 * region_px[..., None]) + tint * (
                0.35 * region_px[..., None]
            )
            view = view * (1 - 0.32 * extra[..., None]) + amber * (
                0.32 * extra[..., None]
            )
            panel_image = Image.fromarray(
                np.clip(view, 0, 255).astype(np.uint8), mode="RGB"
            )
            drawer = ImageDraw.Draw(panel_image, "RGBA")
            for group in pred:
                indices = group[0].tolist()
                rows_i = [i // GRID for i in indices]
                cols_i = [i % GRID for i in indices]
                drawer.rectangle(
                    [min(cols_i) * PATCH, min(rows_i) * PATCH,
                     (max(cols_i) + 1) * PATCH - 1, (max(rows_i) + 1) * PATCH - 1],
                    fill=(255, 60, 60, 85), outline=(255, 60, 60),
                )
            drawer.text((5, 5), f"retina in targets {coverage * 100:.0f}%",
                        fill=(255, 240, 120))
            columns.append(panel_image)
            samples.append({
                "threshold": threshold,
                "coverage": coverage,
                "region_cells": int((occupancy >= threshold).sum()),
                "accept": stats["accept_rate"],
                "fill": stats["mean_block_fill"],
                "visible": stats["retina_visible"],
            })
            records.append({"image": str(image_path), **samples[-1]})
        rows.append({
            "name": f"{image_path.parent.parent.name}/{image_path.parent.name}",
            "columns": columns, "samples": samples,
            "valid": bool(valid),
            "region": int((occupancy >= 0.5).sum()),
        })

    panel = Image.new(
        "RGB",
        (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        f"Proposed masking (fill >= {args.fill:.0%}, retina visible >= "
        f"{args.visible:.0%}, no dilation).  GREEN = MIRAGE retina, AMBER = "
        "boundary patches admitted by the looser threshold, RED = target blocks.",
        fill="black",
    )
    for column, name in enumerate(titles):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["name"], fill="black")
        draw.text((4, top + 22), f"retina {row['region']}/256", fill="black")
        draw.text((4, top + 38), "guide OK" if row["valid"] else "guide INVALID",
                  fill="green" if row["valid"] else "red")
        for column, image in enumerate(row["columns"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
        for column, entry in enumerate(row["samples"]):
            x = ROW_LABEL + (column + 2) * TILE + 4
            draw.text((x, top + TILE - 32),
                      f"region {entry['region_cells']}/256", fill=(255, 210, 120))
            draw.text((x, top + TILE - 18),
                      f"accept {entry['accept']:.2f}  fill {entry['fill']:.2f}",
                      fill=(255, 240, 120))
    panel.save(args.output / "threshold_masking.png", optimize=True)
    (args.output / "threshold_masking.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    summary = {}
    for threshold in thresholds:
        subset = [r for r in records if r["threshold"] == threshold]
        summary[f"thr{threshold}"] = {
            "mean_retina_in_targets": round(
                float(np.mean([r["coverage"] for r in subset])), 4
            ),
            "mean_region_cells": round(
                float(np.mean([r["region_cells"] for r in subset])), 1
            ),
            "mean_accept": round(float(np.mean([r["accept"] for r in subset])), 3),
        }
    print(json.dumps(summary, indent=2))


def run_patch_dilation_sweep(args, model, device, generator) -> None:
    """Grow the region by whole patches, optionally across several fill rules."""
    fills = args.fills if args.fills else [args.fill]
    combos = [(p, f) for p in args.dilate_patches for f in fills]
    titles = ["Original OCT"] + [
        f"+{p}p / fill {f:.0%}" for p, f in combos
    ]
    rows, records = [], []
    for index, image_path in enumerate(args.images):
        image_200 = np.asarray(Image.open(image_path).convert("L"))
        labels = segment(model, image_200, device)
        envelope, valid, _stats = repair_union(
            build_union(labels), params=DEFAULT_REPAIR
        )
        envelope_256 = (
            np.asarray(
                Image.fromarray(envelope.astype(np.uint8) * 255, mode="L").resize(
                    (CROP, CROP), Image.Resampling.NEAREST
                )
            )
            > 127
        )
        occupancy = patch_occupancy(envelope_256, patch_size=PATCH)
        base_region = occupancy >= constrained.MIN_OCCUPANCY
        slice_256 = np.asarray(
            Image.fromarray(image_200, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )
        columns = [Image.fromarray(slice_256, mode="L").convert("RGB")]
        samples, region_sizes = [], []
        for patches, fill in combos:
            placement = dilate_patch_grid(base_region, patches)
            sample = constrained.sample_constrained(
                generator,
                occupancy,
                args.seed + index * 17,
                fill,
                args.visible,
                placement_region=placement.astype(float),
            )
            columns.append(
                constrained.render(
                    slice_256, placement.astype(np.float32), sample
                )
            )
            samples.append(sample)
            region_sizes.append(int(placement.sum()))
            records.append(
                {
                    "image": str(image_path),
                    "dilate_patches": patches,
                    "fill_threshold": fill,
                    "mirage_cells": int(base_region.sum()),
                    "placement_cells": int(placement.sum()),
                    "guide_valid": bool(valid),
                    "accepted": bool(sample["accepted"]),
                    "fallback": bool(sample.get("fallback", False)),
                    "retina_visible": round(sample["retina_visible"], 4),
                    "mean_block_fill": round(sample["mean_fill"], 4),
                    "masked_area": round(len(sample["union"]) / 256.0, 4),
                    "attempts": sample["attempts"],
                }
            )
        rows.append(
            {
                "name": f"{image_path.parent.parent.name}/{image_path.parent.name}",
                "columns": columns,
                "samples": samples,
                "region_sizes": region_sizes,
                "guide_valid": bool(valid),
                "mirage_cells": int(base_region.sum()),
            }
        )

    panel = Image.new(
        "RGB",
        (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        f"Patch-level region dilation at block fill >= {args.fill:.0%}, "
        f"retina visible >= {args.visible:.0%};  green = placement region, "
        "red = target blocks (retina visibility always measured on the TRUE MIRAGE region)",
        fill="black",
    )
    for column, name in enumerate(titles):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["name"], fill="black")
        draw.text((4, top + 22), f"MIRAGE {row['mirage_cells']}/256", fill="black")
        draw.text(
            (4, top + 38),
            "guide OK" if row["guide_valid"] else "guide INVALID",
            fill="green" if row["guide_valid"] else "red",
        )
        for column, image in enumerate(row["columns"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
        for column, sample in enumerate(row["samples"]):
            x = ROW_LABEL + (column + 1) * TILE + 4
            draw.text(
                (x, top + TILE - 46),
                "accepted" if sample["accepted"] else "RELAXED",
                fill=(120, 255, 160) if sample["accepted"] else (255, 110, 110),
            )
            draw.text(
                (x, top + TILE - 32),
                f"placement {row['region_sizes'][column]}/256",
                fill=(160, 255, 200),
            )
            draw.text(
                (x, top + TILE - 18),
                f"vis {sample['retina_visible'] * 100:.0f}%  "
                f"fill {sample['mean_fill'] * 100:.0f}%",
                fill=(255, 240, 120),
            )
    panel.save(args.output / "patch_dilation_sweep.png", optimize=True)
    (args.output / "patch_dilation_summary.json").write_text(
        json.dumps({"fill": args.fill, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for patches, fill in combos:
        subset = [
            r
            for r in records
            if r["dilate_patches"] == patches and r["fill_threshold"] == fill
        ]
        summary[f"+{patches}p fill{fill:.0%}"] = {
            "accepted": f"{sum(r['accepted'] for r in subset)}/{len(subset)}",
            "placement_cells": round(
                float(np.mean([r["placement_cells"] for r in subset])), 1
            ),
            "retina_visible": round(
                float(np.mean([r["retina_visible"] for r in subset])), 3
            ),
            "mean_block_fill": round(
                float(np.mean([r["mean_block_fill"] for r in subset])), 3
            ),
            "masked_area": round(float(np.mean([r["masked_area"] for r in subset])), 3),
        }
    print(json.dumps(summary, indent=2))


def run_expansion_sweep(args, model, device, generator) -> None:
    """Grow the MIRAGE envelope in steps at a fixed block-fill threshold."""
    titles = ["Original OCT"] + [f"+{e:.0%} expand" for e in args.expansions]
    rows, records = [], []
    for index, image_path in enumerate(args.images):
        image_200 = np.asarray(Image.open(image_path).convert("L"))
        labels = segment(model, image_200, device)
        envelope, valid, _stats = repair_union(
            build_union(labels), params=DEFAULT_REPAIR
        )
        slice_256 = np.asarray(
            Image.fromarray(image_200, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )
        columns = [Image.fromarray(slice_256, mode="L").convert("RGB")]
        samples, region_sizes = [], []
        for expansion in args.expansions:
            grown, achieved = expand_envelope(envelope, expansion)
            grown_256 = (
                np.asarray(
                    Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                        (CROP, CROP), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
            occupancy = patch_occupancy(grown_256, patch_size=PATCH)
            sample = constrained.sample_constrained(
                generator,
                occupancy,
                args.seed + index * 17,
                args.fill,
                args.visible,
            )
            columns.append(constrained.render(slice_256, occupancy, sample))
            samples.append(sample)
            cells = int((occupancy >= constrained.MIN_OCCUPANCY).sum())
            region_sizes.append(cells)
            records.append(
                {
                    "image": str(image_path),
                    "expansion": expansion,
                    "achieved_expansion": round(achieved, 4),
                    "region_cells": cells,
                    "guide_valid": bool(valid),
                    "accepted": bool(sample["accepted"]),
                    "fallback": bool(sample.get("fallback", False)),
                    "retina_visible": round(sample["retina_visible"], 4),
                    "mean_block_fill": round(sample["mean_fill"], 4),
                    "masked_area": round(len(sample["union"]) / 256.0, 4),
                    "attempts": sample["attempts"],
                }
            )
        rows.append(
            {
                "name": f"{image_path.parent.parent.name}/{image_path.parent.name}",
                "columns": columns,
                "samples": samples,
                "region_sizes": region_sizes,
                "guide_valid": bool(valid),
            }
        )

    panel = Image.new(
        "RGB",
        (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        f"Envelope expansion sweep at block fill >= {args.fill:.0%}, "
        f"retina visible >= {args.visible:.0%};  "
        "green = MIRAGE region, red = target blocks, yellow % = block retina share",
        fill="black",
    )
    for column, name in enumerate(titles):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["name"], fill="black")
        draw.text(
            (4, top + 24),
            "guide OK" if row["guide_valid"] else "guide INVALID",
            fill="green" if row["guide_valid"] else "red",
        )
        for column, image in enumerate(row["columns"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
        for column, sample in enumerate(row["samples"]):
            x = ROW_LABEL + (column + 1) * TILE + 4
            draw.text(
                (x, top + TILE - 46),
                "accepted" if sample["accepted"] else "RELAXED",
                fill=(120, 255, 160) if sample["accepted"] else (255, 110, 110),
            )
            draw.text(
                (x, top + TILE - 32),
                f"region {row['region_sizes'][column]}/256",
                fill=(160, 255, 200),
            )
            draw.text(
                (x, top + TILE - 18),
                f"vis {sample['retina_visible'] * 100:.0f}%  "
                f"fill {sample['mean_fill'] * 100:.0f}%",
                fill=(255, 240, 120),
            )
    panel.save(args.output / "expansion_sweep.png", optimize=True)
    (args.output / "expansion_summary.json").write_text(
        json.dumps({"fill": args.fill, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {}
    for expansion in args.expansions:
        subset = [r for r in records if r["expansion"] == expansion]
        summary[f"+{expansion:.0%}"] = {
            "accepted": f"{sum(r['accepted'] for r in subset)}/{len(subset)}",
            "region_cells": round(float(np.mean([r["region_cells"] for r in subset])), 1),
            "retina_visible": round(
                float(np.mean([r["retina_visible"] for r in subset])), 3
            ),
            "mean_block_fill": round(
                float(np.mean([r["mean_block_fill"] for r in subset])), 3
            ),
            "masked_area": round(float(np.mean([r["masked_area"] for r in subset])), 3),
        }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--fills", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--fill", type=float, default=0.30)
    parser.add_argument(
        "--expansions", type=float, nargs="+", default=None,
        help="Sweep envelope expansion at a fixed fill threshold.",
    )
    parser.add_argument(
        "--dilate-patches", type=int, nargs="+", default=None,
        help="Sweep whole-patch region dilation at a fixed fill threshold.",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Sweep the patch occupancy threshold (no dilation).",
    )
    parser.add_argument("--visible", type=float, default=0.25)
    parser.add_argument("--expansion", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import orchestrate

    model = orchestrate.build_trained_model(device)

    generator = constrained.make_generator()
    if args.thresholds:
        run_threshold_sweep(args, model, device)
        return
    if args.dilate_patches:
        run_patch_dilation_sweep(args, model, device, generator)
        return
    if args.expansions:
        run_expansion_sweep(args, model, device, generator)
        return
    titles = ["Original OCT", "MIRAGE envelope"] + [
        f"fill >= {f:.0%}" for f in args.fills
    ] + ["CASCADE 50/40/30"]
    rows, records = [], []
    for index, image_path in enumerate(args.images):
        image_200 = np.asarray(Image.open(image_path).convert("L"))
        labels = segment(model, image_200, device)
        envelope, valid, stats = repair_union(build_union(labels), params=DEFAULT_REPAIR)
        grown, achieved = expand_envelope(envelope, args.expansion)
        grown_256 = (
            np.asarray(
                Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                    (CROP, CROP), Image.Resampling.NEAREST
                )
            )
            > 127
        )
        occupancy = patch_occupancy(grown_256, patch_size=PATCH)
        slice_256 = np.asarray(
            Image.fromarray(image_200, mode="L").resize((CROP, CROP), Image.BILINEAR)
        )

        region_view = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
        region_pixels = np.kron(
            occupancy >= constrained.MIN_OCCUPANCY,
            np.ones((PATCH, PATCH), dtype=bool),
        )
        tint = np.array([40.0, 220.0, 90.0], dtype=np.float32)
        alpha = 0.35 * region_pixels[..., None].astype(np.float32)
        region_image = Image.fromarray(
            np.clip(region_view * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8),
            mode="RGB",
        )

        columns = [
            Image.fromarray(slice_256, mode="L").convert("RGB"),
            region_image,
        ]
        samples = []
        for fill in args.fills:
            sample = constrained.sample_constrained(
                generator, occupancy, args.seed + index * 17, fill, args.visible
            )
            columns.append(constrained.render(slice_256, occupancy, sample))
            samples.append(sample)
            records.append(
                {
                    "image": str(image_path),
                    "fill_threshold": fill,
                    "accepted": bool(sample["accepted"]),
                    "retina_visible": round(sample["retina_visible"], 4),
                    "mean_block_fill": round(sample["mean_fill"], 4),
                    "min_block_fill": round(sample["min_fill"], 4),
                    "masked_area": round(len(sample["union"]) / 256.0, 4),
                    "attempts": sample["attempts"],
                }
            )
        cascade = constrained.sample_cascaded(
            generator,
            occupancy,
            args.seed + index * 17,
            min_retina_visible=args.visible,
        )
        columns.append(constrained.render(slice_256, occupancy, cascade))
        samples.append(cascade)
        records.append(
            {
                "image": str(image_path),
                "fill_threshold": f"cascade->{cascade.get('fill_threshold_used')}",
                "accepted": bool(cascade["accepted"]),
                "retina_visible": round(cascade["retina_visible"], 4),
                "mean_block_fill": round(cascade["mean_fill"], 4),
                "min_block_fill": round(cascade["min_fill"], 4),
                "masked_area": round(len(cascade["union"]) / 256.0, 4),
                "attempts": cascade["attempts"],
            }
        )
        rows.append(
            {
                "name": f"{image_path.parent.parent.name}/{image_path.parent.name}",
                "columns": columns,
                "samples": samples,
                "guide_valid": bool(valid),
                "region_cells": int((occupancy >= constrained.MIN_OCCUPANCY).sum()),
                "achieved_expansion": round(achieved, 4),
            }
        )

    panel = Image.new(
        "RGB",
        (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        f"Constrained MIRAGE masking   envelope +{args.expansion:.0%}, "
        f"retina visible >= {args.visible:.0%};  yellow % = share of block that is retina",
        fill="black",
    )
    for column, name in enumerate(titles):
        draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
    for index, row in enumerate(rows):
        top = HEADER * 2 + index * TILE
        draw.text((4, top + 6), row["name"], fill="black")
        draw.text((4, top + 22), f"region {row['region_cells']}/256", fill="black")
        draw.text(
            (4, top + 38),
            "guide OK" if row["guide_valid"] else "guide INVALID",
            fill="green" if row["guide_valid"] else "red",
        )
        for column, image in enumerate(row["columns"]):
            panel.paste(image, (ROW_LABEL + column * TILE, top))
        for column, sample in enumerate(row["samples"]):
            x = ROW_LABEL + (column + 2) * TILE + 4
            draw.text(
                (x, top + TILE - 34),
                "accepted" if sample["accepted"] else "RELAXED",
                fill=(120, 255, 160) if sample["accepted"] else (255, 110, 110),
            )
            draw.text(
                (x, top + TILE - 20),
                f"retina vis {sample['retina_visible'] * 100:.0f}%  "
                f"fill {sample['mean_fill'] * 100:.0f}%",
                fill=(255, 240, 120),
            )
    panel.save(args.output / "selected_slices.png", optimize=True)
    (args.output / "summary.json").write_text(
        json.dumps({"visible": args.visible, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
