#!/usr/bin/env python
"""Score saved CNN stages and render selected channel atlases."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


_MODELS = ("resnet50", "convnext_tiny")
_STAGES = ("stage1", "stage2", "stage3", "stage4")
_SEEDS = {
    "resnet50": {
        "stage1": 1001,
        "stage2": 1002,
        "stage3": 1003,
        "stage4": 1004,
    },
    "convnext_tiny": {
        "stage1": 2001,
        "stage2": 2002,
        "stage3": 2003,
        "stage4": 2004,
    },
}
_EPSILON = 1e-6


@dataclass
class StageStatistics:
    maxima: np.ndarray
    mean_maximum: np.ndarray
    std_maximum: np.ndarray
    selectivity: np.ndarray
    threshold_99: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score saved CNN stages and render channel grids."
    )
    parser.add_argument("--activation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--n-selective", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-mb", type=float, default=400.0)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stage_stack(
    activation_records: list[dict[str, object]],
    stage: str,
) -> np.ndarray:
    arrays = []
    for record in activation_records:
        path = Path(str(record["activation_path"]))
        with np.load(path, allow_pickle=False) as data:
            array = data[stage].astype(np.float32, copy=True)
        if not np.isfinite(array).all():
            raise RuntimeError("nonfinite activation values in %s" % path)
        arrays.append(array)
    return np.stack(arrays, axis=0)


def calculate_stage_statistics(
    activations: np.ndarray,
    top_k: int = 9,
) -> StageStatistics:
    if activations.ndim != 4:
        raise ValueError(
            "expected [N,C,H,W], got %s" % (activations.shape,)
        )
    n_images, channels, _, _ = activations.shape
    if not 1 <= top_k <= n_images:
        raise ValueError("top_k must be between one and the image count")

    maxima = activations.reshape(n_images, channels, -1).max(axis=2)
    mean_maximum = maxima.mean(axis=0)
    std_maximum = maxima.std(axis=0)
    top_values = np.partition(
        maxima,
        kth=n_images - top_k,
        axis=0,
    )[-top_k:, :]
    selectivity = (
        top_values.mean(axis=0) - mean_maximum
    ) / (std_maximum + _EPSILON)
    selectivity = np.where(
        std_maximum > _EPSILON,
        selectivity,
        -np.inf,
    ).astype(np.float32)

    channel_values = activations.transpose(1, 0, 2, 3).reshape(
        channels, -1
    )
    threshold_99 = np.quantile(
        channel_values,
        q=0.99,
        axis=1,
        method="linear",
    ).astype(np.float32)
    return StageStatistics(
        maxima=maxima.astype(np.float32, copy=False),
        mean_maximum=mean_maximum.astype(np.float32, copy=False),
        std_maximum=std_maximum.astype(np.float32, copy=False),
        selectivity=selectivity,
        threshold_99=threshold_99,
    )


def save_stage_statistics(path: Path, stats: StageStatistics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maxima=stats.maxima,
        mean_maximum=stats.mean_maximum,
        std_maximum=stats.std_maximum,
        selectivity=stats.selectivity,
        threshold_99=stats.threshold_99,
    )


def select_channels(
    selectivity: np.ndarray,
    random_seed: int,
    n_selective: int = 20,
    n_random: int = 20,
) -> dict[str, list[int]]:
    if selectivity.ndim != 1:
        raise ValueError("selectivity must be one-dimensional")
    channels = len(selectivity)
    if n_selective + n_random > channels:
        raise ValueError("not enough channels for the requested selection")

    selective = np.argsort(
        -selectivity,
        kind="stable",
    )[:n_selective]
    if not np.isfinite(selectivity[selective]).all():
        raise RuntimeError("not enough finite selective channels")

    remaining = np.setdiff1d(
        np.arange(channels),
        selective,
        assume_unique=False,
    )
    rng = np.random.default_rng(random_seed)
    random_channels = np.sort(
        rng.choice(remaining, size=n_random, replace=False)
    )
    return {
        "selective": selective.astype(int).tolist(),
        "random": random_channels.astype(int).tolist(),
    }


def normalized_heatmap(activation: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(activation).float()[None, None]
    resized = F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    minimum = float(resized.min())
    maximum = float(resized.max())
    return (resized - minimum) / (maximum - minimum + 1e-8)


def channel_region(
    activation: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if activation.ndim != 2:
        raise ValueError("expected [H,W], got %s" % (activation.shape,))
    if not np.isfinite(activation).all():
        raise RuntimeError("activation contains nonfinite values")

    heatmap_224 = normalized_heatmap(activation)
    native_region = activation > threshold
    max_row, max_col = np.unravel_index(
        int(np.argmax(activation)),
        activation.shape,
    )
    if not native_region[max_row, max_col]:
        return heatmap_224, None

    labels, _ = ndimage.label(native_region)
    component_id = labels[max_row, max_col]
    component = labels == component_id
    component_tensor = torch.from_numpy(
        component.astype(np.float32)
    )[None, None]
    region_224 = F.interpolate(
        component_tensor,
        size=(224, 224),
        mode="nearest",
    )[0, 0].numpy() >= 0.5
    return heatmap_224, region_224


def region_box(region: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(region)
    if not len(rows):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def render_overlay(
    crop: Image.Image,
    heatmap: np.ndarray,
    region: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    image = np.asarray(crop, dtype=np.float32) / 255.0
    color = colormaps["magma"](heatmap)[..., :3]
    overlay = Image.fromarray(
        np.clip((0.55 * image + 0.45 * color) * 255.0, 0, 255).astype(
            np.uint8
        )
    )
    box = region_box(region)
    if box is None:
        raise RuntimeError("active region has no bounding box")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (box[0], box[1], box[2] - 1, box[3] - 1),
        outline=(0, 255, 255),
        width=3,
    )
    region_crop = crop.crop(box)
    region_crop = ImageOps.pad(
        region_crop,
        (224, 224),
        method=Image.Resampling.LANCZOS,
        color="white",
    )
    return overlay, region_crop


def render_channel_grid(
    model_name: str,
    stage: str,
    channel: int,
    reason: str,
    threshold: float,
    activations: np.ndarray,
    maxima: np.ndarray,
    image_records: list[dict[str, object]],
    output_path: Path,
    top_k: int,
) -> dict[str, object]:
    active = np.flatnonzero(maxima[:, channel] > threshold)
    ranked = active[
        np.argsort(-maxima[active, channel], kind="stable")
    ][:top_k]

    cell_width = 448
    cell_height = 252
    title_height = 28
    grid = Image.new(
        "RGB",
        (cell_width * 3, title_height + cell_height * 3),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    draw.text(
        (6, 7),
        "%s %s channel %04d (%s), active %d/50"
        % (model_name, stage, channel, reason, len(active)),
        fill="black",
    )
    rendered = 0
    for position, image_index in enumerate(ranked):
        activation = activations[image_index, channel]
        heatmap, region = channel_region(activation, threshold)
        if region is None:
            continue
        crop_path = Path(str(image_records[image_index]["preprocessed_path"]))
        with Image.open(crop_path) as image:
            crop = image.convert("RGB")
        overlay, region_crop = render_overlay(crop, heatmap, region)
        row = rendered // 3
        column = rendered % 3
        x = column * cell_width
        y = title_height + row * cell_height
        grid.paste(overlay, (x, y))
        grid.paste(region_crop, (x + 224, y))
        draw.text(
            (x + 3, y + 228),
            "#%d image %02d max %.4f threshold %.4f"
            % (
                rendered + 1,
                image_index,
                maxima[image_index, channel],
                threshold,
            ),
            fill="black",
        )
        rendered += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="JPEG", quality=86, optimize=True)
    return {
        "channel": channel,
        "reason": reason,
        "active_image_count": int(len(active)),
        "rendered_image_count": rendered,
        "grid_path": os.path.abspath(output_path),
    }


def render_review_sheets(
    model_name: str,
    stage: str,
    channel_payloads: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    output_paths = []
    for sheet_index in range(0, len(channel_payloads), 10):
        entries = channel_payloads[sheet_index : sheet_index + 10]
        sheet = Image.new("RGB", (1600, 2380), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (8, 8),
            "%s %s channels %d-%d"
            % (
                model_name,
                stage,
                sheet_index + 1,
                sheet_index + len(entries),
            ),
            fill="black",
        )
        for position, entry in enumerate(entries):
            with Image.open(entry["grid_path"]) as grid:
                thumbnail = grid.convert("RGB").resize(
                    (780, 450),
                    Image.Resampling.LANCZOS,
                )
            column = position % 2
            row = position // 2
            x = column * 800 + 10
            y = row * 470 + 30
            sheet.paste(thumbnail, (x, y))
            draw.text(
                (x, y + 452),
                "channel %04d %s"
                % (entry["channel"], entry["reason"]),
                fill="black",
            )
        output_path = output_dir / (
            "%s_%s_%d.jpg"
            % (model_name, stage, sheet_index // 10 + 1)
        )
        sheet.save(output_path, format="JPEG", quality=78, optimize=True)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    args = parse_args()
    activation_root = Path(args.activation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    max_output_bytes = int(args.max_output_mb * 1024 * 1024)

    extraction_run = load_json(activation_root / "run.json")
    image_manifest = load_json(Path(extraction_run["manifest_path"]))
    image_records = image_manifest["images"]
    total_bytes = 0
    total_grids = 0
    review_paths = []

    atomic_write_json(
        output_dir / "run.json",
        {
            "activation_root": os.path.abspath(activation_root),
            "image_count": len(image_records),
            "models": list(_MODELS),
            "stages": list(_STAGES),
            "top_k": args.top_k,
            "n_selective": args.n_selective,
            "n_random": args.n_random,
            "threshold_quantile": 0.99,
            "threshold_comparison": ">",
            "heatmap_interpolation": "bilinear",
            "region_interpolation": "nearest",
            "seeds": _SEEDS,
        },
    )

    for model_name in _MODELS:
        activation_manifest = load_json(
            activation_root / model_name / "activation_manifest.json"
        )
        activation_records = activation_manifest["images"]
        if len(activation_records) != len(image_records):
            raise RuntimeError("activation and image manifest counts differ")
        model_selection = {}

        for stage in _STAGES:
            if time.monotonic() - started_at > args.max_seconds:
                raise TimeoutError("processing exceeded the time limit")
            activations = load_stage_stack(activation_records, stage)
            stats = calculate_stage_statistics(activations, args.top_k)
            statistics_path = (
                output_dir / model_name / "statistics" / ("%s.npz" % stage)
            )
            save_stage_statistics(statistics_path, stats)

            selected = select_channels(
                stats.selectivity,
                _SEEDS[model_name][stage],
                args.n_selective,
                args.n_random,
            )
            model_selection[stage] = {
                "seed": _SEEDS[model_name][stage],
                **selected,
            }
            channel_payloads = []
            for reason in ("selective", "random"):
                for channel in selected[reason]:
                    output_path = (
                        output_dir
                        / model_name
                        / stage
                        / ("%s_channel_%04d.jpg" % (reason, channel))
                    )
                    payload = render_channel_grid(
                        model_name,
                        stage,
                        channel,
                        reason,
                        float(stats.threshold_99[channel]),
                        activations,
                        stats.maxima,
                        image_records,
                        output_path,
                        args.top_k,
                    )
                    channel_payloads.append(payload)
                    total_grids += 1
                    total_bytes += output_path.stat().st_size
                    if total_bytes > max_output_bytes:
                        raise RuntimeError("processing exceeded output limit")

            atomic_write_json(
                output_dir / model_name / stage / "channels.json",
                {
                    "model": model_name,
                    "stage": stage,
                    "threshold_quantile": 0.99,
                    "channels": channel_payloads,
                },
            )
            if stage == "stage3":
                review_dir = output_dir / "review_sheets"
                review_dir.mkdir(parents=True, exist_ok=True)
                paths = render_review_sheets(
                    model_name,
                    stage,
                    channel_payloads,
                    review_dir,
                )
                review_paths.extend(paths)
                total_bytes += sum(path.stat().st_size for path in paths)

            print(
                "[%s] %s complete: %d channel grids"
                % (model_name, stage, len(channel_payloads)),
                flush=True,
            )
            del activations, stats

        atomic_write_json(
            output_dir / model_name / "selected_channels.json",
            model_selection,
        )

    atomic_write_json(
        output_dir / "complete.json",
        {
            "elapsed_seconds": time.monotonic() - started_at,
            "channel_grid_count": total_grids,
            "review_sheet_count": len(review_paths),
            "output_bytes_counted": total_bytes,
        },
    )
    print("Saved channel atlas to %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
