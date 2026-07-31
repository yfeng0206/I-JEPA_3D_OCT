#!/usr/bin/env python
"""Render one combined heatmap and overlay for every saved CNN stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


_MODELS = ("resnet50", "convnext_tiny")
_STAGES = ("stage1", "stage2", "stage3", "stage4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-image combined CNN stage maps."
    )
    parser.add_argument("--activation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--max-output-mb", type=float, default=200.0)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def combined_stage_map(activation: np.ndarray) -> np.ndarray:
    if activation.ndim != 3:
        raise ValueError("expected [C,H,W], got %s" % (activation.shape,))
    if not np.isfinite(activation).all():
        raise RuntimeError("activation contains nonfinite values")

    energy = np.sqrt(
        np.mean(np.square(activation, dtype=np.float32), axis=0)
    )
    tensor = torch.from_numpy(energy)[None, None]
    resized = F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    minimum = float(resized.min())
    maximum = float(resized.max())
    return (resized - minimum) / (maximum - minimum + 1e-8)


def render_heatmap(values: np.ndarray) -> Image.Image:
    colors = colormaps["magma"](values)[..., :3]
    return Image.fromarray(
        np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    )


def main() -> None:
    args = parse_args()
    activation_root = Path(args.activation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    max_output_bytes = int(args.max_output_mb * 1024 * 1024)
    output_bytes = 0
    output_files = 0

    atomic_write_json(
        output_dir / "run.json",
        {
            "activation_root": os.path.abspath(activation_root),
            "models": list(_MODELS),
            "stages": list(_STAGES),
            "combined_map": "sqrt(mean(channel_activation_squared))",
            "resize": "bilinear",
            "display_normalization": "per-image-stage minmax",
            "overlay_heatmap_weight": 0.45,
        },
    )

    for model_name in _MODELS:
        manifest = load_json(
            activation_root / model_name / "activation_manifest.json"
        )
        records = manifest["images"]
        for position, record in enumerate(records):
            if time.monotonic() - started_at > args.max_seconds:
                raise TimeoutError("stage-map rendering exceeded time limit")

            image_dir = output_dir / model_name / str(record["image_id"])
            image_dir.mkdir(parents=True, exist_ok=False)
            original_path = image_dir / "original.png"
            shutil.copyfile(record["crop_path"], original_path)
            output_bytes += original_path.stat().st_size
            output_files += 1

            with Image.open(original_path) as image:
                original = image.convert("RGB")
            with np.load(
                record["activation_path"],
                allow_pickle=False,
            ) as activations:
                for stage in _STAGES:
                    stage_dir = image_dir / stage
                    stage_dir.mkdir()
                    values = combined_stage_map(
                        activations[stage].astype(np.float32, copy=False)
                    )
                    heatmap = render_heatmap(values)
                    overlay = Image.blend(original, heatmap, 0.45)
                    heatmap_path = stage_dir / "heatmap.png"
                    overlay_path = stage_dir / "overlay.png"
                    heatmap.save(
                        heatmap_path,
                        format="PNG",
                        optimize=True,
                    )
                    overlay.save(
                        overlay_path,
                        format="PNG",
                        optimize=True,
                    )
                    output_bytes += (
                        heatmap_path.stat().st_size
                        + overlay_path.stat().st_size
                    )
                    output_files += 2

            if output_bytes > max_output_bytes:
                raise RuntimeError("stage maps exceeded output-size limit")

        print(
            "[%s] rendered %d images" % (model_name, len(records)),
            flush=True,
        )

    atomic_write_json(
        output_dir / "complete.json",
        {
            "elapsed_seconds": time.monotonic() - started_at,
            "model_count": len(_MODELS),
            "image_count_per_model": 50,
            "image_file_count": output_files,
            "output_bytes": output_bytes,
        },
    )
    print("Saved stage maps to %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
