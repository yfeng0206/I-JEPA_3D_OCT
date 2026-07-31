#!/usr/bin/env python
"""Extract four frozen CNN stages for manifest images, one image at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_PROJECT_ROOT))

from src.guides.cnn_stages import (  # noqa: E402
    ConvNeXtBaseStages,
    ConvNeXtTinyStages,
    ResNet50Stages,
    ResNeXt10132X8DStages,
)


_STAGES = ("stage1", "stage2", "stage3", "stage4")
_MODEL_NAMES = (
    "resnet50",
    "convnext_tiny",
    "resnext101_32x8d",
    "convnext_base",
)
_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract four frozen CNN stages one image at a time."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=_MODEL_NAMES,
        default=("resnet50", "convnext_tiny"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-mb", type=float, default=400.0)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_manifest(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a nonempty images list")
    if payload.get("count") != len(records):
        raise ValueError("manifest count does not match its image list")
    return payload, records


def load_manifest_crop(record: dict[str, object]) -> tuple[Tensor, Image.Image]:
    path = Path(str(record["preprocessed_path"]))
    if not path.is_file():
        raise FileNotFoundError("manifest crop is missing: %s" % path)
    expected_hash = str(record["preprocessed_sha256"])
    if file_sha256(path) != expected_hash:
        raise RuntimeError("manifest crop hash mismatch: %s" % path)
    with Image.open(path) as image:
        crop = image.convert("RGB").copy()
    if crop.size != (224, 224):
        raise ValueError("manifest crop must be 224x224: %s" % path)
    tensor = TF.pil_to_tensor(crop).to(torch.float32).div_(255.0)
    tensor = (tensor - _MEAN) / _STD
    return tensor, crop


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_shapes(
    arrays: dict[str, np.ndarray],
    expected_shapes: dict[str, tuple[int, int, int]],
) -> None:
    if tuple(arrays) != _STAGES:
        raise RuntimeError("model did not return Stage 1-4 in order")
    actual = {stage: tuple(array.shape) for stage, array in arrays.items()}
    if actual != expected_shapes:
        raise RuntimeError(
            "unexpected stage shapes: expected %s, got %s"
            % (expected_shapes, actual)
        )


def extract_model_activations(
    model: nn.Module,
    model_name: str,
    records: list[dict[str, object]],
    root: Path,
    device: torch.device,
    started_at: float,
    max_seconds: float,
    max_output_bytes: int,
    bytes_written: int,
) -> int:
    activation_dir = root / model_name / "activations"
    activation_dir.mkdir(parents=True, exist_ok=False)
    model = model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output_records = []

    for index, record in enumerate(records):
        if time.monotonic() - started_at > max_seconds:
            raise TimeoutError("extraction exceeded the configured time limit")

        image_id = "%02d_%s" % (
            index,
            str(record["preprocessed_sha256"])[:12],
        )
        tensor, _ = load_manifest_crop(record)
        x = tensor.unsqueeze(0).to(device)

        with torch.inference_mode():
            features = model(x)

        arrays = {
            stage: (
                features[stage][0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            for stage in _STAGES
        }
        validate_shapes(arrays, model.expected_shapes)

        output_path = activation_dir / ("%s.npz" % image_id)
        save_npz_atomic(output_path, arrays)
        file_bytes = output_path.stat().st_size
        bytes_written += file_bytes
        if bytes_written > max_output_bytes:
            raise RuntimeError("extraction exceeded the output-size limit")

        output_records.append(
            {
                "index": index,
                "image_id": image_id,
                "crop_path": str(record["preprocessed_path"]),
                "crop_sha256": str(record["preprocessed_sha256"]),
                "activation_path": os.path.abspath(output_path),
                "activation_bytes": file_bytes,
                "dtype": "float32",
                "shapes": {
                    stage: list(arrays[stage].shape) for stage in _STAGES
                },
            }
        )
        atomic_write_json(
            root / model_name / "activation_manifest.json",
            {
                "model": model_name,
                "weights": model.weights_name,
                "parameter_count": parameter_count,
                "count": len(output_records),
                "images": output_records,
            },
        )

        del x, features, arrays, tensor
        print(
            "[%s] %02d/%02d %s"
            % (model_name, index + 1, len(records), image_id),
            flush=True,
        )

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bytes_written


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0.0 or args.max_output_mb <= 0.0:
        raise ValueError("resource limits must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must not contain duplicates")

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_payload, records = load_manifest(manifest_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started_at = time.monotonic()
    bytes_written = 0
    max_output_bytes = int(args.max_output_mb * 1024 * 1024)
    model_factories = {
        "resnet50": ResNet50Stages,
        "convnext_tiny": ConvNeXtTinyStages,
        "resnext101_32x8d": ResNeXt10132X8DStages,
        "convnext_base": ConvNeXtBaseStages,
    }

    atomic_write_json(
        output_dir / "run.json",
        {
            "manifest_path": os.path.abspath(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_count": len(records),
            "manifest_experiment_id": manifest_payload.get("experiment_id"),
            "models": list(args.models),
            "device": str(device),
            "dtype": "float32",
            "processing": "one_image_at_a_time",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "max_seconds": args.max_seconds,
            "max_output_mb": args.max_output_mb,
        },
    )

    for model_name in args.models:
        bytes_written = extract_model_activations(
            model_factories[model_name](),
            model_name,
            records,
            output_dir,
            device,
            started_at,
            args.max_seconds,
            max_output_bytes,
            bytes_written,
        )

    atomic_write_json(
        output_dir / "complete.json",
        {
            "image_count": len(records),
            "model_count": len(args.models),
            "activation_file_count": len(records) * len(args.models),
            "output_bytes": bytes_written,
            "elapsed_seconds": time.monotonic() - started_at,
        },
    )
    print("Saved activations to %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
