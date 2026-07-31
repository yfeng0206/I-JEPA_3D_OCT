#!/usr/bin/env python
"""Build Stage-3 to Stage-2 spatial feature hierarchies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


_MODEL_NAMES = (
    "resnet50",
    "convnext_tiny",
    "resnext101_32x8d",
    "convnext_base",
)
_DEFAULT_MODELS = ("resnet50", "convnext_tiny")
_MODEL_SEED_OFFSETS = {
    "resnet50": 0,
    "resnext101_32x8d": 0,
    "convnext_tiny": 100_000,
    "convnext_base": 100_000,
}
_COARSE_CLUSTERS = 4
_CHILDREN_PER_PARENT = 3
_KMEANS_RESTARTS = 20
_OUTPUT_SIZE = (224, 224)
_OVERLAY_WEIGHT = 0.45
_LEVEL1_COLORS = np.asarray(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (148, 103, 189),
    ],
    dtype=np.uint8,
)
_LEVEL2_COLORS = np.asarray(
    [
        (31, 119, 180),
        (102, 166, 205),
        (8, 81, 156),
        (255, 127, 14),
        (255, 187, 120),
        (230, 85, 13),
        (44, 160, 44),
        (116, 196, 118),
        (0, 109, 44),
        (148, 103, 189),
        (188, 128, 189),
        (106, 61, 154),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Stage 3 into parents and Stage 2 into children."
    )
    parser.add_argument("--activation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=_MODEL_NAMES,
        default=_DEFAULT_MODELS,
    )
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--max-seconds", type=float, default=900.0)
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


def atomic_save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        suffix=".png",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def spatial_vectors(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("expected [C,H,W], got %s" % (features.shape,))
    if not np.isfinite(features).all():
        raise RuntimeError("features contain nonfinite values")

    channels = features.shape[0]
    vectors = features.transpose(1, 2, 0).reshape(-1, channels)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-8)
    if not np.isfinite(normalized).all():
        raise RuntimeError("normalized vectors contain nonfinite values")
    return normalized.astype(np.float32, copy=False)


def fit_labels(
    vectors: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    if len(vectors) < n_clusters:
        raise ValueError(
            "cannot fit %d clusters to %d vectors"
            % (n_clusters, len(vectors))
        )
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=_KMEANS_RESTARTS,
        random_state=seed,
        algorithm="lloyd",
    ).fit_predict(vectors)
    if len(np.unique(labels)) != n_clusters:
        raise RuntimeError("KMeans returned fewer clusters than requested")
    return labels.astype(np.int16, copy=False)


def build_hierarchy(
    stage3: np.ndarray,
    stage2: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    stage3_height, stage3_width = stage3.shape[1:]
    stage2_height, stage2_width = stage2.shape[1:]
    if (stage2_height, stage2_width) != (
        stage3_height * 2,
        stage3_width * 2,
    ):
        raise ValueError(
            "Stage 2 must be exactly twice Stage 3 spatially: %s vs %s"
            % (stage2.shape, stage3.shape)
        )

    coarse = fit_labels(
        spatial_vectors(stage3),
        n_clusters=_COARSE_CLUSTERS,
        seed=seed,
    ).reshape(stage3_height, stage3_width)
    coarse_at_stage2 = np.repeat(
        np.repeat(coarse, 2, axis=0),
        2,
        axis=1,
    )
    stage2_vectors = spatial_vectors(stage2)
    fine = np.full(
        (stage2_height, stage2_width),
        fill_value=-1,
        dtype=np.int16,
    )
    fine_flat = fine.reshape(-1)
    coarse_flat = coarse_at_stage2.reshape(-1)
    parent_records = []

    for parent in range(_COARSE_CLUSTERS):
        locations = coarse_flat == parent
        location_count = int(locations.sum())
        children = fit_labels(
            stage2_vectors[locations],
            n_clusters=_CHILDREN_PER_PARENT,
            seed=seed + parent + 1,
        )
        hierarchy_labels = parent * _CHILDREN_PER_PARENT + children
        fine_flat[locations] = hierarchy_labels
        parent_records.append(
            {
                "parent_label": parent,
                "stage2_location_count": location_count,
                "child_labels": [
                    parent * _CHILDREN_PER_PARENT + child
                    for child in range(_CHILDREN_PER_PARENT)
                ],
                "child_location_counts": {
                    str(parent * _CHILDREN_PER_PARENT + child): int(
                        np.sum(children == child)
                    )
                    for child in range(_CHILDREN_PER_PARENT)
                },
            }
        )

    if np.any(fine < 0):
        raise RuntimeError("some Stage-2 locations were not assigned")
    return coarse, fine, parent_records


def upsample_labels(
    labels: np.ndarray,
    output_size: tuple[int, int] = _OUTPUT_SIZE,
) -> np.ndarray:
    output_height, output_width = output_size
    height, width = labels.shape
    if output_height % height or output_width % width:
        raise ValueError(
            "output size %s is not divisible by label shape %s"
            % (output_size, labels.shape)
        )
    return np.repeat(
        np.repeat(labels, output_height // height, axis=0),
        output_width // width,
        axis=1,
    )


def categorical_image(
    labels: np.ndarray,
    colors: np.ndarray,
) -> Image.Image:
    if labels.min() < 0 or labels.max() >= len(colors):
        raise ValueError("labels exceed the categorical palette")
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[: len(colors)] = colors
    image = Image.fromarray(labels.astype(np.uint8, copy=False))
    image.putpalette(palette.reshape(-1).tolist())
    return image


def categorical_overlay(
    original: Image.Image,
    labels: np.ndarray,
    colors: np.ndarray,
) -> Image.Image:
    original_array = np.asarray(original, dtype=np.float32)
    categorical = colors[labels].astype(np.float32)
    blended = (
        (1.0 - _OVERLAY_WEIGHT) * original_array
        + _OVERLAY_WEIGHT * categorical
    )
    return Image.fromarray(
        np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    )


def add_output_file(
    path: Path,
    current_bytes: int,
    current_files: int,
    max_output_bytes: int,
) -> tuple[int, int]:
    current_bytes += path.stat().st_size
    current_files += 1
    if current_bytes > max_output_bytes:
        raise RuntimeError("stage hierarchy exceeded output-size limit")
    return current_bytes, current_files


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0.0 or args.max_output_mb <= 0.0:
        raise ValueError("resource limits must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must not contain duplicates")

    activation_root = Path(args.activation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    max_output_bytes = int(args.max_output_mb * 1024 * 1024)
    output_bytes = 0
    output_files = 0

    run_path = output_dir / "run.json"
    atomic_write_json(
        run_path,
        {
            "activation_root": os.path.abspath(activation_root),
            "models": list(args.models),
            "hierarchy": {
                "level1": {
                    "source_stage": "stage3",
                    "clusters": _COARSE_CLUSTERS,
                },
                "level2": {
                    "source_stage": "stage2",
                    "children_per_parent": _CHILDREN_PER_PARENT,
                    "combined_labels": _COARSE_CLUSTERS
                    * _CHILDREN_PER_PARENT,
                },
            },
            "feature_normalization": "L2 per spatial vector",
            "kmeans_n_init": _KMEANS_RESTARTS,
            "base_seed": args.seed,
            "seed_offsets": {
                model_name: _MODEL_SEED_OFFSETS[model_name]
                for model_name in args.models
            },
            "label_resize": "nearest",
            "overlay_weight": _OVERLAY_WEIGHT,
            "processing": "one model/image activation file at a time",
            "max_seconds": args.max_seconds,
            "max_output_mb": args.max_output_mb,
        },
    )
    output_bytes, output_files = add_output_file(
        run_path,
        output_bytes,
        output_files,
        max_output_bytes,
    )

    for model_name in args.models:
        manifest = load_json(
            activation_root / model_name / "activation_manifest.json"
        )
        if not isinstance(manifest, dict):
            raise ValueError("activation manifest must be a JSON object")
        records = manifest.get("images")
        if not isinstance(records, list) or len(records) != 50:
            raise ValueError(
                "%s activation manifest must contain 50 images" % model_name
            )

        for position, record in enumerate(records):
            if time.monotonic() - started_at > args.max_seconds:
                raise TimeoutError(
                    "stage hierarchy exceeded the configured time limit"
                )
            if not isinstance(record, dict):
                raise ValueError("activation image record must be an object")

            image_id = str(record["image_id"])
            image_dir = output_dir / model_name / image_id
            image_dir.mkdir(parents=True, exist_ok=False)

            original_path = image_dir / "original.png"
            shutil.copyfile(str(record["crop_path"]), original_path)
            output_bytes, output_files = add_output_file(
                original_path,
                output_bytes,
                output_files,
                max_output_bytes,
            )
            with Image.open(original_path) as image:
                original = image.convert("RGB").copy()
            if original.size != _OUTPUT_SIZE[::-1]:
                raise ValueError(
                    "original crop must be 224x224: %s" % original_path
                )

            activation_path = Path(str(record["activation_path"]))
            with np.load(activation_path, allow_pickle=False) as activations:
                stage2 = activations["stage2"].astype(
                    np.float32,
                    copy=True,
                )
                stage3 = activations["stage3"].astype(
                    np.float32,
                    copy=True,
                )

            image_seed = (
                args.seed
                + _MODEL_SEED_OFFSETS[model_name]
                + position * 10
            )
            coarse, fine, parent_records = build_hierarchy(
                stage3=stage3,
                stage2=stage2,
                seed=image_seed,
            )
            level1 = upsample_labels(coarse)
            level2 = upsample_labels(fine)

            outputs = {
                "level1_labels.png": categorical_image(
                    level1,
                    _LEVEL1_COLORS,
                ),
                "level1_overlay.png": categorical_overlay(
                    original,
                    level1,
                    _LEVEL1_COLORS,
                ),
                "level2_labels.png": categorical_image(
                    level2,
                    _LEVEL2_COLORS,
                ),
                "level2_overlay.png": categorical_overlay(
                    original,
                    level2,
                    _LEVEL2_COLORS,
                ),
            }
            for filename, image in outputs.items():
                path = image_dir / filename
                atomic_save_png(path, image)
                output_bytes, output_files = add_output_file(
                    path,
                    output_bytes,
                    output_files,
                    max_output_bytes,
                )

            hierarchy_path = image_dir / "hierarchy.json"
            atomic_write_json(
                hierarchy_path,
                {
                    "model": model_name,
                    "image_id": image_id,
                    "activation_path": os.path.abspath(activation_path),
                    "seed": image_seed,
                    "feature_normalization": "L2 per spatial vector",
                    "kmeans_n_init": _KMEANS_RESTARTS,
                    "level1": {
                        "source_stage": "stage3",
                        "native_shape": list(coarse.shape),
                        "clusters": _COARSE_CLUSTERS,
                        "location_counts": {
                            str(label): int(np.sum(coarse == label))
                            for label in range(_COARSE_CLUSTERS)
                        },
                    },
                    "level2": {
                        "source_stage": "stage2",
                        "native_shape": list(fine.shape),
                        "children_per_parent": _CHILDREN_PER_PARENT,
                        "combined_label_formula": (
                            "parent * children_per_parent + child"
                        ),
                        "parents": parent_records,
                    },
                    "output_shape": list(_OUTPUT_SIZE),
                    "label_resize": "nearest",
                },
            )
            output_bytes, output_files = add_output_file(
                hierarchy_path,
                output_bytes,
                output_files,
                max_output_bytes,
            )

            del stage2, stage3, coarse, fine, level1, level2, outputs
            if (position + 1) % 10 == 0:
                print(
                    "[%s] hierarchy %02d/%02d"
                    % (model_name, position + 1, len(records)),
                    flush=True,
                )

    complete_path = output_dir / "complete.json"
    atomic_write_json(
        complete_path,
        {
            "elapsed_seconds": time.monotonic() - started_at,
            "model_count": len(args.models),
            "image_count_per_model": 50,
            "hierarchy_count": 50 * len(args.models),
            "output_file_count_excluding_complete": output_files,
            "output_bytes_excluding_complete": output_bytes,
        },
    )
    output_bytes, output_files = add_output_file(
        complete_path,
        output_bytes,
        output_files,
        max_output_bytes,
    )
    print(
        "Saved %d files (%d bytes) to %s"
        % (output_files, output_bytes, output_dir),
        flush=True,
    )


if __name__ == "__main__":
    main()
