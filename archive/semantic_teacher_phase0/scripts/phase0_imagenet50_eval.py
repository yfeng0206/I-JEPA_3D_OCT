#!/usr/bin/env python
"""Cache frozen ImageNet-50 features and run fixed kNN/linear probes."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.imagenet_subset import (  # noqa: E402
    ImageNetSubsetDataset,
    wnid_manifest_sha256,
)
from src.evaluation.feature_cache import (  # noqa: E402
    FeatureCacheWriter,
    atomic_write_json,
    dataset_snapshot,
    load_feature_cache,
)
from src.evaluation.imagenet_frozen import (  # noqa: E402
    classification_metrics,
    fit_linear_probe,
    weighted_knn_predict,
)
from src.guides import available_guides, build_guide  # noqa: E402

_VLM_GUIDES = frozenset(("qwen3_vl", "molmo"))
_STANDARD_TRANSFORM_PROFILE = {
    "name": "resize_256_center_crop_224",
    "resize_short_side": 256,
    "center_crop": 224,
    "interpolation": "bicubic",
    "antialias": True,
    "tensor_range": "float32_[0,1]",
}
_VLM_TRANSFORM_PROFILES = {
    "qwen3_vl": {
        "name": "full_source_to_tensor_qwen_dynamic_resolution",
        "resize": None,
        "center_crop": None,
        "preserves_source_aspect_ratio": True,
        "official_processor": "aspect_preserving_dynamic_resolution",
        "tensor_range": "float32_[0,1]",
    },
    "molmo": {
        "name": "full_source_to_tensor_molmo_global_plus_local_crops",
        "resize": None,
        "center_crop": None,
        "preserves_source_aspect_ratio": True,
        "official_processor": "global_plus_up_to_24_local_crops",
        "tensor_range": "float32_[0,1]",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen Phase-0 ImageNet-50 feature extraction/evaluation."
    )
    parser.add_argument(
        "--config",
        default="configs/semantic_maps/phase0_guides.yaml",
    )
    parser.add_argument("--data-root")
    parser.add_argument(
        "--wnid-manifest",
        default=(
            "configs/semantic_maps/manifests/"
            "phase0-cmc-in100-prefix50-v1.wnids.txt"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=r"D:\jepa_phase0\cache\frozen_features",
    )
    parser.add_argument(
        "--results-dir",
        default=r"D:\jepa_phase0\results\probes",
    )
    parser.add_argument(
        "--dataset-manifest-dir",
        default=r"D:\jepa_phase0\results\manifests\dataset",
    )
    parser.add_argument(
        "--mode",
        choices=("extract", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--guides", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.07)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-dataset-snapshot",
        action="store_true",
        help="Re-hash every source image before replacing its snapshot.",
    )
    parser.add_argument(
        "--skip-cache-verification",
        action="store_true",
        help="Skip completed-cache SHA-256 checks before probe fitting.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def selected_guides(config, requested):
    guide_config = config.get("guides", {})
    names = requested or [
        name
        for name, values in guide_config.items()
        if bool((values or {}).get("enabled", False))
    ]
    unknown = sorted(set(names) - set(available_guides()))
    if unknown:
        raise ValueError("unknown guides: %s" % unknown)
    if not names:
        raise RuntimeError("no guides selected")
    return names


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluation_transform_profile(guide_name):
    """Return the frozen source transform recorded in feature provenance."""

    name = str(guide_name).lower()
    profile = _VLM_TRANSFORM_PROFILES.get(
        name, _STANDARD_TRANSFORM_PROFILE
    )
    return dict(profile)


def guide_transform(guide_name):
    if str(guide_name).lower() in _VLM_GUIDES:
        return transforms.ToTensor()
    return transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def validate_evaluation_batch_size(guide_name, batch_size):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if guide_name in _VLM_GUIDES and batch_size != 1:
        raise ValueError(
            "%s full-source feature extraction requires batch size 1"
            % guide_name
        )
    if str(guide_name).lower() in _VLM_GUIDES and batch_size != 1:
        raise ValueError(
            "%s frozen evaluation requires batch size 1" % guide_name
        )
    return batch_size


def shared_transform():
    """Backward-compatible name for the I-JEPA/DINO evaluation transform."""

    return guide_transform("ijepa")


def build_dataset(root, split, wnid_manifest, guide_name="ijepa"):
    return ImageNetSubsetDataset(
        root=root,
        split=split,
        wnid_manifest=wnid_manifest,
        transform=guide_transform(guide_name),
        strict=True,
    )


def dataset_vectors(dataset):
    records = dataset.records()
    return (
        [record["sample_id"] for record in records],
        np.asarray([record["label"] for record in records], dtype=np.int64),
    )


def cache_path(cache_root, dataset_id, guide_name, split):
    return os.path.join(cache_root, dataset_id, guide_name, split)


_FEATURE_IDENTITY_KEYS = {
    "official_model_id",
    "model_id",
    "revision",
    "weights_path",
    "weights_sha256",
    "checkpoint_key",
    "input_size",
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "dtype",
}


def feature_guide_identity(guide_values, guide_name=None):
    """Keep cache identity limited to fields that can change frozen features."""
    keys = set(_FEATURE_IDENTITY_KEYS)
    if str(guide_name).lower() in _VLM_GUIDES:
        keys.discard("input_size")
    return {
        key: _plain(value)
        for key, value in guide_values.items()
        if key in keys
    }


def prepare_writers(args, config, guide_name, guide_values):
    dataset_id = str(config["dataset"]["id"])
    wnid_hash = wnid_manifest_sha256(args.wnid_manifest)
    prepared = {}
    transform_profile = evaluation_transform_profile(guide_name)
    for split in ("train", "val"):
        dataset = build_dataset(
            args.data_root,
            split,
            args.wnid_manifest,
            guide_name=guide_name,
        )
        snapshot_path = os.path.join(
            args.dataset_manifest_dir,
            dataset_id,
            "%s.json" % split,
        )
        snapshot = dataset_snapshot(
            dataset,
            snapshot_path,
            overwrite=args.refresh_dataset_snapshot,
        )
        sample_ids, labels = dataset_vectors(dataset)
        provenance = {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "split": split,
            "dataset_content_sha256": snapshot["content_sha256"],
            "wnid_manifest_sha256": wnid_hash,
            "guide": guide_name,
            "guide_config": feature_guide_identity(
                guide_values, guide_name=guide_name
            ),
            "input_transform_profile": transform_profile,
            "feature_api": "encode_features",
            "feature_contract_version": 2,
        }
        writer = FeatureCacheWriter(
            cache_path(args.cache_dir, dataset_id, guide_name, split),
            provenance,
            sample_ids,
            labels,
            overwrite=args.overwrite,
        )
        prepared[split] = {
            "dataset": dataset,
            "snapshot": snapshot,
            "writer": writer,
        }
    return prepared


def batch_memory(output, device):
    allocated = 0
    reserved = 0
    if output.memory_telemetry:
        allocated = max(
            int(item["max_cuda_allocated_bytes"])
            for item in output.memory_telemetry
        )
        reserved = max(
            int(item["max_cuda_reserved_bytes"])
            for item in output.memory_telemetry
        )
    elif device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    return allocated, reserved


def extract_split(
    guide,
    dataset,
    writer,
    batch_size,
    num_workers,
    device,
    model_load_seconds,
):
    if writer.is_complete:
        return writer.finalize()
    if writer.completed == writer.count:
        return writer.finalize(
            {"model_load_seconds_last_process": float(model_load_seconds)}
        )

    indices = range(writer.completed, len(dataset))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    cursor = writer.completed
    for images, labels, sample_ids in loader:
        expected_ids = writer.sample_ids[cursor:cursor + len(sample_ids)]
        if list(sample_ids) != expected_ids:
            raise RuntimeError("DataLoader sample order differs from cache order")
        expected_labels = writer.labels[
            cursor:cursor + len(sample_ids)
        ].tolist()
        if labels.long().tolist() != expected_labels:
            raise RuntimeError("DataLoader labels differ from cache labels")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = guide.encode_features(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if output.global_token is None:
            raise RuntimeError(
                "%s encode_features returned no frozen global readout"
                % guide.name
            )
        features = (
            F.normalize(output.global_token.detach().float(), dim=1)
            .cpu()
            .numpy()
        )
        if features.shape[0] != len(sample_ids):
            raise RuntimeError("guide feature batch size does not match input")
        allocated, reserved = batch_memory(output, device)
        metadata = {
            "guide_output": _plain(output.metadata),
            "model": _plain(output.model_metadata),
            "grid_size": list(output.grid_size),
            "readout_shape": list(features.shape[1:]),
        }
        writer.write_batch(
            features,
            cursor,
            elapsed_seconds=elapsed,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
            feature_metadata=metadata,
        )
        cursor += len(sample_ids)
        print(
            "%s %s: %d/%d"
            % (guide.name, dataset.split, cursor, len(dataset)),
            flush=True,
        )
    return writer.finalize(
        {"model_load_seconds_last_process": float(model_load_seconds)}
    )


def run_extraction(args, config, names):
    if not args.data_root:
        raise ValueError("--data-root is required for feature extraction")
    guide_config = config.get("guides", {})
    for guide_name in names:
        values = dict(guide_config.get(guide_name, {}) or {})
        values.pop("enabled", None)
        if guide_name in _VLM_GUIDES:
            values.pop("input_size", None)
        batch_size = int(
            args.batch_size
            if args.batch_size is not None
            else values.pop("evaluation_batch_size", 1)
        )
        batch_size = validate_evaluation_batch_size(
            guide_name, batch_size
        )
        prepared = prepare_writers(
            args, config, guide_name, values
        )
        if all(item["writer"].is_complete for item in prepared.values()):
            print("%s caches already complete" % guide_name, flush=True)
            continue

        values["device"] = args.device
        print("Loading %s..." % guide_name, flush=True)
        started = time.perf_counter()
        guide = build_guide(guide_name, **values)
        load_seconds = time.perf_counter() - started
        try:
            for split in ("train", "val"):
                item = prepared[split]
                extract_split(
                    guide,
                    item["dataset"],
                    item["writer"],
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    device=guide.device,
                    model_load_seconds=load_seconds,
                )
        finally:
            guide.cleanup()


def evaluate_one(args, config, guide_name):
    dataset_id = str(config["dataset"]["id"])
    root = os.path.join(args.cache_dir, dataset_id, guide_name)
    train = load_feature_cache(
        os.path.join(root, "train"),
        verify=not args.skip_cache_verification,
    )
    validation = load_feature_cache(
        os.path.join(root, "val"),
        verify=not args.skip_cache_verification,
    )
    num_classes = int(config["dataset"]["num_classes"])
    evaluation_device = args.device
    if evaluation_device == "auto":
        evaluation_device = "cuda" if torch.cuda.is_available() else "cpu"
    knn_started = time.perf_counter()
    knn_scores = weighted_knn_predict(
        train["features"],
        train["labels"],
        validation["features"],
        num_classes=num_classes,
        k=args.knn_k,
        temperature=args.knn_temperature,
        device=evaluation_device,
        train_chunk_size=int(
            config.get("evaluation", {}).get("knn_train_chunk_size", 8192)
        ),
        query_chunk_size=int(
            config.get("evaluation", {}).get("knn_query_chunk_size", 256)
        ),
    )
    knn_seconds = time.perf_counter() - knn_started
    knn_metrics = classification_metrics(
        knn_scores, validation["labels"], topk=(1, 5)
    )

    linear_config = config.get("evaluation", {}).get("linear_probe", {})
    c_values = linear_config.get("c_values")
    linear_started = time.perf_counter()
    linear = fit_linear_probe(
        train["features"],
        train["labels"],
        train["sample_ids"],
        validation["features"],
        c_values=c_values,
        validation_fraction=float(
            linear_config.get("validation_fraction", 0.1)
        ),
        max_iter=int(linear_config.get("max_iter", 1000)),
        tolerance=float(linear_config.get("tolerance", 1e-12)),
    )
    linear_seconds = time.perf_counter() - linear_started
    linear_metrics = classification_metrics(
        linear["scores"], validation["labels"], topk=(1, 5)
    )

    guide_values = dict(config["guides"][guide_name] or {})
    result = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "guide": guide_name,
        "model_id": guide_values.get("official_model_id")
        or guide_values.get("model_id"),
        "revision": guide_values.get("revision"),
        "published_parameters": guide_values.get("published_parameters"),
        "frozen": True,
        "train_count": int(train["features"].shape[0]),
        "validation_count": int(validation["features"].shape[0]),
        "feature_dim": int(train["features"].shape[1]),
        "feature_readout": train["manifest"].get("feature_metadata", {}),
        "cache_manifests": {
            "train": train["manifest"],
            "val": validation["manifest"],
        },
        "knn": {
            "k": int(args.knn_k),
            "temperature": float(args.knn_temperature),
            "metrics": knn_metrics,
            "seconds": knn_seconds,
        },
        "linear_probe": {
            "classifier": "multinomial L2 logistic regression",
            "encoder_frozen": True,
            "selected_c": linear["selected_c"],
            "development_top1": linear["development_top1"],
            "candidates": linear["candidates"],
            "metrics": linear_metrics,
            "seconds": linear_seconds,
        },
    }
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(
        args.results_dir, "%s__%s.json" % (dataset_id, guide_name)
    )
    atomic_write_json(output_path, _plain(result))
    print("Saved %s" % output_path, flush=True)
    return result


def run_evaluation(args, config, names):
    for guide_name in names:
        evaluate_one(args, config, guide_name)


def main():
    args = parse_args()
    args.config = os.path.abspath(args.config)
    args.wnid_manifest = os.path.abspath(args.wnid_manifest)
    config = load_config(args.config)
    names = selected_guides(config, args.guides)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.mode in ("extract", "all"):
        run_extraction(args, config, names)
    if args.mode in ("evaluate", "all"):
        run_evaluation(args, config, names)


if __name__ == "__main__":
    main()
