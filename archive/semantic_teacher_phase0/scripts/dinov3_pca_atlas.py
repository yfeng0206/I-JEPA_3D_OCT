#!/usr/bin/env python
"""Reproduce released DINOv3 PCA visualizations with explicit adaptations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml
from sklearn.linear_model import LogisticRegression

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides import build_guide  # noqa: E402
from src.guides.dino_pca import (  # noqa: E402
    classifier_state,
    dinov2_two_stage_pca,
    median_foreground_mask,
    normalize_imagenet,
    paper_style_pca,
    pca_orientation_variants,
    predict_foreground_probability,
    quantize_foreground_mask,
    resize_aspect_to_patch_grid,
)


FOREGROUND_IMAGES_URL = (
    "https://dl.fbaipublicfiles.com/dinov3/notebooks/"
    "foreground_segmentation/foreground_segmentation_images.tar.gz"
)
FOREGROUND_LABELS_URL = (
    "https://dl.fbaipublicfiles.com/dinov3/notebooks/"
    "foreground_segmentation/foreground_segmentation_labels.tar.gz"
)
OFFICIAL_DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render DINOv3 high-resolution PCA using the released supervised "
            "foreground tutorial and an unsupervised DINOv2-style adaptation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/semantic_maps/phase0_guides.yaml",
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default=r"D:\jepa_phase0\results\pca\dinov3_7b",
    )
    parser.add_argument(
        "--foreground-data-dir",
        default=r"D:\jepa_phase0\official\dinov3_foreground",
    )
    parser.add_argument(
        "--classifier-path",
        default=(
            r"D:\jepa_phase0\models\dinov3"
            r"\foreground_classifier_7b_768.npz"
        ),
    )
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retrain-foreground",
        action="store_true",
        help="Discard the cached ViT-7B foreground classifier.",
    )
    return parser.parse_args()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_output_dir(path, overwrite):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(
            "output directory exists; use --overwrite: %s" % output
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=output.name + ".tmp-",
            dir=output.parent,
        )
    )
    return output, temporary


def commit_output_dir(temporary, output, overwrite):
    backup = None
    if output.exists():
        if not overwrite:
            raise FileExistsError("output directory exists")
        backup = output.with_name(output.name + ".backup-%d" % os.getpid())
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, output)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_manifest(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input manifest must contain images")
    selected = []
    seen = set()
    for row in rows[:limit]:
        source = Path(row["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = file_sha256(source)
        if observed != row["image_sha256"]:
            raise RuntimeError("input checksum mismatch: %s" % row["sample_id"])
        identifier = "img_" + observed[:16]
        if identifier in seen:
            raise ValueError("input manifest contains duplicate image content")
        seen.add(identifier)
        item = dict(row)
        item["image_id"] = identifier
        selected.append(item)
    return selected, {
        key: value for key, value in payload.items() if key != "images"
    }


def download_file(url, path):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    os.replace(temporary, path)


def load_tar_images(path):
    images = []
    with tarfile.open(path, "r:gz") as archive:
        members = sorted(
            (member for member in archive if member.isfile()),
            key=lambda member: member.name,
        )
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("could not read %s" % member.name)
            with source, Image.open(source) as image:
                images.append(image.copy())
    return images


class DINOFeatureExtractor:
    def __init__(self, guide, image_height):
        self.guide = guide
        self.image_height = int(image_height)
        self.patch_size = int(guide.patch_size)

    @torch.inference_mode()
    def extract(self, image):
        tensor, grid = resize_aspect_to_patch_grid(
            image,
            image_height=self.image_height,
            patch_size=self.patch_size,
        )
        normalized = normalize_imagenet(tensor).unsqueeze(0).to(
            self.guide.device, dtype=self.guide.dtype
        )
        if self.guide.device.type == "cuda":
            torch.cuda.synchronize(self.guide.device)
            torch.cuda.reset_peak_memory_stats(self.guide.device)
        started = time.perf_counter()
        output = self.guide.model(
            pixel_values=normalized,
            output_attentions=False,
            return_dict=True,
        )
        if self.guide.device.type == "cuda":
            torch.cuda.synchronize(self.guide.device)
        elapsed = time.perf_counter() - started
        patch_count = grid[0] * grid[1]
        prefix_count = int(output.last_hidden_state.size(1) - patch_count)
        if prefix_count < 1:
            raise RuntimeError("DINO output has no expected prefix tokens")
        patches = (
            output.last_hidden_state[0, prefix_count:]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if patches.shape[0] != patch_count:
            raise RuntimeError("DINO patch count does not match resize grid")
        if self.guide.device.type == "cuda":
            memory = {
                "max_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.guide.device)
                ),
                "max_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.guide.device)
                ),
            }
        else:
            memory = {
                "max_cuda_allocated_bytes": 0,
                "max_cuda_reserved_bytes": 0,
            }
        return {
            "features": patches,
            "grid_size": grid,
            "resized_tensor": tensor,
            "latency_seconds": elapsed,
            "memory_telemetry": memory,
            "special_token_count": prefix_count,
        }


def classifier_paths(path):
    path = Path(path)
    return path, path.with_suffix(".json")


def load_classifier(path, model_revision, image_height):
    state_path, metadata_path = classifier_paths(path)
    if not state_path.is_file() or not metadata_path.is_file():
        return None, None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        metadata.get("model_revision") != model_revision
        or int(metadata.get("image_height", -1)) != int(image_height)
        or metadata.get("official_source_commit") != OFFICIAL_DINOV3_COMMIT
    ):
        return None, None
    loaded = np.load(state_path, allow_pickle=False)
    state = {
        "coef": loaded["coef"],
        "intercept": loaded["intercept"],
        "classes": loaded["classes"],
    }
    return state, metadata


def train_classifier(extractor, data_dir, output_path, model_revision):
    data_dir = Path(data_dir)
    image_archive = data_dir / "foreground_segmentation_images.tar.gz"
    label_archive = data_dir / "foreground_segmentation_labels.tar.gz"
    download_file(FOREGROUND_IMAGES_URL, image_archive)
    download_file(FOREGROUND_LABELS_URL, label_archive)
    images = load_tar_images(image_archive)
    labels = load_tar_images(label_archive)
    if len(images) != 9 or len(labels) != 9:
        raise RuntimeError("official foreground data must contain nine pairs")

    features = []
    targets = []
    image_ids = []
    for index, (image, label) in enumerate(zip(images, labels)):
        result = extractor.extract(image.convert("RGB"))
        alpha = label.split()[-1]
        quantized = quantize_foreground_mask(
            alpha,
            result["grid_size"],
            patch_size=extractor.patch_size,
        ).numpy()
        pure = (quantized < 0.01) | (quantized > 0.99)
        features.append(result["features"][pure])
        targets.append((quantized[pure] > 0).astype(np.int64))
        image_ids.append(
            np.full(int(pure.sum()), index, dtype=np.int64)
        )
        print(
            "foreground training image %d/9: %d pure patches"
            % (index + 1, int(pure.sum())),
            flush=True,
        )
    values = np.concatenate(features, axis=0)
    labels_array = np.concatenate(targets, axis=0)
    ids = np.concatenate(image_ids, axis=0)
    classifier = LogisticRegression(
        random_state=0,
        C=0.1,
        max_iter=10000,
    ).fit(values, labels_array)
    state = classifier_state(classifier)
    state_path, metadata_path = classifier_paths(output_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, **state)
    metadata = {
        "schema_version": 1,
        "method": "DINOv3 notebook-style supervised foreground classifier",
        "adaptation": "retrained on 4096-D ViT-7B final normalized features",
        "official_source_commit": OFFICIAL_DINOV3_COMMIT,
        "model_revision": model_revision,
        "image_height": extractor.image_height,
        "patch_size": extractor.patch_size,
        "training_images": len(images),
        "training_pure_patches": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "regularization_c": 0.1,
        "image_archive_sha256": file_sha256(image_archive),
        "label_archive_sha256": file_sha256(label_archive),
        "image_ids_recorded": int(ids.size),
    }
    atomic_json(metadata_path, metadata)
    return state, metadata


def save_rgb(path, rgb):
    values = np.asarray(rgb, dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    Image.fromarray(np.uint8(values * 255), mode="RGB").save(path)


def plot_panel(
    original,
    foreground_score,
    supervised_rgb,
    unsupervised,
    output_path,
):
    figure, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False)
    axes = axes[0]
    axes[0].imshow(original)
    axes[0].set_title("full source")
    axes[1].imshow(foreground_score, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("supervised foreground score")
    axes[2].imshow(supervised_rgb)
    axes[2].set_title("notebook-style PCA\ncanonical orientation")
    for index, polarity in enumerate(unsupervised["polarities"]):
        if polarity["rgb"] is not None:
            axes[3 + index].imshow(polarity["rgb"])
        axes[3 + index].set_title(
            "DINOv2-style\n%s" % polarity["polarity"]
        )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_orientations(variants, output_path):
    figure, axes = plt.subplots(8, 6, figsize=(12, 16), squeeze=False)
    for axis, variant in zip(axes.flat, variants):
        axis.imshow(variant["rgb"])
        axis.set_title(
            "%s / %s" % (variant["order"], variant["signs"]),
            fontsize=6,
        )
        axis.axis("off")
    figure.suptitle(
        "All 48 PCA sign/RGB orientations; no automatic aesthetic selection",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.image_height <= 0 or args.max_images <= 0:
        raise ValueError("image height and count must be positive")
    config = load_config(args.config)
    values = dict(config["guides"]["dinov3"])
    for key in (
        "enabled",
        "official_model_id",
        "published_parameters",
        "evaluation_batch_size",
        "atlas_batch_size",
    ):
        values.pop(key, None)
    values["device"] = args.device
    values["attn_implementation"] = "sdpa"
    entries, selection = load_manifest(
        args.input_manifest, args.max_images
    )
    output, temporary = prepare_output_dir(
        args.output_dir, overwrite=args.overwrite
    )
    moved = False
    guide = None
    try:
        print("Loading DINOv3 for high-resolution PCA...", flush=True)
        started = time.perf_counter()
        guide = build_guide("dinov3", **values)
        load_seconds = time.perf_counter() - started
        extractor = DINOFeatureExtractor(guide, args.image_height)
        if args.retrain_foreground:
            state_path, metadata_path = classifier_paths(
                args.classifier_path
            )
            state_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        classifier, classifier_metadata = load_classifier(
            args.classifier_path,
            values.get("revision"),
            args.image_height,
        )
        if classifier is None:
            classifier, classifier_metadata = train_classifier(
                extractor,
                args.foreground_data_dir,
                args.classifier_path,
                values.get("revision"),
            )
        records = []
        for index, entry in enumerate(entries):
            with Image.open(entry["path"]) as source:
                image = source.convert("RGB")
            result = extractor.extract(image)
            probability = predict_foreground_probability(
                result["features"], classifier
            )
            foreground_score, foreground_mask = median_foreground_mask(
                probability, result["grid_size"], threshold=0.5
            )
            supervised = paper_style_pca(
                result["features"],
                result["grid_size"],
                foreground_mask,
                whiten=True,
            )
            variants = pca_orientation_variants(
                supervised["projected"], foreground_mask
            )
            unsupervised = dinov2_two_stage_pca(
                result["features"], result["grid_size"], threshold=0.5
            )
            image_dir = temporary / entry["image_id"]
            image_dir.mkdir(parents=True)
            image.save(image_dir / "source.jpg", quality=95)
            save_rgb(image_dir / "paper_style_canonical.png", supervised["rgb"])
            for polarity in unsupervised["polarities"]:
                if polarity["rgb"] is not None:
                    save_rgb(
                        image_dir
                        / ("unsupervised_%s.png" % polarity["polarity"]),
                        polarity["rgb"],
                    )
            plot_panel(
                image,
                foreground_score,
                supervised["rgb"],
                unsupervised,
                image_dir / "comparison_panel.png",
            )
            plot_orientations(
                variants, image_dir / "all_48_orientations.png"
            )
            np.savez_compressed(
                image_dir / "maps.npz",
                foreground_score=foreground_score.astype(np.float32),
                foreground_mask=foreground_mask,
                supervised_projected=supervised["projected"],
                supervised_rgb=supervised["rgb"],
                unsupervised_pc1=unsupervised["pc1"],
                unsupervised_pc1_normalized=unsupervised[
                    "pc1_normalized"
                ],
            )
            record = {
                "image_id": entry["image_id"],
                "input_index": index,
                "sample_id": entry.get("sample_id"),
                "image_sha256": entry["image_sha256"],
                "grid_size": list(result["grid_size"]),
                "feature_dim": int(result["features"].shape[1]),
                "foreground_patch_count": int(foreground_mask.sum()),
                "latency_seconds": result["latency_seconds"],
                "memory_telemetry": result["memory_telemetry"],
                "special_token_count": result["special_token_count"],
                "orientation_variants": 48,
                "automatic_orientation_selection": False,
            }
            records.append(record)
            print(
                "PCA image %d/%d: %s grid=%s foreground=%d"
                % (
                    index + 1,
                    len(entries),
                    entry["image_id"],
                    result["grid_size"],
                    int(foreground_mask.sum()),
                ),
                flush=True,
            )
        atomic_json(
            temporary / "summary.json",
            {
                "schema_version": 1,
                "model": {
                    "official_model_id": config["guides"]["dinov3"][
                        "official_model_id"
                    ],
                    "revision": values.get("revision"),
                    "dtype": str(guide.dtype).replace("torch.", ""),
                    "load_seconds": load_seconds,
                },
                "method": {
                    "paper_style": (
                        "DINOv3 notebook-style ViT-7B adaptation with "
                        "supervised foreground mask"
                    ),
                    "unsupervised": (
                        "DINOv2-style two-stage per-image PCA adaptation "
                        "with both unresolved PC1 polarities"
                    ),
                    "image_height": args.image_height,
                    "patch_size": extractor.patch_size,
                    "paper_orientation_policy": (
                        "all 48 variants retained; no automatic aesthetic "
                        "selection"
                    ),
                    "official_source_commit": OFFICIAL_DINOV3_COMMIT,
                },
                "selection": selection,
                "classifier": classifier_metadata,
                "records": records,
            },
        )
        commit_output_dir(temporary, output, overwrite=args.overwrite)
        moved = True
        print("Saved DINOv3 PCA atlas to %s" % output)
    finally:
        if guide is not None:
            guide.cleanup()
        if not moved and temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
