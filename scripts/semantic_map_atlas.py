#!/usr/bin/env python
"""Extract and visualize frozen semantic-guide maps for Phase 0."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides import available_guides, build_guide
from src.guides.maps import (
    extract_maps,
    minmax_normalize,
    summarize_map,
    top_fraction_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render comparable DINOv3/SigLIP2/CLIP semantic maps."
    )
    parser.add_argument(
        "--config",
        default="configs/semantic_maps/phase0_guides.yaml",
        help="YAML guide configuration.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Image files or glob patterns.",
    )
    parser.add_argument(
        "--guides",
        nargs="*",
        default=None,
        help="Guide names to run; defaults to enabled config entries.",
    )
    parser.add_argument("--output-dir", default="results/semantic_maps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Shared deterministic center crop before guide preprocessing.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument(
        "--save-tokens",
        action="store_true",
        help="Save patch tokens in NPZ files (large).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on a guide load/inference error instead of recording it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--include-source-paths",
        action="store_true",
        help="Include absolute input paths in JSON (off by default).",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def resolve_inputs(patterns, limit):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.isfile(pattern):
            paths.append(pattern)
    unique = []
    seen = set()
    for path in paths:
        absolute = os.path.abspath(path)
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    if not unique:
        raise RuntimeError("no input images matched")
    return unique[:limit]


def image_id(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "img_" + digest.hexdigest()[:16]


def sanitize_error(error, input_paths):
    text = str(error)
    replacements = []
    for path in input_paths:
        replacements.append((path, "<input>"))
    replacements.extend(
        [
            (_PROJECT_ROOT, "<project>"),
            (os.path.expanduser("~"), "~"),
        ]
    )
    for source, replacement in replacements:
        if source:
            text = text.replace(source, replacement)
    return text


def safe_model_id(model_id):
    value = str(model_id)
    if os.path.isabs(value) or os.path.exists(value):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return "local_model_" + digest
    return value


def sanitize_metadata(value):
    if isinstance(value, dict):
        return {
            str(key): sanitize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return safe_model_id(value)
    return value


def prepare_output_dir(path, overwrite):
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(absolute):
        if not overwrite:
            raise FileExistsError(
                "output directory already exists: %s (use --overwrite)"
                % absolute
            )
    temporary = tempfile.mkdtemp(
        prefix=os.path.basename(absolute) + ".tmp-",
        dir=parent,
    )
    return absolute, temporary


def commit_output_dir(run_dir, output_dir, overwrite):
    backup_dir = None
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError("output directory already exists")
        backup_dir = "%s.backup-%d" % (output_dir, os.getpid())
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        os.replace(output_dir, backup_dir)
    try:
        os.replace(run_dir, output_dir)
    except Exception:
        if backup_dir is not None and os.path.exists(backup_dir):
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir is not None and os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)


def load_images(paths, crop_size):
    tensors = []
    originals = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = TF.to_tensor(image)
            tensor = TF.resize(tensor, crop_size, antialias=True)
            tensor = TF.center_crop(tensor, [crop_size, crop_size])
            originals.append(TF.to_pil_image(tensor))
            tensors.append(tensor)
    return originals, tensors


def stack_batch(tensors):
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("all shared crops must have the same tensor shape")
    return torch.stack(tensors, dim=0)


def tensor_values(values, index):
    return {
        key: float(value[index].detach().cpu().item())
        for key, value in values.items()
    }


def save_map_npz(
    output_dir,
    image_ids,
    guide_name,
    maps,
    output,
    save_tokens,
):
    map_dir = os.path.join(output_dir, "maps")
    os.makedirs(map_dir, exist_ok=True)
    if len(image_ids) != output.batch_size:
        raise ValueError("image ID count must match guide output batch size")
    for index, identifier in enumerate(image_ids):
        payload = {
            "grid_size": np.asarray(output.grid_size, dtype=np.int64),
        }
        for map_name, score_map in maps.items():
            payload[map_name] = score_map[index].detach().float().cpu().numpy()
        if save_tokens:
            payload["patch_tokens"] = (
                output.patch_tokens[index].detach().float().cpu().numpy()
            )
        np.savez_compressed(
            os.path.join(map_dir, "%s__%s.npz" % (identifier, guide_name)),
            **payload
        )


def plot_atlas(original, image_name, guide_maps, output_path, top_fraction):
    rows = max(1, len(guide_maps))
    columns = 5
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.0 * rows),
        squeeze=False,
    )
    for row, (guide_name, maps) in enumerate(sorted(guide_maps.items())):
        primary_name = (
            "native"
            if "native" in maps
            else "global_cosine"
            if "global_cosine" in maps
            else "token_pca"
        )
        primary = minmax_normalize(maps[primary_name].unsqueeze(0))[0]
        pca = minmax_normalize(maps["token_pca"].unsqueeze(0))[0]
        top = top_fraction_mask(
            primary.unsqueeze(0), fraction=top_fraction
        )[0]

        axes[row, 0].imshow(original)
        axes[row, 0].set_title("%s: image" % guide_name)
        axes[row, 1].imshow(primary.numpy(), cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title(primary_name)
        axes[row, 2].imshow(pca.numpy(), cmap="coolwarm", vmin=0, vmax=1)
        axes[row, 2].set_title("token PCA")
        axes[row, 3].imshow(original)
        axes[row, 3].imshow(
            np.asarray(
                Image.fromarray(
                    np.uint8(primary.numpy() * 255)
                ).resize(original.size, Image.Resampling.NEAREST)
            ),
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=255,
        )
        axes[row, 3].set_title("overlay")
        axes[row, 4].imshow(top.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 4].set_title("top %.0f%%" % (100.0 * top_fraction))
        for column in range(columns):
            axes[row, column].axis("off")
    figure.suptitle(image_name)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    config = load_config(args.config)
    guide_config = config.get("guides", {})
    selected = args.guides or [
        name
        for name, values in guide_config.items()
        if bool((values or {}).get("enabled", False))
    ]
    unknown = sorted(set(selected) - set(available_guides()))
    if unknown:
        raise ValueError("unknown guides: %s" % unknown)
    if not selected:
        raise RuntimeError("no guides selected")

    paths = resolve_inputs(args.inputs, args.max_images)
    identifiers = [image_id(path) for path in paths]
    originals, tensors = load_images(paths, args.crop_size)
    output_dir, run_dir = prepare_output_dir(
        args.output_dir, overwrite=args.overwrite
    )
    os.makedirs(os.path.join(run_dir, "atlases"), exist_ok=True)

    atlas_maps = [dict() for _ in paths]
    records = []
    failures = []

    moved = False
    try:
        for guide_name in selected:
            values = dict(guide_config.get(guide_name, {}) or {})
            values.pop("enabled", None)
            values["device"] = args.device
            guide = None
            try:
                print("Loading guide %s..." % guide_name, flush=True)
                guide = build_guide(guide_name, **values)
                metric_temperature = float(
                    (config.get("metrics", {}) or {}).get(
                        "temperature", 0.15
                    )
                )
                for start in range(0, len(paths), args.batch_size):
                    stop = min(start + args.batch_size, len(paths))
                    batch = stack_batch(tensors[start:stop])
                    output = guide.encode(batch)
                    maps = extract_maps(output)
                    for local_index, global_index in enumerate(
                        range(start, stop)
                    ):
                        per_image_maps = {
                            name: score_map[local_index]
                            .detach()
                            .float()
                            .cpu()
                            for name, score_map in maps.items()
                        }
                        atlas_maps[global_index][guide_name] = per_image_maps
                        primary_name = (
                            "native"
                            if "native" in per_image_maps
                            else "global_cosine"
                            if "global_cosine" in per_image_maps
                            else "token_pca"
                        )
                        metrics = summarize_map(
                            per_image_maps[primary_name].unsqueeze(0),
                            temperature=metric_temperature,
                        )
                        record = {
                            "image_id": identifiers[global_index],
                            "input_index": global_index,
                            "guide": guide_name,
                            "model_id": safe_model_id(guide.model_id),
                            "primary_map": primary_name,
                            "grid_size": list(output.grid_size),
                            "metric_temperature": metric_temperature,
                            "metrics": tensor_values(metrics, 0),
                            "metadata": sanitize_metadata(output.metadata),
                        }
                        if args.include_source_paths:
                            record["source_path"] = paths[global_index]
                        records.append(record)
                    save_map_npz(
                        run_dir,
                        identifiers[start:stop],
                        guide_name,
                        maps,
                        output,
                        args.save_tokens,
                    )
            except Exception as exc:
                failure = {
                    "guide": guide_name,
                    "error_type": type(exc).__name__,
                    "error": sanitize_error(exc, paths),
                }
                failures.append(failure)
                print(
                    "[guide failure] %s: %s" % (guide_name, exc), flush=True
                )
                if args.strict:
                    raise
            finally:
                if guide is not None:
                    guide.cleanup()

        if not records:
            raise RuntimeError("all selected guides failed; no maps produced")

        for index, original in enumerate(originals):
            if not atlas_maps[index]:
                continue
            title = (
                paths[index]
                if args.include_source_paths
                else identifiers[index]
            )
            plot_atlas(
                original,
                title,
                atlas_maps[index],
                os.path.join(
                    run_dir, "atlases", identifiers[index] + ".png"
                ),
                args.top_fraction,
            )

        image_manifest = [
            {
                "image_id": identifier,
                "input_index": index,
                **(
                    {"source_path": paths[index]}
                    if args.include_source_paths
                    else {}
                ),
            }
            for index, identifier in enumerate(identifiers)
        ]
        with open(
            os.path.join(run_dir, "semantic_map_metrics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "config": os.path.basename(args.config),
                    "images": image_manifest,
                    "records": records,
                    "failures": failures,
                },
                handle,
                indent=2,
            )
        commit_output_dir(run_dir, output_dir, overwrite=args.overwrite)
        moved = True
        print(
            "Saved %d records, %d guide failures to %s"
            % (len(records), len(failures), output_dir)
        )
    finally:
        if not moved and os.path.isdir(run_dir):
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
