#!/usr/bin/env python
"""Generate manifest-driven frozen semantic maps and grounded VLM atlases."""

import argparse
from dataclasses import asdict, is_dataclass
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.imagenet_subset import file_sha256  # noqa: E402
from src.guides import available_guides, build_guide  # noqa: E402
from src.guides.maps import (  # noqa: E402
    extract_maps,
    grounding_score_map,
    illustrative_target_rectangles,
    minmax_normalize,
    resize_map,
    summarize_map,
    token_pca_rgb,
)
from src.guides.tokencut import tokencut_partition  # noqa: E402

_VLM_GUIDES = frozenset(("qwen3_vl", "molmo"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render frozen I-JEPA/DINOv3/VLM Phase-0 atlases."
    )
    parser.add_argument(
        "--config",
        default="configs/semantic_maps/phase0_guides.yaml",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-manifest",
        help="Deterministic JSON manifest from phase0_atlas_manifest.py.",
    )
    source.add_argument(
        "--inputs",
        nargs="+",
        help="Ad-hoc image files or glob patterns for debugging only.",
    )
    parser.add_argument(
        "--class-name",
        help="Optional human-review label for ad-hoc input panels.",
    )
    parser.add_argument("--guides", nargs="*", default=None)
    parser.add_argument(
        "--grounding-mode",
        choices=("single_point", "plural_points", "boxes"),
        help=(
            "Override the configured VLM grounding mode. Molmo rejects boxes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\jepa_phase0\results\atlases\phase0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Explicit override; model-specific config is used by default.",
    )
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="I-JEPA/DINO center-crop size; VLMs always receive full sources.",
    )
    parser.add_argument(
        "--save-tokens",
        action="store_true",
        help="Save patch tokens in NPZ files (large).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on a guide or map-method failure.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-source-paths",
        action="store_true",
        help="Include absolute input paths in output JSON.",
    )
    parser.add_argument(
        "--show-class-labels",
        action="store_true",
        help="Show manifest class labels in panels (off for blinded review).",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def image_id(path):
    return "img_" + file_sha256(path)[:16]


def resolve_inputs(patterns, limit):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.isfile(pattern):
            paths.append(pattern)
    entries = []
    seen = set()
    for path in paths:
        absolute = os.path.abspath(path)
        identifier = image_id(absolute)
        if identifier in seen:
            continue
        seen.add(identifier)
        entries.append(
            {
                "path": absolute,
                "image_sha256": file_sha256(absolute),
                "image_id": identifier,
                "selection_mode": "ad_hoc_debug",
            }
        )
    if not entries:
        raise RuntimeError("no input images matched")
    return entries[:limit]


def load_input_manifest(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input manifest must contain a nonempty images list")
    entries = []
    seen = set()
    for index, row in enumerate(rows[:limit]):
        if not isinstance(row, dict):
            raise ValueError("input manifest rows must be objects")
        source_path = row.get("path")
        expected_sha = row.get("image_sha256")
        if not isinstance(source_path, str) or not os.path.isfile(source_path):
            raise FileNotFoundError(
                "manifest image does not exist at row %d: %s"
                % (index, source_path)
            )
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("manifest row %d has no valid SHA-256" % index)
        observed_sha = file_sha256(source_path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                "manifest image checksum mismatch at row %d" % index
            )
        identifier = "img_" + observed_sha[:16]
        if identifier in seen:
            raise ValueError("input manifest contains duplicate image content")
        seen.add(identifier)
        entry = dict(row)
        entry["path"] = os.path.abspath(source_path)
        entry["image_id"] = identifier
        entries.append(entry)
    metadata = {
        key: value for key, value in payload.items() if key != "images"
    }
    metadata["manifest_sha256"] = file_sha256(path)
    return entries, metadata


def sanitize_error(error, input_paths):
    text = str(error)
    replacements = [(path, "<input>") for path in input_paths]
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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return sanitize_metadata(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): sanitize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return safe_model_id(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def prepare_output_dir(path, overwrite):
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(absolute) and not overwrite:
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


def load_images(entries, crop_size=None):
    tensors = []
    originals = []
    for entry in entries:
        with Image.open(entry["path"]) as image:
            image = image.convert("RGB")
            originals.append(image.copy())
            tensors.append(TF.to_tensor(image))
    return originals, tensors


def guide_input_transform_profile(guide_name, crop_size=224):
    guide_name = str(guide_name).lower()
    if guide_name in _VLM_GUIDES:
        profile = {
            "name": (
                "full_source_to_tensor_qwen_dynamic_resolution"
                if guide_name == "qwen3_vl"
                else "full_source_to_tensor_molmo_global_plus_local_crops"
            ),
            "resize": None,
            "center_crop": None,
            "preserves_source_aspect_ratio": True,
        }
        profile["official_processor"] = (
            "aspect_preserving_dynamic_resolution"
            if guide_name == "qwen3_vl"
            else "global_plus_up_to_24_local_crops"
        )
        return profile
    resize_short_side = int(round(256 * int(crop_size) / 224.0))
    return {
        "name": "resize_%d_center_crop_%d"
        % (resize_short_side, int(crop_size)),
        "resize_short_side": resize_short_side,
        "center_crop": int(crop_size),
        "interpolation": "bicubic",
        "antialias": True,
    }


def guide_input_tensors(tensors, guide_name, crop_size=224):
    profile = guide_input_transform_profile(guide_name, crop_size)
    if str(guide_name).lower() in _VLM_GUIDES:
        return list(tensors)
    return [
        TF.center_crop(
            TF.resize(
                tensor,
                profile["resize_short_side"],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            [profile["center_crop"], profile["center_crop"]],
        )
        for tensor in tensors
    ]


def validate_atlas_batch_size(guide_name, batch_size):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("atlas batch size must be positive")
    if str(guide_name).lower() in _VLM_GUIDES and batch_size != 1:
        raise ValueError(
            "%s atlas generation requires batch size 1" % guide_name
        )
    return batch_size


def stack_batch(tensors):
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("all tensors in one guide batch must have the same shape")
    return torch.stack(tensors, dim=0)


def tensor_values(values, index):
    return {
        key: float(value[index].detach().cpu().item())
        for key, value in values.items()
    }


def select_primary_map(maps, derived_grounding=None):
    if derived_grounding is not None:
        return "derived_grounding_raster", derived_grounding
    for name in ("native", "global_cosine", "token_pca"):
        if name in maps:
            return name, maps[name]
    return None, None


def per_image_groundings(output, index):
    boxes = (
        output.grounding_regions[index]
        if output.grounding_regions is not None
        else []
    )
    points = (
        output.grounding_points[index]
        if output.grounding_points is not None
        else []
    )
    return boxes, points


def save_guide_npz(
    output_dir,
    identifier,
    guide_name,
    maps,
    pca_rgb,
    derived_grounding,
    token_cut,
    patch_tokens,
):
    map_dir = os.path.join(output_dir, "maps")
    os.makedirs(map_dir, exist_ok=True)
    payload = {}
    for name, value in maps.items():
        payload[name] = value.detach().float().cpu().numpy()
    if pca_rgb is not None:
        payload["token_pca_rgb"] = pca_rgb.detach().float().cpu().numpy()
    if derived_grounding is not None:
        payload["derived_grounding_raster"] = (
            derived_grounding.detach().float().cpu().numpy()
        )
    if token_cut is not None:
        payload["tokencut_mask"] = (
            token_cut["mask"].detach().float().cpu().numpy()
        )
        payload["tokencut_eigenvector"] = (
            token_cut["eigenvector"].detach().float().cpu().numpy()
        )
        payload["tokencut_box"] = np.asarray(
            token_cut["box"], dtype=np.float32
        )
    if patch_tokens is not None:
        payload["patch_tokens"] = (
            patch_tokens.detach().float().cpu().numpy()
        )
    np.savez_compressed(
        os.path.join(map_dir, "%s__%s.npz" % (identifier, guide_name)),
        **payload
    )


def _normalized_rgb(value):
    array = np.asarray(value, dtype=np.float32)
    result = np.zeros_like(array)
    for channel in range(array.shape[-1]):
        plane = array[..., channel]
        minimum = float(plane.min())
        maximum = float(plane.max())
        result[..., channel] = (plane - minimum) / (
            maximum - minimum + 1e-8
        )
    return result


def _map_image(score_map, image_size):
    normalized = minmax_normalize(score_map.unsqueeze(0))[0].numpy()
    return np.asarray(
        Image.fromarray(np.uint8(normalized * 255)).resize(
            image_size, Image.Resampling.NEAREST
        )
    )


def _draw_overlay(axis, original, score_map, title, cmap="jet"):
    axis.imshow(original)
    axis.imshow(
        _map_image(score_map, original.size),
        cmap=cmap,
        alpha=0.45,
        vmin=0,
        vmax=255,
    )
    axis.set_title(title)


def _box_pixels(box, image_size):
    width, height = image_size
    x1, y1, x2, y2 = box.bbox_2d
    if box.coordinate_space == "normalized_1000":
        return (
            x1 * width / 1000.0,
            y1 * height / 1000.0,
            x2 * width / 1000.0,
            y2 * height / 1000.0,
        )
    source_width, source_height = box.image_size
    return (
        x1 * width / source_width,
        y1 * height / source_height,
        x2 * width / source_width,
        y2 * height / source_height,
    )


def _point_pixels(point, image_size):
    width, height = image_size
    x, y = point.point_2d
    if point.coordinate_space == "normalized_1000":
        return x * width / 1000.0, y * height / 1000.0
    source_width, source_height = point.image_size
    return x * width / source_width, y * height / source_height


def _draw_grounding(axis, original, data, guide_name):
    axis.imshow(original)
    for box in data["boxes"]:
        x1, y1, x2, y2 = _box_pixels(box, original.size)
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor=(0.0, 1.0, 0.0, 0.15),
            )
        )
    for point in data["points"]:
        x, y = _point_pixels(point, original.size)
        axis.scatter([x], [y], s=80, c="yellow", edgecolors="black")
    caption = data.get("caption") or "[no generated caption]"
    if len(caption) > 240:
        caption = caption[:237].rstrip() + "..."
    axis.set_title(
        "%s raw point/box grounding\n%s"
        % (guide_name, textwrap.fill(caption, width=36)),
        fontsize=9,
    )


def _draw_targets(axis, original, data, guide_name, title=None):
    _draw_overlay(
        axis,
        original,
        data["primary"],
        title
        or "%s illustrative top blocks\n(not Phase-1 masks)" % guide_name,
    )
    grid_height, grid_width = data["primary"].shape
    image_width, image_height = original.size
    for row, column, height, width, _ in data["rectangles"]:
        axis.add_patch(
            patches.Rectangle(
                (
                    column * image_width / grid_width,
                    row * image_height / grid_height,
                ),
                width * image_width / grid_width,
                height * image_height / grid_height,
                linewidth=2,
                edgecolor="white",
                facecolor=(0.0, 0.0, 0.0, 0.22),
            )
        )


def _draw_tokencut(axis, original, token_cut):
    axis.imshow(original)
    axis.imshow(
        _map_image(token_cut["mask"], original.size),
        cmap="spring",
        alpha=0.4,
        vmin=0,
        vmax=255,
    )
    x1, y1, x2, y2 = token_cut["box"]
    axis.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
        )
    )
    axis.set_title("DINOv3 TokenCut-style NCut")


def difference_map(dino_map, vlm_map, size=(16, 16)):
    dino = minmax_normalize(
        resize_map(dino_map.unsqueeze(0), size)
    )[0]
    vlm = minmax_normalize(
        resize_map(vlm_map.unsqueeze(0), size)
    )[0]
    return dino - vlm


def plot_atlas(
    original,
    entry,
    guide_data,
    output_path,
    show_class_labels=False,
):
    panel_count = 1
    for name, data in guide_data.items():
        if name in ("qwen3_vl", "molmo"):
            panel_count += 1
            if data.get("primary") is not None:
                panel_count += 1
        else:
            panel_count += int(data.get("primary") is not None)
            panel_count += int(data.get("pca_rgb") is not None)
            panel_count += int(data.get("token_cut") is not None)
            panel_count += int(data.get("primary") is not None)
    dino = guide_data.get("dinov3")
    for name in ("qwen3_vl", "molmo"):
        vlm = guide_data.get(name)
        if (
            dino is not None
            and dino.get("primary") is not None
            and vlm is not None
            and vlm.get("derived_grounding_raster") is not None
        ):
            panel_count += 1

    columns = 4
    rows = int(math.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.8 * rows),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    cursor = 0
    flat_axes[cursor].imshow(original)
    class_name = entry.get("class_name") if show_class_labels else None
    flat_axes[cursor].set_title(
        "original" + ("\n%s" % class_name if class_name else "")
    )
    cursor += 1

    order = ("ijepa", "dinov3", "qwen3_vl", "molmo")
    ordered_names = [name for name in order if name in guide_data]
    ordered_names.extend(
        sorted(set(guide_data) - set(ordered_names))
    )
    for name in ordered_names:
        data = guide_data[name]
        guide_image = data.get("input_image", original)
        if name in ("qwen3_vl", "molmo"):
            _draw_grounding(flat_axes[cursor], original, data, name)
            cursor += 1
            if data.get("primary") is not None:
                _draw_targets(
                    flat_axes[cursor],
                    original,
                    data,
                    name,
                    title=(
                        "%s derived grounding raster\n"
                        "illustrative top blocks (not Phase-1 masks)" % name
                    ),
                )
                cursor += 1
            continue
        if data.get("primary") is not None:
            _draw_overlay(
                flat_axes[cursor],
                guide_image,
                data["primary"],
                "%s %s\nmodel input crop"
                % (name, data["primary_name"]),
            )
            cursor += 1
        if data.get("pca_rgb") is not None:
            flat_axes[cursor].imshow(
                _normalized_rgb(data["pca_rgb"].numpy())
            )
            flat_axes[cursor].set_title(
                "%s 3-component token PCA" % name
            )
            cursor += 1
        if data.get("token_cut") is not None:
            _draw_tokencut(
                flat_axes[cursor], guide_image, data["token_cut"]
            )
            cursor += 1
        if data.get("primary") is not None:
            _draw_targets(flat_axes[cursor], guide_image, data, name)
            cursor += 1

    if dino is not None and dino.get("primary") is not None:
        for name in ("qwen3_vl", "molmo"):
            vlm = guide_data.get(name)
            if vlm is None or vlm.get("derived_grounding_raster") is None:
                continue
            difference = difference_map(
                dino["primary"], vlm["derived_grounding_raster"]
            )
            flat_axes[cursor].imshow(
                difference.numpy(),
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )
            flat_axes[cursor].set_title(
                "DINOv3 minus %s derived grounding raster" % name
            )
            cursor += 1

    for axis in flat_axes:
        axis.axis("off")
    for axis in flat_axes[cursor:]:
        axis.set_visible(False)
    figure.suptitle(entry["image_id"])
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def selected_guide_names(config, requested):
    guide_config = config.get("guides", {})
    selected = requested or [
        name
        for name, values in guide_config.items()
        if bool((values or {}).get("enabled", False))
    ]
    unknown = sorted(set(selected) - set(available_guides()))
    if unknown:
        raise ValueError("unknown guides: %s" % unknown)
    if not selected:
        raise RuntimeError("no guides selected")
    return selected


def main():
    args = parse_args()
    config = load_config(args.config)
    selected = selected_guide_names(config, args.guides)
    if args.max_images <= 0 or args.crop_size <= 0:
        raise ValueError("image limits and crop size must be positive")

    if args.input_manifest:
        entries, selection_metadata = load_input_manifest(
            args.input_manifest, args.max_images
        )
    else:
        entries = resolve_inputs(args.inputs, args.max_images)
        for entry in entries:
            entry["class_name"] = args.class_name
        selection_metadata = {
            "selection_mode": "ad_hoc_debug",
            "selection_uses_model_output": False,
        }
    paths = [entry["path"] for entry in entries]
    originals, source_tensors = load_images(entries, args.crop_size)
    output_dir, run_dir = prepare_output_dir(
        args.output_dir, overwrite=args.overwrite
    )
    os.makedirs(os.path.join(run_dir, "atlases"), exist_ok=True)
    atlas_data = [dict() for _ in entries]
    records = []
    failures = []
    map_config = config.get("maps", {})
    metric_temperature = float(
        config.get("metrics", {}).get("temperature", 0.15)
    )

    moved = False
    try:
        for guide_name in selected:
            values = dict(config.get("guides", {}).get(guide_name, {}) or {})
            values.pop("enabled", None)
            if guide_name in _VLM_GUIDES:
                values.pop("input_size", None)
                if args.grounding_mode is not None:
                    values["grounding_mode"] = args.grounding_mode
            atlas_batch_size = int(
                args.batch_size
                if args.batch_size is not None
                else values.pop("atlas_batch_size", 1)
            )
            values.pop("evaluation_batch_size", None)
            values["device"] = args.device
            atlas_batch_size = validate_atlas_batch_size(
                guide_name, atlas_batch_size
            )
            input_profile = guide_input_transform_profile(
                guide_name, args.crop_size
            )
            tensors = guide_input_tensors(
                source_tensors, guide_name, args.crop_size
            )
            guide = None
            try:
                print("Loading guide %s..." % guide_name, flush=True)
                guide = build_guide(guide_name, **values)
                for start in range(0, len(entries), atlas_batch_size):
                    stop = min(start + atlas_batch_size, len(entries))
                    batch = stack_batch(tensors[start:stop])
                    output = guide.encode(batch)
                    maps = extract_maps(output)
                    pca = (
                        token_pca_rgb(output)
                        if output.metadata.get("spatial_token_grid", True)
                        else None
                    )

                    token_cut_result = None
                    method_failures = []
                    token_cut_config = map_config.get("tokencut", {})
                    if (
                        guide_name == "dinov3"
                        and bool(token_cut_config.get("enabled", False))
                    ):
                        try:
                            token_cut_result = tokencut_partition(
                                output.patch_tokens,
                                output.grid_size,
                                (args.crop_size, args.crop_size),
                                tokencut_root=token_cut_config.get("root"),
                                tau=float(token_cut_config.get("tau", 0.2)),
                                eps=float(token_cut_config.get("eps", 1e-5)),
                            )
                        except Exception as exc:
                            method_failures.append(
                                {
                                    "method": "tokencut",
                                    "error_type": type(exc).__name__,
                                    "error": sanitize_error(exc, paths),
                                }
                            )
                            if args.strict:
                                raise

                    boxes_batch = (
                        output.grounding_regions
                        if output.grounding_regions is not None
                        else [[] for _ in range(output.batch_size)]
                    )
                    points_batch = (
                        output.grounding_points
                        if output.grounding_points is not None
                        else [[] for _ in range(output.batch_size)]
                    )
                    has_grounding = any(
                        boxes or points
                        for boxes, points in zip(boxes_batch, points_batch)
                    )
                    derived_grounding_batch = None
                    if has_grounding:
                        grounding_config = map_config.get("grounding", {})
                        derived_grounding_batch = grounding_score_map(
                            boxes_batch,
                            points_batch,
                            grid_size=tuple(
                                grounding_config.get(
                                    "grid_size", (16, 16)
                                )
                            ),
                            point_sigma_fraction=float(
                                grounding_config.get(
                                    "point_sigma_fraction", 0.08
                                )
                            ),
                        )

                    for local_index, global_index in enumerate(
                        range(start, stop)
                    ):
                        per_maps = {
                            name: value[local_index].detach().float().cpu()
                            for name, value in maps.items()
                        }
                        derived_grounding = (
                            derived_grounding_batch[local_index].detach().cpu()
                            if derived_grounding_batch is not None
                            and (
                                boxes_batch[local_index]
                                or points_batch[local_index]
                            )
                            else None
                        )
                        primary_name, primary = select_primary_map(
                            per_maps, derived_grounding
                        )
                        target_config = map_config.get(
                            "illustrative_targets", {}
                        )
                        rectangles = (
                            illustrative_target_rectangles(
                                primary.unsqueeze(0),
                                block_size=tuple(
                                    target_config.get(
                                        "block_size", (4, 4)
                                    )
                                ),
                                count=int(target_config.get("count", 4)),
                            )[0]
                            if primary is not None
                            else ()
                        )
                        token_cut = None
                        if token_cut_result is not None:
                            token_cut = {
                                "mask": token_cut_result.masks[local_index],
                                "eigenvector": token_cut_result.eigenvectors[
                                    local_index
                                ],
                                "box": token_cut_result.boxes[local_index],
                                "seed": token_cut_result.seeds[local_index],
                                "metadata": token_cut_result.metadata,
                            }
                        boxes, points = per_image_groundings(
                            output, local_index
                        )
                        caption = (
                            output.generated_text[local_index]
                            if output.generated_text is not None
                            else None
                        )
                        per_pca = (
                            pca[local_index].detach().cpu()
                            if pca is not None
                            else None
                        )
                        data = {
                            "maps": per_maps,
                            "primary_name": primary_name,
                            "primary": primary,
                            "pca_rgb": per_pca,
                            "derived_grounding_raster": derived_grounding,
                            "boxes": boxes,
                            "points": points,
                            "caption": caption,
                            "rectangles": rectangles,
                            "token_cut": token_cut,
                            "input_image": (
                                originals[global_index]
                                if guide_name in _VLM_GUIDES
                                else TF.to_pil_image(tensors[global_index])
                            ),
                        }
                        atlas_data[global_index][guide_name] = data

                        metric_maps = dict(per_maps)
                        if derived_grounding is not None:
                            metric_maps["derived_grounding_raster"] = (
                                derived_grounding
                            )
                        map_metrics = {
                            name: tensor_values(
                                summarize_map(
                                    value.unsqueeze(0),
                                    temperature=metric_temperature,
                                ),
                                0,
                            )
                            for name, value in metric_maps.items()
                        }
                        record = {
                            "image_id": entries[global_index]["image_id"],
                            "input_index": global_index,
                            "guide": guide_name,
                            "model_id": safe_model_id(guide.model_id),
                            "model_config": sanitize_metadata(values),
                            "input_transform_profile": input_profile,
                            "grounding_mode": output.metadata.get(
                                "grounding_mode"
                            ),
                            "primary_map": primary_name,
                            "grid_size": list(output.grid_size),
                            "patch_token_shape": list(
                                output.patch_tokens[local_index].shape
                            ),
                            "global_token_shape": (
                                list(output.global_token[local_index].shape)
                                if output.global_token is not None
                                else None
                            ),
                            "embedding_dimension": output.embed_dim,
                            "map_metrics": map_metrics,
                            "generated_text": caption,
                            "grounding_regions": sanitize_metadata(boxes),
                            "grounding_points": sanitize_metadata(points),
                            "grounding_valid": bool(boxes or points),
                            "failures": (
                                output.failures[local_index]
                                if output.failures is not None
                                else []
                            ),
                            "method_failures": method_failures,
                            "metadata": sanitize_metadata(output.metadata),
                            "model_metadata": sanitize_metadata(
                                output.model_metadata
                            ),
                            "spatial_metadata": (
                                sanitize_metadata(
                                    output.spatial_metadata[local_index]
                                )
                                if output.spatial_metadata is not None
                                else None
                            ),
                            "raw_generation": (
                                sanitize_metadata(
                                    output.raw_generation[local_index]
                                )
                                if output.raw_generation is not None
                                else None
                            ),
                            "latency_seconds": (
                                output.latency_seconds[local_index]
                                if output.latency_seconds is not None
                                else None
                            ),
                            "memory_telemetry": (
                                output.memory_telemetry[local_index]
                                if output.memory_telemetry is not None
                                else None
                            ),
                            "illustrative_target_rectangles": [
                                {
                                    "row": row,
                                    "column": column,
                                    "height": height,
                                    "width": width,
                                    "mean_score": score,
                                }
                                for row, column, height, width, score
                                in rectangles
                            ],
                            "illustrative_targets_are_phase1_masks": False,
                        }
                        if args.include_source_paths:
                            record["source_path"] = paths[global_index]
                        records.append(record)
                        save_guide_npz(
                            run_dir,
                            entries[global_index]["image_id"],
                            guide_name,
                            per_maps,
                            per_pca,
                            derived_grounding,
                            token_cut,
                            (
                                output.patch_tokens[local_index]
                                if args.save_tokens
                                else None
                            ),
                        )
            except Exception as exc:
                failure = {
                    "guide": guide_name,
                    "error_type": type(exc).__name__,
                    "error": sanitize_error(exc, paths),
                }
                failures.append(failure)
                print(
                    "[guide failure] %s: %s" % (guide_name, exc),
                    flush=True,
                )
                if args.strict:
                    raise
            finally:
                if guide is not None:
                    guide.cleanup()

        if not records:
            raise RuntimeError("all selected guides failed; no outputs produced")

        differences = []
        for index, original in enumerate(originals):
            if not atlas_data[index]:
                continue
            plot_atlas(
                original,
                entries[index],
                atlas_data[index],
                os.path.join(
                    run_dir,
                    "atlases",
                    entries[index]["image_id"] + ".png",
                ),
                show_class_labels=args.show_class_labels,
            )
            dino = atlas_data[index].get("dinov3")
            if dino is None or dino.get("primary") is None:
                continue
            for name in ("qwen3_vl", "molmo"):
                vlm = atlas_data[index].get(name)
                if (
                    vlm is None
                    or vlm.get("derived_grounding_raster") is None
                ):
                    continue
                value = difference_map(
                    dino["primary"], vlm["derived_grounding_raster"]
                )
                np.savez_compressed(
                    os.path.join(
                        run_dir,
                        "maps",
                        "%s__dinov3-minus-%s.npz"
                        % (entries[index]["image_id"], name),
                    ),
                    difference=value.numpy(),
                )
                differences.append(
                    {
                        "image_id": entries[index]["image_id"],
                        "left": "dinov3",
                        "right": name,
                        "mean_absolute_difference": float(
                            value.abs().mean().item()
                        ),
                    }
                )

        image_manifest = []
        for index, entry in enumerate(entries):
            public_entry = {
                key: value
                for key, value in entry.items()
                if key != "path"
            }
            public_entry["input_index"] = index
            if args.include_source_paths:
                public_entry["source_path"] = entry["path"]
            image_manifest.append(public_entry)
        with open(
            os.path.join(run_dir, "semantic_map_metrics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "config": os.path.basename(args.config),
                    "selection": sanitize_metadata(selection_metadata),
                    "images": image_manifest,
                    "records": records,
                    "differences": differences,
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
