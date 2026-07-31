#!/usr/bin/env python
"""Build the model-blind image manifest for the CNN channel atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


COCO_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/"
    "annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}"
)
COCO_TARGETS = {
    "animals": (
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ),
    "vehicles": (
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ),
    "tools": (
        "knife",
        "scissors",
        "toothbrush",
        "hair drier",
        "baseball bat",
        "tennis racket",
    ),
    "food": (
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ),
    "indoor_scenes": (
        "bed",
        "couch",
        "toilet",
        "oven",
        "microwave",
        "sink",
        "refrigerator",
        "tv",
        "laptop",
        "dining table",
    ),
}
CATEGORY_ORDER = ("animals", "vehicles", "tools", "food", "indoor_scenes")
IMAGENET_TARGETS = {
    "vehicles": (
        ("n03785016", "moped"),
        ("n02701002", "ambulance"),
    ),
    "tools": (
        ("n04517823", "vacuum"),
        ("n04485082", "tripod"),
        ("n03085013", "computer keyboard"),
        ("n03062245", "cocktail shaker"),
    ),
}
AREA_RANGES = {
    "animals": (0.08, 0.65),
    "vehicles": (0.08, 0.65),
    "tools": (0.008, 0.45),
    "food": (0.03, 0.60),
    "indoor_scenes": (0.04, 0.65),
}
MIN_BOX_SIDE = {
    "animals": 48.0,
    "vehicles": 48.0,
    "tools": 20.0,
    "food": 32.0,
    "indoor_scenes": 40.0,
}
INDOOR_MARKERS = frozenset(COCO_TARGETS["indoor_scenes"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deterministic, model-blind CNN atlas manifest."
    )
    parser.add_argument("--coco-annotations", required=True)
    parser.add_argument("--coco-images-dir", required=True)
    parser.add_argument("--imagenet-atlas", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--namespace", default="cnn-channel-atlas50-v1")
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(namespace, *parts):
    payload = "\0".join(str(value) for value in (namespace,) + parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preprocessing_geometry(width, height, resize_shorter=256, crop_size=224):
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if resize_shorter < crop_size or crop_size <= 0:
        raise ValueError("resize_shorter must be at least crop_size")
    scale = float(resize_shorter) / min(width, height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    left = (resized_width - crop_size) // 2
    top = (resized_height - crop_size) // 2
    return {
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_box_resized": [
            left,
            top,
            left + crop_size,
            top + crop_size,
        ],
    }


def bbox_retained_fraction(bbox, width, height, geometry):
    if len(bbox) != 4:
        raise ValueError("bbox must contain x, y, width, height")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if box_width <= 0.0 or box_height <= 0.0:
        return 0.0
    scale_x = float(geometry["resized_width"]) / float(width)
    scale_y = float(geometry["resized_height"]) / float(height)
    box = (
        x * scale_x,
        y * scale_y,
        (x + box_width) * scale_x,
        (y + box_height) * scale_y,
    )
    crop = tuple(float(value) for value in geometry["crop_box_resized"])
    intersection_width = max(0.0, min(box[2], crop[2]) - max(box[0], crop[0]))
    intersection_height = max(
        0.0, min(box[3], crop[3]) - max(box[1], crop[1])
    )
    return float(
        intersection_width
        * intersection_height
        / ((box[2] - box[0]) * (box[3] - box[1]))
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_coco_index(payload):
    images = {int(row["id"]): row for row in payload["images"]}
    category_names = {
        int(row["id"]): str(row["name"]) for row in payload["categories"]
    }
    category_ids = {name: key for key, name in category_names.items()}
    annotations_by_category = {
        category_id: {} for category_id in category_names
    }
    image_category_names = {image_id: set() for image_id in images}
    for annotation in payload["annotations"]:
        category_id = int(annotation["category_id"])
        image_id = int(annotation["image_id"])
        annotations_by_category.setdefault(category_id, {}).setdefault(
            image_id, []
        ).append(annotation)
        image_category_names.setdefault(image_id, set()).add(
            category_names[category_id]
        )
    return (
        images,
        category_ids,
        annotations_by_category,
        image_category_names,
    )


def candidate_for_image(
    image,
    annotations,
    broad_category,
    resize_shorter,
    crop_size,
):
    width = int(image["width"])
    height = int(image["height"])
    image_area = float(width * height)
    geometry = preprocessing_geometry(
        width, height, resize_shorter, crop_size
    )
    lower, upper = AREA_RANGES[broad_category]
    candidates = []
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        box_area = float(bbox[2]) * float(bbox[3])
        area_fraction = box_area / image_area
        if not lower <= area_fraction <= upper:
            continue
        if min(float(bbox[2]), float(bbox[3])) < MIN_BOX_SIDE[broad_category]:
            continue
        retained = bbox_retained_fraction(bbox, width, height, geometry)
        if retained < 0.9:
            continue
        target_midpoint = (lower + upper) / 2.0
        candidates.append(
            (
                abs(area_fraction - target_midpoint),
                -box_area,
                int(annotation["id"]),
                annotation,
                area_fraction,
                retained,
            )
        )
    if not candidates:
        return None
    _, _, _, annotation, area_fraction, retained = min(candidates)
    return {
        "annotation_id": int(annotation["id"]),
        "bbox_xywh": [float(value) for value in annotation["bbox"]],
        "bbox_area_fraction": area_fraction,
        "crop_retained_fraction": retained,
    }


def select_coco_rows(payload, namespace, resize_shorter, crop_size):
    (
        images,
        category_ids,
        annotations_by_category,
        image_category_names,
    ) = build_coco_index(payload)
    selected = []
    used_image_ids = set()
    for broad_category in CATEGORY_ORDER:
        for concept in COCO_TARGETS[broad_category]:
            category_id = category_ids.get(concept)
            if category_id is None:
                raise RuntimeError("COCO category is missing: %s" % concept)
            candidates = []
            for image_id, annotations in annotations_by_category[
                category_id
            ].items():
                if image_id in used_image_ids:
                    continue
                category_names = image_category_names[image_id]
                if broad_category == "indoor_scenes":
                    if len(category_names) < 3:
                        continue
                    if len(category_names & INDOOR_MARKERS) < 2:
                        continue
                target = candidate_for_image(
                    images[image_id],
                    annotations,
                    broad_category,
                    resize_shorter,
                    crop_size,
                )
                if target is None:
                    continue
                score = stable_score(
                    namespace, broad_category, concept, image_id
                )
                candidates.append((score, image_id, target))
            if not candidates:
                raise RuntimeError(
                    "no eligible COCO image for %s/%s"
                    % (broad_category, concept)
                )
            _, image_id, target = min(candidates)
            image = images[image_id]
            used_image_ids.add(image_id)
            row = {
                "broad_category": broad_category,
                "target_concept": concept,
                "source_dataset": "coco-2017-val",
                "source_id": str(image_id),
                "file_name": str(image["file_name"]),
                "source_url": COCO_IMAGE_URL.format(
                    file_name=image["file_name"]
                ),
                "original_width": int(image["width"]),
                "original_height": int(image["height"]),
                "source_categories": sorted(image_category_names[image_id]),
                "selection_score": stable_score(
                    namespace, broad_category, concept, image_id
                ),
            }
            row.update(target)
            selected.append(row)
    return selected


def select_imagenet_rows(payload, namespace):
    rows_by_wnid = {}
    for row in payload["images"]:
        rows_by_wnid.setdefault(str(row["wnid"]), []).append(row)
    selected = []
    for broad_category, targets in IMAGENET_TARGETS.items():
        for wnid, concept in targets:
            candidates = rows_by_wnid.get(wnid, [])
            if not candidates:
                raise RuntimeError(
                    "ImageNet atlas has no candidates for %s" % wnid
                )
            source = min(
                candidates,
                key=lambda row: stable_score(
                    namespace,
                    broad_category,
                    concept,
                    row["image_sha256"],
                ),
            )
            selected.append(
                {
                    "broad_category": broad_category,
                    "target_concept": concept,
                    "source_dataset": str(payload["dataset_id"]),
                    "source_id": str(source["sample_id"]),
                    "wnid": wnid,
                    "class_name": str(source["class_name"]),
                    "source_path": os.path.abspath(source["path"]),
                    "expected_image_sha256": str(source["image_sha256"]),
                    "selection_score": stable_score(
                        namespace,
                        broad_category,
                        concept,
                        source["image_sha256"],
                    ),
                }
            )
    return selected


def download_file(url, output_path, retries=3):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for _ in range(int(retries)):
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            os.replace(temporary, output_path)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise RuntimeError("failed to download %s: %s" % (url, last_error))


def preprocess_image(
    source_path,
    output_path,
    resize_shorter,
    crop_size,
):
    source_path = Path(source_path)
    output_path = Path(output_path)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        geometry = preprocessing_geometry(
            original_width,
            original_height,
            resize_shorter,
            crop_size,
        )
        image = image.resize(
            (geometry["resized_width"], geometry["resized_height"]),
            resample=Image.Resampling.BICUBIC,
        )
        image = image.crop(tuple(geometry["crop_box_resized"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
    return {
        "original_width": original_width,
        "original_height": original_height,
        **geometry,
    }


def materialize_rows(
    rows,
    coco_images_dir,
    preprocessed_dir,
    resize_shorter,
    crop_size,
):
    coco_images_dir = Path(coco_images_dir)
    preprocessed_dir = Path(preprocessed_dir)
    materialized = []
    for index, source in enumerate(rows):
        row = dict(source)
        if row["source_dataset"] == "coco-2017-val":
            source_path = coco_images_dir / row["file_name"]
            if not source_path.exists():
                download_file(row["source_url"], source_path)
        else:
            source_path = Path(row["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError("source image is missing: %s" % source_path)

        image_hash = file_sha256(source_path)
        expected = row.get("expected_image_sha256")
        if expected is not None and image_hash != expected:
            raise RuntimeError(
                "source image hash mismatch: %s" % source_path
            )
        safe_concept = row["target_concept"].replace(" ", "_")
        output_path = (
            preprocessed_dir
            / row["broad_category"]
            / ("%02d_%s.png" % (index, safe_concept))
        )
        geometry = preprocess_image(
            source_path,
            output_path,
            resize_shorter,
            crop_size,
        )
        row.update(geometry)
        row.update(
            {
                "index": index,
                "source_path": os.path.abspath(source_path),
                "image_sha256": image_hash,
                "preprocessed_path": os.path.abspath(output_path),
                "preprocessed_sha256": file_sha256(output_path),
            }
        )
        materialized.append(row)
    hashes = [row["image_sha256"] for row in materialized]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("manifest contains duplicate image content")
    return materialized


def assign_stability_subset(rows, namespace):
    by_category = {category: [] for category in CATEGORY_ORDER}
    for row in rows:
        by_category[row["broad_category"]].append(row)
    for broad_category, category_rows in by_category.items():
        selected = sorted(
            category_rows,
            key=lambda row: stable_score(
                namespace,
                "stability",
                broad_category,
                row["image_sha256"],
            ),
        )[:4]
        selected_hashes = {row["image_sha256"] for row in selected}
        for row in category_rows:
            row["stability_subset"] = (
                row["image_sha256"] in selected_hashes
            )


def render_contact_sheet(rows, output_path):
    thumbnails = []
    for row in rows:
        with Image.open(row["preprocessed_path"]) as image:
            thumbnails.append(image.convert("RGB").resize((160, 160)))
    columns = 5
    cell_width = 180
    cell_height = 195
    rows_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * cell_width, rows_count * cell_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image, row) in enumerate(zip(thumbnails, rows)):
        column = index % columns
        row_index = index // columns
        x = column * cell_width + 10
        y = row_index * cell_height + 5
        sheet.paste(image, (x, y))
        draw.text(
            (x, y + 163),
            "%02d %s/%s"
            % (index, row["broad_category"], row["target_concept"]),
            fill="black",
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError("cannot write an empty manifest CSV")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            "output manifest exists; use --overwrite: %s" % manifest_path
        )
    coco_payload = load_json(args.coco_annotations)
    imagenet_payload = load_json(args.imagenet_atlas)
    rows = select_coco_rows(
        coco_payload,
        args.namespace,
        args.resize_shorter,
        args.crop_size,
    )
    rows.extend(select_imagenet_rows(imagenet_payload, args.namespace))
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    rows.sort(
        key=lambda row: (
            order[row["broad_category"]],
            row["selection_score"],
        )
    )
    rows = materialize_rows(
        rows,
        args.coco_images_dir,
        output_dir / "preprocessed",
        args.resize_shorter,
        args.crop_size,
    )
    assign_stability_subset(rows, args.namespace)
    payload = {
        "schema_version": 1,
        "experiment_id": args.namespace,
        "selection_uses_model_output": False,
        "count": len(rows),
        "category_counts": {
            category: sum(
                row["broad_category"] == category for row in rows
            )
            for category in CATEGORY_ORDER
        },
        "preprocessing": {
            "resize_shorter": args.resize_shorter,
            "crop_size": args.crop_size,
            "interpolation": "PIL.Image.Resampling.BICUBIC",
            "rgb_conversion": True,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "sources": {
            "coco_annotations_path": os.path.abspath(args.coco_annotations),
            "coco_annotations_sha256": file_sha256(args.coco_annotations),
            "coco_annotations_url": COCO_ANNOTATIONS_URL,
            "coco_image_url_pattern": COCO_IMAGE_URL,
            "imagenet_atlas_path": os.path.abspath(args.imagenet_atlas),
            "imagenet_atlas_sha256": file_sha256(args.imagenet_atlas),
        },
        "images": rows,
    }
    atomic_write_json(manifest_path, payload)
    write_csv(output_dir / "manifest.csv", rows)
    render_contact_sheet(rows, output_dir / "contact_sheet.jpg")
    print("Saved %d manifest images to %s" % (len(rows), manifest_path))


if __name__ == "__main__":
    main()
