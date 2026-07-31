#!/usr/bin/env python
"""Build a deterministic, model-blind ImageNet-50 atlas manifest."""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.imagenet_subset import (  # noqa: E402
    ImageNetSubsetDataset,
    atlas_selection_score,
    load_class_names,
)
from src.evaluation.feature_cache import (  # noqa: E402
    atomic_write_json,
    dataset_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select ImageNet-50 atlas images without model output."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--wnid-manifest",
        default=(
            "configs/semantic_maps/manifests/"
            "phase0-cmc-in100-prefix50-v1.wnids.txt"
        ),
    )
    parser.add_argument(
        "--class-names",
        default=(
            "configs/semantic_maps/manifests/"
            "phase0-cmc-in100-prefix50-v1.classes.json"
        ),
    )
    parser.add_argument(
        "--dataset-snapshot",
        default=(
            r"D:\jepa_phase0\results\manifests\dataset"
            r"\phase0-cmc-in100-prefix50-v1\train.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            r"D:\jepa_phase0\results\manifests"
            r"\phase0-atlas-inputs.json"
        ),
    )
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument(
        "--namespace", default="phase0-atlas-v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_selection(
    dataset,
    snapshot,
    class_names,
    per_class,
    max_images,
    namespace,
):
    if per_class <= 0 or max_images <= 0:
        raise ValueError("selection counts must be positive")
    hashes = {
        item["sample_id"]: item["sha256"] for item in snapshot["files"]
    }
    grouped = {wnid: [] for wnid in dataset.wnids}
    for record in dataset.records():
        sha256 = hashes.get(record["sample_id"])
        if sha256 is None:
            raise RuntimeError(
                "dataset snapshot is missing %s" % record["sample_id"]
            )
        score = atlas_selection_score(
            record["wnid"], sha256, namespace=namespace
        )
        grouped[record["wnid"]].append((score, sha256, record))

    selected = []
    selected_hashes = set()
    for class_index, wnid in enumerate(dataset.wnids):
        candidates = sorted(grouped[wnid], key=lambda item: item[0])
        if len(candidates) < per_class:
            raise RuntimeError(
                "%s has only %d images, fewer than --per-class=%d"
                % (wnid, len(candidates), per_class)
            )
        accepted = 0
        for score, sha256, record in candidates:
            if sha256 in selected_hashes:
                continue
            selected.append(
                {
                    "path": os.path.abspath(record["path"]),
                    "sample_id": record["sample_id"],
                    "image_sha256": sha256,
                    "wnid": wnid,
                    "class_index": class_index,
                    "class_name": class_names[wnid],
                    "selection_score": score,
                }
            )
            selected_hashes.add(sha256)
            accepted += 1
            if accepted == per_class:
                break
        if accepted != per_class:
            raise RuntimeError(
                "%s has only %d unique images after content deduplication"
                % (wnid, accepted)
            )
    return selected[:max_images]


def main():
    args = parse_args()
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(
            "output manifest exists; use --overwrite: %s" % args.output
        )
    dataset = ImageNetSubsetDataset(
        args.data_root,
        args.split,
        args.wnid_manifest,
        strict=True,
    )
    snapshot = dataset_snapshot(
        dataset,
        args.dataset_snapshot,
        overwrite=False,
    )
    class_names = load_class_names(args.class_names, dataset.wnids)
    images = build_selection(
        dataset,
        snapshot,
        class_names,
        per_class=args.per_class,
        max_images=args.max_images,
        namespace=args.namespace,
    )
    payload = {
        "schema_version": 1,
        "dataset_id": "phase0-cmc-in100-prefix50-v1",
        "split": args.split,
        "selection_namespace": args.namespace,
        "selection_uses_model_output": False,
        "dataset_content_sha256": snapshot["content_sha256"],
        "per_class": args.per_class,
        "count": len(images),
        "images": images,
    }
    atomic_write_json(args.output, payload)
    print("Saved %d atlas inputs to %s" % (len(images), args.output))


if __name__ == "__main__":
    main()
