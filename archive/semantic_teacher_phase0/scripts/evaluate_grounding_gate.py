#!/usr/bin/env python
"""Score generated points/boxes against official ImageNet XML annotations."""

import argparse
import json
import os
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.grounding import evaluate_box_record, evaluate_point_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("boxes", "points"), required=True
    )
    return parser.parse_args()


def atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp-", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main():
    args = parse_args()
    with open(args.metrics, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(args.annotations, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    by_hash = {
        item["image_sha256"]: item for item in annotations["records"]
    }
    results = []
    for image, record in zip(metrics["images"], metrics["records"]):
        if image["image_sha256"] not in by_hash:
            raise RuntimeError(
                "missing annotation for %s" % image["image_sha256"]
            )
        annotation = by_hash[image["image_sha256"]]
        evaluation = (
            evaluate_box_record(record, annotation)
            if args.mode == "boxes"
            else evaluate_point_record(record, annotation)
        )
        results.append(
            {
                "image_id": image["image_id"],
                "sample_id": image.get("sample_id"),
                "class_name": image.get("class_name"),
                "failures": record.get("failures", []),
                "evaluation": evaluation,
            }
        )
    if args.mode == "boxes":
        summary = {
            "count": len(results),
            "valid_prediction_count": sum(
                item["evaluation"]["prediction_count"] > 0
                for item in results
            ),
            "iou_50_count": sum(
                item["evaluation"]["best_iou"] >= 0.5 for item in results
            ),
            "mean_best_iou": sum(
                item["evaluation"]["best_iou"] for item in results
            )
            / len(results),
        }
    else:
        summary = {
            "count": len(results),
            "valid_prediction_count": sum(
                item["evaluation"]["prediction_count"] > 0
                for item in results
            ),
            "any_point_hit_count": sum(
                item["evaluation"]["any_point_hit"] for item in results
            ),
        }
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "mode": args.mode,
            "metrics_source": os.path.abspath(args.metrics),
            "annotation_source": os.path.abspath(args.annotations),
            "summary": summary,
            "records": results,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
