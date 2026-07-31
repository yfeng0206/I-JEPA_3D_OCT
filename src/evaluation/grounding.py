"""Grounding metrics against official ImageNet localization annotations."""


def box_iou(left, right):
    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must contain four coordinates")
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    if lx1 >= lx2 or ly1 >= ly2 or rx1 >= rx2 or ry1 >= ry2:
        raise ValueError("boxes must be ordered with positive area")
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / (left_area + right_area - intersection)


def grounding_box_pixels(box, image_size):
    width, height = (float(image_size[0]), float(image_size[1]))
    coordinates = box["bbox_2d"]
    if box["coordinate_space"] == "normalized_1000":
        x1, y1, x2, y2 = coordinates
        return (
            float(x1) * width / 1000.0,
            float(y1) * height / 1000.0,
            float(x2) * width / 1000.0,
            float(y2) * height / 1000.0,
        )
    if box["coordinate_space"] == "pixels":
        source_width, source_height = box["image_size"]
        x1, y1, x2, y2 = coordinates
        return (
            float(x1) * width / float(source_width),
            float(y1) * height / float(source_height),
            float(x2) * width / float(source_width),
            float(y2) * height / float(source_height),
        )
    raise ValueError("unsupported box coordinate space")


def grounding_point_pixels(point, image_size):
    width, height = (float(image_size[0]), float(image_size[1]))
    x, y = point["point_2d"]
    if point["coordinate_space"] == "normalized_1000":
        return float(x) * width / 1000.0, float(y) * height / 1000.0
    if point["coordinate_space"] == "pixels":
        source_width, source_height = point["image_size"]
        return (
            float(x) * width / float(source_width),
            float(y) * height / float(source_height),
        )
    raise ValueError("unsupported point coordinate space")


def point_inside_box(point, box):
    x, y = (float(value) for value in point)
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 <= x <= x2 and y1 <= y <= y2


def evaluate_box_record(record, annotation):
    image_size = annotation["image_size"]
    truth = [item["bbox_xyxy"] for item in annotation["boxes"]]
    predictions = record.get("grounding_regions") or []
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        predicted_box = grounding_box_pixels(prediction, image_size)
        for truth_index, truth_box in enumerate(truth):
            candidates.append(
                {
                    "iou": box_iou(predicted_box, truth_box),
                    "prediction_index": prediction_index,
                    "truth_index": truth_index,
                    "prediction_label": prediction["label"],
                    "predicted_box": list(predicted_box),
                    "truth_box": list(truth_box),
                }
            )
    best = max(candidates, key=lambda item: item["iou"]) if candidates else None
    return {
        "prediction_count": len(predictions),
        "truth_count": len(truth),
        "best": best,
        "best_iou": float(best["iou"]) if best is not None else 0.0,
    }


def evaluate_point_record(record, annotation):
    image_size = annotation["image_size"]
    truth = [item["bbox_xyxy"] for item in annotation["boxes"]]
    predictions = record.get("grounding_points") or []
    hits = []
    for prediction_index, prediction in enumerate(predictions):
        pixel = grounding_point_pixels(prediction, image_size)
        matching = [
            truth_index
            for truth_index, truth_box in enumerate(truth)
            if point_inside_box(pixel, truth_box)
        ]
        hits.append(
            {
                "prediction_index": prediction_index,
                "prediction_label": prediction["label"],
                "pixel": list(pixel),
                "truth_indices": matching,
                "inside_any_truth_box": bool(matching),
            }
        )
    return {
        "prediction_count": len(predictions),
        "truth_count": len(truth),
        "hits": hits,
        "any_point_hit": any(item["inside_any_truth_box"] for item in hits),
        "point_hit_count": sum(
            item["inside_any_truth_box"] for item in hits
        ),
    }
