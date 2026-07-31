#!/usr/bin/env python
"""Map manually supplied phrases to Qwen boxes, points, and visual-token grids."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides import build_guide  # noqa: E402
from src.guides.maps import grounding_score_map  # noqa: E402
from src.guides.vlm_guides import parse_qwen_grounding  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a manual-keyword Qwen phrase-grounding oracle."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--phrases", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default="configs/semantic_maps/phase0_guides.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate(guide, image, prompt, max_new_tokens):
    inputs = guide._prepare(image, prompt)
    if guide.device.type == "cuda":
        torch.cuda.synchronize(guide.device)
    started = time.perf_counter()
    output = guide.model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
    )
    if guide.device.type == "cuda":
        torch.cuda.synchronize(guide.device)
    generated = output[:, inputs["input_ids"].shape[1] :]
    text = guide.processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, generated[0].detach().cpu().tolist(), inputs, (
        time.perf_counter() - started
    )


def normalized_label(value):
    return " ".join(str(value).lower().strip().split())


def match_phrase(items, phrase):
    target = normalized_label(phrase)
    exact = [item for item in items if normalized_label(item.label) == target]
    if exact:
        return exact[0]
    partial = [
        item
        for item in items
        if target in normalized_label(item.label)
        or normalized_label(item.label) in target
    ]
    if len(partial) == 1:
        return partial[0]
    raise RuntimeError(
        "could not uniquely match phrase %r to labels %s"
        % (phrase, [item.label for item in items])
    )


def token_records(values):
    records = []
    height, width = values.shape
    for row in range(height):
        for column in range(width):
            value = float(values[row, column])
            if value > 0:
                records.append(
                    {
                        "token_index": row * width + column,
                        "row": row,
                        "column": column,
                        "value": value,
                    }
                )
    return records


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            "output directory exists; use --overwrite: %s" % output_dir
        )
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    values = dict(config["guides"]["qwen3_vl"])
    for key in (
        "enabled",
        "official_model_id",
        "published_parameters",
        "published_visual_parameters",
        "evaluation_batch_size",
        "atlas_batch_size",
        "grounding_mode",
        "caption_max_new_tokens",
        "single_point_max_new_tokens",
        "plural_points_max_new_tokens",
        "boxes_max_new_tokens",
    ):
        values.pop(key, None)
    values["device"] = args.device

    with Image.open(args.image) as source:
        image = source.convert("RGB")

    guide = None
    try:
        guide = build_guide("qwen3_vl", **values)
        phrase_results = []
        grid_size = None
        figure, axes = plt.subplots(
            len(args.phrases),
            3,
            figsize=(12, 3.8 * len(args.phrases)),
            squeeze=False,
        )
        for index, phrase in enumerate(args.phrases):
            box_prompt = (
                "Locate the visible %s. Return only one JSON object: "
                '{"bbox_2d": [x1, y1, x2, y2], "label": "%s"}. '
                "Do not include background or any other object."
                % (phrase, phrase)
            )
            point_prompt = (
                "Locate the visible %s. Return only one JSON object: "
                '{"point_2d": [x, y], "label": "%s"}. '
                "Do not include background or any other object."
                % (phrase, phrase)
            )
            box_text, box_ids, box_inputs, box_latency = generate(
                guide, image, box_prompt, args.max_new_tokens
            )
            point_text, point_ids, _, point_latency = generate(
                guide, image, point_prompt, args.max_new_tokens
            )
            boxes = parse_qwen_grounding(
                box_text, "boxes", image_size=image.size
            )
            points = parse_qwen_grounding(
                point_text, "single_point", image_size=image.size
            )
            if len(boxes) != 1 or len(points) != 1:
                raise RuntimeError(
                    "phrase %r did not return exactly one box and point"
                    % phrase
                )
            box = match_phrase(boxes, phrase)
            point = match_phrase(points, phrase)
            grid_thw = (
                box_inputs["image_grid_thw"][0].detach().cpu().tolist()
            )
            if int(grid_thw[0]) != 1:
                raise RuntimeError("Qwen returned a non-image temporal grid")
            merge_size = int(
                guide.model.config.vision_config.spatial_merge_size
            )
            current_grid = (
                int(grid_thw[1]) // merge_size,
                int(grid_thw[2]) // merge_size,
            )
            if grid_size is None:
                grid_size = current_grid
            elif grid_size != current_grid:
                raise RuntimeError("phrase prompts produced different grids")
            box_map = grounding_score_map(
                [[box]], [[]], grid_size=grid_size
            )[0].numpy()
            point_map = grounding_score_map(
                [[]],
                [[point]],
                grid_size=grid_size,
                point_sigma_fraction=0.065,
            )[0].numpy()
            combined = np.maximum(box_map, point_map).astype(np.float32)
            safe_name = "_".join(normalized_label(phrase).split())
            np.save(output_dir / ("%s_box_grid.npy" % safe_name), box_map)
            np.save(output_dir / ("%s_point_grid.npy" % safe_name), point_map)
            np.save(
                output_dir / ("%s_combined_grid.npy" % safe_name), combined
            )
            np.savetxt(
                output_dir / ("%s_combined_grid.csv" % safe_name),
                combined,
                delimiter=",",
                fmt="%.8f",
            )
            phrase_results.append(
                {
                    "phrase": phrase,
                    "box_prompt": box_prompt,
                    "point_prompt": point_prompt,
                    "raw_box_output": box_text,
                    "raw_point_output": point_text,
                    "box_token_ids": box_ids,
                    "point_token_ids": point_ids,
                    "box_latency_seconds": box_latency,
                    "point_latency_seconds": point_latency,
                    "box": {
                        "label": box.label,
                        "bbox_2d": list(box.bbox_2d),
                        "coordinate_space": box.coordinate_space,
                    },
                    "point": {
                        "label": point.label,
                        "point_2d": list(point.point_2d),
                        "coordinate_space": point.coordinate_space,
                    },
                    "box_token_values": token_records(box_map),
                    "point_token_values": token_records(point_map),
                    "combined_token_values": token_records(combined),
                }
            )
            for column, (title, values_map) in enumerate(
                (
                    ("%s box coverage" % phrase, box_map),
                    ("%s point score" % phrase, point_map),
                    ("%s combined" % phrase, combined),
                )
            ):
                rendered = axes[index, column].imshow(
                    values_map,
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                axes[index, column].set_title(title)
                axes[index, column].set_xlabel("visual-token column")
                axes[index, column].set_ylabel("visual-token row")
                figure.colorbar(rendered, ax=axes[index, column])
        figure.tight_layout()
        figure.savefig(
            output_dir / "phrase_token_heatmaps.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)
        metadata = {
            "schema_version": 1,
            "method": "manual-keyword Qwen3-VL grounding oracle",
            "not_automatic_image_only_selection": True,
            "model_id": config["guides"]["qwen3_vl"]["official_model_id"],
            "revision": config["guides"]["qwen3_vl"]["revision"],
            "source_image_size": list(image.size),
            "visual_token_grid": list(grid_size),
            "visual_token_layout": "row-major; token_index=row*width+column",
            "phrases": phrase_results,
        }
        with open(
            output_dir / "phrase_token_mapping.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        print(output_dir / "phrase_token_heatmaps.png")
    finally:
        if guide is not None:
            guide.cleanup()


if __name__ == "__main__":
    main()
