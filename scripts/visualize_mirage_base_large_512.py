"""Render MIRAGE-Base versus MIRAGE-Large segmentation examples at 512."""
from __future__ import annotations

import argparse
import gc
import json
import pathlib

import cv2
import numpy as np
import torch

from compare_mirage_base_large_512 import (
    CLASS_NAMES,
    DEFAULT_BASE,
    DEFAULT_GOALS,
    DEFAULT_LARGE,
    IGNORE_INDEX,
    build_model,
    load_fairvision,
    load_goals,
    predict,
)

DEFAULT_OUT = pathlib.Path(
    r'D:\jepa_phase0\mirage-goals\outputs'
    r'\mergedv3-base-vs-large-512\visuals'
)

TILE = 320
HEADER = 48
PALETTE = np.asarray([
    (0, 0, 0),        # Elsewhere
    (180, 120, 255),  # InnerRetina, BGR
    (255, 200, 0),    # Choroid, BGR
    (0, 0, 255),      # ignored/background
], dtype=np.uint8)


def gray_bgr(image: np.ndarray) -> np.ndarray:
    gray = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def overlay(image: np.ndarray, hard: np.ndarray,
            alpha: float = 0.55) -> np.ndarray:
    base = gray_bgr(image)
    classes = PALETTE[hard]
    foreground = hard != 0
    result = base.copy()
    result[foreground] = (
        base[foreground].astype(np.float32) * (1 - alpha)
        + classes[foreground].astype(np.float32) * alpha
    ).astype(np.uint8)
    return result


def labeled_tile(image: np.ndarray, title: str,
                 subtitle: str = '') -> np.ndarray:
    resized = cv2.resize(image, (TILE, TILE), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TILE + HEADER, TILE, 3), dtype=np.uint8)
    canvas[HEADER:] = resized
    cv2.putText(canvas, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (7, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def add_legend(panel: np.ndarray, description: str) -> np.ndarray:
    strip = np.zeros((56, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, description, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (230, 230, 230), 1, cv2.LINE_AA)
    x = 8
    for class_index, name in ((1, 'InnerRetina'), (2, 'Choroid'),
                              (3, 'Background/ignore')):
        cv2.rectangle(strip, (x, 30), (x + 18, 48),
                      tuple(int(v) for v in PALETTE[class_index]), -1)
        cv2.putText(strip, name, (x + 24, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (230, 230, 230), 1, cv2.LINE_AA)
        x += 155
    return np.vstack([strip, panel])


def class_dice(prediction: np.ndarray, target: np.ndarray,
               class_index: int) -> float:
    valid = target != IGNORE_INDEX
    pred = (prediction == class_index) & valid
    truth = (target == class_index) & valid
    denominator = float(pred.sum() + truth.sum())
    return (2 * float((pred & truth).sum()) / denominator
            if denominator else float('nan'))


def disagreement_view(image: np.ndarray, base: np.ndarray,
                      large: np.ndarray) -> np.ndarray:
    result = (gray_bgr(image).astype(np.float32) * 0.32).astype(np.uint8)
    base_only = (base > 0) & (large == 0)
    large_only = (large > 0) & (base == 0)
    class_swap = (base > 0) & (large > 0) & (base != large)
    result[base_only] = (60, 60, 255)
    result[large_only] = (255, 150, 60)
    result[class_swap] = (0, 230, 255)
    return result


def select_goals(base: np.ndarray, large: np.ndarray,
                 targets: np.ndarray, count: int) -> list[int]:
    disagreement = np.mean(base != large, axis=(1, 2))
    base_fg = np.asarray([
        np.mean([class_dice(b, t, c) for c in (1, 2)])
        for b, t in zip(base, targets)
    ])
    large_fg = np.asarray([
        np.mean([class_dice(l, t, c) for c in (1, 2)])
        for l, t in zip(large, targets)
    ])
    advantage = base_fg - large_fg
    ordered = np.argsort(disagreement)
    candidates = [
        int(np.argmax(advantage)),
        int(np.argmin(advantage)),
        int(ordered[-1]),
        int(ordered[len(ordered) // 2]),
        int(ordered[len(ordered) // 4]),
        int(ordered[3 * len(ordered) // 4]),
    ]
    selected = []
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    for index in ordered:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == count:
            break
    return selected


def select_disagreement_quantiles(base: np.ndarray, large: np.ndarray,
                                  count: int) -> list[int]:
    disagreement = np.mean(base != large, axis=(1, 2))
    ordered = np.argsort(disagreement)
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [int(ordered[position]) for position in positions]


def probability_maps(model, image: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(image)[None, None].to(
        device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(
            device_type='cuda', dtype=torch.float16,
            enabled=device == 'cuda'):
        output = model({'bscan': tensor})
    logits = output['semseg'] if isinstance(output, dict) else output
    logits = logits.float().clone()
    logits[:, IGNORE_INDEX] = float('-inf')
    return torch.softmax(logits[:, :3], dim=1)[0].cpu().numpy()


def heatmap(probability: np.ndarray) -> np.ndarray:
    image = (np.clip(probability, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)


def render_goals(output: pathlib.Path, inputs: np.ndarray,
                 targets: np.ndarray, names: list[str],
                 predictions: dict[str, np.ndarray],
                 selected: list[int]) -> None:
    rows = []
    for index in selected:
        base = predictions['base'][index]
        large = predictions['large'][index]
        base_inner = class_dice(base, targets[index], 1)
        base_choroid = class_dice(base, targets[index], 2)
        large_inner = class_dice(large, targets[index], 1)
        large_choroid = class_dice(large, targets[index], 2)
        disagreement = float(np.mean(base != large))
        rows.append(np.hstack([
            labeled_tile(gray_bgr(inputs[index]), names[index],
                         'input B-scan'),
            labeled_tile(overlay(inputs[index], targets[index]), 'Ground truth',
                         'InnerRetina + Choroid'),
            labeled_tile(overlay(inputs[index], base), 'MIRAGE-Base',
                         'Dice I %.3f  C %.3f' % (
                             base_inner, base_choroid)),
            labeled_tile(overlay(inputs[index], large), 'MIRAGE-Large',
                         'Dice I %.3f  C %.3f' % (
                             large_inner, large_choroid)),
            labeled_tile(disagreement_view(
                inputs[index], base, large), 'Disagreement',
                '%.2f%% of pixels' % (100 * disagreement)),
        ]))
    panel = add_legend(
        np.vstack(rows),
        'GOALS test: same image, ground truth, Base, Large, disagreement. '
        'Disagreement: red=Base-only, blue=Large-only, yellow=class swap.',
    )
    cv2.imwrite(str(output / 'goals_examples.png'), panel)


def render_fairvision(output: pathlib.Path, inputs: np.ndarray,
                      predictions: dict[str, np.ndarray],
                      selected: list[int], depths: list[int]) -> None:
    rows = []
    slices_per_volume = len(depths)
    for index in selected:
        base = predictions['base'][index]
        large = predictions['large'][index]
        base_inner = float(np.mean(base == 1))
        base_choroid = float(np.mean(base == 2))
        large_inner = float(np.mean(large == 1))
        large_choroid = float(np.mean(large == 2))
        disagreement = float(np.mean(base != large))
        volume = index // slices_per_volume
        depth = depths[index % slices_per_volume]
        rows.append(np.hstack([
            labeled_tile(gray_bgr(inputs[index]),
                         'FairVision volume %02d' % volume,
                         'slice %d' % depth),
            labeled_tile(overlay(inputs[index], base), 'MIRAGE-Base',
                         'area I %.3f  C %.3f' % (
                             base_inner, base_choroid)),
            labeled_tile(overlay(inputs[index], large), 'MIRAGE-Large',
                         'area I %.3f  C %.3f' % (
                             large_inner, large_choroid)),
            labeled_tile(disagreement_view(
                inputs[index], base, large), 'Disagreement',
                '%.2f%% of pixels' % (100 * disagreement)),
        ]))
    panel = add_legend(
        np.vstack(rows),
        'FairVision transfer: no ground truth; examples span disagreement '
        'quantiles from lowest to highest.',
    )
    cv2.imwrite(str(output / 'fairvision_examples.png'), panel)


def render_probabilities(output: pathlib.Path, image: np.ndarray,
                         target: np.ndarray, name: str,
                         predictions: dict[str, np.ndarray],
                         probabilities: dict[str, np.ndarray]) -> None:
    top = np.hstack([
        labeled_tile(gray_bgr(image), name, 'input B-scan'),
        labeled_tile(overlay(image, target), 'Ground truth',
                     'hard pixel classes'),
        labeled_tile(np.zeros((512, 512, 3), dtype=np.uint8),
                     'Probability maps', 'yellow=high, purple=low'),
        labeled_tile(np.zeros((512, 512, 3), dtype=np.uint8),
                     'Classes', 'Elsewhere / Inner / Choroid'),
    ])
    rows = [top]
    for arm, label in (('base', 'MIRAGE-Base'), ('large', 'MIRAGE-Large')):
        rows.append(np.hstack([
            labeled_tile(overlay(image, predictions[arm]), label,
                         'hard classification'),
            labeled_tile(heatmap(probabilities[arm][0]), 'P(Elsewhere)',
                         'mean %.3f' % probabilities[arm][0].mean()),
            labeled_tile(heatmap(probabilities[arm][1]), 'P(InnerRetina)',
                         'mean %.3f' % probabilities[arm][1].mean()),
            labeled_tile(heatmap(probabilities[arm][2]), 'P(Choroid)',
                         'mean %.3f' % probabilities[arm][2].mean()),
        ]))
    panel = add_legend(
        np.vstack(rows),
        'Per-pixel classification: each model assigns three usable class '
        'probabilities, then argmax produces the hard segmentation.',
    )
    filename = 'class_probabilities_%s.png' % pathlib.Path(name).stem
    cv2.imwrite(str(output / filename), panel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ckpt', default=str(DEFAULT_BASE))
    parser.add_argument('--large-ckpt', default=str(DEFAULT_LARGE))
    parser.add_argument('--goals-root', default=str(DEFAULT_GOALS))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    output = pathlib.Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    goals_inputs, goals_targets, goals_names = load_goals(
        pathlib.Path(args.goals_root))
    fairvision_inputs, fairvision_meta = load_fairvision(20, 5, 7)
    all_inputs = np.concatenate([goals_inputs, fairvision_inputs])
    probability_index = len(goals_inputs) // 2

    checkpoints = {
        'base': pathlib.Path(args.base_ckpt),
        'large': pathlib.Path(args.large_ckpt),
    }
    goals_predictions = {}
    fairvision_predictions = {}
    probabilities = {}
    for arm in ('base', 'large'):
        print('running %s' % arm)
        model, _ = build_model(checkpoints[arm], device)
        predictions, _ = predict(model, all_inputs, device, args.batch_size)
        probabilities[arm] = probability_maps(
            model, goals_inputs[probability_index], device)
        goals_predictions[arm] = predictions[:len(goals_inputs)]
        fairvision_predictions[arm] = predictions[len(goals_inputs):]
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    goals_selected = select_goals(
        goals_predictions['base'],
        goals_predictions['large'],
        goals_targets,
        args.examples,
    )
    fairvision_selected = select_disagreement_quantiles(
        fairvision_predictions['base'],
        fairvision_predictions['large'],
        args.examples,
    )
    render_goals(
        output, goals_inputs, goals_targets, goals_names,
        goals_predictions, goals_selected)
    render_fairvision(
        output, fairvision_inputs, fairvision_predictions,
        fairvision_selected, fairvision_meta['depths'])
    render_probabilities(
        output,
        goals_inputs[probability_index],
        goals_targets[probability_index],
        goals_names[probability_index],
        {arm: goals_predictions[arm][probability_index]
         for arm in ('base', 'large')},
        probabilities,
    )

    manifest = {
        'goals_examples': [goals_names[index] for index in goals_selected],
        'goals_selection': (
            'Base-best advantage, Large-best advantage, maximum disagreement, '
            'and disagreement quartiles'
        ),
        'fairvision_examples': fairvision_selected,
        'fairvision_selection': 'disagreement quantiles from minimum to maximum',
        'probability_example': goals_names[probability_index],
        'classes': list(CLASS_NAMES),
        'palette_bgr': PALETTE.tolist(),
        'files': [
            'goals_examples.png',
            'fairvision_examples.png',
            'class_probabilities_%s.png'
            % pathlib.Path(goals_names[probability_index]).stem,
        ],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('wrote visuals to %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
