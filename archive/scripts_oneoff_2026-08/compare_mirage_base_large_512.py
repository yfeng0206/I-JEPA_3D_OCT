"""Compare MergedV3 MIRAGE-Base and MIRAGE-Large checkpoints at 512.

Both arms use the same strict checkpoint loader, preprocessing, void-channel
suppression, GOALS test set, and FairVision plausibility protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import sys
import time
from argparse import Namespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fairvision_model_compare import (  # noqa: E402
    DATA,
    MIRAGE_WS,
    col_coverage,
    components,
    largest_gap,
    mean_runs,
    topology_violation,
)

DEFAULT_BASE = (
    MIRAGE_WS / 'outputs' / 'mergedv3-base-512' / 'MergedV3' /
    'MIRAGE-Base_frozen_convnext_CEGDice-ignore' / 'checkpoint-best.pth'
)
DEFAULT_LARGE = (
    MIRAGE_WS / 'outputs' / 'mergedv3-512' / 'MergedV3' /
    'MIRAGE-Large_frozen_convnext_CEGDice-ignore' / 'checkpoint-34.pth'
)
DEFAULT_GOALS = pathlib.Path(
    r'D:\jepa_phase0\mirage-datasets\MergedV3\test'
)
DEFAULT_OUT = (
    MIRAGE_WS / 'outputs' / 'mergedv3-base-vs-large-512' / 'comparison.json'
)

SIZE = 512
PATCH_SIZE = 32
IGNORE_INDEX = 3
CLASS_NAMES = ('Elsewhere', 'InnerRetina', 'Choroid')
VALUE_TO_INDEX = {0: 0, 128: 1, 255: 2, 1: IGNORE_INDEX}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_spec(state: dict) -> tuple[str, int]:
    dim = int(state['global_tokens'].shape[-1])
    factories = {768: 'mirage-base', 1024: 'mirage-large'}
    if dim not in factories:
        raise ValueError('unsupported MIRAGE embedding dimension: %d' % dim)
    pos = state['input_adapters.bscan.pos_emb']
    if tuple(pos.shape[-2:]) != (SIZE // PATCH_SIZE, SIZE // PATCH_SIZE):
        raise ValueError('checkpoint is not on the native 16x16 grid: %s'
                         % (tuple(pos.shape),))
    return factories[dim], dim


def build_model(checkpoint: pathlib.Path, device: str):
    sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
    from fm_seg_config import fm_factory
    from mirage.model import model_factory
    from mirage.output_adapters import ConvNeXtAdapter

    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    factory, dim = checkpoint_spec(state)

    cfg = fm_factory[factory]()
    cfg.build_domain_conf()
    grid = SIZE // PATCH_SIZE
    runtime_args = Namespace(
        grid_sizes={'bscan': [grid, grid]},
        input_size={'bscan': [SIZE, SIZE]},
    )
    input_adapters = {
        'bscan': cfg.domain_conf['bscan']['input_adapter'](
            stride_level=1,
            patch_size_full=[PATCH_SIZE, PATCH_SIZE],
            image_size=[SIZE, SIZE],
            learnable_pos_emb=False,
        )
    }
    output_adapters = {
        'semseg': ConvNeXtAdapter(
            num_classes=4,
            preds_per_patch=16,
            depth=4,
            interpolate_mode='bilinear',
            main_tasks=['bscan'],
            embed_dim=6144,
            patch_size=[PATCH_SIZE, PATCH_SIZE],
            task='semseg',
            image_size=[SIZE, SIZE],
        )
    }
    model = model_factory[cfg.model](
        args=runtime_args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=1,
        drop_path_rate=0.1,
    )
    model.load_state_dict(state, strict=True)
    del blob, state
    gc.collect()
    return model.to(device).eval(), {
        'factory': factory,
        'embed_dim': dim,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
    }


def predict(model, inputs: np.ndarray, device: str,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    ignore_rates = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size, None]).to(
            device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16,
                enabled=device == 'cuda'):
            result = model({'bscan': batch})
        logits = result['semseg'] if isinstance(result, dict) else result
        logits = logits.float().clone()
        raw = logits.argmax(1)
        ignore_rates.extend(
            (raw == IGNORE_INDEX).float().mean(dim=(1, 2)).cpu().tolist())
        logits[:, IGNORE_INDEX] = float('-inf')
        outputs.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(outputs), np.asarray(ignore_rates, dtype=np.float64)


def normalize_resize(array: np.ndarray) -> np.ndarray:
    import cv2

    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(array.min()), float(array.max())
    unit = ((array - lo) / (hi - lo)
            if hi > lo else np.zeros_like(array))
    return cv2.resize(unit, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def load_goals(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import cv2

    image_dir = root / 'bscan'
    mask_dir = root / 'semseg'
    image_paths = sorted(image_dir.glob('*.png'))
    if not image_paths:
        raise FileNotFoundError('no GOALS test images in %s' % image_dir)

    inputs, targets, names = [], [], []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        inputs.append(normalize_resize(
            np.array(Image.open(image_path).convert('L'))))

        raw = np.array(Image.open(mask_path).convert('L'))
        target = np.full(raw.shape, 255, dtype=np.uint8)
        for value, index in VALUE_TO_INDEX.items():
            target[raw == value] = index
        if (target == 255).any():
            raise ValueError('unmapped values in %s: %s' % (
                mask_path, np.unique(raw[target == 255]).tolist()))
        targets.append(cv2.resize(
            target, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
        names.append(image_path.name)
    return np.stack(inputs), np.stack(targets), names


def class_metrics(predictions: np.ndarray,
                  targets: np.ndarray) -> dict:
    per_class = {}
    aggregate = {}
    for class_index, name in enumerate(CLASS_NAMES):
        dice, iou = [], []
        total_intersection = total_pred = total_truth = 0.0
        for prediction, target in zip(predictions, targets):
            valid = target != IGNORE_INDEX
            pred = (prediction == class_index) & valid
            truth = (target == class_index) & valid
            intersection = float((pred & truth).sum())
            pred_sum, truth_sum = float(pred.sum()), float(truth.sum())
            total_intersection += intersection
            total_pred += pred_sum
            total_truth += truth_sum
            union = pred_sum + truth_sum - intersection
            if pred_sum + truth_sum:
                dice.append(2 * intersection / (pred_sum + truth_sum))
            if union:
                iou.append(intersection / union)
        per_class[name] = {
            'dice': float(np.mean(dice)),
            'iou': float(np.mean(iou)),
            'n': len(dice),
        }
        total_union = total_pred + total_truth - total_intersection
        aggregate[name] = {
            'dice': 2 * total_intersection / (total_pred + total_truth),
            'iou': total_intersection / total_union,
        }

    all_dice = float(np.mean(
        [per_class[name]['dice'] for name in CLASS_NAMES]))
    foreground_dice = float(np.mean(
        [per_class[name]['dice'] for name in CLASS_NAMES[1:]]))
    mean_iou = float(np.mean(
        [per_class[name]['iou'] for name in CLASS_NAMES]))
    aggregate_all_dice = float(np.mean(
        [aggregate[name]['dice'] for name in CLASS_NAMES]))
    aggregate_foreground_dice = float(np.mean(
        [aggregate[name]['dice'] for name in CLASS_NAMES[1:]]))
    aggregate_mean_iou = float(np.mean(
        [aggregate[name]['iou'] for name in CLASS_NAMES]))
    return {
        'per_class': per_class,
        'all_class_mean_dice': all_dice,
        'foreground_mean_dice': foreground_dice,
        'mean_iou': mean_iou,
        'aggregate_per_class': aggregate,
        'aggregate_all_class_mean_dice': aggregate_all_dice,
        'aggregate_foreground_mean_dice': aggregate_foreground_dice,
        'aggregate_mean_iou': aggregate_mean_iou,
    }


def load_fairvision(volumes: int, slices: int,
                    seed: int) -> tuple[np.ndarray, dict]:
    files = sorted(DATA.glob('*.npz'))
    if not files:
        raise FileNotFoundError('no FairVision volumes in %s' % DATA)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(volumes, len(files)),
                       replace=False)
    depths = np.linspace(20, 180, num=slices).astype(int)
    inputs = []
    for volume_index in picks:
        with np.load(files[int(volume_index)], allow_pickle=True) as archive:
            volume = archive['oct_bscans']
        for depth in depths:
            inputs.append(normalize_resize(volume[int(depth)]))
    metadata = {
        'volumes': len(picks),
        'slices_per_volume': len(depths),
        'depths': depths.tolist(),
        'seed': seed,
        'n_slices': len(inputs),
    }
    return np.stack(inputs), metadata


def fairvision_metrics(inputs: np.ndarray,
                       predictions: np.ndarray) -> dict:
    import cv2

    rpe_rows = []
    for array in inputs:
        small = cv2.resize(array, (200, 200),
                           interpolation=cv2.INTER_LINEAR)
        profile = cv2.GaussianBlur(small, (1, 9), 0)
        rpe_rows.append(np.argmax(profile, axis=0))

    bad = total = 0
    runs, area, ncomp, biggest = [], [], [], []
    inner_area, choroid_area = [], []
    inner_coverage, choroid_coverage, inner_gap = [], [], []
    inner_ncomp, inner_biggest = [], []
    inner_wrong, choroid_wrong = [], []

    for index, prediction in enumerate(predictions):
        hard = cv2.resize(
            prediction, (200, 200), interpolation=cv2.INTER_NEAREST)
        inner = hard == 1
        choroid = hard == 2
        union = inner | choroid

        violations, evaluable = topology_violation(inner, choroid)
        bad += violations
        total += evaluable
        runs += mean_runs(union)
        area.append(float(union.mean()))
        count, largest = components(union)
        ncomp.append(count)
        biggest.append(largest)

        inner_area.append(float(inner.mean()))
        choroid_area.append(float(choroid.mean()))
        inner_coverage.append(col_coverage(inner))
        choroid_coverage.append(col_coverage(choroid))
        inner_gap.append(largest_gap(inner))
        count, largest = components(inner)
        inner_ncomp.append(count)
        inner_biggest.append(largest)

        row_indices = np.arange(hard.shape[0])[:, None]
        below_rpe = row_indices > rpe_rows[index][None, :]
        if inner.any():
            inner_wrong.append(float(
                (inner & below_rpe).sum() / inner.sum()))
        if choroid.any():
            choroid_wrong.append(float(
                (choroid & ~below_rpe).sum() / choroid.sum()))

    return {
        'topology_violation': bad / total if total else float('nan'),
        'evaluable_columns': total,
        'runs_per_column': float(np.mean(runs)),
        'union_area': float(np.mean(area)),
        'n_components': float(np.mean(ncomp)),
        'largest_component_share': float(np.mean(biggest)),
        'inner_area': float(np.mean(inner_area)),
        'choroid_area': float(np.mean(choroid_area)),
        'inner_col_coverage': float(np.mean(inner_coverage)),
        'choroid_col_coverage': float(np.mean(choroid_coverage)),
        'inner_largest_gap': float(np.mean(inner_gap)),
        'inner_n_components': float(np.mean(inner_ncomp)),
        'inner_largest_share': float(np.mean(inner_biggest)),
        'inner_below_rpe': float(np.mean(inner_wrong)),
        'choroid_above_rpe': float(np.mean(choroid_wrong)),
    }


def numeric_delta(base: dict, large: dict) -> dict:
    result = {}
    for key in base.keys() & large.keys():
        if (isinstance(base[key], (int, float))
                and isinstance(large[key], (int, float))):
            result[key] = float(base[key] - large[key])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ckpt', default=str(DEFAULT_BASE))
    parser.add_argument('--large-ckpt', default=str(DEFAULT_LARGE))
    parser.add_argument('--goals-root', default=str(DEFAULT_GOALS))
    parser.add_argument('--fairvision-volumes', type=int, default=20)
    parser.add_argument('--fairvision-slices', type=int, default=5)
    parser.add_argument('--fairvision-seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    checkpoints = {
        'base': pathlib.Path(args.base_ckpt),
        'large': pathlib.Path(args.large_ckpt),
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    goals_inputs, goals_targets, goals_names = load_goals(
        pathlib.Path(args.goals_root))
    fairvision_inputs, fairvision_meta = load_fairvision(
        args.fairvision_volumes,
        args.fairvision_slices,
        args.fairvision_seed,
    )
    all_inputs = np.concatenate([goals_inputs, fairvision_inputs])

    result = {
        'contract': {
            'input_size': SIZE,
            'patch_size': PATCH_SIZE,
            'token_grid': [16, 16],
            'void_suppressed_before_argmax': True,
            'strict_checkpoint_loading': True,
            'goals_images': goals_names,
            'fairvision': fairvision_meta,
        },
        'checkpoints': {},
        'goals': {'arms': {}},
        'fairvision': {'arms': {}},
    }
    fairvision_predictions = {}

    for arm in ('base', 'large'):
        checkpoint = checkpoints[arm]
        print('[%s] loading %s' % (arm, checkpoint))
        model, spec = build_model(checkpoint, device)
        predict(model, all_inputs[:args.batch_size], device, args.batch_size)
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        predictions, ignore_rates = predict(
            model, all_inputs, device, args.batch_size)
        if device == 'cuda':
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        goals_predictions = predictions[:len(goals_inputs)]
        fv_predictions = predictions[len(goals_inputs):]
        fairvision_predictions[arm] = fv_predictions

        goals_score = class_metrics(goals_predictions, goals_targets)
        goals_score['predicted_ignore_rate'] = float(
            ignore_rates[:len(goals_inputs)].mean())
        fv_score = fairvision_metrics(fairvision_inputs, fv_predictions)
        fv_score['predicted_ignore_rate'] = float(
            ignore_rates[len(goals_inputs):].mean())

        result['checkpoints'][arm] = {
            'path': str(checkpoint),
            'bytes': checkpoint.stat().st_size,
            'sha256': sha256(checkpoint),
            **spec,
        }
        result['checkpoints'][arm]['inference'] = {
            'images': len(all_inputs),
            'batch_size': args.batch_size,
            'seconds': inference_seconds,
            'images_per_second': len(all_inputs) / inference_seconds,
        }
        result['goals']['arms'][arm] = goals_score
        result['fairvision']['arms'][arm] = fv_score
        print('  GOALS aggregate Dice %.4f  inner %.4f  choroid %.4f'
              % (goals_score['aggregate_all_class_mean_dice'],
                 goals_score['aggregate_per_class']['InnerRetina']['dice'],
                 goals_score['aggregate_per_class']['Choroid']['dice']))
        print('  throughput %.1f images/s'
              % result['checkpoints'][arm]['inference']['images_per_second'])

    result['goals']['base_minus_large'] = {
        'all_class_mean_dice': (
            result['goals']['arms']['base']['all_class_mean_dice']
            - result['goals']['arms']['large']['all_class_mean_dice']),
        'foreground_mean_dice': (
            result['goals']['arms']['base']['foreground_mean_dice']
            - result['goals']['arms']['large']['foreground_mean_dice']),
        'mean_iou': (
            result['goals']['arms']['base']['mean_iou']
            - result['goals']['arms']['large']['mean_iou']),
        'aggregate_all_class_mean_dice': (
            result['goals']['arms']['base']['aggregate_all_class_mean_dice']
            - result['goals']['arms']['large']['aggregate_all_class_mean_dice']),
        'aggregate_foreground_mean_dice': (
            result['goals']['arms']['base']['aggregate_foreground_mean_dice']
            - result['goals']['arms']['large']['aggregate_foreground_mean_dice']),
        'aggregate_mean_iou': (
            result['goals']['arms']['base']['aggregate_mean_iou']
            - result['goals']['arms']['large']['aggregate_mean_iou']),
        'per_class_dice': {
            name: (
                result['goals']['arms']['base']['per_class'][name]['dice']
                - result['goals']['arms']['large']['per_class'][name]['dice'])
            for name in CLASS_NAMES
        },
        'aggregate_per_class_dice': {
            name: (
                result['goals']['arms']['base']['aggregate_per_class'][name]['dice']
                - result['goals']['arms']['large']['aggregate_per_class'][name]['dice'])
            for name in CLASS_NAMES
        },
    }
    result['fairvision']['base_minus_large'] = numeric_delta(
        result['fairvision']['arms']['base'],
        result['fairvision']['arms']['large'],
    )
    import cv2
    base_union = np.stack([
        cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST) > 0
        for mask in fairvision_predictions['base']
    ])
    large_union = np.stack([
        cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST) > 0
        for mask in fairvision_predictions['large']
    ])
    result['fairvision']['envelope_agreement'] = float(
        np.mean(base_union == large_union))
    intersection = float((base_union & large_union).sum())
    union = float((base_union | large_union).sum())
    result['fairvision']['envelope_iou'] = intersection / union

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('wrote %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
