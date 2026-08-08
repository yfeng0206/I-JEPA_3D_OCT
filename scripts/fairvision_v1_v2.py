"""Two-arm FairVision figure: V1 (raw out-of-the-box MIRAGE) vs V2 (merged retrain).

This is the visual companion to the V1 -> V2 connectivity claim.  It draws the
*raw* argmax output of both arms -- no envelope repair, no morphological
cleanup -- because the question it exists to answer is what the trained weights
emit before ``src/guides/mirage_envelope.py`` patches anything up.

Slices are the ten volumes / five depths already materialised under
``fairvision-transfer``, which come from the FairVision **Test** split (held out
from every training run).  Pixels are recomputed from the source ``.npz`` at
full float precision using the same per-slice min-max + bilinear 1024 resize
recorded in each slice's ``metadata.json``, so the inputs match the pipeline
behind the reported metrics rather than the uint8 PNG previews.

Both taxonomies collapse to the same two regions via the ARMS table:
  V1 baseline (GOALS 4-class): inner = {1,2}, choroid = {3}
  V2 merged   (3-class + void): inner = {1}, choroid = {2}, class 3 suppressed

Per-panel stats are the raw-union connectivity numbers: connected components,
largest-component share, and mean vertical runs per occupied column.  They are
computed with the same helpers fairvision_model_compare uses, so a number here
is directly comparable to a number in the metric tables.

Two stages, because inference needs the MIRAGE venv (no matplotlib) and drawing
needs the repo venv:
  stage 1  --dump masks.npz        (MIRAGE venv)
  stage 2  --plot-from masks.npz   (repo venv)
"""
from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fairvision_model_compare import (  # noqa: E402
    ARMS, BASELINE_CKPT, MIRAGE_WS, build_model, components, mean_runs,
)

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')

# Volume/slice pairs already published under fairvision-transfer.  The first is
# the slice the user inspected by hand; the rest spread across volumes and depth.
DEFAULT_PICKS = (
    'data_07266:199',
    'data_07287:100',
    'data_08300:50',
    'data_09001:149',
    'data_09345:100',
)

INNER_RGB = (0, 190, 210)
CHOR_RGB = (250, 176, 40)


def overlay(gray: np.ndarray, idx: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    g = np.clip(gray.astype(np.float32), 0.0, 1.0) * 255.0
    base = np.repeat(g[..., None], 3, axis=2)
    lab = np.zeros_like(base)
    lab[idx == 1] = INNER_RGB
    lab[idx == 2] = CHOR_RGB
    m = (idx != 0)[..., None]
    return np.clip(base * (1 - alpha * m) + lab * (alpha * m), 0, 255).astype(np.uint8)


def union_stats(coll: np.ndarray) -> tuple:
    """Raw-union connectivity, using fairvision_model_compare's definitions."""
    union = coll > 0
    n_comp, largest = components(union)
    runs = mean_runs(union)
    return n_comp, largest, (float(np.mean(runs)) if runs else 0.0)


def run_ckpt(ckpt: pathlib.Path, arm: str, inputs: list, device: str) -> list:
    inner_cls, chor_cls, n_logits, ignore_cls = ARMS[arm]
    model = build_model(n_logits, ckpt, device)
    outs = []
    for arr in inputs:
        t = torch.from_numpy(arr)[None, None].to(device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            out = model({'bscan': t})
        logits = out['semseg'] if isinstance(out, dict) else out
        logits = logits.float().clone()  # inference_mode tensors are read-only
        if ignore_cls is not None:
            logits[:, ignore_cls] = float('-inf')
        hard = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        coll = np.zeros_like(hard)
        coll[np.isin(hard, inner_cls)] = 1
        coll[np.isin(hard, chor_cls)] = 2
        outs.append(coll)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def load_inputs(picks: list) -> tuple:
    import cv2
    inputs, names = [], []
    for spec in picks:
        vol_id, _, sl = spec.partition(':')
        path = TEST / ('%s.npz' % vol_id)
        if not path.exists():
            raise SystemExit('missing volume: %s' % path)
        with np.load(path, allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = int(sl)
        if not 0 <= d < len(vol):
            raise SystemExit('slice %d out of range for %s (%d)' % (d, vol_id, len(vol)))
        raw = np.asarray(vol[d], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        inputs.append(cv2.resize(unit, (1024, 1024), interpolation=cv2.INTER_LINEAR))
        names.append('%s  slice %d' % (vol_id, d))
    return inputs, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2-ckpt')
    ap.add_argument('--v3-ckpt', help='optional third arm')
    ap.add_argument('--baseline-ckpt', default=str(BASELINE_CKPT))
    ap.add_argument('--picks', nargs='*', default=list(DEFAULT_PICKS),
                    help='volume:slice pairs, e.g. data_07266:199')
    ap.add_argument('--dump', help='stage 1: write masks npz (MIRAGE venv)')
    ap.add_argument('--plot-from', help='stage 2: draw from npz (repo venv)')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.plot_from:
        return plot(a.plot_from, a.out)
    if not (a.v2_ckpt and a.dump):
        raise SystemExit('need --v2-ckpt --dump, or --plot-from')

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs, names = load_inputs(a.picks)
    print('slices: %d' % len(inputs))

    plan = [('V1  raw MIRAGE (GOALS-only)', pathlib.Path(a.baseline_ckpt), 'baseline'),
            ('V2  merged retrain', pathlib.Path(a.v2_ckpt), 'merged')]
    if a.v3_ckpt:
        plan.append(('V3  + bounded choroid band', pathlib.Path(a.v3_ckpt), 'merged'))

    labels, stacks = [], []
    for label, ck, arm in plan:
        if not ck.exists():
            raise SystemExit('missing checkpoint: %s' % ck)
        print('running %s  <- %s' % (label, ck.name))
        labels.append(label)
        stacks.append(np.stack(run_ckpt(ck, arm, inputs, device)))

    masks = np.stack(stacks)                      # (arm, slice, H, W)
    stats = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.float32)
    for ai in range(masks.shape[0]):
        for si in range(masks.shape[1]):
            stats[ai, si] = union_stats(masks[ai, si])

    for ai, lab in enumerate(labels):
        print('%-30s n_comp %.2f  largest %.4f  runs/col %.3f' % (
            lab, stats[ai, :, 0].mean(), stats[ai, :, 1].mean(), stats[ai, :, 2].mean()))

    out = pathlib.Path(a.dump)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, inputs=np.stack(inputs).astype(np.float16),
                        names=np.array(names), labels=np.array(labels),
                        masks=masks, stats=stats)
    print('wrote %s' % out)
    return 0


def plot(npz_path: str, out_path: str) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    z = np.load(npz_path, allow_pickle=True)
    inputs, names = z['inputs'].astype(np.float32), list(z['names'])
    labels, masks, stats = list(z['labels']), z['masks'], z['stats']
    n = len(names)
    na = len(labels)
    ncol = 1 + 2 * na

    fig, axes = plt.subplots(n, ncol, figsize=(3.8 * ncol, 3.9 * n))
    if n == 1:
        axes = axes[None, :]
    heads = ['B-scan (FairVision Test)']
    for ai in range(na):
        heads += [labels[ai], '%s union (raw)' % labels[ai].split()[0]]

    for r in range(n):
        panels = [overlay(inputs[r], np.zeros_like(masks[0, r]))]
        for ai in range(na):
            panels.append(overlay(inputs[r], masks[ai, r]))
            panels.append((masks[ai, r] > 0).astype(np.uint8) * 255)
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=None if img.ndim == 3 else 'gray',
                      vmin=None if img.ndim == 3 else 0,
                      vmax=None if img.ndim == 3 else 255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(heads[c], fontsize=10)
        for ai in range(na):
            axes[r, 2 + 2 * ai].set_xlabel(
                'components %d   largest %.3f   runs/col %.2f'
                % (int(stats[ai, r, 0]), stats[ai, r, 1], stats[ai, r, 2]),
                fontsize=9)
        axes[r, 0].set_ylabel(names[r], fontsize=10)

    fig.legend(handles=[Patch(facecolor=np.array(INNER_RGB) / 255, label='InnerRetina'),
                        Patch(facecolor=np.array(CHOR_RGB) / 255, label='Choroid')],
               loc='lower center', ncol=2, frameon=False, fontsize=11)
    fig.suptitle(
        'MIRAGE arms on FairVision transfer - RAW output, no envelope repair\n'
        'lower components / runs-per-column = better connected; the mid-retina gap is a '
        'taxonomy hole and survives in EVERY arm',
        fontsize=13)
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=108)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
