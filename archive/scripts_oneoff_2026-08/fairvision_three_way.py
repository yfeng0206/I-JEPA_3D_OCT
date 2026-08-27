"""Three-arm FairVision figure: baseline vs merged-V2 vs merged-V3.

FairVision has no layer ground truth, so unlike the GOALS figure nothing
here can be scored right or wrong.  What it CAN show is the finding that
motivated V3: V2 lost the choroid on transfer (evidence dilution -- only
55 of 1248 merged training images carried any choroid label), and V3's
bounded synthetic band recovered it.

All three arms see byte-identical inputs -- same volumes, same slice
indices, same min-max + resize -- reproduced from fairvision_model_compare
so the pixels drawn here match the pixels behind the reported metrics.
Checkpoints are loaded and freed one at a time so peak VRAM stays at a
single model, since a training run may be holding the GPU.

Both taxonomies collapse to the same two regions before drawing, using the
ARMS table as the single source of truth:
  baseline (GOALS 4-class): inner = {1,2}, choroid = {3}
  merged   (3-class)      : inner = {1},   choroid = {2}, void 3 suppressed
Without that collapse the arms are not visually comparable.
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
    ARMS, BASELINE_CKPT, DATA, MIRAGE_WS, build_model,
)

# matplotlib is deliberately NOT imported at module level.  Inference must
# run in the MIRAGE venv, which has no matplotlib, and installing it there
# risks pip resolving a different numpy/torch underneath a live training
# run.  So the script has two stages: --dump computes masks in the MIRAGE
# venv and writes an npz, --plot-from draws in the repo venv.

INNER_RGB = (0, 190, 210)
CHOR_RGB = (250, 176, 40)


def overlay(gray: np.ndarray, idx: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    g = gray.astype(np.float32) * 255.0
    base = np.repeat(g[..., None], 3, axis=2)
    lab = np.zeros_like(base)
    lab[idx == 1] = INNER_RGB
    lab[idx == 2] = CHOR_RGB
    m = (idx != 0)[..., None]
    return np.clip(base * (1 - alpha * m) + lab * (alpha * m), 0, 255).astype(np.uint8)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2-ckpt')
    ap.add_argument('--v3-ckpt')
    ap.add_argument('--baseline-ckpt', default=str(BASELINE_CKPT))
    ap.add_argument('--volumes', type=int, default=4)
    ap.add_argument('--slices', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dump', help='stage 1: write masks npz (MIRAGE venv)')
    ap.add_argument('--plot-from', help='stage 2: draw from npz (repo venv)')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.plot_from:
        return plot(a.plot_from, a.out)
    if not (a.v2_ckpt and a.v3_ckpt and a.dump):
        raise SystemExit('need --v2-ckpt --v3-ckpt --dump, or --plot-from')

    os.chdir(MIRAGE_WS)
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = sorted(DATA.glob('*.npz'))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(files), size=min(a.volumes, len(files)), replace=False)
    depths = np.linspace(20, 180, num=a.slices).astype(int)

    inputs, names = [], []
    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            inputs.append(cv2.resize(unit, (1024, 1024),
                                     interpolation=cv2.INTER_LINEAR))
            names.append('%s slice %d' % (files[int(vi)].stem, int(d)))
    print('slices: %d' % len(inputs))

    labels, stacks = [], []
    for label, ck, arm in [
            ('Baseline (GOALS-only)', pathlib.Path(a.baseline_ckpt), 'baseline'),
            ('Merged V2 (real labels only)', pathlib.Path(a.v2_ckpt), 'merged'),
            ('Merged V3 (+ bounded band)', pathlib.Path(a.v3_ckpt), 'merged')]:
        if not ck.exists():
            raise SystemExit('missing checkpoint: %s' % ck)
        print('running %s' % label)
        labels.append(label)
        stacks.append(np.stack(run_ckpt(ck, arm, inputs, device)))

    out = pathlib.Path(a.dump)
    out.parent.mkdir(parents=True, exist_ok=True)
    # float16 for the B-scans: they are already min-max scaled to [0,1] and
    # only used for display, so full precision would triple the file size.
    np.savez_compressed(out, inputs=np.stack(inputs).astype(np.float16),
                        names=np.array(names), labels=np.array(labels),
                        masks=np.stack(stacks))
    print('wrote %s' % out)
    return 0


def plot(npz_path: str, out_path: str) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    z = np.load(npz_path, allow_pickle=True)
    inputs, names = z['inputs'].astype(np.float32), list(z['names'])
    labels, masks = list(z['labels']), z['masks']

    rows = len(inputs)
    fig, axes = plt.subplots(rows, 4, figsize=(17.5, 4.5 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r in range(rows):
        g = inputs[r]
        axes[r, 0].imshow(g, cmap='gray')
        if r == 0:
            axes[r, 0].set_title('B-scan', fontsize=13, fontweight='bold')
        for c, label in enumerate(labels):
            m = masks[c][r]
            axes[r, c + 1].imshow(overlay(g, m))
            if r == 0:
                axes[r, c + 1].set_title(label, fontsize=13, fontweight='bold')
            axes[r, c + 1].set_xlabel('choroid area %.3f' % float((m == 2).mean()),
                                      fontsize=11)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(names[r], fontsize=8)

    handles = [Patch(color=np.array(INNER_RGB) / 255, label='InnerRetina (RNFL+GCIPL)'),
               Patch(color=np.array(CHOR_RGB) / 255, label='Choroid')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('FairVision transfer - no ground truth, so judge continuity '
                 'and anatomical plausibility', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.028, 1, 0.972])
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
