"""Raw, unmodified MIRAGE segmentation output for every arm - nothing applied.

The other comparison figures deliberately make the arms *comparable*: they fuse
V1's RNFL+GCIPL into one region and they force the merged arms' void logit to
-inf so it cannot win the argmax.  Both are display conveniences.  This script
removes them.  What is drawn here is exactly what each network emits:

    hard = logits.argmax(1)

No class collapse.  No void suppression.  No envelope repair, no morphological
closing, no component filtering, no gap bridging.  Each arm is shown in its own
native taxonomy, which is the point -- the taxonomies are NOT the same:

    V1 (GOALS 4-class)  0 Elsewhere  1 RNFL         2 GCIPL    3 Choroid
    V2/V3 (merged)      0 Elsewhere  1 InnerRetina  2 Choroid  3 Background/void

V1's colours reproduce the published convention in each slice's metadata.json
(RNFL amber, GCIPL magenta, choroid cyan) so this figure can be checked against
the pre-existing ``03_hard_segmentation_1024.png`` previews.  For the merged
arms the choroid keeps the same cyan so it stays visually comparable across
arms, InnerRetina gets a fusion colour, and the void class is drawn in ALARM RED
- if any red appears, the merged head is emitting the class that the metric
scripts normally suppress.

Two stages, since inference needs the MIRAGE venv (no matplotlib):
  stage 1  --dump raw.npz        (MIRAGE venv)
  stage 2  --plot-from raw.npz   (repo venv)
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
from fairvision_model_compare import BASELINE_CKPT, MIRAGE_WS, build_model  # noqa: E402
from fairvision_v1_v2 import DEFAULT_PICKS, load_inputs  # noqa: E402

# native GOALS palette, from fairvision-transfer/*/*/metadata.json
V1_PALETTE = [(0, 0, 0), (255, 196, 0), (224, 64, 224), (0, 200, 255)]
V1_NAMES = ['Elsewhere', 'RNFL', 'GCIPL', 'Choroid']

# merged palette: choroid deliberately the SAME cyan as V1 so the two arms can
# be compared by eye; void is red so it is impossible to miss.
MERGED_PALETTE = [(0, 0, 0), (255, 120, 180), (0, 200, 255), (255, 0, 0)]
MERGED_NAMES = ['Elsewhere', 'InnerRetina', 'Choroid', 'Background/void']


def colour(gray: np.ndarray, hard: np.ndarray, palette, alpha: float = 0.45) -> np.ndarray:
    g = np.clip(gray.astype(np.float32), 0.0, 1.0) * 255.0
    base = np.repeat(g[..., None], 3, axis=2)
    lab = np.zeros_like(base)
    for ci, rgb in enumerate(palette):
        if ci == 0:
            continue
        lab[hard == ci] = rgb
    m = (hard != 0)[..., None]
    return np.clip(base * (1 - alpha * m) + lab * (alpha * m), 0, 255).astype(np.uint8)


def run_raw(ckpt: pathlib.Path, inputs: list, device: str) -> np.ndarray:
    """Pure argmax over all 4 logits.  Nothing suppressed, nothing remapped."""
    model = build_model(4, ckpt, device)
    outs = []
    for arr in inputs:
        t = torch.from_numpy(arr)[None, None].to(device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            out = model({'bscan': t})
        logits = out['semseg'] if isinstance(out, dict) else out
        outs.append(logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2-ckpt')
    ap.add_argument('--v3-ckpt')
    ap.add_argument('--baseline-ckpt', default=str(BASELINE_CKPT))
    ap.add_argument('--picks', nargs='*', default=list(DEFAULT_PICKS))
    ap.add_argument('--dump')
    ap.add_argument('--plot-from')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.plot_from:
        return plot(a.plot_from, a.out)
    if not (a.v2_ckpt and a.dump):
        raise SystemExit('need --v2-ckpt [--v3-ckpt] --dump, or --plot-from')

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs, names = load_inputs(a.picks)
    print('slices: %d   RAW argmax, no suppression, no repair' % len(inputs))

    plan = [('V1  raw MIRAGE (GOALS 4-class)', pathlib.Path(a.baseline_ckpt), 'v1'),
            ('V2  merged retrain', pathlib.Path(a.v2_ckpt), 'merged')]
    if a.v3_ckpt:
        plan.append(('V3  + bounded choroid band', pathlib.Path(a.v3_ckpt), 'merged'))

    labels, kinds, stacks = [], [], []
    for label, ck, kind in plan:
        if not ck.exists():
            raise SystemExit('missing checkpoint: %s' % ck)
        print('running %s  <- %s' % (label, ck.name))
        labels.append(label)
        kinds.append(kind)
        stacks.append(run_raw(ck, inputs, device))

    masks = np.stack(stacks)
    for ai, lab in enumerate(labels):
        nm = V1_NAMES if kinds[ai] == 'v1' else MERGED_NAMES
        frac = [float((masks[ai] == c).mean()) for c in range(4)]
        print('  %-32s %s' % (lab, '  '.join(
            '%s %.4f' % (nm[c][:11], frac[c]) for c in range(4))))

    out = pathlib.Path(a.dump)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, inputs=np.stack(inputs).astype(np.float16),
                        names=np.array(names), labels=np.array(labels),
                        kinds=np.array(kinds), masks=masks)
    print('wrote %s' % out)
    return 0


def plot(npz_path: str, out_path: str) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    z = np.load(npz_path, allow_pickle=True)
    inputs, names = z['inputs'].astype(np.float32), list(z['names'])
    labels, kinds, masks = list(z['labels']), list(z['kinds']), z['masks']
    n, na = len(names), len(labels)

    fig, axes = plt.subplots(n, 1 + na, figsize=(4.0 * (1 + na), 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    for r in range(n):
        ax = axes[r, 0]
        ax.imshow(np.clip(inputs[r], 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(names[r], fontsize=10)
        if r == 0:
            ax.set_title('B-scan (FairVision Test)', fontsize=10)
        for ai in range(na):
            pal = V1_PALETTE if kinds[ai] == 'v1' else MERGED_PALETTE
            nm = V1_NAMES if kinds[ai] == 'v1' else MERGED_NAMES
            ax = axes[r, 1 + ai]
            ax.imshow(colour(inputs[r], masks[ai, r], pal))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(labels[ai], fontsize=10)
            f = [float((masks[ai, r] == c).mean()) for c in range(4)]
            ax.set_xlabel('  '.join('%s %.3f' % (nm[c][:5], f[c])
                                    for c in (1, 2, 3)), fontsize=9)

    h1 = [Patch(facecolor=np.array(c) / 255, label='V1: %s' % l)
          for c, l in zip(V1_PALETTE[1:], V1_NAMES[1:])]
    h2 = [Patch(facecolor=np.array(c) / 255, label='V2/V3: %s' % l)
          for c, l in zip(MERGED_PALETTE[1:], MERGED_NAMES[1:])]
    fig.legend(handles=h1 + h2, loc='lower center', ncol=6, frameon=False, fontsize=10)
    fig.suptitle(
        'RAW MIRAGE output - pure argmax, NOTHING applied\n'
        'no class fusion, no void suppression, no envelope repair, no morphology, '
        'no component filtering\n'
        'each arm in its OWN taxonomy: V1 keeps RNFL and GCIPL separate, V2/V3 fuse '
        'them into InnerRetina',
        fontsize=12)
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=108)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
