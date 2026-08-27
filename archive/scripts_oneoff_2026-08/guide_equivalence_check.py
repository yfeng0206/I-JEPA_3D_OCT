"""Is the V1 soft guide actually a DIFFERENT experiment from the shipped MIRAGE arm?

If the pooled 16x16 location scores produced by V3-merged soft occupancy are
near-identical to the shipped repaired-envelope guide, then running "V1 with the
adapter frozen at zero" reproduces the existing MIRAGE arm and is not a new
experiment.  Worth knowing before committing GPU weeks.

Two things differ between the arms:
  1. checkpoint   -- shipped = original GOALS-tuned MIRAGE; V1 = V3-merged
  2. post-process -- shipped = repair_union hard envelope; V1 = raw soft occupancy

The shipped guides are bit-packed binary masks in the native 200x200 label space
(5000 bytes = 40000 bits), which is exactly the space repair_union runs in.

Stage 1 (MIRAGE venv): --dump    Stage 2 (repo venv): --plot-from
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

GUIDES = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training')
IMAGES = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
CKPT = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
        r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
NATIVE, RES, GRID, ANATOMY = 200, 1024, 16, (1, 2)


def pool_to_grid(a, grid=GRID):
    """Mean-pool to (grid,grid); trims the remainder like patch_occupancy does."""
    h, w = a.shape
    ch, cw = h // grid, w // grid
    return a[:grid * ch, :grid * cw].reshape(grid, ch, grid, cw).mean(axis=(1, 3))


def dump(out_path, n_vols, per_vol):
    import os
    import cv2
    from compare_512_vs_1024 import build
    from fairvision_model_compare import MIRAGE_WS

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(RES, CKPT, device)
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(L0=o.detach()))

    vols = sorted(p.stem for p in GUIDES.glob('data_*.npz'))[:n_vols]
    ship, soft, names = [], [], []
    for vol_id in vols:
        g = np.load(GUIDES / ('%s.npz' % vol_id), allow_pickle=True)
        packed, sl_idx, valid = g['packed_envelopes'], g['slice_indices'], g['valid']
        with np.load(IMAGES / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        pick = np.linspace(0, len(sl_idx) - 1, per_vol).astype(int)
        for pi in pick:
            if not valid[pi]:
                continue
            env = np.unpackbits(packed[pi])[:NATIVE * NATIVE].reshape(NATIVE, NATIVE)
            raw = np.asarray(vol[int(sl_idx[pi])], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            img = cv2.resize(unit, (RES, RES), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(img)[None, None].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                model({'bscan': x})
            M = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)
            s = F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0].cpu().numpy()
            ship.append(pool_to_grid(env.astype(np.float32)))
            soft.append(s)
            names.append('%s:%d' % (vol_id, int(sl_idx[pi])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, shipped=np.stack(ship), soft=np.stack(soft),
                        names=np.array(names))
    print('wrote %s   n=%d' % (out_path, len(names)))


def analyse(npz_path, out_png):
    z = np.load(npz_path, allow_pickle=True)
    sh, so = z['shipped'], z['soft']
    n = len(sh)
    rep = {
        'n_slices': n,
        'shipped_area_mean': float(sh.mean()), 'soft_area_mean': float(so.mean()),
        'corr_global': float(np.corrcoef(sh.ravel(), so.ravel())[0, 1]),
        'corr_per_slice_mean': float(np.mean(
            [np.corrcoef(sh[i].ravel(), so[i].ravel())[0, 1] for i in range(n)])),
        'mean_abs_diff': float(np.abs(sh - so).mean()),
    }
    # agreement of the admissible-cell SETS, which is what the sampler consumes
    for th_s, th_n in ((0.25, 0.10), (0.25, 0.25), (0.5, 0.10)):
        A, B = sh >= th_s, so >= th_n
        inter = (A & B).sum(axis=(1, 2))
        union = (A | B).sum(axis=(1, 2)).clip(min=1)
        rep[f'iou_shipped{th_s}_soft{th_n}'] = float((inter / union).mean())
        rep[f'cells_shipped{th_s}'] = float(A.sum(axis=(1, 2)).mean())
        rep[f'cells_soft{th_n}'] = float(B.sum(axis=(1, 2)).mean())
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_png).parent / 'guide_equivalence.json').write_text(
        json.dumps(rep, indent=2))

    print('n=%d' % n)
    print('  mean area   shipped %.4f   soft %.4f' %
          (rep['shipped_area_mean'], rep['soft_area_mean']))
    print('  corr        global %.4f   per-slice %.4f' %
          (rep['corr_global'], rep['corr_per_slice_mean']))
    print('  mean |diff| %.4f' % rep['mean_abs_diff'])
    for k in sorted(k for k in rep if k.startswith('iou_')):
        print('  %-28s %.4f' % (k, rep[k]))
    for k in sorted(k for k in rep if k.startswith('cells_')):
        print('  %-28s %.1f / 256' % (k, rep[k]))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
    ax[0].scatter(sh.ravel(), so.ravel(), s=3, alpha=0.15)
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[0].set_xlabel('shipped repaired envelope'); ax[0].set_ylabel('V1 soft occupancy')
    ax[0].set_title('per-cell agreement\nr = %.4f' % rep['corr_global'])
    im = ax[1].imshow(sh[0], cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('shipped guide  %s' % str(z['names'][0]))
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(so[0], cmap='viridis', vmin=0, vmax=1)
    ax[2].set_title('V1 soft guide (V3-merged @1024)')
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    im = ax[3].imshow(so[0] - sh[0], cmap='coolwarm', vmin=-1, vmax=1)
    ax[3].set_title('soft − shipped')
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    for a_ in ax[1:]:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('Is V1-with-frozen-adapter a new arm, or a re-run of the shipped MIRAGE arm?',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--out')
    ap.add_argument('--n-vols', type=int, default=12)
    ap.add_argument('--per-vol', type=int, default=4)
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_vols, a.per_vol)
    if a.plot_from:
        return analyse(a.plot_from, a.out)
    raise SystemExit('need --dump or --plot-from')


if __name__ == '__main__':
    main()
