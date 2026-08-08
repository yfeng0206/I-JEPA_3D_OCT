"""Visualise the four anatomy-budgeted connected targets on real OCT slices.

Draws, per slice, what the mask constructor actually produces:

    T1 red   T2 blue   T3 green   T4 orange     overlap shown as a blend
    union    = what the context encoder never sees
    context  = what it does see

The count is FOUR targets per image, always -- one seeded per horizontal band --
matching I-JEPA's default (its ablation reports 1/2/3/4 targets -> 9.0/22.0/
48.5/54.2 on 1% ImageNet, so the count is not a free parameter).  What varies
per image is the SIZE, because the budget is relative to how much anatomy the
slice contains:

    grow the union until   sum_{i in U} a_i  >=  rho * sum_i a_i

Stage 1 (MIRAGE venv, GPU): --dump    Stage 2 (repo venv): --plot-from
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
CK_BASE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-base-512\MergedV3'
           r'\MIRAGE-Base_frozen_convnext_CEGDice-ignore\checkpoint-best.pth')
MIRAGE_RES, JEPA_RES, PATCH, GRID = 512, 256, 16, 16
ANATOMY = (1, 2)
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.20, 0.75, 0.35), (0.98, 0.65, 0.10)]


def dump(out_path, picks, seed):
    import os
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from base512_vs_large1024_guide import build
    from fairvision_model_compare import MIRAGE_WS

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, interp = build('base', MIRAGE_RES, CK_BASE, device)
    assert not interp
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(L0=o.detach()))

    imgs, grids, names = [], [], []
    for k, spec in enumerate(picks):
        vol_id, _, sl = spec.partition(':')
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        raw = np.asarray(vol[int(sl)], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        # ONE crop, applied at both resolutions -- the two branches must see the
        # same field of view or the guide points at the wrong anatomy
        torch.manual_seed(seed + k)
        pil = Image.fromarray((unit * 255).astype(np.uint8))
        i, j, h, w = T.RandomResizedCrop.get_params(
            pil, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.))
        bic = T.InterpolationMode.BICUBIC
        m_img = np.asarray(TF.resized_crop(pil, i, j, h, w,
                                           [MIRAGE_RES, MIRAGE_RES], bic),
                           dtype=np.float32) / 255.
        j_img = np.asarray(TF.resized_crop(pil, i, j, h, w,
                                           [JEPA_RES, JEPA_RES], bic),
                           dtype=np.float32) / 255.
        x = torch.from_numpy(m_img)[None, None].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            model({'bscan': x})
        M = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)
        grids.append(F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0].cpu().numpy())
        imgs.append(j_img)
        names.append(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, imgs=np.stack(imgs), grids=np.stack(grids),
                        names=np.array(names))
    print('wrote %s  %d slices' % (out_path, len(imgs)))


def plot(npz_path, out_png, rho):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from anatomy_target_sampler import build_targets, topology_of

    z = np.load(npz_path, allow_pickle=True)
    imgs, grids, names = z['imgs'], z['grids'], [str(s) for s in z['names']]
    n = len(imgs)
    fig, ax = plt.subplots(n, 4, figsize=(17, 4.05 * n))
    if n == 1:
        ax = ax[None]
    stats = []

    for r in range(n):
        a, img = grids[r], imgs[r]
        A = float(a.sum())
        parts, U = build_targets(a, rho=rho, overlap=0.24)
        union = np.zeros_like(U)
        for p in parts:
            union |= p
        t = topology_of(union)
        sizes = [int(p.sum()) for p in parts]
        st = {'slice': names[r], 'anatomy_mass': A,
              'part_cells': sizes, 'target_spread': max(sizes) - min(sizes),
              'union_cells': int(union.sum()),
              'union_components': t['components'],
              'retina_visible': float(1 - (a * union).sum() / A),
              'purity': float((a * union).sum() / max(union.sum(), 1)),
              'parts_connected': [topology_of(p)['components'] for p in parts],
              'union_holes': t['holes'],
              'context_cells': int(256 - union.sum())}
        stats.append(st)

        ax[r, 0].imshow(img, cmap='gray')
        ax[r, 0].set_ylabel(names[r], fontsize=9)
        ax[r, 0].set_title('OCT crop 256x256' if r == 0 else '', fontsize=11)

        im = ax[r, 1].imshow(a, cmap='viridis', vmin=0, vmax=1)
        ax[r, 1].set_title('MIRAGE anatomy $a_i$ (16x16)\n'
                           'mass %.1f cells' % A if r == 0
                           else 'mass %.1f cells' % A, fontsize=10)

        # targets, each its own colour, drawn on the grid
        canvas = np.ones((GRID, GRID, 3)) * 0.12
        for k, p in enumerate(parts):
            canvas[p] = 0.5 * canvas[p] + 0.5 * np.array(COLORS[k])
        ax[r, 2].imshow(canvas)
        ax[r, 2].set_title('4 targets: %s cells (spread %d)\nunion %d, %d comp, visible %.1f%%'
                           % ('/'.join(str(c) for c in st['part_cells']),
                              st['target_spread'], st['union_cells'],
                              st['union_components'], 100 * st['retina_visible']),
                           fontsize=10)

        # targets over the B-scan at pixel scale
        ov = np.repeat(img[..., None], 3, axis=2).copy()
        for k, p in enumerate(parts):
            up = np.kron(p, np.ones((PATCH, PATCH), dtype=bool))
            ov[up] = 0.55 * ov[up] + 0.45 * np.array(COLORS[k])
        ax[r, 3].imshow(np.clip(ov, 0, 1))
        ax[r, 3].set_title('targets on the B-scan\ncontext keeps %d of 256 tokens'
                           % st['context_cells'], fontsize=10)

        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    handles = [Patch(facecolor=COLORS[k], label='target %d' % (k + 1)) for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11)
    fig.suptitle(r'Anatomy-budgeted connected masking  ($\rho$=%.2f): '
                 '4 targets per image, size adapts to how much retina is present'
                 % rho, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)

    pathlib.Path(out_png).with_suffix('.json').write_text(json.dumps(stats, indent=2))
    print('rho=%.2f   4 targets per image, always\n' % rho)
    print('%-18s %7s %-18s %7s %7s %8s %8s %9s' %
          ('slice', 'anat', 'target cells', 'spread', 'union', 'visible', 'purity', 'context'))
    for s in stats:
        print('%-18s %7.1f %-18s %7d %7d %7.1f%% %8.3f %9d' %
              (s['slice'], s['anatomy_mass'],
               '/'.join(str(c) for c in s['part_cells']), s['target_spread'],
               s['union_cells'], 100 * s['retina_visible'], s['purity'],
               s['context_cells']))
    allc = [c for s in stats for c in s['parts_connected']]
    print('\nevery target connected: %s   union components: %s   union holes: %s'
          % (all(c == 1 for c in allc), [s['union_components'] for s in stats],
             [s['union_holes'] for s in stats]))
    print('wrote %s' % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--out')
    ap.add_argument('--rho', type=float, default=0.70)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--picks', nargs='*',
                    default=['data_07050:4', 'data_07225:24',
                             'data_08508:64', 'data_09186:144'])
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.picks, a.seed)
    if a.plot_from:
        return plot(a.plot_from, a.out, a.rho)
    raise SystemExit('need --dump or --plot-from')


if __name__ == '__main__':
    main()
