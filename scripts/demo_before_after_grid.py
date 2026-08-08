#!/usr/bin/env python3
"""Before/after masks on N random HELD-OUT slices, compact grid.

BEFORE = frozen MIRAGE targets.  AFTER = adapted score, budget-locked so the
cell count comes from the frozen reference and only placement can change.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                    # noqa: E402
from adapter_sweep import Adapter, CACHE                 # noqa: E402
from jepa_to_mirage_probe import build_mirage            # noqa: E402

OUT = REPO / 'results/masking/adapter_guardrails'
GRID, ANATOMY = 16, (1, 2)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument('--n-show', type=int, default=20)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    a = ap.parse_args()

    rep = json.loads((a.out / 'guardrails.json').read_text())
    cfg = rep['config']
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')

    perm = np.random.default_rng(0).permutation(
        min(cfg['n_images'], im512.shape[0]))
    ho = np.sort(perm[:cfg['n_heldout']])
    pick = np.sort(np.random.default_rng(a.seed).choice(
        ho, size=a.n_show, replace=False))

    hd = np.load(CACHE / 'head.npz')
    head = nn.Conv2d(384, 4, 1).to(dev)
    head.weight.data = torch.tensor(hd['w']).to(dev)
    head.bias.data = torch.tensor(hd['b']).to(dev)
    mod = Adapter(depth=cfg['depth'], width=cfg['width'],
                  alpha=cfg['alpha']).to(dev)
    mod.load_state_dict(torch.load(a.out / 'adapter_cfg7.pt'))
    mod.eval()

    mir = build_mirage(dev)
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach()))

    G0, G1 = [], []
    for s in range(0, len(pick), 8):
        idx = pick[s:s + 8]
        x = torch.from_numpy(im512[idx].astype(np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
            H0 = grab['H'].float()
            H = mod(H0)
            G0.append(F.adaptive_avg_pool2d(
                head(H0).float().softmax(1)[:, ANATOMY], (GRID, GRID)).cpu().numpy())
            G1.append(F.adaptive_avg_pool2d(
                head(H).float().softmax(1)[:, ANATOMY], (GRID, GRID)).cpu().numpy())
    G0, G1 = np.concatenate(G0), np.concatenate(G1)

    C = a.cols
    R = int(np.ceil(a.n_show / C))
    fig, axes = plt.subplots(R * 2, C, figsize=(3.15 * C, 3.45 * R * 2))
    axes = np.atleast_2d(axes)
    jac, d_cells = [], []
    for k in range(a.n_show):
        r, c = divmod(k, C)
        g0, g1 = G0[k], G1[k]
        m0 = np.logical_or.reduce(A.build_targets([g0[0], g0[1]], 4)[0])
        fr = A.build_targets([g0[0], g0[1]], 4)[1]
        half = len(A.grow_components(g0[0]))
        bud = [int(sum(int(U.sum()) for U in fr[:half])),
               int(sum(int(U.sum()) for U in fr[half:]))]
        m1 = np.logical_or.reduce(
            A.build_targets_fixed_cells([g1[0], g1[1]], bud, 4)[0])
        J = (m0 & m1).sum() / max((m0 | m1).sum(), 1)
        jac.append(J)
        d_cells.append(int(m1.sum()) - int(m0.sum()))
        img = im256[pick[k]]
        for row, (m, col, tag) in enumerate((
                (m0, (.95, .15, .15, .5), 'BEFORE'),
                (m1, (.15, .55, .95, .5), 'AFTER'))):
            ax = axes[r * 2 + row, c]
            ax.imshow(img, cmap='gray')
            ov = np.zeros((256, 256, 4))
            ov[np.kron(m, np.ones((16, 16))).astype(bool)] = col
            ax.imshow(ov)
            t = ('%s  slice %d   %d cells' % (tag, pick[k], m.sum()) if row == 0
                 else '%s   %d cells   J=%.2f' % (tag, m.sum(), J))
            ax.set_title(t, fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
    for k in range(a.n_show, R * C):
        r, c = divmod(k, C)
        axes[r * 2, c].axis('off')
        axes[r * 2 + 1, c].axis('off')

    h = rep['heldout']
    fig.suptitle('BEFORE (frozen MIRAGE) vs AFTER (adapted, budget-locked)   '
                 '%d random HELD-OUT slices   |   mean Jaccard %.3f, '
                 'mean cell change %+.1f   |   $L_{rel}$ %.4f -> %.4f (%.1f%%)'
                 % (a.n_show, np.mean(jac), np.mean(d_cells),
                    h['L_rel_before'], h['L_rel_after'],
                    h['L_rel_reduction_pct']), fontsize=13, y=1.0)
    fig.tight_layout()
    f = a.out / ('before_after_%d.png' % a.n_show)
    fig.savefig(f, dpi=100, bbox_inches='tight', facecolor='white')
    print('wrote %s' % f)
    print('mean Jaccard %.4f   mean cell change %+.2f   min J %.2f  max J %.2f'
          % (np.mean(jac), np.mean(d_cells), np.min(jac), np.max(jac)))


if __name__ == '__main__':
    main()
