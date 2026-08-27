#!/usr/bin/env python3
"""Visualise what L_rel's gradient does to the MIRAGE masking area.

Reads the adapter trained by adapter_guardrails.py and renders, on HELD-OUT
FairVision images:

  row 0   loss curve, and the three guardrail summaries
  row 1   B-scan  |  frozen anatomy S0  |  adapted S  |  change  |  margin
  row 2   masks BEFORE (frozen)  vs  AFTER free  vs  AFTER budget-locked
  row 3   where the change lands relative to MIRAGE confidence
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
from adapter_sweep import Adapter, gram, CACHE           # noqa: E402
from jepa_to_mirage_probe import build_mirage            # noqa: E402

OUT = REPO / 'results/masking/adapter_guardrails'
GRID, ANATOMY = 16, (1, 2)
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.15, 0.70, 0.30), (0.95, 0.65, 0.10)]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument('--n-show', type=int, default=5)
    ap.add_argument('--n-stat', type=int, default=128)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    a = ap.parse_args()

    rep = json.loads((a.out / 'guardrails.json').read_text())
    cfg = rep['config']
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')

    rng = np.random.default_rng(0)
    perm = rng.permutation(min(cfg['n_images'], im512.shape[0]))
    ho = np.sort(perm[:cfg['n_heldout']])

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

    def grids(idx):
        x = torch.from_numpy(im512[idx].astype(np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
            H0 = grab['H'].float()
            H = mod(H0)
            L0, Lf = head(H0), head(H)
            P0 = L0.float().softmax(1)
            g0 = F.adaptive_avg_pool2d(P0[:, ANATOMY], (GRID, GRID)).cpu().numpy()
            g1 = F.adaptive_avg_pool2d(Lf.float().softmax(1)[:, ANATOMY],
                                       (GRID, GRID)).cpu().numpy()
            t2 = P0.topk(2, dim=1).values
            marg = F.adaptive_avg_pool2d((t2[:, 0] - t2[:, 1])[:, None],
                                         (GRID, GRID))[:, 0].cpu().numpy()
        return g0, g1, marg

    # ---- statistics over held-out images ---------------------------------
    dv, mv, cf, cl, jf, jl = [], [], [], [], [], []
    B = 16
    stat_idx = ho[:a.n_stat]
    for s in range(0, len(stat_idx), B):
        idx = stat_idx[s:s + B]
        g0, g1, marg = grids(idx)
        dv.append(np.abs(g1.sum(1) - g0.sum(1)).ravel())
        mv.append(marg.ravel())
        for j in range(g0.shape[0]):
            m0 = np.logical_or.reduce(A.build_targets([g0[j][0], g0[j][1]], 4)[0])
            m1 = np.logical_or.reduce(A.build_targets([g1[j][0], g1[j][1]], 4)[0])
            bud = [int(sum(int(U.sum()) for U in A.grow_components(g0[j][c])))
                   for c in (0, 1)]
            m2 = np.logical_or.reduce(
                A.build_targets_fixed_cells([g1[j][0], g1[j][1]], bud, 4)[0])
            cf.append(m0.sum()); cl.append(m1.sum())
            jf.append((m0 & m1).sum() / max((m0 | m1).sum(), 1))
            jl.append((m0 & m2).sum() / max((m0 | m2).sum(), 1))
    dv = np.concatenate(dv); mv = np.concatenate(mv)

    show = ho[:a.n_show]
    g0s, g1s, margs = grids(show)

    # ------------------------------------------------------------------ plot
    n = a.n_show
    fig = plt.figure(figsize=(4.1 * n + 4, 16.5))
    gs = fig.add_gridspec(4, n + 1, hspace=.30, wspace=.16,
                          left=.04, right=.985, top=.915, bottom=.035)
    K = dict(fontsize=10, pad=6)

    # --- row 0 -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    L = np.array(rep['loss_curve'])
    ax.plot(L, lw=.7, alpha=.3, color='#2a9d8f')
    w = max(1, len(L) // 60)
    ax.plot(np.arange(len(L) - w + 1), np.convolve(L, np.ones(w) / w, 'valid'),
            lw=2.4, color='#264653')
    t, h = rep['train'], rep['heldout']
    ax.axhline(h['L_rel_before'], ls='--', c='#e76f51', lw=1.4)
    ax.axhline(h['L_rel_after'], ls='--', c='#2a9d8f', lw=1.4)
    ax.text(len(L) * .55, h['L_rel_before'], ' frozen MIRAGE  %.4f'
            % h['L_rel_before'], fontsize=8.5, va='bottom', color='#e76f51')
    ax.text(len(L) * .55, h['L_rel_after'], ' after adapter  %.4f'
            % h['L_rel_after'], fontsize=8.5, va='top', color='#2a9d8f')
    ax.set_xlabel('step  (%d train images, batch %d)' % (
        cfg['n_images'] - cfg['n_heldout'], cfg['batch']))
    ax.set_ylabel(r'$L_{rel}$')
    ax.set_title(r'$L_{rel}$ during training   (dashed = HELD-OUT)', **K)
    ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[0, 2])
    b = ax.bar(['train', 'HELD-OUT'],
               [t['L_rel_reduction_pct'], h['L_rel_reduction_pct']],
               color=['#8d99ae', '#2a9d8f'], width=.55)
    for r_, v in zip(b, [t['L_rel_reduction_pct'], h['L_rel_reduction_pct']]):
        ax.text(r_.get_x() + r_.get_width()/2, v, '%.1f%%' % v,
                ha='center', va='bottom', fontsize=11)
    ax.set_title('T1  generalisation\nno train/held-out gap', **K)
    ax.set_ylabel(r'$L_{rel}$ reduced (%)'); ax.grid(axis='y', alpha=.25)

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(3)
    vals = [h['cells_frozen'], h['cells_free'], h['cells_locked']]
    bb = ax.bar(x, vals, color=['#8d99ae', '#e76f51', '#2a9d8f'], width=.6)
    for r_, v in zip(bb, vals):
        ax.text(r_.get_x() + r_.get_width()/2, v, '%.1f' % v,
                ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(['frozen', 'free', 'LOCKED'])
    ax.set_ylabel('mask cells')
    ax.set_title('T2  budget lock\nJaccard %.3f free / %.3f locked'
                 % (h['jaccard_free'], h['jaccard_locked']), **K)
    ax.grid(axis='y', alpha=.25)

    ax = fig.add_subplot(gs[0, 4:])
    sub = rng.choice(len(dv), size=min(6000, len(dv)), replace=False)
    ax.scatter(mv[sub], dv[sub], s=3, alpha=.16, color='#264653')
    ax.set_xlabel('frozen MIRAGE confidence margin  $p_{top1}-p_{top2}$')
    ax.set_ylabel(r'$|\Delta S|$  adapter change')
    ax.set_title('T3  the adapter edits UNCERTAIN cells\n'
                 r'corr $%+.3f$   unsure/sure ratio %.1fx'
                 % (h['corr_change_vs_margin'],
                    h['change_unsure_margin_lt0.5']
                    / max(h['change_sure_margin_ge0.9'], 1e-9)), **K)
    ax.grid(alpha=.25)

    # --- rows 1-3 : per image ---------------------------------------------
    def up(m):
        return np.kron(m.astype(float), np.ones((16, 16)))

    for j in range(n):
        idx = show[j]
        a0, a1 = g0s[j].sum(0), g1s[j].sum(0)
        d = a1 - a0
        m0 = np.logical_or.reduce(A.build_targets([g0s[j][0], g0s[j][1]], 4)[0])
        p1 = A.build_targets([g1s[j][0], g1s[j][1]], 4)[0]
        m1 = np.logical_or.reduce(p1)
        bud = [int(sum(int(U.sum()) for U in A.grow_components(g0s[j][c])))
               for c in (0, 1)]
        p2 = A.build_targets_fixed_cells([g1s[j][0], g1s[j][1]], bud, 4)[0]
        m2 = np.logical_or.reduce(p2)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(im256[idx], cmap='gray')
        ov = np.zeros((256, 256, 4))
        ov[up(m0).astype(bool)] = (.95, .15, .15, .5)
        ax.imshow(ov)
        ax.set_title('BEFORE  frozen MIRAGE\n%d cells' % m0.sum(), **K)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('masks', fontsize=11)

        ax = fig.add_subplot(gs[2, j])
        ax.imshow(im256[idx], cmap='gray')
        ov = np.zeros((256, 256, 4))
        ov[up(m2).astype(bool)] = (.15, .55, .95, .5)
        ax.imshow(ov)
        ax.set_title('AFTER  budget-locked\n%d cells   J=%.2f'
                     % (m2.sum(), (m0 & m2).sum() / max((m0 | m2).sum(), 1)), **K)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[3, j])
        v = max(abs(d).max(), 1e-8)
        im = ax.imshow(d, cmap='bwr', vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title(r'$\Delta S$ from $L_{rel}$ backprop' + '\nmax %.3f' % v, **K)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=.046)

    for r_, (g, t_) in enumerate(((g0s[0].sum(0), 'frozen  $S_0$'),
                                  (g1s[0].sum(0), 'adapted  $S$'))):
        ax = fig.add_subplot(gs[1 + r_, n])
        ax.imshow(g, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(t_, **K); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[3, n])
    ax.imshow(margs[0], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('frozen confidence margin\n(dark = MIRAGE unsure)', **K)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('What $L_{rel}$ backprop does to the MIRAGE masking area   '
                 '(depth %d, width %d, peak lr %.0e, $\\alpha$ %.2f;  '
                 'MIRAGE fully frozen, adapter only;  HELD-OUT images)'
                 % (cfg['depth'], cfg['width'], cfg['lr'], cfg['alpha']),
                 fontsize=14, y=.962)
    f = a.out / 'backprop_effect.png'
    fig.savefig(f, dpi=108, facecolor='white')
    print('wrote %s' % f)
    print('held-out: L_rel %.5f -> %.5f (%.1f%%)   cells %.1f -> %.1f free / %.1f locked'
          % (h['L_rel_before'], h['L_rel_after'], h['L_rel_reduction_pct'],
             h['cells_frozen'], h['cells_free'], h['cells_locked']))


if __name__ == '__main__':
    main()
