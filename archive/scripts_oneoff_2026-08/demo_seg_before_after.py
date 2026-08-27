#!/usr/bin/env python3
"""Raw MIRAGE segmentation BEFORE vs AFTER the adapter, plus the difference.

This looks at MIRAGE's own output, not the 16x16 JEPA masks: the 64x64 decoder
argmax and the continuous anatomy score S = P_InnerRetina + P_Choroid.

  col 1  B-scan
  col 2  segmentation BEFORE   (frozen MIRAGE argmax)
  col 3  segmentation AFTER    (adapted argmax)
  col 4  class CHANGED         (which pixels moved, and to what)
  col 5  anatomy score BEFORE
  col 6  anatomy score AFTER
  col 7  delta S               (red = adapter raised, blue = lowered)

IMPORTANT CAVEAT.  FairVision has no anatomy ground truth, so "changed" is NOT
"improved".  Whether these edits are correct can only be decided against a
labelled set such as GOALS.  This figure shows WHAT the adapter did, not that
it did the right thing.
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

from adapter_sweep import Adapter, CACHE                 # noqa: E402
from jepa_to_mirage_probe import build_mirage            # noqa: E402

OUT = REPO / 'results/masking/adapter_guardrails'
ANATOMY = (1, 2)
CLASS_NAMES = ('Elsewhere', 'InnerRetina', 'Choroid', 'void')
# Elsewhere=black, InnerRetina=cyan, Choroid=orange, void=grey
CLASS_RGB = np.array([[0.05, 0.05, 0.08],
                      [0.15, 0.75, 0.85],
                      [0.95, 0.55, 0.15],
                      [0.45, 0.45, 0.45]])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ap = argparse.ArgumentParser()
    ap.add_argument('--n-show', type=int, default=6)
    ap.add_argument('--n-stat', type=int, default=192)
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

    def forward(idx):
        x = torch.from_numpy(im512[idx].astype(np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
            H0 = grab['H'].float()
            H = mod(H0)
            L0, Lf = head(H0), head(H)
            P0, Pf = L0.softmax(1), Lf.softmax(1)
        return (L0.argmax(1).cpu().numpy(), Lf.argmax(1).cpu().numpy(),
                P0[:, ANATOMY].sum(1).cpu().numpy(),
                Pf[:, ANATOMY].sum(1).cpu().numpy(),
                (P0.topk(2, 1).values[:, 0] - P0.topk(2, 1).values[:, 1]).cpu().numpy())

    # ---- statistics over held-out images ---------------------------------
    n_px = 0
    changed = 0
    trans = np.zeros((4, 4), np.int64)
    d_area = {1: [], 2: []}
    marg_ch, marg_un = [], []
    B = 16
    stat = ho[:a.n_stat]
    for s in range(0, len(stat), B):
        A0, A1, S0, S1, M0 = forward(stat[s:s + B])
        n_px += A0.size
        ch = A0 != A1
        changed += int(ch.sum())
        for i in range(4):
            for j in range(4):
                trans[i, j] += int(((A0 == i) & (A1 == j)).sum())
        for c in (1, 2):
            d_area[c].append(((A1 == c).sum(axis=(1, 2))
                              - (A0 == c).sum(axis=(1, 2)))
                             / max(A0[0].size, 1))
        marg_ch.append(M0[ch])
        marg_un.append(M0[~ch][::37])
    marg_ch = np.concatenate(marg_ch)
    marg_un = np.concatenate(marg_un)

    stats = {
        'n_images': int(len(stat)),
        'pct_pixels_changed': 100 * changed / n_px,
        'mean_margin_where_changed': float(marg_ch.mean()),
        'mean_margin_where_unchanged': float(marg_un.mean()),
        'inner_area_change_pp': float(np.concatenate(d_area[1]).mean() * 100),
        'choroid_area_change_pp': float(np.concatenate(d_area[2]).mean() * 100),
        'transitions': {'%s->%s' % (CLASS_NAMES[i], CLASS_NAMES[j]):
                        int(trans[i, j]) for i in range(4) for j in range(4)
                        if i != j and trans[i, j] > 0},
    }
    (a.out / 'seg_before_after.json').write_text(json.dumps(stats, indent=2))

    A0, A1, S0, S1, M0 = forward(pick)

    # ------------------------------------------------------------------ plot
    n = a.n_show
    fig = plt.figure(figsize=(25.5, 3.55 * n + 2.4))
    gs = fig.add_gridspec(n + 1, 7, hspace=.20, wspace=.06,
                          height_ratios=[1] * n + [0.62],
                          left=.028, right=.985, top=.945, bottom=.02)
    K = dict(fontsize=10.5, pad=6)

    for r in range(n):
        seg0 = CLASS_RGB[A0[r]]
        seg1 = CLASS_RGB[A1[r]]
        ch = A0[r] != A1[r]
        # colour the change by DESTINATION class
        chimg = np.ones((*ch.shape, 3)) * 0.97
        for c in range(4):
            chimg[ch & (A1[r] == c)] = CLASS_RGB[c]
        d = S1[r] - S0[r]
        v = max(abs(d).max(), 1e-8)

        panels = [
            (im256[pick[r]], 'B-scan  slice %d' % pick[r], 'gray', None, None),
            (seg0, 'segmentation BEFORE  (frozen)', None, None, None),
            (seg1, 'segmentation AFTER  (adapted)', None, None, None),
            (chimg, 'CHANGED  %.2f%% of pixels'
             % (100 * ch.mean()), None, None, None),
            (S0[r], 'anatomy score BEFORE', 'magma', 0, 1),
            (S1[r], 'anatomy score AFTER', 'magma', 0, 1),
            (d, r'$\Delta S$   max %.3f' % v, 'bwr', -v, v),
        ]
        for c, (im, t, cm, lo, hi) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            h = ax.imshow(im, cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
            if r == 0:
                ax.set_title(t, **K)
            elif c in (3, 6):
                ax.set_title(t, fontsize=9, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 6:
                plt.colorbar(h, ax=ax, fraction=.046)

    ax = fig.add_subplot(gs[n, :3])
    ax.axis('off')
    ax.legend(handles=[Patch(facecolor=CLASS_RGB[i], label=CLASS_NAMES[i])
                       for i in range(4)],
              loc='upper left', ncol=4, fontsize=11, frameon=False,
              title='MIRAGE classes   (change column is coloured by DESTINATION class)',
              title_fontsize=11)

    ax = fig.add_subplot(gs[n, 3:5])
    keys = sorted(stats['transitions'], key=stats['transitions'].get, reverse=True)[:6]
    vals = [stats['transitions'][k] for k in keys]
    ax.barh(range(len(keys))[::-1], vals, color='#2a9d8f')
    ax.set_yticks(range(len(keys))[::-1])
    ax.set_yticklabels(keys, fontsize=9)
    ax.set_xlabel('pixels (%d held-out images)' % stats['n_images'], fontsize=9)
    ax.set_title('which class transitions happen', fontsize=10.5)
    ax.grid(axis='x', alpha=.25)

    ax = fig.add_subplot(gs[n, 5:])
    ax.axis('off')
    txt = (
        'HELD-OUT, %d images\n\n'
        'pixels changing class        %.2f%%\n'
        'InnerRetina area change      %+.3f pp\n'
        'Choroid area change          %+.3f pp\n\n'
        'frozen confidence margin\n'
        '  where a pixel CHANGED      %.4f\n'
        '  where it did NOT           %.4f\n'
        '  -> edits land where MIRAGE was unsure\n\n'
        'L_rel  %.4f -> %.4f  (%.1f%%)\n\n'
        'CAVEAT  FairVision has no anatomy labels,\n'
        'so CHANGED is not IMPROVED.  Correctness\n'
        'needs a labelled set (GOALS).'
        % (stats['n_images'], stats['pct_pixels_changed'],
           stats['inner_area_change_pp'], stats['choroid_area_change_pp'],
           stats['mean_margin_where_changed'], stats['mean_margin_where_unchanged'],
           rep['heldout']['L_rel_before'], rep['heldout']['L_rel_after'],
           rep['heldout']['L_rel_reduction_pct'])
    )
    ax.text(0, 1, txt, va='top', family='monospace', fontsize=9.5,
            linespacing=1.5)

    fig.suptitle('Raw MIRAGE segmentation before and after JEPA distillation   '
                 '(64x64 decoder output, HELD-OUT slices;  MIRAGE fully frozen, '
                 'adapter only:  $H = H_0 + %.2f\\tanh(A(H_0))$)' % cfg['alpha'],
                 fontsize=14, y=.977)
    f = a.out / 'seg_before_after.png'
    fig.savefig(f, dpi=104, facecolor='white')
    print('wrote %s' % f)
    print('pixels changed %.2f%%   inner %+.3f pp   choroid %+.3f pp'
          % (stats['pct_pixels_changed'], stats['inner_area_change_pp'],
             stats['choroid_area_change_pp']))
    print('margin where changed %.4f  vs unchanged %.4f'
          % (stats['mean_margin_where_changed'],
             stats['mean_margin_where_unchanged']))
    for k in keys:
        print('   %-28s %d' % (k, stats['transitions'][k]))


if __name__ == '__main__':
    main()
