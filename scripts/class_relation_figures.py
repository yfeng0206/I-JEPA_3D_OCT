#!/usr/bin/env python
"""Figures for the class-relation analysis and the segmentation before/after."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_stage import ResBlock                                   # noqa: E402
from goals_eval import load_pairs, dice_iou, VOID, RES               # noqa: E402
from jepa_to_mirage_probe import build_mirage                        # noqa: E402
from class_relation_probe import EncAdapter                          # noqa: E402

OUT = REPO / 'results/masking/class_relations'
PLACE = REPO / 'results/masking/placement'
CLS = ('bg', 'inner', 'choroid')
RGB = {0: (15, 15, 25), 1: (0, 190, 210), 2: (250, 176, 40)}


def fig_blocks():
    res = json.load(open(OUT / 'class_relations.json'))
    order = ['MIRAGE H0', 'MIRAGE enc', 'MIRAGE enc +adapt a0.05',
             'MIRAGE enc +adapt a0.50', 'JEPA ep100 (envelope)',
             'JEPA ep30 (anatomy)', 'JEPA untrained (control)']
    order = [o for o in order if o in res]
    fig, axes = plt.subplots(1, len(order), figsize=(2.35 * len(order), 3.1))
    for ax, nm in zip(axes, order):
        B = np.array(res[nm]['block'])
        im = ax.imshow(B, vmin=-0.6, vmax=1.0, cmap='RdYlBu_r')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, '%.2f' % B[i, j], ha='center', va='center',
                        fontsize=7.5,
                        color='white' if abs(B[i, j]) > 0.65 else 'black')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLS, fontsize=7.5, rotation=45)
        ax.set_yticklabels(CLS, fontsize=7.5)
        t = nm.replace(' +adapt', '\n+adapt').replace(' (', '\n(')
        ax.set_title(t, fontsize=8.5)
    fig.suptitle('Patch-to-patch cosine similarity by tissue class '
                 '(GOALS ground truth, 30 held-out images)', fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    plt.savefig(OUT / 'class_blocks.png', dpi=160, bbox_inches='tight')
    print('wrote', OUT / 'class_blocks.png')

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = [o.replace(' (control)', '').replace(' (envelope)', '')
             .replace(' (anatomy)', '') for o in order]
    col = ['#d62728' if 'MIRAGE' in o and 'adapt' not in o else
           '#ff9896' if 'adapt' in o else '#1f77b4' for o in order]
    ax[0].bar(names, [res[o]['contrast_inner_vs_choroid'] for o in order],
              color=col)
    ax[0].set_ylabel('within-class  −  inner/choroid similarity')
    ax[0].set_title('Does the model separate INNER RETINA from CHOROID?')
    ax[0].tick_params(axis='x', rotation=38, labelsize=7.5)
    ax[0].grid(alpha=.3, axis='y')
    ax[1].bar(names, [res[o]['contrast_tissue_vs_bg'] for o in order], color=col)
    ax[1].set_ylabel('tissue  −  background similarity')
    ax[1].set_title('Does the model separate TISSUE from BACKGROUND?')
    ax[1].tick_params(axis='x', rotation=38, labelsize=7.5)
    ax[1].grid(alpha=.3, axis='y')
    for x in ax:
        plt.setp(x.get_xticklabels(), ha='right')
    plt.tight_layout()
    plt.savefig(OUT / 'class_contrast.png', dpi=160)
    print('wrote', OUT / 'class_contrast.png')


def colourise(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, v in RGB.items():
        out[lab == k] = v
    return out


def fig_segmentation():
    dev = 'cuda'
    mir = build_mirage(dev)
    sem = mir.output_adapters['semseg']
    imgs, gts, names = load_pairs()

    def predict(adapter=None):
        h = None
        if adapter is not None:
            def pre(m, args):
                x = args[0]
                return (adapter(x.float()).to(x.dtype),) + args[1:]
            h = sem.proj_dec.register_forward_pre_hook(pre)
        out = []
        try:
            for s in range(0, len(imgs), 8):
                x = torch.from_numpy(imgs[s:s + 8])[:, None].to(dev)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    o = mir({'bscan': x})
                lg = (o['semseg'] if isinstance(o, dict) else o).float()
                if lg.shape[-1] != RES:
                    lg = F.interpolate(lg, size=(RES, RES), mode='bilinear',
                                       align_corners=False)
                lg[:, VOID] = float('-inf')
                out.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
        finally:
            if h is not None:
                h.remove()
        return np.concatenate(out)

    variants = [('frozen MIRAGE', None)]
    for al in (0.05, 0.50):
        p = PLACE / ('adapter_enc_a%.2f.pt' % al)
        if p.exists():
            ck = torch.load(p, map_location=dev)
            m = EncAdapter(ck['cfg']['ch'], ck['cfg']['depth'],
                           ck['cfg']['width'], ck['cfg']['alpha']).to(dev)
            m.load_state_dict(ck['state_dict']); m.eval()
            variants.append((r'encoder adapter $\alpha$=%.2f' % al, m))

    preds, dices = {}, {}
    for nm, m in variants:
        p = predict(m)
        preds[nm] = p
        per = []
        for i in range(len(gts)):
            ds = [dice_iou(p[i], gts[i], c)[0] for c in (1, 2)]
            per.append(np.mean([d for d in ds if d is not None]))
        dices[nm] = (float(np.mean(per)), np.array(per))

    # Show the images where the strong adapter changes the most, so the figure
    # cannot be accused of hiding the damage.
    worst = np.argsort(dices[variants[-1][0]][1] - dices['frozen MIRAGE'][1])[:4]
    ncol = 2 + len(variants)
    fig, axes = plt.subplots(len(worst), ncol, figsize=(2.5 * ncol, 2.6 * len(worst)))
    for r, i in enumerate(worst):
        axes[r, 0].imshow(imgs[i], cmap='gray')
        axes[r, 0].set_ylabel(names[i].replace('.png', ''), fontsize=8)
        axes[r, 1].imshow(colourise(gts[i]))
        for c, (nm, _) in enumerate(variants):
            axes[r, 2 + c].imshow(colourise(preds[nm][i]))
            d = dices[nm][1][i]
            axes[r, 2 + c].set_title('Dice %.4f' % d, fontsize=8)
        if r == 0:
            axes[r, 0].set_title('B-scan', fontsize=9)
            axes[r, 1].set_title('ground truth', fontsize=9)
            for c, (nm, _) in enumerate(variants):
                axes[r, 2 + c].set_title('%s\nDice %.4f'
                                         % (nm, dices[nm][1][worst[0]]), fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(RGB[1]) / 255, label='inner retina'),
          Patch(color=np.array(RGB[2]) / 255, label='choroid')]
    fig.legend(handles=hl, loc='lower center', ncol=2, fontsize=9)
    ttl = '  |  '.join('%s %.4f' % (n, dices[n][0]) for n, _ in variants)
    fig.suptitle('GOALS segmentation, 4 most-changed images\nmean anatomy Dice:  '
                 + ttl, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(OUT / 'segmentation_before_after.png', dpi=150)
    print('wrote', OUT / 'segmentation_before_after.png')
    for nm, _ in variants:
        print('   %-28s mean anatomy Dice %.4f' % (nm, dices[nm][0]))


if __name__ == '__main__':
    fig_blocks()
    fig_segmentation()
