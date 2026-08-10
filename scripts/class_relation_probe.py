#!/usr/bin/env python
"""Do MIRAGE and I-JEPA agree about how the tissue classes relate?

L_rel assumes the two models should agree about which patches resemble which.
This decomposes that assumption by CLASS, using GOALS human labels.

For every model, every ordered pair of 16x16 grid cells within an image is
binned by the ground-truth class of its two endpoints, giving a 3x3 block
matrix of mean cosine similarity:

              background   inner retina   choroid
  background      s00           s01          s02
  inner                         s11          s12
  choroid                                    s22

The diagonal is within-class cohesion (do same-class patches look alike?), the
off-diagonal is between-class similarity (are the classes separated?).  Their
difference is the contrast the segmentation decoder relies on.

If MIRAGE and I-JEPA produce the same block structure, relational distillation
is well posed.  If they do not, L_rel is asking MIRAGE to adopt a class
geometry that contradicts its own -- which is what the GOALS Dice degradation
would then reflect.

An untrained I-JEPA encoder is included as a control: any structure it shows is
inherited from raw pixel statistics, not learned.
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

from adapter_stage import ResBlock                                   # noqa: E402
from goals_eval import load_pairs                                    # noqa: E402
from jepa_to_mirage_probe import (build_mirage, build_jepa,          # noqa: E402
                                  IMNET_MEAN, IMNET_STD)

GRID = 16
PURITY = 0.7
OUT = REPO / 'results/masking/class_relations'
CLASSES = ('background', 'inner', 'choroid')
PLACE = REPO / 'results/masking/placement'


class EncAdapter(nn.Module):
    def __init__(self, ch, depth, width, alpha):
        super().__init__()
        layers = [nn.Conv2d(ch, width, 1), nn.GELU()]
        layers += [ResBlock(width) for _ in range(depth)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, ch, 1)
        self.alpha = alpha

    def forward(self, x):
        tok = x.dim() == 3
        if tok:
            b, n, c = x.shape
            g = int(round(n ** 0.5))
            x = x.transpose(1, 2).reshape(b, c, g, g)
        y = x + self.alpha * torch.tanh(self.out(self.trunk(x)))
        if tok:
            y = y.flatten(2).transpose(1, 2)
        return y


def token_labels(gt, grid=GRID):
    h = gt.shape[0] // grid
    cells = gt.reshape(grid, h, grid, h).transpose(0, 2, 1, 3).reshape(grid * grid, -1)
    lab = np.zeros(grid * grid, np.int64)
    pur = np.zeros(grid * grid, np.float32)
    for i, c in enumerate(cells):
        cnt = np.bincount(c, minlength=4)
        lab[i] = cnt.argmax()
        pur[i] = cnt.max() / cnt.sum()
    return lab, pur


def block_matrix(feats, labels, purity):
    """Mean cosine similarity per class-pair, aggregated over images.

    Similarity is computed WITHIN each image, matching L_rel, which compares
    per-image Gram matrices.  The self-similarity diagonal is excluded.
    """
    acc = np.zeros((3, 3)); cnt = np.zeros((3, 3))
    for z, lab, pur in zip(feats, labels, purity):
        keep = (pur >= PURITY) & (lab < 3)
        if keep.sum() < 4:
            continue
        v = F.normalize(torch.from_numpy(z[keep]).float(), dim=-1)
        s = (v @ v.T).numpy()
        li = lab[keep]
        np.fill_diagonal(s, np.nan)
        for a in range(3):
            for b in range(3):
                m = np.outer(li == a, li == b)
                vals = s[m]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    acc[a, b] += vals.sum(); cnt[a, b] += len(vals)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa-ep100', default=str(
        r'D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar'))
    ap.add_argument('--jepa-ep30', default=str(
        r'D:\jepa_phase0\runs\patch_mirage_anatomy\jepa_patch_mirage-ep30.pth.tar'))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'

    mir = build_mirage(dev)
    sem = mir.output_adapters['semseg']
    grab = {}
    sem.proj_dec.register_forward_hook(
        lambda m, i, o: grab.update(enc=i[0].detach()))
    sem.final_layer.register_forward_hook(
        lambda m, i, o: grab.update(h0=i[0].detach()))

    imgs, gts, names = load_pairs()
    labels, purity = [], []
    for g in gts:
        l, p = token_labels(g)
        labels.append(l); purity.append(p)
    kept = int(sum(((p >= PURITY) & (l < 3)).sum()
                   for l, p in zip(labels, purity)))
    print('GOALS images %d   grid %dx%d   tokens kept (purity>=%.0f%%) %d'
          % (len(imgs), GRID, GRID, 100 * PURITY, kept))
    for c, nm in enumerate(CLASSES):
        n = int(sum(((p >= PURITY) & (l == c)).sum()
                    for l, p in zip(labels, purity)))
        print('   %-11s %5d tokens' % (nm, n))
    print()

    adapters = {}
    for al in (0.05, 0.50):
        p = PLACE / ('adapter_enc_a%.2f.pt' % al)
        if p.exists():
            ck = torch.load(p, map_location=dev)
            m = EncAdapter(ck['cfg']['ch'], ck['cfg']['depth'],
                           ck['cfg']['width'], ck['cfg']['alpha']).to(dev)
            m.load_state_dict(ck['state_dict']); m.eval()
            adapters[al] = m

    feats = {'MIRAGE enc': [], 'MIRAGE H0': []}
    for al in adapters:
        feats['MIRAGE enc +adapt a%.2f' % al] = []
    for i in range(len(imgs)):
        x = torch.from_numpy(imgs[i])[None, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x})
        e = grab['enc'].float()
        h = F.adaptive_avg_pool2d(grab['h0'].float(),
                                  (GRID, GRID)).flatten(2).transpose(1, 2)
        feats['MIRAGE enc'].append(e[0].cpu().numpy())
        feats['MIRAGE H0'].append(h[0].cpu().numpy())
        with torch.no_grad():
            for al, m in adapters.items():
                feats['MIRAGE enc +adapt a%.2f' % al].append(
                    m(e)[0].cpu().numpy())

    import cv2
    jepa_models = {
        'JEPA ep100 (envelope)': a.jepa_ep100,
        'JEPA ep30 (anatomy)': a.jepa_ep30,
        'JEPA untrained (control)': None,
    }
    for nm, ck in jepa_models.items():
        if ck is None:
            from src.models.vision_transformer import vit_base
            torch.manual_seed(0)
            enc = vit_base(img_size=[256], patch_size=16).to(dev).eval()
        else:
            enc = build_jepa(pathlib.Path(ck), dev)
        feats[nm] = []
        for i in range(len(imgs)):
            b = cv2.resize(imgs[i], (256, 256), interpolation=cv2.INTER_LINEAR)
            rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
            x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(dev)
            with torch.no_grad():
                z = F.layer_norm(enc(x), (768,))
            feats[nm].append(z[0].cpu().numpy())
        del enc
        torch.cuda.empty_cache()

    order = ['MIRAGE enc', 'MIRAGE H0']
    order += [k for k in feats if k.startswith('MIRAGE enc +adapt')]
    order += [k for k in jepa_models]
    res = {}
    print('%-26s %8s %8s %8s %8s %8s %8s' %
          ('model', 'bg-bg', 'in-in', 'ch-ch', 'in-ch', 'in-bg', 'ch-bg'))
    print('-' * 80)
    for nm in order:
        B = block_matrix(feats[nm], labels, purity)
        res[nm] = {'block': B.tolist(),
                   'within_inner': B[1, 1], 'within_choroid': B[2, 2],
                   'inner_choroid': B[1, 2], 'inner_bg': B[0, 1],
                   'choroid_bg': B[0, 2], 'bg_bg': B[0, 0],
                   'contrast_inner_vs_choroid':
                       0.5 * (B[1, 1] + B[2, 2]) - B[1, 2],
                   'contrast_tissue_vs_bg':
                       0.5 * (B[1, 1] + B[2, 2]) - 0.5 * (B[0, 1] + B[0, 2])}
        print('%-26s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f'
              % (nm, B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 1], B[0, 2]))

    print()
    print('%-26s %14s %14s %10s' %
          ('model', 'inner-vs-chor', 'tissue-vs-bg', 'chor>inner?'))
    print('-' * 68)
    for nm in order:
        r = res[nm]
        print('%-26s %+14.4f %+14.4f %10s'
              % (nm, r['contrast_inner_vs_choroid'], r['contrast_tissue_vs_bg'],
                 'yes' if r['within_choroid'] > r['within_inner'] else 'no'))

    # Does each JEPA agree with MIRAGE about the SHAPE of the block matrix?
    print()
    print('AGREEMENT WITH MIRAGE  (correlation over the 6 unique block values)')
    keys = ['bg_bg', 'within_inner', 'within_choroid', 'inner_choroid',
            'inner_bg', 'choroid_bg']
    for ref in ('MIRAGE enc', 'MIRAGE H0'):
        v0 = np.array([res[ref][k] for k in keys])
        print('  vs %s:' % ref)
        for nm in order:
            if nm == ref:
                continue
            v1 = np.array([res[nm][k] for k in keys])
            r = float(np.corrcoef(v0, v1)[0, 1])
            print('     %-26s r = %+.4f' % (nm, r))
    for nm in res:
        res[nm] = {k: (float(v) if not isinstance(v, list) else v)
                   for k, v in res[nm].items()}
    (OUT / 'class_relations.json').write_text(json.dumps(res, indent=2))
    print('\nwrote', OUT / 'class_relations.json')


if __name__ == '__main__':
    main()
