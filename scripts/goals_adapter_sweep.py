#!/usr/bin/env python3
"""Sweep adapter hyperparameters using GOALS ground truth as the selection metric.

Every previous adapter sweep optimised L_rel, which is the training objective
and says nothing about whether the resulting segmentation is any GOOD.  GOALS
has real layer labels, so this scores accuracy directly.

The baseline measurement that motivates the sweep: cfg-7 as currently
configured makes GOALS anatomy Dice slightly WORSE, 0.9457 frozen -> 0.9425
adapted.  L_rel contains no segmentation term, so nothing in the objective
rewards preserving Dice.  The question this sweep answers is whether any
setting buys relational alignment without paying for it in segmentation
quality.

Two numbers are reported per config and they pull in opposite directions:

    L_rel reduction   how much the adapter achieved its objective (higher better)
    GOALS anat Dice   whether segmentation survived it (higher better)

alpha=0 is the degenerate control: the adapter is the identity, L_rel reduction
is 0 and Dice is exactly the frozen baseline.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_stage import ResBlock, gram                     # noqa: E402
from goals_eval import load_pairs, predict, score            # noqa: E402
from jepa_to_mirage_probe import build_mirage, build_jepa    # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
OUT = REPO / 'results/masking/goals_sweep'
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = 16


class Adapter(nn.Module):
    """cfg-7 family with optional dropout.

    Dropout carries no parameters, so a state_dict from the dropout-free
    version loads into this unchanged.
    """

    def __init__(self, depth=2, width=128, alpha=0.5, dropout=0.0):
        super().__init__()
        self.alpha = alpha
        layers = [nn.Conv2d(384, width, 1), nn.GELU()]
        if depth == 0:
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        else:
            layers += [ResBlock(width) for _ in range(depth)]
        if dropout > 0:
            layers += [nn.Dropout2d(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h0):
        return h0 + self.alpha * torch.tanh(self.out(self.trunk(h0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa-ckpt', required=True)
    ap.add_argument('--n-train', type=int, default=4800)
    ap.add_argument('--n-eval', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--stage', default='all')
    ap.add_argument('--tag', default='sweep')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'

    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    perm = rng.permutation(a.n_train + a.n_eval)
    idx_ev = np.sort(perm[:a.n_eval])
    idx_tr = np.sort(perm[a.n_eval:])

    mir = build_mirage(dev)
    grab = {}
    head = mir.output_adapters['semseg'].final_layer
    head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))
    enc = build_jepa(pathlib.Path(a.jepa_ckpt), dev)
    for p in enc.parameters():
        p.requires_grad_(False)

    g_imgs, g_gts, _ = load_pairs()
    print('teacher      %s' % pathlib.Path(a.jepa_ckpt).name)
    print('GOALS        %d held-out images' % len(g_imgs))
    print('adapter data %d train / %d eval FairVision slices' % (len(idx_tr), len(idx_ev)))
    print()

    def h0(idx):
        x = torch.from_numpy(np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x})
        return grab['H'].float()

    def rj(idx):
        b = np.asarray(im256[idx], np.float32) / 255.
        rgbv = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgbv.transpose(0, 3, 1, 2)).to(dev)
        with torch.no_grad():
            z = F.layer_norm(enc(x), (768,))
            return gram(z)

    base_pred = predict(mir, head, None, g_imgs, 8, dev)
    base = score(base_pred, g_gts)
    print('%-42s %10s %11s %11s' % ('config', 'L_rel red', 'GOALS Dice', 'vs frozen'))
    print('-' * 78)
    print('%-42s %9s  %11.4f %11s' % ('frozen MIRAGE (no adapter)', '-',
                                      base['anatomy_mean_dice'], 'baseline'))

    def run(depth, width, alpha, dropout, lr):
        torch.manual_seed(0)
        mod = Adapter(depth, width, alpha, dropout).to(dev)
        nst = (len(idx_tr) + a.batch - 1) // a.batch
        opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                  total_steps=nst, pct_start=0.1)
        for s in range(0, len(idx_tr), a.batch):
            i = np.sort(idx_tr[s:s + a.batch])
            H = mod(h0(i))
            U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            F.mse_loss(gram(U), rj(i)).backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        mod.eval()
        r0, r1 = [], []
        with torch.no_grad():
            for s in range(0, len(idx_ev), a.batch):
                i = np.sort(idx_ev[s:s + a.batch])
                H0 = h0(i); H = mod(H0); R = rj(i)
                U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
                U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
                r0.append(float(F.mse_loss(gram(U0), R)))
                r1.append(float(F.mse_loss(gram(U1), R)))
        b, c = float(np.mean(r0)), float(np.mean(r1))
        red = 100 * (b - c) / b
        g = score(predict(mir, head, mod, g_imgs, 8, dev), g_gts)
        return red, g, mod

    configs = []
    if a.stage in ('all', '1'):
        for al in (0.1, 0.25, 0.5, 1.0):
            configs.append(dict(depth=2, width=128, alpha=al, dropout=0.0, lr=1e-3))
    if a.stage in ('all', '2'):
        for d in (0, 4):
            configs.append(dict(depth=d, width=128, alpha=0.25, dropout=0.0, lr=1e-3))
        configs.append(dict(depth=2, width=64, alpha=0.25, dropout=0.0, lr=1e-3))
    if a.stage in ('all', '3'):
        configs.append(dict(depth=2, width=128, alpha=0.25, dropout=0.0, lr=1e-4))
        configs.append(dict(depth=2, width=128, alpha=0.25, dropout=0.1, lr=1e-3))

    res = {'frozen': base, 'teacher': str(a.jepa_ckpt), 'configs': []}
    best = (-1e9, None, None)
    for cfg in configs:
        t0 = time.perf_counter()
        red, g, mod = run(**cfg)
        d = g['anatomy_mean_dice'] - base['anatomy_mean_dice']
        nm = 'd%d w%d a%.2f dr%.1f lr%.0e' % (cfg['depth'], cfg['width'],
                                              cfg['alpha'], cfg['dropout'], cfg['lr'])
        print('%-42s %9.2f%% %11.4f %+11.4f  (%.0fs)'
              % (nm, red, g['anatomy_mean_dice'], d, time.perf_counter() - t0))
        res['configs'].append({**cfg, 'name': nm, 'L_rel_reduction_pct': red,
                               'goals': g, 'dice_delta': d})
        # Prefer preserving segmentation; break ties on relational alignment.
        sc = d * 1000 + red * 0.01
        if sc > best[0]:
            best = (sc, nm, (cfg, mod))
        del mod
        torch.cuda.empty_cache()

    print()
    print('best by (Dice delta, then L_rel): %s' % best[1])
    if best[2] is not None:
        cfg, mod = best[2]
        p = OUT / ('%s_best.pt' % a.tag)
        torch.save({'state_dict': mod.state_dict(),
                    'cfg': {k: cfg[k] for k in ('depth', 'width', 'alpha')},
                    'dropout': cfg['dropout'], 'lr': cfg['lr'],
                    'jepa_ckpt': str(a.jepa_ckpt)}, p)
        print('wrote', p)
    res['best'] = best[1]
    (OUT / ('%s.json' % a.tag)).write_text(json.dumps(res, indent=2))
    print('wrote', OUT / ('%s.json' % a.tag))


if __name__ == '__main__':
    main()
