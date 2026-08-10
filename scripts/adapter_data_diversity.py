#!/usr/bin/env python
"""Does adapter training data diversity matter, and where does it saturate?

Two facts motivate this:

  1. The cache every adapter experiment has used (`adapter_sweep`) contains
     6,000 volumes at **slice 100 only** -- one fixed depth out of 200.  The
     adapter has never seen a peripheral B-scan.
  2. The saturation claim ("no gain past ~2,400 images") was measured on that
     same single-depth cache, so it may be saturation of a narrow distribution
     rather than of the adapter.

A stratified cache (`slice_pos`) exists: 6,000 volumes sampled across 100
depths spanning 0-199.  This trains identically on each and cross-evaluates,
so generalisation across depth is measured rather than assumed.

The saturation curve comes free: a single pass is evaluated at 1,200 / 2,400 /
4,800 images seen.

Reported per checkpoint:
  L_rel red (own)   held-out reduction on the SAME distribution it trained on
  L_rel red (other) held-out reduction on the OTHER depth distribution
  GOALS Dice        ground-truth segmentation cost
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

from adapter_placement_ablation import Adapter, gram, to_tokens      # noqa: E402
from goals_eval import load_pairs, dice_iou, VOID, RES               # noqa: E402
from jepa_to_mirage_probe import (build_mirage, build_jepa,          # noqa: E402
                                  IMNET_MEAN, IMNET_STD)

CACHES = {
    'middle': pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\adapter_sweep'),
    'stratified': pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos'),
}
OUT = REPO / 'results/masking/data_diversity'
GRID = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=4800)
    ap.add_argument('--n-eval', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--tap', default='enc')
    ap.add_argument('--checkpoints', default='1200,2400,4800')
    ap.add_argument('--jepa-ckpt', default=str(
        r'D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar'))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    ckpts = [int(v) for v in a.checkpoints.split(',')]

    mir = build_mirage(dev)
    enc = build_jepa(pathlib.Path(a.jepa_ckpt), dev)
    for p in enc.parameters():
        p.requires_grad_(False)
    sem = mir.output_adapters['semseg']
    TAP = {'enc': (sem.proj_dec, 768), 'mid': (sem.blocks, 384),
           'h0': (sem.final_layer, 384)}
    grab = {}
    for nm, (mod, _) in TAP.items():
        mod.register_forward_hook(
            lambda m, i, o, nm=nm: grab.update({nm: i[0].detach()}))
    tap_mod, tap_ch = TAP[a.tap]

    data = {}
    for nm, root in CACHES.items():
        im512 = np.load(root / 'im512.npy', mmap_mode='r')
        im256 = np.load(root / 'im256.npy', mmap_mode='r')
        rng = np.random.default_rng(0)
        perm = rng.permutation(min(len(im512), a.n_train + a.n_eval))
        data[nm] = {'im512': im512, 'im256': im256,
                    'ev': np.sort(perm[:a.n_eval]),
                    'tr': np.sort(perm[a.n_eval:])}
        pos = root / 'pos.npy'
        depths = (len(np.unique(np.load(pos))) if pos.exists() else 1)
        print('%-11s %d samples, %d distinct slice depths' % (nm, len(im512), depths))
    print()

    def taps_of(cache, idx):
        x = torch.from_numpy(
            np.asarray(data[cache]['im512'][idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mir({'bscan': x})
        return grab[a.tap].float()

    def jgram(cache, idx):
        b = np.asarray(data[cache]['im256'][idx], np.float32) / 255.
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2).astype(np.float32)).to(dev)
        with torch.no_grad():
            return gram(F.layer_norm(enc(x), (768,)))

    RJ = {}
    for cache in CACHES:
        for split in ('tr', 'ev'):
            for s in range(0, len(data[cache][split]), a.batch):
                i = np.sort(data[cache][split][s:s + a.batch])
                RJ[(cache, i.tobytes())] = jgram(cache, i).half().cpu()
    print('JEPA grams precomputed\n')

    g_imgs, g_gts, _ = load_pairs()

    def goals_dice(adapter=None):
        h = None
        if adapter is not None:
            def pre(m, args):
                x = args[0]
                return (adapter(x.float()).to(x.dtype),) + args[1:]
            h = tap_mod.register_forward_pre_hook(pre)
        out = []
        try:
            for s in range(0, len(g_imgs), 8):
                x = torch.from_numpy(g_imgs[s:s + 8])[:, None].to(dev)
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
        pred = np.concatenate(out)
        per = []
        for i in range(len(g_gts)):
            ds = [dice_iou(pred[i], g_gts[i], c)[0] for c in (1, 2)]
            per.append(np.mean([d for d in ds if d is not None]))
        return float(np.mean(per))

    def rel_eval(cache, model):
        b, c = [], []
        with torch.no_grad():
            for s in range(0, len(data[cache]['ev']), a.batch):
                i = np.sort(data[cache]['ev'][s:s + a.batch])
                z = taps_of(cache, i)
                r = RJ[(cache, i.tobytes())].to(dev).float()
                b.append(float(F.mse_loss(gram(to_tokens(z)), r)))
                c.append(float(F.mse_loss(gram(to_tokens(model(z))), r)))
        b, c = float(np.mean(b)), float(np.mean(c))
        return 100 * (b - c) / b

    base = goals_dice()
    print('frozen MIRAGE GOALS Dice %.4f' % base)
    print()
    hdr = ('%-12s %8s %13s %13s %9s %10s' %
           ('train data', 'images', 'L_rel own', 'L_rel other', 'Dice', 'vs frozen'))
    print(hdr); print('-' * len(hdr))

    res = {'frozen_dice': base, 'alpha': a.alpha, 'tap': a.tap, 'runs': {}}
    for cache in CACHES:
        other = [c for c in CACHES if c != cache][0]
        torch.manual_seed(0)
        ad = Adapter(tap_ch, 2, 128, a.alpha).to(dev)
        opt = torch.optim.AdamW(ad.parameters(), lr=a.lr, weight_decay=1e-4)
        nst = (len(data[cache]['tr']) + a.batch - 1) // a.batch
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=nst, pct_start=0.1)
        seen = 0
        for s in range(0, len(data[cache]['tr']), a.batch):
            i = np.sort(data[cache]['tr'][s:s + a.batch])
            z = taps_of(cache, i)
            r = RJ[(cache, i.tobytes())].to(dev).float()
            loss = F.mse_loss(gram(to_tokens(ad(z))), r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            seen += len(i)
            if ckpts and seen >= ckpts[0]:
                n = ckpts.pop(0)
                ad.eval()
                own, oth = rel_eval(cache, ad), rel_eval(other, ad)
                d = goals_dice(ad)
                ad.train()
                print('%-12s %8d %12.2f%% %12.2f%% %9.4f %+10.5f'
                      % (cache, seen, own, oth, d, d - base))
                res['runs']['%s_n%d' % (cache, n)] = {
                    'train_cache': cache, 'images': seen,
                    'rel_red_own_pct': own, 'rel_red_other_pct': oth,
                    'goals_dice': d, 'dice_delta': d - base}
        ckpts = [int(v) for v in a.checkpoints.split(',')]

    (OUT / 'diversity.json').write_text(json.dumps(res, indent=2))
    print('\nwrote', OUT / 'diversity.json')


if __name__ == '__main__':
    main()
