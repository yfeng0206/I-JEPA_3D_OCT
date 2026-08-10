#!/usr/bin/env python3
"""Honest anatomy-vs-random comparison through the PRODUCTION collator only.

Two claims need checking, and I got one of them wrong before:

  claim 1  anatomy masks the important region more often than random
  claim 2  anatomy leaves MORE context than random

Claim 2 was previously "measured" by taking the naive complement of the
target union (~200 tokens).  That is not what the encoder receives.  The real
I-JEPA policy samples an encoder block at enc_mask_scale, subtracts the target
union from it, and then stacks the batch on its minimum length.  Every number
here comes from CurriculumMaskGenerator.generate(), so the context column is
what the encoder actually sees.

Arms, all on identical guides and identical seeds:

  random_default    rectangles, ramp cold (r_t=0), pred_mask_scale as shipped
  random_matched    rectangles, ramp cold, scale lowered to match anatomy area
  envelope_default  rectangles aimed at the retina, ramp hot
  anatomy           connected anatomy-shaped targets, ramp hot

random_matched is the arm that isolates SHAPE: without it, anatomy differs
from random in both where it masks and how much, and the comparison cannot
separate the two.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.masks.curriculum import CurriculumMaskGenerator     # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/fair'
G = 16


def make_guides(per):
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)
    plc = (occ >= 0.25).astype(np.float32)
    return torch.from_numpy(np.stack([occ, plc], 1)), occ


def arm(mode, scale, epoch, per, guides, occ, batch, k, seed=0):
    cfg = {'mode': mode, 'enabled': True, 'T_warm': 25, 'T_total': 30,
           'r_max': 1.0, 'ramp_shape': 'linear',
           'anatomy_mass_cap': 0.90, 'anatomy_tau': 0.10}
    gen = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16, npred=4, nenc=1,
        pred_mask_scale=scale,
        pred_target_k=k if mode == 'mirage_anatomy' else None,
        curriculum_cfg=cfg)
    gen.set_epoch(epoch, 100)
    n = len(per)
    ctx, hid, on_anat, conn, dead, comps = [], [], [], [], [], []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        random.seed(seed + s); torch.manual_seed(seed + s); np.random.seed(seed + s)
        enc, pred = gen.generate(
            batch_size=b, guide_grids=guides[s:s + b],
            guide_valid=torch.ones(b, dtype=torch.bool))
        ctx.extend([int(enc[0].shape[1])] * b)
        for j in range(b):
            o = occ[s + j].ravel()
            u = set()
            for p in pred:
                idx = p[j].numpy()
                u.update(idx.tolist())
                m = np.zeros(G * G, bool); m[idx] = True
                nc = ndimage.label(m.reshape(G, G), structure=np.ones((3, 3)))[1]
                comps.append(nc)
                conn.append(int(nc == 1))
                # A target is DEAD if it lands almost entirely off anatomy.
                dead.append(int(o[idx].mean() < 0.10))
            idx = np.array(sorted(u))
            hid.append(len(idx))
            on_anat.append(float(o[idx].mean()))
    return {'mode': mode, 'scale': list(scale), 'epoch': epoch,
            'context_tokens': float(np.mean(ctx)),
            'hidden_cells': float(np.mean(hid)),
            'on_anatomy_pct': 100 * float(np.mean(on_anat)),
            'connected_pct': 100 * float(np.mean(conn)),
            'dead_target_pct': 100 * float(np.mean(dead)),
            'mean_components': float(np.mean(comps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    guides, occ = make_guides(per)

    arms = [
        ('random_default', 'mirage_envelope', (0.15, 0.2), 0),
        ('random_matched', 'mirage_envelope', (0.055, 0.075), 0),
        ('envelope_default', 'mirage_envelope', (0.15, 0.2), 50),
        # Rectangles aimed at the retina AND shrunk to anatomy's masked area.
        # envelope_default differs from anatomy in three ways at once (shape,
        # area, context), so it cannot attribute anything to shape.  This arm
        # holds placement and area fixed so shape is the only free variable.
        ('envelope_matched', 'mirage_envelope', (0.055, 0.075), 50),
        ('anatomy', 'mirage_anatomy', (0.15, 0.2), 50),
    ]
    res = {}
    hdr = ('%-18s %9s %9s %11s %11s %10s' %
           ('arm', 'context', 'hidden', 'on-anatomy', 'connected', 'dead tgt'))
    print('anatomy vs random through the PRODUCTION collator (%d slices)' % len(per))
    print()
    print(hdr); print('-' * len(hdr))
    for tag, mode, scale, epoch in arms:
        r = arm(mode, scale, epoch, per, guides, occ, a.batch, a.k)
        res[tag] = r
        print('%-18s %9.1f %9.1f %10.1f%% %10.1f%% %9.2f%%'
              % (tag, r['context_tokens'], r['hidden_cells'],
                 r['on_anatomy_pct'], r['connected_pct'], r['dead_target_pct']))

    an, rm, rd = res['anatomy'], res['random_matched'], res['random_default']
    print()
    print('CLAIM 1  masks the important region more than random')
    print('   anatomy %.1f%% on-anatomy vs random_matched %.1f%%  -> %.2fx'
          % (an['on_anatomy_pct'], rm['on_anatomy_pct'],
             an['on_anatomy_pct'] / max(rm['on_anatomy_pct'], 1e-9)))
    print('   dead targets  anatomy %.2f%% vs random_matched %.2f%%'
          % (an['dead_target_pct'], rm['dead_target_pct']))
    print()
    print('CLAIM 2  leaves more context than random')
    print('   vs random_default  %.1f vs %.1f tokens  -> %s'
          % (an['context_tokens'], rd['context_tokens'],
             'HOLDS' if an['context_tokens'] > rd['context_tokens'] else 'FAILS'))
    print('   vs random_matched  %.1f vs %.1f tokens  -> %s'
          % (an['context_tokens'], rm['context_tokens'],
             'HOLDS' if an['context_tokens'] > rm['context_tokens'] else 'FAILS'))
    print('   (matched is the honest control: same masked area, only shape differs)')
    (OUT / 'fair.json').write_text(json.dumps(res, indent=2))
    print()
    print('wrote', OUT / 'fair.json')


if __name__ == '__main__':
    main()
