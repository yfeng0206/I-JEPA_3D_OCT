#!/usr/bin/env python
"""All masking arms on the same 1,000 slices, with honest connectivity.

`fair_compare.py` reports `connected_pct` using `ndimage.label` with a 3x3
structure, i.e. 8-connectivity, where a diagonal touch counts as adjacency.
Under that rule every arm scores 100% -- including targets that are visibly
checkerboards. This reports BOTH rules so the difference is explicit:

    8-conn   diagonal touch counts (what the docs have been quoting)
    4-conn   must share an edge (what "a connected region" normally means)

Arms:
    random_default     rectangles, ramp cold
    random_matched     rectangles, ramp cold, area matched to anatomy
    envelope_default   rectangles placed on the retina, ramp hot
    anatomy_8conn      current production: 8-neighbour growth
    anatomy_4conn      proposed: 4-neighbour growth
    anatomy_4conn_rim  proposed + one ring of rim growth

Everything goes through the production CurriculumMaskGenerator on identical
guides and seeds.  CPU only.
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

import src.masks.anatomy as AN                                     # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator           # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/fair'
G = 16
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
BOX = np.ones((3, 3))
NB4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def grow_rim(occ, radius, directions=None, rim_score=0.30, tau=0.10):
    """Grow the anatomy region outward by `radius`, never leaving the grid.

    The region is defined by `occ >= tau`, NOT by `occ > 0`. Pooling a soft
    200x200 MIRAGE map down to 16x16 leaves a small non-zero value in every
    cell, so an `occ > 0` test selects the whole grid and the rim adds nothing
    -- measured at 0.00 cells/slice before this was fixed.
    """
    if radius <= 0:
        return occ.copy()
    out = occ >= tau
    dirs = [DIRS[d] for d in (directions or DIRS)]
    h, w = occ.shape
    for _ in range(radius):
        g = out.copy()
        for dr, dc in dirs:
            s = np.zeros_like(out)
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            s[r0:r1, c0:c1] = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            g |= s
        out = g
    grown = occ.copy()
    new = out & (occ < tau)
    grown[new] = rim_score
    return grown


def arm(mode, scale, epoch, per, occ_all, batch, k, nb, rim, seed=0):
    AN.NB8[:] = nb
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
    ctx, hid, on, c4, c8, hol, dead = [], [], [], [], [], [], []
    anat_h, bg_h, anat_ctx = [], [], []
    rng = np.random.default_rng(seed)
    for s in range(0, n, batch):
        b = min(batch, n - s)
        random.seed(seed + s); torch.manual_seed(seed + s); np.random.seed(seed + s)
        occ_b = occ_all[s:s + b]
        if rim > 0:
            fed = np.stack([grow_rim(o, rim,
                                     None if rng.random() < 0.5 else
                                     ['down', 'left', 'right'])
                            for o in occ_b])
        else:
            fed = occ_b
        t = torch.from_numpy(fed).float()
        guides = torch.stack([t, (t >= 0.25).float()], 1)
        enc, pred = gen.generate(batch_size=b, guide_grids=guides,
                                 guide_valid=torch.ones(b, dtype=torch.bool))
        ctx.extend([int(enc[0].shape[1])] * b)
        for j in range(b):
            o = occ_b[j].ravel()          # score against the TRUE guide always
            u = set()
            for p in pred:
                idx = p[j].numpy()
                u.update(idx.tolist())
                m = np.zeros(G * G, bool); m[idx] = True
                M = m.reshape(G, G)
                c8.append(ndimage.label(M, structure=BOX)[1] == 1)
                c4.append(ndimage.label(M, structure=CROSS)[1] == 1)
                hol.append(int((ndimage.binary_fill_holes(M) & ~M).sum()))
                dead.append(int(o[idx].mean() < 0.10))
            idx = np.array(sorted(u))
            hid.append(len(idx))
            on.append(float(o[idx].mean()))
            a = float(o[idx].sum())
            anat_h.append(a); bg_h.append(len(idx) - a)
            anat_ctx.append(float(o[enc[0][j].numpy()].sum()))
    return {
        'context': float(np.mean(ctx)), 'hidden': float(np.mean(hid)),
        'anat_hidden': float(np.mean(anat_h)), 'bg_hidden': float(np.mean(bg_h)),
        'anat_ctx': float(np.mean(anat_ctx)),
        'on_anatomy_pct': 100 * float(np.mean(on)),
        'conn8_pct': 100 * float(np.mean(c8)),
        'conn4_pct': 100 * float(np.mean(c4)),
        'holes': float(np.mean(hol)),
        'dead_pct': 100 * float(np.mean(dead)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=250)
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)

    arms = [
        ('random_default',    'mirage_envelope', (0.15, 0.2),   0, NB8, 0),
        ('random_matched',    'mirage_envelope', (0.055, 0.075), 0, NB8, 0),
        ('envelope_default',  'mirage_envelope', (0.15, 0.2),  50, NB8, 0),
        ('anatomy_8conn',     'mirage_anatomy',  (0.15, 0.2),  50, NB8, 0),
        ('anatomy_4conn',     'mirage_anatomy',  (0.15, 0.2),  50, NB4, 0),
        ('anatomy_4conn_rim', 'mirage_anatomy',  (0.15, 0.2),  50, NB4, 1),
    ]
    print('all arms on the SAME %d slices, production collator\n' % len(per))
    hdr = ('%-19s %8s %7s %8s %7s %9s %8s %8s %7s %7s' %
           ('arm', 'context', 'hidden', 'anat_hid', 'bg_hid',
            'on-anat', '8-conn', '4-conn', 'holes', 'dead'))
    print(hdr); print('-' * len(hdr))
    res = {}
    for tag, mode, scale, ep, nb, rim in arms:
        r = arm(mode, scale, ep, per, occ, a.batch, a.k, nb, rim)
        res[tag] = r
        print('%-19s %8.1f %7.1f %8.1f %7.1f %8.1f%% %7.1f%% %7.1f%% %7.2f %6.1f%%'
              % (tag, r['context'], r['hidden'], r['anat_hidden'], r['bg_hidden'],
                 r['on_anatomy_pct'], r['conn8_pct'], r['conn4_pct'],
                 r['holes'], r['dead_pct']))
    (OUT / 'arms_with_connectivity.json').write_text(json.dumps(res, indent=2))
    print('\nwrote', OUT / 'arms_with_connectivity.json')

    e, a8, a4 = res['envelope_default'], res['anatomy_8conn'], res['anatomy_4conn']
    print('\nkey contrasts')
    print('  budget on tissue      envelope %.0f%% vs anatomy %.0f%%'
          % (100 * e['anat_hidden'] / max(e['hidden'], 1),
             100 * a4['anat_hidden'] / max(a4['hidden'], 1)))
    print('  wasted on background  envelope %.1f cells vs anatomy %.1f cells'
          % (e['bg_hidden'], a4['bg_hidden']))
    print('  truly connected       envelope %.0f%%  anatomy(8) %.0f%%  anatomy(4) %.0f%%'
          % (e['conn4_pct'], a8['conn4_pct'], a4['conn4_pct']))


if __name__ == '__main__':
    main()
