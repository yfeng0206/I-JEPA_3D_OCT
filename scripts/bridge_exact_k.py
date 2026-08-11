#!/usr/bin/env python
"""Exact-K diagonal bridging, measured through the production collator.

Why exact-K matters
-------------------
`utils.resample_to_k` forces every target to exactly `pred_target_k` before it
reaches the predictor.  A target bridged from 16 to 18 cells would therefore be
subsampled back to 16 UNIFORMLY -- the very operation `shrink_to_k`'s docstring
says "destroys the property the method exists for".  So bridging must return
exactly K cells, or it undoes itself at collation.

    grow NB8 -> shrink_to_k(K) -> bridge diagonals -> trim back to K,
    removing only cells whose deletion keeps the target 4-connected,
    lowest anatomy occupancy first.

Net effect: identical hidden budget, identical K, but every target is one
edge-connected region.

Also re-measures the failure modes documented in `anatomy.is_viable`:
fallback rate, dead targets, and all-4-dead, against the random arm those
fallbacks land in.
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
import src.masks.curriculum as CU                                  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator           # noqa: E402
from scripts.bridge_diagonals_sweep import bridge_diagonals        # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/fair'
G = 16
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
BOX = np.ones((3, 3))
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STATS = {'bridged': 0, 'trimmed': 0, 'calls': 0, 'short': 0, 'notconn': 0}


def trim_to_k_nb4(m, k, score):
    """Drop cells down to `k`, never breaking 4-connectivity.

    A cell is removable only if the rest of the target stays a single
    4-connected component without it (i.e. it is not an articulation point);
    among removable cells the lowest anatomy occupancy goes first.
    """
    out = np.asarray(m, bool).copy()
    while int(out.sum()) > k:
        best, bs = None, None
        for r, c in zip(*np.nonzero(out)):
            out[r, c] = False
            ok = out.any() and ndimage.label(out, structure=CROSS)[1] == 1
            out[r, c] = True
            if not ok:
                continue
            s = float(score[r, c])
            if bs is None or s < bs:
                best, bs = (r, c), s
        if best is None:                  # every remaining cell is a cut vertex
            break
        out[best] = False
    return out


def bridge_exact_k(mask, k, score=None):
    """Production `shrink_to_k`, then bridge, then trim back to exactly k."""
    m = AN.shrink_to_k(mask, k, score)
    sc = np.asarray(score, float) if score is not None else m.astype(float)
    before = int(m.sum())
    b, nadd = bridge_diagonals(m, sc)
    grown = int(b.sum())
    if grown > k:
        b = trim_to_k_nb4(b, k, sc)
    STATS['calls'] += 1
    STATS['bridged'] += nadd
    STATS['trimmed'] += grown - int(b.sum())
    if int(b.sum()) < min(k, before):
        STATS['short'] += 1
    if b.any() and ndimage.label(b, structure=CROSS)[1] != 1:
        STATS['notconn'] += 1
    return b


def arm(mode, scale, epoch, per, occ_all, batch, k, bridge, seed=0):
    AN.NB8[:] = NB8
    CU.anatomy_shrink_to_k = bridge_exact_k if bridge else AN.shrink_to_k
    for key in STATS:
        STATS[key] = 0
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
    ctx, hid, on, c4, c8, hol, dead, alldead = [], [], [], [], [], [], [], []
    anat_h, bg_h, lens = [], [], []
    fb = 0
    for s in range(0, n, batch):
        b = min(batch, n - s)
        random.seed(seed + s); torch.manual_seed(seed + s); np.random.seed(seed + s)
        occ_b = occ_all[s:s + b]
        t = torch.from_numpy(occ_b).float()
        guides = torch.stack([t, (t >= 0.25).float()], 1)
        enc, pred = gen.generate(batch_size=b, guide_grids=guides,
                                 guide_valid=torch.ones(b, dtype=torch.bool))
        fb += gen._mirage_stats.get('fallbacks', 0) if gen._mirage_stats else 0
        ctx.extend([int(enc[0].shape[1])] * b)
        for j in range(b):
            o = occ_b[j].ravel()
            u, d = set(), 0
            for p in pred:
                idx = p[j].numpy()
                u.update(idx.tolist())
                lens.append(len(set(idx.tolist())))
                m = np.zeros(G * G, bool); m[idx] = True
                M = m.reshape(G, G)
                c8.append(ndimage.label(M, structure=BOX)[1] == 1)
                c4.append(ndimage.label(M, structure=CROSS)[1] == 1)
                hol.append(int((ndimage.binary_fill_holes(M) & ~M).sum()))
                dd = int(o[idx].mean() < 0.10)
                dead.append(dd); d += dd
            alldead.append(int(d == 4))
            idx = np.array(sorted(u))
            hid.append(len(idx))
            on.append(float(o[idx].mean()))
            a = float(o[idx].sum())
            anat_h.append(a); bg_h.append(len(idx) - a)
    return {
        'context': float(np.mean(ctx)), 'hidden': float(np.mean(hid)),
        'anat_hidden': float(np.mean(anat_h)), 'bg_hidden': float(np.mean(bg_h)),
        'on_anatomy_pct': 100 * float(np.mean(on)),
        'conn8_pct': 100 * float(np.mean(c8)),
        'conn4_pct': 100 * float(np.mean(c4)),
        'holes': float(np.mean(hol)),
        'dead_pct': 100 * float(np.mean(dead)),
        'all4dead_pct': 100 * float(np.mean(alldead)),
        'fallback_pct': 100.0 * fb / n,
        'unique_cells_per_target': float(np.mean(lens)),
        'bridges_per_target': STATS['bridged'] / max(STATS['calls'], 1),
        'trimmed_per_target': STATS['trimmed'] / max(STATS['calls'], 1),
        'undersized_pct': 100.0 * STATS['short'] / max(STATS['calls'], 1),
        'bridge_failed_pct': 100.0 * STATS['notconn'] / max(STATS['calls'], 1),
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
        ('random_default',  'mirage_envelope', (0.15, 0.2),   0, False),
        ('anatomy_8conn',   'mirage_anatomy',  (0.15, 0.2),  50, False),
        ('anatomy_bridge',  'mirage_anatomy',  (0.15, 0.2),  50, True),
    ]
    print('%d slices, production collator\n' % len(per))
    hdr = ('%-16s %8s %7s %8s %7s %9s %8s %8s %7s' %
           ('arm', 'context', 'hidden', 'anat_hid', 'bg_hid', 'on-anat',
            '8-conn', '4-conn', 'holes'))
    print(hdr); print('-' * len(hdr))
    res = {}
    for tag, mode, scale, ep, br in arms:
        r = arm(mode, scale, ep, per, occ, a.batch, a.k, br)
        res[tag] = r
        print('%-16s %8.1f %7.1f %8.1f %7.1f %8.1f%% %7.1f%% %7.1f%% %7.2f'
              % (tag, r['context'], r['hidden'], r['anat_hidden'], r['bg_hidden'],
                 r['on_anatomy_pct'], r['conn8_pct'], r['conn4_pct'], r['holes']))

    print('\nfailure modes')
    hdr2 = ('%-16s %10s %10s %12s %11s %10s' %
            ('arm', 'dead tgt', 'all4 dead', 'fallback', 'cells/tgt', 'undersized'))
    print(hdr2); print('-' * len(hdr2))
    for tag, *_ in arms:
        r = res[tag]
        print('%-16s %9.2f%% %9.2f%% %11.2f%% %11.2f %9.2f%%'
              % (tag, r['dead_pct'], r['all4dead_pct'], r['fallback_pct'],
                 r['unique_cells_per_target'], r['undersized_pct']))

    b = res['anatomy_bridge']
    print('\nbridge mechanics: %.2f cells added, %.2f trimmed back, '
          'bridge failed on %.2f%% of targets'
          % (b['bridges_per_target'], b['trimmed_per_target'],
             b['bridge_failed_pct']))
    (OUT / 'bridge_exact_k.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'bridge_exact_k.json')


if __name__ == '__main__':
    main()
