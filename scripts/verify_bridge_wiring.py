#!/usr/bin/env python
"""Verify diagonal bridging through the CONFIG FLAG, with no monkeypatching.

`bridge_exact_k.py` proved the idea by swapping `curriculum.anatomy_shrink_to_k`
at runtime.  This script instead sets `anatomy_bridge_diagonals` in the
curriculum config, exactly as `configs/patch_anatomy_v2.yaml` does, so what is
measured here is the code path that will actually run in training.

Also times the sampler, because the run is I/O-bound with the GPU at ~50% duty
cycle: any CPU added inside the DataLoader workers comes straight off the
loader's throughput budget.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.masks.anatomy import CROSS4                               # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator           # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/fair'
G = 16
BOX = np.ones((3, 3))


def arm(per, occ_all, batch, k, bridge, mode='mirage_anatomy',
        scale=(0.15, 0.2), epoch=50, seed=0):
    cfg = {'mode': mode, 'enabled': True, 'T_warm': 25, 'T_total': 30,
           'r_max': 1.0, 'ramp_shape': 'linear',
           'anatomy_mass_cap': 0.90, 'anatomy_tau': 0.10,
           'anatomy_bridge_diagonals': bridge}
    gen = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16, npred=4, nenc=1,
        pred_mask_scale=scale,
        pred_target_k=k if mode == 'mirage_anatomy' else None,
        curriculum_cfg=cfg)
    gen.set_epoch(epoch, 100)
    assert gen.anatomy_bridge_diagonals is bridge

    n = len(per)
    ctx, hid, on, c4, c8, hol, dead, alldead = [], [], [], [], [], [], [], []
    anat_h, bg_h, lens = [], [], []
    fb, elapsed = 0, 0.0
    for s in range(0, n, batch):
        b = min(batch, n - s)
        random.seed(seed + s); torch.manual_seed(seed + s); np.random.seed(seed + s)
        occ_b = occ_all[s:s + b]
        t = torch.from_numpy(occ_b).float()
        guides = torch.stack([t, (t >= 0.25).float()], 1)
        t0 = time.perf_counter()
        enc, pred = gen.generate(batch_size=b, guide_grids=guides,
                                 guide_valid=torch.ones(b, dtype=torch.bool))
        elapsed += time.perf_counter() - t0
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
                c4.append(ndimage.label(M, structure=CROSS4)[1] == 1)
                hol.append(int((ndimage.binary_fill_holes(M) & ~M).sum()))
                dd = int(o[idx].mean() < 0.10)
                dead.append(dd); d += dd
            alldead.append(int(d == 4))
            idx = np.array(sorted(u))
            hid.append(len(idx)); on.append(float(o[idx].mean()))
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
        'ms_per_image': 1000.0 * elapsed / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)
    print('%d slices, batch %d, config flag only (no monkeypatching)\n'
          % (len(per), a.batch))

    res = {}
    res['anatomy_bridge_off'] = arm(per, occ, a.batch, a.k, False)
    res['anatomy_bridge_on'] = arm(per, occ, a.batch, a.k, True)

    hdr = ('%-22s %8s %7s %8s %7s %9s %8s %8s %7s' %
           ('arm', 'context', 'hidden', 'anat_hid', 'bg_hid', 'on-anat',
            '8-conn', '4-conn', 'holes'))
    print(hdr); print('-' * len(hdr))
    for tag, r in res.items():
        print('%-22s %8.1f %7.1f %8.1f %7.1f %8.1f%% %7.1f%% %7.1f%% %7.2f'
              % (tag, r['context'], r['hidden'], r['anat_hidden'],
                 r['bg_hidden'], r['on_anatomy_pct'], r['conn8_pct'],
                 r['conn4_pct'], r['holes']))

    print('\n%-22s %10s %10s %11s %11s %12s' %
          ('arm', 'dead tgt', 'all4 dead', 'fallback', 'cells/tgt', 'ms/image'))
    print('-' * 82)
    for tag, r in res.items():
        print('%-22s %9.2f%% %9.2f%% %10.2f%% %11.2f %12.3f'
              % (tag, r['dead_pct'], r['all4dead_pct'], r['fallback_pct'],
                 r['unique_cells_per_target'], r['ms_per_image']))

    off, on_ = res['anatomy_bridge_off'], res['anatomy_bridge_on']
    over = on_['ms_per_image'] - off['ms_per_image']
    print('\nsampler overhead: %+.3f ms/image (%+.1f%%)'
          % (over, 100 * over / off['ms_per_image']))
    print('at 117 images/s that is %+.1f%% of one core' % (100 * over * 117 / 1000))
    (OUT / 'bridge_wiring_verified.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'bridge_wiring_verified.json')


if __name__ == '__main__':
    main()
