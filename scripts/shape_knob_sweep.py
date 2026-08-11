#!/usr/bin/env python
"""Recover anatomy coverage under 4-connected growth.

Switching growth from an 8- to a 4-neighbourhood fixes the scatter (targets
edge-connected 51% -> 91%, holes 0.06 -> 0.00) but costs coverage:

    arm              hidden  anat_hid  on-anat  4-conn
    anatomy_8conn      54.2      39.3    72.1%   51.3%
    anatomy_4conn      51.5      34.9    67.5%   91.3%

The retina is a thin diagonal band, and a 4-neighbourhood has to stair-step
along it, so growth stops earlier and captures less tissue.

This sweeps the shape knobs to find a 4-connected setting that gets back to
roughly 8-conn coverage:

    max_hole   holes the grower is allowed to close (already exists in
               `fill_small_holes`, currently 2)
    mass_cap   fraction of class mass growth tries to cover
    tau        occupancy above which a cell counts as anatomy support

Target: hidden ~54, anat_hid >=39, on-anat ~72%, 4-conn >=90%.
Filling holes deliberately admits some interior background -- acceptable,
since a solid shape is the point.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.masks.anatomy as AN                                     # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/fair'
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
NB4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def evaluate(occ_all, nb, mass_cap, tau, max_hole, k=16, fill=False):
    AN.NB8[:] = nb
    hid, anat, on, c4, hol = [], [], [], [], []
    for occ in occ_all:
        parts, _ = AN.build_targets([occ, occ * 0], n=4, mass_cap=mass_cap,
                                    tau=tau, max_hole=max_hole)
        u = np.zeros(occ.shape, bool)
        for p in parts:
            if p.sum() == 0:
                continue
            m = AN.shrink_to_k(p, k, score=occ)
            if fill:
                # Close interior holes so the target is a solid shape. Only
                # cells fully enclosed by the target are added, so this cannot
                # grow the outer boundary.
                filled = ndimage.binary_fill_holes(m)
                if filled.sum() <= k + 2:
                    m = filled
            c4.append(ndimage.label(m, structure=CROSS)[1] == 1)
            hol.append(int((ndimage.binary_fill_holes(m) & ~m).sum()))
            u |= m
        idx = np.nonzero(u.ravel())[0]
        if len(idx) == 0:
            continue
        o = occ.ravel()
        hid.append(len(idx))
        anat.append(float(o[idx].sum()))
        on.append(float(o[idx].mean()))
    return (float(np.mean(hid)), float(np.mean(anat)), 100 * float(np.mean(on)),
            100 * float(np.mean(c4)), float(np.mean(hol)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)[:a.n]

    print('%d slices\n' % len(occ))
    hdr = ('%-46s %7s %9s %8s %8s %6s' %
           ('setting', 'hidden', 'anat_hid', 'on-anat', '4-conn', 'holes'))
    print(hdr); print('-' * len(hdr))
    rows = {}

    h, an, on, c4, hl = evaluate(occ, NB8, 0.90, 0.10, 2)
    print('%-46s %7.1f %9.1f %7.1f%% %7.1f%% %6.2f'
          % ('NB8 baseline (current production)', h, an, on, c4, hl))
    rows['nb8_baseline'] = dict(hidden=h, anat=an, on=on, c4=c4, holes=hl)
    target_anat, target_on = an, on

    combos = list(itertools.product([2, 4, 8], [0.90, 0.95], [0.10, 0.05], [False, True]))
    best = None
    for mh, mc, tau, fill in combos:
        h, an, on, c4, hl = evaluate(occ, NB4, mc, tau, mh, fill=fill)
        tag = 'NB4 max_hole=%d mass_cap=%.2f tau=%.2f%s' % (
            mh, mc, tau, ' +fill' if fill else '')
        print('%-46s %7.1f %9.1f %7.1f%% %7.1f%% %6.2f' % (tag, h, an, on, c4, hl))
        rows[tag] = dict(hidden=h, anat=an, on=on, c4=c4, holes=hl)
        if c4 >= 88.0:
            # closest to matching 8-conn coverage while staying connected
            score = abs(an - target_anat) / target_anat + abs(on - target_on) / target_on
            if best is None or score < best[0]:
                best = (score, tag, (h, an, on, c4, hl))
    if best:
        _, tag, (h, an, on, c4, hl) = best
        print('\nbest 4-connected match to the NB8 baseline:')
        print('  %s' % tag)
        print('  hidden %.1f (vs %.1f)   anat_hid %.1f (vs %.1f)   on-anat %.1f%% (vs %.1f%%)'
              % (h, rows['nb8_baseline']['hidden'], an, target_anat, on, target_on))
        print('  4-conn %.1f%%   holes %.2f' % (c4, hl))
    (OUT / 'shape_knob_sweep.json').write_text(json.dumps(rows, indent=2))
    print('\nwrote', OUT / 'shape_knob_sweep.json')


if __name__ == '__main__':
    main()
