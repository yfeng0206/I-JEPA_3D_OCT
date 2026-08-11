#!/usr/bin/env python
"""Keep 8-connected growth, then BRIDGE the diagonal steps.

The dilemma measured in `shape_knob_sweep.py`:

  * NB8 growth hugs the thin diagonal retinal band, so 16 cells capture a lot
    of tissue (on-anat 72.6%) -- but the result is a diagonal chain, only 49%
    edge-connected.
  * NB4 growth produces solid shapes (90% edge-connected) but has to
    stair-step along the band, so the same 16 cells capture less tissue
    (on-anat 63-68%). No combination of max_hole / mass_cap / tau recovers it.

Both properties are obtainable: grow with NB8 for coverage, then add the
minimum cells needed to turn every corner-touch into an edge-touch. For a
diagonal pair (r,c)-(r+1,c+1) either (r+1,c) or (r,c+1) bridges it; the one
with higher anatomy occupancy is chosen, so bridging prefers tissue.

This is the user's "if we are patching to make a good shape it is fine to fill
in the hole", applied to diagonal steps rather than enclosed holes.
"""
from __future__ import annotations

import argparse
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


def bridge_diagonals(m, score, max_add=None):
    """Add the fewest cells that make `m` 4-connected.

    Only cells that complete an existing corner-touch are added, so the target
    cannot grow beyond the footprint its 8-connected form already occupied.
    """
    out = np.asarray(m, bool).copy()
    h, w = out.shape
    added = 0
    for _ in range(4):                       # a bridge can expose another
        if ndimage.label(out, structure=CROSS)[1] == 1:
            break
        cand = {}
        for r in range(h - 1):
            for c in range(w - 1):
                # the two diagonal orientations inside each 2x2 block
                for (a, b), (x, y) in ((((r, c), (r + 1, c + 1)),
                                        ((r + 1, c), (r, c + 1))),
                                       (((r, c + 1), (r + 1, c)),
                                        ((r, c), (r + 1, c + 1)))):
                    if out[a] and out[b] and not out[x] and not out[y]:
                        for cell in (x, y):
                            cand[cell] = max(cand.get(cell, -1.0), float(score[cell]))
        if not cand:
            break
        # take the highest-occupancy bridge first
        for cell, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
            if max_add is not None and added >= max_add:
                break
            out[cell] = True
            added += 1
            if ndimage.label(out, structure=CROSS)[1] == 1:
                break
    return out, added


def evaluate(occ_all, nb, k, bridge, mass_cap=0.90, tau=0.10, max_hole=2):
    AN.NB8[:] = nb
    hid, anat, on, c4, hol, add = [], [], [], [], [], []
    for occ in occ_all:
        parts, _ = AN.build_targets([occ, occ * 0], n=4, mass_cap=mass_cap,
                                    tau=tau, max_hole=max_hole)
        u = np.zeros(occ.shape, bool)
        for p in parts:
            if p.sum() == 0:
                continue
            m = AN.shrink_to_k(p, k, score=occ)
            if bridge:
                m, n_add = bridge_diagonals(m, occ)
                add.append(n_add)
            c4.append(ndimage.label(m, structure=CROSS)[1] == 1)
            hol.append(int((ndimage.binary_fill_holes(m) & ~m).sum()))
            u |= m
        idx = np.nonzero(u.ravel())[0]
        if len(idx) == 0:
            continue
        o = occ.ravel()
        hid.append(len(idx)); anat.append(float(o[idx].sum()))
        on.append(float(o[idx].mean()))
    return dict(hidden=float(np.mean(hid)), anat=float(np.mean(anat)),
                on=100 * float(np.mean(on)), c4=100 * float(np.mean(c4)),
                holes=float(np.mean(hol)),
                bridges=float(np.mean(add)) if add else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)[:a.n]
    print('%d slices\n' % len(occ))

    hdr = ('%-40s %7s %9s %8s %8s %6s %8s' %
           ('setting', 'hidden', 'anat_hid', 'on-anat', '4-conn', 'holes', 'bridges'))
    print(hdr); print('-' * len(hdr))
    rows = {}
    for tag, nb, k, br in (
            ('NB8 k=16 (current production)', NB8, 16, False),
            ('NB4 k=16', NB4, 16, False),
            ('NB8 k=16 + bridge', NB8, 16, True),
            ('NB8 k=14 + bridge', NB8, 14, True),
            ('NB8 k=13 + bridge', NB8, 13, True),
            ('NB8 k=12 + bridge', NB8, 12, True)):
        r = evaluate(occ, nb, k, br)
        rows[tag] = r
        print('%-40s %7.1f %9.1f %7.1f%% %7.1f%% %6.2f %8.2f'
              % (tag, r['hidden'], r['anat'], r['on'], r['c4'], r['holes'],
                 r['bridges']))
    (OUT / 'bridge_sweep.json').write_text(json.dumps(rows, indent=2))
    print('\nwrote', OUT / 'bridge_sweep.json')


if __name__ == '__main__':
    main()
