"""Stage 2: replay the REAL guided sampler on 512 vs 1024 occupancy grids.

Reads the grids dumped by ``sampler_equivalence_dump.py`` and drives the actual
production sampler -- ``CurriculumMaskGenerator._sample_mirage_blocks`` -- so
the comparison exercises the same admissibility, spread, overlap and accept
logic the training run would use.  Nothing is reimplemented.

The comparison is strictly paired: for every slice both arms get the same
pre-drawn block sizes, the same ``biased_flags``, the same ``fixed_uniform``
fallbacks and the same global RNG seed, so any difference in the accepted
geometry is attributable to the occupancy grid alone.

Reported per arm:
  accepted            fraction of slices whose four blocks pass the
                      retina-visible accept test
  feasible            fraction where every block found an admissible window
  mean_block_fill     mean fraction of each placed block that is on region
  retina_visible      fraction of anatomy cells left OUTSIDE the four blocks
  unique targets      |union of the four blocks| in cells (A' changes this,
                      so it is logged explicitly)
  purity              fraction of union cells that are truly anatomy
  A' purity           the same after intersecting each window with anatomy

Plus the paired agreement between arms: identical block top-left positions,
and IoU of the target unions.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

CURRICULUM_CFG = {
    'mode': 'mirage_envelope',
    'T_warm': 25, 'T_total': 30, 'r_max': 1.0, 'ramp_shape': 'linear',
    'mirage_dilate_patches': 0,
    'mirage_min_block_fill': 0.40,
    'mirage_min_retina_visible': 0.25,
    'mirage_max_attempts': 30,
    'mirage_occupancy_threshold': 0.25,
    'mirage_spread': True,
    'mirage_overlap_tolerance': 0.25,
}
THRESH = CURRICULUM_CFG['mirage_occupancy_threshold']


def make_gen() -> CurriculumMaskGenerator:
    gen = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False, curriculum_cfg=dict(CURRICULUM_CFG))
    gen.set_epoch(99, 100)          # r_t = 1.0, fully guided
    return gen


def run_arm(gen, grids: np.ndarray, plans: list, thresh: float = None,
            ref_truth: np.ndarray = None) -> dict:
    """Drive the production sampler over every slice with a fixed plan.

    ``ref_truth`` is an ARM-INDEPENDENT boolean anatomy reference used to score
    purity and visibility.  Without it those two metrics are circular: they
    would be measured against each arm's own thresholded grid, so lowering the
    threshold would inflate purity for free.
    """
    thresh = THRESH if thresh is None else thresh
    prev = gen.mirage_occupancy_threshold
    gen.mirage_occupancy_threshold = thresh
    acc = {k: [] for k in ('accepted', 'feasible', 'fill', 'visible',
                           'unique', 'purity', 'purity_ap', 'nblocks_guided')}
    unions, tops = [], []
    for i, (gi, (sizes, flags, fixed, seed)) in enumerate(zip(grids, plans)):
        occ = torch.from_numpy(gi)
        placement = (occ >= thresh)
        random.seed(seed)
        np.random.seed(seed)
        blocks, stats = gen._sample_mirage_blocks(
            sizes, occ, placement, flags, fixed)
        truth = ((gi >= thresh) if ref_truth is None
                 else ref_truth[i]).reshape(-1)
        union = sorted(set().union(*[set(b) for b in blocks])) if blocks else []
        u = np.array(union, dtype=int)
        acc['accepted'].append(float(stats['accepted']))
        acc['feasible'].append(float(stats['feasible']))
        acc['fill'].append(stats['mean_block_fill'])
        tc = int(truth.sum())
        masked = int(truth[u].sum()) if u.size else 0
        acc['visible'].append((tc - masked) / tc if tc > 0 else 1.0)
        acc['unique'].append(len(u))
        acc['nblocks_guided'].append(stats['guided_blocks'])
        acc['purity'].append(float(truth[u].mean()) if u.size else 0.0)
        # A': window INTERSECT anatomy -> zero background target cells by
        # construction, but a smaller and variable target area.
        keep = u[truth[u]] if u.size else u
        acc['purity_ap'].append(1.0 if keep.size else 0.0)
        unions.append(set(union))
        tops.append(tuple(b[0] for b in blocks))
        acc.setdefault('unique_ap', []).append(int(keep.size))
    gen.mirage_occupancy_threshold = prev
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    return out, unions, tops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()

    z = np.load(a.grids, allow_pickle=True)
    g1024, g512 = z['g1024'], z['g512']
    n = len(g1024)
    print('slices: %d   mean occupancy  1024 %.4f   512 %.4f'
          % (n, g1024.mean(), g512.mean()))

    gen = make_gen()
    # One shared plan per slice: identical block sizes / flags / seeds per arm.
    rng = np.random.default_rng(a.seed)
    plans = []
    for i in range(n):
        random.seed(a.seed + i)
        tg = torch.Generator()
        tg.manual_seed(a.seed + i)
        sizes = [gen._sample_block_size(gen.pred_mask_scale, tg)
                 for _ in range(gen.npred)]
        flags = [True] * gen.npred
        fixed = [None] * gen.npred
        plans.append((sizes, flags, fixed, int(rng.integers(2 ** 31))))

    REF = (g1024 >= THRESH)
    r1024, u1024, t1024 = run_arm(gen, g1024, plans, ref_truth=REF)
    r512, u512, t512 = run_arm(gen, g512, plans, ref_truth=REF)

    # Calibrate: 512 is systematically less confident, so a threshold tuned at
    # 1024 cuts away more region.  Find t* making the mean admissible AREA
    # match, which is the quantity the sampler's feasibility actually depends
    # on, and re-run.  If this recovers equivalence the 512 deficit is a
    # calibration artefact, not a capability difference.
    target_area = float((g1024 >= THRESH).mean())
    lo, hi = 0.01, 0.99
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float((g512 >= mid).mean()) > target_area:
            lo = mid
        else:
            hi = mid
    tstar = 0.5 * (lo + hi)
    print('calibrated 512 threshold: %.4f  (area %.4f vs 1024 %.4f at 0.25)'
          % (tstar, float((g512 >= tstar).mean()), target_area))
    rcal, ucal, tcal = run_arm(gen, g512, plans, thresh=tstar, ref_truth=REF)

    keys = ['accepted', 'feasible', 'fill', 'visible', 'unique', 'unique_ap',
            'purity', 'nblocks_guided']
    label = {'accepted': 'accept rate', 'feasible': 'feasible rate',
             'fill': 'mean block fill', 'visible': 'retina visible',
             'unique': 'unique target cells', 'unique_ap': "unique cells (A')",
             'purity': 'target purity', 'nblocks_guided': 'guided blocks /4'}
    print('\n%-22s %10s %10s %10s %12s' % ('metric', '1024', '512@0.25',
                                           '512@cal', 'cal-1024'))
    for k in keys:
        print('%-22s %10.4f %10.4f %10.4f %+12.4f'
              % (label[k], r1024[k], r512[k], rcal[k], rcal[k] - r1024[k]))

    for tag, uu, tt in (('512@0.25', u512, t512), ('512@cal', ucal, tcal)):
        same_top = np.mean([a == b for a, b in zip(t1024, tt)])
        ious = [len(x & y) / max(len(x | y), 1) for x, y in zip(u1024, uu)]
        print('\nPAIRED AGREEMENT vs 1024  [%s]' % tag)
        print('  identical block positions : %.3f of slices' % same_top)
        print('  target-union IoU          : mean %.4f  p10 %.4f'
              % (np.mean(ious), np.percentile(ious, 10)))
    cell = np.mean([np.mean((g1024[i] >= THRESH) == (g512[i] >= tstar))
                    for i in range(n)])
    print('  boolean placement agree   : %.4f of the 256 cells (calibrated)'
          % cell)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
