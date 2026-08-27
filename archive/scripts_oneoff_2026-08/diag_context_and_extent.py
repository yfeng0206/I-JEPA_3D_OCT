"""Two diagnostics the review asked for, before any sampler rewrite.

D1  CONTEXT POLICY.  The demo reports "context keeps 206/256", which is exactly
    256 - |U|.  That is the COMPLEMENT of the target union, not I-JEPA's context
    policy: the real collator samples a context BLOCK (enc_mask_scale 0.85-1.0,
    one block) and then subtracts the target union, and the audited runs saw
    only 71-87 tokens reach the transformer.  If the anatomy path silently
    switched to "everything except the targets", the encoder sees far more of
    the image than any previous arm, and the comparison is broken.

D2  MASS vs EXTENT.  grow_union stops when

        sum_{i in U} a_i  >=  rho * sum_i a_i

    which accumulates PROBABILITY MASS, not spatial coverage.  Because the
    confident core of the band carries most of the mass, 70% of mass may
    correspond to far less than 70% of the cells MIRAGE considers anatomy.
    Measured here against a support set S = {a_i > tau}.

Neither diagnostic changes any code; both only measure what the current
pipeline does.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.masks.multiblock import MaskCollator                 # noqa: E402
from anatomy_target_sampler import build_targets              # noqa: E402

GRID = 16


def d1_context(n_batches=100, batch=8):
    """What does the REAL I-JEPA collator hand the context encoder?"""
    coll = MaskCollator(input_size=(256, 256), patch_size=16)
    imgs = [torch.randn(3, 256, 256) for _ in range(batch)]
    ctx, tgt_union, blk = [], [], []
    for _ in range(n_batches):
        _, me, mp = coll(imgs)
        for b in range(batch):
            c = set(me[0][b].tolist())
            u = set()
            for m in mp:
                u |= set(m[b].tolist())
            ctx.append(len(c))
            tgt_union.append(len(u))
            blk.append(len(c) + len(u))       # context block before subtraction
    return {
        'real_context_tokens_mean': float(np.mean(ctx)),
        'real_context_tokens_min': int(np.min(ctx)),
        'real_context_tokens_max': int(np.max(ctx)),
        'real_target_union_mean': float(np.mean(tgt_union)),
        'complement_would_be': float(256 - np.mean(tgt_union)),
        'ratio_complement_over_real': float((256 - np.mean(tgt_union)) / np.mean(ctx)),
    }


def d2_mass_vs_extent(grids, rho=0.70, taus=(0.05, 0.10, 0.25, 0.50)):
    """Does a 70% MASS budget cover 70% of the anatomy's spatial EXTENT?"""
    out = {}
    for tau in taus:
        mass, ext, sup, uni = [], [], [], []
        for a in grids:
            A = float(a.sum())
            if A <= 1e-6:
                continue
            parts, U = build_targets(a, rho=rho, overlap=0.24)
            union = np.zeros_like(U)
            for p in parts:
                union |= p
            S = a > tau
            if S.sum() == 0:
                continue
            mass.append(float((a * union).sum() / A))
            ext.append(float((union & S).sum() / S.sum()))
            sup.append(int(S.sum()))
            uni.append(int(union.sum()))
        out['tau%.2f' % tau] = {
            'mass_covered': float(np.mean(mass)),
            'extent_covered': float(np.mean(ext)),
            'support_cells': float(np.mean(sup)),
            'union_cells': float(np.mean(uni)),
            'gap_mass_minus_extent': float(np.mean(mass) - np.mean(ext)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_base.npz'))
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--rho', type=float, default=0.70)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/diagnostics')
    a = ap.parse_args()

    rep = {}
    print('=== D1: what context does the REAL I-JEPA collator produce? ===')
    d1 = d1_context()
    rep['d1_context'] = d1
    print('  real context tokens        : mean %.1f  (min %d, max %d)'
          % (d1['real_context_tokens_mean'], d1['real_context_tokens_min'],
             d1['real_context_tokens_max']))
    print('  real target union          : mean %.1f' % d1['real_target_union_mean'])
    print('  256 - union (the complement): %.1f' % d1['complement_would_be'])
    print('  complement / real          : %.2fx   <-- how much MORE the encoder'
          ' would see' % d1['ratio_complement_over_real'])

    print('\n=== D2: does a %.0f%% MASS budget cover %.0f%% of spatial EXTENT? ==='
          % (100 * a.rho, 100 * a.rho))
    G = np.load(a.grids, allow_pickle=True)['grids'][:a.n]
    d2 = d2_mass_vs_extent(G, a.rho)
    rep['d2_mass_vs_extent'] = d2
    print('  support S = {a_i > tau}    (%d slices)' % len(G))
    print('  %8s %10s %12s %14s %12s %10s'
          % ('tau', '|S| cells', 'union cells', 'mass covered', 'extent cov', 'gap'))
    for k, v in d2.items():
        print('  %8s %10.1f %12.1f %13.3f %12.3f %10.3f'
              % (k.replace('tau', ''), v['support_cells'], v['union_cells'],
                 v['mass_covered'], v['extent_covered'], v['gap_mass_minus_extent']))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'context_and_extent.json').write_text(json.dumps(rep, indent=2))
    print('\nwrote %s' % (a.out / 'context_and_extent.json'))


if __name__ == '__main__':
    main()
