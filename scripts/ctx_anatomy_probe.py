#!/usr/bin/env python3
"""Does the surviving context still contain IMPORTANT (anatomical) region?

Not "is it non-black" -- that was ctx_informative_probe.py, which additionally
hard-coded enc_mask_scale=(0.4,0.5) instead of the real (0.85,1.0) and so
reported a context larger than its own context block. This probe uses the
production MaskCollator policy verbatim.

Arms are PAIRED: identical block sizes and identical RNG stream per image, so
the only difference is which target union gets subtracted from the context.
"""
import sys
import json
import math
import random
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from anatomy_target_sampler_v2 import build_targets  # noqa: E402
from src.masks.multiblock import MaskCollator  # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = pathlib.Path('results/masking/diagnostics')
OUT.mkdir(parents=True, exist_ok=True)

TAU_SUPPORT = 0.10   # MIRAGE "meaningfully belongs to anatomy"
TAU_CONF = 0.50      # confident anatomy
MASS_CAP = 0.80      # sampler default


def sample_arm(coll, pred_sizes, enc_sizes, target_union, seed):
    """Run the collator's real context policy against a given target union."""
    random.seed(seed)
    # Burn the same draws the rect arm uses for target locations so the
    # context block sees an identical RNG position in both arms.
    for bh, bw in pred_sizes:
        coll._sample_block_location(bh, bw, coll.height, coll.width)

    bh, bw = enc_sizes[0]
    retries = 0
    kept = None
    block = None
    for attempt in range(50):
        top, left = coll._sample_block_location(bh, bw, coll.height, coll.width)
        idx = coll._block_to_indices(top, left, bh, bw)
        cand = [i for i in idx if i not in target_union]
        if len(cand) >= coll.min_keep:
            kept, block, retries = cand, idx, attempt
            break
    if kept is None:
        retries = 50
        block = list(range(coll.num_patches))
        kept = [i for i in block if i not in target_union]
        if len(kept) < coll.min_keep:
            kept = block[:coll.min_keep]
    return set(kept), set(block), retries


def rect_targets(coll, pred_sizes, seed):
    random.seed(seed)
    union = set()
    for bh, bw in pred_sizes:
        top, left = coll._sample_block_location(bh, bw, coll.height, coll.width)
        union.update(coll._block_to_indices(top, left, bh, bw))
    return union


def score(ctx, a_flat, inner_flat, chor_flat):
    """Anatomy content of a context set."""
    A = float(a_flat.sum())
    ci = np.array(sorted(ctx), dtype=int)
    if ci.size == 0:
        return dict(n_ctx=0, mass_frac=0.0, n_support=0, n_conf=0,
                    inner_frac=0.0, chor_frac=0.0)
    v = a_flat[ci]
    return dict(
        n_ctx=int(ci.size),
        mass_frac=float(v.sum() / A) if A > 0 else 0.0,
        n_support=int((v > TAU_SUPPORT).sum()),
        n_conf=int((v > TAU_CONF).sum()),
        inner_frac=(float(inner_flat[ci].sum() / inner_flat.sum())
                    if inner_flat.sum() > 0 else float('nan')),
        chor_frac=(float(chor_flat[ci].sum() / chor_flat.sum())
                   if chor_flat.sum() > 0 else float('nan')),
    )


def main():
    per = np.load(GRIDS)['per']  # (N,2,16,16)
    N = per.shape[0]
    coll = MaskCollator()  # production defaults
    g = torch.Generator()

    rows = {'rect': [], 'anat': []}
    n_empty_target = 0
    n_zero_anat = 0

    for i in range(N):
        P_inner, P_chor = per[i, 0], per[i, 1]
        a = P_inner + P_chor
        a_flat, inner_flat, chor_flat = a.ravel(), P_inner.ravel(), P_chor.ravel()
        if a_flat.sum() <= 1e-6:
            n_zero_anat += 1

        seed = 10_000 + i
        g.manual_seed(seed)
        pred_sizes = [coll._sample_block_size(coll.pred_mask_scale, g)
                      for _ in range(coll.npred)]
        enc_sizes = [coll._sample_block_size(coll.enc_mask_scale, g)
                     for _ in range(coll.nenc)]

        u_rect = rect_targets(coll, pred_sizes, seed)

        parts, _ = build_targets([P_inner, P_chor], 4,
                                 mass_cap=MASS_CAP, tau=0.10, overlap=0.0)
        u_anat_mask = np.logical_or.reduce(parts)
        u_anat = set(np.flatnonzero(u_anat_mask.ravel()).tolist())
        if len(u_anat) == 0:
            n_empty_target += 1

        for name, union in (('rect', u_rect), ('anat', u_anat)):
            ctx, block, retries = sample_arm(coll, pred_sizes, enc_sizes,
                                             union, seed)
            r = score(ctx, a_flat, inner_flat, chor_flat)
            r.update(n_block=len(block), n_target=len(union), retries=retries,
                     anat_total=float(a_flat.sum()))
            rows[name].append(r)

        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{N}', flush=True)

    def agg(rs):
        def col(k):
            return np.array([r[k] for r in rs], dtype=float)
        mass, sup, conf = col('mass_frac'), col('n_support'), col('n_conf')
        inner, chor = col('inner_frac'), col('chor_frac')
        return dict(
            n_block=float(col('n_block').mean()),
            n_target=float(col('n_target').mean()),
            n_ctx=float(col('n_ctx').mean()),
            retries=float(col('retries').mean()),
            mass_mean=float(mass.mean()),
            mass_p05=float(np.percentile(mass, 5)),
            mass_p01=float(np.percentile(mass, 1)),
            mass_min=float(mass.min()),
            support_mean=float(sup.mean()),
            support_p05=float(np.percentile(sup, 5)),
            support_min=float(sup.min()),
            conf_mean=float(conf.mean()),
            conf_p05=float(np.percentile(conf, 5)),
            inner_mean=float(np.nanmean(inner)),
            chor_mean=float(np.nanmean(chor)),
            pct_support_0=float((sup == 0).mean() * 100),
            pct_support_lt5=float((sup < 5).mean() * 100),
            pct_mass_lt05=float((mass < 0.05).mean() * 100),
            pct_conf_0=float((conf == 0).mean() * 100),
            pct_inner_0=float(np.nanmean(inner < 1e-6) * 100),
            pct_chor_0=float(np.nanmean(chor < 1e-6) * 100),
        )

    res = {k: agg(v) for k, v in rows.items()}
    res['_meta'] = dict(n=N, zero_anatomy_slices=n_zero_anat,
                        empty_anatomy_targets=n_empty_target,
                        tau_support=TAU_SUPPORT, tau_conf=TAU_CONF)

    (OUT / 'ctx_anatomy.json').write_text(json.dumps(res, indent=2))

    L = [
        ('context block (pre-removal)', 'n_block', '{:.1f}'),
        ('target union', 'n_target', '{:.1f}'),
        ('CONTEXT after removal', 'n_ctx', '{:.1f}'),
        ('block retries', 'retries', '{:.2f}'),
        ('-- IMPORTANT REGION IN CONTEXT --', None, None),
        ('anatomy mass retained', 'mass_mean', '{:.3f}'),
        ('   p05', 'mass_p05', '{:.3f}'),
        ('   p01', 'mass_p01', '{:.3f}'),
        ('   min', 'mass_min', '{:.3f}'),
        ('ctx tokens w/ anatomy >0.10', 'support_mean', '{:.1f}'),
        ('   p05', 'support_p05', '{:.1f}'),
        ('   min', 'support_min', '{:.0f}'),
        ('ctx tokens w/ anatomy >0.50', 'conf_mean', '{:.1f}'),
        ('   p05', 'conf_p05', '{:.1f}'),
        ('-- PER CLASS (mass retained) --', None, None),
        ('inner retina', 'inner_mean', '{:.3f}'),
        ('choroid', 'chor_mean', '{:.3f}'),
        ('-- TAIL RISK (% of slices) --', None, None),
        ('zero anatomy tokens', 'pct_support_0', '{:.2f}%'),
        ('<5 anatomy tokens', 'pct_support_lt5', '{:.2f}%'),
        ('<5% anatomy mass', 'pct_mass_lt05', '{:.2f}%'),
        ('zero CONFIDENT anatomy', 'pct_conf_0', '{:.2f}%'),
        ('zero inner retina', 'pct_inner_0', '{:.2f}%'),
        ('zero choroid', 'pct_chor_0', '{:.2f}%'),
    ]
    print(f"\n{'metric':<32}{'RANDOM rect':>14}{'ANATOMY':>14}")
    print('-' * 60)
    for label, key, fmt in L:
        if key is None:
            print(label)
            continue
        print(f'{label:<32}{fmt.format(res["rect"][key]):>14}'
              f'{fmt.format(res["anat"][key]):>14}')
    print('-' * 60)
    print(f'slices={N}  zero-anatomy slices={n_zero_anat}  '
          f'empty anatomy targets={n_empty_target}')
    print(f'saved -> {OUT / "ctx_anatomy.json"}')


if __name__ == '__main__':
    main()
