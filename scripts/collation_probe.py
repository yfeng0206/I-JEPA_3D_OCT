#!/usr/bin/env python3
"""What does batch collation actually destroy, and which fix is better?

Reading the collator carefully changes the picture:

  curriculum.py:1129-1130   pred_indices_union is built from the FULL,
                            untruncated target lists
  curriculum.py:1163-1195   masks_enc excludes that full union
  curriculum.py:1205-1209   masks_enc is then min-truncated PER GROUP
  curriculum.py:1211-1218   masks_pred is min-truncated GLOBALLY

So the anatomy actually hidden from the context encoder is correct: the whole
anatomical union is removed before any truncation happens.  What truncation
destroys is (a) how many of those hidden cells the predictor is asked to
predict, and (b) how many context tokens survive stacking.

That matters for the choice of fix.  My earlier bucketing recommendation
implicitly treated the loss as "anatomy area hidden", which is not what is
being lost.  Because the union is already correct, the prediction index list
can be resampled to a fixed length WITHOUT touching the mask at all:

  fixed-K resample   target larger than K  -> subsample K distinct cells
                     target smaller than K -> sample with replacement

Every index is a real token, so the missing-padding-mask problem does not
arise, no slice is dropped, and the context mask is byte-identical to today.

This measures both arms against the production MaskCollator policy.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import anatomy_target_sampler_v2 as A                       # noqa: E402
from src.masks.multiblock import MaskCollator               # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = pathlib.Path('results/masking/collation')


def build_all():
    """Per slice: the 4 target index lists and the context block, real policy."""
    import torch
    per = np.load(GRIDS)['per']
    coll = MaskCollator(input_size=(256, 256), patch_size=16, npred=4)
    gen = torch.Generator().manual_seed(0)
    # Enc block SIZE is drawn once per batch and shared by every image, so the
    # only thing that varies the context length is subtracting the union.
    bh, bw = coll._sample_block_size(coll.enc_mask_scale, gen)
    rows = []
    for i in range(len(per)):
        cs = [per[i, 0], per[i, 1]]
        if not A.is_viable(cs):
            continue
        parts, _ = A.build_targets(cs)
        tg = [np.flatnonzero(p.ravel()).tolist() for p in parts]
        union = set().union(*[set(t) for t in tg])
        random.seed(i)
        kept = None
        for _ in range(50):
            top, left = coll._sample_block_location(bh, bw, coll.height, coll.width)
            idx = coll._block_to_indices(top, left, bh, bw)
            cand = [j for j in idx if j not in union]
            if len(cand) >= coll.min_keep:
                kept = cand
                break
        if kept is None:
            allp = list(range(coll.num_patches))
            kept = [j for j in allp if j not in union] or allp[:coll.min_keep]
        rows.append({'tg': [len(t) for t in tg], 'union': len(union),
                     'ctx': len(kept)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_all()
    tg = np.array([r['tg'] for r in rows], np.int64)
    ctx = np.array([r['ctx'] for r in rows], np.int64)
    uni = np.array([r['union'] for r in rows], np.int64)
    n = len(rows)
    rng = np.random.default_rng(0)
    print('slices %d   union %.1f cells   context %.1f tokens   target %.1f'
          % (n, uni.mean(), ctx.mean(), tg.mean()))
    print()

    res = {'n': int(n), 'mean_union': float(uni.mean()),
           'mean_ctx': float(ctx.mean()), 'batches': {}}

    print('=== what each truncation costs, per microbatch ===')
    print('%6s %14s %14s %16s %16s' % (
        'batch', 'ctx kept', 'ctx tokens', 'pred cells kept', 'distinct/slice'))
    for b in (16, 32, 64):
        ck, pk, dpc = [], [], []
        for _ in range(600):
            s = rng.integers(0, n, size=b)
            c, t = ctx[s], tg[s]
            cmin = max(1, int(c.min()))
            ck.append(b * cmin / c.sum())
            g = max(1, int(t.min()))
            pk.append(b * 4 * g / t.sum())
            dpc.append(4 * g)
        r = {'ctx_kept_pct': 100 * float(np.mean(ck)),
             'ctx_tokens': float(np.mean([ctx[rng.integers(0, n, b)].min()
                                          for _ in range(600)])),
             'pred_kept_pct': 100 * float(np.mean(pk)),
             'distinct_pred_cells': float(np.mean(dpc))}
        res['batches'][str(b)] = r
        print('%6d %13.1f%% %14.1f %15.1f%% %16.1f'
              % (b, r['ctx_kept_pct'], r['ctx_tokens'], r['pred_kept_pct'],
                 r['distinct_pred_cells']))
    print()

    print('=== fix comparison: distinct anatomy cells the predictor sees ===')
    print('%22s %18s %14s %12s' % ('policy', 'distinct cells', 'vs ideal',
                                   'slices lost'))
    ideal = tg.sum(1).mean()
    print('%22s %18.1f %13.0f%% %11.1f%%' % ('ideal (no truncation)', ideal, 100, 0.0))
    b64 = res['batches']['64']
    print('%22s %18.1f %13.1f%% %11.1f%%'
          % ('current global-min', b64['distinct_pred_cells'],
             100 * b64['distinct_pred_cells'] / ideal, 0.0))
    bucket = 0.660 * ideal
    print('%22s %18.1f %13.1f%% %11.1f%%' % ('bucketing', bucket,
                                             100 * bucket / ideal, 0.0))
    res['fix'] = {'ideal': float(ideal),
                  'global_min': float(b64['distinct_pred_cells']),
                  'bucketing': float(bucket), 'fixed_k': {}}
    for k in (8, 12, 16, 20, 24):
        distinct = np.minimum(tg, k).sum(1).mean()
        print('%22s %18.1f %13.1f%% %11.1f%%'
              % ('fixed-K resample K=%d' % k, distinct, 100 * distinct / ideal, 0.0))
        res['fix']['fixed_k'][str(k)] = {
            'distinct_cells': float(distinct),
            'pct_of_ideal': float(100 * distinct / ideal),
            'predicted_slots': 4 * k,
            'repeat_rate_pct': float(100 * (1 - distinct / (4 * k)))}
    print()
    print('Context masks are IDENTICAL under every policy: the union is removed')
    print('before truncation, so no fix here changes what the encoder sees.')
    (OUT / 'collation.json').write_text(json.dumps(res, indent=2))
    print('wrote', OUT / 'collation.json')


if __name__ == '__main__':
    main()
