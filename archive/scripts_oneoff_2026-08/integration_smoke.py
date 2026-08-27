#!/usr/bin/env python3
"""End-to-end integration smoke test: does the whole thing actually run?

Everything measured so far has been offline probes on cached tensors.  Not
one real training step has ever been executed with irregular anatomy
targets, and that is exactly the blind spot that let the collation bug hide
-- every mask metric looked healthy while 92.8% of the signal was being
discarded downstream of the mask.

This runs the REAL components: init_patch_model, the EMA target encoder,
apply_masks, repeat_interleave_batch and the same smooth_l1 loss as
src/train_patch.py, over real FairVision B-scans, and logs every stage.

Three arms, all on identical images and identical guides:

  rect_default   production mirage_envelope rectangles, pred_mask_scale as
                 configured (this is what actually trained ep100)
  rect_matched   same rectangles, masking ratio lowered to match anatomy
  anatomy        irregular connected targets + fixed-K resampling

rect_matched exists because swapping the sampler changes masked AREA by
2.13x as well as target shape.  Without it, any difference confounds better
targeting with an easier task.
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
import torch.nn.functional as F
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                        # noqa: E402
from src.helper import init_patch_model                      # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator     # noqa: E402
from src.masks.utils import apply_masks, resample_to_k       # noqa: E402
from src.utils.tensors import repeat_interleave_batch        # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUT = REPO / 'results/masking/integration'
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = 16


def anatomy_masks(scores, k, npred=4, min_keep=10, rng=None):
    """Irregular targets -> the collator contract, with fixed-K resampling.

    `is_viable` is NOT optional.  About 1.9% of slices cannot fill 4 targets
    -- MIRAGE finds too little anatomy -- and without the gate the first such
    slice raises out of resample_to_k.  That is the correct failure mode
    (loud, not silent), but production has to route those samples somewhere,
    so they fall back to random blocks exactly as the collator does.
    """
    rng = rng or np.random.default_rng(0)
    B = len(scores)
    enc, pred = [[]], [[] for _ in range(npred)]
    stats = {'cells': [], 'on_anat': [], 'comps': [], 'ctx': [], 'distinct': [],
             'fallback': 0}
    for b in range(B):
        cs = [scores[b, 0], scores[b, 1]]
        if A.is_viable(cs):
            parts, _ = A.build_targets(cs)
            comps = int(max(ndimage.label(p_, structure=np.ones((3, 3)))[1]
                            for p_ in parts))
        else:
            stats['fallback'] += 1
            parts = []
            for _ in range(npred):
                m = np.zeros((GRID, GRID), bool)
                r0 = rng.integers(0, GRID - 4)
                c0 = rng.integers(0, GRID - 4)
                m[r0:r0 + 4, c0:c0 + 4] = True
                parts.append(m)
            comps = 1
        union = np.logical_or.reduce(parts)
        # Context = complement of the FULL union, exactly as the collator does.
        ctx = np.flatnonzero(~union.ravel())
        if len(ctx) < min_keep:
            ctx = np.arange(min_keep)
        enc[0].append(torch.from_numpy(ctx).long())
        seen = set()
        for p in range(npred):
            fl = np.flatnonzero(parts[p].ravel())
            idx = resample_to_k(torch.from_numpy(fl).long(), k)
            pred[p].append(idx)
            seen.update(idx.tolist())
        anat = (np.asarray(cs[0]) + np.asarray(cs[1])).ravel()
        stats['cells'].append(int(union.sum()))
        stats['on_anat'].append(float((anat[np.flatnonzero(union.ravel())] > 0.5).mean()))
        stats['comps'].append(comps)
        stats['ctx'].append(len(ctx))
        stats['distinct'].append(len(seen))
    # Context lengths vary; stack on the batch minimum as the collator does.
    m = min(t.numel() for t in enc[0])
    enc = [torch.stack([t[:m] for t in enc[0]], 0)]
    pred = [torch.stack(g, 0) for g in pred]
    return enc, pred, stats


def rect_masks(gen, guides, B):
    enc, pred = gen.generate(
        batch_size=B, guide_grids=guides,
        guide_valid=torch.ones(B, dtype=torch.bool))
    stats = {'cells': [], 'on_anat': [], 'comps': [], 'ctx': [], 'distinct': []}
    env = guides.numpy().reshape(B, -1)
    for b in range(B):
        u = set()
        for p in pred:
            u.update(p[b].numpy().tolist())
        idx = np.array(sorted(u))
        stats['cells'].append(len(u))
        stats['on_anat'].append(float(env[b][idx].mean()) if len(idx) else 0.0)
        stats['comps'].append(1)
        stats['ctx'].append(int(enc[0].shape[1]))
        stats['distinct'].append(len(u))
    return enc, pred, stats


def run_arm(tag, steps, B, k, dev, im256, scores, guides, lr, log):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    encoder, predictor = init_patch_model(device=dev, patch_size=16, crop_size=256)
    import copy
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    gen = None
    if tag.startswith('rect'):
        cfg = {'mode': 'mirage_envelope', 'enabled': True}
        scale = (0.15, 0.2) if tag == 'rect_default' else (0.055, 0.075)
        gen = CurriculumMaskGenerator(
            input_size=(256, 256), patch_size=16, npred=4, nenc=1,
            pred_mask_scale=scale, curriculum_cfg=cfg)

    agg = {kk: [] for kk in ('cells', 'on_anat', 'comps', 'ctx', 'distinct')}
    losses, gnorms = [], []
    nfallback = 0
    t0 = time.perf_counter()
    for s in range(steps):
        sel = np.arange(s * B, (s + 1) * B) % len(im256)
        b = im256[sel].astype(np.float32) / 255.
        rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
        imgs = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(dev)

        if gen is not None:
            enc_m, pred_m, st = rect_masks(gen, guides[sel], B)
        else:
            enc_m, pred_m, st = anatomy_masks(scores[sel], k)
            nfallback += st['fallback']
        for kk in agg:
            agg[kk].extend(st[kk])
        enc_m = [t.to(dev) for t in enc_m]
        pred_m = [t.to(dev) for t in pred_m]

        with torch.no_grad():
            h = target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            h_pred = apply_masks(h, pred_m)
            h_rep = repeat_interleave_batch(h_pred, B, repeat=len(enc_m))
        z = encoder(imgs, enc_m)
        z = predictor(z, enc_m, pred_m)
        assert z.shape == h_rep.shape, (tag, z.shape, h_rep.shape)
        loss = F.smooth_l1_loss(z, h_rep)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 10.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), '%s: non-finite loss at step %d' % (tag, s)
        losses.append(float(loss)); gnorms.append(float(gn))
        if s == 0:
            log(' %-13s z %s  h_rep %s  ctx %s  pred %s'
                % (tag, tuple(z.shape), tuple(h_rep.shape),
                   tuple(enc_m[0].shape), tuple(pred_m[0].shape)))
    dt = time.perf_counter() - t0
    r = {'tag': tag, 'steps': steps, 'batch': B,
         'loss_first': losses[0], 'loss_last': losses[-1],
         'loss_mean': float(np.mean(losses)),
         'loss_drop_pct': 100 * (losses[0] - losses[-1]) / losses[0],
         'all_finite': bool(np.all(np.isfinite(losses))),
         'grad_norm_mean': float(np.mean(gnorms)),
         'cells_hidden': float(np.mean(agg['cells'])),
         'on_anatomy_pct': 100 * float(np.mean(agg['on_anat'])),
         'components_max': int(np.max(agg['comps'])),
         'context_tokens': float(np.mean(agg['ctx'])),
         'distinct_target_cells': float(np.mean(agg['distinct'])),
         'fallback_pct': 100.0 * nfallback / (steps * B),
         'sec_per_step': dt / steps,
         'vram_mb': torch.cuda.max_memory_allocated() / 1024 ** 2}
    del encoder, predictor, target_encoder, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')
    per = np.load(GRIDS)['per']
    n = min(len(per), a.steps * a.batch)
    scores = per[:n]
    guides = torch.from_numpy(((per[:n, 0] + per[:n, 1]) > 0.5).astype(np.float32))
    im256 = np.asarray(im256[:n])

    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)

    log('integration smoke: %d steps x batch %d on real FairVision B-scans'
        % (a.steps, a.batch))
    log('')
    res = []
    for tag in ('rect_default', 'rect_matched', 'anatomy'):
        res.append(run_arm(tag, a.steps, a.batch, a.k, dev, im256, scores,
                           guides, a.lr, log))
    log('')
    hdr = ('%-14s %8s %8s %9s %8s %10s %9s %8s %8s %7s'
           % ('arm', 'hidden', 'on-anat', 'context', 'to-loss', 'loss drop',
              'grad', 'fallback', 's/step', 'VRAM'))
    log(hdr); log('-' * len(hdr))
    for r in res:
        log('%-14s %8.1f %7.1f%% %9.1f %8.1f %9.1f%% %9.2f %7.1f%% %8.2f %6.0fM'
            % (r['tag'], r['cells_hidden'], r['on_anatomy_pct'],
               r['context_tokens'], r['distinct_target_cells'],
               r['loss_drop_pct'], r['grad_norm_mean'], r['fallback_pct'],
               r['sec_per_step'], r['vram_mb']))
    log('')
    log('all losses finite: %s' % all(r['all_finite'] for r in res))
    log('max connected components per target: %s'
        % {r['tag']: r['components_max'] for r in res})
    (OUT / 'integration.json').write_text(json.dumps(res, indent=2))
    (OUT / 'integration.log').write_text('\n'.join(lines))
    print('wrote', OUT / 'integration.json')


if __name__ == '__main__':
    main()
