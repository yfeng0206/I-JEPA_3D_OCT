#!/usr/bin/env python
"""Reconcile the profiled 13.2 GB with the 17.5 GB the run actually logs.

`train_patch.py` logs `torch.cuda.max_memory_allocated`, which is a running
ALL-TIME maximum that is never reset.  So the number in the CSV is not the
training step's cost -- it is the largest peak anything in the process has
ever reached, and validation is a different, larger shape:

    training    context ~160 tokens, 4 targets x pred_target_k=16 cells
    validation  uniform MaskCollator, pred_mask_scale 0.15-0.2 of 256
                -> 4 targets x ~42 cells, at the same batch size

Both run at batch_size 64.  This measures each shape separately at the real
batch size and reports which one sets the high-water mark.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import torch
import torch.nn.functional as F
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.helper import init_patch_model, init_opt                   # noqa: E402
from src.masks.utils import apply_masks                             # noqa: E402
from src.utils.tensors import repeat_interleave_batch               # noqa: E402

GB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_anatomy_v2.yaml')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(REPO / a.config))
    data_cfg, mask_cfg = cfg['data'], cfg['mask']
    meta_cfg, opt_cfg = cfg['meta'], cfg['optimization']
    crop, patch = data_cfg['crop_size'], mask_cfg['patch_size']
    ntok = (crop // patch) ** 2
    bs = data_cfg['batch_size']
    npred = mask_cfg['num_pred_masks']
    dev = torch.device('cuda')

    enc, pred = init_patch_model(
        device=dev, patch_size=patch, crop_size=crop,
        model_name=meta_cfg['model_name'], pred_depth=meta_cfg['pred_depth'],
        pred_emb_dim=meta_cfg['pred_emb_dim'])
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    optimizer, scaler, _, _ = init_opt(
        encoder=enc, predictor=pred,
        wd=opt_cfg['weight_decay'], final_wd=opt_cfg['final_weight_decay'],
        start_lr=opt_cfg['start_lr'], ref_lr=opt_cfg['lr'],
        final_lr=opt_cfg['final_lr'], iterations_per_epoch=100,
        warmup=opt_cfg['warmup'], num_epochs=opt_cfg['epochs'],
        ipe_scale=opt_cfg.get('ipe_scale', 1.0),
        use_bfloat16=meta_cfg['use_bfloat16'])

    imgs = torch.randn(bs, 3, crop, crop, device=dev)

    def run(nctx, kcells, train):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        menc = [torch.stack([torch.randperm(ntok, device=dev)[:nctx]
                             for _ in range(bs)])]
        mpred = [torch.stack([torch.randperm(ntok, device=dev)[:kcells]
                              for _ in range(bs)]) for _ in range(npred)]
        if train:
            with torch.no_grad():
                h = tgt(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, mpred)
                h = repeat_interleave_batch(h, bs, repeat=len(menc))
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                z = pred(enc(imgs, menc), menc, mpred)
                loss = F.smooth_l1_loss(z, h)
            scaler.scale(loss).backward(); scaler.step(optimizer)
            scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            # evaluate_val(): no autocast, no grad, full fp32
            with torch.no_grad():
                h = tgt(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, mpred)
                h = repeat_interleave_batch(h, bs, repeat=len(menc))
                z = pred(enc(imgs, menc), menc, mpred)
                loss = F.smooth_l1_loss(z, h)
        peak = torch.cuda.max_memory_allocated() / GB
        del menc, mpred, h, z, loss
        return peak

    print('batch %d, %d tokens\n' % (bs, ntok))
    print('%-44s %10s' % ('phase', 'peak GB'))
    print('-' * 56)
    k = int(mask_cfg['pred_target_k'])
    peaks = {}
    for nctx in (160, 180, 200, 220, 240):
        peaks[nctx] = run(nctx, k, True)
        print('%-44s %10.2f' % ('training  ctx %d, 4 x %d cells' % (nctx, k),
                                peaks[nctx]))
    v_small = run(218, 42, False)
    print('%-44s %10.2f' % ('validation ctx 218, 4 x 42 cells (fp32)', v_small))

    hi = max(list(peaks.values()) + [v_small])
    print('\nhigh-water mark %.2f GB' % hi)
    print('enc_mask_scale is %s of %d tokens, so the encoder block is %d-%d '
          'tokens BEFORE the target union is subtracted; the average context '
          'is ~160 but the PEAK is far higher, and max_memory_allocated '
          'records the peak and never resets.'
          % (mask_cfg['enc_mask_scale'], ntok,
             int(mask_cfg['enc_mask_scale'][0] * ntok),
             int(mask_cfg['enc_mask_scale'][1] * ntok)))
    print('logged by the real run: 17.5 GB')


if __name__ == '__main__':
    main()
