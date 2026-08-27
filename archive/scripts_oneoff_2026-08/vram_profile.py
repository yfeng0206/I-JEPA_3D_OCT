#!/usr/bin/env python
"""Where the pretraining VRAM actually goes.

MIRAGE is NOT part of this: the guides are precomputed to disk and
`train_patch.py` never imports or instantiates a segmentation model.  It reads
`.npy` files in the DataLoader workers, which costs host RAM, not VRAM.

Measures the real cost at small batches and extrapolates, so it can run
alongside other GPU work instead of reserving the full training footprint.
Activation memory is linear in batch size; parameters, gradients and optimizer
state are not, so the two are reported separately.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
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


def mb(x):
    return x / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_anatomy_v2.yaml')
    ap.add_argument('--batches', type=int, nargs='*', default=[2, 4, 8, 16])
    a = ap.parse_args()

    cfg = yaml.safe_load(open(REPO / a.config))
    data_cfg, mask_cfg = cfg['data'], cfg['mask']
    meta_cfg, opt_cfg = cfg['meta'], cfg['optimization']
    crop = data_cfg['crop_size']
    patch = mask_cfg['patch_size']
    ntok = (crop // patch) ** 2
    k = int(mask_cfg['pred_target_k'])
    npred = mask_cfg['num_pred_masks']
    target_bs = data_cfg['batch_size']
    dev = torch.device('cuda')

    print('crop %d, patch %d -> %d tokens | target batch %d, accum %d '
          '(effective %d)' % (crop, patch, ntok, target_bs,
                              opt_cfg['accum_steps'],
                              target_bs * opt_cfg['accum_steps']))
    print('use_bfloat16=%s -> GradScaler %s, autocast fp16 %s\n'
          % (meta_cfg['use_bfloat16'], 'off' if meta_cfg['use_bfloat16'] else 'ON',
             'off' if meta_cfg['use_bfloat16'] else 'ON'))

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    enc, pred = init_patch_model(
        device=dev, patch_size=patch, crop_size=crop,
        model_name=meta_cfg['model_name'], pred_depth=meta_cfg['pred_depth'],
        pred_emb_dim=meta_cfg['pred_emb_dim'])
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    after_models = torch.cuda.memory_allocated()

    n_enc = sum(p.numel() for p in enc.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    n_tgt = sum(p.numel() for p in tgt.parameters())
    print('%-34s %12s %10s' % ('component', 'params', 'VRAM MB'))
    print('-' * 58)
    print('%-34s %12s %10.1f' % ('encoder (online, trainable)',
                                 '{:,}'.format(n_enc), mb(n_enc * 4)))
    print('%-34s %12s %10.1f' % ('predictor (trainable)',
                                 '{:,}'.format(n_pred), mb(n_pred * 4)))
    print('%-34s %12s %10.1f' % ('target_encoder (EMA, frozen)',
                                 '{:,}'.format(n_tgt), mb(n_tgt * 4)))
    print('%-34s %12s %10.1f' % ('MIRAGE segmentation model', '0 (not loaded)', 0.0))
    print('%-34s %12s %10.1f\n' % ('  measured weights total', '',
                                   mb(after_models - base)))

    optimizer, scaler, lr_sched, wd_sched = init_opt(
        encoder=enc, predictor=pred,
        wd=opt_cfg['weight_decay'], final_wd=opt_cfg['final_weight_decay'],
        start_lr=opt_cfg['start_lr'], ref_lr=opt_cfg['lr'],
        final_lr=opt_cfg['final_lr'], iterations_per_epoch=100,
        warmup=opt_cfg['warmup'], num_epochs=opt_cfg['epochs'],
        ipe_scale=opt_cfg.get('ipe_scale', 1.0),
        use_bfloat16=meta_cfg['use_bfloat16'])

    def step(bs):
        """One real fwd/bwd/optimizer step at batch `bs`."""
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        imgs = torch.randn(bs, 3, crop, crop, device=dev)
        # context ~160 of 256 tokens, matching the measured production value
        menc = [torch.stack([torch.randperm(ntok, device=dev)[:160]
                             for _ in range(bs)])]
        mpred = [torch.stack([torch.randperm(ntok, device=dev)[:k]
                              for _ in range(bs)]) for _ in range(npred)]
        with torch.no_grad():
            h = tgt(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, mpred)
            h = repeat_interleave_batch(h, bs, repeat=len(menc))
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            z = pred(enc(imgs, menc), menc, mpred)
            loss = F.smooth_l1_loss(z, h)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        res = torch.cuda.max_memory_reserved()
        del imgs, menc, mpred, h, z, loss
        return peak, res

    step(2)                       # warm-up: materialises optimizer state
    steady = torch.cuda.memory_allocated()
    print('%-34s %10.2f GB' % ('weights + grads + Adam state', steady / GB))

    print('\n%-8s %11s %11s %13s %13s' % ('batch', 'peak GB', 'reserved GB',
                                          'activations', 'GB per sample'))
    print('-' * 62)
    xs, ys, rs = [], [], []
    for bs in a.batches:
        peak, res = step(bs)
        act = (peak - steady) / GB
        xs.append(bs); ys.append(act); rs.append(res / GB)
        print('%-8d %11.2f %11.2f %13.2f %13.4f'
              % (bs, peak / GB, res / GB, act, act / bs))

    slope, intercept = np.polyfit(xs, ys, 1)
    est = steady / GB + slope * target_bs + intercept
    overhead = float(np.mean([r / ((p) / GB) for r, p in
                              zip(rs, [steady + y * GB for y in ys])]))
    print('\nactivations = %.4f GB/sample x batch + %.2f GB' % (slope, intercept))
    print('extrapolated to batch %d: %.1f GB weights+state + %.1f GB activations'
          ' = %.1f GB allocated' % (target_bs, steady / GB,
                                    slope * target_bs + intercept, est))
    print('caching allocator reserves ~%.2fx allocated -> ~%.1f GB shown by '
          'nvidia-smi' % (overhead, est * overhead))
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print('device has %.1f GB\n' % total)

    print('what would actually shrink it (same effective batch %d):'
          % (target_bs * opt_cfg['accum_steps']))
    for bs in (16, 32, 64):
        accum = target_bs * opt_cfg['accum_steps'] // bs
        e = steady / GB + slope * bs + intercept
        print('  batch %-3d accum %-3d -> ~%.1f GB   (%.0f%% of current)'
              % (bs, accum, e, 100 * e / est))


if __name__ == '__main__':
    main()
