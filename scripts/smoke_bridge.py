#!/usr/bin/env python
"""End-to-end smoke test for `anatomy_bridge_diagonals`, on real data.

Builds the SAME dataset, paired transform and `MirageMaskCollator` that
`train_patch.py` builds from the config, loads the epoch-28 weights the run
will resume from, and pushes real batches through both maskings.

The two arms see byte-identical images and identical RNG state, so any loss
difference is attributable to the mask distribution alone.

Forward-only under `no_grad`: the point is to confirm the path runs and to
measure the loss shift, not to train.  That keeps VRAM to a few GB so it can
share the GPU with whatever else is running.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset      # noqa: E402
from src.helper import init_patch_model                               # noqa: E402
from src.masks.anatomy import CROSS4                                  # noqa: E402
from src.masks.curriculum import MirageMaskCollator                   # noqa: E402
from src.masks.utils import apply_masks                               # noqa: E402
from src.transforms import make_paired_transforms                     # noqa: E402
from src.utils.tensors import repeat_interleave_batch                 # noqa: E402

OUT = REPO / 'results/masking/fair'


def _identity(batch):
    """Module-level so it pickles to spawn workers; a lambda does not."""
    return batch


def rss_gb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def sys_free_gb():
    import psutil
    return psutil.virtual_memory().available / 1024 ** 3


def mask_stats(masks_pred, grid, occ=None):
    c4, c8, cells, union, overlap = [], [], [], [], []
    B = masks_pred[0].shape[0]
    for j in range(B):
        counts = np.zeros(grid * grid, np.int32)
        for p in masks_pred:
            idx = np.unique(p[j].numpy())
            counts[idx] += 1
            m = np.zeros(grid * grid, bool); m[idx] = True
            M = m.reshape(grid, grid)
            c4.append(ndimage.label(M, structure=CROSS4)[1] == 1)
            c8.append(ndimage.label(M, structure=np.ones((3, 3)))[1] == 1)
            cells.append(len(idx))
        union.append(int((counts > 0).sum()))
        # cells claimed by more than one target: bridging must not inflate this
        overlap.append(int((counts > 1).sum()))
    return {'conn4': 100 * float(np.mean(c4)), 'conn8': 100 * float(np.mean(c8)),
            'cells': float(np.mean(cells)), 'union': float(np.mean(union)),
            'overlap': float(np.mean(overlap))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_anatomy_v2.yaml')
    ap.add_argument('--batches', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=0,
                    help='0 = use the config value')
    ap.add_argument('--workers', type=int, default=4,
                    help='kept below the training value to limit RAM')
    ap.add_argument('--epochs-to-test', type=int, nargs='*', default=[28, 50])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(REPO / a.config))
    data_cfg, mask_cfg = cfg['data'], cfg['mask']
    meta_cfg, opt_cfg = cfg['meta'], cfg['optimization']
    curr = mask_cfg['curriculum']
    crop = data_cfg['crop_size']
    grid = crop // mask_cfg['patch_size']
    bs = a.batch_size or data_cfg['batch_size']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('config            %s' % a.config)
    print('device            %s' % dev)
    print('batch/workers     %d / %d  (training uses %d workers)'
          % (bs, a.workers, data_cfg['num_workers']))
    print('bridge in config  %s\n' % curr.get('anatomy_bridge_diagonals'))

    # ---- data: exactly what train_patch builds ---------------------------
    paired = make_paired_transforms(
        crop_size=crop, crop_scale=tuple(data_cfg.get('crop_scale', (0.3, 1.0))),
        gaussian_blur=data_cfg.get('use_gaussian_blur', False),
        horizontal_flip=data_cfg.get('use_horizontal_flip', False),
        color_distortion=data_cfg.get('use_color_distortion', False),
        color_jitter=data_cfg.get('color_jitter_strength', 0.0))
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(data_cfg['data_dir'], 'Training'),
        guide_dir=os.path.join(curr['mirage_guide_dir'], 'Training'),
        num_slices=data_cfg['num_slices'], slice_size=crop, transform=paired,
        patch_size=mask_cfg['patch_size'],
        dilate_patches=int(curr.get('mirage_dilate_patches', 1)),
        occupancy_threshold=float(curr.get('mirage_occupancy_threshold', 0.5)),
        slice_cache=os.path.join(data_cfg['slice_cache_dir'], 'Training'))
    print('dataset           %d slices, %d volumes'
          % (len(ds), len(ds.file_paths)))

    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=a.workers,
                        pin_memory=False, drop_last=True,
                        collate_fn=_identity,
                        **({'prefetch_factor': 2} if a.workers > 0 else {}))

    def make_collator(bridge):
        c = copy.deepcopy(curr)
        c['anatomy_bridge_diagonals'] = bridge
        return MirageMaskCollator(
            input_size=(crop, crop), patch_size=mask_cfg['patch_size'],
            enc_mask_scale=tuple(mask_cfg['enc_mask_scale']),
            pred_mask_scale=tuple(mask_cfg['pred_mask_scale']),
            aspect_ratio=tuple(mask_cfg['aspect_ratio']),
            nenc=mask_cfg['num_enc_masks'], npred=mask_cfg['num_pred_masks'],
            min_keep=mask_cfg['min_keep'], allow_overlap=mask_cfg['allow_overlap'],
            pred_target_k=mask_cfg.get('pred_target_k'), curriculum_cfg=c)

    # ---- model: the weights the run will resume from ---------------------
    enc, pred = init_patch_model(
        device=dev, patch_size=mask_cfg['patch_size'], crop_size=crop,
        model_name=meta_cfg['model_name'], pred_depth=meta_cfg['pred_depth'],
        pred_emb_dim=meta_cfg['pred_emb_dim'])
    tgt = copy.deepcopy(enc)
    ck = torch.load(meta_cfg['read_checkpoint'], map_location='cpu',
                    weights_only=False)
    enc.load_state_dict(ck['encoder']); pred.load_state_dict(ck['predictor'])
    tgt.load_state_dict(ck['target_encoder'])
    for m in (enc, pred, tgt):
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
    print('checkpoint        epoch %s  (%s)\n'
          % (ck.get('epoch'), os.path.basename(meta_cfg['read_checkpoint'])))

    # ---- pull the batches ONCE so both arms see identical images ---------
    print('fetching %d batches ...' % a.batches)
    t0 = time.time()
    raw = []
    it = iter(loader)
    for _ in range(a.batches):
        raw.append(next(it))
    fetch = time.time() - t0
    print('  %.1fs for %d x %d = %.1f samples/s (loader needs 117 with %d workers)\n'
          % (fetch, a.batches, bs, a.batches * bs / fetch, data_cfg['num_workers']))
    del it, loader

    results = {}
    peak_rss, peak_vram = rss_gb(), 0.0
    for ep in a.epochs_to_test:
        row = {}
        for bridge in (False, True):
            col = make_collator(bridge)
            col.set_epoch(ep, opt_cfg['epochs'])
            losses, ms, ctxs = [], [], []
            for bi, batch in enumerate(raw):
                # identical RNG for both arms -> only the sampler differs
                random.seed(1234 + bi); np.random.seed(1234 + bi)
                torch.manual_seed(1234 + bi)
                imgs, m_enc, m_pred, stats = col(batch)
                ms.append(mask_stats(m_pred, grid))
                ctxs.append(int(m_enc[0].shape[1]))
                imgs = imgs.to(dev, non_blocking=True)
                me = [m.to(dev) for m in m_enc]
                mp = [m.to(dev) for m in m_pred]
                with torch.no_grad():
                    h = tgt(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, mp)
                    h = repeat_interleave_batch(h, imgs.size(0), repeat=len(me))
                    z = pred(enc(imgs, me), me, mp)
                    losses.append(F.smooth_l1_loss(z, h).item())
                peak_rss = max(peak_rss, rss_gb())
                if dev.type == 'cuda':
                    peak_vram = max(peak_vram,
                                    torch.cuda.max_memory_allocated() / 1024 ** 3)
            row['on' if bridge else 'off'] = {
                'loss': float(np.mean(losses)), 'loss_std': float(np.std(losses)),
                'conn4_pct': float(np.mean([x['conn4'] for x in ms])),
                'conn8_pct': float(np.mean([x['conn8'] for x in ms])),
                'cells_per_target': float(np.mean([x['cells'] for x in ms])),
                'hidden_union': float(np.mean([x['union'] for x in ms])),
                'overlap_cells': float(np.mean([x['overlap'] for x in ms])),
                'context_tokens': float(np.mean(ctxs)),
                'r_t': float(col._get_generator()._r_t),
            }
            del col
        results['epoch_%d' % ep] = row

    print('%-7s %-8s %6s %10s %8s %8s %9s %8s %8s %7s'
          % ('epoch', 'bridge', 'r_t', 'loss', '4-conn', '8-conn',
             'cells/tgt', 'union', 'overlap', 'context'))
    print('-' * 92)
    for ep in a.epochs_to_test:
        r = results['epoch_%d' % ep]
        for k in ('off', 'on'):
            v = r[k]
            print('%-7d %-8s %6.2f %10.6f %7.1f%% %7.1f%% %9.2f %8.1f %8.2f %7.1f'
                  % (ep, k, v['r_t'], v['loss'], v['conn4_pct'], v['conn8_pct'],
                     v['cells_per_target'], v['hidden_union'],
                     v['overlap_cells'], v['context_tokens']))
        d = r['on']['loss'] - r['off']['loss']
        print('%-7s %-8s %6s %+10.6f  (%+.2f%%)\n'
              % ('', 'delta', '', d, 100 * d / r['off']['loss']))

    print('resources: peak RSS %.1f GB, peak VRAM %.1f GB, system free %.1f GB'
          % (peak_rss, peak_vram, sys_free_gb()))
    results['_resources'] = {'peak_rss_gb': peak_rss, 'peak_vram_gb': peak_vram,
                             'batches': a.batches, 'batch_size': bs}
    (OUT / 'smoke_bridge.json').write_text(json.dumps(results, indent=2))
    print('wrote', OUT / 'smoke_bridge.json')


if __name__ == '__main__':
    main()
