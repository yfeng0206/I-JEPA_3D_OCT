#!/usr/bin/env python
"""Find the dataloader settings that stop starving the GPU.

The run was measured at 50.4% GPU duty cycle in a square wave: ~2.4 s busy,
~2.4 s idle. CPU sat at 21% of 16 logical cores and physical disk read at
9-11 MB/s, so the loop was I/O bound on the HDD, not compute bound.

This sweeps worker count and prefetch depth against the REAL dataset and the
REAL collator, and reports samples/s plus peak RAM. It runs no model, so it is
safe to sweep aggressively and the numbers isolate the input pipeline.

Peak RAM matters: workers were measured at ~2.2 GB RSS each, so 12 workers
could approach the 32 GB limit. Windows surfaces exhaustion as
"Couldn't open shared file mapping ... error code 1455", which has killed a run
in this project before.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import MirageMaskCollator               # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402


def build(cfg, guide_root, cache_root):
    d, m = cfg['data'], cfg['mask']
    cur = m['curriculum']
    tf = make_paired_transforms(
        crop_size=d['crop_size'],
        crop_scale=tuple(d.get('crop_scale', (0.3, 1.0))),
        gaussian_blur=d.get('use_gaussian_blur', False),
        horizontal_flip=d.get('use_horizontal_flip', False),
        color_distortion=d.get('use_color_distortion', False),
        color_jitter=d.get('color_jitter_strength', 0.0),
    )
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(d['data_dir'], 'Training'),
        guide_dir=os.path.join(guide_root, 'Training'),
        num_slices=d['num_slices'], slice_size=d['crop_size'],
        transform=tf,
        patch_size=m['patch_size'],
        dilate_patches=int(cur.get('mirage_dilate_patches', 0)),
        occupancy_threshold=float(cur.get('mirage_occupancy_threshold', 0.25)),
        slice_cache=os.path.join(cache_root, 'Training'),
        require_guides=True,
    )
    coll = MirageMaskCollator(
        input_size=(d['crop_size'], d['crop_size']),
        patch_size=m['patch_size'],
        enc_mask_scale=tuple(m['enc_mask_scale']),
        pred_mask_scale=tuple(m['pred_mask_scale']),
        aspect_ratio=tuple(m['aspect_ratio']),
        nenc=m['num_enc_masks'], npred=m['num_pred_masks'],
        min_keep=m.get('min_keep', 10), allow_overlap=m.get('allow_overlap', False),
        pred_target_k=m.get('pred_target_k'),
        curriculum_cfg=cur,
    )
    coll.set_epoch(50, cfg['optimization']['epochs'])
    return ds, coll


def bench(ds, coll, batch, workers, prefetch, n_batches):
    kw = {}
    if workers > 0:
        kw['prefetch_factor'] = prefetch
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, collate_fn=coll, **kw)
    it = iter(dl)
    for _ in range(3):            # warm up workers and page cache
        next(it)
    proc = psutil.Process()
    peak = 0.0
    t0 = time.perf_counter()
    for _ in range(n_batches):
        next(it)
        rss = sum(c.memory_info().rss for c in proc.children(recursive=True))
        peak = max(peak, (rss + proc.memory_info().rss) / (1 << 30))
    dt = time.perf_counter() - t0
    del it, dl
    return batch * n_batches / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_anatomy_v2.yaml')
    ap.add_argument('--guide-root', required=True)
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--workers', default='6,8,12')
    ap.add_argument('--prefetch', default='2,4')
    ap.add_argument('--batches', type=int, default=25)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    batch = cfg['data']['batch_size']

    print('guides %s' % a.guide_root)
    print('cache  %s' % a.cache_root)
    ds, coll = build(cfg, a.guide_root, a.cache_root)
    print('dataset %d slices, batch %d\n' % (len(ds), batch))

    # A GPU step was measured at ~0.546 s for batch 64 => this is the rate the
    # loader must beat for the GPU to stop waiting.
    need = batch / 0.546
    print('loader must exceed %.0f samples/s to saturate the GPU\n' % need)
    print('%8s %9s %12s %10s %9s' %
          ('workers', 'prefetch', 'samples/s', 'peak RAM', 'verdict'))
    print('-' * 54)
    best = (0, None)
    for w in [int(x) for x in a.workers.split(',')]:
        for p in [int(x) for x in a.prefetch.split(',')]:
            if w == 0 and p != 2:
                continue
            try:
                rate, ram = bench(ds, coll, batch, w, p, a.batches)
            except Exception as exc:
                print('%8d %9d %12s %10s %9s' % (w, p, 'FAILED', '-', str(exc)[:30]))
                continue
            verdict = 'saturates' if rate >= need else 'starves'
            print('%8d %9d %12.0f %9.1fG %9s' % (w, p, rate, ram, verdict))
            if rate > best[0]:
                best = (rate, (w, p, ram))
    if best[1]:
        w, p, ram = best[1]
        print('\nbest: workers=%d prefetch=%d  %.0f samples/s  peak RAM %.1f GB'
              % (w, p, best[0], ram))
        print('projected epoch: %.0f s (%.2f h)'
              % (600000 / best[0], 600000 / best[0] / 3600))


if __name__ == '__main__':
    main()
