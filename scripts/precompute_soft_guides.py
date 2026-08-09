#!/usr/bin/env python3
"""Stage 4: ONE MIRAGE pass over all FairVision slices, cached to disk.

    FairVision slice -> frozen MIRAGE -> frozen cfg-7 adapter
                     -> frozen seg head -> softmax
                     -> P_inner, P_choroid at native 200x200 -> disk

After this, JEPA epochs never touch MIRAGE.  One pass instead of 75:

    live MIRAGE every epoch   75 x ~36 min  = ~45 h
    this                      1  x ~36 min

It also keeps fine-tuning and downstream evaluation cheap, because MIRAGE's
95.6M parameters never need to be resident.

WHY NATIVE 200x200 AND POST-SOFTMAX
The guide is cropped in tandem with the image by PairedRandomResizedCrop and
only then pooled to the 16x16 token grid.  Storing the score AFTER the
softmax and BEFORE any pooling preserves the required softmax-then-pool
ordering through the crop.  Pooling first and softmaxing later is not the
same function, and was measured to change the final masks badly: Jaccard
0.587, 0/200 identical, 40% fewer cells.

WHY NOT CACHE THE FEATURE
The adapter input H0 is (384,64,64) = 3.00 MiB/sample in fp16, so one epoch
of 600,000 slices would be 1.72 TiB.  The post-softmax score is 2 x 200 x 200
uint8 = 80 KB/sample, about 45 GiB for the whole training split, and unlike a
cached feature it can be validly cropped.

The cache records the MIRAGE checkpoint hash AND the adapter hash, so a stale
or mismatched guide fails loudly instead of silently masking the wrong thing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_stage import Adapter, sha                       # noqa: E402
from jepa_to_mirage_probe import build_mirage                # noqa: E402

SCHEMA = 2
NATIVE = 200
RES = 512
ANATOMY = (1, 2)


def slice_grid(num_slices):
    """The exact depth indices OCTSliceDataset samples."""
    return np.linspace(0, 199, num=num_slices, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='Training')
    ap.add_argument('--data-root',
                    default=r'D:\jepa_phase0\fairvision-glaucoma\data')
    ap.add_argument('--out-root',
                    default=r'D:\jepa_phase0\fairvision-glaucoma\mirage_soft_guides')
    ap.add_argument('--adapter', default=None,
                    help='cfg-7 adapter checkpoint; omit for the frozen baseline guide')
    ap.add_argument('--num-slices', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--yes', action='store_true',
                    help='required when the estimate exceeds 10 GiB')
    a = ap.parse_args()

    import cv2
    dev = 'cuda'
    src = pathlib.Path(a.data_root) / a.split
    vols = sorted(src.glob('data_*.npz'))
    if a.limit:
        vols = vols[:a.limit]
    idxs = slice_grid(a.num_slices)
    n_slices = len(vols) * len(idxs)
    per_vol = len(idxs) * 2 * NATIVE * NATIVE
    est_gib = len(vols) * per_vol / 1024 ** 3

    print('split            %s' % a.split)
    print('volumes          %d' % len(vols))
    print('slices/volume    %d   (np.linspace(0,199,%d))' % (len(idxs), a.num_slices))
    print('TOTAL 2D slices  %d' % n_slices)
    print('estimated size   %.2f GiB   (%.0f KB/slice)'
          % (est_gib, per_vol / len(idxs) / 1024))
    free = __import__('shutil').disk_usage(a.out_root[:3]).free / 1024 ** 3
    print('free on target   %.0f GiB' % free)
    if est_gib > 10 and not a.yes:
        print('\nRefusing to write %.1f GiB without --yes.' % est_gib)
        return
    if est_gib > free * 0.9:
        print('\nRefusing: estimate exceeds 90%% of free space.')
        return

    mir = build_mirage(dev)
    assert not any(p.requires_grad for p in mir.parameters()), 'MIRAGE not frozen'
    grab = {}
    head = mir.output_adapters['semseg'].final_layer
    head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))

    adapter = None
    adapter_sha = 'none'
    if a.adapter:
        ck = torch.load(a.adapter, map_location='cpu', weights_only=False)
        adapter = Adapter(**ck['cfg']).to(dev)
        adapter.load_state_dict(ck['state_dict'])
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter_sha = sha(a.adapter)
        print('adapter          %s  (sha %s, taught by %s)'
              % (a.adapter, adapter_sha, ck.get('jepa_ckpt')))
    else:
        print('adapter          NONE - caching the frozen baseline guide')

    tag = 'base512_%s' % ('cfg7_' + adapter_sha if adapter else 'frozen')
    out = pathlib.Path(a.out_root) / tag / a.split
    out.mkdir(parents=True, exist_ok=True)

    from jepa_to_mirage_probe import CK_MIRAGE
    meta = {'schema_version': SCHEMA, 'split': a.split,
            'mirage_ckpt': str(CK_MIRAGE), 'mirage_sha': sha(CK_MIRAGE),
            'adapter_ckpt': str(a.adapter) if a.adapter else None,
            'adapter_sha': adapter_sha,
            'num_slices': int(a.num_slices),
            'slice_indices': idxs.tolist(),
            'native_size': NATIVE, 'mirage_input_res': RES,
            'dtype': 'uint8', 'channels': ['P_inner', 'P_choroid'],
            'ordering': 'softmax BEFORE any pooling or cropping',
            'amp': bool(a.amp), 'n_volumes': len(vols),
            'n_slices_total': int(n_slices),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S')}

    t0 = time.perf_counter()
    written = 0
    for vi, vp in enumerate(vols):
        dst = out / vp.name
        with np.load(vp, allow_pickle=True) as z:
            vol = z['oct_bscans']
        buf = np.zeros((len(idxs), 2, NATIVE, NATIVE), np.uint8)
        for s in range(0, len(idxs), a.batch):
            sl = idxs[s:s + a.batch]
            imgs = np.zeros((len(sl), RES, RES), np.float32)
            for j, d in enumerate(sl):
                raw = np.asarray(vol[int(d)], np.float32)
                lo, hi = raw.min(), raw.max()
                u = (raw - lo) / (hi - lo) if hi > lo else raw * 0
                imgs[j] = cv2.resize(u, (RES, RES), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(imgs)[:, None].to(dev)
            with torch.no_grad():
                if a.amp:
                    with torch.autocast('cuda', dtype=torch.float16):
                        mir({'bscan': x})
                else:
                    mir({'bscan': x})
                H = grab['H'].float()
                if adapter is not None:
                    H = adapter(H)
                # softmax FIRST, at the decoder's own 64x64 resolution
                P = head(H).float().softmax(1)[:, ANATOMY]
                P = F.interpolate(P, size=(NATIVE, NATIVE), mode='bilinear',
                                  align_corners=False)
                buf[s:s + len(sl)] = (P.clamp(0, 1) * 255).round().byte().cpu().numpy()
        np.savez_compressed(
            dst, soft_scores=buf, source_filename=vp.name,
            schema_version=SCHEMA, slice_indices=idxs,
            mirage_sha=meta['mirage_sha'], adapter_sha=adapter_sha)
        written += 1
        if written % 100 == 0:
            el = time.perf_counter() - t0
            print('   %d/%d volumes  %.0fs  eta %.0fs'
                  % (written, len(vols), el, el / written * (len(vols) - written)),
                  flush=True)
    meta['seconds'] = time.perf_counter() - t0
    (out.parent / 'cache_meta.json').write_text(json.dumps(meta, indent=2))
    print('done: %d volumes in %.0fs -> %s' % (written, meta['seconds'], out))


if __name__ == '__main__':
    main()
