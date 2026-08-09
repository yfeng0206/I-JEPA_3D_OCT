#!/usr/bin/env python3
"""Score MIRAGE (with or without the cfg-7 adapter) against GOALS ground truth.

Everything measured so far about the adapter has been a CHANGE metric -- drift,
mask Jaccard, agreement with the frozen output -- because FairVision has no
anatomy labels.  GOALS does.  This is the first time the adapter can be scored
for ACCURACY rather than movement, which makes it usable as a selection metric
for hyperparameters.

Evaluation set: the 30 GOALS images in MergedV3/test.  The 55 GOALS images in
MergedV3/train are NOT used -- MIRAGE was trained on them, so scoring there
would measure memorisation.  val is also excluded as it was available for model
selection.

Preprocessing matches scripts/infer_seg_model.py exactly: per-image min-max to
[0,1], bilinear resize, no ImageNet normalisation, and the void channel (index
3) suppressed before argmax, which is what MIRAGE's own evaluation loop does.

Classes: 0 Elsewhere, 1 InnerRetina, 2 Choroid, 3 Background/void.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_stage import Adapter                            # noqa: E402
from jepa_to_mirage_probe import build_mirage                # noqa: E402

GOALS = pathlib.Path(r'D:\jepa_phase0\mirage-datasets\MergedV3\test')
OUT = REPO / 'results/masking/goals_eval'
RES = 512
VOID = 3
PALETTE = {0: 0, 128: 1, 255: 2}
NAMES = {0: 'Elsewhere', 1: 'InnerRetina', 2: 'Choroid'}


def load_pairs(limit=0):
    import cv2
    bs = sorted((GOALS / 'bscan').glob('*.png'))
    ms = sorted((GOALS / 'semseg').glob('*.png'))
    assert [b.name for b in bs] == [m.name for m in ms], 'bscan/semseg mismatch'
    if limit:
        bs, ms = bs[:limit], ms[:limit]
    imgs, gts, names = [], [], []
    for b, m in zip(bs, ms):
        arr = np.asarray(Image.open(b), np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        unit = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        imgs.append(cv2.resize(unit, (RES, RES), interpolation=cv2.INTER_LINEAR))
        raw = np.asarray(Image.open(m))
        gt = np.zeros_like(raw, np.uint8)
        for v, c in PALETTE.items():
            gt[raw == v] = c
        # NEAREST for labels: bilinear would invent classes that do not exist.
        gts.append(cv2.resize(gt, (RES, RES), interpolation=cv2.INTER_NEAREST))
        names.append(b.name)
    return np.stack(imgs), np.stack(gts), names


def dice_iou(pred, gt, cls):
    p, g = pred == cls, gt == cls
    inter = np.logical_and(p, g).sum()
    ps, gs = p.sum(), g.sum()
    union = np.logical_or(p, g).sum()
    if ps + gs == 0:
        return None, None
    return 2.0 * inter / (ps + gs), (inter / union if union else 0.0)


def predict(mir, head, adapter, imgs, batch, dev, amp=True):
    grab = {}
    h = head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))
    out = []
    try:
        for s in range(0, len(imgs), batch):
            x = torch.from_numpy(imgs[s:s + batch])[:, None].to(dev)
            with torch.no_grad():
                if amp:
                    with torch.autocast('cuda', dtype=torch.float16):
                        mir({'bscan': x})
                else:
                    mir({'bscan': x})
                H = grab['H'].float()
                if adapter is not None:
                    H = adapter(H)
                logits = head(H).float().clone()
                # The head emits 4x64x64; ConvNeXtAdapter's own forward then
                # bilinearly upsamples to image_size. Verified exact: bilinear
                # upsampling head(H) reproduces model['semseg'] with max |diff|
                # 0.0. Doing it here keeps the adapter in the path while
                # matching MIRAGE's native output resolution.
                logits = F.interpolate(logits, size=(RES, RES),
                                       mode='bilinear', align_corners=False)
                logits[:, VOID] = float('-inf')     # MIRAGE's own eval does this
                out.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    finally:
        h.remove()
    return np.concatenate(out)


def score(pred, gts):
    res = {}
    for c, nm in NAMES.items():
        ds, js = [], []
        for i in range(len(gts)):
            d, j = dice_iou(pred[i], gts[i], c)
            if d is not None:
                ds.append(d); js.append(j)
        res[nm] = {'dice': float(np.mean(ds)) if ds else None,
                   'iou': float(np.mean(js)) if js else None,
                   'n': len(ds)}
    an = [v['dice'] for k, v in res.items() if k in ('InnerRetina', 'Choroid')
          and v['dice'] is not None]
    res['anatomy_mean_dice'] = float(np.mean(an)) if an else None
    res['pixel_acc'] = float(np.mean([(pred[i] == gts[i]).mean()
                                      for i in range(len(gts))]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=None, action='append',
                    help='adapter checkpoint; repeatable. Omit for frozen only.')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='goals_eval')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'

    imgs, gts, names = load_pairs(a.limit)
    print('GOALS held-out test: %d images at %dx%d' % (len(imgs), RES, RES))
    print('(MergedV3/train GOALS images excluded - MIRAGE trained on them)')
    print()

    mir = build_mirage(dev)
    head = mir.output_adapters['semseg'].final_layer

    runs = [('frozen MIRAGE', None)]
    for p in (a.adapter or []):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        m = Adapter(**ck['cfg']).to(dev)
        m.load_state_dict(ck['state_dict'])
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
        runs.append((pathlib.Path(p).stem, m))

    hdr = '%-32s %11s %11s %11s %11s' % ('model', 'inner Dice', 'chor Dice',
                                         'anat mean', 'pixel acc')
    print(hdr); print('-' * len(hdr))
    allres, base = {}, None
    for nm, mod in runs:
        pred = predict(mir, head, mod, imgs, a.batch, dev)
        r = score(pred, gts)
        allres[nm] = r
        if base is None:
            base = r
        d = ''
        if r is not base and r['anatomy_mean_dice'] and base['anatomy_mean_dice']:
            d = '  %+.4f' % (r['anatomy_mean_dice'] - base['anatomy_mean_dice'])
        print('%-32s %11.4f %11.4f %11.4f %11.4f%s'
              % (nm, r['InnerRetina']['dice'], r['Choroid']['dice'],
                 r['anatomy_mean_dice'], r['pixel_acc'], d))
    (OUT / ('%s.json' % a.tag)).write_text(json.dumps(
        {'n_images': len(imgs), 'images': names, 'results': allres}, indent=2))
    print()
    print('wrote', OUT / ('%s.json' % a.tag))


if __name__ == '__main__':
    main()
