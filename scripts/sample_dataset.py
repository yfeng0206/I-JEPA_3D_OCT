#!/usr/bin/env python
"""Pull random B-scans from any candidate dataset and open them for viewing.

FairVision gives 200 CONSECUTIVE slices per volume.  The public segmentation
datasets give only sparse annotated slices (Duke DME: 11 per subject from 10
subjects; AROI: 19-103 per patient from 23 patients), so "a volume" is not
comparable across them.  This pulls a random sample from whichever dataset you
name and writes a single panel, at TRUE pixel scale by default so resolution
differences are visible rather than hidden by resizing.

    python scripts/sample_dataset.py --dataset fairvision --n 6 --open
    python scripts/sample_dataset.py --dataset aroi --n 6 --open
    python scripts/sample_dataset.py --dataset duke_dme --n 6 --open
    python scripts/sample_dataset.py --dataset goals --n 6 --open
    python scripts/sample_dataset.py --dataset all --n 3 --open
"""

from __future__ import annotations

import argparse
import io
import os
import random
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ZIPS = {
    'goals': r'D:\jepa_phase0\mirage-goals\downloads\GOALS.zip',
    'duke_dme': r'D:\jepa_phase0\mirage-datasets\Duke_DME.zip',
    'aroi': r'D:\jepa_phase0\mirage-datasets\AROI.zip',
}
FAIRVISION = r'D:\jepa_phase0\fairvision-glaucoma\data\Training'
MIRAGE_MASKS = r'D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training'

PALETTE = [(240, 92, 92), (250, 190, 70), (120, 150, 255), (128, 194, 100),
           (220, 120, 220), (90, 220, 220), (250, 140, 60), (180, 180, 180),
           (140, 100, 200)]


def _font(size, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def sample_zip(key, n, rng):
    """Random (name, bscan, semseg) triples from a MIRAGE-format zip."""
    path = ZIPS[key]
    if not os.path.isfile(path):
        return []
    out = []
    with zipfile.ZipFile(path) as z:
        bs = sorted(x for x in z.namelist() if '/bscan/' in x and x.endswith('.png'))
        for b in rng.sample(bs, min(n, len(bs))):
            s = b.replace('/bscan/', '/semseg/')
            img = np.array(Image.open(io.BytesIO(z.read(b))).convert('L'))
            seg = (np.array(Image.open(io.BytesIO(z.read(s))).convert('L'))
                   if s in z.namelist() else np.zeros_like(img))
            out.append((b.split('/')[-1], img, seg))
    return out


def sample_fairvision(n, rng):
    """Random volume + random slice, with the MIRAGE mask we generated."""
    files = sorted(f for f in os.listdir(FAIRVISION) if f.endswith('.npz'))
    out = []
    tries = 0
    while len(out) < n and tries < n * 12:
        tries += 1
        f = rng.choice(files)
        mp = os.path.join(MIRAGE_MASKS, f)
        if not os.path.isfile(mp):
            continue
        with np.load(os.path.join(FAIRVISION, f), allow_pickle=True) as z:
            vol = z['oct_bscans']
        with np.load(mp, allow_pickle=False) as z:
            hard, idx = z['hard_masks'], z['slice_indices']
        k = rng.randrange(len(idx))
        depth = int(idx[k])
        out.append(('%s  slice %d/200' % (f.replace('.npz', ''), depth),
                    np.array(vol[depth]), hard[k] * 80))
    return out


def build(items, title, scale, out_path):
    if not items:
        raise SystemExit('no samples found for %s' % title)
    tile_w = max(i[1].shape[1] for i in items)
    tile_h = max(i[1].shape[0] for i in items)
    tw, th = int(tile_w * scale), int(tile_h * scale)
    gap, head, lab_h = 8, 66, 20
    cols = min(len(items), 3)
    rows = (len(items) + cols - 1) // cols
    W = 16 + cols * (tw * 2 + gap * 2)
    H = head + rows * (th + lab_h + gap)
    fig = Image.new('RGB', (W, H), (14, 14, 16))
    d = ImageDraw.Draw(fig)
    d.text((14, 12), title, font=_font(17, True), fill=(245, 245, 245))
    d.text((14, 38), 'left = B-scan, right = label mask.  Shown at %.2fx true pixel scale '
                     '(no upscaling, so resolution differences are real).' % scale,
           font=_font(12), fill=(176, 176, 182))

    for i, (name, img, seg) in enumerate(items):
        r, c = divmod(i, cols)
        x = 14 + c * (tw * 2 + gap * 2)
        y = head + r * (th + lab_h + gap)
        bi = Image.fromarray(img).convert('RGB')
        bi = bi.resize((int(img.shape[1] * scale), int(img.shape[0] * scale)),
                       Image.NEAREST if scale >= 1 else Image.BILINEAR)
        sm = np.zeros(seg.shape + (3,), np.uint8)
        for j, v in enumerate([v for v in np.unique(seg) if v != 0]):
            sm[seg == v] = PALETTE[j % len(PALETTE)]
        si = Image.fromarray(sm).resize(bi.size, Image.NEAREST)
        blend = (np.array(bi).astype(np.float32) * 0.45
                 + np.array(si).astype(np.float32) * 0.55).astype(np.uint8)
        fig.paste(bi, (x, y))
        fig.paste(Image.fromarray(blend), (x + bi.size[0] + gap, y))
        d.text((x, y + bi.size[1] + 3),
               '%s   %dx%d' % (name, img.shape[0], img.shape[1]),
               font=_font(11), fill=(200, 200, 208))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fairvision',
                    choices=['fairvision', 'goals', 'duke_dme', 'aroi', 'all'])
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--scale', type=float, default=1.0,
                    help='1.0 = true pixel size. Use 2 to magnify FairVision.')
    ap.add_argument('--out', default=None)
    ap.add_argument('--open', action='store_true', help='open in the default viewer')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    targets = (['fairvision', 'goals', 'duke_dme', 'aroi']
               if args.dataset == 'all' else [args.dataset])
    written = []
    for t in targets:
        items = sample_fairvision(args.n, rng) if t == 'fairvision' else sample_zip(t, args.n, rng)
        if not items:
            print('%-11s not available locally' % t)
            continue
        out = args.out or os.path.join('results', 'masking', 'sample_%s.png' % t)
        if args.dataset == 'all':
            out = os.path.join('results', 'masking', 'sample_%s.png' % t)
        p = build(items, 'Random sample - %s' % t.upper(), args.scale, out)
        h, w = items[0][1].shape
        print('%-11s %d samples  %dx%d  -> %s' % (t, len(items), h, w, p))
        written.append(p)

    if args.open:
        for p in written:
            try:
                os.startfile(os.path.abspath(p))
            except Exception:
                subprocess.run(['explorer', os.path.abspath(p)], check=False)


if __name__ == '__main__':
    main()
