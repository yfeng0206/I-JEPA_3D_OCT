#!/usr/bin/env python
"""Visual and statistical comparison of candidate OCT segmentation datasets.

Question being answered: do GOALS, Duke DME, AROI and our FairVision target
actually look alike?  If they do, combining them to fine-tune the segmentation
model is reasonable.  If they differ in scan geometry, contrast or sampling,
naive combination will not transfer.

Renders one row per dataset (B-scan, its label mask, and an intensity profile)
and prints geometry/intensity statistics side by side.

    python scripts/dataset_visual_compare.py
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ZIPS = r'D:\jepa_phase0\mirage-datasets'
GOALS_ZIP = r'D:\jepa_phase0\mirage-goals\downloads\GOALS.zip'
FAIRVISION = r'D:\jepa_phase0\fairvision-glaucoma\data\Training'
MIRAGE_MASKS = r'D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training'


def _font(size, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def from_zip(path, want=3):
    """Return (bscan, semseg) uint8 pairs from a MIRAGE-format dataset zip."""
    out = []
    with zipfile.ZipFile(path) as z:
        bs = sorted(x for x in z.namelist() if '/bscan/' in x and x.endswith('.png'))
        for b in bs[:: max(1, len(bs) // (want + 1))][:want]:
            s = b.replace('/bscan/', '/semseg/')
            if s not in z.namelist():
                continue
            img = np.array(Image.open(io.BytesIO(z.read(b))).convert('L'))
            seg = np.array(Image.open(io.BytesIO(z.read(s))).convert('L'))
            out.append((img, seg))
    return out


def from_fairvision(want=3):
    """FairVision slices with the MIRAGE hard mask we generated for them."""
    files = sorted(f for f in os.listdir(FAIRVISION) if f.endswith('.npz'))[:40]
    out = []
    for f in files:
        mp = os.path.join(MIRAGE_MASKS, f)
        if not os.path.isfile(mp):
            continue
        with np.load(os.path.join(FAIRVISION, f), allow_pickle=True) as z:
            vol = z['oct_bscans']
        with np.load(mp, allow_pickle=False) as z:
            hard = z['hard_masks']
            idx = z['slice_indices']
        k = len(idx) // 2
        out.append((np.array(vol[int(idx[k])]), hard[k] * 80))
        if len(out) >= want:
            break
    return out


def stats(img):
    a = img.astype(np.float32)
    return {
        'h': img.shape[0], 'w': img.shape[1],
        'mean': a.mean(), 'std': a.std(),
        'p01': np.percentile(a, 1), 'p99': np.percentile(a, 99),
        # Vertical sharpness: how abruptly intensity changes down a column.
        # Layered OCT with good axial resolution has high values.
        'grad_y': np.abs(np.diff(a, axis=0)).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/masking/dataset_compare.png')
    ap.add_argument('--cols', type=int, default=3)
    args = ap.parse_args()

    sets = []
    if os.path.isfile(GOALS_ZIP):
        sets.append(('GOALS  (Topcon SS, circumpapillary)', from_zip(GOALS_ZIP, args.cols)))
    for n, lbl in (('Duke_DME', 'Duke DME  (Spectralis, macular)'),
                   ('AROI', 'AROI  (Zeiss Cirrus, macular)')):
        p = os.path.join(ZIPS, n + '.zip')
        if os.path.isfile(p):
            sets.append((lbl, from_zip(p, args.cols)))
    sets.append(('FairVision  (our target, 200x200)', from_fairvision(args.cols)))

    tile = 240
    lab, head, gap = 250, 96, 8
    W = lab + args.cols * 2 * (tile + gap)
    H = head + len(sets) * (tile + 34)
    fig = Image.new('RGB', (W, H), (14, 14, 16))
    d = ImageDraw.Draw(fig)
    d.text((16, 12), 'Candidate segmentation datasets vs our FairVision target',
           font=_font(19, True), fill=(245, 245, 245))
    d.text((16, 40),
           'Pairs are (B-scan, label mask). All resized to a square tile for display, so ASPECT RATIO IS NOT PRESERVED here '
           '-- see the printed geometry table for true sizes.',
           font=_font(12), fill=(176, 176, 182))
    d.text((16, 58),
           'What to compare: layer contrast, how much of the frame the retina fills, speckle texture, and whether the band is '
           'flat (macular) or curved around a disc (circumpapillary).',
           font=_font(12), fill=(176, 176, 182))

    print('%-38s %6s %6s %7s %7s %7s' % ('dataset', 'H', 'W', 'mean', 'std', 'grad_y'))
    for r, (name, items) in enumerate(sets):
        y = head + r * (tile + 34)
        d.text((14, y + 4), name.split('  ')[0], font=_font(13, True), fill=(240, 240, 240))
        if '  ' in name:
            for i, part in enumerate(name.split('  ')[1:]):
                d.text((14, y + 24 + i * 15), part, font=_font(11), fill=(150, 150, 158))
        if not items:
            d.text((lab, y + 8), 'not available locally', font=_font(13), fill=(220, 120, 120))
            continue
        s0 = stats(items[0][0])
        d.text((14, y + 58), '%d x %d' % (s0['h'], s0['w']), font=_font(11), fill=(0, 210, 190))
        d.text((14, y + 74), 'grad_y %.2f' % s0['grad_y'], font=_font(11), fill=(150, 150, 158))
        print('%-38s %6d %6d %7.1f %7.1f %7.2f'
              % (name, s0['h'], s0['w'], s0['mean'], s0['std'], s0['grad_y']))

        for c, (img, seg) in enumerate(items[: args.cols]):
            bi = Image.fromarray(img).convert('RGB').resize((tile, tile), Image.BILINEAR)
            # Colour the mask by class value so layers are distinguishable.
            sm = np.zeros(seg.shape + (3,), np.uint8)
            vals = [v for v in np.unique(seg) if v != 0]
            palette = [(240, 92, 92), (250, 190, 70), (120, 150, 255), (128, 194, 100),
                       (220, 120, 220), (90, 220, 220), (250, 140, 60), (180, 180, 180),
                       (140, 100, 200)]
            for i, v in enumerate(vals):
                sm[seg == v] = palette[i % len(palette)]
            si = Image.fromarray(sm).resize((tile, tile), Image.NEAREST)
            base = np.array(bi).astype(np.float32) * 0.45 + np.array(si).astype(np.float32) * 0.55
            si = Image.fromarray(base.astype(np.uint8))
            fig.paste(bi, (lab + c * 2 * (tile + gap), y))
            fig.paste(si, (lab + c * 2 * (tile + gap) + tile + gap, y))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.save(args.out, optimize=True)
    print()
    print('saved %s (%.0f KB)' % (args.out, os.path.getsize(args.out) / 1024))


if __name__ == '__main__':
    main()
