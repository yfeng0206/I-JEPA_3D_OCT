#!/usr/bin/env python
"""Every stage of the I-JEPA input pipeline, on one B-scan.

    1  raw B-scan, with the random crop window drawn on it
    2  the crop, resized to 256 -- this is what actually enters the model
    3  the MIRAGE guide carried through the SAME crop, pooled to the 16x16
       token grid the sampler works on
    4  the 4 prediction targets (hidden from the context encoder)
    5  what the context encoder actually receives: the encoder block minus the
       target union, everything else blacked out

Stage 1 matters because the guide must be cropped with the image or it points
at the wrong anatomy; `PairedRandomResizedCrop` draws one rectangle and applies
it to both. Stage 5 matters because the context is NOT simply "the image minus
the targets" -- an encoder block is sampled at enc_mask_scale first, and the
target union is then subtracted from it.

Rendered with 4-connected growth and randomised rim growth, the two changes
under consideration. Nothing in src/ is modified.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch, Rectangle                    # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.masks.anatomy as AN                                     # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator           # noqa: E402

OUT = REPO / 'results/masking/dilation'
GRID, PATCH, RES = 16, 16, 256
TARGET_RGB = [(239, 71, 111), (255, 209, 102), (6, 214, 160), (17, 138, 178)]
NB4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIRS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def grow_rim(occ, radius, directions=None, rim_score=0.30):
    if radius <= 0:
        return occ.copy()
    out = occ > 0
    dirs = [DIRS[d] for d in (directions or DIRS)]
    h, w = occ.shape
    for _ in range(radius):
        g = out.copy()
        for dr, dc in dirs:
            s = np.zeros_like(out)
            r0, r1 = max(0, dr), min(h, h + dr)
            c0, c1 = max(0, dc), min(w, w + dc)
            s[r0:r1, c0:c1] = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            g |= s
        out = g
    grown = occ.copy()
    grown[out & (occ == 0)] = rim_score
    return grown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', default=(
        r'C:\jepa_data\mirage_soft_guides'
        r'\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training'))
    ap.add_argument('--data', default=r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--rim', type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    AN.NB8[:] = NB4

    rng = np.random.default_rng(a.seed)
    files = sorted(pathlib.Path(a.guides).glob('*.npy'))
    picks = [files[i] for i in rng.choice(len(files), a.n, replace=False)]

    cfg = {'mode': 'mirage_anatomy', 'enabled': True, 'T_warm': 25, 'T_total': 30,
           'r_max': 1.0, 'ramp_shape': 'linear', 'anatomy_mass_cap': 0.90,
           'anatomy_tau': 0.10, 'mirage_occupancy_threshold': 0.25}

    rows = []
    for k, gf in enumerate(picks):
        vol = np.load(gf, mmap_mode='r')
        sl = int(rng.integers(25, 75))
        soft = np.asarray(vol[sl], np.float32) / 255.0            # (2,200,200)
        with np.load(pathlib.Path(a.data) / (gf.stem + '.npz'), allow_pickle=True) as z:
            raw = np.asarray(z['oct_bscans'][sl * 2], np.float32)
        lo, hi = raw.min(), raw.max()
        raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0

        img_pil = Image.fromarray((raw * 255).astype(np.uint8))
        occ_full = np.clip(soft[0] + soft[1], 0, 1)
        gp = Image.fromarray((occ_full * 255).astype(np.uint8)).resize(
            img_pil.size, Image.NEAREST)

        # One crop rectangle, applied to image AND guide -- the real behaviour.
        torch.manual_seed(1000 + k)
        top, left, h, w = T.RandomResizedCrop.get_params(
            img_pil, [0.3, 1.0], [3.0 / 4.0, 4.0 / 3.0])
        crop_img = np.asarray(TF.resized_crop(
            img_pil, top, left, h, w, [RES, RES],
            T.InterpolationMode.BICUBIC), np.float32) / 255.
        crop_g = np.asarray(TF.resized_crop(
            gp, top, left, h, w, [RES, RES],
            T.InterpolationMode.NEAREST), np.float32) / 255.

        occ = F.adaptive_avg_pool2d(
            torch.from_numpy(crop_g)[None, None], (GRID, GRID))[0, 0].numpy()
        if occ.sum() < 0.5:
            continue
        mode = str(rng.choice(['all', 'up', 'down']))
        dirs = None if mode == 'all' else (
            ['up', 'left', 'right'] if mode == 'up' else ['down', 'left', 'right'])
        guide_grid = grow_rim(occ, a.rim, dirs)

        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        gen = CurriculumMaskGenerator(
            input_size=(RES, RES), patch_size=PATCH, npred=4, nenc=1,
            pred_mask_scale=(0.15, 0.2), pred_target_k=16, curriculum_cfg=cfg)
        gen.set_epoch(50, 100)
        t = torch.from_numpy(guide_grid)[None].float()
        enc, pred = gen.generate(
            batch_size=1, guide_grids=torch.stack([t, (t >= 0.25).float()], 1),
            guide_valid=torch.ones(1, dtype=torch.bool))
        rows.append((gf.stem, sl, raw, (top, left, h, w), crop_img, occ,
                     guide_grid, enc[0][0].numpy(),
                     [p[0].numpy() for p in pred], mode))

    fig, axes = plt.subplots(len(rows), 5, figsize=(18, 3.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    for r, (nm, sl, raw, (top, left, h, w), cimg, occ, gg, eidx, pidx, mode) in enumerate(rows):
        axes[r, 0].imshow(raw, cmap='gray')
        axes[r, 0].add_patch(Rectangle((left, top), w, h, fill=False,
                                       edgecolor='#ff3860', lw=2.2))
        axes[r, 0].set_ylabel('%s\nslice %d' % (nm, sl), fontsize=8)
        axes[r, 0].set_xlabel('crop %dx%d at (%d,%d)' % (w, h, left, top), fontsize=8)

        axes[r, 1].imshow(cimg, cmap='gray')
        axes[r, 1].set_xlabel('resized to %d, enters the model' % RES, fontsize=8)

        axes[r, 2].imshow(cimg, cmap='gray')
        axes[r, 2].imshow(np.kron(gg, np.ones((PATCH, PATCH))), cmap='YlGn',
                          alpha=0.5, vmin=0, vmax=1)
        axes[r, 2].set_xlabel('guide after the SAME crop -> 16x16', fontsize=8)

        base = (np.stack([cimg] * 3, -1) * 255).astype(np.uint8)
        tgt = (base * 0.35).astype(np.uint8)
        hidden = set()
        for ti, idx in enumerate(pidx):
            col = np.array(TARGET_RGB[ti % 4], np.uint8)
            for i in idx:
                rr, cc = int(i) // GRID, int(i) % GRID
                hidden.add(int(i))
                y, x = rr * PATCH, cc * PATCH
                blk = base[y:y + PATCH, x:x + PATCH].astype(np.float32)
                tgt[y:y + PATCH, x:x + PATCH] = (0.4 * blk + 0.6 * col).astype(np.uint8)
        axes[r, 3].imshow(tgt)
        axes[r, 3].set_xlabel('4 targets = HIDDEN (%d cells, rim +%d %s)'
                              % (len(hidden), 1, mode), fontsize=8)

        ctx = np.zeros_like(base)
        for i in eidx:
            rr, cc = int(i) // GRID, int(i) % GRID
            y, x = rr * PATCH, cc * PATCH
            ctx[y:y + PATCH, x:x + PATCH] = base[y:y + PATCH, x:x + PATCH]
        axes[r, 4].imshow(ctx)
        axes[r, 4].set_xlabel('CONTEXT the encoder receives (%d of 256 tokens)'
                              % len(eidx), fontsize=8)

        if r == 0:
            for c, t_ in enumerate(['1. raw B-scan + crop window',
                                    '2. cropped input',
                                    '3. guide, same crop',
                                    '4. prediction targets (hidden)',
                                    '5. context encoder input']):
                axes[r, c].set_title(t_, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    hl = [Patch(color=np.array(c) / 255, label='target %d' % (i + 1))
          for i, c in enumerate(TARGET_RGB)]
    hl.append(Patch(facecolor='none', edgecolor='#ff3860', label='random crop window'))
    fig.legend(handles=hl, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('Full I-JEPA input pipeline: random crop -> guide carried through '
                 'the same crop -> targets hidden -> context encoder input',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(OUT / 'full_pipeline.png', dpi=140)
    print('wrote', OUT / 'full_pipeline.png')
    for nm, sl, _, crop, _, _, _, e, p, mode in rows:
        hid = len(set(np.concatenate(p).tolist()))
        print('  %-12s slice %2d  crop %s  hidden %2d  context %3d  rim %s'
              % (nm, sl, crop, hid, len(e), mode))


if __name__ == '__main__':
    main()
