"""Render what the occupancy-threshold fix actually changed.

Draws the real training masks side by side at threshold 0.50 (the silent bug:
``train_patch.py`` never passed the configured value, so the dataset kept its
default) and 0.25 (the calibrated policy now in force).

Both columns come from the SAME slice, the SAME crop and the SAME RNG seed, so
the four block sizes are identical and only the admissible region -- and
therefore where the blocks land -- differs.  Masks are produced by the actual
``CurriculumMaskGenerator`` used in training, not a reimplementation.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
from src.masks.curriculum import CurriculumMaskGenerator
from src.transforms import make_paired_transforms

BLOCK_COLOURS = [
    (228, 62, 58),    # red
    (66, 133, 220),   # blue
    (233, 168, 47),   # amber
    (128, 194, 100),  # green
]
REGION_COLOUR = (0, 210, 190)   # cyan: admissible placement region
CELL = 3                        # upscale factor for the 16x16 patch grid


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_uint8(tensor):
    """Undo ImageNet normalisation for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (tensor * std + mean).clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def _blend(base, mask, colour, alpha):
    out = base.astype(np.float32)
    for c in range(3):
        out[..., c] = np.where(
            mask, out[..., c] * (1 - alpha) + colour[c] * alpha, out[..., c]
        )
    return out.astype(np.uint8)


def _grid_overlay(image, grid_size, patch, region=None, blocks=None, context=None):
    """Paint the patch-level masks onto a copy of the slice."""
    out = image.copy()
    if context is not None:
        # Everything the encoder cannot see is dimmed.
        hidden = np.repeat(np.repeat(~context, patch, 0), patch, 1)
        out = _blend(out, hidden, (0, 0, 0), 0.55)
    if region is not None:
        big = np.repeat(np.repeat(region, patch, 0), patch, 1)
        out = _blend(out, big, REGION_COLOUR, 0.22)
    if blocks is not None:
        for i, blk in enumerate(blocks):
            big = np.repeat(np.repeat(blk, patch, 0), patch, 1)
            out = _blend(out, big, BLOCK_COLOURS[i % len(BLOCK_COLOURS)], 0.62)
    img = Image.fromarray(out).resize(
        (image.shape[1] * CELL // 2, image.shape[0] * CELL // 2), Image.NEAREST
    )
    draw = ImageDraw.Draw(img)
    step = img.width / grid_size
    for k in range(grid_size + 1):
        p = k * step
        draw.line([(p, 0), (p, img.height)], fill=(255, 255, 255, 40), width=1)
        draw.line([(0, p), (img.width, p)], fill=(255, 255, 255, 40), width=1)
    return img


def _masks_for(generator, grid, placement, valid, seed, grid_size):
    """Run the real generator with a pinned RNG so block sizes are shared."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    guide = torch.from_numpy(
        np.stack([grid, placement.astype(np.float32)], axis=0)
    ).unsqueeze(0)
    enc, pred = generator.generate(
        batch_size=1,
        guide_grids=guide,
        guide_valid=torch.tensor([valid], dtype=torch.bool),
    )
    blocks = []
    for m in pred:
        flat = np.zeros(grid_size * grid_size, dtype=bool)
        flat[m[0].numpy()] = True
        blocks.append(flat.reshape(grid_size, grid_size))
    ctx = np.zeros(grid_size * grid_size, dtype=bool)
    ctx[enc[0][0].numpy()] = True
    stats = dict(generator.mirage_stats)
    return blocks, ctx.reshape(grid_size, grid_size), stats


def _aggregate(ds, generator, grid_size, batches, batch_size, seed):
    """Paired A/B over real batches: same slices, same seeds, two thresholds.

    Runs at the true batch size because the collator truncates every target to
    the shortest in the batch, so per-image numbers do not extrapolate.
    """
    rng = random.Random(seed)
    totals = {0.50: {}, 0.25: {}}
    for key in totals:
        totals[key] = {
            'unique': 0.0, 'context': 0.0, 'on_region': 0.0, 'accept': 0.0,
            'infeasible': 0.0, 'fallbacks': 0.0, 'region_cells': 0.0,
            'retina_visible': 0.0, 'images': 0,
        }
    for b in range(batches):
        grids, valids = [], []
        while len(grids) < batch_size:
            idx = rng.randrange(len(ds))
            _, guide_t, valid_t = ds[idx]
            grids.append(guide_t[0].numpy())
            valids.append(bool(valid_t))
        grid_batch = np.stack(grids)
        valid_batch = torch.tensor(valids, dtype=torch.bool)
        for thr in (0.50, 0.25):
            placement = (grid_batch >= thr).astype(np.float32)
            guide = torch.from_numpy(
                np.stack([grid_batch, placement], axis=1)
            )
            random.seed(seed * 977 + b)
            np.random.seed((seed * 977 + b) % (2 ** 32))
            torch.manual_seed(seed * 977 + b)
            enc, pred = generator.generate(
                batch_size=batch_size, guide_grids=guide, guide_valid=valid_batch
            )
            s = generator.mirage_stats
            t = totals[thr]
            t['unique'] += float(s['unique_target_patches'])
            t['context'] += float(s['context_patches'])
            t['on_region'] += float(s['target_on_region'])
            t['accept'] += float(s['accept_rate'])
            t['infeasible'] += float(s['infeasible'])
            t['fallbacks'] += float(s['fallbacks'])
            t['retina_visible'] += float(s['retina_visible'])
            t['images'] += batch_size
        totals[0.50]['region_cells'] += float((grid_batch >= 0.50).sum()) / batch_size
        totals[0.25]['region_cells'] += float((grid_batch >= 0.25).sum()) / batch_size
        print('  batch %d/%d' % (b + 1, batches), flush=True)

    print()
    print('Paired A/B over %d images (batch %d), full guidance r_t=1.0'
          % (batches * batch_size, batch_size))
    print('%-26s %12s %12s %10s' % ('metric', 'thr 0.50', 'thr 0.25', 'change'))
    rows = [
        ('admissible region (cells)', 'region_cells'),
        ('unique target patches', 'unique'),
        ('context tokens', 'context'),
        ('target on region', 'on_region'),
        ('accept rate', 'accept'),
        ('retina visible', 'retina_visible'),
        ('infeasible /batch', 'infeasible'),
        ('invalid-guide fallbacks /batch', 'fallbacks'),
    ]
    for label, key in rows:
        a = totals[0.50][key] / batches
        c = totals[0.25][key] / batches
        pct = ((c - a) / a * 100.0) if a else float('nan')
        print('%-26s %12.2f %12.2f %9.1f%%' % (label, a, c, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_mirage_envelope.yaml')
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--output', default=None)
    ap.add_argument('--aggregate', type=int, default=0,
                    help='Also run a paired A/B over this many batches.')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, m, curr = cfg['data'], cfg['mask'], cfg['mask']['curriculum']
    crop, patch = d['crop_size'], m['patch_size']
    grid_size = crop // patch
    out_dir = args.output or os.path.join(
        os.path.dirname(d['slice_cache_dir']), 'threshold_fix_masks'
    )
    os.makedirs(out_dir, exist_ok=True)

    paired = make_paired_transforms(
        crop_size=crop, crop_scale=tuple(d['crop_scale']),
        horizontal_flip=False, gaussian_blur=False,
        color_distortion=False, color_jitter=0.0,
    )
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(d['data_dir'], 'Training'),
        guide_dir=os.path.join(curr['mirage_guide_dir'], 'Training'),
        num_slices=d['num_slices'], slice_size=crop, transform=paired,
        patch_size=patch, dilate_patches=int(curr['mirage_dilate_patches']),
        occupancy_threshold=float(curr['mirage_occupancy_threshold']),
        slice_cache=os.path.join(d['slice_cache_dir'], 'Training'),
    )
    generator = CurriculumMaskGenerator(
        input_size=(crop, crop), patch_size=patch,
        enc_mask_scale=tuple(m['enc_mask_scale']),
        pred_mask_scale=tuple(m['pred_mask_scale']),
        aspect_ratio=tuple(m['aspect_ratio']),
        nenc=m['num_enc_masks'], npred=m['num_pred_masks'],
        min_keep=m['min_keep'], allow_overlap=m['allow_overlap'],
        curriculum_cfg=curr,
    )
    generator.set_epoch(60, cfg['optimization']['epochs'])  # full guidance

    rng = random.Random(args.seed)
    picks = []
    tried = 0
    while len(picks) < args.rows and tried < args.rows * 40:
        tried += 1
        idx = rng.randrange(len(ds))
        img_t, guide_t, valid_t = ds[idx]
        if not bool(valid_t):
            continue
        grid = guide_t[0].numpy()
        if (grid >= 0.25).sum() < 25:      # skip near-empty crops
            continue
        picks.append((idx, img_t, grid, bool(valid_t)))

    font = _font(15)
    small = _font(13)
    tile = grid_size * patch * CELL // 2
    label_w = 240
    header = 78
    panel = Image.new(
        'RGB',
        (label_w + tile * 4 + 60, header + len(picks) * (tile + 26)),
        (14, 14, 16),
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (16, 12),
        'What the occupancy-threshold fix changed. Same slice, same crop, same RNG seed '
        '-> the four block sizes are identical; only the admissible region differs.',
        font=font, fill=(240, 240, 240),
    )
    draw.text(
        (16, 36),
        'Cyan = region a target block may be drawn from.  Red/blue/amber/green = the 4 target blocks.  '
        'Dimmed = hidden from the encoder.',
        font=small, fill=(170, 170, 175),
    )
    titles = [
        'Original OCT',
        'MIRAGE occupancy',
        'thr 0.50  (the bug)',
        'thr 0.25  (now running)',
    ]
    for c, title in enumerate(titles):
        draw.text((label_w + c * tile + 8, header - 20), title, font=small,
                  fill=(235, 235, 235))

    summary = []
    for r, (idx, img_t, grid, valid) in enumerate(picks):
        image = _to_uint8(img_t)
        y = header + r * (tile + 26)
        seed = args.seed * 1000 + r

        p05 = grid >= 0.50
        p025 = grid >= 0.25
        b05, c05, s05 = _masks_for(generator, grid, p05, valid, seed, grid_size)
        b025, c025, s025 = _masks_for(generator, grid, p025, valid, seed, grid_size)

        cols = [
            _grid_overlay(image, grid_size, patch),
            _grid_overlay(image, grid_size, patch, region=grid >= 0.25),
            _grid_overlay(image, grid_size, patch, region=p05, blocks=b05, context=c05),
            _grid_overlay(image, grid_size, patch, region=p025, blocks=b025, context=c025),
        ]
        for c, im in enumerate(cols):
            panel.paste(im, (label_w + c * tile, y))

        u05 = int(np.any(np.stack(b05), axis=0).sum())
        u025 = int(np.any(np.stack(b025), axis=0).sum())
        vol = os.path.basename(ds.file_paths[idx // ds.num_slices]).replace('.npz', '')
        lines = [
            ('%s s%d' % (vol, idx % ds.num_slices), (235, 235, 235)),
            ('region 0.50: %d cells' % int(p05.sum()), (150, 150, 155)),
            ('region 0.25: %d cells' % int(p025.sum()), (0, 210, 190)),
            ('masked 0.50: %d' % u05, (235, 130, 130)),
            ('masked 0.25: %d' % u025, (140, 220, 150)),
            ('context 0.50: %d' % int(c05.sum()), (150, 150, 155)),
            ('context 0.25: %d' % int(c025.sum()), (150, 150, 155)),
        ]
        for li, (text, colour) in enumerate(lines):
            draw.text((16, y + 6 + li * 17), text, font=small, fill=colour)
        summary.append((vol, int(p05.sum()), int(p025.sum()), u05, u025,
                        int(c05.sum()), int(c025.sum())))

    path = os.path.join(out_dir, 'threshold_fix_masks.png')
    panel.save(path)
    print('saved %s' % path)
    print()
    print('%-14s %9s %9s %9s %9s %9s %9s'
          % ('volume', 'reg0.50', 'reg0.25', 'msk0.50', 'msk0.25', 'ctx0.50', 'ctx0.25'))
    for row in summary:
        print('%-14s %9d %9d %9d %9d %9d %9d' % row)
    arr = np.array([r[1:] for r in summary], dtype=float)
    print('%-14s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f' % (('mean',) + tuple(arr.mean(0))))
    print()
    print('NOTE: the panel above renders one image at a time, so these per-slice')
    print('counts do NOT reflect the batch-level truncation that training applies.')
    print('Use --aggregate for numbers that do.')

    if args.aggregate:
        print()
        _aggregate(ds, generator, grid_size, args.aggregate, args.batch_size,
                   args.seed)


if __name__ == '__main__':
    main()
