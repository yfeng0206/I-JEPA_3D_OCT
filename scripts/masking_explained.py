"""Visualise the occupancy-threshold bug and the I-JEPA predictor's view.

Two figures:

``bug.png``
    The bug itself.  MIRAGE gives a fractional occupancy per 16x16 patch.  Two
    separate places turn that into a boolean:
      * the DATASET builds the *placement region* -- the cells a target block
        may be drawn from (``oct_slices_guided.py:213``)
      * the COLLATOR builds the *scoring truth* -- what "on retina" and
        "retina still visible" are measured against (``curriculum.py:564``)
    The config set both to 0.25, but ``train_patch.py`` never passed the value
    to the dataset, so the dataset silently kept its 0.5 default.  Blocks were
    placed from the 0.5 region and scored against the 0.25 truth.  The figure
    shows the two regions, the cells that disagree, and the resulting masks.

``predictor.png``
    What the model actually sees per sample: the context the encoder is given,
    and the four target blocks the predictor must reconstruct from it -- each
    as its own prediction problem.
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

BLOCK_COLOURS = [(228, 62, 58), (66, 133, 220), (233, 168, 47), (128, 194, 100)]
REGION = (0, 210, 190)
DISAGREE = (255, 0, 170)
SCALE = 2


def _font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_uint8(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (tensor * std + mean).clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def _blend(base, mask, colour, alpha):
    out = base.astype(np.float32)
    for c in range(3):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + colour[c] * alpha,
                               out[..., c])
    return out.astype(np.uint8)


def _render(image, patch, grid_size, layers=(), hide=None, grid=True, size=None):
    """layers = sequence of (cell_mask, colour, alpha)."""
    out = image.copy()
    if hide is not None:
        out = _blend(out, np.repeat(np.repeat(hide, patch, 0), patch, 1),
                     (0, 0, 0), 0.82)
    for cells, colour, alpha in layers:
        out = _blend(out, np.repeat(np.repeat(cells, patch, 0), patch, 1),
                     colour, alpha)
    side = size or image.shape[0] * SCALE // 2
    img = Image.fromarray(out).resize((side, side), Image.NEAREST)
    if grid:
        draw = ImageDraw.Draw(img)
        step = side / grid_size
        for k in range(grid_size + 1):
            p = k * step
            draw.line([(p, 0), (p, side)], fill=(255, 255, 255, 32))
            draw.line([(0, p), (side, p)], fill=(255, 255, 255, 32))
    return img


def _sample(generator, grid, placement, valid, seed, grid_size):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    guide = torch.from_numpy(np.stack([grid, placement.astype(np.float32)], 0)).unsqueeze(0)
    enc, pred = generator.generate(
        batch_size=1, guide_grids=guide,
        guide_valid=torch.tensor([valid], dtype=torch.bool),
    )
    blocks = []
    for m in pred:
        flat = np.zeros(grid_size * grid_size, dtype=bool)
        flat[m[0].numpy()] = True
        blocks.append(flat.reshape(grid_size, grid_size))
    ctx = np.zeros(grid_size * grid_size, dtype=bool)
    ctx[enc[0][0].numpy()] = True
    return blocks, ctx.reshape(grid_size, grid_size)


def _pick(ds, seed, count):
    rng = random.Random(seed)
    picks = []
    guard = 0
    while len(picks) < count and guard < count * 60:
        guard += 1
        idx = rng.randrange(len(ds))
        img_t, guide_t, valid_t = ds[idx]
        if not bool(valid_t):
            continue
        grid = guide_t[0].numpy()
        # Require a real difference between the two thresholds so the figure
        # shows the bug rather than a slice where it happens not to matter.
        if (grid >= 0.25).sum() < 30 or ((grid >= 0.25) & ~(grid >= 0.50)).sum() < 5:
            continue
        picks.append((idx, img_t, grid, bool(valid_t)))
    return picks


def figure_bug(ds, generator, picks, patch, grid_size, out_path):
    tile = grid_size * patch * SCALE // 2
    lab = 250
    head = 152
    gap = 14
    cols = 6
    W = lab + cols * (tile + gap)
    H = head + len(picks) * (tile + 30)
    fig = Image.new('RGB', (W, H), (13, 13, 15))
    d = ImageDraw.Draw(fig)
    d.text((18, 14), 'The bug: target blocks were PLACED from one grid and SCORED against another',
           font=_font(21, True), fill=(245, 245, 245))
    body = [
        'MIRAGE outputs a fractional retina occupancy per 16x16 patch. Two independent places turned that fraction into a yes/no cell:',
        '   the DATASET built the placement region (oct_slices_guided.py:213)   and   the COLLATOR built the scoring truth (curriculum.py:564).',
        'configs/patch_mirage_envelope.yaml set 0.25 for both, but train_patch.py never passed it to the dataset, so the dataset kept its 0.5 default.',
        'Nothing crashed. Blocks were drawn from the SMALLER 0.5 region while "on retina" and "retina still visible" were judged on the LARGER 0.25 truth.',
    ]
    for i, line in enumerate(body):
        d.text((18, 46 + i * 19), line, font=_font(14),
               fill=(180, 180, 186) if i else (215, 215, 220))

    titles = [
        'Original B-scan',
        'region @ 0.50  (used)',
        'region @ 0.25  (intended)',
        'cells only 0.25 allows',
        'BEFORE  blocks from 0.50',
        'AFTER   blocks from 0.25',
    ]
    for c, t in enumerate(titles):
        colour = (255, 150, 150) if c == 4 else (150, 230, 170) if c == 5 else (225, 225, 230)
        d.text((lab + c * (tile + gap) + 4, head - 22), t, font=_font(13, True), fill=colour)

    stats = []
    for r, (idx, img_t, grid, valid) in enumerate(picks):
        img = _to_uint8(img_t)
        y = head + r * (tile + 30)
        seed = 4242 + r * 17
        r50, r25 = grid >= 0.50, grid >= 0.25
        only25 = r25 & ~r50
        b50, c50 = _sample(generator, grid, r50, valid, seed, grid_size)
        b25, c25 = _sample(generator, grid, r25, valid, seed, grid_size)

        panes = [
            _render(img, patch, grid_size),
            _render(img, patch, grid_size, [(r50, REGION, 0.42)]),
            _render(img, patch, grid_size, [(r25, REGION, 0.42)]),
            _render(img, patch, grid_size, [(r50, REGION, 0.16), (only25, DISAGREE, 0.72)]),
            _render(img, patch, grid_size,
                    [(b, BLOCK_COLOURS[i], 0.66) for i, b in enumerate(b50)], hide=~c50),
            _render(img, patch, grid_size,
                    [(b, BLOCK_COLOURS[i], 0.66) for i, b in enumerate(b25)], hide=~c25),
        ]
        for c, pane in enumerate(panes):
            fig.paste(pane, (lab + c * (tile + gap), y))

        u50 = int(np.any(np.stack(b50), 0).sum())
        u25 = int(np.any(np.stack(b25), 0).sum())
        vol = os.path.basename(ds.file_paths[idx // ds.num_slices]).replace('.npz', '')
        rows = [
            ('%s  s%d' % (vol, idx % ds.num_slices), (240, 240, 240), True),
            ('', None, False),
            ('region 0.50 : %3d cells' % r50.sum(), (150, 150, 158), False),
            ('region 0.25 : %3d cells' % r25.sum(), (0, 210, 190), False),
            ('disagree    : %3d cells' % only25.sum(), DISAGREE, False),
            ('', None, False),
            ('masked 0.50 : %3d patches' % u50, (255, 150, 150), False),
            ('masked 0.25 : %3d patches' % u25, (150, 230, 170), False),
            ('context 0.50: %3d tokens' % c50.sum(), (150, 150, 158), False),
            ('context 0.25: %3d tokens' % c25.sum(), (150, 150, 158), False),
        ]
        yy = y + 4
        for text, colour, bold in rows:
            if text:
                d.text((18, yy), text, font=_font(13, bold), fill=colour)
            yy += 17
        stats.append((vol, int(r50.sum()), int(r25.sum()), int(only25.sum()),
                      u50, u25, int(c50.sum()), int(c25.sum())))
    fig.save(out_path)
    return stats


def figure_predictor(ds, generator, picks, patch, grid_size, out_path):
    tile = grid_size * patch * SCALE // 2
    lab = 250
    head = 160
    gap = 14
    cols = 7
    W = lab + cols * (tile + gap)
    H = head + len(picks) * (tile + 30)
    fig = Image.new('RGB', (W, H), (13, 13, 15))
    d = ImageDraw.Draw(fig)
    d.text((18, 14), 'What the encoder sees and what the predictor must reconstruct',
           font=_font(21, True), fill=(245, 245, 245))
    body = [
        'I-JEPA never reconstructs pixels. The encoder sees ONLY the bright context tokens; everything dark is withheld from it.',
        'The predictor then receives those context embeddings plus the positions of one target block, and must output that block\'s',
        'embedding, which is compared by Smooth-L1 to the EMA target encoder\'s embedding of the FULL image at the same positions.',
        'Four target blocks = four separate prediction problems from the same context. MIRAGE only changes WHERE those blocks land.',
    ]
    for i, line in enumerate(body):
        d.text((18, 46 + i * 19), line, font=_font(14),
               fill=(215, 215, 220) if i == 0 else (180, 180, 186))

    titles = ['Original B-scan', 'Encoder input (context)', 'All 4 targets',
              'target 1', 'target 2', 'target 3', 'target 4']
    for c, t in enumerate(titles):
        colour = (225, 225, 230)
        if c >= 3:
            colour = BLOCK_COLOURS[c - 3]
        d.text((lab + c * (tile + gap) + 4, head - 22), t, font=_font(13, True), fill=colour)

    for r, (idx, img_t, grid, valid) in enumerate(picks):
        img = _to_uint8(img_t)
        y = head + r * (tile + 30)
        blocks, ctx = _sample(generator, grid, grid >= 0.25, valid, 4242 + r * 17, grid_size)
        panes = [
            _render(img, patch, grid_size),
            _render(img, patch, grid_size, [], hide=~ctx),
            _render(img, patch, grid_size,
                    [(b, BLOCK_COLOURS[i], 0.66) for i, b in enumerate(blocks)], hide=~ctx),
        ]
        for i, blk in enumerate(blocks):
            panes.append(_render(img, patch, grid_size,
                                 [(blk, BLOCK_COLOURS[i], 0.72)], hide=~(ctx | blk)))
        for c, pane in enumerate(panes):
            fig.paste(pane, (lab + c * (tile + gap), y))

        vol = os.path.basename(ds.file_paths[idx // ds.num_slices]).replace('.npz', '')
        union = int(np.any(np.stack(blocks), 0).sum())
        rows = [
            ('%s  s%d' % (vol, idx % ds.num_slices), (240, 240, 240), True),
            ('', None, False),
            ('encoder sees : %3d / 256' % ctx.sum(), (235, 235, 235), False),
            ('hidden       : %3d / 256' % (256 - ctx.sum()), (150, 150, 158), False),
            ('', None, False),
            ('target sizes : %s' % ' '.join(str(int(b.sum())) for b in blocks),
             (200, 200, 206), False),
            ('union        : %3d patches' % union, (200, 200, 206), False),
            ('retina cells : %3d' % int((grid >= 0.25).sum()), REGION, False),
        ]
        yy = y + 4
        for text, colour, bold in rows:
            if text:
                d.text((18, yy), text, font=_font(13, bold), fill=colour)
            yy += 17
    fig.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/patch_mirage_envelope.yaml')
    ap.add_argument('--rows', type=int, default=5)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, m, curr = cfg['data'], cfg['mask'], cfg['mask']['curriculum']
    crop, patch = d['crop_size'], m['patch_size']
    grid_size = crop // patch
    out_dir = args.output or os.path.join(
        os.path.dirname(d['slice_cache_dir']), 'masking_explained')
    os.makedirs(out_dir, exist_ok=True)

    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(d['data_dir'], 'Training'),
        guide_dir=os.path.join(curr['mirage_guide_dir'], 'Training'),
        num_slices=d['num_slices'], slice_size=crop,
        transform=make_paired_transforms(
            crop_size=crop, crop_scale=tuple(d['crop_scale']),
            horizontal_flip=False, gaussian_blur=False,
            color_distortion=False, color_jitter=0.0),
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
    generator.set_epoch(60, cfg['optimization']['epochs'])

    picks = _pick(ds, args.seed, args.rows)
    bug_path = os.path.join(out_dir, 'bug.png')
    pred_path = os.path.join(out_dir, 'predictor.png')
    stats = figure_bug(ds, generator, picks, patch, grid_size, bug_path)
    figure_predictor(ds, generator, picks, patch, grid_size, pred_path)
    print('saved %s' % bug_path)
    print('saved %s' % pred_path)
    print()
    print('%-14s %8s %8s %9s %8s %8s %8s %8s'
          % ('volume', 'reg.50', 'reg.25', 'disagree', 'msk.50', 'msk.25',
             'ctx.50', 'ctx.25'))
    for row in stats:
        print('%-14s %8d %8d %9d %8d %8d %8d %8d' % row)
    arr = np.array([r[1:] for r in stats], float)
    print('%-14s %8.1f %8.1f %9.1f %8.1f %8.1f %8.1f %8.1f'
          % (('mean',) + tuple(arr.mean(0))))


if __name__ == '__main__':
    main()
