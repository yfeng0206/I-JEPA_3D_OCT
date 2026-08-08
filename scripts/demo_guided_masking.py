"""One-shot end-to-end demo of semantic multi-block masking (design A').

Renders every stage between a raw OCT B-scan and the token set the I-JEPA
context encoder actually receives, using the PRODUCTION sampler
(``CurriculumMaskGenerator._sample_mirage_blocks``) and the PRODUCTION paired
crop (``PairedRandomResizedCrop``).  Nothing is reimplemented, so what is drawn
is what a training step would do.

Pipeline
    native B-scan 200x200
      -> frozen MIRAGE  -> P(InnerRetina)+P(Choroid)   [precomputed, no repair]
      -> PairedRandomResizedCrop applied to IMAGE AND GUIDE with one rectangle
      -> mean-pool the cropped guide to the 16x16 patch grid  = occupancy
      -> placement = occupancy >= threshold
      -> sampler places 4 connected blocks (spread, overlap tolerance, retries)
      -> A': target = block INTERSECT anatomy      (zero background targets)
      -> context = every patch outside the block union

Design notes worth stating explicitly:

* MIRAGE is frozen and runs on the FULL slice, then its probability map is
  cropped with the image.  Cropping a precomputed map is both cheaper than
  in-loop inference and more faithful to MIRAGE's training distribution than
  running it on a 30%-scale crop.  In-loop inference is only required once
  MIRAGE itself becomes trainable, which is a later phase.
* The paired transform resamples the guide with NEAREST.  For a soft
  probability map that is a conservative choice: it cannot overshoot outside
  [0, 1] the way bicubic can, and the subsequent mean-pool to 16x16 averages
  many source pixels anyway.
* A' shrinks the target set relative to the rectangles, so ``unique target
  cells`` is logged before and after the intersection.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.transforms import PairedRandomResizedCrop  # noqa: E402

GRID = 16
PATCH = 16
CROP = 256

CURRICULUM_CFG = {
    'mode': 'mirage_envelope',
    'T_warm': 25, 'T_total': 30, 'r_max': 1.0, 'ramp_shape': 'linear',
    'mirage_dilate_patches': 0,
    'mirage_min_block_fill': 0.40,
    'mirage_min_retina_visible': 0.25,
    'mirage_max_attempts': 30,
    'mirage_occupancy_threshold': 0.25,
    'mirage_spread': True,
    'mirage_overlap_tolerance': 0.25,
}


def pool_to_grid(a: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Mean-pool a [0,1] map to (grid, grid).  Soft analogue of patch_occupancy."""
    h, w = a.shape
    ch, cw = h // grid, w // grid
    return a[:grid * ch, :grid * cw].reshape(grid, ch, grid, cw).mean(axis=(1, 3))


def run_once(image_u8, prob_u8, gen, thresh, seed):
    """One full masking draw.  Returns everything needed to draw it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    img_pil = Image.fromarray(image_u8).convert('RGB')
    guide_pil = Image.fromarray(prob_u8, mode='L')
    tf = PairedRandomResizedCrop(crop_size=CROP, crop_scale=(0.3, 1.0),
                                 ratio=(3.0 / 4.0, 4.0 / 3.0))
    tensor, cropped_guide = tf(img_pil, guide_pil)

    soft = np.asarray(cropped_guide, dtype=np.float32) / 255.0
    occ = pool_to_grid(soft).astype(np.float32)
    placement = occ >= thresh
    anatomy = occ >= thresh

    tg = torch.Generator()
    # Decorrelate block-size draws from the crop draw.  Both previously derived
    # from `seed`, so RandomResizedCrop's first uniforms and the first block's
    # scale/aspect uniforms were the SAME numbers -- small crops were paired
    # with small blocks, biasing every retention statistic.
    tg.manual_seed((seed * 2654435761 + 12345) % (2 ** 31))
    sizes = [gen._sample_block_size(gen.pred_mask_scale, tg)
             for _ in range(gen.npred)]
    gen.mirage_occupancy_threshold = thresh
    blocks, stats = gen._sample_mirage_blocks(
        sizes, torch.from_numpy(occ), torch.from_numpy(placement),
        [True] * gen.npred, [None] * gen.npred)

    union = sorted(set().union(*[set(b) for b in blocks])) if blocks else []
    flat_anat = anatomy.reshape(-1)
    targets_ap = [int(i) for i in union if flat_anat[i]]     # A' intersection
    context = [i for i in range(GRID * GRID) if i not in set(union)]
    return dict(tensor=tensor, soft=soft, occ=occ, placement=placement,
                blocks=blocks, sizes=sizes, stats=stats, union=union,
                targets_ap=targets_ap, context=context, anatomy=anatomy)


def cells_to_image(cells, grid=GRID):
    m = np.zeros(grid * grid, dtype=bool)
    if len(cells):
        m[np.array(list(cells), dtype=int)] = True
    return m.reshape(grid, grid)


def denorm(tensor):
    from src.transforms import IMAGENET_MEAN, IMAGENET_STD
    x = tensor.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        x[c] = x[c] * s + m
    return x.permute(1, 2, 0).clamp(0, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--guides', required=True, help='npz from mirage_soft_guide_dump')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.0868,
                    help='calibrated for 512; 1024 uses 0.25')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    z = np.load(a.guides, allow_pickle=True)
    img = z['images'][a.index]
    prob = z['probs'][a.index]
    name = str(z['names'][a.index])
    res = int(z['res'])

    gen = CurriculumMaskGenerator(
        input_size=(CROP, CROP), patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False, curriculum_cfg=dict(CURRICULUM_CFG))
    gen.set_epoch(99, 100)

    r = run_once(img, prob, gen, a.threshold, a.seed)
    view = denorm(r['tensor'])
    st = r['stats']

    print('slice %s   MIRAGE @%d   threshold %.4f' % (name, res, a.threshold))
    print('  block sizes (h,w)   : %s' % (r['sizes'],))
    print('  guided blocks       : %d / 4   feasible=%s  accepted=%s  attempts=%d'
          % (st['guided_blocks'], st['feasible'], st['accepted'], st['attempts']))
    print('  mean block fill     : %.4f' % st['mean_block_fill'])
    print('  retina visible      : %.4f  (accept bar %.2f)'
          % (st['retina_visible'], CURRICULUM_CFG['mirage_min_retina_visible']))
    print('  admissible cells    : %d / 256' % int(r['placement'].sum()))
    print('  target cells rect   : %d' % len(r['union']))
    print("  target cells A'     : %d   (%.1f%% of rectangles)"
          % (len(r['targets_ap']),
             100.0 * len(r['targets_ap']) / max(len(r['union']), 1)))
    print('  context cells       : %d' % len(r['context']))
    bg = len(r['union']) - len(r['targets_ap'])
    print('  background cells in rectangles: %d  ->  A\' removes all of them' % bg)

    fig, ax = plt.subplots(2, 4, figsize=(19, 9.6))
    ax = ax.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('1. native B-scan 200x200', fontsize=11)

    ax[1].imshow(prob / 255.0, cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('2. frozen MIRAGE @%d\nP(inner)+P(choroid), NO repair' % res,
                    fontsize=11)

    ax[2].imshow(view)
    ax[2].set_title('3. paired crop -> 256x256\n(same rectangle as guide)',
                    fontsize=11)

    ax[3].imshow(r['soft'], cmap='viridis', vmin=0, vmax=1)
    ax[3].set_title('4. guide under the SAME crop', fontsize=11)

    im = ax[4].imshow(r['occ'], cmap='viridis', vmin=0, vmax=1,
                      interpolation='nearest')
    for rr in range(GRID):
        for cc in range(GRID):
            if r['occ'][rr, cc] > 0.01:
                ax[4].text(cc, rr, '%.2f' % r['occ'][rr, cc], ha='center',
                           va='center', fontsize=4.5, color='w')
    ax[4].set_title('5. mean-pool -> 16x16 occupancy', fontsize=11)
    fig.colorbar(im, ax=ax[4], fraction=0.046)

    ax[5].imshow(r['placement'], cmap='gray', interpolation='nearest')
    ax[5].set_title('6. placement = occ >= %.3f\n%d / 256 admissible'
                    % (a.threshold, int(r['placement'].sum())), fontsize=11)

    ax[6].imshow(view, extent=(0, GRID, GRID, 0))
    colors = ['#ff3b30', '#34c759', '#0a84ff', '#ffd60a']
    for bi, (blk, (bh, bw)) in enumerate(zip(r['blocks'], r['sizes'])):
        top, left = blk[0] // GRID, blk[0] % GRID
        ax[6].add_patch(Rectangle((left, top), bw, bh, fill=False,
                                  edgecolor=colors[bi % 4], linewidth=2.5))
    ax[6].set_xlim(0, GRID); ax[6].set_ylim(GRID, 0)
    ax[6].set_title('7. four connected blocks\n(rectangles, I-JEPA prior kept)',
                    fontsize=11)

    comp = np.zeros((GRID, GRID, 3), dtype=np.float32)
    comp[cells_to_image(r['context'])] = (0.20, 0.55, 0.95)      # context
    comp[cells_to_image(r['union'])] = (0.35, 0.35, 0.35)        # rect, dropped
    comp[cells_to_image(r['targets_ap'])] = (1.0, 0.45, 0.0)     # A' targets
    ax[7].imshow(comp, interpolation='nearest')
    ax[7].set_title("8. blue=context %d   orange=A' target %d\ngrey=%d "
                    'background cells A\' drops'
                    % (len(r['context']), len(r['targets_ap']), bg), fontsize=11)

    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle('Semantic multi-block masking (A\') - one training step, '
                 'production sampler - %s' % name, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
