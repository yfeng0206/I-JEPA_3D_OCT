#!/usr/bin/env python3
"""One picture explaining the three masking methods on real B-scans."""
from __future__ import annotations

import pathlib
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import anatomy_target_sampler_v2 as A                        # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator     # noqa: E402

GRIDS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
IMS = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos\im256.npy')
OUT = REPO / 'results/masking/explain'
G, NROW = 16, 3


def overlay(ax, img, mask, title, sub, color):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    big = np.kron(mask.astype(float), np.ones((16, 16)))
    rgba = np.zeros((256, 256, 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = big * 0.72
    ax.imshow(rgba)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(sub, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = np.load(GRIDS)['per']
    ims = np.load(IMS, mmap_mode='r')
    rows = [i for i in range(200) if A.is_viable([per[i, 0], per[i, 1]])][:NROW]

    guides = torch.from_numpy(((per[:, 0] + per[:, 1]) > 0.5).astype(np.float32))
    cfg = {'mode': 'mirage_envelope', 'enabled': True,
           'T_warm': 25, 'T_total': 30, 'r_max': 1.0, 'ramp_shape': 'linear'}
    g_rand = CurriculumMaskGenerator(input_size=(256, 256), patch_size=16, npred=4,
                                     nenc=1, curriculum_cfg=dict(cfg))
    g_rand.set_epoch(0, 100)                       # r_t = 0  -> no guidance
    g_guid = CurriculumMaskGenerator(input_size=(256, 256), patch_size=16, npred=4,
                                     nenc=1, curriculum_cfg=dict(cfg))
    g_guid.set_epoch(50, 100)                      # r_t = 1  -> full guidance

    fig, ax = plt.subplots(NROW, 4, figsize=(14.5, 3.7 * NROW))
    for r, i in enumerate(rows):
        img = np.asarray(ims[i])
        anat = (per[i, 0] + per[i, 1]) > 0.5

        def mk(gen, seed):
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
            _, pred = gen.generate(batch_size=1, guide_grids=guides[i:i + 1],
                                   guide_valid=torch.ones(1, dtype=torch.bool))
            m = np.zeros(G * G, bool)
            for p in pred:
                m[p[0].numpy()] = True
            return m.reshape(G, G)

        m_rand = mk(g_rand, 3)
        m_guid = mk(g_guid, 3)
        parts, _ = A.build_targets([per[i, 0], per[i, 1]])
        m_anat = np.logical_or.reduce(parts)

        ax[r, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        ax[r, 0].set_title('the B-scan' if r == 0 else '', fontsize=10)
        ax[r, 0].set_xlabel('retina = the bright curved band', fontsize=8)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])

        overlay(ax[r, 1], img, m_rand,
                'METHOD 1  random blocks' if r == 0 else '',
                '4 squares dropped anywhere - %d cells' % m_rand.sum(), (1, .2, .2))
        overlay(ax[r, 2], img, m_guid,
                'METHOD 2  guided rectangles  <- WHAT ACTUALLY TRAINED' if r == 0 else '',
                'still 4 squares, aimed at retina - %d cells' % m_guid.sum(),
                (1, .6, 0))
        overlay(ax[r, 3], img, m_anat,
                'METHOD 3  anatomy shapes  <- WHAT WE DESIGNED' if r == 0 else '',
                'follows the retina outline - %d cells' % m_anat.sum(), (0, .85, .4))

    fig.suptitle('Three ways to choose what to hide from the model.\n'
                 'Method 2 is what every run so far used. Method 3 exists only in '
                 'scripts and has never trained anything.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'three_methods.png', dpi=115)
    print('wrote', OUT / 'three_methods.png')


if __name__ == '__main__':
    main()
