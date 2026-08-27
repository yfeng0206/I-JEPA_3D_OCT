"""Sampler-equivalence check: does MIRAGE-512 place the same blocks as MIRAGE-1024?

Decision gate before committing the guided pretraining run to 512.  Segmentation
Dice at 512 is 0.0084 lower overall / 0.0140 lower on choroid, but the guided
sampler never sees a pixel map -- it sees a 16x16 fractional occupancy grid.  So
the question that actually matters is whether the two resolutions produce
*equivalent block placements*, not equivalent segmentations.

Stage 1 (MIRAGE venv) runs both checkpoints over the same held-out FairVision
Test slices and writes the pooled 16x16 occupancy for each.  Occupancy is built
the way the proposed design builds it -- softmax over the four classes, sum of
the two anatomy channels, adaptive average pool to 16x16 -- NOT from a repaired
binary envelope, because the whole point of the new design is to consume raw
MIRAGE probability.

Stage 2 (repo venv) replays the REAL sampler
(``CurriculumMaskGenerator._sample_mirage_blocks``) on both grids with identical
seeds and identical pre-drawn block sizes, then compares accepted geometry.

Nothing here trains anything and nothing is written back into the pipeline.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
CK1024 = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
          r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
CK512 = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-512\MergedV3'
         r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')

# merged taxonomy: 0 Elsewhere, 1 InnerRetina, 2 Choroid, 3 void
ANATOMY = (1, 2)
GRID = 16


def build(res: int, ckpt: str):
    """MIRAGE at an arbitrary square resolution.

    ``pos_emb`` must be INTERPOLATED from the checkpoint, never regenerated:
    although the adapter is built with ``learnable_pos_emb=False``, the weights
    carried in these checkpoints differ from a freshly generated sin-cos table
    by up to 2.003 (they were themselves interpolated up from MIRAGE's native
    16x16 grid during the 1024 fine-tune).  Regenerating silently destroys
    accuracy -- measured Inner Dice 0.969 -> 0.606.
    """
    import torch.nn.functional as F
    from argparse import Namespace
    from fm_seg_config import fm_factory
    from mirage.model import model_factory
    from mirage.output_adapters import ConvNeXtAdapter

    g = res // 32
    cfg = fm_factory['mirage-large']()
    cfg.build_domain_conf()
    ra = Namespace(grid_sizes={'bscan': [g, g]}, input_size={'bscan': [res, res]})
    ia = {'bscan': cfg.domain_conf['bscan']['input_adapter'](
        stride_level=1, patch_size_full=[32, 32], image_size=[res, res],
        learnable_pos_emb=False)}
    oa = {'semseg': ConvNeXtAdapter(
        num_classes=4, preds_per_patch=16, depth=4, interpolate_mode='bilinear',
        main_tasks=['bscan'], embed_dim=6144, patch_size=[32, 32], task='semseg',
        image_size=[res, res])}
    model = model_factory[cfg.model](
        args=ra, input_adapters=ia, output_adapters=oa,
        num_global_tokens=1, drop_path_rate=0.1)
    sd = dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    pe = sd['input_adapters.bscan.pos_emb']
    if pe.shape[-1] != g:
        sd['input_adapters.bscan.pos_emb'] = F.interpolate(
            pe.float(), size=(g, g), mode='bicubic', align_corners=False)
    model.load_state_dict(sd, strict=True)
    return model.cuda().eval()


def occupancy_grids(model, inputs, res: int) -> np.ndarray:
    """softmax -> P(inner)+P(choroid) -> adaptive avg pool to 16x16."""
    import torch.nn.functional as F
    out = []
    for arr in inputs:
        t = torch.from_numpy(arr)[None, None].cuda()
        with torch.inference_mode(), torch.autocast('cuda', torch.float16):
            o = model({'bscan': t})
        logits = (o['semseg'] if isinstance(o, dict) else o).float()
        prob = logits.softmax(dim=1)
        anat = prob[:, ANATOMY[0]] + prob[:, ANATOMY[1]]
        grid = F.adaptive_avg_pool2d(anat[:, None], (GRID, GRID))
        out.append(grid[0, 0].cpu().numpy().astype(np.float32))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--volumes', type=int, default=40)
    ap.add_argument('--slices', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dump', required=True)
    a = ap.parse_args()

    os.chdir(r'D:\jepa_phase0\mirage-goals')
    sys.path.insert(0, r'D:\jepa_phase0\mirage-goals\MIRAGE')
    import cv2

    files = sorted(TEST.glob('*.npz'))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(files), size=min(a.volumes, len(files)), replace=False)
    depths = np.linspace(20, 180, num=a.slices).astype(int)

    raw, names = [], []
    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            s = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = s.min(), s.max()
            raw.append((s - lo) / (hi - lo) if hi > lo else np.zeros_like(s))
            names.append('%s:%d' % (files[int(vi)].stem, int(d)))
    print('slices: %d' % len(raw))

    grids = {}
    for res, ck, tag in [(1024, CK1024, 'g1024'), (512, CK512, 'g512')]:
        big = [cv2.resize(s, (res, res), interpolation=cv2.INTER_LINEAR) for s in raw]
        m = build(res, ck)
        grids[tag] = occupancy_grids(m, big, res)
        del m
        torch.cuda.empty_cache()
        print('%s done  mean occ %.4f' % (tag, grids[tag].mean()))

    dst = pathlib.Path(a.dump)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, names=np.array(names), **grids)
    print('wrote %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
