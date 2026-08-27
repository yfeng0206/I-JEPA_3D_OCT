"""Dump RAW MIRAGE anatomy probability at native resolution -- no repair.

Phase-1 guide source for semantic multi-block masking.  Differs from
``mirage_precompute_guides.py`` in two deliberate ways:

1. It stores a SOFT probability map, not a bit-packed binary envelope, so the
   sampler can consume raw MIRAGE confidence rather than a hard decision.
2. It does NOT run ``repair_union``.  The repair existed to bridge the
   unlabelled mid-retina, which is a taxonomy hole present in 100% of GOALS
   columns with mean height 0.0504 of the image.  Measured at the 16x16 I-JEPA
   mask grid, where one cell spans 0.0625 of the image, that hole closes on its
   own: of 1,695 cells lying between anatomy, exactly 1 (0.06%) is empty.  The
   repair is therefore unnecessary at the resolution the mask actually uses,
   and removing it deletes the only non-differentiable step in the guide path.

Probability is quantised to uint8 (p * 255).  At native 200x200 that is 40 KB
per slice before compression, versus 80 KB for float16, and the sampler
thresholds at ~0.09-0.25 where a 1/255 quantisation step is irrelevant.

The map is stored at NATIVE resolution on purpose: the paired random-resized
crop has to be applied to the image and the guide together, so the guide must
still be croppable.  Pooling to 16x16 happens after the crop, in the dataset.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch

NATIVE = 200
ANATOMY = (1, 2)            # merged taxonomy: 1 InnerRetina, 2 Choroid

CK = {
    1024: (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
           r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth'),
    512: (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-512\MergedV3'
          r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth'),
}


def build(res: int, ckpt: str, device: str = 'cuda'):
    """MIRAGE at an arbitrary square resolution.

    ``pos_emb`` is INTERPOLATED from the checkpoint, never regenerated: despite
    ``learnable_pos_emb=False`` these checkpoints carry a table that differs
    from freshly generated sin-cos by up to 2.003, because the 1024 fine-tune
    itself interpolated MIRAGE's native 16x16 grid upward.  Regenerating it
    silently destroys accuracy (measured Inner Dice 0.969 -> 0.606).
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
        main_tasks=['bscan'], embed_dim=6144, patch_size=[32, 32],
        task='semseg', image_size=[res, res])}
    model = model_factory[cfg.model](
        args=ra, input_adapters=ia, output_adapters=oa,
        num_global_tokens=1, drop_path_rate=0.1)
    sd = dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    pe = sd['input_adapters.bscan.pos_emb']
    if pe.shape[-1] != g:
        sd['input_adapters.bscan.pos_emb'] = F.interpolate(
            pe.float(), size=(g, g), mode='bicubic', align_corners=False)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def anatomy_prob(model, unit: np.ndarray, res: int, device: str) -> np.ndarray:
    """P(InnerRetina) + P(Choroid) resampled back to native resolution."""
    import cv2
    import torch.nn.functional as F
    big = cv2.resize(unit, (res, res), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(big)[None, None].to(device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(
            device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
        out = model({'bscan': t})
    logits = (out['semseg'] if isinstance(out, dict) else out).float()
    prob = logits.softmax(dim=1)
    anat = prob[:, ANATOMY[0]] + prob[:, ANATOMY[1]]
    small = F.interpolate(anat[:, None], size=(NATIVE, NATIVE),
                          mode='bilinear', align_corners=False)
    return small[0, 0].clamp(0, 1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='Test')
    ap.add_argument('--res', type=int, default=512, choices=(512, 1024))
    ap.add_argument('--volumes', type=int, default=1)
    ap.add_argument('--slices', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.chdir(r'D:\jepa_phase0\mirage-goals')
    sys.path.insert(0, r'D:\jepa_phase0\mirage-goals\MIRAGE')

    root = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data') / a.split
    files = sorted(root.glob('*.npz'))
    if not files:
        raise SystemExit('no volumes under %s' % root)
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(files), size=min(a.volumes, len(files)), replace=False)
    depths = np.linspace(20, 180, num=a.slices).astype(int)

    device = a.device if torch.cuda.is_available() or a.device == 'cpu' else 'cpu'
    model = build(a.res, CK[a.res], device)

    imgs, probs, names = [], [], []
    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            imgs.append((unit * 255).astype(np.uint8))
            probs.append((anatomy_prob(model, unit, a.res, device) * 255
                          ).astype(np.uint8))
            names.append('%s:%d' % (files[int(vi)].stem, int(d)))
    print('slices: %d   mean anatomy prob %.4f'
          % (len(imgs), np.mean(probs) / 255.0))

    dst = pathlib.Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, images=np.stack(imgs), probs=np.stack(probs),
                        names=np.array(names), res=np.array(a.res))
    print('wrote %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
