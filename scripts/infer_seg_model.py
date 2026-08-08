"""Run a trained MIRAGE segmentation checkpoint over a directory of B-scans and
write hard predictions as PNGs.

We deliberately do NOT use MIRAGE's own `--infer_only --test` path.  A code
review established two defects in it:

* `run_seg_tuning.py` passes `log_images=True` on the test calls, and the eval
  loop then saves images INSTEAD of collecting predictions and loss, so it
  computes metrics over empty lists.
* the "best" checkpoint reload prepends `module.` to every key and loads with
  `strict=False`, so every key is silently rejected and the last-epoch model
  stays active.

This script loads the checkpoint with `strict=True` (so a key mismatch is a
hard error, not a silent no-op) and reproduces MIRAGE's validation transform:
per-image min-max to [0,1], then bilinear resize to 1024, with no ImageNet
normalisation -- which is what `norm='minmax'` does in the official pipeline.

The ignore/void channel is suppressed before argmax, matching the behaviour
MIRAGE documents in its evaluation loop.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

MIRAGE_WS = pathlib.Path(r'D:\jepa_phase0\mirage-goals')


def build_model(num_classes: int, ckpt: pathlib.Path, device: str):
    from argparse import Namespace
    sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
    from fm_seg_config import fm_factory
    from mirage.model import model_factory
    from mirage.output_adapters import ConvNeXtAdapter

    cfg = fm_factory['mirage-large']()
    cfg.build_domain_conf()
    runtime_args = Namespace(grid_sizes={'bscan': [32, 32]},
                             input_size={'bscan': [1024, 1024]})
    input_adapters = {
        'bscan': cfg.domain_conf['bscan']['input_adapter'](
            stride_level=1, patch_size_full=[32, 32],
            image_size=[1024, 1024], learnable_pos_emb=False)
    }
    output_adapters = {
        'semseg': ConvNeXtAdapter(
            num_classes=num_classes, preds_per_patch=16, depth=4,
            interpolate_mode='bilinear', main_tasks=['bscan'], embed_dim=6144,
            patch_size=[32, 32], task='semseg', image_size=[1024, 1024])
    }
    model = model_factory[cfg.model](
        args=runtime_args, input_adapters=input_adapters,
        output_adapters=output_adapters, num_global_tokens=1,
        drop_path_rate=0.1)
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    model.load_state_dict(state, strict=True)
    meta = {}
    if isinstance(blob, dict):
        for k in ('epoch', 'best_miou', 'max_miou'):
            if k in blob:
                meta[k] = blob[k]
    return model.to(device).eval(), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--images', required=True,
                    help='directory of B-scan PNGs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-classes', type=int, default=4)
    ap.add_argument('--ignore-index', type=int, default=3,
                    help='suppressed before argmax; -1 to disable')
    ap.add_argument('--palette', default='0,128,255',
                    help='output pixel value per class index, in order')
    ap.add_argument('--strip-prefix', action='store_true',
                    help="write '0071.png' for an input named 'GOALS__0071.png'")
    args = ap.parse_args()

    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = pathlib.Path(args.ckpt)
    model, meta = build_model(args.num_classes, ckpt, device)
    print('loaded', ckpt, 'meta:', meta)

    palette = [int(x) for x in args.palette.split(',')]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in pathlib.Path(args.images).iterdir()
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif',
                                            '.tiff', '.bmp'))
    if not images:
        raise SystemExit('no images in %s' % args.images)

    ig_hits = 0
    total_px = 0
    for p in images:
        arr = np.array(Image.open(p).convert('L'), dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        unit = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        big = cv2.resize(unit, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(big)[None, None].to(device=device,
                                                 dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16,
                enabled=device == 'cuda'):
            out = model({'bscan': t})
        # .clone() is required: tensors produced inside inference_mode reject
        # in-place writes, and we need to overwrite the void channel below.
        logits = (out['semseg'] if isinstance(out, dict) else out).float().clone()

        if 0 <= args.ignore_index < logits.shape[1]:
            raw = logits.argmax(1)[0]
            ig_hits += int((raw == args.ignore_index).sum())
            total_px += int(raw.numel())
            logits[:, args.ignore_index] = float('-inf')

        hard = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        vis = np.zeros_like(hard)
        for idx, value in enumerate(palette):
            vis[hard == idx] = value

        name = p.name.split('__', 1)[-1] if args.strip_prefix else p.name
        Image.fromarray(vis).save(out_dir / name)

    print('wrote %d predictions to %s' % (len(images), out_dir))
    if total_px:
        print('void channel would have won %.6f of pixels (suppressed)'
              % (ig_hits / total_px))
    (out_dir / '_inference.json').write_text(json.dumps({
        'checkpoint': str(ckpt), 'checkpoint_meta': meta,
        'n_images': len(images), 'palette': palette,
        'ignore_index': args.ignore_index,
        'void_argmax_rate': (ig_hits / total_px) if total_px else None,
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
