"""Visual comparison of the two trained MIRAGE-Large arms: 512 vs 1024 input.

Both arms are the SAME model -- MIRAGE-Large, 24 blocks, dim 1024, fine-tuned on
MergedV3 with the identical recipe -- and differ only in the input resolution
they were fine-tuned at:

    1024: outputs\\mergedv3       , pos_emb (1,1024,32,32), 32x32 token grid
     512: outputs\\mergedv3-512   , pos_emb (1,1024,16,16), 16x16 token grid

512 is MIRAGE's NATIVE pretraining resolution -- the released MIRAGE-Large.pth
carries a (1,1024,16,16) pos_emb -- so the 1024 arm is the one running away from
the pretrained grid, not the other way round.

Raw argmax, void logit suppressed, no envelope repair and no morphology, so what
is drawn is what the network emits.  Per-panel stats are the raw-union
connectivity numbers from fairvision_model_compare, so they are comparable to
every other table in this workstream.

Two stages, because inference needs the MIRAGE venv (no matplotlib):
  stage 1  --dump masks.npz        (MIRAGE venv)
  stage 2  --plot-from masks.npz   (repo venv)
"""
from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fairvision_model_compare import MIRAGE_WS, components, mean_runs  # noqa: E402
from fairvision_v1_v2 import DEFAULT_PICKS, load_inputs, overlay  # noqa: E402

CK = {
    1024: (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
           r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth'),
    512: (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-512\MergedV3'
          r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth'),
}


def build(res: int, ckpt: str, device: str):
    """MIRAGE-Large at an arbitrary square resolution.

    pos_emb is INTERPOLATED from the checkpoint, never regenerated: despite
    learnable_pos_emb=False these tables differ from freshly generated sin-cos
    by up to 2.003, and regenerating drops Inner Dice 0.969 -> 0.606.
    """
    import torch.nn.functional as F
    from argparse import Namespace
    sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
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


def run(res: int, inputs: list, device: str) -> np.ndarray:
    """Raw argmax with the void logit suppressed, resampled to a common size."""
    import cv2
    model = build(res, CK[res], device)
    outs = []
    for arr in inputs:
        small = cv2.resize(arr, (res, res), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(small)[None, None].to(device=device,
                                                   dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            out = model({'bscan': t})
        logits = (out['semseg'] if isinstance(out, dict) else out).float().clone()
        logits[:, 3] = float('-inf')       # void never wins the argmax
        hard = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        if hard.shape[0] != 1024:
            hard = cv2.resize(hard, (1024, 1024),
                              interpolation=cv2.INTER_NEAREST)
        outs.append(hard)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(outs)


def stats_for(mask: np.ndarray) -> tuple:
    union = mask > 0
    n_comp, largest = components(union)
    runs = mean_runs(union)
    return n_comp, largest, (float(np.mean(runs)) if runs else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--picks', nargs='*', default=list(DEFAULT_PICKS))
    ap.add_argument('--dump')
    ap.add_argument('--plot-from')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.plot_from:
        return plot(a.plot_from, a.out)
    if not a.dump:
        raise SystemExit('need --dump, or --plot-from')

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs, names = load_inputs(a.picks)
    print('slices: %d' % len(inputs))

    labels = ['MIRAGE-Large @1024', 'MIRAGE-Large @512 (native grid)']
    stacks = []
    for res in (1024, 512):
        print('running %d ...' % res)
        stacks.append(run(res, inputs, device))

    masks = np.stack(stacks)
    st = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.float32)
    for ai in range(masks.shape[0]):
        for si in range(masks.shape[1]):
            st[ai, si] = stats_for(masks[ai, si])
    for ai, lab in enumerate(labels):
        inner = float((masks[ai] == 1).mean())
        chor = float((masks[ai] == 2).mean())
        print('  %-32s inner %.4f  choroid %.4f  n_comp %.2f  runs/col %.3f'
              % (lab, inner, chor, st[ai, :, 0].mean(), st[ai, :, 2].mean()))

    dst = pathlib.Path(a.dump)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, inputs=np.stack(inputs).astype(np.float16),
                        names=np.array(names), labels=np.array(labels),
                        masks=masks, stats=st)
    print('wrote %s' % dst)
    return 0


def plot(npz_path: str, out_path: str) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from fairvision_v1_v2 import CHOR_RGB, INNER_RGB

    z = np.load(npz_path, allow_pickle=True)
    inputs, names = z['inputs'].astype(np.float32), list(z['names'])
    labels, masks, st = list(z['labels']), z['masks'], z['stats']
    n = len(names)

    fig, axes = plt.subplots(n, 5, figsize=(19.5, 4.0 * n))
    if n == 1:
        axes = axes[None, :]
    heads = ['B-scan (FairVision Test)', labels[0], labels[1],
             'disagreement', 'union outline']

    for r in range(n):
        m1024, m512 = masks[0, r], masks[1, r]
        axes[r, 0].imshow(np.clip(inputs[r], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[r, 1].imshow(overlay(inputs[r], m1024))
        axes[r, 2].imshow(overlay(inputs[r], m512))

        # where the two arms disagree, and in which direction
        diff = np.zeros(m1024.shape + (3,), dtype=np.uint8)
        diff[(m1024 > 0) & (m512 == 0)] = (255, 60, 60)     # only 1024
        diff[(m512 > 0) & (m1024 == 0)] = (60, 160, 255)    # only 512
        diff[(m512 > 0) & (m1024 > 0)] = (70, 70, 70)       # both
        axes[r, 3].imshow(diff)

        both = np.zeros(m1024.shape + (3,), dtype=np.uint8)
        both[m1024 > 0] = (200, 60, 60)
        both[m512 > 0] = (60, 160, 255)
        both[(m1024 > 0) & (m512 > 0)] = (245, 245, 245)
        axes[r, 4].imshow(both)

        inter = float(((m1024 > 0) & (m512 > 0)).sum())
        uni = float(((m1024 > 0) | (m512 > 0)).sum())
        axes[r, 3].set_xlabel('union IoU %.4f' % (inter / max(uni, 1)), fontsize=9)
        for c, ai in ((1, 0), (2, 1)):
            axes[r, c].set_xlabel(
                'components %d   runs/col %.2f   inner %.3f  choroid %.3f'
                % (int(st[ai, r, 0]), st[ai, r, 2],
                   float((masks[ai, r] == 1).mean()),
                   float((masks[ai, r] == 2).mean())), fontsize=8.5)
        axes[r, 0].set_ylabel(names[r], fontsize=10)
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(heads[c], fontsize=11)

    fig.legend(handles=[
        Patch(facecolor=np.array(INNER_RGB) / 255, label='InnerRetina'),
        Patch(facecolor=np.array(CHOR_RGB) / 255, label='Choroid'),
        Patch(facecolor=(1.0, 0.235, 0.235), label='only 1024 predicts retina'),
        Patch(facecolor=(0.235, 0.63, 1.0), label='only 512 predicts retina'),
        Patch(facecolor=(0.96, 0.96, 0.96), label='both agree'),
    ], loc='lower center', ncol=5, frameon=False, fontsize=10)
    fig.suptitle('Same model, same recipe, different input resolution: '
                 'MIRAGE-Large @1024 vs @512\n'
                 'both ViT-L (24 blocks, dim 1024); 512 is MIRAGE\'s native '
                 'pretraining grid - RAW argmax, no repair', fontsize=13)
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=104)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
