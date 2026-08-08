"""Dump every decoder-stage tensor for ONE B-scan as inspectable PNGs.

Traces a single slice through the segmentation head and writes each intermediate
to its own file, so the shapes quoted in the pipeline table can be seen rather
than taken on faith:

    encoder tokens   (1, 1025, 1024)      -> 32x32 token grid
    blocks INPUT     (1, 384, 128, 128)   <- proj_dec output, reshaped
    blocks OUTPUT    (1, 384, 128, 128)   <- after 4 ConvNeXt blocks
    final_layer      (1, 4, 128, 128)     <- the real prediction resolution
    bilinear x8      (1, 4, 1024, 1024)   <- interpolation only, no new detail
    argmax           (1024, 1024)         <- hard segmentation
    patch occupancy  (16, 16)             <- the I-JEPA mask grid

Nothing is repaired, closed, filtered or smoothed anywhere in this script.

Low-resolution stages are ALSO written upscaled with NEAREST interpolation, so
the true blockiness stays visible instead of being hidden by viewer smoothing.
Every file records its true shape in the manifest.

Two stages, because inference needs the MIRAGE venv (no matplotlib):
  stage 1  --dump stages.npz        (MIRAGE venv)
  stage 2  --render-from stages.npz (repo venv)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fairvision_model_compare import MIRAGE_WS, build_model  # noqa: E402
from fairvision_raw_native import V1_NAMES, V1_PALETTE, MERGED_NAMES, MERGED_PALETTE  # noqa: E402

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')

# shipped guide policy, see configs/patch_mirage_envelope.yaml
OCCUPANCY_THRESHOLD = 0.25
MASK_GRID = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--arm', choices=['v1', 'merged'], default='v1')
    ap.add_argument('--volume', default='data_07266')
    ap.add_argument('--slice', type=int, default=199)
    ap.add_argument('--dump')
    ap.add_argument('--render-from')
    ap.add_argument('--outdir')
    a = ap.parse_args()

    if a.render_from:
        return render(a.render_from, a.outdir)
    if not (a.ckpt and a.dump):
        raise SystemExit('need --ckpt --dump, or --render-from')

    os.chdir(MIRAGE_WS)
    import cv2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = TEST / ('%s.npz' % a.volume)
    if not path.exists():
        raise SystemExit('missing volume: %s' % path)
    with np.load(path, allow_pickle=True) as z:
        vol = z['oct_bscans']
    raw = np.asarray(vol[a.slice], dtype=np.float32)
    lo, hi = raw.min(), raw.max()
    unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    img = cv2.resize(unit, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    model = build_model(4, pathlib.Path(a.ckpt), device)
    caught = {}

    def grab(key, which):
        def hook(mod, inp, out):
            t = (inp[0] if which == 'in' else out)
            caught[key] = t.detach().float().cpu().numpy()
        return hook

    mods = dict(model.named_modules())
    mods['encoder'].register_forward_hook(grab('encoder_out', 'out'))
    mods['output_adapters.semseg.blocks'].register_forward_hook(grab('proj_dec_reshaped', 'in'))
    mods['output_adapters.semseg.blocks'].register_forward_hook(grab('convnext_out', 'out'))
    mods['output_adapters.semseg.final_layer'].register_forward_hook(grab('logits_128', 'out'))

    t = torch.from_numpy(img)[None, None].to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        out = model({'bscan': t})
    logits_1024 = (out['semseg'] if isinstance(out, dict) else out).float().cpu().numpy()

    hard = logits_1024[0].argmax(0).astype(np.uint8)

    # I-JEPA mask grid: fractional occupancy of the raw union per 64x64 patch.
    union = (hard > 0).astype(np.float32)
    cell = union.shape[0] // MASK_GRID
    occ = union.reshape(MASK_GRID, cell, MASK_GRID, cell).mean(axis=(1, 3))

    payload = dict(
        input_1024=img.astype(np.float32),
        encoder_out=caught['encoder_out'],
        proj_dec_reshaped=caught['proj_dec_reshaped'],
        convnext_out=caught['convnext_out'],
        logits_128=caught['logits_128'],
        logits_1024=logits_1024,
        hard_1024=hard,
        occupancy_16=occ.astype(np.float32),
        arm=np.array(a.arm),
        title=np.array('%s  slice %d' % (a.volume, a.slice)),
        ckpt=np.array(str(a.ckpt)),
    )
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.ndim > 0:
            print('  %-20s %s' % (k, v.shape))

    dst = pathlib.Path(a.dump)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    print('wrote %s' % dst)
    return 0


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def render(npz_path: str, outdir: str) -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    z = np.load(npz_path, allow_pickle=True)
    arm = str(z['arm'])
    title = str(z['title'])
    names = V1_NAMES if arm == 'v1' else MERGED_NAMES
    palette = V1_PALETTE if arm == 'v1' else MERGED_PALETTE

    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save_gray(name, arr, upto=None, desc='', truth=None):
        a8 = (np.clip(_norm(arr), 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(a8)
        if upto:
            im = im.resize((upto, upto), Image.NEAREST)
        im.save(out / name)
        manifest.append(dict(file=name, true_shape=list(truth or arr.shape), desc=desc))

    def save_rgb(name, arr, upto=None, desc='', truth=None):
        im = Image.fromarray(arr)
        if upto:
            im = im.resize((upto, upto), Image.NEAREST)
        im.save(out / name)
        manifest.append(dict(file=name, true_shape=list(truth or arr.shape[:2]), desc=desc))

    inp = z['input_1024']
    save_gray('00_input_1024.png', inp, desc='input B-scan, min-max scaled')

    # ---- encoder tokens: 1025 tokens -> drop global -> 32x32 grid ----
    enc = z['encoder_out'][0]                     # (1025, 1024)
    tok = enc[1:] if enc.shape[0] == 1025 else enc
    side = int(round(tok.shape[0] ** 0.5))
    tokmap = np.linalg.norm(tok.reshape(side, side, -1), axis=2)
    save_gray('01_encoder_tokens_%dx%d_L2.png' % (side, side), tokmap, 512,
              'encoder output token L2 norm, TRUE size %dx%d' % (side, side),
              truth=(side, side))

    # ---- decoder feature maps ----
    for key, fname, desc in [
            ('proj_dec_reshaped', '02_proj_dec_reshaped_384x128x128',
             'proj_dec output reshaped to spatial, 384ch'),
            ('convnext_out', '03_convnext_out_384x128x128',
             'after 4 ConvNeXt blocks, 384ch')]:
        f = z[key][0]                              # (384, 128, 128)
        save_gray('%s_mean.png' % fname, f.mean(axis=0), 512,
                  '%s - channel mean' % desc, truth=f.shape[1:])
        var = f.reshape(f.shape[0], -1).var(axis=1)
        top = np.argsort(var)[::-1][:16]
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i, ci in enumerate(top):
            axi = ax[i // 4, i % 4]
            axi.imshow(_norm(f[ci]), cmap='magma')
            axi.set_title('ch %d' % ci, fontsize=9)
            axi.set_xticks([]); axi.set_yticks([])
        fig.suptitle('%s\n16 highest-variance channels of 384, each %dx%d'
                     % (desc, f.shape[1], f.shape[2]), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / ('%s_top16ch.png' % fname), dpi=100)
        plt.close(fig)
        manifest.append(dict(file='%s_top16ch.png' % fname,
                             true_shape=[384, int(f.shape[1]), int(f.shape[2])],
                             desc='%s - 16 highest-variance channels' % desc))

    # ---- final_layer logits at TRUE 128x128, per class ----
    lg128 = z['logits_128'][0]                     # (4, 128, 128)
    for c in range(lg128.shape[0]):
        save_gray('04_logits_128_class%d_%s.png' % (c, names[c].replace('/', '-')),
                  lg128[c], 512,
                  'final_layer logit, class %d %s, TRUE 128x128' % (c, names[c]),
                  truth=lg128.shape[1:])

    e = np.exp(lg128 - lg128.max(axis=0, keepdims=True))
    prob = e / e.sum(axis=0, keepdims=True)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for c in range(4):
        im = ax[c].imshow(prob[c], cmap='viridis', vmin=0, vmax=1)
        ax[c].set_title('%s  (softmax)' % names[c], fontsize=11)
        ax[c].set_xticks([]); ax[c].set_yticks([])
        fig.colorbar(im, ax=ax[c], fraction=0.046)
    fig.suptitle('Class probabilities at the TRUE prediction resolution 128x128 - %s'
                 % title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / '05_softmax_128_panel.png', dpi=100)
    plt.close(fig)
    manifest.append(dict(file='05_softmax_128_panel.png', true_shape=[4, 128, 128],
                         desc='softmax over the 4 classes at 128x128'))

    # ---- after bilinear x8 ----
    lg1024 = z['logits_1024'][0]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for c in range(4):
        ax[c].imshow(_norm(lg1024[c]), cmap='magma')
        ax[c].set_title('%s' % names[c], fontsize=11)
        ax[c].set_xticks([]); ax[c].set_yticks([])
    fig.suptitle('After bilinear x8: (4, 1024, 1024) - interpolation only, '
                 'no detail finer than 8px', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / '06_logits_1024_panel.png', dpi=100)
    plt.close(fig)
    manifest.append(dict(file='06_logits_1024_panel.png', true_shape=[4, 1024, 1024],
                         desc='upsampled logits, all 4 classes'))

    # ---- hard segmentation ----
    hard = z['hard_1024']
    hard128 = lg128.argmax(0).astype(np.uint8)

    def colourise(h):
        rgb = np.zeros(h.shape + (3,), dtype=np.uint8)
        for ci, col in enumerate(palette):
            rgb[h == ci] = col
        return rgb

    save_rgb('07_hard_seg_128.png', colourise(hard128), 512,
             'argmax at TRUE 128x128 (before upsampling)', truth=(128, 128))
    save_rgb('08_hard_seg_1024.png', colourise(hard),
             desc='argmax at 1024x1024 - the hard segmentation')

    g = np.repeat((np.clip(inp, 0, 1) * 255)[..., None], 3, axis=2)
    lab = colourise(hard).astype(np.float32)
    m = (hard != 0)[..., None]
    ov = np.clip(g * (1 - 0.45 * m) + lab * (0.45 * m), 0, 255).astype(np.uint8)
    save_rgb('09_hard_seg_overlay_1024.png', ov, desc='hard segmentation over the B-scan')

    # ---- patch level: the I-JEPA mask grid ----
    occ = z['occupancy_16']
    save_gray('10_patch_occupancy_16.png', occ, 512,
              'fractional union occupancy per 64x64 patch, TRUE 16x16', truth=occ.shape)
    sel = (occ >= OCCUPANCY_THRESHOLD).astype(np.uint8) * 255
    save_gray('11_patch_placement_16_thr%.2f.png' % OCCUPANCY_THRESHOLD, sel, 512,
              'boolean placement region at occupancy >= %.2f, TRUE 16x16'
              % OCCUPANCY_THRESHOLD, truth=occ.shape)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(np.clip(inp, 0, 1), cmap='gray')
    ax[0].set_title('B-scan 1024x1024', fontsize=11)
    ax[1].imshow(ov)
    ax[1].set_title('hard segmentation (argmax)', fontsize=11)
    im = ax[2].imshow(occ, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax[2].set_title('patch occupancy 16x16', fontsize=11)
    for c in range(16):
        for r in range(16):
            if occ[r, c] > 0.01:
                ax[2].text(c, r, '%.2f' % occ[r, c], ha='center', va='center',
                           fontsize=5, color='w')
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for axi in ax[:2]:
        axi.set_xticks([]); axi.set_yticks([])
    fig.suptitle('%s  -  pixel level to patch level' % title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / '12_summary_panel.png', dpi=110)
    plt.close(fig)
    manifest.append(dict(file='12_summary_panel.png', true_shape=[],
                         desc='B-scan / hard seg / 16x16 occupancy with values'))

    frac = {names[c]: float((hard == c).mean()) for c in range(4)}
    (out / 'manifest.json').write_text(json.dumps(dict(
        title=title, arm=arm, ckpt=str(z['ckpt']),
        class_fractions_1024=frac,
        true_prediction_resolution='128x128 (everything above is bilinear x8)',
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        files=manifest), indent=2))
    print('wrote %d files to %s' % (len(manifest) + 1, out))
    for m_ in manifest:
        print('   %-46s true %s' % (m_['file'], m_['true_shape']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
