"""Full chain from MIRAGE's raw ViT decoder output to the JEPA target masks.

Shows every intermediate the guide passes through, at its true resolution, so
the anatomy signal can be checked against the actual B-scan pixels rather than
only at the 16x16 grid the sampler consumes.

    MIRAGE-Base @512
        patchify 32   ->  16x16 = 256 tokens (+1 global), ViT-B x12
        proj_dec      ->  384 x 64 x 64
        final_layer   ->  L0 : 4 x 64 x 64          <- RAW LOGITS, hooked here
        bilinear x8   ->  4 x 512 x 512             <- what MIRAGE itself emits
        argmax        ->  hard labels

    guide path (never uses argmax)
        softmax(L0)   ->  P  : 4 x 64 x 64
        P1 + P2       ->  M  : 64 x 64   soft anatomy occupancy in [0,1]
        avgpool 4x4   ->  a  : 16 x 16   the JEPA patch grid
        build_targets ->  4 connected targets

Panels: B-scan, per-class raw logits, softmax anatomy at 64x64, the same
upsampled to 512 and overlaid on the pixels, the argmax segmentation, the
pooled 16x16 grid, and the final targets.

Stage 1 (MIRAGE venv, GPU): --dump    Stage 2 (repo venv): --plot-from
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
CK_BASE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-base-512\MergedV3'
           r'\MIRAGE-Base_frozen_convnext_CEGDice-ignore\checkpoint-best.pth')
MIRAGE_RES, JEPA_RES, PATCH, GRID = 512, 256, 16, 16
ANATOMY = (1, 2)
CLASS_NAMES = ('0 Elsewhere', '1 InnerRetina', '2 Choroid', '3 Background')
CLASS_RGB = {1: (0.00, 0.75, 0.85), 2: (0.98, 0.69, 0.16)}
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.20, 0.75, 0.35), (0.98, 0.65, 0.10)]


def dump(out_path, picks, seed):
    import os
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from base512_vs_large1024_guide import build
    from fairvision_model_compare import MIRAGE_WS

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, interp = build('base', MIRAGE_RES, CK_BASE, device)
    assert not interp
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(L0=o.detach(), H=i[0].detach()))

    mimgs, jimgs, L0s, names = [], [], [], []
    for k, spec in enumerate(picks):
        vol_id, _, sl = spec.partition(':')
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        raw = np.asarray(vol[int(sl)], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        torch.manual_seed(seed + k)
        pil = Image.fromarray((unit * 255).astype(np.uint8))
        i, j, h, w = T.RandomResizedCrop.get_params(
            pil, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.))
        bic = T.InterpolationMode.BICUBIC
        m_img = np.asarray(TF.resized_crop(pil, i, j, h, w,
                                           [MIRAGE_RES, MIRAGE_RES], bic),
                           dtype=np.float32) / 255.
        j_img = np.asarray(TF.resized_crop(pil, i, j, h, w,
                                           [JEPA_RES, JEPA_RES], bic),
                           dtype=np.float32) / 255.
        x = torch.from_numpy(m_img)[None, None].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            model({'bscan': x})
        mimgs.append(m_img)
        jimgs.append(j_img)
        L0s.append(grab['L0'][0].float().cpu().numpy())     # (4, 64, 64) RAW
        names.append(spec)
        print('  %s   L0 %s   H %s' % (spec, tuple(grab['L0'].shape[1:]),
                                       tuple(grab['H'].shape[1:])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mimgs=np.stack(mimgs), jimgs=np.stack(jimgs),
                        L0=np.stack(L0s), names=np.array(names))
    print('wrote %s' % out_path)


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def upsample(m, size):
    """Bilinear upsample a 2-D map, matching MIRAGE's own final interpolate."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(m)[None, None].float()
    return F.interpolate(t, size=(size, size), mode='bilinear',
                         align_corners=False)[0, 0].numpy()


def plot(npz_path, out_png, rho):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from anatomy_target_sampler import build_targets

    z = np.load(npz_path, allow_pickle=True)
    mimgs, jimgs, L0s = z['mimgs'], z['jimgs'], z['L0']
    names = [str(s) for s in z['names']]
    n = len(names)
    ncol = 8
    fig, ax = plt.subplots(n, ncol, figsize=(3.0 * ncol, 3.15 * n))
    if n == 1:
        ax = ax[None]
    rows = []

    for r in range(n):
        L0, mimg, jimg = L0s[r], mimgs[r], jimgs[r]
        P = softmax_np(L0, axis=0)
        M = P[list(ANATOMY)].sum(axis=0)                       # (64,64) in [0,1]
        hard = L0.argmax(axis=0)                               # (64,64)
        g = M.shape[0]
        pool = g // GRID
        a = M.reshape(GRID, pool, GRID, pool).mean(axis=(1, 3))   # (16,16)
        parts, U = build_targets(a, rho=rho, overlap=0.24)

        c = 0
        ax[r, c].imshow(mimg, cmap='gray')
        ax[r, c].set_ylabel(names[r], fontsize=9)
        if r == 0:
            ax[r, c].set_title('1. OCT crop 512', fontsize=10)
        c += 1

        # raw logits: the two anatomy channels, at native decoder resolution
        for cls in ANATOMY:
            im = ax[r, c].imshow(L0[cls], cmap='RdBu_r',
                                 vmin=-abs(L0).max(), vmax=abs(L0).max())
            if r == 0:
                ax[r, c].set_title('2. RAW logit %s\n(%d x %d, pre-upsample)'
                                   % (CLASS_NAMES[cls], g, g), fontsize=9)
            plt.colorbar(im, ax=ax[r, c], fraction=0.046)
            c += 1

        im = ax[r, c].imshow(M, cmap='viridis', vmin=0, vmax=1)
        if r == 0:
            ax[r, c].set_title('3. softmax anatomy\n$M=P_1+P_2$ (%d x %d)' % (g, g),
                               fontsize=9)
        plt.colorbar(im, ax=ax[r, c], fraction=0.046)
        c += 1

        # the soft map upsampled to pixel resolution and laid over the B-scan
        Mup = upsample(M, MIRAGE_RES)
        ax[r, c].imshow(mimg, cmap='gray')
        ax[r, c].imshow(Mup, cmap='inferno', alpha=0.55, vmin=0, vmax=1)
        if r == 0:
            ax[r, c].set_title('4. $M$ upsampled to 512\nover the pixels', fontsize=9)
        c += 1

        # argmax segmentation, what MIRAGE itself would output
        hup = np.rint(upsample(hard.astype(np.float32), MIRAGE_RES)).astype(int)
        ov = np.repeat(mimg[..., None], 3, axis=2).copy()
        for cls, rgb in CLASS_RGB.items():
            m = hup == cls
            ov[m] = 0.5 * ov[m] + 0.5 * np.array(rgb)
        ax[r, c].imshow(np.clip(ov, 0, 1))
        if r == 0:
            ax[r, c].set_title('5. argmax segmentation\n(cyan inner, orange choroid)',
                               fontsize=9)
        c += 1

        im = ax[r, c].imshow(a, cmap='viridis', vmin=0, vmax=1)
        if r == 0:
            ax[r, c].set_title('6. avgpool %dx%d -> 16x16\nthe sampler input'
                               % (pool, pool), fontsize=9)
        plt.colorbar(im, ax=ax[r, c], fraction=0.046)
        c += 1

        ovj = np.repeat(jimg[..., None], 3, axis=2).copy()
        for k, p in enumerate(parts):
            up = np.kron(p, np.ones((PATCH, PATCH), dtype=bool))
            ovj[up] = 0.55 * ovj[up] + 0.45 * np.array(COLORS[k])
        ax[r, c].imshow(np.clip(ovj, 0, 1))
        sizes = [int(p.sum()) for p in parts]
        if r == 0:
            ax[r, c].set_title('7. targets on JEPA 256', fontsize=10)
        ax[r, c].set_xlabel('%s cells' % '/'.join(map(str, sizes)), fontsize=8)

        rows.append({'slice': names[r], 'logit_min': float(L0.min()),
                     'logit_max': float(L0.max()),
                     'M_mean': float(M.mean()),
                     'anatomy_mass_16': float(a.sum()),
                     'argmax_class_frac': {CLASS_NAMES[k]: float((hard == k).mean())
                                           for k in range(4)},
                     'target_cells': sizes})
        for cc in range(ncol):
            ax[r, cc].set_xticks([]); ax[r, cc].set_yticks([])

    handles = [Patch(facecolor=COLORS[k], label='target %d' % (k + 1)) for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11)
    fig.suptitle('MIRAGE-Base@512 raw ViT decoder output through to JEPA targets '
                 r'($\rho$=%.2f)' % rho, fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125, bbox_inches='tight')
    plt.close(fig)
    pathlib.Path(out_png).with_suffix('.json').write_text(json.dumps(rows, indent=2))

    print('%-16s %9s %9s %8s %9s  %s' %
          ('slice', 'logit_min', 'logit_max', 'M_mean', 'anat16', 'argmax class fractions'))
    for x in rows:
        f = x['argmax_class_frac']
        print('%-16s %9.2f %9.2f %8.4f %9.1f  %s' %
              (x['slice'], x['logit_min'], x['logit_max'], x['M_mean'],
               x['anatomy_mass_16'],
               ' '.join('%s=%.3f' % (k.split()[0], v) for k, v in f.items())))
    print('\nwrote %s' % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--out')
    ap.add_argument('--rho', type=float, default=0.70)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--picks', nargs='*',
                    default=['data_07050:4', 'data_08508:64', 'data_09186:144'])
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.picks, a.seed)
    if a.plot_from:
        return plot(a.plot_from, a.out, a.rho)
    raise SystemExit('need --dump or --plot-from')


if __name__ == '__main__':
    main()
