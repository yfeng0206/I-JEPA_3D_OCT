"""v2 masking visual: class-aware targets AND the correct I-JEPA context policy.

Two things this figure shows that the earlier demos did not.

CLASS-AWARE TARGETS.  InnerRetina and Choroid are grown as separate connected
regions and each receives its own share of the four targets, instead of forcing
P1+P2 into one object and bridging across the unlabelled mid-retina.

THE REAL CONTEXT.  The earlier demo drew context = all patches - target union,
which gives the encoder ~214 of 256 tokens.  The production collator hands it a
mean of 84.2 (min 38, max 119).  MIRAGE must replace ONLY masks_pred; the
context block still comes from the original I-JEPA sampler and then has the
target union removed.  Both are drawn here so the difference is visible.

Also reports visibility under BOTH definitions, because quoting only the mass
figure is what made the masks look implausibly small:

    mass visible    1 - sum(a_i for i in U) / sum(a)
    extent visible  1 - |U ^ S| / |S|,  S = {a_i > tau}

Reads the per-class npz written by the dump step.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

GRID, PATCH = 16, 16
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
    TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
    CK = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-base-512\MergedV3'
          r'\MIRAGE-Base_frozen_convnext_CEGDice-ignore\checkpoint-best.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, interp = build('base', 512, CK, device)
    assert not interp
    g = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: g.update(L0=o.detach()))

    jimgs, pers, names = [], [], []
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
        m_img = np.asarray(TF.resized_crop(pil, i, j, h, w, [512, 512], bic),
                           dtype=np.float32) / 255.
        j_img = np.asarray(TF.resized_crop(pil, i, j, h, w, [256, 256], bic),
                           dtype=np.float32) / 255.
        x = torch.from_numpy(m_img)[None, None].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            model({'bscan': x})
        P = g['L0'].softmax(1)
        per = torch.stack([F.adaptive_avg_pool2d(P[:, c:c + 1], (GRID, GRID))[0, 0]
                           for c in (1, 2)]).cpu().numpy()
        jimgs.append(j_img)
        pers.append(per)
        names.append(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, jimgs=np.stack(jimgs), per=np.stack(pers),
                        names=np.array(names))
    print('wrote %s  %d slices' % (out_path, len(names)))


def plot(npz_path, out_png, mass_cap, tau):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import torch
    from src.masks.multiblock import MaskCollator
    from anatomy_target_sampler_v2 import build_targets, topology_of, n_components

    z = np.load(npz_path, allow_pickle=True)
    jimgs, pers, names = z['jimgs'], z['per'], [str(s) for s in z['names']]
    n = len(names)
    coll = MaskCollator(input_size=(256, 256), patch_size=16)
    ncol = 6
    fig, ax = plt.subplots(n, ncol, figsize=(3.25 * ncol, 3.4 * n))
    if n == 1:
        ax = ax[None]
    rows = []

    for r in range(n):
        per, img = pers[r], jimgs[r]
        a = per.sum(0)
        A = float(a.sum())
        S = a > tau
        parts, regions = build_targets([per[0], per[1]], n=4,
                                       mass_cap=mass_cap, tau=tau, overlap=0.0)
        U = np.zeros_like(parts[0])
        for p in parts:
            U |= p

        # the REAL I-JEPA context block, then remove the target union.
        # MaskCollator draws block LOCATIONS from Python's `random`
        # (multiblock.py:93-94) and re-seeds its torch size-generator from
        # random.randint on every call, so torch.manual_seed alone has no
        # effect here and the reported context size would change every run.
        random.seed(1234 + r)
        torch.manual_seed(1234 + r)
        _, me, _ = coll([torch.from_numpy(img)[None].repeat(3, 1, 1)])
        c0 = set(me[0][0].tolist())
        ctx = sorted(c0 - set(np.flatnonzero(U.ravel()).tolist()))
        ctx_mask = np.zeros(GRID * GRID, bool)
        ctx_mask[np.array(ctx, dtype=int)] = True
        ctx_mask = ctx_mask.reshape(GRID, GRID)

        st = {'slice': names[r], 'anatomy_mass': A, 'support_cells': int(S.sum()),
              'target_cells': [int(p.sum()) for p in parts],
              'union_cells': int(U.sum()),
              'targets_connected': all(p.sum() > 0 and n_components(p) == 1
                                       for p in parts),
              'mass_visible': float(1 - (a * U).sum() / A),
              'extent_visible': float(1 - (U & S).sum() / max(S.sum(), 1)),
              'ijepa_context_block': len(c0),
              'context_after_removing_union': len(ctx),
              'naive_complement_would_be': int(256 - U.sum()),
              'retina_mass_in_context': float((a * ctx_mask).sum() / A),
              'union_holes': topology_of(U)['holes']}
        rows.append(st)

        c = 0
        ax[r, c].imshow(img, cmap='gray')
        ax[r, c].set_ylabel(names[r], fontsize=9)
        if r == 0:
            ax[r, c].set_title('OCT crop 256', fontsize=10)
        c += 1

        rgb = np.zeros((GRID, GRID, 3))
        rgb[..., 0] = per[1] * 0.98            # choroid -> warm
        rgb[..., 1] = per[0] * 0.75            # inner   -> green/cyan
        rgb[..., 2] = per[0] * 0.85
        ax[r, c].imshow(np.clip(rgb, 0, 1))
        if r == 0:
            ax[r, c].set_title('per-class anatomy\ncyan=Inner  orange=Choroid',
                               fontsize=9)
        c += 1

        reg = np.ones((GRID, GRID, 3)) * 0.12
        for ri, R in enumerate(regions):
            col = np.array([(0.2, 0.8, 0.9), (0.98, 0.65, 0.1)][ri % 2])
            reg[R] = 0.45 * reg[R] + 0.55 * col
        ax[r, c].imshow(reg)
        if r == 0:
            ax[r, c].set_title('per-class REGIONS\n(grown inside support S)', fontsize=9)
        c += 1

        canvas = np.ones((GRID, GRID, 3)) * 0.12
        for k, p in enumerate(parts):
            canvas[p] = 0.5 * canvas[p] + 0.5 * np.array(COLORS[k])
        ax[r, c].imshow(canvas)
        ax[r, c].set_title('4 targets %s\nunion %d cells'
                           % ('/'.join(map(str, st['target_cells'])),
                              st['union_cells']), fontsize=9)
        c += 1

        ov = np.repeat(img[..., None], 3, axis=2).copy()
        for k, p in enumerate(parts):
            up = np.kron(p, np.ones((PATCH, PATCH), dtype=bool))
            ov[up] = 0.55 * ov[up] + 0.45 * np.array(COLORS[k])
        ax[r, c].imshow(np.clip(ov, 0, 1))
        ax[r, c].set_title('targets on the B-scan\nvisible: mass %.0f%%  extent %.0f%%'
                           % (100 * st['mass_visible'],
                              100 * st['extent_visible']), fontsize=9)
        c += 1

        cv = np.repeat(img[..., None], 3, axis=2).copy() * 0.30
        upc = np.kron(ctx_mask, np.ones((PATCH, PATCH), dtype=bool))
        base = np.repeat(img[..., None], 3, axis=2)
        cv[upc] = base[upc]
        ax[r, c].imshow(np.clip(cv, 0, 1))
        ax[r, c].set_title('REAL context: %d tokens\n(naive complement would be %d)'
                           % (st['context_after_removing_union'],
                              st['naive_complement_would_be']),
                           fontsize=9, color='darkgreen')

        for cc in range(ncol):
            ax[r, cc].set_xticks([]); ax[r, cc].set_yticks([])

    handles = [Patch(facecolor=COLORS[k], label='target %d' % (k + 1)) for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11)
    fig.suptitle('v2 sampler: class-aware targets + the real I-JEPA context policy   '
                 r'(mass cap %.2f, $\tau$=%.2f, overlap 0)' % (mass_cap, tau),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125, bbox_inches='tight')
    plt.close(fig)
    pathlib.Path(out_png).with_suffix('.json').write_text(json.dumps(rows, indent=2))

    print('%-16s %6s %6s %-16s %6s %8s %8s %9s %9s' %
          ('slice', 'anat', '|S|', 'targets', 'union', 'vis_mass', 'vis_ext',
           'ctx REAL', 'ctx naive'))
    for x in rows:
        print('%-16s %6.1f %6d %-16s %6d %7.0f%% %7.0f%% %9d %9d' %
              (x['slice'], x['anatomy_mass'], x['support_cells'],
               '/'.join(map(str, x['target_cells'])), x['union_cells'],
               100 * x['mass_visible'], 100 * x['extent_visible'],
               x['context_after_removing_union'], x['naive_complement_would_be']))
    print('\nall targets connected & non-empty: %s   holes: %s'
          % (all(x['targets_connected'] for x in rows),
             [x['union_holes'] for x in rows]))
    print('wrote %s' % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--out')
    ap.add_argument('--mass-cap', type=float, default=0.80)
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--picks', nargs='*',
                    default=['data_07050:4', 'data_07225:24',
                             'data_08508:64', 'data_09186:144'])
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.picks, a.seed)
    if a.plot_from:
        return plot(a.plot_from, a.out, a.mass_cap, a.tau)
    raise SystemExit('need --dump or --plot-from')


if __name__ == '__main__':
    main()
