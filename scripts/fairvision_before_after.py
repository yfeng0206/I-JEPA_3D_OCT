#!/usr/bin/env python
"""FairVision before/after: what the adapter does to the anatomy guide.

GOALS gives ground truth but is only 30 images and is not what we pretrain on.
This shows the effect on FairVision -- the actual pretraining distribution --
for both an early and a late I-JEPA teacher, under the old Gram-MSE loss and
the new class-conditioned structural loss.

Two rows of evidence per image:
  segmentation   frozen MIRAGE vs adapted, with the disagreement highlighted
  masking guide  the 16x16 occupancy grid the sampler actually consumes

Slices are drawn from the stratified cache, so peripheral depths are included
rather than only the mid-volume B-scan.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter_placement_ablation import Adapter                       # noqa: E402
from goals_eval import VOID, RES                                     # noqa: E402
from jepa_to_mirage_probe import build_mirage                        # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
SRC = REPO / 'results/masking/structural_loss'
OUT = REPO / 'results/masking/structural_loss'
GRID = 16
OCC = 0.25
RGB = {0: (12, 12, 20), 1: (0, 190, 210), 2: (250, 176, 40)}


def colourise(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, v in RGB.items():
        out[lab == k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4, help='FairVision slices to show')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--adapters', default=(
        'late_ep100_gram_mse_a050,late_ep100_structl100_a050,'
        'early_ep27_structl100_a050'),
        help='comma list of adapter stems under results/masking/structural_loss')
    a = ap.parse_args()
    dev = 'cuda'

    mir = build_mirage(dev)
    sem = mir.output_adapters['semseg']

    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    pos = np.load(CACHE / 'pos.npy') if (CACHE / 'pos.npy').exists() else None
    # Spread the sample across slice depth so the figure is not all mid-volume.
    if pos is not None:
        order = np.argsort(pos)
        pick = order[np.linspace(0, len(order) - 1, a.n).astype(int)]
    else:
        pick = np.linspace(0, len(im512) - 1, a.n).astype(int)
    imgs = np.asarray(im512[np.sort(pick)], np.float32) / 255.
    depths = pos[np.sort(pick)] if pos is not None else [None] * a.n

    def predict(adapter=None):
        h = None
        if adapter is not None:
            def pre(m, args):
                x = args[0]
                return (adapter(x.float()).to(x.dtype),) + args[1:]
            h = sem.proj_dec.register_forward_pre_hook(pre)
        try:
            x = torch.from_numpy(imgs)[:, None].to(dev)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                o = mir({'bscan': x})
            lg = (o['semseg'] if isinstance(o, dict) else o).float()
            if lg.shape[-1] != RES:
                lg = F.interpolate(lg, size=(RES, RES), mode='bilinear',
                                   align_corners=False)
            lg[:, VOID] = float('-inf')
            pred = lg.argmax(1).cpu().numpy().astype(np.uint8)
        finally:
            if h is not None:
                h.remove()
        m = torch.from_numpy(np.isin(pred, (1, 2)).astype(np.float32))
        guide = (F.adaptive_avg_pool2d(m[:, None], (GRID, GRID))[:, 0]
                 >= OCC).numpy()
        return pred, guide

    variants = [('frozen MIRAGE', None)]
    for stem in a.adapters.split(','):
        f = SRC / ('adapter_%s.pt' % stem.strip())
        if not f.exists():
            hits = sorted(SRC.glob('adapter_%s*.pt' % stem.strip()))
            f = hits[0] if hits else f
        if not f.exists():
            print('missing %s -- skipped' % f.name)
            continue
        ck = torch.load(f, map_location=dev)
        m = Adapter(ck['cfg']['ch'], ck['cfg']['depth'], ck['cfg']['width'],
                    ck['cfg']['alpha']).to(dev)
        m.load_state_dict(ck['state_dict']); m.eval()
        nm = (stem.strip().replace('_gram_mse', '\nGram-MSE')
              .replace('_structl', '\nstructural lam='))
        variants.append((nm, m))
    if len(variants) == 1:
        raise SystemExit('none of the requested adapters loaded: %s' % a.adapters)
    print('variants: %s' % ', '.join(n.replace('\n', ' ') for n, _ in variants))

    preds, guides = {}, {}
    for nm, m in variants:
        preds[nm], guides[nm] = predict(m)

    base = variants[0][0]
    ncol = 1 + len(variants)
    fig, ax = plt.subplots(a.n, ncol, figsize=(2.45 * ncol, 2.6 * a.n))
    for r in range(a.n):
        ax[r, 0].imshow(imgs[r], cmap='gray')
        ax[r, 0].set_ylabel('slice %s' % depths[r], fontsize=8)
        if r == 0:
            ax[r, 0].set_title('FairVision B-scan', fontsize=9)
        for c, (nm, _) in enumerate(variants):
            im = colourise(preds[nm][r]).copy()
            if nm != base:
                diff = preds[nm][r] != preds[base][r]
                im[diff] = (255, 40, 40)
            ax[r, 1 + c].imshow(im)
            if nm == base:
                t = nm
            else:
                ch = 100 * (preds[nm][r] != preds[base][r]).mean()
                t = '%s\n%.2f%% px changed' % (nm, ch)
            if r == 0:
                ax[r, 1 + c].set_title(t, fontsize=7.5)
            elif nm != base:
                ax[r, 1 + c].set_title('%.2f%% changed'
                                       % (100 * (preds[nm][r] != preds[base][r]).mean()),
                                       fontsize=7.5)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    hl = [Patch(color=np.array(RGB[1]) / 255, label='inner retina'),
          Patch(color=np.array(RGB[2]) / 255, label='choroid'),
          Patch(color=(1, 0.16, 0.16), label='changed vs frozen')]
    fig.legend(handles=hl, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('FairVision segmentation: frozen MIRAGE vs JEPA-adapted '
                 r'($\alpha$=%.2f, encoder tap)' % a.alpha, fontsize=11)
    plt.tight_layout(rect=[0, 0.045, 1, 0.95])
    plt.savefig(OUT / 'fairvision_before_after.png', dpi=150)
    print('wrote', OUT / 'fairvision_before_after.png')

    fig, ax = plt.subplots(a.n, ncol - 1, figsize=(2.3 * (ncol - 1), 2.4 * a.n))
    if a.n == 1:
        ax = ax[None]
    for r in range(a.n):
        for c, (nm, _) in enumerate(variants):
            g = guides[nm][r]
            im = np.zeros(g.shape + (3,), np.uint8)
            im[g] = (70, 200, 120)
            if nm != base:
                gone = guides[base][r] & ~g
                new = g & ~guides[base][r]
                im[gone] = (200, 60, 60); im[new] = (250, 220, 60)
            ax[r, c].imshow(im, interpolation='nearest')
            if r == 0:
                ax[r, c].set_title(nm, fontsize=7.5)
            if nm != base:
                inter = (g & guides[base][r]).sum()
                union = max((g | guides[base][r]).sum(), 1)
                ax[r, c].set_xlabel('J=%.3f' % (inter / union), fontsize=7.5)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    hl = [Patch(color=(70 / 255, 200 / 255, 120 / 255), label='guide cell'),
          Patch(color=(200 / 255, 60 / 255, 60 / 255), label='lost vs frozen'),
          Patch(color=(250 / 255, 220 / 255, 60 / 255), label='gained vs frozen')]
    fig.legend(handles=hl, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('The 16x16 masking guide the sampler consumes', fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUT / 'fairvision_guide_change.png', dpi=150)
    print('wrote', OUT / 'fairvision_guide_change.png')

    stats = {}
    for nm, _ in variants[1:]:
        px = float(np.mean([(preds[nm][r] != preds[base][r]).mean()
                            for r in range(a.n)]))
        jj = float(np.mean([
            (guides[nm][r] & guides[base][r]).sum()
            / max((guides[nm][r] | guides[base][r]).sum(), 1)
            for r in range(a.n)]))
        stats[nm] = {'pixels_changed_pct': 100 * px, 'guide_jaccard': jj}
        print('  %-30s %.2f%% pixels changed, guide J %.3f' % (nm, 100 * px, jj))
    (OUT / 'fairvision_visual_stats.json').write_text(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

