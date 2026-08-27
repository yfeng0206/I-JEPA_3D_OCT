#!/usr/bin/env python3
"""Full MIRAGE -> JEPA masking pipeline, every stage rendered with its shape.

Two figures:
  pipeline_trace.png   one B-scan through every tensor in the chain
  class_balance.png    inner-retina vs choroid masking, random / oracle / anatomy

Run with the venv that has BOTH matplotlib and MIRAGE:
  D:\\jepa_phase0\\.venv\\Scripts\\python.exe scripts/demo_pipeline_trace.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

MW = pathlib.Path(r'D:\jepa_phase0\mirage-goals')
CK = (MW / 'outputs/mergedv3-base-512/MergedV3/'
           'MIRAGE-Base_frozen_convnext_CEGDice-ignore/checkpoint-best.pth')
TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
OUT = REPO / 'results/masking/pipeline'
RES, GRID, TAU = 512, 16, 0.10
ANATOMY = (1, 2)
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.15, 0.70, 0.30), (0.95, 0.65, 0.10)]


def build_mirage(device):
    from argparse import Namespace
    if str(MW / 'MIRAGE') not in sys.path:
        sys.path.insert(0, str(MW / 'MIRAGE'))
    cwd = os.getcwd()
    os.chdir(MW)
    try:
        from fm_seg_config import fm_factory
        from mirage.model import model_factory
        from mirage.output_adapters import ConvNeXtAdapter
        g = RES // 32
        cfg = fm_factory['mirage-base']()
        cfg.build_domain_conf()
        ra = Namespace(grid_sizes={'bscan': [g, g]}, input_size={'bscan': [RES, RES]})
        ia = {'bscan': cfg.domain_conf['bscan']['input_adapter'](
            stride_level=1, patch_size_full=[32, 32], image_size=[RES, RES],
            learnable_pos_emb=False)}
        oa = {'semseg': ConvNeXtAdapter(
            num_classes=4, preds_per_patch=16, depth=4,
            interpolate_mode='bilinear', main_tasks=['bscan'], embed_dim=6144,
            patch_size=[32, 32], task='semseg', image_size=[RES, RES])}
        m = model_factory[cfg.model](args=ra, input_adapters=ia, output_adapters=oa,
                                     num_global_tokens=1, drop_path_rate=0.1)
        sd = dict(torch.load(CK, map_location='cpu', weights_only=False)['model'])
        m.load_state_dict(sd, strict=True)
    finally:
        os.chdir(cwd)
    return m.to(device).eval()


def oracle_band(profile16):
    """Intensity/anatomy-centroid ribbon, from src/masks/curriculum.py:893-940."""
    H = W = GRID
    RF, LF, MR, eps = 0.28, 0.6, 3, 1e-6
    prof = torch.from_numpy(profile16 - profile16.min()).float()
    ys = torch.arange(H, dtype=torch.float32)
    rm = prof.sum(1)
    tot = float(rm.sum())
    gc = float((ys * rm).sum() / tot) if tot > eps else H / 2
    cm = prof.sum(0)
    cc = torch.where(cm > eps, (ys.view(H, 1) * prof).sum(0) / cm.clamp(min=eps),
                     torch.full((W,), gc))
    cc = F.avg_pool1d(cc.view(1, 1, W), 3, 1, 1, count_include_pad=False).view(W)
    bh = max(MR, min(int(round((RF / max(LF, eps)) * H)), H))
    xk = max(1, min(int(round(LF * W)), W))
    x0 = (W - xk) // 2
    O = np.zeros((H, W), bool)
    for x in range(x0, x0 + xk):
        c = int(round(float(cc[x])))
        t = max(0, min(c - bh // 2, H - bh))
        O[t:t + bh, x] = True
    return O


def dump(out_npz, n_slices, pick):
    import cv2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_mirage(device)
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach(), L0=o.detach()))

    vols = sorted(p.stem for p in TRAIN.glob('data_*.npz'))
    rec = {k: [] for k in ('img512', 'img256', 'H', 'L0', 'S64', 'grid', 'name')}
    for vi in pick[:n_slices]:
        v = vols[vi % len(vols)]
        with np.load(TRAIN / (v + '.npz'), allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = len(vol) // 2
        raw = np.asarray(vol[d], np.float32)
        lo, hi = raw.min(), raw.max()
        u = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        x512 = cv2.resize(u, (RES, RES), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(x512)[None, None].to(device)
        with torch.no_grad():
            model({'bscan': x})
        L0 = grab['L0'].float()
        P = L0.softmax(1)
        S64 = P[:, ANATOMY].sum(1)
        grid = F.adaptive_avg_pool2d(P[:, ANATOMY], (GRID, GRID))[0].cpu().numpy()
        rec['img512'].append(x512.astype(np.float32))
        rec['img256'].append(cv2.resize(u, (256, 256),
                                        interpolation=cv2.INTER_LINEAR).astype(np.float32))
        # keep only a small slice of H: full (384,64,64) per image is heavy
        rec['H'].append(grab['H'][0, :8].cpu().numpy().astype(np.float32))
        rec['L0'].append(L0[0].cpu().numpy().astype(np.float32))
        rec['S64'].append(S64[0].cpu().numpy().astype(np.float32))
        rec['grid'].append(grid.astype(np.float32))
        rec['name'].append('%s:%d' % (v, d))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{k: np.array(v) for k, v in rec.items()})
    print('wrote %s' % out_npz)


def real_context(union_flat, seed):
    """The production collator's context policy, targets replaced by ours."""
    from src.masks.multiblock import MaskCollator
    coll = MaskCollator()
    g = torch.Generator().manual_seed(seed)
    ps = [coll._sample_block_size(coll.pred_mask_scale, g) for _ in range(coll.npred)]
    es = [coll._sample_block_size(coll.enc_mask_scale, g) for _ in range(coll.nenc)]
    random.seed(seed)
    rect = set()
    for bh, bw in ps:
        t, l = coll._sample_block_location(bh, bw, GRID, GRID)
        rect.update(coll._block_to_indices(t, l, bh, bw))

    def ctx(u):
        random.seed(seed)
        for bh, bw in ps:
            coll._sample_block_location(bh, bw, GRID, GRID)
        bh, bw = es[0]
        for _ in range(50):
            t, l = coll._sample_block_location(bh, bw, GRID, GRID)
            idx = coll._block_to_indices(t, l, bh, bw)
            keep = [i for i in idx if i not in u]
            if len(keep) >= coll.min_keep:
                return set(keep), set(idx)
        return set(range(GRID * GRID)) - u, set(range(GRID * GRID))
    return rect, ctx


def plot(npz, out_dir, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import anatomy_target_sampler_v2 as A
    from demo_split_fix import old_build

    z = np.load(npz, allow_pickle=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    i = row
    img512, img256 = z['img512'][i], z['img256'][i]
    Hf, L0, S64, grid = z['H'][i], z['L0'][i], z['S64'][i], z['grid'][i]
    Pi, Pc = grid[0], grid[1]
    a = Pi + Pc
    name = str(z['name'][i])

    parts, regions = A.build_targets([Pi, Pc], 4)
    U = np.logical_or.reduce(parts)
    caps = A.region_capacity([Pi, Pc])
    viable = A.is_viable([Pi, Pc], 4, min_cells=4)
    rect, ctxfn = real_context(set(np.flatnonzero(U.ravel()).tolist()), 1234)
    ctx, block = ctxfn(set(np.flatnonzero(U.ravel()).tolist()))

    def up(m):
        return np.kron(m.astype(float), np.ones((16, 16)))

    fig = plt.figure(figsize=(26, 12.5))
    gs = fig.add_gridspec(3, 7, hspace=0.30, wspace=0.16,
                          left=0.025, right=0.985, top=0.885, bottom=0.045)
    K = dict(fontsize=10.5, pad=7)

    def panel(r, c, im, title, sub, cmap='gray', vmin=None, vmax=None):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, **K)
        ax.set_xlabel(sub, fontsize=9, color='#333')
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    # ---- row 0 : MIRAGE forward -------------------------------------------
    panel(0, 0, img512, '1  input B-scan', '(1, 512, 512)  raw min-max, NOT ImageNet')
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.concatenate([np.concatenate(list(Hf[j*2:j*2+2]), 1) for j in range(4)], 0),
              cmap='viridis')
    ax.set_title('2  decoder feature H  (8 of 384 ch)', **K)
    ax.set_xlabel('(384, 64, 64)   frozen ViT-B + ConvNeXt', fontsize=9, color='#333')
    ax.set_xticks([]); ax.set_yticks([])
    names = ['Elsewhere', 'InnerRetina', 'Choroid', 'void/ignore']
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.concatenate([np.concatenate(list(L0[:2]), 1),
                              np.concatenate(list(L0[2:]), 1)], 0), cmap='coolwarm')
    ax.set_title('3  seg logits  Conv2d(384,4,1x1)', **K)
    ax.set_xlabel('(4, 64, 64)   %s' % ' | '.join(names), fontsize=8, color='#333')
    ax.set_xticks([]); ax.set_yticks([])
    P = torch.from_numpy(L0)[None].softmax(1)[0].numpy()
    panel(0, 3, np.concatenate([np.concatenate(list(P[:2]), 1),
                                np.concatenate(list(P[2:]), 1)], 0),
          '4  softmax P', '(4, 64, 64)   softmax BEFORE pooling', 'magma', 0, 1)
    panel(0, 4, S64, '5  anatomy  S = P_inner + P_choroid',
          '(1, 64, 64)   values in [0,1]', 'magma', 0, 1)
    panel(0, 5, a, '6  AvgPool 4x4 -> JEPA grid',
          '(1, 16, 16)   1 cell = 1 JEPA patch', 'magma', 0, 1)
    ax = fig.add_subplot(gs[0, 6])
    ax.imshow(img512, cmap='gray')
    ax.imshow(np.kron(a, np.ones((32, 32))), cmap='magma', alpha=0.45, vmin=0, vmax=1)
    ax.set_title('7  grid mapped back to pixels', **K)
    ax.set_xlabel('each cell = 32x32 px at 512  (16x16 px at 256)', fontsize=9, color='#333')
    ax.set_xticks([]); ax.set_yticks([])

    # ---- row 1 : sampler ---------------------------------------------------
    panel(1, 0, (a > TAU).astype(float), '8  support  S = {score > tau}',
          'tau=%.2f   %d of 256 cells' % (TAU, int((a > TAU).sum())), 'gray')
    reg = np.zeros((GRID, GRID))
    for k, R in enumerate(regions):
        reg[R] = k + 1
    panel(1, 1, reg, '9  grow_components  per class AND per component',
          'mass_cap=0.90   capacity %s = %d cells' % (caps, sum(caps)), 'tab10', 0, 9)
    rgb = np.zeros((GRID, GRID, 3))
    for k, p in enumerate(parts):
        rgb[p] = COLORS[k]
    panel(1, 2, rgb, '10  geodesic_partition + rebalance',
          'GEOMETRY only  sizes %s' % [int(p.sum()) for p in parts])
    panel(1, 3, U.astype(float), '11  target union  (masks_pred)',
          '%d cells   viable=%s' % (int(U.sum()), viable), 'gray')
    cm = np.zeros((GRID, GRID))
    for idx in block:
        cm[idx // GRID, idx % GRID] = 1
    for idx in ctx:
        cm[idx // GRID, idx % GRID] = 2
    panel(1, 4, cm, '12  I-JEPA context block, minus targets',
          'block %d -> context %d tokens' % (len(block), len(ctx)), 'viridis', 0, 2)
    ax = fig.add_subplot(gs[1, 5])
    ax.imshow(img256, cmap='gray')
    ov = np.zeros((256, 256, 4))
    for k, p in enumerate(parts):
        m = up(p).astype(bool)
        ov[m] = (*COLORS[k], 0.55)
    ax.imshow(ov)
    ax.set_title('13  targets on the B-scan', **K)
    ax.set_xlabel('what the predictor must reconstruct', fontsize=9, color='#333')
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 6])
    cmask = np.zeros((GRID, GRID), bool)
    for idx in ctx:
        cmask[idx // GRID, idx % GRID] = True
    ax.imshow(img256 * up(cmask), cmap='gray')
    ax.set_title('14  what the ENCODER actually sees', **K)
    ax.set_xlabel('%d context tokens' % len(ctx), fontsize=9, color='#333')
    ax.set_xticks([]); ax.set_yticks([])

    # ---- row 2 : three-way comparison + the single-component fix ----------
    R = np.zeros((GRID, GRID), bool)
    for idx in rect:
        R[idx // GRID, idx % GRID] = True
    O = oracle_band(a)
    pb, _rb = old_build([Pi, Pc], 4, mass_cap=0.80)
    Ub = np.logical_or.reduce(pb)
    for c, (m, lab) in enumerate(((R, 'RANDOM rect'), (O, 'ORACLE ribbon'),
                                  (Ub, 'ANATOMY  BEFORE\ncap 0.80, single-component'),
                                  (U, 'ANATOMY  AFTER\ncap 0.90, multi-component'))):
        ax = fig.add_subplot(gs[2, c])
        ax.imshow(img256, cmap='gray')
        ovl = np.zeros((256, 256, 4))
        col = (0.15, 0.55, 0.95, 0.5) if c == 3 else (0.95, 0.15, 0.15, 0.5)
        ovl[up(m).astype(bool)] = col
        ax.imshow(ovl)
        inn = Pi[m].sum() / max(Pi.sum(), 1e-9)
        ch = Pc[m].sum() / max(Pc.sum(), 1e-9)
        ax.set_title('%s   %d cells' % (lab, int(m.sum())), **K)
        ax.set_xlabel('inner %.0f%%   choroid %.0f%%   mass %.0f%%'
                      % (100*inn, 100*ch, 100*a[m].sum()/max(a.sum(), 1e-9)),
                      fontsize=9, color='#333')
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2, 4])
    d = np.zeros((GRID, GRID))
    d[U & ~Ub] = 1
    d[Ub & ~U] = -1
    ax.imshow(d, cmap='bwr', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('recovered (red) / dropped (blue)', **K)
    ax.set_xlabel('%+d cells   %+.0f pp anatomy mass'
                  % (int(U.sum()) - int(Ub.sum()),
                     100*(a[U].sum() - a[Ub].sum())/max(a.sum(), 1e-9)),
                  fontsize=9, color='#333')
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 5:])
    ax.axis('off')
    txt = (
        'GRADIENT PATHS\n'
        '  masking   H -> FROZEN 1x1 head -> softmax -> P1+P2 -> pool\n'
        '            -> DETACH -> masks.  No gradient crosses this path.\n'
        '  distill   H -> pool 4x4 -> (256,384) -> L2 -> Gram (256,256)\n'
        '            vs JEPA h_full (256,768) -> Gram, stop-grad.\n'
        '            384 and 768 never meet; no projector.\n\n'
        'FROZEN      patchify, ViT-B, decoder, AND the seg head\n'
        'TRAINABLE   adapter only:  H = H0 + alpha*tanh(A(H0))\n'
        '            zero-init, so step 0 reproduces MIRAGE exactly\n\n'
        'The frozen head reads the ADAPTED H, which is what makes the\n'
        'path live.  Routing the adapter into a SEPARATE residual logit\n'
        'head instead was measured to be a dead end: seg agreement and\n'
        'mask Jaccard both exactly 1.000000.\n\n'
        'Measured, one pass over 6000 images, depth-2 adapter, alpha 0.5:\n'
        '  L_rel reduced 29.9%%   seg agreement 0.971   mask Jaccard 0.745\n\n'
        'this slice: %s' % name
    )
    ax.text(0, 1, txt, va='top', ha='left', fontsize=9, family='monospace',
            linespacing=1.5)

    fig.suptitle('MIRAGE-Base@512  ->  I-JEPA anatomy-guided masking : every stage '
                 'with its tensor shape', fontsize=15, y=0.955)
    f = out_dir / ('pipeline_trace_row%d.png' % row)
    fig.savefig(f, dpi=115, facecolor='white')
    plt.close(fig)
    print('wrote %s' % f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    ap.add_argument('--n-slices', type=int, default=6)
    ap.add_argument('--row', type=int, default=0)
    ap.add_argument('--pick', type=int, nargs='*', default=[3, 11, 27, 44, 61, 80])
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_slices, a.pick)
    if a.plot_from:
        return plot(a.plot_from, a.out, a.row)
    ap.error('need --dump or --plot-from')


if __name__ == '__main__':
    main()
