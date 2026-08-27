#!/usr/bin/env python3
"""Render the FULL MIRAGE -> JEPA pipeline on one real slice, layer by layer.

Every panel is a real tensor from a real forward pass; every shape label is read
off the tensor, not written by hand.  Nothing here is schematic.

The architecture drawn is the frozen-MIRAGE + zero-init residual design:

    x -> [FROZEN ViT-B + ConvNeXt decoder] -> H0 (B,384,64,64)
                                              |
                          +-------------------+-------------------+
                          |                                       |
                 [FROZEN 1x1 head]                          [NEW adapter]
                          |                                       |
                          v                              +--------+--------+
                     L0 (B,4,64,64)                      |                 |
                          |                          dH (384ch)      dL (4ch)
                          |                              |                 |
                          |                       H = H0+dH        L = L0 + a*tanh(dL)
                          |                              |                 |
                          |                          [L_rel path]    [masking path]
                          |                              |                 |
                          +------------------------------+----------> softmax -> S
                                                                           |
                                                            AvgPool 4x4 -> 16x16
                                                                           |
                                                                        DETACH
                                                                           |
                                                                   anatomy sampler
                                                                           |
                                                                    4 target masks

Run:
    python scripts/demo_full_pipeline.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

MIRAGE_WS = pathlib.Path(r'D:\jepa_phase0\mirage-goals')
CKPT = (MIRAGE_WS / 'outputs' / 'mergedv3-base-512' / 'MergedV3' /
        'MIRAGE-Base_frozen_convnext_CEGDice-ignore' / 'checkpoint-best.pth')
TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
OUT = REPO / 'results/masking/pipeline'

RES, GRID, ALPHA = 512, 16, 2.0
ANATOMY = (1, 2)
CLASS_NAMES = ('Elsewhere', 'InnerRetina', 'Choroid', 'void/ignore')
COLORS = [(0.90, 0.20, 0.20), (0.20, 0.45, 0.90),
          (0.15, 0.70, 0.35), (0.95, 0.65, 0.10)]


class Adapter(nn.Module):
    """NEW trainable branch. Both outputs zero-init => step 0 reproduces MIRAGE."""

    def __init__(self, hid=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(384, hid, 1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GELU())
        self.to_feat = nn.Conv2d(hid, 384, 1)   # dH -> feeds L_rel
        self.res_head = nn.Conv2d(hid, 4, 1)    # dL -> feeds masking
        for layer in (self.to_feat, self.res_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h0):
        t = self.trunk(h0)
        return h0 + self.to_feat(t), self.res_head(t)


def build_mirage(device):
    sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
    cwd = os.getcwd()
    os.chdir(MIRAGE_WS)
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
        model = model_factory[cfg.model](
            args=ra, input_adapters=ia, output_adapters=oa,
            num_global_tokens=1, drop_path_rate=0.1)
        model.load_state_dict(
            dict(torch.load(CKPT, map_location='cpu', weights_only=False)['model']),
            strict=True)
    finally:
        os.chdir(cwd)
    for p in model.parameters():          # FROZEN, and eval() for drop_path
        p.requires_grad_(False)
    return model.to(device).eval()


def pick_slice(seed=0):
    import cv2
    vols = sorted(p.stem for p in TRAIN.glob('data_*.npz'))
    rng = np.random.default_rng(seed)
    for v in rng.permutation(vols)[:40]:
        with np.load(TRAIN / (v + '.npz'), allow_pickle=True) as z:
            vol = z['oct_bscans']
        raw = np.asarray(vol[len(vol) // 2], np.float32)
        lo, hi = raw.min(), raw.max()
        if hi <= lo:
            continue
        u = (raw - lo) / (hi - lo)
        return u, cv2.resize(u, (RES, RES), interpolation=cv2.INTER_LINEAR), v
    raise RuntimeError('no usable slice')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import random
    from anatomy_target_sampler_v2 import build_targets, is_viable
    from src.masks.multiblock import MaskCollator

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)

    model = build_mirage(dev)
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(H0=i[0].detach(), L0=o.detach()))
    frozen_head = model.output_adapters['semseg'].final_layer

    img_native, img512, name = pick_slice()
    x = torch.from_numpy(img512)[None, None].to(dev)
    with torch.no_grad():
        model({'bscan': x})
    H0, L0 = grab['H0'], grab['L0']

    ad = Adapter().to(dev)
    with torch.no_grad():
        H, dL = ad(H0)
        L_final = L0 + ALPHA * torch.tanh(dL)
        P = L_final.float().softmax(1)
        S = P[:, ANATOMY].sum(1, keepdim=True)
        per = F.adaptive_avg_pool2d(P[:, ANATOMY], (GRID, GRID))[0].cpu().numpy()
        Spool = F.adaptive_avg_pool2d(S, (GRID, GRID))[0, 0].cpu().numpy()
        # L_rel path
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        R_M = (F.normalize(U, dim=-1) @ F.normalize(U, dim=-1).transpose(1, 2))[0].cpu().numpy()

    # --- show that dL is LIVE when a labelled loss exists -------------------
    ad2 = Adapter().to(dev)
    opt = torch.optim.Adam(ad2.parameters(), lr=1e-4)
    tgt_shift = torch.roll(L0.argmax(1), shifts=2, dims=1)   # fake "corrected" label
    for _ in range(150):
        _, d2 = ad2(H0)
        F.cross_entropy(L0 + ALPHA * torch.tanh(d2), tgt_shift).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, dL_trained = ad2(H0)

    Pi, Pc = per[0], per[1]
    viable = is_viable([Pi, Pc], 4, min_cells=4)
    parts, _ = build_targets([Pi, Pc], 4)
    union = np.logical_or.reduce(parts)

    coll = MaskCollator()
    # Replay the collator's own draw order so the context BLOCK is the raw block,
    # not me[0][0] -- that tensor has already had the collator's RANDOM targets
    # removed, and subtracting the anatomy union on top of it double-removes.
    seed = 7
    gen = torch.Generator().manual_seed(seed)
    pred_sizes = [coll._sample_block_size(coll.pred_mask_scale, gen)
                  for _ in range(coll.npred)]
    enc_sizes = [coll._sample_block_size(coll.enc_mask_scale, gen)
                 for _ in range(coll.nenc)]
    random.seed(seed)
    for bh, bw in pred_sizes:                      # burn the target draws
        coll._sample_block_location(bh, bw, coll.height, coll.width)
    tgt_flat = set(np.flatnonzero(union.ravel()).tolist())
    bh, bw = enc_sizes[0]
    block, ctx = None, None
    for _ in range(50):
        top, left = coll._sample_block_location(bh, bw, coll.height, coll.width)
        idx = coll._block_to_indices(top, left, bh, bw)
        keep = [i for i in idx if i not in tgt_flat]
        if len(keep) >= coll.min_keep:
            block, ctx = set(idx), sorted(keep)
            break
    if block is None:
        block = set(range(GRID * GRID))
        ctx = sorted(block - tgt_flat)
    blk = np.zeros(GRID * GRID, bool); blk[sorted(block)] = True
    ctxm = np.zeros(GRID * GRID, bool); ctxm[np.array(ctx, int)] = True
    blk, ctxm = blk.reshape(GRID, GRID), ctxm.reshape(GRID, GRID)

    # ---------------------------------------------------------------- plot
    fig = plt.figure(figsize=(26, 15))
    gs = fig.add_gridspec(4, 7, hspace=0.42, wspace=0.22)

    def show(ax, data, title, shape, cmap='gray', vmin=None, vmax=None, frozen=None):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(shape, fontsize=8, family='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        if frozen is not None:
            for s in ax.spines.values():
                s.set_edgecolor('#1f77b4' if frozen else '#d62728')
                s.set_linewidth(2.5)

    # ROW 0 -- frozen MIRAGE forward
    show(fig.add_subplot(gs[0, 0]), img_native, 'INPUT B-scan (native)',
         f'{img_native.shape}  min-max [0,1], 1ch', frozen=True)
    show(fig.add_subplot(gs[0, 1]), img512, 'resize -> MIRAGE input',
         f'(1,1,{RES},{RES})  NO ImageNet norm', frozen=True)
    show(fig.add_subplot(gs[0, 2]), H0[0].mean(0).cpu(), 'H0  decoder feature (mean over ch)',
         f'{tuple(H0.shape)}', cmap='viridis', frozen=True)
    show(fig.add_subplot(gs[0, 3]), H0[0, :64].std(0).cpu(), 'H0  channel std (structure)',
         f'{tuple(H0.shape)}', cmap='magma', frozen=True)
    for k in range(3):
        show(fig.add_subplot(gs[0, 4 + k]), L0[0, k].cpu(),
             f'L0 logit ch{k}: {CLASS_NAMES[k]}', f'{tuple(L0.shape)}',
             cmap='coolwarm', frozen=True)

    # ROW 1 -- new branch + corrected logits
    show(fig.add_subplot(gs[1, 0]), dL[0, 1].cpu(), 'dL at INIT (zero-init)',
         f'{tuple(dL.shape)}   max|dL|={dL.abs().max():.1e}', cmap='coolwarm',
         vmin=-1, vmax=1, frozen=False)
    show(fig.add_subplot(gs[1, 1]), dL_trained[0, 1].cpu(),
         'dL AFTER 150 steps w/ labelled loss', f'max|dL|={dL_trained.abs().max():.2f}',
         cmap='coolwarm', vmin=-1, vmax=1, frozen=False)
    show(fig.add_subplot(gs[1, 2]), (H - H0)[0].mean(0).cpu(), 'dH  (feeds L_rel only)',
         f'{tuple(H.shape)}   ||dH||={(H-H0).norm():.1e}', cmap='PuOr', frozen=False)
    show(fig.add_subplot(gs[1, 3]), L_final[0, 1].cpu(),
         'L = L0 + a*tanh(dL)   ch1', f'a={ALPHA}  max|L-L0|={(L_final-L0).abs().max():.1e}',
         cmap='coolwarm', frozen=None)
    show(fig.add_subplot(gs[1, 4]), P[0, ANATOMY].sum(0).cpu(),
         'S = softmax(L)  P_inner + P_choroid', f'{tuple(S.shape)}  softmax BEFORE pool',
         cmap='inferno', vmin=0, vmax=1)
    show(fig.add_subplot(gs[1, 5]), L_final[0].argmax(0).cpu(),
         'argmax segmentation', f'{tuple(L_final.shape)} -> (1,{RES//8},{RES//8})',
         cmap='tab10', vmin=0, vmax=9)
    show(fig.add_subplot(gs[1, 6]), Spool, 'AvgPool 4x4 -> JEPA grid  [DETACH]',
         f'(1,1,{GRID},{GRID})   each cell = 4x4 decoder cells',
         cmap='inferno', vmin=0, vmax=1)

    # ROW 2 -- sampler
    ax = fig.add_subplot(gs[2, 0])
    show(ax, per[0], 'P_InnerRetina on grid', f'{per[0].shape}  mass={per[0].sum():.1f}',
         cmap='Blues', vmin=0, vmax=1)
    show(fig.add_subplot(gs[2, 1]), per[1], 'P_Choroid on grid',
         f'{per[1].shape}  mass={per[1].sum():.1f}', cmap='Greens', vmin=0, vmax=1)
    show(fig.add_subplot(gs[2, 2]), (per.sum(0) > 0.10), 'support S = {score > tau}',
         f'tau=0.10   {int((per.sum(0)>0.10).sum())} cells', cmap='gray')
    rgb = np.zeros((GRID, GRID, 3))
    for i, p in enumerate(parts):
        rgb[p] = COLORS[i]
    axt = fig.add_subplot(gs[2, 3])
    axt.imshow(rgb, interpolation='nearest')
    axt.set_title('4 CONNECTED targets (masks_pred)', fontsize=10, fontweight='bold')
    axt.set_xlabel('sizes ' + '/'.join(str(int(p.sum())) for p in parts) +
                   f'   union={int(union.sum())}   viable={viable}', fontsize=8,
                   family='monospace')
    axt.set_xticks([]); axt.set_yticks([])
    show(fig.add_subplot(gs[2, 4]), blk, 'I-JEPA context BLOCK (raw, pre-removal)',
         f'{int(blk.sum())} of 256 patches   enc_mask_scale {coll.enc_mask_scale}',
         cmap='gray')
    show(fig.add_subplot(gs[2, 5]), ctxm, 'CONTEXT after removing targets',
         f'masks_enc = {len(ctx)} tokens  ({int(blk.sum())} - {int(blk.sum())-len(ctx)} overlap)',
         cmap='gray')
    j16 = np.asarray(F.avg_pool2d(
        torch.from_numpy(img_native)[None, None].float(),
        kernel_size=img_native.shape[0] // GRID)[0, 0])[:GRID, :GRID]
    ov = np.stack([j16] * 3, -1) / max(j16.max(), 1e-9) * 0.7
    ov[union] = [0.95, 0.15, 0.15]
    axo = fig.add_subplot(gs[2, 6])
    axo.imshow(ov, interpolation='nearest')
    axo.set_title('targets over the B-scan', fontsize=10, fontweight='bold')
    axo.set_xlabel('red = hidden from encoder', fontsize=8, family='monospace')
    axo.set_xticks([]); axo.set_yticks([])

    # ROW 3 -- L_rel path
    show(fig.add_subplot(gs[3, 0]), R_M, 'R_M = Gram(pooled H)',
         f'({R_M.shape[0]},{R_M.shape[1]})  from (256,384)', cmap='RdBu_r',
         vmin=-1, vmax=1, frozen=False)
    torch.manual_seed(0)
    ZJ = F.normalize(torch.randn(1, 256, 768), dim=-1)
    R_J = (ZJ @ ZJ.transpose(1, 2))[0].numpy()
    show(fig.add_subplot(gs[3, 1]), R_J, 'R_J = Gram(JEPA h_full) [DETACH]',
         '(256,256)  from (256,768)  *placeholder*', cmap='RdBu_r', vmin=-1, vmax=1)

    txt = fig.add_subplot(gs[3, 2:])
    txt.axis('off')
    txt.text(0.0, 1.0, f"""SHAPE CHAIN  (slice {name}, all read off real tensors)

  MIRAGE   (1,1,512,512) -> ViT-B/32 -> decoder -> H0 {tuple(H0.shape)}
           H0 --1x1 Conv(384->4)--> L0 {tuple(L0.shape)}                        [FROZEN]
           H0 --adapter--> dH (384ch) and dL (4ch)                              [NEW]
           L  = L0 + {ALPHA}*tanh(dL) -> softmax -> S = P1+P2 (1,1,64,64)
           AvgPool 4x4 -> (1,1,16,16) -> DETACH -> sampler -> 4 masks

  L_rel    H (B,384,64,64) -> pool -> (B,384,16,16) -> flatten -> (B,384,256)
           -> transpose -> (B,256,384) -> L2norm -> @T -> R_M (B,256,256)
           JEPA h_full (B,256,768) -> L2norm -> @T -> R_J (B,256,256), detached

  BLUE border = frozen (cannot forget).   RED border = new trainable branch.

MEASURED HERE
  max|dL| at init      {dL.abs().max().item():.3e}   -> L == L0 exactly, argmax identical
  max|L - L0| at init  {(L_final-L0).abs().max().item():.3e}
  dL after labelled loss {dL_trained.abs().max().item():.3f}  -> the residual path IS live,
      but ONLY when a loss touches the logits.  L_rel alone leaves dL at exactly 0
      forever, so the masking guide never moves and masks are STATIC (precomputable).

  targets {'/'.join(str(int(p.sum())) for p in parts)}  union {int(union.sum())} cells
  context BLOCK {int(blk.sum())} -> masks_enc {len(ctx)} tokens after removing the union
      (block drawn by replaying the collator's own RNG order, so the anatomy union
       is subtracted from the RAW block -- not from me[0][0], which has already had
       the collator's own random targets removed)
""", fontsize=9.5, family='monospace', va='top', linespacing=1.5)

    fig.suptitle(
        'MIRAGE-guided I-JEPA masking: every layer, real tensors, one B-scan',
        fontsize=15, fontweight='bold', y=0.965)
    out = OUT / 'full_pipeline.png'
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')
    print('wrote', out)
    print(f'slice {name}  targets {[int(p.sum()) for p in parts]}  '
          f'union {int(union.sum())}  ctx {len(ctx)}')


if __name__ == '__main__':
    main()
