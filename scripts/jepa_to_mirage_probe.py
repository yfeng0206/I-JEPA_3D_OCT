#!/usr/bin/env python3
"""Does the JEPA representation actually change MIRAGE?  ep30 vs ep100.

The distillation path under test:

    JEPA EMA TARGET encoder (full image, no context mask)
        h_full (B,256,768) -> LayerNorm -> L2 -> Gram R_J (B,256,256) -> stop-grad
    MIRAGE
        H = H0 + A(H0)     -> pool 4x4 -> (B,256,384) -> L2 -> Gram R_M
    L_rel = MSE(R_M, sg(R_J))

NOTE it is the TARGET encoder, not the context encoder.  The context encoder
only ever sees the unmasked subset, so its output is not defined on the 256-token
grid that R_J needs.

Two teachers are compared so the question "what changes after we learn from
JEPA" has a control: if ep30 and ep100 push MIRAGE to the same place, then what
MIRAGE learns is not specific to what JEPA learned.

CAVEAT: both checkpoints come from patch_mirage_envelope, a run whose own masks
were MIRAGE-guided.  This is circular and is why the result is reported as a
sensitivity measurement, not as evidence that JEPA improves segmentation.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

MW = pathlib.Path(r'D:\jepa_phase0\mirage-goals')
CK_MIRAGE = (MW / 'outputs/mergedv3-base-512/MergedV3/'
                  'MIRAGE-Base_frozen_convnext_CEGDice-ignore/checkpoint-best.pth')
RUNS = pathlib.Path(r'D:\jepa_phase0\runs\patch_mirage_envelope')
TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
OUT = REPO / 'results/masking/jepa_to_mirage'
RES, GRID, ANATOMY = 512, 16, (1, 2)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
ALPHA = 2.0


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
            num_classes=4, preds_per_patch=16, depth=4, interpolate_mode='bilinear',
            main_tasks=['bscan'], embed_dim=6144, patch_size=[32, 32],
            task='semseg', image_size=[RES, RES])}
        m = model_factory[cfg.model](args=ra, input_adapters=ia, output_adapters=oa,
                                     num_global_tokens=1, drop_path_rate=0.1)
        m.load_state_dict(dict(torch.load(CK_MIRAGE, map_location='cpu',
                                          weights_only=False)['model']), strict=True)
    finally:
        os.chdir(cwd)
    return m.to(device).eval()


def build_jepa(ckpt, device):
    from src.models.vision_transformer import vit_base
    enc = vit_base(img_size=[256], patch_size=16)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['target_encoder']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not [k for k in missing if 'decoder' not in k], missing
    return enc.to(device).eval()


def dump(out_npz, n_slices, epochs):
    import cv2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mir = build_mirage(dev)
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach(), L0=o.detach()))

    vols = sorted(p.stem for p in TRAIN.glob('data_*.npz'))[:n_slices]
    imgs512, imgs256, names = [], [], []
    for v in vols:
        with np.load(TRAIN / (v + '.npz'), allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = len(vol) // 2
        raw = np.asarray(vol[d], np.float32)
        lo, hi = raw.min(), raw.max()
        u = (raw - lo) / (hi - lo) if hi > lo else raw * 0
        imgs512.append(cv2.resize(u, (RES, RES), interpolation=cv2.INTER_LINEAR))
        imgs256.append(cv2.resize(u, (256, 256), interpolation=cv2.INTER_LINEAR))
        names.append('%s:%d' % (v, d))

    H0, L0 = [], []
    for s in imgs512:
        x = torch.from_numpy(s)[None, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
        H0.append(grab['H'].cpu())
        L0.append(grab['L0'].cpu())
    H0 = torch.cat(H0)
    L0 = torch.cat(L0)
    fl = mir.output_adapters['semseg'].final_layer
    mir_head_w = fl.weight.detach().cpu()
    mir_head_b = fl.bias.detach().cpu()
    del mir
    torch.cuda.empty_cache()

    out = {'H0': H0.numpy(), 'L0': L0.numpy(),
           'head_w': mir_head_w.numpy(), 'head_b': mir_head_b.numpy(),
           'img256': np.array(imgs256, np.float32), 'names': np.array(names)}
    for ep in epochs:
        ck = RUNS / ('jepa_patch_mirage-ep%d.pth.tar' % ep)
        enc = build_jepa(ck, dev)
        zs = []
        for s in imgs256:
            rgb = np.repeat(s[..., None], 3, -1)
            rgb = (rgb - IMNET_MEAN) / IMNET_STD
            x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
            with torch.no_grad():
                h = enc(x)
                h = F.layer_norm(h, (h.size(-1),))   # matches train_patch.py:631
            zs.append(h.cpu())
        out['zj_ep%d' % ep] = torch.cat(zs).numpy()
        del enc
        torch.cuda.empty_cache()
        print('  JEPA ep%d -> %s' % (ep, out['zj_ep%d' % ep].shape))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    print('wrote %s' % out_npz)


class Adapter(nn.Module):
    """FairVision-only wiring.

        H = H0 + alpha * tanh(A(H0))          bounded residual on FEATURES
        L = FrozenSegHead(H)                  the frozen head sees the ADAPTED H

    Zero-init output => A(H0)=0 => H=H0 => L=L0 at step 0, i.e. the system
    starts as exactly pretrained MIRAGE.

    This differs from an earlier variant that routed the adapter into a SEPARATE
    trainable residual logit head and left the frozen head reading H0.  That
    wiring was measured to be a structural dead end: the frozen head never saw
    the adapted features, the residual head received 0.000e+00 gradient without a
    labelled L_seg, and segmentation/masks were bit-identical (agreement and
    Jaccard both exactly 1.000000).  Keep the frozen head on the adapted path.
    """

    def __init__(self, hid=64, alpha=ALPHA):
        super().__init__()
        self.alpha = alpha
        self.trunk = nn.Sequential(
            nn.Conv2d(384, hid, 1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hid, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h0):
        return h0 + self.alpha * torch.tanh(self.out(self.trunk(h0)))


def gram(x):
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)


def analyze(npz, out_dir, epochs, steps, lr, lam):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import anatomy_target_sampler_v2 as A

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(npz, allow_pickle=True)
    H0 = torch.tensor(z['H0']).to(dev)
    L0 = torch.tensor(z['L0']).to(dev)
    n = H0.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # the ORIGINAL MIRAGE segmentation head, frozen, applied to the ADAPTED H
    head = nn.Conv2d(384, 4, 1).to(dev)
    head.weight.data = torch.tensor(z['head_w']).to(dev)
    head.bias.data = torch.tensor(z['head_b']).to(dev)
    for p in head.parameters():
        p.requires_grad_(False)

    RJ = {}
    for ep in epochs:
        RJ[ep] = gram(torch.tensor(z['zj_ep%d' % ep]).to(dev)).detach()

    rep = {'n_slices': n, 'steps': steps, 'lr': lr, 'lambda': lam,
           'alpha_tanh': ALPHA}

    # how different are the two teachers from each other?
    if len(epochs) == 2:
        a, b = epochs
        rep['teacher_gram_cosine_ep%d_vs_ep%d' % (a, b)] = float(
            F.cosine_similarity(RJ[a].reshape(n, -1), RJ[b].reshape(n, -1),
                                dim=-1).mean())
        rep['teacher_gram_mse'] = float(F.mse_loss(RJ[a], RJ[b]))

    def masks_from(logits):
        P = logits.float().softmax(1)
        g = F.adaptive_avg_pool2d(P[:, ANATOMY], (GRID, GRID)).cpu().numpy()
        return np.array([np.logical_or.reduce(
            A.build_targets([g[i, 0], g[i, 1]], 4)[0]) for i in range(g.shape[0])])

    base_masks = masks_from(L0)
    base_area = torch.from_numpy(
        F.adaptive_avg_pool2d(L0.float().softmax(1)[:, ANATOMY],
                              (GRID, GRID)).sum((1, 2, 3)).cpu().numpy())

    trained = {}
    for ep in epochs:
        torch.manual_seed(0)
        mod = Adapter().to(dev)
        opt = torch.optim.Adam(mod.parameters(), lr=lr)
        hist = []
        for s in range(steps):
            H = mod(H0)
            U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            l_rel = F.mse_loss(gram(U), RJ[ep])
            # FairVision-only objective: L_rel alone trains the adapter.
            (lam * l_rel).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if s % 20 == 0 or s == steps - 1:
                hist.append((s, float(l_rel), 0.0))
        with torch.no_grad():
            H = mod(H0)
            Lf = head(H)                      # frozen head on the ADAPTED features
            U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            m = masks_from(Lf)
            inter = (m & base_masks).sum(axis=(1, 2))
            uni = (m | base_masks).sum(axis=(1, 2)).clip(min=1)
            Pf = Lf.float().softmax(1)
            gf = F.adaptive_avg_pool2d(Pf[:, ANATOMY], (GRID, GRID))
            g0 = F.adaptive_avg_pool2d(L0.float().softmax(1)[:, ANATOMY],
                                       (GRID, GRID))
            rep['ep%d' % ep] = {
                'L_rel_start': hist[0][1], 'L_rel_end': hist[-1][1],
                'L_rel_reduction_pct': 100 * (hist[0][1] - hist[-1][1]) / hist[0][1],
                'feature_drift': float((H - H0).norm() / H0.norm()),
                'max_abs_dL': float((Lf - L0).abs().max()),
                'mean_abs_dL': float((Lf - L0).abs().mean()),
                'argmax_agreement': float((Lf.argmax(1) == L0.argmax(1)).float().mean()),
                'mask_jaccard': float((inter / uni).mean()),
                'mask_cells_before': float(base_masks.sum(axis=(1, 2)).mean()),
                'mask_cells_after': float(m.sum(axis=(1, 2)).mean()),
                'anatomy_score_max_abs_change': float((gf - g0).abs().max()),
                'anatomy_score_mean_abs_change': float((gf - g0).abs().mean()),
                'anatomy_area_change': float(
                    (gf.sum((1, 2, 3)).cpu() - base_area).abs().mean()),
                'gram_cosine_with_teacher': float(F.cosine_similarity(
                    gram(U).reshape(n, -1), RJ[ep].reshape(n, -1), dim=-1).mean()),
            }
            trained[ep] = dict(H=H.cpu(), Lf=Lf.cpu(), masks=m, hist=hist)

    # do the two teachers move MIRAGE to the SAME place?
    if len(epochs) == 2:
        a, b = epochs
        Ha, Hb = trained[a]['H'].to(dev), trained[b]['H'].to(dev)
        rep['adapted_feature_cosine_between_teachers'] = float(
            F.cosine_similarity(Ha.flatten(1), Hb.flatten(1), dim=-1).mean())
        rep['adapted_feature_relative_diff'] = float(
            (Ha - Hb).norm() / ((Ha - H0).norm() + 1e-9))
        ja = trained[a]['masks']
        jb = trained[b]['masks']
        rep['mask_jaccard_between_teachers'] = float(
            ((ja & jb).sum(axis=(1, 2)) / (ja | jb).sum(axis=(1, 2)).clip(min=1)).mean())

    (out_dir / 'jepa_to_mirage.json').write_text(json.dumps(rep, indent=2))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(21, 11))
    gs = fig.add_gridspec(3, 5, hspace=0.34, wspace=0.22,
                          left=0.05, right=0.98, top=0.89, bottom=0.06)

    ax = fig.add_subplot(gs[0, 0])
    for ep in epochs:
        h = np.array(trained[ep]['hist'])
        ax.plot(h[:, 0], h[:, 1], lw=2, label='teacher ep%d' % ep)
    ax.set_xlabel('adapter step'); ax.set_ylabel(r'$L_{rel}$')
    ax.set_title('Distillation loss', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=.25)

    metrics = [('feature_drift', 'MIRAGE feature drift\n||H-H0|| / ||H0||'),
               ('argmax_agreement', 'segmentation agreement\nwith frozen MIRAGE'),
               ('mask_jaccard', 'mask Jaccard\nvs frozen MIRAGE'),
               ('gram_cosine_with_teacher', 'Gram cosine\nwith JEPA teacher')]
    for c, (k, t) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, c + 1])
        vals = [rep['ep%d' % ep][k] for ep in epochs]
        b = ax.bar([str(e) for e in epochs], vals,
                   color=['#8d99ae', '#2a9d8f'][:len(epochs)], width=.55)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width()/2, v, '%.4f' % v,
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel('JEPA teacher epoch')
        ax.grid(axis='y', alpha=.25)
        if k in ('argmax_agreement', 'mask_jaccard'):
            ax.set_ylim(0, 1.15)

    img = z['img256']
    show = min(3, n)
    for r in range(show):
        ax = fig.add_subplot(gs[1, r])
        ax.imshow(img[r], cmap='gray')
        ov = np.zeros((256, 256, 4))
        ov[np.kron(base_masks[r], np.ones((16, 16))).astype(bool)] = (.95, .15, .15, .5)
        ax.imshow(ov)
        ax.set_title('frozen MIRAGE masks  %s' % str(z['names'][r]), fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    for c, ep in enumerate(epochs[:2]):
        ax = fig.add_subplot(gs[1, 3 + c])
        d = (trained[ep]['Lf'].softmax(1)[:, ANATOMY].sum(1)
             - L0.cpu().softmax(1)[:, ANATOMY].sum(1))[0].numpy()
        v = max(abs(d).max(), 1e-8)
        im = ax.imshow(d, cmap='bwr', vmin=-v, vmax=v)
        ax.set_title('anatomy score CHANGE, teacher ep%d\nmax |change| %.2e'
                     % (ep, v), fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=.046)

    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    lines = ['%-34s %14s %14s' % ('metric', 'teacher ep%d' % epochs[0],
                                  'teacher ep%d' % epochs[-1])]
    lines.append('-' * 66)
    for k, t in [('L_rel_reduction_pct', 'L_rel reduced (%)'),
                 ('feature_drift', 'MIRAGE feature drift'),
                 ('gram_cosine_with_teacher', 'Gram cosine w/ teacher'),
                 ('argmax_agreement', 'segmentation agreement'),
                 ('mask_jaccard', 'mask Jaccard vs frozen'),
                 ('mask_cells_after', 'mask cells (frozen %.1f)'
                  % rep['ep%d' % epochs[0]]['mask_cells_before']),
                 ('max_abs_dL', 'max |delta logit|'),
                 ('anatomy_score_max_abs_change', 'max |anatomy score change|')]:
        lines.append('%-34s %14.5f %14.5f'
                     % (t, rep['ep%d' % epochs[0]][k], rep['ep%d' % epochs[-1]][k]))
    if len(epochs) == 2:
        lines += ['', 'ARE THE TWO TEACHERS DIFFERENT?',
                  '  teacher Gram cosine ep%d vs ep%d      %.6f'
                  % (epochs[0], epochs[1],
                     rep['teacher_gram_cosine_ep%d_vs_ep%d' % tuple(epochs)]),
                  '  adapted MIRAGE feature cosine         %.6f'
                  % rep['adapted_feature_cosine_between_teachers'],
                  '  mask Jaccard between the two          %.6f'
                  % rep['mask_jaccard_between_teachers']]
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left',
            family='monospace', fontsize=11, linespacing=1.5)

    fig.suptitle('What does JEPA actually change in MIRAGE?   '
                 'EMA target encoder ep%d vs ep%d,  %d slices,  %d adapter steps'
                 % (epochs[0], epochs[-1], n, steps), fontsize=14, y=0.955)
    f = out_dir / 'jepa_to_mirage.png'
    fig.savefig(f, dpi=115, facecolor='white')
    print('wrote %s' % f)

    print('\n%-34s %14s %14s' % ('metric', 'ep%d' % epochs[0], 'ep%d' % epochs[-1]))
    for k in ('L_rel_reduction_pct', 'feature_drift', 'gram_cosine_with_teacher',
              'argmax_agreement', 'mask_jaccard', 'max_abs_dL'):
        print('%-34s %14.6f %14.6f'
              % (k, rep['ep%d' % epochs[0]][k], rep['ep%d' % epochs[-1]][k]))
    if len(epochs) == 2:
        print('\nteacher Gram cosine ep%d vs ep%d : %.6f'
              % (epochs[0], epochs[1],
                 rep['teacher_gram_cosine_ep%d_vs_ep%d' % tuple(epochs)]))
        print('adapted feature cosine between   : %.6f'
              % rep['adapted_feature_cosine_between_teachers'])
        print('mask Jaccard between teachers    : %.6f'
              % rep['mask_jaccard_between_teachers'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--analyze-from', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    ap.add_argument('--n-slices', type=int, default=24)
    ap.add_argument('--epochs', type=int, nargs='+', default=[30, 100])
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lam', type=float, default=1.0)
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_slices, a.epochs)
    if a.analyze_from:
        return analyze(a.analyze_from, a.out, a.epochs, a.steps, a.lr, a.lam)
    ap.error('need --dump or --analyze-from')


if __name__ == '__main__':
    main()
