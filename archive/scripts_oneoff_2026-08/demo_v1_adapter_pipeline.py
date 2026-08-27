"""End-to-end V1 guided-masking pipeline: frozen MIRAGE + trainable residual adapter.

Implements exactly the agreed design and verifies its three load-bearing claims
on real data, in one forward/backward pass:

    same OCT crop
        |-- resize 1024 --> FROZEN MIRAGE --> H (384x128x128), L0 (4x128x128)
        |                                          |
        |                        tiny residual head R_phi(H) -> dL
        |                                          |
        |                    L = L0 + alpha * tanh(dL)      <- only trainable path
        |                                          |
        |                        softmax -> M = P_inner + P_choroid (128x128)
        |                                          |
        |                        avgpool 8x8 -> 16x16 location scores
        |                                          |
        |                        production block sampler -> 4 target blocks
        |                                          |
        '-- resize 256 ---------------------------> I-JEPA context / target masks

Claims verified:
  C1  zero-init  =>  L == L0 bit-exactly, so training starts at MIRAGE behaviour
  C2  gradient reaches the adapter's parameters
  C3  gradient reaches NO MIRAGE parameter (structurally impossible to forget)

Taxonomy (V3-merged): 0 Elsewhere | 1 InnerRetina | 2 Choroid | 3 Background/ignore

Two stages, because MIRAGE inference needs the MIRAGE venv (no matplotlib):
  stage 1  --dump demo.npz         (MIRAGE venv)
  stage 2  --plot-from demo.npz    (repo venv)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
        r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')

MIRAGE_RES, JEPA_RES, PATCH, GRID = 1024, 256, 16, 16
POOL = 8                      # 128 // 16
ANATOMY = (1, 2)
ALPHA = 2.0                   # chosen from scripts/mirage_logit_scale.py

CURRICULUM_CFG = {
    'mirage_min_block_fill': 0.5,
    'mirage_min_retina_visible': 0.20,
    'mirage_max_attempts': 20,
    'mirage_spread': True,
    'mirage_overlap_tolerance': 0.25,
}


class ResidualHead(nn.Module):
    """Tiny trainable correction on MIRAGE's decoder features.

    Zero-initialised final conv, so dL == 0 at step 0 and the whole system
    reproduces frozen MIRAGE exactly.  ~62k parameters against MIRAGE's 315M.
    """

    def __init__(self, in_ch=384, hidden=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h)


def aligned_crop(raw, seed):
    """One crop rectangle, applied at two output resolutions.

    The MIRAGE branch and the JEPA branch must see the same field of view or
    the guide points at the wrong anatomy, so crop parameters are drawn once.
    """
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    torch.manual_seed(seed)
    lo, hi = raw.min(), raw.max()
    unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    pil = Image.fromarray((unit * 255).astype(np.uint8))
    i, j, h, w = T.RandomResizedCrop.get_params(pil, scale=(0.3, 1.0),
                                                ratio=(3. / 4., 4. / 3.))
    bic = T.InterpolationMode.BICUBIC
    mir = TF.resized_crop(pil, i, j, h, w, [MIRAGE_RES, MIRAGE_RES], bic)
    jep = TF.resized_crop(pil, i, j, h, w, [JEPA_RES, JEPA_RES], bic)
    return (np.asarray(mir, dtype=np.float32) / 255.0,
            np.asarray(jep, dtype=np.float32) / 255.0,
            dict(top=i, left=j, height=h, width=w))


def build_mirage(device):
    from compare_512_vs_1024 import build
    return build(MIRAGE_RES, CKPT, device)


def dump(out_path, slice_name, seed, alpha, threshold):
    import os
    from fairvision_model_compare import MIRAGE_WS
    from src.masks.curriculum import CurriculumMaskGenerator

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_mirage(device)
    for p in model.parameters():                 # frozen, and asserted below
        p.requires_grad_(False)

    vol_id, _, sl = slice_name.partition(':')
    with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
        raw = np.asarray(z['oct_bscans'][int(sl)], dtype=np.float32)
    mir_img, jep_img, crop = aligned_crop(raw, seed)

    # ---- capture H (input to final_layer) and L0 (its output) ------------
    grab = {}
    layer = model.output_adapters['semseg'].final_layer
    hk = layer.register_forward_hook(
        lambda m, i, o: grab.update(H=i[0].detach(), L0=o.detach()))
    x = torch.from_numpy(mir_img)[None, None].to(device=device, dtype=torch.float32)
    with torch.no_grad():                        # MIRAGE never needs activations
        model({'bscan': x})
    hk.remove()
    H, L0 = grab['H'], grab['L0']

    head = ResidualHead(in_ch=H.shape[1]).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    n_mirage = sum(p.numel() for p in model.parameters())

    # ---- C1: zero-init reproduces MIRAGE exactly ------------------------
    dL = head(H)
    L = L0 + alpha * torch.tanh(dL)
    c1_maxdiff = float((L - L0).abs().max())

    # ---- pipeline: softmax -> anatomy -> pool ---------------------------
    P = L.softmax(dim=1)
    M = P[:, ANATOMY].sum(dim=1)                 # (1,128,128)
    scores = F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0]

    # ---- C2/C3: where does gradient go? ---------------------------------
    probe_loss = -scores.mean()                  # "prefer more anatomy" probe
    probe_loss.backward()
    head_grad = float(sum(p.grad.abs().sum() for p in head.parameters()
                          if p.grad is not None))
    named_zero = [n for n, p in head.named_parameters()
                  if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    # `.grad is None` on a requires_grad_(False) parameter proves nothing -- a
    # frozen parameter has .grad None even when gradient flows THROUGH it.  The
    # real guarantee is that the captured feature carries no autograd history,
    # so there is no path back into MIRAGE at all.  Assert that directly.
    mirage_grad = sum(1 for p in model.parameters() if p.grad is not None)
    c3_H_detached = (H.grad_fn is None) and (not H.requires_grad)
    c3_L0_detached = (L0.grad_fn is None) and (not L0.requires_grad)

    # Zero-init means the LAST conv has W = 0, so at step 0 the chain rule
    # sends exactly zero gradient to every earlier layer.  This is the standard
    # zero-init residual behaviour (ReZero / LayerScale / ControlNet), not a
    # bug: after one optimiser step the last conv is non-zero and gradient
    # reaches the whole head.  Verified rather than asserted.
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)
    L2 = L0 + alpha * torch.tanh(head(H))
    s2 = F.adaptive_avg_pool2d(
        L2.softmax(dim=1)[:, ANATOMY].sum(dim=1)[:, None], (GRID, GRID))[0, 0]
    (-s2.mean()).backward()
    named_zero_after = [n for n, p in head.named_parameters()
                        if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    step1_shift = float((s2 - scores).abs().max())

    # ---- production sampler on the resulting scores ----------------------
    occ = scores.detach().float().cpu().numpy()
    gen = CurriculumMaskGenerator(
        input_size=(JEPA_RES, JEPA_RES), patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False, curriculum_cfg=dict(CURRICULUM_CFG))
    gen.set_epoch(99, 100)
    random.seed(seed); np.random.seed(seed)
    tg = torch.Generator(); tg.manual_seed((seed * 2654435761 + 12345) % (2 ** 31))
    sizes = [gen._sample_block_size(gen.pred_mask_scale, tg) for _ in range(gen.npred)]
    gen.mirage_occupancy_threshold = threshold
    placement = occ >= threshold
    blocks, stats = gen._sample_mirage_blocks(
        sizes, torch.from_numpy(occ), torch.from_numpy(placement),
        [True] * gen.npred, [None] * gen.npred)
    union = sorted(set().union(*[set(b) for b in blocks])) if blocks else []
    context = [i for i in range(GRID * GRID) if i not in set(union)]

    # feasibility across a threshold range, since 0.25 was calibrated for the
    # repaired hard envelope rather than this soft occupancy
    feas = {}
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        feas[str(th)] = int((occ >= th).sum())

    rep = {
        'slice': slice_name, 'seed': seed, 'alpha': alpha, 'threshold': threshold,
        'crop': crop,
        'shapes': {'mirage_in': list(x.shape), 'H': list(H.shape), 'L0': list(L0.shape),
                   'M': list(M.shape), 'scores': list(scores.shape),
                   'jepa_in': [1, 3, JEPA_RES, JEPA_RES]},
        'params': {'mirage_total': n_mirage, 'adapter_total': n_head,
                   'adapter_frac_pct': 100.0 * n_head / (n_mirage + n_head)},
        'C1_zero_init_max_abs_L_minus_L0': c1_maxdiff,
        'C2_adapter_grad_abs_sum': head_grad,
        'C2_zero_grad_params_at_step0': named_zero,
        'C2_zero_grad_params_after_1_step': named_zero_after,
        'C2_score_shift_after_1_step': step1_shift,
        'C3_mirage_params_with_grad': mirage_grad,
        'C3_H_has_no_autograd_history': bool(c3_H_detached),
        'C3_L0_has_no_autograd_history': bool(c3_L0_detached),
        'sampler': {'sizes': [list(s) for s in sizes],
                    'guided_blocks': int(stats.get('guided_blocks', -1)),
                    'feasible': bool(stats.get('feasible', False)),
                    'accepted': bool(stats.get('accepted', False)),
                    'attempts': int(stats.get('attempts', -1)),
                    'mean_block_fill': float(stats.get('mean_block_fill', float('nan'))),
                    'retina_visible': float(stats.get('retina_visible', float('nan'))),
                    'n_target_cells': len(union), 'n_context_cells': len(context),
                    'admissible_cells': int(placement.sum())},
        'admissible_cells_by_threshold': feas,
        'score_mean': float(occ.mean()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, jepa_img=jep_img, mirage_img=mir_img,
        L0=L0[0].cpu().numpy(), M=M[0].detach().cpu().numpy(), scores=occ,
        placement=placement, union=np.array(union, dtype=np.int32),
        context=np.array(context, dtype=np.int32),
        blocks=np.array([np.array(b, dtype=np.int32) for b in blocks], dtype=object),
        report=json.dumps(rep))
    print(json.dumps(rep, indent=2))
    print('wrote %s' % out_path)


def plot(npz_path, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = np.load(npz_path, allow_pickle=True)
    rep = json.loads(str(z['report']))
    jep, L0, M, sc = z['jepa_img'], z['L0'], z['M'], z['scores']
    union, context, placement = z['union'], z['context'], z['placement']

    def cells(ix):
        m = np.zeros(GRID * GRID, dtype=bool)
        if len(ix):
            m[np.asarray(ix, dtype=int)] = True
        return m.reshape(GRID, GRID)

    fig, ax = plt.subplots(2, 4, figsize=(19, 9.4))
    ax = ax.ravel()

    ax[0].imshow(jep, cmap='gray')
    ax[0].set_title('1. aligned crop -> JEPA 256x256')
    ax[1].imshow(L0.argmax(0), cmap='tab10', vmin=0, vmax=9)
    ax[1].set_title('2. frozen MIRAGE argmax $L_0$\n(4x128x128, shown for reference only)')
    im = ax[2].imshow(M, cmap='viridis', vmin=0, vmax=1)
    ax[2].set_title('3. soft anatomy $M=P_1+P_2$\n128x128, no argmax')
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    im = ax[3].imshow(sc, cmap='viridis', vmin=0, vmax=1)
    ax[3].set_title('4. pooled 8x8 -> 16x16 scores\nmean %.3f' % rep['score_mean'])
    plt.colorbar(im, ax=ax[3], fraction=0.046)

    ax[4].imshow(placement, cmap='Greys_r')
    ax[4].set_title('5. admissible cells (score >= %.2f)\n%d / 256'
                    % (rep['threshold'], rep['sampler']['admissible_cells']))
    ax[5].imshow(cells(union), cmap='Reds', vmin=0, vmax=1)
    s = rep['sampler']
    ax[5].set_title('6. target blocks: %d cells\nguided %d/4  accepted=%s'
                    % (s['n_target_cells'], s['guided_blocks'], s['accepted']))
    ax[6].imshow(cells(context), cmap='Blues', vmin=0, vmax=1)
    ax[6].set_title('7. context (visible): %d cells' % s['n_context_cells'])

    up = np.kron(cells(union), np.ones((PATCH, PATCH), dtype=bool))
    ov = np.repeat(jep[..., None], 3, axis=2).copy()
    ov[up] = 0.55 * ov[up] + 0.45 * np.array([0.95, 0.15, 0.15])
    ax[7].imshow(np.clip(ov, 0, 1))
    ax[7].set_title('8. targets on the B-scan')

    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    p = rep['params']
    fig.suptitle('V1 guided masking: frozen MIRAGE (%.1fM) + residual adapter (%dk, %.3f%%)   '
                 r'$L = L_0 + %.1f\tanh(\Delta L)$   |   C1 max$|L-L_0|$=%.0e   '
                 'C2 adapter grad=%.3e   C3 MIRAGE params with grad=%d'
                 % (p['mirage_total'] / 1e6, p['adapter_total'] // 1000,
                    p['adapter_frac_pct'], rep['alpha'],
                    rep['C1_zero_init_max_abs_L_minus_L0'],
                    rep['C2_adapter_grad_abs_sum'], rep['C3_mirage_params_with_grad']),
                 fontsize=11)
    fig.tight_layout()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % out_png)


def stats_run(out_path, n_slices, n_seeds, alpha, threshold, aspect):
    """Distribution of delivered target area over many draws.

    A single draw is not enough to know whether this configuration is
    area-matched to the existing comparison arms (~102-112 target cells).  If
    it is not, any AUC difference is confounded by masked area rather than by
    the guidance being tested.
    """
    import os
    from fairvision_model_compare import MIRAGE_WS
    from src.masks.curriculum import CurriculumMaskGenerator

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_mirage(device)
    for p in model.parameters():
        p.requires_grad_(False)
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(L0=o.detach()))

    vols = sorted(p.stem for p in TEST.glob('data_*.npz'))[:n_slices]
    gen = CurriculumMaskGenerator(
        input_size=(JEPA_RES, JEPA_RES), patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=tuple(aspect), nenc=1, npred=4, min_keep=10,
        allow_overlap=False, curriculum_cfg=dict(CURRICULUM_CFG))
    gen.set_epoch(99, 100)
    gen.mirage_occupancy_threshold = threshold

    tgt, acc, feas, vis, fill, purity, hgt = [], [], [], [], [], [], []
    for vi, vol_id in enumerate(vols):
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        raw = np.asarray(vol[len(vol) // 2], dtype=np.float32)
        for s in range(n_seeds):
            seed = vi * 1000 + s
            mir_img, _, _ = aligned_crop(raw, seed)
            x = torch.from_numpy(mir_img)[None, None].to(device=device,
                                                         dtype=torch.float32)
            with torch.no_grad():
                model({'bscan': x})
            M = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)
            occ = F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0].cpu().numpy()
            placement = occ >= threshold
            random.seed(seed); np.random.seed(seed)
            tg = torch.Generator(); tg.manual_seed((seed * 2654435761 + 12345) % (2 ** 31))
            sizes = [gen._sample_block_size(gen.pred_mask_scale, tg)
                     for _ in range(gen.npred)]
            hgt.extend(h for h, _ in sizes)
            blocks, st = gen._sample_mirage_blocks(
                sizes, torch.from_numpy(occ), torch.from_numpy(placement),
                [True] * gen.npred, [None] * gen.npred)
            union = sorted(set().union(*[set(b) for b in blocks])) if blocks else []
            tgt.append(len(union))
            acc.append(bool(st.get('accepted', False)))
            feas.append(bool(st.get('feasible', False)))
            vis.append(float(st.get('retina_visible', np.nan)))
            fill.append(float(st.get('mean_block_fill', np.nan)))
            if union:
                purity.append(float(occ.reshape(-1)[np.array(union)].mean()))

    band = np.array([h for h in hgt], dtype=float)
    rep = {
        'n_draws': len(tgt), 'n_slices': len(vols), 'n_seeds': n_seeds,
        'alpha': alpha, 'threshold': threshold, 'aspect_ratio': list(aspect),
        'target_cells': {'mean': float(np.mean(tgt)), 'std': float(np.std(tgt)),
                         'p10': float(np.percentile(tgt, 10)),
                         'p50': float(np.percentile(tgt, 50)),
                         'p90': float(np.percentile(tgt, 90))},
        'accept_rate': float(np.mean(acc)), 'feasible_rate': float(np.mean(feas)),
        'retina_visible_mean': float(np.nanmean(vis)),
        'mean_block_fill': float(np.nanmean(fill)),
        'target_purity_mean': float(np.mean(purity)) if purity else float('nan'),
        'block_height_mean': float(band.mean()),
        'arms_reference_target_cells': '~102-112 (shipped MIRAGE / oracle / random arms)',
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2))
    t = rep['target_cells']
    print('draws %d   aspect %s   threshold %.2f' % (rep['n_draws'], aspect, threshold))
    print('  target cells   mean %.1f  std %.1f  p10 %.0f  p50 %.0f  p90 %.0f'
          % (t['mean'], t['std'], t['p10'], t['p50'], t['p90']))
    print('  accept %.4f   feasible %.4f   retina visible %.4f   block fill %.4f'
          % (rep['accept_rate'], rep['feasible_rate'],
             rep['retina_visible_mean'], rep['mean_block_fill']))
    print('  target purity  %.4f   mean block height %.2f cells'
          % (rep['target_purity_mean'], rep['block_height_mean']))
    print('  arms deliver   %s' % rep['arms_reference_target_cells'])
    print('wrote %s' % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--plot-from', type=pathlib.Path)
    ap.add_argument('--stats', type=pathlib.Path, help='area-matching statistics json')
    ap.add_argument('--n-slices', type=int, default=20)
    ap.add_argument('--n-seeds', type=int, default=10)
    ap.add_argument('--aspect', type=float, nargs=2, default=(0.75, 1.5))
    ap.add_argument('--out')
    ap.add_argument('--slice', default='data_07050:4')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--threshold', type=float, default=0.25)
    a = ap.parse_args()
    if a.stats:
        return stats_run(a.stats, a.n_slices, a.n_seeds, a.alpha, a.threshold, a.aspect)
    if a.dump:
        return dump(a.dump, a.slice, a.seed, a.alpha, a.threshold)
    if a.plot_from:
        return plot(a.plot_from, a.out)
    raise SystemExit('need --dump, --plot-from or --stats')


if __name__ == '__main__':
    main()
