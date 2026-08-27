"""Does L_sem actually FIGHT L_seg, or is it orthogonal?

My earlier probe showed JEPA features are weaker linear predictors of anatomy than
MIRAGE's own (R2 0.78 vs 0.97) and concluded L_sem must be harmful.  That
conclusion does not follow: an auxiliary task need not CONTAIN the target
information to help -- it only needs to shape the representation usefully, and
the design already has L_seg to veto shapes that hurt segmentation.

The testable version of the disagreement is the standard multi-task question:

    cos( dL_seg/dtheta , dL_sem/dtheta )   on the SHARED adapter parameters

    cos <  0  -> the two objectives conflict; lambda is a direct trade dial and
                 "improve both" is false in the region where L_sem has any effect
    cos ~= 0  -> orthogonal; L_sem is close to free, L_seg's veto costs little
    cos >  0  -> they agree; L_sem genuinely assists segmentation

Then the equilibrium question: train L_anchor + lambda*L_sem and measure what
actually happens to segmentation fidelity as lambda grows.  That is the direct
test of "L_seg pushes back".

IMPORTANT PROXY: ground-truth segmentation labels are not wired up here, so
L_seg is proxied by L_anchor = agreement with the FROZEN MIRAGE logits.  That
captures L_seg's PROTECTIVE role (frozen MIRAGE is currently correct: GOALS Dice
0.9691 inner / 0.9420 choroid) but NOT its ability to improve beyond the frozen
model.  So this bounds the harm; it cannot measure the upside.

final_layer is Conv2d(384, 4, kernel=1) -- pointwise linear -- and average
pooling is linear, so final_layer(pool(H)) == pool(final_layer(H)) exactly.
The frozen segmentation path is therefore exact on the cached pooled features,
not an approximation.  The adapter is non-linear, so its own behaviour at 16x16
is an approximation of its behaviour at 128x128; noted as a limitation.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ANATOMY = (1, 2)


class Adapter(nn.Module):
    """Same shape as the V1 residual adapter, zero-initialised output."""

    def __init__(self, dim=384, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(h)


def flat_grad(params):
    return torch.cat([p.grad.reshape(-1) if p.grad is not None
                      else torch.zeros(p.numel()) for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\bidir\feats.npz'))
    ap.add_argument('--final-layer', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\bidir\final_layer.pt'))
    ap.add_argument('--lam', type=float, nargs='+', default=[0.0, 0.01, 0.1, 0.5, 1.0])
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/bidirectional'))
    a = ap.parse_args()

    torch.manual_seed(0)
    z = np.load(a.feats, allow_pickle=True)
    n, npatch = z['hm'].shape[0], z['hm'].shape[1]
    g = int(round(npatch ** 0.5))
    H = torch.tensor(z['hm'], dtype=torch.float32).reshape(n, g, g, -1).permute(0, 3, 1, 2)
    ZJ = torch.tensor(z['zj'], dtype=torch.float32)                       # (n,256,768)

    fl = torch.load(a.final_layer, map_location='cpu')
    W = fl['output_adapters.semseg.final_layer.weight']                   # (4,384,1,1)
    b = fl['output_adapters.semseg.final_layer.bias']
    seg_head = nn.Conv2d(384, 4, 1)
    seg_head.weight.data, seg_head.bias.data = W, b
    for p in seg_head.parameters():
        p.requires_grad_(False)                                           # frozen head

    with torch.no_grad():
        L0 = seg_head(H)                                                  # frozen logits
        anat0 = L0.softmax(1)[:, ANATOMY].sum(1)                          # (n,g,g)

    rep = {'n_slices': n, 'grid': g, 'steps': a.steps,
           'note': 'L_seg proxied by agreement with frozen MIRAGE logits'}

    # ---- 1. gradient conflict at initialisation -------------------------
    ad = Adapter()
    proj = nn.Linear(384, ZJ.shape[-1])
    params = list(ad.parameters())

    def losses(adapter, projector):
        H2 = adapter(H)
        seg = seg_head(H2)
        l_anchor = F.mse_loss(seg, L0)                                    # protective role
        feats = H2.permute(0, 2, 3, 1).reshape(n, npatch, -1)
        l_sem = (1 - F.cosine_similarity(projector(feats), ZJ, dim=-1)).mean()
        return l_anchor, l_sem

    # zero-init adapter gives an exactly-zero anchor gradient (seg == L0), so
    # perturb it first: the conflict question is about the training regime, not
    # the single degenerate point at initialisation.
    with torch.no_grad():
        for p in ad.net[-1].parameters():
            p.add_(0.01 * torch.randn_like(p))

    l_anchor, l_sem = losses(ad, proj)
    ad.zero_grad(); l_anchor.backward(retain_graph=True)
    g_anchor = flat_grad(params).clone()
    ad.zero_grad(); l_sem.backward()
    g_sem = flat_grad(params).clone()

    cos = float(F.cosine_similarity(g_anchor[None], g_sem[None]).item())
    # PCGrad view: how much of L_sem survives after removing the conflicting part
    proj_coef = float((g_sem @ g_anchor) / (g_anchor @ g_anchor + 1e-12))
    g_sem_free = g_sem - min(proj_coef, 0.0) * g_anchor
    rep['gradient_conflict'] = {
        'cos_g_anchor_g_sem': cos,
        'g_anchor_norm': float(g_anchor.norm()),
        'g_sem_norm': float(g_sem.norm()),
        'frac_of_Lsem_grad_conflicting': float(max(-cos, 0.0)),
        'frac_of_Lsem_grad_surviving_PCGrad': float(g_sem_free.norm() / (g_sem.norm() + 1e-12)),
    }

    # ---- 2. equilibrium: does L_seg actually hold the line? -------------
    rep['lambda_sweep'] = {}
    for lam in a.lam:
        torch.manual_seed(0)
        ad = Adapter()
        proj = nn.Linear(384, ZJ.shape[-1])
        opt = torch.optim.Adam(list(ad.parameters()) + list(proj.parameters()), lr=a.lr)
        for _ in range(a.steps):
            l_anchor, l_sem = losses(ad, proj)
            (l_anchor + lam * l_sem).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            H2 = ad(H)
            seg = seg_head(H2)
            anat = seg.softmax(1)[:, ANATOMY].sum(1)
            l_anchor, l_sem = losses(ad, proj)
            agree = float((seg.argmax(1) == L0.argmax(1)).float().mean())
            rep['lambda_sweep'][str(lam)] = {
                'L_anchor': float(l_anchor), 'L_sem': float(l_sem),
                'argmax_agreement_with_frozen': agree,
                'anatomy_mean_abs_shift': float((anat - anat0).abs().mean()),
                'anatomy_mean_signed_shift': float((anat - anat0).mean()),
            }

    # ---- 3. is the ADAPTER learning, or is the projector absorbing it all? ----
    # L_sem can be driven down by the free linear projector alone, without H
    # changing at all.  If so the loss is satisfied while MIRAGE learns nothing:
    # safe, but useless.  Ablation: train the projector with the adapter frozen.
    rep['projector_absorption'] = {}
    for mode in ('projector_only', 'adapter_and_projector'):
        torch.manual_seed(0)
        ad = Adapter()
        proj = nn.Linear(384, ZJ.shape[-1])
        train = list(proj.parameters())
        if mode == 'adapter_and_projector':
            train += list(ad.parameters())
        opt = torch.optim.Adam(train, lr=a.lr)
        for _ in range(a.steps):
            l_anchor, l_sem = losses(ad, proj)
            (l_anchor + 1.0 * l_sem).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            l_anchor, l_sem = losses(ad, proj)
            H2 = ad(H)
            drift = float((H2 - H).norm() / H.norm())
        rep['projector_absorption'][mode] = {
            'L_sem': float(l_sem), 'L_anchor': float(l_anchor),
            'relative_feature_drift': drift,
        }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'gradient_conflict.json').write_text(json.dumps(rep, indent=2))

    c = rep['gradient_conflict']
    print('=== Does L_sem fight L_seg? ===')
    print('  (L_seg proxied by agreement with FROZEN MIRAGE logits)')
    print('\n  cos(grad L_anchor, grad L_sem) on shared adapter : %+.4f' % c['cos_g_anchor_g_sem'])
    print('  |grad L_anchor| %.4e   |grad L_sem| %.4e'
          % (c['g_anchor_norm'], c['g_sem_norm']))
    print('  fraction of L_sem gradient that conflicts       : %.4f'
          % c['frac_of_Lsem_grad_conflicting'])
    print('  fraction surviving PCGrad projection            : %.4f'
          % c['frac_of_Lsem_grad_surviving_PCGrad'])
    print('\n  lambda sweep after %d steps of  L_anchor + lambda*L_sem :' % a.steps)
    print('  %8s %12s %10s %20s %16s' %
          ('lambda', 'L_anchor', 'L_sem', 'argmax agreement', '|anat shift|'))
    for lam in a.lam:
        r = rep['lambda_sweep'][str(lam)]
        print('  %8.2f %12.6f %10.5f %19.4f %16.5f' %
              (lam, r['L_anchor'], r['L_sem'],
               r['argmax_agreement_with_frozen'], r['anatomy_mean_abs_shift']))
    print('\n  is the ADAPTER learning, or does the projector absorb L_sem?')
    print('  %24s %10s %12s %22s' % ('mode', 'L_sem', 'L_anchor', 'feature drift |dH|/|H|'))
    for mode in ('projector_only', 'adapter_and_projector'):
        r = rep['projector_absorption'][mode]
        print('  %24s %10.5f %12.6f %22.6f'
              % (mode, r['L_sem'], r['L_anchor'], r['relative_feature_drift']))
    po = rep['projector_absorption']['projector_only']['L_sem']
    ap_ = rep['projector_absorption']['adapter_and_projector']['L_sem']
    print('  L_sem improvement attributable to the adapter: %.5f (%.1f%% of the gap from 1.0)'
          % (po - ap_, 100 * (po - ap_) / max(1.0 - ap_, 1e-9)))

    print('\nwrote %s' % (a.out / 'gradient_conflict.json'))


if __name__ == '__main__':
    main()
