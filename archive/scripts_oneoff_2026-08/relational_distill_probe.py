"""Relational distillation: can JEPA teach MIRAGE without a projector to absorb it?

Pointwise alignment  L_sem = 1 - cos(P_psi(h^M), sg(z^J))  was measured to be
satisfied almost entirely by the throwaway 384->768 projector: only 1.7-3.2% of
the improvement was attributable to MIRAGE's own features, and constraining the
projector did not help (a frozen random projector gave 233% feature churn and
0.0% gain).

Relational distillation removes the projector entirely.  Compare the patch-to-
patch SIMILARITY STRUCTURE of the two representations, which is dimension-free:

    C_M = normalize(H_M) normalize(H_M)^T          (256 x 256)
    C_J = normalize(Z_J) normalize(Z_J)^T          (256 x 256, stop-grad)
    L_rel = || C_M - sg(C_J) ||_F^2  /  N^2

There is no free linear map between 384-d and 768-d for the loss to hide in, so
any reduction must come from the MIRAGE-side representation.

Reported per run:
  * how much of L_rel the adapter actually achieves (there is nothing else to
    train, so this is the whole loss by construction -- the real question is
    whether it moves the representation in a direction the segmentation head
    tolerates)
  * agreement with frozen MIRAGE logits, i.e. does the anchor hold
  * gradient conflict cos(grad L_anchor, grad L_rel)
  * an L_rel-only control with NO anchor, to see where the representation goes
    when nothing pushes back

CAVEAT, same as before: L_seg is proxied by agreement with the FROZEN MIRAGE
logits.  That measures the protective role only.  Whether L_rel IMPROVES
segmentation can only be answered against real GOALS ground truth, which this
script does not do.
"""
from __future__ import annotations

import argparse
import json
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

from lsem_lseg_conflict import Adapter                        # noqa: E402

ANATOMY = (1, 2)


def gram(x):
    """Row-normalised similarity matrix, (B, N, N)."""
    return F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).transpose(1, 2)


def flat_grad(params):
    return torch.cat([p.grad.reshape(-1) if p.grad is not None
                      else torch.zeros(p.numel()) for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\bidir\feats.npz'))
    ap.add_argument('--final-layer', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\bidir\final_layer.pt'))
    ap.add_argument('--lam', type=float, nargs='+', default=[0.0, 0.1, 1.0, 10.0])
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/bidirectional')
    a = ap.parse_args()

    torch.manual_seed(0)
    z = np.load(a.feats, allow_pickle=True)
    n, npatch = z['hm'].shape[0], z['hm'].shape[1]
    g = int(round(npatch ** 0.5))
    H = torch.tensor(z['hm'], dtype=torch.float32).reshape(n, g, g, -1).permute(0, 3, 1, 2)
    ZJ = torch.tensor(z['zj'], dtype=torch.float32)

    fl = torch.load(a.final_layer, map_location='cpu')
    head = nn.Conv2d(384, 4, 1)
    head.weight.data = fl['output_adapters.semseg.final_layer.weight']
    head.bias.data = fl['output_adapters.semseg.final_layer.bias']
    for p in head.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        L0 = head(H)
        CJ = gram(ZJ)                                    # stop-grad target
        CM0 = gram(H.permute(0, 2, 3, 1).reshape(n, npatch, -1))
        base_rel = float(((CM0 - CJ) ** 2).mean())
        base_agree_struct = float(F.cosine_similarity(
            CM0.reshape(n, -1), CJ.reshape(n, -1), dim=-1).mean())

    rep = {'n_slices': n, 'steps': a.steps,
           'L_rel_at_init': base_rel,
           'gram_cosine_MIRAGE_vs_JEPA_at_init': base_agree_struct,
           'note': 'L_seg proxied by agreement with frozen MIRAGE logits'}

    def losses(ad):
        H2 = ad(H)
        l_anchor = F.mse_loss(head(H2), L0)
        CM = gram(H2.permute(0, 2, 3, 1).reshape(n, npatch, -1))
        l_rel = ((CM - CJ) ** 2).mean()
        return l_anchor, l_rel, H2

    # ---- gradient conflict, away from the degenerate zero-init point -----
    ad = Adapter()
    with torch.no_grad():
        for p in ad.net[-1].parameters():
            p.add_(0.01 * torch.randn_like(p))
    la, lr, _ = losses(ad)
    params = list(ad.parameters())
    ad.zero_grad(); la.backward(retain_graph=True); ga = flat_grad(params).clone()
    ad.zero_grad(); lr.backward(); gr = flat_grad(params).clone()
    rep['cos_grad_anchor_vs_rel'] = float(F.cosine_similarity(ga[None], gr[None]).item())
    rep['grad_norm_anchor'] = float(ga.norm())
    rep['grad_norm_rel'] = float(gr.norm())

    # ---- lambda sweep ----------------------------------------------------
    rep['lambda_sweep'] = {}
    for lam in a.lam:
        torch.manual_seed(0)
        ad = Adapter()
        opt = torch.optim.Adam(ad.parameters(), lr=a.lr)
        for _ in range(a.steps):
            la, lr, _ = losses(ad)
            (la + lam * lr).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            la, lr, H2 = losses(ad)
            rep['lambda_sweep'][str(lam)] = {
                'L_anchor': float(la), 'L_rel': float(lr),
                'L_rel_reduction_pct': 100 * (base_rel - float(lr)) / base_rel,
                'argmax_agreement_with_frozen': float(
                    (head(H2).argmax(1) == L0.argmax(1)).float().mean()),
                'relative_feature_drift': float((H2 - H).norm() / H.norm()),
                'anatomy_mean_abs_shift': float(
                    (head(H2).softmax(1)[:, ANATOMY].sum(1)
                     - L0.softmax(1)[:, ANATOMY].sum(1)).abs().mean()),
            }

    # ---- control: no anchor at all --------------------------------------
    torch.manual_seed(0)
    ad = Adapter()
    opt = torch.optim.Adam(ad.parameters(), lr=a.lr)
    for _ in range(a.steps):
        _, lr, _ = losses(ad)
        lr.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        la, lr, H2 = losses(ad)
        rep['no_anchor_control'] = {
            'L_anchor': float(la), 'L_rel': float(lr),
            'L_rel_reduction_pct': 100 * (base_rel - float(lr)) / base_rel,
            'argmax_agreement_with_frozen': float(
                (head(H2).argmax(1) == L0.argmax(1)).float().mean()),
            'relative_feature_drift': float((H2 - H).norm() / H.norm()),
        }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'relational_distill.json').write_text(json.dumps(rep, indent=2))

    print('=== Relational distillation (no projector to absorb the loss) ===')
    print('  L_rel at init %.6f   gram cosine(MIRAGE, JEPA) %.4f'
          % (base_rel, base_agree_struct))
    print('  cos(grad L_anchor, grad L_rel) = %+.4f   |ga| %.3e  |gr| %.3e'
          % (rep['cos_grad_anchor_vs_rel'], rep['grad_norm_anchor'], rep['grad_norm_rel']))
    print('\n  %8s %11s %10s %9s %19s %12s'
          % ('lambda', 'L_anchor', 'L_rel', 'L_rel dn%', 'argmax agreement', 'feat drift'))
    for lam in a.lam:
        r = rep['lambda_sweep'][str(lam)]
        print('  %8.2f %11.6f %10.6f %8.1f%% %19.4f %12.4f'
              % (lam, r['L_anchor'], r['L_rel'], r['L_rel_reduction_pct'],
                 r['argmax_agreement_with_frozen'], r['relative_feature_drift']))
    c = rep['no_anchor_control']
    print('  %8s %11.6f %10.6f %8.1f%% %19.4f %12.4f'
          % ('no anch', c['L_anchor'], c['L_rel'], c['L_rel_reduction_pct'],
             c['argmax_agreement_with_frozen'], c['relative_feature_drift']))
    print('\nwrote %s' % (a.out / 'relational_distill.json'))


if __name__ == '__main__':
    main()
