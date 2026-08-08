"""Does a fixed mask budget K actually prevent the scorer from cheating?

The token-level proposal argues that letting MIRAGE choose WHICH K patches to
hide, while the system fixes HOW MANY, is "the single most important
anti-cheating constraint" (its section 8).  That claim is only half true and
this script measures the other half.

Fixed K removes two failure modes:
    hide nothing      (loss -> trivially low)
    hide everything   (loss -> manipulated)

It does NOT remove the third, which is the one that actually killed the earlier
designs:

    hide the K EASIEST patches.

Analytically, with hard budget K and soft relaxation h_i = sigmoid((q_i - tau)/T),
the JEPA loss over the hidden set is

    L = (1/K) * sum_i h_i * e_i

so

    dL/dq_i = (1/K) * e_i * h_i (1 - h_i) / T   >= 0        (smooth-L1 e_i >= 0)

Gradient descent therefore DECREASES q_i wherever e_i is large: high-error
patches are pushed out of the top-K and low-error patches are pulled in.  The
budget only guarantees that as one leaves, another enters.  The selection is
still degenerate.

This script does not argue that; it runs the actual proposed optimisation using
REAL measured per-patch JEPA errors and REAL MIRAGE anatomy scores, and reports
where the mask lands.

Inputs (both already on disk, no GPU, no model):
  results/masking/error_vs_anatomy/per_ckpt_errors.npz   real per-patch errors
  <L0.npz>                                               real MIRAGE logits
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GRID, POOL, ANATOMY = 16, 8, (1, 2)


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def anatomy_grid(L0):
    M = softmax_np(L0, axis=1)[:, ANATOMY].sum(axis=1)
    n, h, w = M.shape
    return M.reshape(n, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4))


def soft_topk(q, K, T, iters=60):
    """h = sigmoid((q - tau)/T) with tau bisected so that sum(h) == K."""
    lo = q.min(dim=-1, keepdim=True).values - 10.0
    hi = q.max(dim=-1, keepdim=True).values + 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = torch.sigmoid((q - mid) / T).sum(dim=-1, keepdim=True)
        too_many = (s > K)
        lo = torch.where(too_many, mid, lo)
        hi = torch.where(too_many, hi, mid)
    tau = 0.5 * (lo + hi)
    return torch.sigmoid((q - tau.detach()) / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--errors', type=pathlib.Path,
                    default=REPO / 'results/masking/error_vs_anatomy/per_ckpt_errors.npz')
    ap.add_argument('--l0', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\logit-scale\L0.npz'))
    ap.add_argument('--K', type=int, default=105, help='mask budget (arms deliver ~102-112)')
    ap.add_argument('--T', type=float, nargs='+', default=[0.1, 0.5, 2.0, 5.0],
                    help='sigmoid temperature sweep')
    ap.add_argument('--alpha', type=float, nargs='+', default=[2.0, 5.0, 10.0, 1e9],
                    help='authority sweep; 1e9 = effectively unbounded residual')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/topk_degeneracy')
    a = ap.parse_args()

    ez = np.load(a.errors, allow_pickle=True)
    err = ez['ep100'].reshape(len(ez['ep100']), -1)          # (N,256) real errors
    anat = ez['anat'].reshape(err.shape)                     # (N,256) real anatomy

    # distance-to-context proxy: measured to dominate error (r = +0.57).
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)

    e = torch.tensor(err, dtype=torch.float32)
    an = torch.tensor(anat, dtype=torch.float32)
    # base score q0 = anatomy logit, exactly as the proposal specifies
    q0 = torch.logit(an.clamp(1e-4, 1 - 1e-4))
    dq = torch.zeros_like(q0, requires_grad=True)            # the trainable residual
    opt = torch.optim.Adam([dq], lr=a.lr)

    rep = {'K': a.K, 'T_sweep': a.T, 'alpha_sweep': a.alpha, 'steps': a.steps,
           'n_slices': int(err.shape[0])}

    def snapshot(h):
        hard = torch.zeros_like(h)
        idx = h.topk(a.K, dim=-1).indices
        hard.scatter_(1, idx, 1.0)
        sel = hard.bool().numpy()
        return {
            'mean_err_of_hidden': float(err[sel].mean()),
            'mean_anat_of_hidden': float(anat[sel].mean()),
            'frac_hidden_on_anatomy': float((anat[sel] > 0.5).mean()),
            'budget_check': float(hard.sum(dim=-1).mean()),
        }

    with torch.no_grad():
        h0 = soft_topk(q0, a.K, a.T[0])
    rep['start'] = snapshot(h0)
    rep['baseline_mean_err_all_patches'] = float(err.mean())
    # the K globally easiest patches: the endpoint an unconstrained cheater reaches
    idx_easy = torch.tensor(err).argsort(dim=-1)[:, :a.K].numpy()
    sel_easy = np.zeros_like(err, dtype=bool)
    np.put_along_axis(sel_easy, idx_easy, True, axis=1)
    rep['oracle_cheat'] = {
        'mean_err_of_hidden': float(err[sel_easy].mean()),
        'mean_anat_of_hidden': float(anat[sel_easy].mean()),
        'frac_hidden_on_anatomy': float((anat[sel_easy] > 0.5).mean()),
    }

    print('=== Does a fixed budget K stop the scorer cheating? ===')
    print('  budget K = %d   temperatures %s   %d slices   %d steps'
          % (a.K, a.T, rep['n_slices'], a.steps))
    print('  mean error over ALL patches        : %.5f'
          % rep['baseline_mean_err_all_patches'])
    print('  MIRAGE-anatomy start               : err %.5f   anat %.4f'
          % (rep['start']['mean_err_of_hidden'], rep['start']['mean_anat_of_hidden']))
    print('  pure cheat (K globally easiest)    : err %.5f   anat %.4f'
          % (rep['oracle_cheat']['mean_err_of_hidden'],
             rep['oracle_cheat']['mean_anat_of_hidden']))
    print('\n  %8s %8s %10s %12s %13s %10s %9s' %
          ('alpha', 'T', 'loss', 'err(hidden)', 'anat(hidden)', 'live%', 'corr(g,e)'))

    rep['runs'] = {}
    for T in a.T:
        for alpha in a.alpha:
            dq = torch.zeros_like(q0, requires_grad=True)
            opt = torch.optim.Adam([dq], lr=a.lr)
            corr0, live0, ste0 = None, None, None
            for step in range(a.steps):
                resid = dq if alpha >= 1e8 else alpha * torch.tanh(dq)
                q = q0 + resid
                h = soft_topk(q, a.K, T)
                loss = (h * e).sum(dim=-1).mean() / a.K      # exactly the proposal's L
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if step == 0:                                # the sign check
                    g = dq.grad.detach().numpy().ravel()
                    m = np.abs(g) > 1e-12
                    corr0 = float(np.corrcoef(g[m], err.ravel()[m])[0, 1])
                    with torch.no_grad():
                        # fraction of patches whose soft gate can carry gradient
                        live0 = float((h * (1 - h) > 0.01).float().mean())
                        # STE fidelity: how far the soft surrogate is from the
                        # hard mask it stands in for during the forward pass
                        hard = torch.zeros_like(h)
                        hard.scatter_(1, h.topk(a.K, dim=-1).indices, 1.0)
                        ste0 = float((h - hard).abs().mean())
                opt.step()
            with torch.no_grad():
                resid = dq if alpha >= 1e8 else alpha * torch.tanh(dq)
                s = snapshot(soft_topk(q0 + resid, a.K, T))
            s.update(corr_grad_vs_error_at_init=corr0, final_loss=float(loss),
                     live_gate_frac=live0, ste_gap=ste0, T=T,
                     alpha=('unbounded' if alpha >= 1e8 else alpha))
            rep['runs']['T%.2f_a%s' % (T, 'unb' if alpha >= 1e8 else alpha)] = s
            print('  %8s %8.2f %10.5f %12.5f %13.4f %9.1f%% %+9.4f' %
                  ('unbnd' if alpha >= 1e8 else '%.1f' % alpha, T, s['final_loss'],
                   s['mean_err_of_hidden'], s['mean_anat_of_hidden'],
                   100 * live0, corr0))
        print('    T=%.2f  STE gap |h - hard| = %.4f' % (T, ste0))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'topk_degeneracy.json').write_text(json.dumps(rep, indent=2))
    print('\nwrote %s' % (a.out / 'topk_degeneracy.json'))


if __name__ == '__main__':
    main()
