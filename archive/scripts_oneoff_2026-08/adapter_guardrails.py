#!/usr/bin/env python3
"""Three guardrail tests for the JEPA -> MIRAGE adapter, before any long run.

T1  HELD-OUT.  Train the adapter on a subset, measure L_rel reduction on
    FairVision images it never updated on.  This is the decisive test: an
    earlier 24-slice probe reported 76.8% reduction that turned out to be
    memorisation, and the honest single-pass number was 30.3%.

T2  BUDGET LOCK.  The adapter widens the guide (37.1 -> 43.8 cells, +18%).  If
    it may change both WHERE targets go and HOW MANY cells they cover, a
    downstream JEPA gain is confounded.  Arm B takes the cell budget from the
    FROZEN score and lets the adapted score set only ranking/geometry.

T3  UNCERTAINTY.  Where does the adapter act?  Compute the frozen confidence
    margin m = p_top1 - p_top2 and correlate it with |S_adapted - S_frozen|.
    We want changes concentrated where MIRAGE was UNSURE, i.e. corr < 0 against
    margin.  Changing confident interiors as much as boundaries is less
    convincing.

FairVision only.  No labels anywhere.
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

import anatomy_target_sampler_v2 as A                       # noqa: E402
from adapter_sweep import Adapter, gram, CACHE              # noqa: E402
from jepa_to_mirage_probe import build_mirage               # noqa: E402

OUT = REPO / 'results/masking/adapter_guardrails'
GRID, ANATOMY = 16, (1, 2)


def masks_and_grid(logits):
    P = logits.float().softmax(1)
    g = F.adaptive_avg_pool2d(P[:, ANATOMY], (GRID, GRID)).cpu().numpy()
    return g


def build_from_grid(g, budget=None):
    """4 targets from a (2,16,16) grid; `budget` locks the per-class cells."""
    if budget is None:
        parts, _ = A.build_targets([g[0], g[1]], 4)
    else:
        parts, _ = A.build_targets_fixed_cells([g[0], g[1]], budget, 4)
    return np.logical_or.reduce(parts), parts


def run(args):
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    RJ = np.load(CACHE / 'RJ.npy', mmap_mode='r')
    n_all = min(args.n_images, im512.shape[0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_all)
    n_ho = args.n_heldout
    ho_idx = np.sort(perm[:n_ho])
    tr_idx = np.sort(perm[n_ho:])
    print('train %d   held-out %d' % (len(tr_idx), len(ho_idx)), flush=True)

    hd = np.load(CACHE / 'head.npz')
    head = nn.Conv2d(384, 4, 1).to(dev)
    head.weight.data = torch.tensor(hd['w']).to(dev)
    head.bias.data = torch.tensor(hd['b']).to(dev)
    for p in head.parameters():
        p.requires_grad_(False)

    mir = build_mirage(dev)
    grab = {}
    mir.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda mo, i, o: grab.update(H=i[0].detach()))

    def h0_of(idx):
        x = torch.from_numpy(im512[idx].astype(np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            mir({'bscan': x})
        return grab['H'].float()

    # ---------------------------------------------------------------- train
    torch.manual_seed(0)
    mod = Adapter(depth=args.depth, width=args.width, alpha=args.alpha).to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=args.lr, weight_decay=1e-4)
    B = args.batch
    nsteps = (len(tr_idx) + B - 1) // B
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=nsteps * args.epochs,
                                                pct_start=0.1)
    curve = []
    for ep in range(args.epochs):
        order = tr_idx[np.random.default_rng(ep).permutation(len(tr_idx))]
        for s in range(0, len(order), B):
            idx = np.sort(order[s:s + B])
            H = mod(h0_of(idx))
            U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
            rj = torch.from_numpy(RJ[idx].astype(np.float32)).to(dev)
            loss = F.mse_loss(gram(U), rj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            curve.append(float(loss))
        print('  epoch %d/%d  L_rel %.5f' % (ep + 1, args.epochs,
                                             np.mean(curve[-nsteps:])), flush=True)
    mod.eval()

    # -------------------------------------------------------------- measure
    def measure(idx_set, tag):
        rel0, rel1 = [], []
        jac_free, jac_lock = [], []
        cells0, cells_free, cells_lock = [], [], []
        agree = []
        d_abs, margins = [], []
        drift = []
        for s in range(0, len(idx_set), B):
            idx = idx_set[s:s + B]
            H0 = h0_of(idx)
            with torch.no_grad():
                H = mod(H0)
                L0, Lf = head(H0), head(H)
                U0 = F.adaptive_avg_pool2d(H0, (GRID, GRID)).flatten(2).transpose(1, 2)
                U1 = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
                rj = torch.from_numpy(RJ[idx].astype(np.float32)).to(dev)
                rel0.append(float(F.mse_loss(gram(U0), rj)))
                rel1.append(float(F.mse_loss(gram(U1), rj)))
                agree.append(float((Lf.argmax(1) == L0.argmax(1)).float().mean()))
                drift.append(float((H - H0).norm() / H0.norm()))
                P0 = L0.float().softmax(1)
                top2 = P0.topk(2, dim=1).values
                marg = (top2[:, 0] - top2[:, 1])
                margins.append(F.adaptive_avg_pool2d(marg[:, None], (GRID, GRID))
                               [:, 0].cpu().numpy())
                g0 = masks_and_grid(L0)
                g1 = masks_and_grid(Lf)
                d_abs.append(np.abs(g1.sum(1) - g0.sum(1)))
            for j in range(g0.shape[0]):
                m0, p0 = build_from_grid(g0[j])
                # budget = cells the FROZEN build actually uses, per class.
                # Taking it from an uncapped grow_components over-counts,
                # because build_targets caps components at n.
                fr = A.build_targets([g0[j][0], g0[j][1]], 4)[1]
                half = len(A.grow_components(g0[j][0]))
                budget = [int(sum(int(U.sum()) for U in fr[:half])),
                          int(sum(int(U.sum()) for U in fr[half:]))]
                m1, _ = build_from_grid(g1[j])
                m2, _ = build_from_grid(g1[j], budget=budget)
                cells0.append(int(m0.sum()))
                cells_free.append(int(m1.sum()))
                cells_lock.append(int(m2.sum()))
                jac_free.append((m0 & m1).sum() / max((m0 | m1).sum(), 1))
                jac_lock.append((m0 & m2).sum() / max((m0 | m2).sum(), 1))
        d_abs = np.concatenate(d_abs).ravel()
        margins = np.concatenate(margins).ravel()
        # The margin distribution is saturated near 1.0, so a strict `>q75`
        # bucket can come out EMPTY and yield nan.  Bucket by fixed margin
        # thresholds instead, which is also easier to interpret.
        lo, hi = margins < 0.5, margins >= 0.9
        r0, r1 = float(np.mean(rel0)), float(np.mean(rel1))
        return {
            'tag': tag, 'n': int(len(idx_set)),
            'L_rel_before': r0, 'L_rel_after': r1,
            'L_rel_reduction_pct': 100 * (r0 - r1) / r0,
            'seg_agreement': float(np.mean(agree)),
            'feature_drift': float(np.mean(drift)),
            'cells_frozen': float(np.mean(cells0)),
            'cells_free': float(np.mean(cells_free)),
            'cells_locked': float(np.mean(cells_lock)),
            'jaccard_free': float(np.mean(jac_free)),
            'jaccard_locked': float(np.mean(jac_lock)),
            'corr_change_vs_margin': float(np.corrcoef(d_abs, margins)[0, 1]),
            'frac_margin_below_0.5': float(lo.mean()),
            'frac_margin_above_0.9': float(hi.mean()),
            'change_unsure_margin_lt0.5': float(d_abs[lo].mean()) if lo.any() else 0.0,
            'change_sure_margin_ge0.9': float(d_abs[hi].mean()) if hi.any() else 0.0,
        }

    res = {'config': vars(args),
           'train': measure(tr_idx[:args.n_eval], 'train'),
           'heldout': measure(ho_idx[:args.n_eval], 'heldout'),
           'loss_curve': curve}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'guardrails.json').write_text(json.dumps(res, indent=2))
    torch.save(mod.state_dict(), OUT / 'adapter_cfg7.pt')

    t, h = res['train'], res['heldout']
    print('\n=== T1  HELD-OUT GENERALISATION ===')
    print('%-30s%12s%12s' % ('', 'train', 'HELD-OUT'))
    for k, lab in (('L_rel_before', 'L_rel before'),
                   ('L_rel_after', 'L_rel after'),
                   ('L_rel_reduction_pct', 'reduction (%)'),
                   ('feature_drift', 'feature drift'),
                   ('seg_agreement', 'seg agreement')):
        print('%-30s%12.5f%12.5f' % (lab, t[k], h[k]))
    print('\n=== T2  BUDGET LOCK  (held-out) ===')
    print('%-30s%12s%12s' % ('', 'free', 'LOCKED'))
    print('%-30s%12.1f%12.1f' % ('mask cells (frozen %.1f)' % h['cells_frozen'],
                                 h['cells_free'], h['cells_locked']))
    print('%-30s%12.4f%12.4f' % ('mask Jaccard vs frozen',
                                 h['jaccard_free'], h['jaccard_locked']))
    print('\n=== T3  WHERE DOES THE ADAPTER ACT?  (held-out) ===')
    print('corr(|change|, frozen confidence margin) = %+.4f  (want < 0)'
          % h['corr_change_vs_margin'])
    print('mean |change| where MIRAGE UNSURE (margin < 0.5, %.1f%% of cells)  %.5f'
          % (100 * h['frac_margin_below_0.5'], h['change_unsure_margin_lt0.5']))
    print('mean |change| where MIRAGE SURE   (margin >= 0.9, %.1f%% of cells) %.5f'
          % (100 * h['frac_margin_above_0.9'], h['change_sure_margin_ge0.9']))
    if h['change_sure_margin_ge0.9'] > 0:
        print('ratio unsure/sure = %.2fx'
              % (h['change_unsure_margin_lt0.5'] / h['change_sure_margin_ge0.9']))
    print('\nwrote %s' % (OUT / 'guardrails.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=6000)
    ap.add_argument('--n-heldout', type=int, default=1200)
    ap.add_argument('--n-eval', type=int, default=192)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.5)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
