"""B2 gate: does the real I-JEPA predictor accept anatomy-budgeted irregular targets?

The anatomy-relative budget makes the number of target cells VARY per image
(a slice with more retina gets a larger target).  I-JEPA's collation stacks the
per-image index tensors, which requires them to be equal length, and the
curriculum collator resolves that by truncating every target to the batch-wide
minimum (src/masks/curriculum.py:1210):

    global_min_pred = max(1, min(t.numel() for group in masks_pred for t in group))
    collated_masks_pred.append(torch.stack([t[:global_min_pred] for t in group], dim=0))

Rectangular masking survives this because every block has the same size by
construction.  A variable anatomy budget does not, so this probe measures the
loss of target area BEFORE any training is launched, and compares two fixes.

Also verified, on the real modules, with real MIRAGE grids:
  1. masks_pred still contains FOUR tensors (I-JEPA's target-count ablation
     reports 1/2/3/4 targets -> 9.0/22.0/48.5/54.2 on 1% ImageNet, so this is
     not a detail that can be quietly changed)
  2. tokens marked hidden are genuinely ABSENT from the context encoder
  3. the target encoder still receives all 256 tokens
  4. predictor output and loss shapes are correct
  5. gradients reach the JEPA encoder and predictor
  6. several hundred optimisation steps give a finite, decreasing loss
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.helper import init_patch_model                       # noqa: E402
from src.masks.utils import apply_masks                       # noqa: E402
from src.utils.tensors import repeat_interleave_batch         # noqa: E402
from anatomy_target_sampler import build_targets              # noqa: E402

CROP, PATCH, GRID = 256, 16, 16


def build_target_lists(grids, rho):
    """Per-image list of 4 connected irregular target index lists."""
    out = []
    for a in grids:
        parts, _ = build_targets(a, rho=rho, overlap=0.24)
        out.append([np.flatnonzero(p.ravel()).tolist() for p in parts])
    return out


def collate(per_image, mode):
    """Stack per-image targets into 4 tensors of shape (B, n).

    truncate : I-JEPA's existing behaviour -- cut everything to the batch minimum
    pad_pool : keep every index by sampling WITH replacement up to the batch max,
               so no anatomy is discarded (duplicates re-weight, they do not drop)
    """
    B = len(per_image)
    npred = len(per_image[0])
    groups = [[per_image[b][p] for b in range(B)] for p in range(npred)]
    sizes = [len(t) for g in groups for t in g]
    out, stats = [], {'min': int(min(sizes)), 'max': int(max(sizes)),
                      'mean': float(np.mean(sizes))}
    if mode == 'truncate':
        n = max(1, min(sizes))
        for g in groups:
            out.append(torch.tensor([t[:n] for t in g], dtype=torch.long))
        stats['kept'] = n * npred * B
    else:
        n = max(sizes)
        rng = np.random.default_rng(0)
        for g in groups:
            rows = []
            for t in g:
                t = t if t else [0]
                extra = rng.choice(t, size=n - len(t), replace=True).tolist() \
                    if n > len(t) else []
                rows.append(list(t) + extra)
            out.append(torch.tensor(rows, dtype=torch.long))
        stats['kept'] = sum(sizes)
    stats['delivered_unique'] = float(np.mean(
        [len(set(t[:len(t)].tolist())) for g in out for t in g]))
    stats['n_tensors'] = len(out)
    stats['tensor_shape'] = list(out[0].shape)
    return out, stats


def context_from(per_image, B):
    """Context = every cell not in the union of that image's 4 targets."""
    rows = []
    for b in range(B):
        un = set()
        for p in per_image[b]:
            un |= set(p)
        rows.append(sorted(set(range(GRID * GRID)) - un))
    n = min(len(r) for r in rows)
    return [torch.tensor([r[:n] for r in rows], dtype=torch.long)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_base.npz'))
    ap.add_argument('--rho', type=float, default=0.70)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/b2_probe')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    G = np.load(a.grids, allow_pickle=True)['grids'][:a.batch]
    per_image = build_target_lists(G, a.rho)
    rep = {'rho': a.rho, 'batch': a.batch, 'device': str(device)}

    # ---- collation: how much anatomy survives each policy? --------------
    for mode in ('truncate', 'pad_pool'):
        _, st = collate(per_image, mode)
        rep['collate_%s' % mode] = st
    raw = [len(p) for im in per_image for p in im]
    rep['per_target_cells'] = {'mean': float(np.mean(raw)), 'min': int(min(raw)),
                               'max': int(max(raw))}
    rep['truncation_loss_frac'] = 1 - rep['collate_truncate']['kept'] / sum(raw)

    # ---- real modules ----------------------------------------------------
    encoder, predictor = init_patch_model(device, patch_size=PATCH, crop_size=CROP,
                                          model_name='vit_base')
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    x = torch.randn(a.batch, 3, CROP, CROP, device=device)

    results = {}
    for mode in ('truncate', 'pad_pool'):
        torch.manual_seed(0)
        enc = copy.deepcopy(encoder).to(device)
        prd = copy.deepcopy(predictor).to(device)
        tgt = copy.deepcopy(target_encoder).to(device)
        mp, st = collate(per_image, mode)
        mp = [m.to(device) for m in mp]
        me = [m.to(device) for m in context_from(per_image, a.batch)]

        # --- structural checks (once) ---
        with torch.no_grad():
            z_ctx = enc(x, me)
            h_full = tgt(x)
        chk = {
            'n_masks_pred': len(mp),
            'masks_pred_shape': list(mp[0].shape),
            'context_shape': list(z_ctx.shape),
            'target_encoder_sees_all_256': list(h_full.shape) == [a.batch, 256, 768],
            'context_excludes_all_targets': all(
                not (set(me[0][b].tolist()) & set(t[b].tolist()))
                for t in mp for b in range(a.batch)),
        }
        z = prd(enc(x, me), me, mp)
        h = F.layer_norm(h_full, (h_full.size(-1),))
        h_rep = repeat_interleave_batch(apply_masks(h, mp), a.batch, repeat=len(me))
        chk['pred_shape'] = list(z.shape)
        chk['target_shape'] = list(h_rep.shape)
        chk['shapes_match'] = list(z.shape) == list(h_rep.shape)

        # --- optimisation ---
        opt = torch.optim.AdamW(list(enc.parameters()) + list(prd.parameters()), lr=a.lr)
        losses = []
        for step in range(a.steps):
            z = prd(enc(x, me), me, mp)
            with torch.no_grad():
                h = F.layer_norm(tgt(x), (768,))
                h_rep = repeat_interleave_batch(apply_masks(h, mp), a.batch, repeat=len(me))
            loss = F.smooth_l1_loss(z, h_rep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                chk['grad_encoder'] = float(sum(
                    p.grad.abs().sum() for p in enc.parameters() if p.grad is not None))
                chk['grad_predictor'] = float(sum(
                    p.grad.abs().sum() for p in prd.parameters() if p.grad is not None))
                chk['grad_target_encoder'] = float(sum(
                    p.grad.abs().sum() for p in tgt.parameters() if p.grad is not None))
            opt.step()
            losses.append(float(loss))
        chk['loss_first'] = losses[0]
        chk['loss_last'] = losses[-1]
        chk['loss_finite'] = bool(np.all(np.isfinite(losses)))
        chk['loss_decreased'] = losses[-1] < losses[0]
        chk['collate'] = st
        results[mode] = chk
        del enc, prd, tgt
        torch.cuda.empty_cache()

    rep['runs'] = results
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'b2_predictor_probe.json').write_text(json.dumps(rep, indent=2))

    print('=== B2: real predictor with anatomy-budgeted irregular targets ===')
    print('  rho %.2f   batch %d   device %s' % (a.rho, a.batch, device))
    c = rep['per_target_cells']
    print('  per-target cells: mean %.1f  min %d  max %d   <-- VARIABLE, unlike rectangles'
          % (c['mean'], c['min'], c['max']))
    print('  truncate-to-batch-min would discard %.1f%% of target cells'
          % (100 * rep['truncation_loss_frac']))
    for mode, r in results.items():
        print('\n  --- %s ---' % mode)
        print('    masks_pred tensors      : %d   shape %s'
              % (r['n_masks_pred'], r['masks_pred_shape']))
        print('    context shape           : %s' % r['context_shape'])
        print('    target encoder all 256  : %s' % r['target_encoder_sees_all_256'])
        print('    context excludes targets: %s' % r['context_excludes_all_targets'])
        print('    pred %s  vs  target %s  match %s'
              % (r['pred_shape'], r['target_shape'], r['shapes_match']))
        print('    grad encoder %.3e  predictor %.3e  target-encoder %.1f'
              % (r['grad_encoder'], r['grad_predictor'], r['grad_target_encoder']))
        print('    loss %.5f -> %.5f   finite %s   decreased %s'
              % (r['loss_first'], r['loss_last'], r['loss_finite'], r['loss_decreased']))
    print('\nwrote %s' % (a.out / 'b2_predictor_probe.json'))


if __name__ == '__main__':
    main()
