#!/usr/bin/env python
"""Class-conditioned structural loss: transfer strengths, preserve strengths.

Measurement (`docs/experiments/masking/class_relations.md`) showed that full
Gram-MSE asks I-JEPA to teach MIRAGE the one relationship I-JEPA is worst at:

    inner-vs-choroid discrimination AUC
        MIRAGE encoder  0.9773   (Cohen's d 2.82)
        I-JEPA ep100    0.6945   (Cohen's d 0.65)
        untrained JEPA  0.8288   <- even random weights beat trained I-JEPA

and that driving L_rel down moves MIRAGE's inner-choroid similarity from 0.304
toward I-JEPA's 0.719, collapsing the separation the segmentation head needs.

This replaces the objective with two terms.

1. STRUCTURAL TRANSFER on safe blocks only.  Each 16x16 cell gets a coarse
   pseudo-class from FROZEN MIRAGE itself (no external labels): I inner, C
   choroid, B background.  Pairs are binned, and I-JEPA teaches only

       S = { I-I, C-C, I-B, C-B }

   The I-C block -- the one I-JEPA gets wrong -- is excluded, as is B-B, which
   is background-to-background and carries nothing worth learning.

   Absolute cosine scales differ between independently trained representations
   (MIRAGE within 0.667 / between 0.304 vs I-JEPA 0.800 / 0.718), so matching
   raw values imports I-JEPA's scale rather than its structure.  Both sides are
   therefore z-scored over the safe set per image before comparison.

2. SEPARATION BARRIER.  A one-sided hinge protects what MIRAGE already knows:

       delta = (mu_II + mu_CC)/2 - mu_IC
       L_sep = relu(delta_frozen - delta_adapted)^2

   No penalty while the adapted model separates the tissues at least as well as
   frozen MIRAGE; it only pushes back when adaptation starts erasing that.

    L = L_struct + lambda_sep * L_sep

Pseudo-labels are taken from the FROZEN forward, so the class assignment cannot
drift as the adapter trains.
"""
from __future__ import annotations

import argparse
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

from adapter_placement_ablation import Adapter, gram, to_tokens      # noqa: E402
from goals_eval import load_pairs, dice_iou, VOID, RES               # noqa: E402
from jepa_to_mirage_probe import (build_mirage, build_jepa,          # noqa: E402
                                  IMNET_MEAN, IMNET_STD)

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
OUT = REPO / 'results/masking/structural_loss'
GRID = 16
B_, I_, C_ = 0, 1, 2


def pseudo_labels(logits):
    """Coarse per-cell class from frozen MIRAGE. (B, 256) in {0,1,2}."""
    lg = logits.float().clone()
    lg[:, VOID] = float('-inf')
    hard = lg.argmax(1)                                   # (B, H, W)
    onehot = torch.stack([(hard == c).float() for c in (B_, I_, C_)], 1)
    pooled = F.adaptive_avg_pool2d(onehot, (GRID, GRID))  # (B, 3, 16, 16)
    return pooled.flatten(2).argmax(1)                    # (B, 256)


def block_masks(lab):
    """Boolean pair masks for one image's label vector (N,)."""
    a, b = lab[:, None], lab[None, :]
    off = ~torch.eye(len(lab), dtype=torch.bool, device=lab.device)
    m = {
        'II': (a == I_) & (b == I_) & off,
        'CC': (a == C_) & (b == C_) & off,
        'IC': ((a == I_) & (b == C_)) | ((a == C_) & (b == I_)),
        'IB': ((a == I_) & (b == B_)) | ((a == B_) & (b == I_)),
        'CB': ((a == C_) & (b == B_)) | ((a == B_) & (b == C_)),
    }
    m['safe'] = m['II'] | m['CC'] | m['IB'] | m['CB']
    return m


def zscore(v, eps=1e-6):
    return (v - v.mean()) / (v.std() + eps)


def separation(R, m):
    """delta = mean(within-tissue) - mean(inner-choroid); None if undefined."""
    parts = []
    for k in ('II', 'CC'):
        if m[k].any():
            parts.append(R[m[k]].mean())
    if not parts or not m['IC'].any():
        return None
    return torch.stack(parts).mean() - R[m['IC']].mean()


def structural_loss(RM, RM0, RJ, labs, lam):
    """Returns (total, L_struct, L_sep, n_used)."""
    ls, lp, used = [], [], 0
    for k in range(len(labs)):
        m = block_masks(labs[k])
        if m['safe'].sum() < 8:
            continue
        used += 1
        ls.append(F.mse_loss(zscore(RM[k][m['safe']]),
                             zscore(RJ[k][m['safe']]).detach()))
        d_now = separation(RM[k], m)
        d_ref = separation(RM0[k], m)
        if d_now is not None and d_ref is not None:
            lp.append(F.relu(d_ref.detach() - d_now) ** 2)
    if not used:
        z = torch.zeros((), device=RM.device, requires_grad=True)
        return z, z, z, 0
    L_struct = torch.stack(ls).mean()
    L_sep = torch.stack(lp).mean() if lp else torch.zeros((), device=RM.device)
    return L_struct + lam * L_sep, L_struct, L_sep, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=4800)
    ap.add_argument('--n-eval', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--lam-sep', type=float, default=10.0)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--teachers', default=(
        r'early_ep27=D:\jepa_phase0\runs\patch_mirage_envelope\resume-ep27.pth.tar;'
        r'late_ep100=D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar'))
    ap.add_argument('--tag', default='structural')
    ap.add_argument('--losses', default='gram_mse,structural:10',
                    help='comma list; "structural:<lam_sep>" may repeat, e.g. '
                         'gram_mse,structural:0,structural:10,structural:100. '
                         'lam_sep=0 isolates safe-block+z-score from the barrier.')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'

    mir = build_mirage(dev)
    sem = mir.output_adapters['semseg']
    tap_mod, tap_ch = sem.proj_dec, 768
    grab = {}
    sem.proj_dec.register_forward_hook(
        lambda m, i, o: grab.update(enc=i[0].detach()))

    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    im256 = np.load(CACHE / 'im256.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    perm = rng.permutation(min(len(im512), a.n_train + a.n_eval))
    idx_ev, idx_tr = np.sort(perm[:a.n_eval]), np.sort(perm[a.n_eval:])

    def mirage_pass(idx):
        """Frozen encoder tokens, seg logits and pseudo-labels."""
        x = torch.from_numpy(
            np.asarray(im512[idx], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            o = mir({'bscan': x})
        lg = (o['semseg'] if isinstance(o, dict) else o)
        return grab['enc'].float(), pseudo_labels(lg)

    g_imgs, g_gts, g_names = load_pairs()
    gt_labs = []
    for g in g_gts:
        h = g.shape[0] // GRID
        cells = g.reshape(GRID, h, GRID, h).transpose(0, 2, 1, 3).reshape(GRID * GRID, -1)
        lab = np.array([np.bincount(c, minlength=4).argmax() for c in cells])
        pur = np.array([np.bincount(c, minlength=4).max() / c.size for c in cells])
        gt_labs.append((lab, pur))

    def goals_eval(adapter=None):
        """GOALS Dice plus GT-labelled inner/choroid separation of the tap."""
        h = None
        if adapter is not None:
            def pre(m, args):
                x = args[0]
                return (adapter(x.float()).to(x.dtype),) + args[1:]
            h = tap_mod.register_forward_pre_hook(pre)
        preds, feats = [], []
        try:
            for s in range(0, len(g_imgs), 8):
                x = torch.from_numpy(g_imgs[s:s + 8])[:, None].to(dev)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    o = mir({'bscan': x})
                lg = (o['semseg'] if isinstance(o, dict) else o).float()
                if lg.shape[-1] != RES:
                    lg = F.interpolate(lg, size=(RES, RES), mode='bilinear',
                                       align_corners=False)
                lg[:, VOID] = float('-inf')
                preds.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
                feats.append(grab['enc'].float().cpu().numpy())
        finally:
            if h is not None:
                h.remove()
        pred = np.concatenate(preds)
        feat = np.concatenate(feats)
        per = []
        for i in range(len(g_gts)):
            ds = [dice_iou(pred[i], g_gts[i], c)[0] for c in (I_, C_)]
            per.append(np.mean([d for d in ds if d is not None]))
        win, btw = [], []
        for i, (lab, pur) in enumerate(gt_labs):
            keep = (pur >= 0.7) & np.isin(lab, (I_, C_))
            if keep.sum() < 4:
                continue
            v = F.normalize(torch.from_numpy(feat[i][keep]).float(), dim=-1)
            s = (v @ v.T).numpy()
            li = lab[keep]
            iu = np.triu_indices(len(li), k=1)
            same = (li[:, None] == li[None, :])[iu]
            win.extend(s[iu][same]); btw.extend(s[iu][~same])
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones(len(win)), np.zeros(len(btw))]
        auc = float(roc_auc_score(y, np.r_[win, btw]))
        return (float(np.mean(per)), np.array(per), pred,
                {'within': float(np.mean(win)), 'between': float(np.mean(btw)),
                 'delta': float(np.mean(win) - np.mean(btw)), 'discrim_auc': auc})

    base_d, base_per, base_pred, base_sep = goals_eval()
    print('frozen MIRAGE   GOALS Dice %.4f   inner/chor delta %+.4f  AUC %.4f'
          % (base_d, base_sep['delta'], base_sep['discrim_auc']))
    print('train %d / eval %d stratified slices, %d epoch(s), alpha %.2f, lam_sep %.1f'
          % (len(idx_tr), len(idx_ev), a.epochs, a.alpha, a.lam_sep))
    print()

    teachers = dict(t.split('=', 1) for t in a.teachers.split(';'))
    res = {'frozen': {'goals_dice': base_d, **base_sep},
           'cfg': {'alpha': a.alpha, 'lam_sep': a.lam_sep, 'epochs': a.epochs,
                   'n_train': int(len(idx_tr)), 'lr': a.lr, 'tap': 'enc',
                   'cache': str(CACHE)},
           'runs': {}}

    hdr = ('%-13s %-14s %10s %9s %9s %9s %9s' %
           ('teacher', 'loss', 'GOALS Dice', 'vs frozen', 'sep delta',
            'sep AUC', 'mask J'))
    print(hdr); print('-' * len(hdr))

    for tname, tpath in teachers.items():
        enc = build_jepa(pathlib.Path(tpath), dev)
        for p in enc.parameters():
            p.requires_grad_(False)

        def jg(idx):
            b = np.asarray(im256[idx], np.float32) / 255.
            rgb = (np.repeat(b[..., None], 3, -1) - IMNET_MEAN) / IMNET_STD
            x = torch.from_numpy(rgb.transpose(0, 3, 1, 2).astype(np.float32)).to(dev)
            with torch.no_grad():
                return gram(F.layer_norm(enc(x), (768,)))

        for loss_spec in a.losses.split(','):
            loss_spec = loss_spec.strip()
            if ':' in loss_spec:
                loss_name, lam = loss_spec.split(':')
                lam = float(lam)
            else:
                loss_name, lam = loss_spec, a.lam_sep
            label = (loss_name if loss_name == 'gram_mse'
                     else 'struct(l=%g)' % lam)
            torch.manual_seed(0)
            ad = Adapter(tap_ch, 2, 128, a.alpha).to(dev)
            opt = torch.optim.AdamW(ad.parameters(), lr=a.lr, weight_decay=1e-4)
            nst = a.epochs * ((len(idx_tr) + a.batch - 1) // a.batch)
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=a.lr, total_steps=nst, pct_start=0.1)
            hist = []
            for ep in range(a.epochs):
                for s in range(0, len(idx_tr), a.batch):
                    i = np.sort(idx_tr[s:s + a.batch])
                    Z, labs = mirage_pass(i)
                    RJ = jg(i)
                    Zp = ad(Z)
                    RM = gram(to_tokens(Zp))
                    if loss_name == 'gram_mse':
                        loss = F.mse_loss(RM, RJ)
                        ls = lp = torch.zeros(())
                    else:
                        with torch.no_grad():
                            RM0 = gram(to_tokens(Z))
                        loss, ls, lp, _ = structural_loss(
                            RM, RM0, RJ, labs, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                    hist.append({'loss': float(loss), 'struct': float(ls),
                                 'sep': float(lp)})
            ad.eval()

            d, per, pred, sep = goals_eval(ad)
            jac = float(np.mean([
                (np.isin(pred[i], (I_, C_)) & np.isin(base_pred[i], (I_, C_))).sum()
                / max((np.isin(pred[i], (I_, C_))
                       | np.isin(base_pred[i], (I_, C_))).sum(), 1)
                for i in range(len(pred))]))
            from scipy import stats
            p = float(stats.ttest_rel(per, base_per).pvalue)
            print('%-13s %-14s %10.4f %+9.5f %+9.4f %9.4f %9.4f'
                  % (tname, label, d, d - base_d, sep['delta'],
                     sep['discrim_auc'], jac))
            res['runs']['%s__%s' % (tname, label)] = {
                'teacher': tname, 'teacher_path': tpath, 'loss': label,
                'lam_sep': lam,
                'goals_dice': d, 'dice_delta': d - base_d, 'dice_p': p,
                'better_on': int((per > base_per).sum()),
                'sep_delta': sep['delta'], 'sep_auc': sep['discrim_auc'],
                'sep_within': sep['within'], 'sep_between': sep['between'],
                'mask_jaccard': jac,
                'final_loss': float(np.mean([h['loss'] for h in hist[-20:]])),
            }
            torch.save({'state_dict': ad.state_dict(),
                        'cfg': {'ch': tap_ch, 'depth': 2, 'width': 128,
                                'alpha': a.alpha},
                        'teacher': tpath, 'loss': loss_name},
                       OUT / ('adapter_%s_%s_a%03d.pt' % (tname, label.replace('(','').replace(')','').replace('=','').replace('.',''), round(a.alpha*100))))
        del enc
        torch.cuda.empty_cache()

    (OUT / ('%s.json' % a.tag)).write_text(json.dumps(res, indent=2))
    print('\nwrote', OUT / ('%s.json' % a.tag))


if __name__ == '__main__':
    main()


