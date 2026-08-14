"""Which patches actually drive the downstream glaucoma decision?

This is an EXACT attribution, not an ablation and not a retrained probe.

The frozen protocol is ``probe_type: mean_pool`` (zero parameters) followed by
``LinearHead`` = ``LayerNorm -> Linear``.  So for one volume::

    f      = mean over slices and patches of h_sj          (768,)
    logit  = w . LayerNorm(f) + b

LayerNorm is affine once its two scalars are fixed by f, so writing
``a_d = w_d * gamma_d / sigma(f)`` and ``A = sum_d a_d``::

    logit = mean_sj [ a . h_sj  -  A * mean_d(h_sj) ]  +  (w . beta + b)

Every patch therefore has a signed, exactly additive contribution::

    contrib(s,j) = ( a . h_sj - A * mean_d(h_sj) ) / (S * 256)

and those contributions sum to the logit (verified per volume, printed as a
residual).  Splitting them by the anatomy mask gives, with no retraining:

  * how much signed decision mass sits on anatomy vs background,
  * and -- the number that matters -- the AUC obtained from each region's
    contribution ALONE, which says how much of the actual discrimination each
    region carries.

Only the Test split is needed, and the classifier head is the one already
trained by the published run, so this costs one forward pass over 3,000 volumes
instead of the full 10,000-volume protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from torch.utils.data import DataLoader                            # noqa: E402
from src.datasets.oct_volumes import OCTVolumeDataset              # noqa: E402
from src.helper import init_patch_model                            # noqa: E402
from scripts.downstream_region_auc import (                        # noqa: E402
    imagenet_normalize, load_mask, CROP, PATCH, GRID, NPATCH)


def load_head(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    h = sd["head"]
    h = {k.replace("module.", ""): v for k, v in h.items()}
    return (h["norm.weight"].float(), h["norm.bias"].float(),
            h["linear.weight"].float().squeeze(0), h["linear.bias"].float().squeeze())


@torch.no_grad()
def attribute(ckpt, head_path, args, device):
    gamma, beta, w, b = [t.to(device) for t in load_head(head_path)]
    encoder, _ = init_patch_model(device, patch_size=PATCH, crop_size=CROP,
                                  model_name="vit_base")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(
        {k.replace("module.", ""): v for k, v in sd["target_encoder"].items()},
        strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ds = OCTVolumeDataset(os.path.join(args.data_dir, "Test"),
                          num_slices=args.num_slices, slice_size=CROP,
                          return_label=True)
    M = load_mask(args.mask_cache, "Test", args.num_slices, len(ds))
    if args.limit_test and args.limit_test < len(ds):
        keep = np.sort(np.random.default_rng(args.seed).permutation(len(ds))[:args.limit_test])
        ds = torch.utils.data.Subset(ds, keep.tolist())
        M = M[keep]
        print(f"  Test subsampled to {len(ds)} volumes", flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=args.amp)

    C_an, C_bg, LOG, LAB, RES = [], [], [], [], []
    # per-PATCH importance, separated from how many patches each region has
    an_abs, bg_abs, an_n, bg_n = 0.0, 0.0, 0, 0
    pmap_sum = np.zeros(NPATCH)
    pabs_sum = np.zeros(NPATCH)
    t0 = time.time()
    for i, (vol, lab) in enumerate(loader):
        flat = vol.squeeze(0).to(device)
        m_np = M[i]
        if args.slices_used and args.slices_used < flat.size(0):
            rg = np.random.default_rng(args.seed * 1000003 + i)
            edges = np.linspace(0, flat.size(0), args.slices_used + 1).astype(int)
            sel = np.unique(np.array([rg.integers(edges[k], max(edges[k + 1], edges[k] + 1))
                                      for k in range(args.slices_used)]))
            flat = flat[torch.from_numpy(sel).to(device)]
            m_np = m_np[sel]
        parts = []
        for j in range(0, flat.size(0), args.chunk):
            with amp:
                parts.append(encoder(imagenet_normalize(flat[j:j + args.chunk])).float())
        h = torch.cat(parts, 0)                       # (S,256,D)
        S = h.size(0)
        f = h.mean((0, 1))                            # (768,)
        mu, sigma = f.mean(), f.std(unbiased=False)
        a = w * gamma / (sigma + 1e-5)
        A = a.sum()
        logit = float((w * (gamma * (f - mu) / (sigma + 1e-5) + beta)).sum() + b)

        contrib = ((h @ a) - A * h.mean(-1)) / (S * NPATCH)   # (S,256)
        const = float((w * beta).sum() + b)
        RES.append(float(contrib.sum()) + const - logit)

        m = torch.from_numpy(m_np).to(device)         # (S,256) bool
        C_an.append(float(contrib[m].sum()))
        C_bg.append(float(contrib[~m].sum()))
        an_abs += float(contrib[m].abs().sum()); an_n += int(m.sum())
        bg_abs += float(contrib[~m].abs().sum()); bg_n += int((~m).sum())
        LOG.append(logit); LAB.append(int(lab.squeeze()))

        c_np = contrib.sum(0).cpu().numpy()
        pmap_sum += c_np
        pabs_sum += np.abs(c_np)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(ds)} ({time.time() - t0:.0f}s)", flush=True)

    from sklearn.metrics import roc_auc_score
    C_an = np.array(C_an); C_bg = np.array(C_bg)
    LOG = np.array(LOG); LAB = np.array(LAB)
    out = dict(
        checkpoint=str(ckpt), head=str(head_path), n_test=int(len(LAB)),
        max_abs_residual=float(np.abs(RES).max()),
        auc_full=float(roc_auc_score(LAB, LOG)),
        auc_anatomy_only=float(roc_auc_score(LAB, C_an)),
        auc_background_only=float(roc_auc_score(LAB, C_bg)),
        mean_abs_share_anatomy=float(np.abs(C_an).mean() /
                                     (np.abs(C_an).mean() + np.abs(C_bg).mean())),
        anatomy_cells_frac=float(M.mean()),
        corr_anat_bg=float(np.corrcoef(C_an, C_bg)[0, 1]),
        std_anat=float(C_an.std()), std_bg=float(C_bg.std()),
        # per-PATCH importance: mean |contribution| of ONE cell of each type.
        # This is the intensive quantity; multiplying by the cell count gives
        # the extensive one, and the two answer different questions.
        per_patch_anatomy=an_abs / max(an_n, 1),
        per_patch_background=bg_abs / max(bg_n, 1),
        ratio_per_patch=(an_abs / max(an_n, 1)) / max(bg_abs / max(bg_n, 1), 1e-12),
        n_cells_anatomy=int(an_n), n_cells_background=int(bg_n),
        total_abs_anatomy=an_abs, total_abs_background=bg_abs,
        ratio_total=an_abs / max(bg_abs, 1e-12),
    )
    np.savez_compressed(
        pathlib.Path(args.out) / f"{args.tag}_attrib.npz",
        C_an=C_an, C_bg=C_bg, logit=LOG, label=LAB,
        patch_mean=pmap_sum / len(LAB), patch_absmean=pabs_sum / len(LAB))
    del encoder
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--mask_cache", default=r"D:\jepa_phase0\reports\anatomy_mask_cache")
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--slices_used", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\patch_attribution")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = attribute(args.ckpt, args.head, args, device)
    r["tag"] = args.tag
    print(json.dumps(r, indent=2), flush=True)
    dst = out / f"{args.tag}_attrib.json"
    dst.write_text(json.dumps(r, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
