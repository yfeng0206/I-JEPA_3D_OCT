"""Fit the published-format classifier head from cached region features.

``patch_attribution.py`` needs a head in the exact ``LinearHead`` form
(``LayerNorm -> Linear``), because the exact per-patch decomposition is derived
from those four tensors.  Published runs saved such a head, but the random and
oracle arms were trained elsewhere and only their AUCs were published.

The region-feature cache written by ``downstream_region_auc.py`` already holds
the pooled 768-vectors for those arms, so the head can be refitted in seconds
without touching the encoder again.  Trains on the ``all`` pooling, which is the
published protocol, and selects on validation AUC.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval_downstream import LinearHead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat_cache", default=r"D:\jepa_phase0\reports\region_features")
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--region", default="all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\refit_heads")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    src = pathlib.Path(args.feat_cache) / f"{args.tag}_s{args.num_slices}.pt"
    d = torch.load(src, map_location="cpu")
    F, L = d["feats"], d["labels"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = F["Training"][args.region].to(dev)
    Xva = F["Validation"][args.region].to(dev)
    Xte = F["Test"][args.region].to(dev)
    ytr = L["Training"].float().to(dev)
    yva, yte = L["Validation"].numpy(), L["Test"].numpy()

    torch.manual_seed(args.seed)
    head = LinearHead(in_dim=Xtr.size(1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    best = dict(val=-1.0)
    for ep in range(1, args.epochs + 1):
        lr = args.lr * (ep / max(args.warmup, 1) if ep <= args.warmup else
                        0.5 * (1 + np.cos(np.pi * (ep - args.warmup) /
                                          max(args.epochs - args.warmup, 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        head.train()
        perm = torch.randperm(Xtr.size(0), device=dev)
        for s in range(0, Xtr.size(0), args.batch):
            idx = perm[s:s + args.batch]
            opt.zero_grad()
            crit(head(Xtr[idx]).squeeze(-1), ytr[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva).squeeze(-1).cpu().numpy()
            pt = head(Xte).squeeze(-1).cpu().numpy()
        va = roc_auc_score(yva, pv)
        if va > best["val"]:
            best = dict(val=float(va), test=float(roc_auc_score(yte, pt)),
                        epoch=ep,
                        state={k: v.detach().cpu().clone()
                               for k, v in head.state_dict().items()})

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.tag}_head.pt"
    torch.save({"head": best["state"], "val_auc": best["val"],
                "test_auc": best["test"], "epoch": best["epoch"]}, dst)
    print(f"{args.tag}: val {best['val']:.4f}  test {best['test']:.4f} "
          f"(ep {best['epoch']})  -> {dst}")


if __name__ == "__main__":
    main()
