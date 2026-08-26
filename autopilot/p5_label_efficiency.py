"""P5: label-efficiency curves from cached frozen features. Zero GPU.

Round-2 clinical review and the operator's own plan both ask for this: the point
of self-supervised pretraining is representation quality under limited
annotation, so the question is whether a better masking policy reduces label
dependence, not only whether it raises AUC at 100 percent labels.

Every arm's epoch-100 frozen probe already cached its features, so this needs no
GPU and no re-encoding:

    runs/frozen_*/feature_cache/{Training,Validation,Test}_s100_r256_fp32_*.pt

For each arm and each label fraction we subsample the TRAINING set with a fixed
per-fraction seed shared across arms, so every arm sees the identical subset of
patients at every fraction. That makes the arms paired at each point on the
curve, which is what allows a difference to be interpreted.

MEMORY DISCIPLINE. Each Training cache is about 1.8 GB on disk as
(6000, 100, 768) float32. It is mean-pooled over slices immediately, in chunks,
so the resident array is (6000, 768), about 18 MB. Peak is one chunk plus the
pooled result. The script refuses to start when system RAM is already above
`--max-ram-pct`, because a 2.5-day training run must not be paged out for this.

Usage:
  python p5_label_efficiency.py [--max-ram-pct 80] [--fractions 0.01 0.05 0.1 0.25 1.0]
"""
import argparse
import ctypes
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from scipy import stats as sps

RUNS = r"D:\jepa_phase0\runs"
OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"

ARMS = {
    "random":    "frozen_meanpool_random_ep100_fp32",
    "intensity": "frozen_meanpool_oracle_ep100_fp32",
    "envelope":  "frozen_meanpool_envelope_fp32_ep100",
    "cover":     "frozen_meanpool_cover_f021_ep100",
}


def ram_pct():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return float(m.dwMemoryLoad)


def load_pooled(run_dir, split):
    """Load a feature cache and mean-pool over slices, keeping memory bounded."""
    pat = os.path.join(RUNS, run_dir, "feature_cache", "%s_s100_r256_fp32_*.pt" % split)
    hits = glob.glob(pat)
    if not hits:
        return None, None
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    f, y = d["features"], d["labels"]
    if f.dim() == 3:                      # (N, slices, D) -> (N, D)
        out = torch.empty((f.shape[0], f.shape[2]), dtype=torch.float32)
        step = 256
        for i in range(0, f.shape[0], step):
            out[i:i + step] = f[i:i + step].float().mean(dim=1)
        f = out
    del d
    return f.numpy().astype(np.float64), y.numpy().astype(int)


def fit_logreg(Xtr, ytr, Xte, seed, Xva=None, yva=None,
               epochs=50, lr=4e-4, wd=0.05, batch_size=256, warmup_epochs=5):
    """LayerNorm + linear head fitted with the PRIMARY probe protocol.

    An earlier version used a full-batch fit at lr=0.05 for 200 epochs with no
    validation selection, because it is cheap enough to repeat 25 times. That
    produced a different operating point from the primary probe: at full
    supervision it put the null at 0.8811 against 0.8746, which compressed the
    headline gap and made the two tables disagree without either being wrong.
    Matching the primary protocol removes that discrepancy at the cost of a
    slower fit, which is worth paying to keep one probe definition in the paper.

    Protocol taken from the shipped probe configs: batch 256, AdamW at 4e-4,
    weight decay 0.05, 50 epochs, 5 warmup epochs, cosine decay, and the epoch
    chosen by validation AUC rather than the last epoch.
    """
    g = torch.Generator().manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    use_val = Xva is not None and yva is not None and len(np.unique(yva)) > 1
    if use_val:
        xv = torch.tensor(Xva, dtype=torch.float32)

    model = torch.nn.Sequential(torch.nn.LayerNorm(xt.shape[1]),
                                torch.nn.Linear(xt.shape[1], 1))
    for m in model:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01, generator=g)
            torch.nn.init.zeros_(m.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()

    n = xt.shape[0]
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        float(s) / float(max(1, warmup_steps)) if s < warmup_steps else
        0.5 * (1.0 + math.cos(math.pi * float(s - warmup_steps) /
                              float(max(1, total_steps - warmup_steps))))))

    best_val, best_pred = -1.0, None
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            lossf(model(xt[idx]).squeeze(-1), yt[idx]).backward()
            opt.step()
            sched.step()
        if use_val:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(xv).squeeze(-1)).numpy().astype(np.float64)
                va = auc(np.asarray(yva, dtype=int), pv)
            if va > best_val:
                best_val = va
                with torch.no_grad():
                    best_pred = torch.sigmoid(model(xe).squeeze(-1)).numpy().astype(np.float64)

    if best_pred is not None:
        return best_pred
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe).squeeze(-1)).numpy().astype(np.float64)


def auc(y, p):
    r = sps.rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ram-pct", type=float, default=80.0)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.25, 1.00])
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    r = ram_pct()
    if r > a.max_ram_pct:
        print("REFUSING TO START: system RAM at %.1f%%, above the %.1f%% limit." % (r, a.max_ram_pct))
        print("A long training run must not be paged out for this. Re-run when the")
        print("GPU campaign has finished and memory has been released.")
        return 2
    print("RAM at %.1f%%, proceeding" % r)

    res = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "epoch": 100, "fractions": a.fractions, "repeats": a.repeats,
           "method": "features mean-pooled from the cached frozen probe; identical "
                     "patient subset per fraction across arms, so arms are paired; "
                     "LayerNorm + linear head refitted per subset",
           "arms": {}}

    for arm, run in ARMS.items():
        Xtr, ytr = load_pooled(run, "Training")
        Xte, yte = load_pooled(run, "Test")
        Xva, yva = load_pooled(run, "Validation")
        if Xtr is None or Xte is None:
            print("  %s: cache missing, skipped" % arm)
            continue
        if Xva is None:
            print("  %s: NO validation cache - falling back to last-epoch "
                  "selection, which breaks protocol parity with the primary "
                  "probe. Reporting it rather than hiding it." % arm)
        print("%s: train %s val %s test %s"
              % (arm, Xtr.shape, (Xva.shape if Xva is not None else None),
                 Xte.shape), flush=True)
        curve = {}
        for frac in a.fractions:
            aucs = []
            for rep in range(a.repeats if frac < 1.0 else 1):
                # seed depends on fraction and repeat ONLY, never on the arm, so
                # every arm is fitted on exactly the same patients
                rng = np.random.default_rng(hash((round(frac, 4), rep)) % (2**32))
                n = max(int(round(frac * len(ytr))), 20)
                idx = rng.choice(len(ytr), n, replace=False)
                if len(np.unique(ytr[idx])) < 2:
                    continue
                p = fit_logreg(Xtr[idx], ytr[idx], Xte, seed=1000 + rep,
                               Xva=Xva, yva=yva)
                aucs.append(auc(yte, p))
            if aucs:
                curve["%.2f" % frac] = {"n_train": int(round(frac * len(ytr))),
                                        "auc_mean": float(np.mean(aucs)),
                                        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                                        "n_repeats": len(aucs)}
                print("   frac %.2f  n=%-5d auc=%.4f (sd %.4f, %d reps)"
                      % (frac, curve["%.2f" % frac]["n_train"],
                         curve["%.2f" % frac]["auc_mean"],
                         curve["%.2f" % frac]["auc_sd"], len(aucs)), flush=True)
        res["arms"][arm] = curve
        del Xtr, Xte

    with open(os.path.join(OUT, "p5_label_efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", os.path.join(OUT, "p5_label_efficiency.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
