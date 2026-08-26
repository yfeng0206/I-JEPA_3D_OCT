"""CPU-only. Re-analyse ALREADY-SAVED pooled features:
  D:\jepa_phase0\reports\region_features\{arm}_ep50_s100.pt
Each holds Training(2000)/Validation(600)/Test(1000) mean-pooled 768-d vectors
for three cell regions: all / anatomy / background, plus labels.

New question this file answers (not in docs/experiments/masking/background_signal.md):
  does the BACKGROUND-position pool carry class information that is NOT linearly
  recoverable from the ANATOMY-position pool?  That is the incremental-information
  test for H-a; the published table only reports each region probed in isolation.
"""
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ARMS = ["random", "oracle", "envelope", "blob"]
SRC = r"D:\jepa_phase0\reports\region_features\{}_ep50_s100.pt"
CS = [0.001, 0.01, 0.1, 1.0, 10.0]
RNG = np.random.default_rng(42)


def probe(Xtr, ytr, Xva, yva, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    a, b, c = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = (-1, None, None)
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(a, ytr)
        v = roc_auc_score(yva, m.predict_proba(b)[:, 1])
        if v > best[0]:
            best = (v, C, m.predict_proba(c)[:, 1])
    return {"val_auc": float(best[0]), "C": best[1],
            "test_auc": float(roc_auc_score(yte, best[2]))}, best[2]


def boot_ci(y, s, n=2000):
    y = np.asarray(y); s = np.asarray(s)
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def boot_delta_ci(y, s1, s2, n=2000):
    y = np.asarray(y)
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out))]


res = {"source": SRC, "note": "ep50 checkpoints, 25 stratified slices/volume, paired splits"}
for arm in ARMS:
    d = torch.load(SRC.format(arm), map_location="cpu")
    F, L = d["feats"], d["labels"]
    y = {k: L[k].numpy() for k in L}
    X = {k: {r: F[k][r].numpy().astype(np.float64) for r in F[k]} for k in F}
    r = {}
    scores = {}
    for reg in ["all", "anatomy", "background"]:
        r[reg], scores[reg] = probe(X["Training"][reg], y["Training"],
                                    X["Validation"][reg], y["Validation"],
                                    X["Test"][reg], y["Test"])
        r[reg]["test_auc_ci95"] = boot_ci(y["Test"], scores[reg])

    cat = {k: np.concatenate([X[k]["anatomy"], X[k]["background"]], 1) for k in X}
    r["anatomy+background"], scores["cat"] = probe(cat["Training"], y["Training"],
                                                   cat["Validation"], y["Validation"],
                                                   cat["Test"], y["Test"])
    r["anatomy+background"]["test_auc_ci95"] = boot_ci(y["Test"], scores["cat"])

    # background residualised on anatomy (Ridge fit on Training only)
    rg = Ridge(alpha=10.0).fit(X["Training"]["anatomy"], X["Training"]["background"])
    bres = {k: X[k]["background"] - rg.predict(X[k]["anatomy"]) for k in X}
    r2 = 1.0 - (bres["Test"] ** 2).sum() / (
        (X["Test"]["background"] - X["Training"]["background"].mean(0)) ** 2).sum()
    r["bg_residual_on_anatomy"], scores["bres"] = probe(
        bres["Training"], y["Training"], bres["Validation"], y["Validation"],
        bres["Test"], y["Test"])
    r["bg_residual_on_anatomy"]["test_auc_ci95"] = boot_ci(y["Test"], scores["bres"])
    r["bg_residual_on_anatomy"]["ridge_test_R2_bg_from_anatomy"] = float(r2)

    # anatomy residualised on background (symmetric control)
    rg2 = Ridge(alpha=10.0).fit(X["Training"]["background"], X["Training"]["anatomy"])
    ares = {k: X[k]["anatomy"] - rg2.predict(X[k]["background"]) for k in X}
    r2b = 1.0 - (ares["Test"] ** 2).sum() / (
        (X["Test"]["anatomy"] - X["Training"]["anatomy"].mean(0)) ** 2).sum()
    r["anat_residual_on_bg"], scores["ares"] = probe(
        ares["Training"], y["Training"], ares["Validation"], y["Validation"],
        ares["Test"], y["Test"])
    r["anat_residual_on_bg"]["test_auc_ci95"] = boot_ci(y["Test"], scores["ares"])
    r["anat_residual_on_bg"]["ridge_test_R2_anatomy_from_bg"] = float(r2b)

    r["delta_cat_minus_anatomy_ci95_and_mean"] = boot_delta_ci(
        y["Test"], scores["cat"], scores["anatomy"])
    r["delta_anatomy_minus_all_ci95_and_mean"] = boot_delta_ci(
        y["Test"], scores["anatomy"], scores["all"])
    r["n"] = {k: int(len(y[k])) for k in y}
    r["test_prevalence"] = float(y["Test"].mean())
    res[arm] = r
    print("== %s ==" % arm)
    for k in ["all", "anatomy", "background", "anatomy+background",
              "bg_residual_on_anatomy", "anat_residual_on_bg"]:
        e = r[k]
        print("  %-24s testAUC %.4f  CI[%.4f,%.4f]  (val %.4f, C=%s)"
              % (k, e["test_auc"], e["test_auc_ci95"][0], e["test_auc_ci95"][1],
                 e["val_auc"], e["C"]))
    print("  ridge R2 bg<-anatomy (Test): %.4f | anatomy<-bg: %.4f"
          % (r["bg_residual_on_anatomy"]["ridge_test_R2_bg_from_anatomy"],
             r["anat_residual_on_bg"]["ridge_test_R2_anatomy_from_bg"]))
    lo, hi, mu = r["delta_cat_minus_anatomy_ci95_and_mean"]
    print("  delta (anat+bg) - anat = %+.4f  CI[%+.4f,%+.4f]" % (mu, lo, hi))

p = r"C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a2_region_incremental.json"
with open(p, "w") as f:
    json.dump(res, f, indent=2)
print("\nwrote", p)
