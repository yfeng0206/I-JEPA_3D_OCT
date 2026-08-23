"""P7: subgroup / fairness analysis, corrected inventory + fast rank AUC.

Joins saved test predictions to FairVision demographics using the index
alignment proven in p1_test_metadata.py (labels reconstructed from the sorted
file order match every predictions file exactly).

Reports per arm and per protected attribute:
  - subgroup AUC with a stratified bootstrap CI
  - worst-group AUC and the max-min AUC gap
  - TPR and FPR at a threshold selected ONCE on the validation split, so
    operating-point fairness is not tuned on the data it is reported on

Groups with n < 50 or fewer than 10 of either class are reported but marked
underpowered and excluded from gap summaries: an AUC on 22 subjects is not
interpretable and should not drive a fairness claim.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p7_fairness.json
"""
import csv
import json
import os
from collections import Counter

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_curve

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
INV = os.path.join(OUT, "p1b_full_inventory.json")
META = os.path.join(OUT, "test_metadata.csv")
N_BOOT = 3000
MIN_N, MIN_CLASS = 50, 10
RNG = np.random.default_rng(20260822)

GROUPS = [("race_label", "race"), ("sex_label", "sex"),
          ("hispanic_label", "ethnicity"), ("language_label", "language"),
          ("marital_label", "marital_status")]


def fast_auc(y, s):
    r = sps.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_ci(y, p, n_boot=N_BOOT):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    v = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, pos.size, True), RNG.choice(neg, neg.size, True)])
        v[b] = fast_auc(y[idx], p[idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    with open(META, newline="", encoding="utf-8") as f:
        meta = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    y = np.array([int(r["glaucoma"]) for r in meta])

    inv = json.load(open(INV))
    recs = [r for r in inv["records"]
            if r["family"] == "frozen_probe" and r["status"] == "primary"]

    res = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "n_test": int(len(y)), "n_bootstrap": N_BOOT,
           "min_group_n": MIN_N, "min_per_class": MIN_CLASS,
           "threshold_source": "validation split, Youden J, selected once per probe; "
                               "falls back to 0.5 when no val_predictions.npz exists",
           "arms": {}}

    gmasks = {}
    for col, gname in GROUPS:
        vals = sorted(set(r[col] for r in meta))
        gmasks[gname] = [(v, np.array([r[col] == v for r in meta])) for v in vals]

    for r in sorted(recs, key=lambda r: (r["arm"], r["epoch"])):
        z = np.load(r["path"])
        probs = z["probs"].astype(np.float64)
        assert np.array_equal(z["labels"].astype(int), y)

        vp = os.path.join(os.path.dirname(r["path"]),
                          os.path.basename(r["path"]).replace("test_", "val_"))
        thr, thr_src = 0.5, "default_0.5"
        if os.path.exists(vp):
            vz = np.load(vp)
            vy, vpr = vz["labels"].astype(int), vz["probs"].astype(np.float64)
            fpr, tpr, t = roc_curve(vy, vpr)
            thr, thr_src = float(t[np.argmax(tpr - fpr)]), "validation_youden"

        key = "%s@ep%s@%s" % (r["arm"], r["epoch"], r["precision"])
        entry = {"arm": r["arm"], "epoch": r["epoch"], "precision": r["precision"],
                 "tag": r["tag"], "overall_auc": float(fast_auc(y, probs)),
                 "threshold": thr, "threshold_source": thr_src, "groups": {}}

        for gname, items in gmasks.items():
            per = {}
            for v, m in items:
                n = int(m.sum())
                npos = int(y[m].sum())
                nneg = n - npos
                if npos < 1 or nneg < 1:
                    per[v] = {"n": n, "n_pos": npos, "n_neg": nneg,
                              "underpowered": True, "auc": None}
                    continue
                under = (n < MIN_N) or (npos < MIN_CLASS) or (nneg < MIN_CLASS)
                a = float(fast_auc(y[m], probs[m]))
                lo, hi = boot_ci(y[m], probs[m]) if not under else (float("nan"), float("nan"))
                pred = (probs[m] >= thr).astype(int)
                per[v] = {"n": n, "n_pos": npos, "n_neg": nneg, "underpowered": bool(under),
                          "auc": a, "auc_ci95_lo": lo, "auc_ci95_hi": hi,
                          "tpr": float(pred[y[m] == 1].mean()),
                          "fpr": float(pred[y[m] == 0].mean())}
            ok = {k: v for k, v in per.items() if v.get("auc") is not None and not v["underpowered"]}
            summ = {}
            if len(ok) >= 2:
                aucs = {k: v["auc"] for k, v in ok.items()}
                w = min(aucs, key=aucs.get)
                bst = max(aucs, key=aucs.get)
                summ = {"n_groups": len(ok), "worst_group": w, "worst_auc": aucs[w],
                        "best_group": bst, "best_auc": aucs[bst],
                        "auc_gap": aucs[bst] - aucs[w],
                        "tpr_gap": max(v["tpr"] for v in ok.values()) - min(v["tpr"] for v in ok.values()),
                        "fpr_gap": max(v["fpr"] for v in ok.values()) - min(v["fpr"] for v in ok.values())}
            entry["groups"][gname] = {"per_group": per, "summary": summ}

        res["arms"][key] = entry
        print("%-26s %-5s overall=%.4f | race worst=%-6s gap=%.4f | sex gap=%.4f" % (
            key, r["precision"], entry["overall_auc"],
            entry["groups"]["race"]["summary"].get("worst_group", "-"),
            entry["groups"]["race"]["summary"].get("auc_gap", float("nan")),
            entry["groups"]["sex"]["summary"].get("auc_gap", float("nan"))), flush=True)

    worst = [e["groups"]["race"]["summary"].get("worst_group")
             for e in res["arms"].values() if e["groups"]["race"]["summary"]]
    res["worst_race_group_across_probes"] = dict(Counter(worst))
    res["n_probes_with_race_summary"] = len(worst)

    with open(os.path.join(OUT, "p7_fairness.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nworst race group across %d probes: %s" % (len(worst), dict(Counter(worst))))
    print("wrote", os.path.join(OUT, "p7_fairness.json"))


if __name__ == "__main__":
    main()
