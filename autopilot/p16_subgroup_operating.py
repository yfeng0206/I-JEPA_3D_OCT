"""P16: subgroup operating points and calibration, zero GPU.

Round-3 clinical review: "The subgroup audit remains AUC-only. Aggregate
operating points and calibration were added, but there is still no subgroup
calibration, subgroup fixed-specificity sensitivity, predictive values."

AUC is threshold-free; a screening tool is deployed at one threshold, and that
threshold is shared across subgroups. So the clinically load-bearing question is
not whether every group's AUC rises, it is what a SINGLE deployed threshold does
to each group. A model can improve every subgroup's AUC and still deliver very
different sensitivity to each group at the operating point actually shipped.

Method, matching p8b for the aggregate case:
  * the threshold is chosen on the VALIDATION split to hit a target specificity,
    then transferred UNCHANGED to test - the honest simulation of deployment
  * one threshold for everyone, exactly as a deployed screen would use
  * per subgroup: sensitivity, achieved specificity, PPV, NPV, Brier, 15-bin ECE
  * the change from the null to the best arm, per subgroup

Threshold selection repeats p8b's fix: roc_curve returns thresholds in
DECREASING order, so specificity is INCREASING along the array. The most
permissive threshold meeting the target is the LAST index that still meets it;
taking the first gives the degenerate +inf threshold and sensitivity 0.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p16_subgroup_operating.json
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_curve, brier_score_loss

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
REPO = r"C:\Users\Gary\Desktop\jepa"
META = os.path.join(OUT, "test_metadata.csv")
TARGET_SPEC = 0.90
EPOCH = 100
MIN_N = 50
MIN_CLASS = 10

ARMS = {
    "random":    r"results\downstream\meanpool_sweep_random\ep%d_test_predictions.npz",
    "envelope":  r"results\downstream\meanpool_sweep_mirage\ep%d_test_predictions.npz",
    "intensity": r"results\downstream\meanpool_sweep_oracle\ep%d_test_predictions.npz",
}
VAL_RANDOM = r"results\downstream\meanpool_sweep_random\ep%d_val_predictions.npz"
GROUPS = [("race_label", "race"), ("sex_label", "sex")]


def load(p):
    z = np.load(p)
    return z["labels"].astype(int), z["probs"].astype(np.float64)


def thr_at_spec(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1.0 - fpr
    ok = np.where(spec >= target)[0]
    if len(ok) == 0:
        return float("inf")
    return float(thr[ok[-1]])


def ece(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}


def main():
    meta = list(csv.DictReader(open(META, encoding="utf-8")))
    print("metadata rows: %d" % len(meta))

    # threshold from the validation split of the NULL arm, transferred unchanged
    yv, pv = load(os.path.join(REPO, VAL_RANDOM % EPOCH))
    thr = thr_at_spec(yv, pv, TARGET_SPEC)
    print("threshold from validation at spec>=%.2f: %.6f" % (TARGET_SPEC, thr))

    res = {"target_specificity": TARGET_SPEC, "epoch": EPOCH,
           "threshold": thr, "threshold_source": "random val split, transferred",
           "note": "one shared threshold for all subgroups, as deployed",
           "arms": {}}

    per_arm_group = {}
    for arm, pat in ARMS.items():
        path = os.path.join(REPO, pat % EPOCH)
        if not os.path.exists(path):
            print("  %-10s MISSING %s" % (arm, path))
            continue
        y, p = load(path)
        if len(y) != len(meta):
            print("  %-10s length mismatch %d vs %d - skipping" % (arm, len(y), len(meta)))
            continue
        entry = {"overall": metrics(y, p, thr), "groups": {}}
        for col, gname in GROUPS:
            vals = sorted({r[col] for r in meta if r[col]})
            g = {}
            for v in vals:
                m = np.array([r[col] == v for r in meta])
                yy, pp = y[m], p[m]
                if len(yy) < MIN_N or yy.sum() < MIN_CLASS or (len(yy) - yy.sum()) < MIN_CLASS:
                    g[v] = {"n": int(len(yy)), "underpowered": True}
                    continue
                g[v] = metrics(yy, pp, thr)
                g[v]["underpowered"] = False
            entry["groups"][gname] = g
        res["arms"][arm] = entry
        per_arm_group[arm] = entry
        print("  %-10s overall sens=%.4f spec=%.4f ece=%.4f"
              % (arm, entry["overall"]["sensitivity"],
                 entry["overall"]["specificity"], entry["overall"]["ece"]))

    # change from null to best arm, per subgroup
    if "random" in per_arm_group and "intensity" in per_arm_group:
        delta = {}
        for col, gname in GROUPS:
            d = {}
            a = per_arm_group["random"]["groups"].get(gname, {})
            b = per_arm_group["intensity"]["groups"].get(gname, {})
            for v in a:
                if a[v].get("underpowered") or b.get(v, {}).get("underpowered"):
                    continue
                d[v] = {"d_sensitivity": b[v]["sensitivity"] - a[v]["sensitivity"],
                        "d_specificity": b[v]["specificity"] - a[v]["specificity"],
                        "d_ece": b[v]["ece"] - a[v]["ece"],
                        "n": a[v]["n"]}
            delta[gname] = d
        res["delta_random_to_intensity"] = delta
        print("\nchange from random to intensity at the shared threshold:")
        for gname, d in delta.items():
            for v, s in sorted(d.items()):
                print("  %-8s %-22s d_sens=%+.4f  d_spec=%+.4f  n=%d"
                      % (gname, v, s["d_sensitivity"], s["d_specificity"], s["n"]))

    outp = os.path.join(OUT, "p16_subgroup_operating.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nwrote %s" % outp)


if __name__ == "__main__":
    main()
