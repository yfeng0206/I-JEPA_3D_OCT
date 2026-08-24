"""P8b: clinical operating points and calibration, zero GPU.

Round-2 clinical review: "The fairness audit is too AUC-centric ... there is no
subgroup calibration, sensitivity at a clinically chosen specificity, predictive
value, or threshold transfer."

AUC is a threshold-free summary. A screening tool is deployed at a threshold, so
this reports what a reader actually needs:

  * sensitivity at a FIXED clinical specificity (0.85 and 0.90), with the
    threshold chosen on the VALIDATION split and then transferred unchanged to
    test - which is the honest simulation of deployment
  * the achieved test specificity after that transfer, which shows whether the
    threshold survives the shift
  * positive and negative predictive value at the cohort prevalence
  * calibration: Brier score and 15-bin expected calibration error

All from stored per-case predictions. Paired bootstrap over subjects gives the
interval on the CHANGE in sensitivity between two arms.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p8b_operating_points.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_curve, brier_score_loss

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
REPO = r"C:\Users\Gary\Desktop\jepa"
N_BOOT = 10000
RNG = np.random.default_rng(20260823)
TARGET_SPEC = [0.85, 0.90]

ARMS = {
    "random":    r"results\downstream\meanpool_sweep_random\ep%d_test_predictions.npz",
    "envelope":  r"results\downstream\meanpool_sweep_mirage\ep%d_test_predictions.npz",
    "intensity": r"results\downstream\meanpool_sweep_oracle\ep%d_test_predictions.npz",
}
VAL = {
    "random": r"results\downstream\meanpool_sweep_random\ep%d_val_predictions.npz",
}


def load(p):
    z = np.load(p)
    return z["labels"].astype(int), z["probs"].astype(np.float64)


def thr_at_spec(y, p, target):
    """Most permissive threshold whose specificity still reaches `target`.

    roc_curve returns thresholds in DECREASING order, so specificity (1 - fpr)
    decreases along the array while sensitivity increases. The operating point we
    want is the LAST index whose specificity still meets the target, because that
    maximises sensitivity subject to the constraint. Taking the first index
    instead selects the degenerate threshold at +inf, which classifies nothing as
    positive and yields sensitivity 0.
    """
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero((1.0 - fpr) >= target)
    if len(ok) == 0:
        return float(np.nanmin(thr[np.isfinite(thr)]))
    t = thr[ok[-1]]
    if not np.isfinite(t):
        finite = thr[np.isfinite(thr)]
        t = finite[0] if len(finite) else 0.5
    return float(t)


def metrics_at(y, p, t, prev=None):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"threshold": t, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "balanced_accuracy": (sens + spec) / 2,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    w = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = w == b
        if m.sum() == 0:
            continue
        e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ep = 100
    # the validation split is shared across arms, so each arm picks its OWN
    # threshold on its OWN validation predictions - never on test
    out = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "epoch": ep, "n_bootstrap": N_BOOT,
           "protocol": "threshold selected on the validation split at a fixed target "
                       "specificity, then transferred unchanged to the test split",
           "target_specificities": TARGET_SPEC, "arms": {}}

    y = probs = None
    S = {}
    for arm, pat in ARMS.items():
        yt, pt = load(os.path.join(REPO, pat % ep))
        if y is None:
            y = yt
        assert np.array_equal(yt, y)
        S[arm] = pt

    # validation predictions exist only for the random sweep in the repo tree;
    # fall back to that shared split for threshold selection and say so.
    vp = os.path.join(REPO, VAL["random"] % ep)
    has_val = os.path.exists(vp)
    out["threshold_source"] = ("per-arm validation split" if has_val
                               else "test split (NO validation available - flagged)")
    vy, vpr = load(vp) if has_val else (y, S["random"])

    prev = float(y.mean())
    out["prevalence"] = prev

    for arm in ARMS:
        e = {"auc_note": "see Table 1", "brier": float(brier_score_loss(y, S[arm])),
             "ece_15bin": ece(y, S[arm]), "at": {}}
        for ts in TARGET_SPEC:
            # threshold from validation predictions of THIS arm if present,
            # else from the shared random validation split
            avp = os.path.join(REPO, ARMS[arm].replace("test_", "val_") % ep)
            if os.path.exists(avp):
                ay, apr = load(avp)
                t = thr_at_spec(ay, apr, ts)
                src = "own validation split"
            else:
                t = thr_at_spec(vy, vpr, ts)
                src = "shared validation split (arm's own not stored)"
            m = metrics_at(y, S[arm], t)
            m["threshold_source"] = src
            e["at"]["spec%02d" % int(ts * 100)] = m
        out["arms"][arm] = e

    # paired bootstrap on the CHANGE in sensitivity at target specificity
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out["contrasts"] = {}
    for b_arm, a_arm in (("intensity", "random"), ("envelope", "random")):
        key = "%s_minus_%s" % (b_arm, a_arm)
        entry = {}
        for ts in TARGET_SPEC:
            k = "spec%02d" % int(ts * 100)
            tb = out["arms"][b_arm]["at"][k]["threshold"]
            ta = out["arms"][a_arm]["at"][k]["threshold"]
            point = out["arms"][b_arm]["at"][k]["sensitivity"] - out["arms"][a_arm]["at"][k]["sensitivity"]
            d = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = np.concatenate([RNG.choice(pos, pos.size, True),
                                      RNG.choice(neg, neg.size, True)])
                yy = y[idx]
                pmask = yy == 1
                sb = (S[b_arm][idx][pmask] >= tb).mean()
                sa = (S[a_arm][idx][pmask] >= ta).mean()
                d[i] = sb - sa
            lo, hi = np.percentile(d, [2.5, 97.5])
            entry[k] = {"delta_sensitivity": float(point),
                        "ci95_lo": float(lo), "ci95_hi": float(hi),
                        "excludes_zero": bool(lo > 0 or hi < 0)}
            print("  %s at spec %.2f: dSens=%+.4f [%+.4f,%+.4f]" % (key, ts, point, lo, hi), flush=True)
        out["contrasts"][key] = entry

    with open(os.path.join(OUT, "p8b_operating_points.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\ncohort prevalence %.4f | threshold source: %s" % (prev, out["threshold_source"]))
    for ts in TARGET_SPEC:
        k = "spec%02d" % int(ts * 100)
        print("\n--- target validation specificity %.2f ---" % ts)
        print("%-11s %9s %9s %7s %7s %8s %8s" % ("arm", "sens", "spec(test)", "ppv", "npv", "brier", "ece"))
        for arm in ARMS:
            m = out["arms"][arm]["at"][k]
            print("%-11s %9.4f %9.4f %7.4f %7.4f %8.4f %8.4f" % (
                arm, m["sensitivity"], m["specificity"], m["ppv"], m["npv"],
                out["arms"][arm]["brier"], out["arms"][arm]["ece_15bin"]))
    print("\nwrote", os.path.join(OUT, "p8b_operating_points.json"))


if __name__ == "__main__":
    main()
