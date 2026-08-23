"""P1c: paired statistics over the corrected evidence inventory.

Two things this fixes relative to p1_paired_stats.py:
  1. correct arm labels (the null is `meanpool_sweep_random`, not the retracted
     `cover_random` arm)
  2. precision is tracked, and cross-precision contrasts are FLAGGED rather than
     silently reported

Comparisons are organised into families so that no contrast silently mixes
either epoch or precision:

  A. within-precision, matched-epoch  -> the defensible primary contrasts
  B. cross-precision, matched-epoch   -> reported but flagged CONFOUNDED
  C. precision robustness             -> same encoder, fp16 vs fp32

A fast exact rank-based AUC (ties handled by average ranks) replaces
sklearn.roc_auc_score inside the bootstrap; it is verified against sklearn to
1e-12 before use.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p1c_stats.json
"""
import json
import os
import sys
import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_paired_stats import delong_test  # noqa: E402

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
INV = os.path.join(OUT, "p1b_full_inventory.json")
N_BOOT = 10000
RNG = np.random.default_rng(20260822)


def fast_auc(y, s):
    """Exact ROC-AUC via average ranks. Equals sklearn.roc_auc_score."""
    r = sps.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    inv = json.load(open(INV))
    recs = [r for r in inv["records"]
            if r["family"] == "frozen_probe" and r["status"] == "primary"]

    y = None
    S = {}
    meta = {}
    for r in recs:
        z = np.load(r["path"])
        lab = z["labels"].astype(int)
        if y is None:
            y = lab
        assert np.array_equal(lab, y)
        key = "%s@ep%s@%s" % (r["arm"], r["epoch"], r["precision"])
        S[key] = z["probs"].astype(np.float64)
        meta[key] = r

    # ---- verify the fast AUC against sklearn before trusting it
    err = max(abs(fast_auc(y, s) - roc_auc_score(y, s)) for s in S.values())
    assert err < 1e-12, "fast_auc disagrees with sklearn by %g" % err
    print("fast_auc verified against sklearn, max abs error %.2e" % err)
    print("%d primary frozen probes on n=%d (pos=%d)" % (len(S), len(y), y.sum()))

    # ---- paired bootstrap, one shared index set per draw
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keys = list(S)
    boot = {k: np.empty(N_BOOT) for k in keys}
    for b in range(N_BOOT):
        idx = np.concatenate([RNG.choice(pos, pos.size, True), RNG.choice(neg, neg.size, True)])
        yy = y[idx]
        for k in keys:
            boot[k][b] = fast_auc(yy, S[k][idx])
        if (b + 1) % 2000 == 0:
            print("  bootstrap %d/%d" % (b + 1, N_BOOT), flush=True)

    table = []
    for k in keys:
        r = meta[k]
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        table.append({"key": k, "arm": r["arm"], "epoch": r["epoch"],
                      "precision": r["precision"], "tag": r["tag"],
                      "auc": float(fast_auc(y, S[k])),
                      "ci95_lo": float(lo), "ci95_hi": float(hi),
                      "boot_sd": float(boot[k].std(ddof=1))})
    table.sort(key=lambda t: (t["arm"], t["epoch"]))

    # ---- contrasts
    contrasts = []
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i >= j:
                continue
            a, b = keys[i], keys[j]
            ra, rb = meta[a], meta[b]
            same_ep = ra["epoch"] == rb["epoch"]
            same_pr = ra["precision"] == rb["precision"]
            same_arm = ra["arm"] == rb["arm"]
            if same_arm and same_ep and not same_pr:
                kind = "C_precision_robustness"
            elif same_ep and same_pr and not same_arm:
                kind = "A_primary_matched"
            elif same_ep and not same_pr and not same_arm:
                kind = "B_CONFOUNDED_cross_precision"
            else:
                kind = "D_other"
            d, se, p, ci = delong_test(y, S[a], S[b])
            db = boot[a] - boot[b]
            contrasts.append({
                "a": a, "b": b, "kind": kind,
                "arm_a": ra["arm"], "arm_b": rb["arm"], "epoch": ra["epoch"] if same_ep else None,
                "prec_a": ra["precision"], "prec_b": rb["precision"],
                "auc_a": float(fast_auc(y, S[a])), "auc_b": float(fast_auc(y, S[b])),
                "delta_a_minus_b": float(d), "delong_se": float(se), "delong_p": float(p),
                "delong_ci95": list(ci),
                "boot_ci95_lo": float(np.percentile(db, 2.5)),
                "boot_ci95_hi": float(np.percentile(db, 97.5)),
                "boot_p_two_sided": float(2 * min((db <= 0).mean(), (db >= 0).mean())),
            })

    # ---- multiplicity -----------------------------------------------------
    # Two families are corrected separately, because pooling them would apply a
    # 13-test penalty to a 9-test confirmatory claim (and the manuscript
    # displays exactly nine).
    #
    #   CONFIRMATORY: the nine fp16 contrasts among random / oracle / envelope
    #   at epochs 50, 75 and 100. These are the contrasts Table 2 reports.
    #   EXPLORATORY : every other matched-epoch, matched-precision contrast.
    def bh(ps):
        ps = np.asarray(ps, dtype=float)
        order = np.argsort(ps)
        n = len(ps)
        q = np.empty(n)
        prev = 1.0
        for rank in range(n - 1, -1, -1):
            k = order[rank]
            prev = min(prev, ps[k] * n / (rank + 1))
            q[k] = prev
        return q

    PRIMARY_ARMS = {"random", "oracle", "envelope"}
    famA = [c for c in contrasts if c["kind"] == "A_primary_matched"]
    confirm = [c for c in famA
               if c["prec_a"] == "fp16" and c["prec_b"] == "fp16"
               and c["arm_a"] in PRIMARY_ARMS and c["arm_b"] in PRIMARY_ARMS]
    explore = [c for c in famA if c not in confirm]

    for c, qq in zip(confirm, bh([c["delong_p"] for c in confirm])):
        c["family"] = "confirmatory"
        c["delong_q_bh"] = float(qq)
        c["bh_family_size"] = len(confirm)
    if explore:
        for c, qq in zip(explore, bh([c["delong_p"] for c in explore])):
            c["family"] = "exploratory"
            c["delong_q_bh"] = float(qq)
            c["bh_family_size"] = len(explore)

    out = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "n_test": int(len(y)), "n_pos": int(y.sum()), "n_bootstrap": N_BOOT,
           "bootstrap_seed": 20260822,
           "family_key": {
               "A_primary_matched": "same epoch, same precision, different arm - defensible",
               "B_CONFOUNDED_cross_precision": "same epoch, DIFFERENT precision - flagged, not for headline claims",
               "C_precision_robustness": "same arm and epoch, fp16 vs fp32 - measures the precision effect",
               "D_other": "different epochs - not a valid arm comparison"},
           "multiplicity": {
               "confirmatory_family_size": len(confirm),
               "exploratory_family_size": len(explore),
               "note": "BH applied separately within each family; the confirmatory family is "
                       "exactly the nine fp16 contrasts among random/oracle/envelope at "
                       "epochs 50/75/100 that the manuscript reports"},
           "table": table, "contrasts": contrasts}
    with open(os.path.join(OUT, "p1c_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "p1c_bootstrap.npz"), **boot)

    print("\n%-26s %-6s %8s  %-20s" % ("arm@epoch", "prec", "AUC", "95% CI"))
    for t in table:
        print("%-26s %-6s %8.4f  [%.4f, %.4f]" % (
            "%s@ep%s" % (t["arm"], t["epoch"]), t["precision"], t["auc"], t["ci95_lo"], t["ci95_hi"]))

    print("\n--- family A: matched epoch AND matched precision ---")
    for c in sorted(famA, key=lambda c: (c["epoch"], c["arm_a"])):
        print("  ep%-4s %-11s - %-11s  d=%+.4f  p=%.5f q=%.5f  CI[%+.4f,%+.4f]" % (
            c["epoch"], c["arm_a"], c["arm_b"], c["delta_a_minus_b"],
            c["delong_p"], c["delong_q_bh"], c["boot_ci95_lo"], c["boot_ci95_hi"]))

    fc = [c for c in contrasts if c["kind"] == "C_precision_robustness"]
    if fc:
        print("\n--- family C: precision robustness (same encoder) ---")
        for c in fc:
            print("  %-11s ep%-4s fp16 vs fp32: d=%+.6f  p=%.4f" % (
                c["arm_a"], c["epoch"], c["delta_a_minus_b"], c["delong_p"]))

    nB = sum(1 for c in contrasts if c["kind"] == "B_CONFOUNDED_cross_precision")
    print("\ncross-precision contrasts flagged CONFOUNDED: %d" % nB)
    print("wrote", os.path.join(OUT, "p1c_stats.json"))


if __name__ == "__main__":
    main()
