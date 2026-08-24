"""P7c: paired intervals on subgroup CHANGES between two policies.

Round-2 review, correctly: "the relevant quantity - paired difference in gaps
between policies - has no interval." The paper reports each subgroup's AUC with
a marginal interval, which cannot answer whether one policy helps a subgroup
MORE than another, because the two policies are scored on the same subjects and
their errors are correlated.

This computes, at a matched epoch, for a pair of arms (B vs A):

  1. per-subgroup   dAUC_g = AUC_g(B) - AUC_g(A), with a paired bootstrap CI
  2. the disparity change  d(gap) = gap(B) - gap(A), also paired
  3. the differential benefit  dAUC_worst - dAUC_best, i.e. whether the
     worst-served group gained more than the best-served one

Every bootstrap draw resamples subjects ONCE and applies the same resampled
index set to both arms and to every subgroup, which preserves the pairing.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p7c_paired_subgroup.json
"""
import csv
import json
import os

import numpy as np
from scipy import stats as sps

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
META = os.path.join(OUT, "test_metadata.csv")
REPO = r"C:\Users\Gary\Desktop\jepa"
RUNS = r"D:\jepa_phase0\runs"
N_BOOT = 10000
RNG = np.random.default_rng(20260823)
MIN_N, MIN_CLASS = 50, 10

# (label, path) at matched epoch 100
ARMS = {
    "random": os.path.join(REPO, r"results\downstream\meanpool_sweep_random\ep100_test_predictions.npz"),
    "envelope": os.path.join(REPO, r"results\downstream\meanpool_sweep_mirage\ep100_test_predictions.npz"),
    "intensity": os.path.join(REPO, r"results\downstream\meanpool_sweep_oracle\ep100_test_predictions.npz"),
}
CONTRASTS = [("intensity", "random"), ("envelope", "random")]
GROUPS = [("race_label", "race"), ("sex_label", "sex")]
SEV_BINS = [(-100, -12, "severe"), (-12, -6, "moderate"), (-6, -2, "mild")]


def fast_auc(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 1 or n0 < 1:
        return np.nan
    r = sps.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    with open(META, newline="", encoding="utf-8") as f:
        meta = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    y = np.array([int(r["glaucoma"]) for r in meta])

    # severity needs md from the released metadata CSV
    md_by_file = {}
    csvp = r"D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv"
    with open(csvp, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            md_by_file[r["filename"]] = float(r["md"])
    md = np.array([md_by_file[r["file"]] for r in meta])

    S = {}
    for k, p in ARMS.items():
        z = np.load(p)
        assert np.array_equal(z["labels"].astype(int), y), "test split mismatch in %s" % k
        S[k] = z["probs"].astype(np.float64)

    # build the subgroup masks once
    masks = {}
    for col, gname in GROUPS:
        for v in sorted(set(r[col] for r in meta)):
            m = np.array([r[col] == v for r in meta])
            npos, nneg = int(y[m].sum()), int((~y[m].astype(bool)).sum())
            if m.sum() < MIN_N or npos < MIN_CLASS or nneg < MIN_CLASS:
                continue
            masks[(gname, v)] = m
    # severity strata are scored against the SHARED pool of all negatives
    neg = y == 0
    for lo, hi, nm in SEV_BINS:
        sel = (md > lo) & (md <= hi) & (y == 1)
        if sel.sum() >= MIN_CLASS:
            masks[("severity", nm)] = sel | neg

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    out = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "epoch": 100, "n_bootstrap": N_BOOT, "seed": 20260823,
           "method": "one subject resample per draw, applied to BOTH arms and every "
                     "subgroup, so all differences below are paired",
           "note": "severity strata are scored against the shared pool of all negatives, "
                   "because the FairVision label is defined by md and within-bin AUC is undefined",
           "contrasts": {}}

    for b_arm, a_arm in CONTRASTS:
        key = "%s_minus_%s" % (b_arm, a_arm)
        keys = sorted(masks)
        point = {}
        for gk in keys:
            m = masks[gk]
            point[gk] = fast_auc(y[m], S[b_arm][m]) - fast_auc(y[m], S[a_arm][m])

        boot = {gk: np.empty(N_BOOT) for gk in keys}
        for i in range(N_BOOT):
            idx = np.concatenate([RNG.choice(pos_idx, pos_idx.size, True),
                                  RNG.choice(neg_idx, neg_idx.size, True)])
            yy = y[idx]
            for gk in keys:
                mm = masks[gk][idx]
                yv = yy[mm]
                if yv.sum() < 2 or (yv == 0).sum() < 2:
                    boot[gk][i] = np.nan
                    continue
                boot[gk][i] = (fast_auc(yv, S[b_arm][idx][mm])
                               - fast_auc(yv, S[a_arm][idx][mm]))
            if (i + 1) % 2500 == 0:
                print("  %s bootstrap %d/%d" % (key, i + 1, N_BOOT), flush=True)

        entry = {"per_group": {}}
        for gk in keys:
            v = boot[gk][~np.isnan(boot[gk])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            entry["per_group"]["%s:%s" % gk] = {
                "delta_auc": float(point[gk]),
                "ci95_lo": float(lo), "ci95_hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n": int(masks[gk].sum()),
            }

        # disparity change per attribute, and differential benefit
        for gname in ("race", "sex"):
            gk = [k for k in keys if k[0] == gname]
            if len(gk) < 2:
                continue
            aucs_a = {k: fast_auc(y[masks[k]], S[a_arm][masks[k]]) for k in gk}
            worst = min(aucs_a, key=aucs_a.get)
            best = max(aucs_a, key=aucs_a.get)
            dboot = boot[worst] - boot[best]
            dboot = dboot[~np.isnan(dboot)]
            lo, hi = np.percentile(dboot, [2.5, 97.5])
            entry["%s_differential_benefit" % gname] = {
                "worst_group_under_%s" % a_arm: "%s:%s" % worst,
                "best_group_under_%s" % a_arm: "%s:%s" % best,
                "delta_worst_minus_delta_best": float(point[worst] - point[best]),
                "ci95_lo": float(lo), "ci95_hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "reading": "positive means the worst-served group gained MORE than the "
                           "best-served one; an interval containing zero means the "
                           "differential benefit is not resolved",
            }
        out["contrasts"][key] = entry

    with open(os.path.join(OUT, "p7c_paired_subgroup.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    for key, e in out["contrasts"].items():
        print("\n=== %s (matched epoch 100) ===" % key)
        print("%-22s %10s %-22s %s" % ("subgroup", "dAUC", "95% CI", "excl 0"))
        for gk, v in sorted(e["per_group"].items()):
            print("%-22s %+10.5f [%+.5f,%+.5f] %s" % (
                gk, v["delta_auc"], v["ci95_lo"], v["ci95_hi"],
                "yes" if v["excludes_zero"] else "no"))
        for gname in ("race", "sex"):
            k = "%s_differential_benefit" % gname
            if k in e:
                d = e[k]
                print("  %s differential: %+.5f [%+.5f,%+.5f] excl0=%s" % (
                    gname, d["delta_worst_minus_delta_best"], d["ci95_lo"],
                    d["ci95_hi"], "yes" if d["excludes_zero"] else "no"))
    print("\nwrote", os.path.join(OUT, "p7c_paired_subgroup.json"))


if __name__ == "__main__":
    main()
