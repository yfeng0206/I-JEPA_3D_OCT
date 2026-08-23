"""Independent validation of the DeLong implementation used in p1_paired_stats.py.

A wrong variance would silently produce wrong p-values throughout the paper, so
the estimator is checked four ways before any of its output is quoted:

  1. AUC agreement: DeLong's internal AUC vs sklearn.roc_auc_score
  2. Variance agreement: DeLong SE vs the empirical SD of a paired bootstrap
  3. Null calibration: on synthetic data where two scores have EQUAL true AUC,
     the p-value distribution must be approximately uniform and the type-I error
     rate at alpha=0.05 must be near 0.05
  4. Self-comparison: DeLong of a score against itself must give delta=0

Output -> D:/jepa_phase0/autopilot_out/p1_stats/delong_validation.json
"""
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_paired_stats import delong_cov, delong_test  # noqa: E402

OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
rng = np.random.default_rng(7)
report = {}

# ---------------------------------------------------------------- 1 & 4
stats = json.load(open(os.path.join(OUT, "p1_master_stats.json")))
runs = {}
for t in stats["table"]:
    p = os.path.join(r"D:\jepa_phase0\runs", t["probe_dir"], "test_predictions.npz")
    z = np.load(p)
    runs[t["probe_dir"]] = (z["labels"].astype(int), z["probs"].astype(float))

labels = list(runs.values())[0][0]
max_auc_err = 0.0
for k, (y, p) in runs.items():
    a_dl, _ = delong_cov(y, [p])
    a_sk = roc_auc_score(y, p)
    max_auc_err = max(max_auc_err, abs(float(a_dl[0]) - a_sk))
report["1_auc_agreement_max_abs_error_vs_sklearn"] = float(max_auc_err)
report["1_pass"] = bool(max_auc_err < 1e-9)

k0 = list(runs)[0]
d, se, p, ci = delong_test(labels, runs[k0][1], runs[k0][1])
report["4_self_comparison_delta"] = float(d)
report["4_pass"] = bool(abs(d) < 1e-12)

# ---------------------------------------------------------------- 2
# DeLong SE of the DIFFERENCE vs empirical SD of the paired bootstrap difference.
boot = np.load(os.path.join(OUT, "bootstrap_draws.npz"))
checks = []
pairs = [("frozen_cover_random_ep100", "frozen_meanpool_mirage_ep100"),
         ("frozen_cover_random_ep50", "frozen_meanpool_bridge_ep50"),
         ("frozen_meanpool_cover_f021_ep50", "frozen_meanpool_mirage_ep50"),
         ("frozen_meanpool_fork_ep25", "frozen_meanpool_cover_f021_ep27")]
for a, b in pairs:
    d, se, pv, ci = delong_test(labels, runs[a][1], runs[b][1])
    emp = float((boot[a] - boot[b]).std(ddof=1))
    checks.append({"a": a, "b": b, "delta": float(d), "delong_se": float(se),
                   "bootstrap_sd": emp,
                   "ratio_delong_over_bootstrap": float(se / emp) if emp else None})
report["2_se_vs_bootstrap"] = checks
report["2_pass"] = all(0.85 <= c["ratio_delong_over_bootstrap"] <= 1.15 for c in checks)

# ---------------------------------------------------------------- 3
# Null calibration: two correlated scores with IDENTICAL true signal.
n_sim, n, alpha = 600, 1500, 0.05
ps = np.empty(n_sim)
for s in range(n_sim):
    y = rng.integers(0, 2, n)
    sig = y + rng.normal(0, 1.0, n)              # shared signal
    s1 = sig + rng.normal(0, 0.55, n)            # equal-strength independent noise
    s2 = sig + rng.normal(0, 0.55, n)
    ps[s] = delong_test(y, s1, s2)[2]
type1 = float((ps < alpha).mean())
ks = sps.kstest(ps, "uniform")
report["3_null_calibration"] = {
    "n_simulations": n_sim, "n_per_sim": n, "alpha": alpha,
    "empirical_type1_error": type1,
    "ks_uniform_stat": float(ks.statistic), "ks_uniform_p": float(ks.pvalue),
    "note": "p-values under a true null must be ~Uniform(0,1); type-I error must be ~alpha",
}
# binomial 95% band for the type-I rate
lo, hi = sps.binom.interval(0.95, n_sim, alpha)
report["3_type1_95pct_band"] = [float(lo / n_sim), float(hi / n_sim)]
report["3_pass"] = bool(lo / n_sim <= type1 <= hi / n_sim and ks.pvalue > 0.01)

report["ALL_PASS"] = all(report[k] for k in ("1_pass", "2_pass", "3_pass", "4_pass"))

with open(os.path.join(OUT, "delong_validation.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

print(json.dumps(report, indent=1))
print("\nALL_PASS =", report["ALL_PASS"])
