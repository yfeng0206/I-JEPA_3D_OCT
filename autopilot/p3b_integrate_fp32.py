"""P3b: integrate completed fp32 re-probes into the evidence base.

Idempotent. Run after any subset of the GPU queue completes; it uses whatever
fp32 probes exist and reports what is still outstanding.

Produces, for each arm and epoch where BOTH precisions exist:
  - the fp16 AUC, the fp32 AUC, and their paired difference with a DeLong p
  - a fully fp32 replication of the primary contrasts

That converts the precision mismatch from an undisclosed confound into a
measured robustness result.

Outputs
  D:/jepa_phase0/autopilot_out/p1_stats/p3b_fp32.json
  paper/genai4health2026/auto/table_fp32.tex
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_paired_stats import delong_test  # noqa: E402

RUNS = r"D:\jepa_phase0\runs"
STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
AUTO = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026\auto"
REPO = r"C:\Users\Gary\Desktop\jepa"
N_BOOT = 10000
RNG = np.random.default_rng(20260822)

# fp32 re-probe dir -> (arm, epoch)
FP32 = {
    "frozen_meanpool_envelope_fp32_ep50": ("envelope", 50),
    "frozen_meanpool_envelope_fp32_ep75": ("envelope", 75),
    "frozen_meanpool_envelope_fp32_ep100": ("envelope", 100),
    "frozen_meanpool_oracle_ep50_fp32": ("oracle", 50),
    "frozen_meanpool_oracle_ep75_fp32": ("oracle", 75),
    "frozen_meanpool_oracle_ep100_fp32": ("oracle", 100),
    "frozen_meanpool_random_ep50_fp32": ("random", 50),
    "frozen_meanpool_random_ep75_fp32": ("random", 75),
    "frozen_meanpool_random_ep100_fp32": ("random", 100),
}
FP16 = {("random", ep): r"results\downstream\meanpool_sweep_random\ep%d_test_predictions.npz" % ep
        for ep in (50, 75, 100)}
FP16.update({("oracle", ep): r"results\downstream\meanpool_sweep_oracle\ep%d_test_predictions.npz" % ep
             for ep in (50, 75, 100)})
FP16.update({("envelope", ep): r"results\downstream\meanpool_sweep_mirage\ep%d_test_predictions.npz" % ep
             for ep in (50, 75, 100)})


def fast_auc(y, s):
    r = sps.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    have32, missing = {}, []
    for d, (arm, ep) in FP32.items():
        p = os.path.join(RUNS, d, "test_predictions.npz")
        g = os.path.join(r"D:\jepa_phase0\autopilot_out\probe_guards",
                         "guard_%s.json" % d.replace("frozen_", ""))
        if os.path.exists(p):
            guard = json.load(open(g)) if os.path.exists(g) else {}
            if guard and not guard.get("encoder_unchanged", True):
                print("[INVALID] %s: encoder hash changed, refusing to use" % d)
                continue
            have32[(arm, ep)] = (p, guard)
        else:
            missing.append(d)

    print("fp32 probes available: %d / %d" % (len(have32), len(FP32)))
    for m in missing:
        print("   still pending:", m)
    if not have32:
        print("nothing to integrate yet")
        return

    y = None
    rows = []
    for (arm, ep), (p, guard) in sorted(have32.items()):
        z = np.load(p)
        lab = z["labels"].astype(int)
        pr32 = z["probs"].astype(np.float64)
        if y is None:
            y = lab
        assert np.array_equal(lab, y), "test split mismatch in %s" % p
        a32 = float(fast_auc(y, pr32))

        f16p = os.path.join(REPO, FP16[(arm, ep)])
        row = {"arm": arm, "epoch": ep, "auc_fp32": a32,
               "encoder_sha256": guard.get("sha256_before"),
               "encoder_unchanged": guard.get("encoder_unchanged"),
               "elapsed_min": round(guard.get("elapsed_sec", 0) / 60.0, 1)}
        if os.path.exists(f16p):
            pr16 = np.load(f16p)["probs"].astype(np.float64)
            a16 = float(fast_auc(y, pr16))
            d, se, pv, ci = delong_test(y, pr32, pr16)
            row.update({"auc_fp16": a16, "delta_fp32_minus_fp16": d,
                        "delong_p": pv, "delong_ci95": list(ci)})
        rows.append(row)

    # fully-fp32 replication of the primary contrasts
    contrasts = []
    S32 = {}
    for (arm, ep), (p, _) in have32.items():
        S32[(arm, ep)] = np.load(p)["probs"].astype(np.float64)
    for ep in (50, 75, 100):
        for a, b in (("envelope", "random"), ("oracle", "random"), ("oracle", "envelope")):
            if (a, ep) in S32 and (b, ep) in S32:
                d, se, pv, ci = delong_test(y, S32[(a, ep)], S32[(b, ep)])
                pos = np.flatnonzero(y == 1)
                neg = np.flatnonzero(y == 0)
                db = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    idx = np.concatenate([RNG.choice(pos, pos.size, True),
                                          RNG.choice(neg, neg.size, True)])
                    yy = y[idx]
                    db[i] = fast_auc(yy, S32[(a, ep)][idx]) - fast_auc(yy, S32[(b, ep)][idx])
                contrasts.append({"a": a, "b": b, "epoch": ep, "precision": "fp32",
                                  "delta": d, "delong_p": pv,
                                  "boot_ci95_lo": float(np.percentile(db, 2.5)),
                                  "boot_ci95_hi": float(np.percentile(db, 97.5))})

    out = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "n_available": len(have32), "n_expected": len(FP32), "pending": missing,
           "rows": rows, "fp32_contrasts": contrasts}
    with open(os.path.join(STATS, "p3b_fp32.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # ---- LaTeX table
    tl = [r"\begin{tabular}{lccccc}", r"\toprule",
          r"policy & epoch & fp16 AUC & fp32 AUC & $\Delta$ & $p$ \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["epoch"])):
        if "auc_fp16" in r:
            tl.append("\\textsc{%s} & %d & %.6f & %.6f & %+.6f & %.3f \\\\" % (
                r["arm"], r["epoch"], r["auc_fp16"], r["auc_fp32"],
                r["delta_fp32_minus_fp16"], r["delong_p"]))
        else:
            tl.append("\\textsc{%s} & %d & --- & %.6f & --- & --- \\\\" % (
                r["arm"], r["epoch"], r["auc_fp32"]))
    tl += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(AUTO, "table_fp32.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tl) + "\n")

    print("\n%-10s %-5s %-11s %-11s %-11s %s" % ("arm", "ep", "fp16", "fp32", "delta", "p"))
    for r in sorted(rows, key=lambda r: (r["arm"], r["epoch"])):
        if "auc_fp16" in r:
            print("%-10s %-5d %-11.6f %-11.6f %+.6f  %.4f" % (
                r["arm"], r["epoch"], r["auc_fp16"], r["auc_fp32"],
                r["delta_fp32_minus_fp16"], r["delong_p"]))
        else:
            print("%-10s %-5d %-11s %-11.6f" % (r["arm"], r["epoch"], "-", r["auc_fp32"]))
    if contrasts:
        print("\nfully-fp32 replication of primary contrasts:")
        for c in contrasts:
            print("  ep%-4d %-9s - %-9s d=%+.4f p=%.5f CI[%+.4f,%+.4f]" % (
                c["epoch"], c["a"], c["b"], c["delta"], c["delong_p"],
                c["boot_ci95_lo"], c["boot_ci95_hi"]))
    print("\nwrote", os.path.join(STATS, "p3b_fp32.json"))


if __name__ == "__main__":
    main()
