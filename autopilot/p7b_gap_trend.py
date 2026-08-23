"""P7b: does improving the pretraining objective close or widen subgroup gaps?

Consumes the verified output of paper/genai4health2026/scripts/subgroup_analysis.py
(which independently reproduces the AUCs computed in p1c_stats.py) and asks one
question across every attribute:

    as aggregate AUC improves across masking policies, does the max-min
    subgroup gap shrink, stay flat, or widen?

This is the trustworthy-AI claim the workshop cares about, and it is answerable
without any GPU because it reuses saved predictions.

Retracted probes are excluded from the correlation; they are reported separately.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p7b_gap_trend.json
"""
import json
import os
import re
import numpy as np
from scipy import stats as sps

SUB = r"D:\jepa_phase0\autopilot_out\subgroup\subgroup_auc.json"
OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
INV = os.path.join(OUT, "p1b_full_inventory.json")

ATTRS = ["gender", "race", "ethnicity", "language", "maritalstatus", "age", "severity"]


def excluded_probe_dirs():
    """Probe directories the evidence inventory marks retracted or excluded.

    subgroup_analysis.py only tags the `frozen_cover_random_*` family as
    RETRACTED; it does not know about the anatomy-v2 epoch-75/92 precision
    splice. Filtering on its status alone silently re-admitted two probes the
    manuscript says are excluded, which inflated the probe count to 21 and
    flipped the severity trend from non-significant to significant. The
    inventory is the single source of truth for exclusion.
    """
    inv = json.load(open(INV))
    bad = set()
    for r in inv["records"]:
        if r["status"] in ("retracted", "excluded"):
            bad.add(os.path.basename(os.path.dirname(r["path"])))
            bad.add(r["tag"])
    return bad


def main():
    data = json.load(open(SUB))
    probes = data if isinstance(data, dict) else {}
    excluded = excluded_probe_dirs()

    rows = []
    dropped = []
    for name, v in probes.items():
        if not isinstance(v, dict) or "overall_auc" not in v:
            continue
        if str(v.get("status", "OK")).upper() == "RETRACTED":
            dropped.append((name, "retracted"))
            continue
        if name in excluded:
            dropped.append((name, "excluded by inventory"))
            continue
        r = {"probe": name, "overall_auc": v["overall_auc"], "status": v.get("status", "OK")}
        for a in ATTRS:
            s = v.get("subgroups", {}).get(a, {})
            r[a + "_gap"] = s.get("gap")
            r[a + "_worst"] = s.get("worst_group")
            r[a + "_worst_auc"] = s.get("worst")
        rows.append(r)

    rows.sort(key=lambda r: r["overall_auc"])
    out = {"generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
           "n_probes": len(rows), "source": SUB,
           "excluded_probes": dropped,
           "note": "retracted AND inventory-excluded probes removed; exclusion status is "
                   "taken from p1b_full_inventory.json, not from subgroup_analysis.py",
           "independence_caveat": "these probes share one test set, one epoch-25 ancestor and "
                                  "one probe seed, and several are checkpoints of the same arm. "
                                  "They are NOT independent, so the Spearman p-values below are "
                                  "not calibrated and must be reported as descriptive.",
           "epoch_precision_caveat": "rows span pretraining epochs 25-100 and both fp16 and fp32 "
                                     "probe precision; the trend is across checkpoints, not a "
                                     "matched-epoch arm comparison.",
           "trends": {}, "worst_group_consistency": {}, "rows": rows}

    print("n probes retained before dedup: %d   (dropped %d)" % (len(rows), len(dropped)))
    for n, why in dropped:
        print("   dropped %-34s %s" % (n, why))

    # ---- branch (arm) identity, for the pseudo-replication correction -------
    # Several rows are different epochs of ONE training branch. Treating them as
    # independent points inflates n from 7 branches to 19 checkpoints. We report
    # both, and treat the branch-level figure as the honest one.
    #
    # Order matters: the more specific keys must be tested first, because
    # e.g. "frozen_meanpool_oracle_ep100_fp32" contains "oracle" but is not a
    # "sweep_oracle" directory.
    def branch_of(name):
        n = name.lower()
        for key, arm in (("cover_f021", "cover-f021"), ("bridge", "anatomy-v2"),
                         ("blob_fp32", "anatomy-v2"), ("anatomy", "anatomy-v1"),
                         ("fork", "ancestor"), ("envelope", "envelope"),
                         ("mirage", "envelope"), ("oracle", "oracle"),
                         ("random", "random")):
            if key in n:
                return arm
        return n

    def epoch_of(name):
        m = re.search(r"ep(\d+)", name.lower())
        return int(m.group(1)) if m else -1

    for r in rows:
        r["branch"] = branch_of(r["probe"])
        r["epoch"] = epoch_of(r["probe"])

    # ---- collapse technical duplicates -------------------------------------
    # An fp32 re-probe and its fp16 original are the SAME frozen encoder scored
    # twice at different probe precision. Counting both inflates n, invents
    # spurious "branches", and shifts every correlation. Keep exactly one probe
    # per (branch, epoch), preferring the fp16 original because that is the run
    # the manuscript reports.
    best = {}
    dup = []
    for r in rows:
        k = (r["branch"], r["epoch"])
        if k not in best:
            best[k] = r
        else:
            keep, drop = (best[k], r) if "fp32" not in best[k]["probe"] else (r, best[k])
            best[k] = keep
            dup.append((drop["probe"], "same encoder as %s" % keep["probe"]))
    if dup:
        rows = sorted(best.values(), key=lambda r: r["overall_auc"])
        out["n_probes"] = len(rows)
        out["rows"] = rows
        out["collapsed_duplicates"] = dup
        print("\ncollapsed %d technical duplicates (same encoder, different probe precision):" % len(dup))
        for a, why in dup:
            print("   %-38s %s" % (a, why))

    # AUC vector must be built AFTER the dedup, or the boolean masks below
    # index a stale array.
    auc = np.array([r["overall_auc"] for r in rows])
    print("\nn probes used: %d   branches: %d"
          % (len(rows), len(set(r["branch"] for r in rows))))

    print("\n%-14s %8s %8s %8s   %-24s %s" %
          ("attribute", "rho", "p", "q(BH7)", "worst group", "gap range"))
    print("-" * 104)
    for a in ATTRS:
        g = np.array([r[a + "_gap"] if r[a + "_gap"] is not None else np.nan for r in rows])
        m = ~np.isnan(g)
        if m.sum() < 4:
            continue
        rho = sps.spearmanr(auc[m], g[m])

        # branch-level: average each branch's gap and AUC, then correlate.
        # This removes the pseudo-replication from probing one branch at
        # several epochs.
        bg, ba = {}, {}
        for r in rows:
            if r[a + "_gap"] is None:
                continue
            bg.setdefault(r["branch"], []).append(r[a + "_gap"])
            ba.setdefault(r["branch"], []).append(r["overall_auc"])
        bx = np.array([np.mean(ba[k]) for k in sorted(bg)])
        by = np.array([np.mean(bg[k]) for k in sorted(bg)])
        brho = sps.spearmanr(bx, by) if len(bx) >= 4 else None

        out["trends"][a] = {"n_checkpoints": int(m.sum()),
                            "spearman_rho": float(rho.statistic),
                            "spearman_p": float(rho.pvalue),
                            "n_branches": int(len(bx)),
                            "branch_spearman_rho": float(brho.statistic) if brho else None,
                            "branch_spearman_p": float(brho.pvalue) if brho else None,
                            "gap_min": float(np.nanmin(g[m])), "gap_max": float(np.nanmax(g[m])),
                            "gap_mean": float(np.nanmean(g[m]))}
        worst = [r[a + "_worst"] for r in rows if r[a + "_worst"]]
        from collections import Counter
        cnt = dict(Counter(worst))
        out["worst_group_consistency"][a] = {"counts": cnt, "n": len(worst),
                                             "unanimous": len(cnt) == 1}

    # ---- Benjamini-Hochberg across the seven attributes tested -------------
    # Seven gap-vs-AUC tests are run; quoting one uncorrected p-value would be
    # selective reporting.
    names = [a for a in ATTRS if a in out["trends"]]
    ps = np.array([out["trends"][a]["spearman_p"] for a in names])
    order = np.argsort(ps)
    n = len(ps)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        k = order[rank]
        prev = min(prev, ps[k] * n / (rank + 1))
        q[k] = prev
    for a, qq in zip(names, q):
        out["trends"][a]["q_bh_across_attributes"] = float(qq)
        t = out["trends"][a]
        t["direction"] = ("widens" if t["spearman_rho"] > 0 else "narrows") \
            if qq < 0.05 else "no trend surviving correction"

    for a in names:
        t = out["trends"][a]
        c = out["worst_group_consistency"].get(a, {})
        cnt = c.get("counts", {})
        top = max(cnt.items(), key=lambda kv: kv[1]) if cnt else ("-", 0)
        print("%-14s %+8.3f %8.4f %8.4f   %-16s %2d/%-2d  [%.4f, %.4f]" % (
            a, t["spearman_rho"], t["spearman_p"], t["q_bh_across_attributes"],
            top[0][:16], top[1], c.get("n", 0), t["gap_min"], t["gap_max"]))

    print("\nbranch-level (pseudo-replication removed, n=%d branches):" %
          out["trends"][names[0]]["n_branches"])
    for a in names:
        t = out["trends"][a]
        print("   %-14s rho=%+.3f  p=%.4f" % (a, t["branch_spearman_rho"], t["branch_spearman_p"]))

    with open(os.path.join(OUT, "p7b_gap_trend.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", os.path.join(OUT, "p7b_gap_trend.json"))


if __name__ == "__main__":
    main()
