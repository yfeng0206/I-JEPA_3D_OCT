"""Derive every intersectional claim in the paper from the two subgroup CSVs.

Reads the outputs of intersectional_analysis.py and prints each claim with the
count that supports it, so no number in the manuscript is asserted by hand.
Retracted arms (fp16 + window contamination) are excluded from every count.
"""
import csv
import os
from collections import defaultdict

REPORTS = os.environ.get("SUBGROUP_DIR", r"D:\jepa_phase0\reports\subgroup")
INTER = os.path.join(REPORTS, "intersectional_auc.csv")
MARG = os.path.join(REPORTS, "subgroup_auc.csv")


def load(path, key):
    rows = defaultdict(dict)
    overall = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                continue
            rows[r["arm"]][r[key]] = float(r["auc"])
            overall[r["arm"]] = float(r["overall_auc"])
    return rows, overall


inter, overall = load(INTER, "subgroup")
marg, _ = load(MARG, "subgroup")

arms = sorted(inter)
n = len(arms)
print(f"non-retracted arms: {n}\n")

# --- orderings -----------------------------------------------------------
worst_bf = sum(min(inter[a], key=inter[a].get) == "black x female" for a in arms)
best_am = sum(max(inter[a], key=inter[a].get) == "asian x male" for a in arms)
print(f"black x female is the WORST cell         : {worst_bf}/{n}")
print(f"asian x male   is the BEST  cell         : {best_am}/{n}")

fem = sum(
    inter[a][f"{r} x female"] < inter[a][f"{r} x male"]
    for a in arms
    for r in ("asian", "black", "white")
)
print(f"female below male within race            : {fem}/{n * 3}")

blk = sum(
    inter[a][f"black x {g}"] < inter[a][f"white x {g}"]
    for a in arms
    for g in ("female", "male")
)
print(f"black below white within gender          : {blk}/{n * 2}")


# --- gap magnitudes ------------------------------------------------------
def spread(d, keys=None):
    v = [d[k] for k in (keys or d)]
    return max(v) - min(v)


g_inter = {a: spread(inter[a]) for a in arms}
g_race = {a: spread(marg[a], ["asian", "black", "white"]) for a in arms}
g_gender = {a: spread(marg[a], ["female", "male"]) for a in arms}

mi = sum(g_inter.values()) / n
mr = sum(g_race.values()) / n
mg = sum(g_gender.values()) / n
print(f"\nmean gap  gender {mg:.4f} | race {mr:.4f} | race x gender {mi:.4f}")
print(f"intersectional exceeds marginal race     : "
      f"{sum(g_inter[a] > g_race[a] for a in arms)}/{n}")
print(f"understatement of worst-cell disadvantage: {(mi - mr) / mr * 100:.1f}%")
print(f"additive prediction {mr + mg:.4f} vs observed {mi:.4f} "
      f"(ratio {mi / (mr + mg):.3f})")

# --- accuracy / equity tension ------------------------------------------
by_auc = sorted(arms, key=lambda a: -overall[a])
best = by_auc[0]
tight = min(arms, key=lambda a: g_inter[a])
print(f"\nbest overall AUC : {best} {overall[best]:.4f}, "
      f"but black x female {inter[best]['black x female']:.4f}")
print(f"smallest gap     : {tight} gap {g_inter[tight]:.4f}, "
      f"overall AUC {overall[tight]:.4f} ranks "
      f"{by_auc.index(tight) + 1}/{n}")

# --- does any arm lift the worst cell above the best arm's worst cell? ---
bf = {a: inter[a]["black x female"] for a in arms}
top_bf = max(bf, key=bf.get)
print(f"highest black x female anywhere: {top_bf} {bf[top_bf]:.4f} "
      f"(vs best-overall arm {inter[best]['black x female']:.4f})")
