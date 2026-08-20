"""Figure: mask policy vs. subgroup disparity.

Left  - per-arm racial AUC gap (max-min across race groups), retracted arms excluded.
Right - severity-stratified detection AUC, showing the mask-policy-invariant
        mild-disease penalty.

Reads D:/jepa_phase0/reports/subgroup/subgroup_auc.json (produced by
subgroup_analysis.py) and writes PDF+PNG next to the other paper figures.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = r"D:\jepa_phase0\reports\subgroup\subgroup_auc.json"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")

# Map probe dir -> (display name, pretraining family). Probes of the same
# family share ONE pretraining run, so they are not independent replicates.
FAMILY = {
    "frozen_meanpool_fork_ep25": ("fork ep25", "shared fork"),
    "frozen_meanpool_envelope_ep30": ("envelope ep30", "envelope"),
    "frozen_meanpool_mirage_ep50": ("envelope ep50", "envelope"),
    "frozen_meanpool_mirage_ep75": ("envelope ep75", "envelope"),
    "frozen_meanpool_mirage_ep100": ("envelope ep100", "envelope"),
    "frozen_meanpool_anatomy_ep30": ("anatomy ep30", "anatomy"),
    "frozen_meanpool_bridge_ep35": ("blob ep35", "blob"),
    "frozen_meanpool_bridge_ep40": ("blob ep40", "blob"),
    "frozen_meanpool_bridge_ep50": ("blob ep50", "blob"),
    "frozen_meanpool_cover_f021_ep27": ("COVER ep27", "COVER f0.21"),
    "frozen_meanpool_cover_f021_ep30": ("COVER ep30", "COVER f0.21"),
    "frozen_meanpool_cover_f021_ep34": ("COVER ep34", "COVER f0.21"),
    "sweep_random_ep50": ("random ep50", "random"),
    "sweep_random_ep75": ("random ep75", "random"),
    "sweep_random_ep100": ("random ep100", "random"),
    "sweep_oracle_ep50": ("oracle ep50", "oracle"),
    "sweep_oracle_ep75": ("oracle ep75", "oracle"),
    "sweep_oracle_ep100": ("oracle ep100", "oracle"),
}
COLORS = {"shared fork": "#7f7f7f", "envelope": "#1f77b4", "anatomy": "#2ca02c",
          "blob": "#d62728", "COVER f0.21": "#ff7f0e", "random": "#9467bd",
          "oracle": "#8c564b"}


def main():
    data = json.load(open(SRC))
    arms = [(k, v) for k, v in data.items()
            if "error" not in v and v.get("status") == "OK" and k in FAMILY]
    arms.sort(key=lambda kv: kv[1]["overall_auc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---- left: racial gap per arm -----------------------------------------
    names = [FAMILY[k][0] for k, _ in arms]
    fams = [FAMILY[k][1] for k, _ in arms]
    gaps = [v["subgroups"]["race"]["gap"] for _, v in arms]
    y = np.arange(len(arms))
    ax1.barh(y, gaps, color=[COLORS[f] for f in fams], edgecolor="black",
             linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Racial AUC gap (max $-$ min across race groups)")
    # Derive the count from the data so the annotation can never drift out of
    # sync with the underlying report again (it previously said 12/12 while the
    # paper said 18/18, and wrongly claimed no gap was significant).
    n_black_worst = sum(1 for _, v in arms
                        if v["subgroups"]["race"]["worst_group"] == "black")
    ax1.set_title("(a) Observed racial AUC gaps across saved probes\n"
                  "(Black lowest point estimate in %d/%d; largest gap "
                  "(blob ep50) significant after adjustment)"
                  % (n_black_worst, len(arms)),
                  fontsize=9)
    for yi, g in zip(y, gaps):
        ax1.text(g + 0.001, yi, "%.3f" % g, va="center", fontsize=7)
    ax1.set_xlim(0, max(gaps) * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[f], ec="black", lw=0.5)
               for f in COLORS]
    ax1.legend(handles, list(COLORS), fontsize=7, loc="lower right",
               title="pretraining run", title_fontsize=7)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)

    # ---- right: severity-stratified detection ------------------------------
    order = ["severe (<=-12)", "moderate (-12,-6]", "mild (-6,-2]"]
    for k, v in arms:
        sev = {e["subgroup"]: e for e in v["subgroups"]["severity"]["levels"]}
        vals = [sev[s]["auc"] for s in order if s in sev]
        if len(vals) != len(order):
            continue
        ax2.plot(range(len(order)), vals, marker="o", ms=4, lw=1.2,
                 color=COLORS[FAMILY[k][1]], alpha=0.85)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(["severe\n(MD$\\leq-12$)", "moderate\n($-12<$MD$\\leq-6$)",
                         "mild\n($-6<$MD$\\leq-2$)"], fontsize=8)
    ax2.set_ylabel("AUC (stratum positives vs. all negatives)")
    ax2.set_title("(b) Severity-gap estimates are similar across tested "
                  "probes\n(gap $\\approx$0.13 in every arm)", fontsize=9)
    ax2.grid(alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "fig6_subgroup_disparity." + ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
