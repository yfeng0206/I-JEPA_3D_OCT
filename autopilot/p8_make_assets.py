"""P8: generate LaTeX tables, macros and figures from verified artifacts only.

Design rule: the manuscript must never contain a hand-typed number. Every
quantity is emitted here as a \newcommand macro, so prose, tables and figures
are guaranteed to agree and a re-run propagates corrections everywhere.

Inputs  (all produced and checked earlier in this run)
  p1b_full_inventory.json  - labelled, de-duplicated evidence inventory
  p1c_stats.json           - AUCs, bootstrap CIs, DeLong contrasts by family
  p7_fairness.json         - subgroup AUCs and gaps
  p7_gap_correlation.json  - AUC-vs-gap correlations

Outputs -> paper/genai4health2026/auto/
  auto_numbers.tex, table_main.tex, table_fairness.tex, table_allprobes.tex
  fig_trajectories_ci.png, fig_fairness.png, fig_roc.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
AUTO = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026\auto"
os.makedirs(AUTO, exist_ok=True)

ARM_TEX = {"random": r"\textsc{random}", "oracle": r"\textsc{intensity}",
           "envelope": r"\textsc{envelope}", "anatomy-v1": r"\textsc{anatomy-v1}",
           "anatomy-v2": r"\textsc{anatomy-v2}", "cover-f021": r"\textsc{cover}",
           "ancestor": r"ancestor"}
# Display label for figures. The internal key stays "oracle" because that is the
# name in the stored artifacts and released checkpoints; only the paper renames.
ARM_PLOT = {"random": "random", "oracle": "intensity", "envelope": "envelope",
            "anatomy-v1": "anatomy-v1", "anatomy-v2": "anatomy-v2",
            "cover-f021": "cover-f021", "ancestor": "ancestor"}
COL = {"random": "#4c4c4c", "oracle": "#c1272d", "envelope": "#1f77b4",
       "anatomy-v1": "#9467bd", "anatomy-v2": "#8c564b", "cover-f021": "#2ca02c",
       "ancestor": "#999999"}


def num(x, d=4):
    return ("%." + str(d) + "f") % x


def signed(x, d=4):
    return ("%+." + str(d) + "f") % x


def pfmt(p):
    if p < 1e-4:
        return "$<$0.0001"
    return "%.4f" % p


def main():
    inv = json.load(open(os.path.join(STATS, "p1b_full_inventory.json")))
    st = json.load(open(os.path.join(STATS, "p1c_stats.json")))
    fair = json.load(open(os.path.join(STATS, "p7_fairness.json")))
    gcorr = json.load(open(os.path.join(STATS, "p7_gap_correlation.json")))

    T = {t["key"]: t for t in st["table"]}
    C = {(c["a"], c["b"]): c for c in st["contrasts"]}

    def contrast(a, b):
        return C.get((a, b)) or C.get((b, a))

    def delta(a, b):
        """AUC(a) - AUC(b) with the correct sign regardless of stored order."""
        c = contrast(a, b)
        if c is None:
            return None
        if c["a"] == a:
            return c["delta_a_minus_b"], c["boot_ci95_lo"], c["boot_ci95_hi"], c["delong_p"], c
        return (-c["delta_a_minus_b"], -c["boot_ci95_hi"], -c["boot_ci95_lo"], c["delong_p"], c)

    M = {}           # name -> value; a dict so a later definition simply wins
                     # rather than emitting a duplicate \newcommand, which is a
                     # hard LaTeX error

    def mac(name, val):
        M[name] = val

    # ---------------------------------------------------------- macros
    mac("Ntest", "%d" % st["n_test"])
    mac("Npos", "%d" % st["n_pos"])
    mac("Nneg", "%d" % (st["n_test"] - st["n_pos"]))
    mac("Nboot", "{:,}".format(st["n_bootstrap"]).replace(",", "{,}"))
    mac("Nprobes", "%d" % len(st["table"]))

    # TeX control sequences may contain letters only, so epochs are spelled out.
    EPW = {50: "EpFifty", 75: "EpSeventyFive", 100: "EpHundred"}
    ARMW = {"random": "Random", "oracle": "Oracle", "envelope": "Envelope"}

    for ep in (50, 75, 100):
        for arm in ("random", "oracle", "envelope"):
            k = "%s@ep%d@fp16" % (arm, ep)
            if k in T:
                mac("AUC%s%s" % (ARMW[arm], EPW[ep]), num(T[k]["auc"]))
                mac("CIlo%s%s" % (ARMW[arm], EPW[ep]), num(T[k]["ci95_lo"]))
                mac("CIhi%s%s" % (ARMW[arm], EPW[ep]), num(T[k]["ci95_hi"]))
        for a, b, tag in (("oracle", "random", "OracleRandom"),
                          ("envelope", "random", "EnvelopeRandom"),
                          ("oracle", "envelope", "OracleEnvelope")):
            r = delta("%s@ep%d@fp16" % (a, ep), "%s@ep%d@fp16" % (b, ep))
            if r:
                d, lo, hi, p, c = r
                mac("D%s%s" % (tag, EPW[ep]), signed(d))
                mac("D%s%sCI" % (tag, EPW[ep]), "[%s,\\,%s]" % (signed(lo), signed(hi)))
                mac("D%s%sP" % (tag, EPW[ep]), pfmt(p))
                if "delong_q_bh" in c:
                    mac("D%s%sQ" % (tag, EPW[ep]), pfmt(c["delong_q_bh"]))

    # fp32 arms at matched epoch 50. Prefer an fp32 null when one exists, so the
    # contrast is precision-matched; fall back to the fp16 null otherwise and let
    # the manuscript flag it.
    null50 = "random@ep50@fp32" if "random@ep50@fp32" in T else "random@ep50@fp16"
    mac("NullEpFiftyPrecision", "fp32" if null50.endswith("fp32") else "fp16")
    mac("HTwoMatched", "yes" if null50.endswith("fp32") else "no")
    if "random@ep50@fp32" in T:
        mac("AUCRandomEpFiftyFPthirtytwo", num(T["random@ep50@fp32"]["auc"]))
    for arm, tag in (("anatomy-v2", "AnatomyTwo"), ("cover-f021", "Cover")):
        k = "%s@ep50@fp32" % arm
        if k in T:
            mac("AUC%sEpFifty" % tag, num(T[k]["auc"]))
            r = delta(k, null50)
            if r:
                mac("D%sRandomEpFifty" % tag, signed(r[0]))
                mac("D%sRandomEpFiftyCI" % tag, "[%s,\\,%s]" % (signed(r[1]), signed(r[2])))
                mac("D%sRandomEpFiftyP" % tag, pfmt(r[3]))
    # ---- cover-f021 at every epoch it has been probed. This arm is fp32, so
    # prefer an fp32 null when one exists at the same epoch; otherwise fall back
    # to the fp16 null and record that the contrast crosses the precision
    # boundary so the manuscript can flag it.
    EPW_ALL = {27: "EpTwentySeven", 30: "EpThirty", 34: "EpThirtyFour",
               50: "EpFifty", 73: "EpSeventyThree", 75: "EpSeventyFive",
               100: "EpHundred"}
    for ep, w in EPW_ALL.items():
        k = "cover-f021@ep%d@fp32" % ep
        if k not in T:
            continue
        mac("AUCCover%s" % w, num(T[k]["auc"]))
        for cand, flag in (("random@ep%d@fp32" % ep, "fp32"),
                           ("random@ep%d@fp16" % ep, "fp16")):
            if cand in T:
                r = delta(k, cand)
                if r:
                    mac("DCoverRandom%s" % w, signed(r[0]))
                    mac("DCoverRandom%sCI" % w, "[%s,\\,%s]" % (signed(r[1]), signed(r[2])))
                    mac("DCoverRandom%sP" % w, pfmt(r[3]))
                    mac("DCoverRandom%sNullPrec" % w, flag)
                break
    # cover's own progress, which is the cleanest statement about the arm
    if "cover-f021@ep75@fp32" in T and "cover-f021@ep50@fp32" in T:
        r = delta("cover-f021@ep75@fp32", "cover-f021@ep50@fp32")
        if r:
            mac("DCoverSelfFiftyToSeventyFive", signed(r[0]))
            mac("DCoverSelfFiftyToSeventyFiveP", pfmt(r[3]))
    if "random@ep75@fp16" in T and "random@ep50@fp16" in T:
        mac("DRandomSelfFiftyToSeventyFive",
            signed(T["random@ep75@fp16"]["auc"] - T["random@ep50@fp16"]["auc"]))
    for a, tag in (("envelope", "Envelope"), ("oracle", "Oracle")):
        k = "cover-f021@ep75@fp32"
        c = "%s@ep75@fp16" % a
        if k in T and c in T:
            r = delta(k, c)
            if r:
                mac("DCover%sEpSeventyFive" % tag, signed(r[0]))
                mac("DCover%sEpSeventyFiveP" % tag, pfmt(r[3]))

    mac("AUCAncestor", num(T["ancestor@ep25@fp32"]["auc"]))

    # fairness
    nblack = fair["arms"]["oracle@ep100@fp16"]["groups"]["race"]["per_group"]["Black"]["n"]
    nwhite = fair["arms"]["oracle@ep100@fp16"]["groups"]["race"]["per_group"]["White"]["n"]
    nasian = fair["arms"]["oracle@ep100@fp16"]["groups"]["race"]["per_group"]["Asian"]["n"]
    mac("NBlack", "%d" % nblack)
    mac("NWhite", "%d" % nwhite)
    mac("NAsian", "%d" % nasian)
    mac("NprobesRace", "%d" % fair["n_probes_with_race_summary"])
    mac("WorstRaceCount", "%d" % fair["worst_race_group_across_probes"].get("Black", 0))
    o = fair["arms"]["oracle@ep100@fp16"]["groups"]["race"]["per_group"]
    mac("BlackAUCOracle", num(o["Black"]["auc"]))
    mac("BlackCIOracle", "[%s,\\,%s]" % (num(o["Black"]["auc_ci95_lo"]), num(o["Black"]["auc_ci95_hi"])))
    mac("WhiteAUCOracle", num(o["White"]["auc"]))
    mac("RaceGapOracle", num(o["White"]["auc"] - o["Black"]["auc"]))
    rho, prho = gcorr["spearman_auc_vs_racegap"]
    mac("GapRho", signed(rho, 3))
    mac("GapRhoP", pfmt(prho))
    srho, sprho = gcorr["spearman_auc_vs_sexgap"]
    mac("SexGapRho", signed(srho, 3))
    mac("SexGapRhoP", pfmt(sprho))

    # ---- 19-probe subgroup trends (authoritative: richer metadata CSV) ----
    trendp = os.path.join(STATS, "p7b_gap_trend.json")
    if os.path.exists(trendp):
        tr = json.load(open(trendp))
        mac("NprobesSub", "%d" % tr["n_probes"])
        WORD = {"gender": "Gender", "race": "Race", "ethnicity": "Ethnicity",
                "language": "Language", "maritalstatus": "Marital",
                "age": "Age", "severity": "Severity"}
        for a, w in WORD.items():
            t = tr["trends"].get(a)
            if not t:
                continue
            mac("Sub%sRho" % w, signed(t["spearman_rho"], 3))
            mac("Sub%sRhoP" % w, pfmt(t["spearman_p"]))
            mac("Sub%sQ" % w, pfmt(t["q_bh_across_attributes"]))
            mac("Sub%sBranchRho" % w, signed(t["branch_spearman_rho"], 3))
            mac("Sub%sBranchP" % w, pfmt(t["branch_spearman_p"]))
            mac("Sub%sGapMin" % w, num(t["gap_min"]))
            mac("Sub%sGapMax" % w, num(t["gap_max"]))
            c = tr["worst_group_consistency"].get(a, {})
            cnt = c.get("counts", {})
            if cnt:
                top, n = max(cnt.items(), key=lambda kv: kv[1])
                safe = (top.replace("<", "$<$").replace(">", "$>$")
                           .replace("-6", "$-6$").replace("-2", "$-2$")
                           .replace("-12", "$-12$").replace("_", r"\_"))
                mac("Sub%sWorst" % w, safe)
                mac("Sub%sWorstN" % w, "%d" % n)
        mac("NbranchesSub", "%d" % tr["trends"]["race"]["n_branches"])
        sv = tr["trends"].get("severity")
        if sv:
            mac("SeverityGapSpread", num(sv["gap_max"] - sv["gap_min"]))
        aucs = [r["overall_auc"] for r in tr["rows"]]
        mac("SubAUCMin", num(min(aucs)))
        mac("SubAUCMax", num(max(aucs)))

    # ---- severity-stratified AUCs, matched epoch 100 ----
    subp = r"D:\jepa_phase0\autopilot_out\subgroup\subgroup_auc.json"
    if os.path.exists(subp):
        sub = json.load(open(subp))
        SEV = {"mild (-6,-2]": "Mild", "moderate (-12,-6]": "Moderate",
               "severe (<=-12)": "Severe"}
        pick = {"sweep_random_ep100": "Random", "sweep_oracle_ep100": "Oracle",
                "frozen_meanpool_mirage_ep100": "Envelope"}
        got = {}
        for probe, armw in pick.items():
            v = sub.get(probe)
            if not v:
                continue
            lv = {e["subgroup"]: e["auc"] for e in v["subgroups"]["severity"]["levels"]}
            for k, w in SEV.items():
                if k in lv:
                    mac("Sev%s%s" % (armw, w), num(lv[k]))
                    got[(armw, w)] = lv[k]
        for w in ("Mild", "Moderate", "Severe"):
            if ("Oracle", w) in got and ("Random", w) in got:
                mac("SevDelta%s" % w, signed(got[("Oracle", w)] - got[("Random", w)]))
        for armw in ("Random", "Oracle"):
            if (armw, "Severe") in got and (armw, "Mild") in got:
                mac("SevGap%s" % armw, num(got[(armw, "Severe")] - got[(armw, "Mild")]))

    # ---- race subgroup AUCs at matched epoch 100 (absolute, not gaps) ----
    for probe, armw in (("random@ep100@fp16", "Random"), ("oracle@ep100@fp16", "Oracle")):
        e = fair["arms"].get(probe)
        if not e:
            continue
        pg = e["groups"]["race"]["per_group"]
        for g in ("White", "Black", "Asian"):
            if g in pg and pg[g].get("auc") is not None:
                mac("Race%s%s" % (armw, g), num(pg[g]["auc"]))
    rb = fair["arms"]["oracle@ep100@fp16"]["groups"]["race"]["per_group"]["Black"]["auc"]
    rr = fair["arms"]["random@ep100@fp16"]["groups"]["race"]["per_group"]["Black"]["auc"]
    mac("BlackGainOracle", signed(rb - rr))

    # ---- fine-tuned heads (separate family, never pooled with frozen probes)
    ft = {}
    for rec in inv["records"]:
        if rec["family"] != "finetune":
            continue
        head = rec["tag"].split("/")[-1]
        ft[(rec["arm"], head)] = rec["auc"]
    if ("oracle", "meanpool") in ft and ("random", "mean_pool") in ft:
        fo = ft[("oracle", "meanpool")]
        fr = ft[("random", "mean_pool")]
        mac("FTOracleMeanpool", num(fo))
        mac("FTRandomMeanpool", num(fr))
        mac("FTDelta", signed(fo - fr))
    best_o = max((v for (a, h), v in ft.items() if a == "oracle"), default=None)
    best_r = max((v for (a, h), v in ft.items() if a == "random"), default=None)
    if best_o and best_r:
        mac("FTOracleBest", num(best_o))
        mac("FTRandomBest", num(best_r))
        mac("FTDeltaBest", signed(best_o - best_r))

    with open(os.path.join(AUTO, "auto_numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED by autopilot/p8_make_assets.py - do not edit by hand\n")
        f.write("% Every number in the manuscript resolves through these macros.\n")
        for name in sorted(M):
            f.write(r"\newcommand{\%s}{%s}" % (name, M[name]) + "\n")
    print("wrote auto_numbers.tex with %d macros" % len(M))

    # ---------------------------------------------------------- Table 1
    rows = []
    for arm in ("random", "envelope", "oracle"):
        cells = [ARM_TEX[arm]]
        for ep in (50, 75, 100):
            k = "%s@ep%d@fp16" % (arm, ep)
            t = T[k]
            cells.append(num(t["auc"]))
            if arm == "random":
                cells.append("---")
            else:
                d, lo, hi, p, c = delta(k, "random@ep%d@fp16" % ep)
                star = r"\textbf{%s}" % signed(d) if p < 0.05 else signed(d)
                cells.append(star)
        rows.append(" & ".join(cells) + r" \\")

    tab = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"& \multicolumn{2}{c}{epoch 50} & \multicolumn{2}{c}{epoch 75} & \multicolumn{2}{c}{epoch 100} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"policy & AUC & $\Delta$ & AUC & $\Delta$ & AUC & $\Delta$ \\",
           r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(AUTO, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED - matched epoch AND matched precision (fp16 family)\n")
        f.write("\n".join(tab) + "\n")
    print("wrote table_main.tex")

    # ---------------------------------------------------------- all probes
    rows = []
    for t in sorted(st["table"], key=lambda t: (t["arm"], t["epoch"])):
        rows.append(" & ".join([ARM_TEX.get(t["arm"], t["arm"]), str(t["epoch"]),
                                t["precision"], num(t["auc"]),
                                "[%s,\\,%s]" % (num(t["ci95_lo"]), num(t["ci95_hi"]))]) + r" \\")
    for r in inv["records"]:
        if r["status"] in ("retracted", "excluded"):
            rows.append(" & ".join([r["arm"].replace("-RETRACTED", ""), str(r["epoch"]),
                                    r["precision"], num(r["auc"]),
                                    r"\emph{%s}" % r["status"]]) + r" \\")
    tab = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"policy & epoch & precision & test AUC & 95\% CI \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(AUTO, "table_allprobes.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tab) + "\n")
    print("wrote table_allprobes.tex")

    # ---------------------------------------------------------- fairness table
    rows = []
    for key in ("random@ep100@fp16", "envelope@ep100@fp16", "oracle@ep100@fp16"):
        e = fair["arms"][key]
        pg = e["groups"]["race"]["per_group"]
        for g in ("White", "Black", "Asian"):
            d = pg[g]
            rows.append(" & ".join([ARM_TEX[e["arm"]] if g == "White" else "",
                                    g, str(d["n"]), num(d["auc"]),
                                    "[%s,\\,%s]" % (num(d["auc_ci95_lo"]), num(d["auc_ci95_hi"]))]) + r" \\")
        rows.append(r"\addlinespace")
    tab = [r"\begin{tabular}{llccc}", r"\toprule",
           r"policy & group & $n$ & AUC & 95\% CI \\", r"\midrule"] + rows[:-1] + \
          [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(AUTO, "table_fairness.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tab) + "\n")
    print("wrote table_fairness.tex")

    # ---------------------------------------------------------- Fig: trajectories
    # Left: trajectories. Right: PAIRED deltas vs the null with 95% CI.
    # Marginal per-arm CIs overlap heavily because they carry between-case
    # variance that cancels in a paired comparison; plotting them alone would
    # understate the evidence. The paired panel is the actual inference.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    anc = T["ancestor@ep25@fp32"]
    ax.plot([25], [anc["auc"]], "o", mfc="white", color="#555555", zorder=5)
    ax.annotate("shared ancestor (ep25)", (25, anc["auc"]), textcoords="offset points",
                xytext=(14, 26), fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))

    for arm in ("random", "oracle", "envelope"):
        pts = sorted([t for t in st["table"] if t["arm"] == arm and t["precision"] == "fp16"],
                     key=lambda t: t["epoch"])
        if not pts:
            continue
        ax.plot([25] + [p["epoch"] for p in pts], [anc["auc"]] + [p["auc"] for p in pts],
                "-o", color=COL[arm], label=ARM_PLOT[arm], lw=2.2, ms=5)

    for arm, mk in (("anatomy-v2", "s"), ("cover-f021", "^"), ("anatomy-v1", "D")):
        pts = sorted([t for t in st["table"] if t["arm"] == arm], key=lambda t: t["epoch"])
        if not pts:
            continue
        ax.plot([25] + [p["epoch"] for p in pts], [anc["auc"]] + [p["auc"] for p in pts],
                "--", marker=mk, color=COL[arm], label=ARM_PLOT[arm] + " (fp32)", lw=1.3, ms=4, alpha=0.8)

    ax.set_xlabel("pretraining epoch")
    ax.set_ylabel("frozen-probe test AUC")
    ax.set_title("(a) Trajectories from a shared checkpoint", fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")

    ax = axes[1]
    eps = [50, 75, 100]
    off = {"envelope": -0.30, "oracle": 0.0, "cover-f021": 0.30}
    for arm in ("envelope", "oracle", "cover-f021"):
        xs, ds, lo, hi = [], [], [], []
        for i, ep in enumerate(eps):
            # cover was probed fp32; prefer an fp32 null where one exists
            src = ("%s@ep%d@fp32" if arm == "cover-f021" else "%s@ep%d@fp16") % (arm, ep)
            if src not in T:
                continue
            nul = next((c for c in ("random@ep%d@fp32" % ep, "random@ep%d@fp16" % ep)
                        if c in T), None)
            if nul is None:
                continue
            r = delta(src, nul)
            if r is None:
                continue
            d, l, h, p, c = r
            xs.append(i + off[arm])
            ds.append(d)
            lo.append(d - l)
            hi.append(h - d)
        if not xs:
            continue
        ax.errorbar(xs, ds, yerr=[lo, hi], fmt="o", color=COL[arm], capsize=4,
                    ms=6, lw=1.8, label=ARM_PLOT[arm] + r" $-$ random")
    ax.axhline(0, color="#333333", lw=1.2, ls="-")
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels(["ep%d" % e for e in eps])
    ax.set_ylabel(r"$\Delta$ AUC vs unguided null")
    ax.set_title("(b) Paired difference, 95%% bootstrap CI\n(%s draws, same test cases)"
                 % st["n_bootstrap"], fontsize=10)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(AUTO, "fig_trajectories_ci.png"), dpi=200)
    plt.close(fig)
    print("wrote fig_trajectories_ci.png")

    # ---------------------------------------------------------- Fig: fairness
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    keys = ["random@ep100@fp16", "envelope@ep100@fp16", "oracle@ep100@fp16"]
    groups = ["White", "Black", "Asian"]
    w = 0.25
    for gi, g in enumerate(groups):
        vals = [fair["arms"][k]["groups"]["race"]["per_group"][g]["auc"] for k in keys]
        errlo = [vals[i] - fair["arms"][k]["groups"]["race"]["per_group"][g]["auc_ci95_lo"]
                 for i, k in enumerate(keys)]
        errhi = [fair["arms"][k]["groups"]["race"]["per_group"][g]["auc_ci95_hi"] - vals[i]
                 for i, k in enumerate(keys)]
        ax.bar(np.arange(len(keys)) + (gi - 1) * w, vals, w, yerr=[errlo, errhi],
               capsize=3, label="%s (n=%d)" % (g, fair["arms"][keys[0]]["groups"]["race"]["per_group"][g]["n"]))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([ARM_PLOT[k.split("@")[0]] for k in keys])
    ax.set_ylim(0.75, 0.93)
    ax.set_ylabel("test AUC")
    ax.set_title("Race-stratified AUC at epoch 100", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25, ls=":")

    ax = axes[1]
    xs, ys = [], []
    for k, e in fair["arms"].items():
        s = e["groups"]["race"]["summary"]
        if not s:
            continue
        xs.append(e["overall_auc"])
        ys.append(s["auc_gap"])
        ax.scatter(e["overall_auc"], s["auc_gap"], color=COL.get(e["arm"], "#333"), s=26)
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 50)
    ax.plot(xr, np.polyval(z, xr), "k--", lw=1)
    ax.set_xlabel("overall test AUC")
    ax.set_ylabel("max-min race AUC gap")
    ax.set_title(r"Gap widens as AUC improves ($\rho=%+.2f$, $p=%.3f$)" % (rho, prho), fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(AUTO, "fig_fairness.png"), dpi=200)
    plt.close(fig)
    print("wrote fig_fairness.png")

    # ---------------------------------------------------------- Fig: ROC
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    recs = {r["tag"] + "@" + str(r["epoch"]): r for r in inv["records"]}
    for arm in ("random", "envelope", "oracle"):
        rr = [r for r in inv["records"] if r["arm"] == arm and r["epoch"] == 100
              and r["family"] == "frozen_probe" and r["precision"] == "fp16"]
        if not rr:
            continue
        z = np.load(rr[0]["path"])
        y = z["labels"].astype(int)
        p = z["probs"].astype(np.float64)
        fpr, tpr, _ = roc_curve(y, p)
        k = "%s@ep100@fp16" % arm
        ax.plot(fpr, tpr, color=COL[arm], lw=1.8,
                label="%s  AUC %s" % (ARM_PLOT[arm], num(T[k]["auc"], 3)))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC at epoch 100 (N=%d)" % st["n_test"], fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(AUTO, "fig_roc.png"), dpi=200)
    plt.close(fig)
    print("wrote fig_roc.png")

    # ---------------------------------------------------------- subgroup trends
    trp = os.path.join(STATS, "p7b_gap_trend.json")
    if os.path.exists(trp):
        tr = json.load(open(trp))
        PRETTY = {"gender": "sex", "race": "race", "ethnicity": "ethnicity",
                  "language": "language", "maritalstatus": "marital status",
                  "age": "age", "severity": "disease severity"}
        rows = []
        for a in ("gender", "race", "ethnicity", "language", "maritalstatus", "age", "severity"):
            t = tr["trends"].get(a)
            if not t:
                continue
            c = tr["worst_group_consistency"].get(a, {})
            cnt = c.get("counts", {})
            top, n = max(cnt.items(), key=lambda kv: kv[1]) if cnt else ("-", 0)
            top = (top.replace("<", "$<$").replace("-6", "$-6$")
                      .replace("-2", "$-2$").replace("-12", "$-12$"))
            rows.append(" & ".join([
                PRETTY[a], "%s (%d/%d)" % (top, n, c.get("n", 0)),
                "%.4f--%.4f" % (t["gap_min"], t["gap_max"]),
                "%+.3f" % t["spearman_rho"], pfmt(t["spearman_p"]),
                pfmt(t["q_bh_across_attributes"]),
                "%+.3f" % t["branch_spearman_rho"], pfmt(t["branch_spearman_p"]),
            ]) + r" \\")
        tab = [r"\begin{tabular}{llccccccc}"[:-1] + "}", r"\toprule",
               r"& & & \multicolumn{3}{c}{per checkpoint ($n{=}%d$)} & \multicolumn{2}{c}{per branch ($n{=}%d$)} \\"
               % (tr["n_probes"], tr["trends"]["race"]["n_branches"]),
               r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
               r"attribute & worst group & gap range & $\rho$ & $p$ & $q$ & $\rho$ & $p$ \\",
               r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(AUTO, "table_subgroup_trends.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tab) + "\n")
        print("wrote table_subgroup_trends.tex")

    print("\nall assets in", AUTO)


if __name__ == "__main__":
    main()
