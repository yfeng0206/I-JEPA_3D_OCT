"""P8: generate LaTeX tables, macros and figures from verified artifacts only.

Generated values retain named artifact lookups. Independent source/artist
checks are still required: sharing a generator does not guarantee agreement
between a table, a plot, and the interpretation in the manuscript.

Inputs  (all produced and checked earlier in this run)
  p1b_full_inventory.json  - labelled, de-duplicated evidence inventory
  p1c_stats.json           - AUCs, bootstrap CIs, DeLong contrasts by family
  p7_fairness.json         - subgroup AUCs and gaps
  p7_gap_correlation.json  - AUC-vs-gap correlations
  results/p17_subgroup_multiplicity.json - simultaneous subgroup intervals

Outputs -> paper/genai4health2026/auto/
  auto_numbers.tex, table_main.tex, table_fairness.tex, table_allprobes.tex
  fig_trajectories_ci.png, fig_fairness.png, fig_roc.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_curve

STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
REPO = r"C:\Users\Gary\Desktop\jepa"
AUTO = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026\auto"
SUBGROUP_ADJUSTED = os.path.join(REPO, r"results\p17_subgroup_multiplicity.json")
os.makedirs(AUTO, exist_ok=True)

ARM_TEX = {"random": r"\textsc{random}", "oracle": r"\ArmBest{}",
           "intensity": r"\ArmBest{}",
           "envelope": r"\textsc{envelope}", "anatomy-v1": r"\textsc{anatomy-v1}",
           "anatomy-v2": r"\textsc{anatomy-v2}", "cover-f021": r"\textsc{cover}",
           "ancestor": r"ancestor"}
# Display label for figures. The internal key stays "oracle" because that is the
# name in the stored artifacts and released checkpoints; only the paper renames.
ARM_PLOT = {"random": "random", "oracle": "centroid", "intensity": "centroid",
            "envelope": "envelope",
            "anatomy-v1": "anatomy-v1", "anatomy-v2": "anatomy-v2",
            "cover-f021": "cover", "cover": "cover", "ancestor": "ancestor"}

# ---------------------------------------------------------------- arm palette
# ONE arm-to-colour mapping for every figure this file emits, and the same
# mapping is mirrored in paper/genai4health2026/scripts/make_story_figures.py.
# Audited offline with the bundled skill tool
#   .agents/skills/scientific-visualization/scripts/palette_audit.py
#   --background FFFFFF --role graphical
# All seven colours clear the 3:1 background screen (worst: anatomy-v1 at 3.06;
# the previous ancestor #999999 failed at 2.85). 19 of 21 pairs clear the
# 10.0 dL* greyscale screen, against 9 failures before (worst pair 0.36).
#
# An exhaustive search over ALL 58 colours bundled with the skill (30 of which
# clear 3:1 on white) shows the largest subset whose every pair clears
# dL* >= 10 has exactly FIVE members, and six is impossible. That five-set is
# also near-monochromatic blue, so it is useless as a qualitative palette. Six
# or seven arms therefore cannot be separated by colour alone: the marker and
# line-style redundancy below is a necessity, not a preference.
COL = {"random": "#000000",      # L* 0.00,  contrast 21.00
       "ancestor": "#333333",    # L* 21.25, contrast 12.63
       "oracle": "#882255",      # L* 31.88, contrast  8.73  (Tol muted)
       "anatomy-v2": "#666666",  # L* 43.19, contrast  5.74
       "envelope": "#0072B2",    # L* 45.97, contrast  5.19  (Okabe-Ito blue)
       "cover-f021": "#009E73",  # L* 57.74, contrast  3.42  (Okabe-Ito green)
       "anatomy-v1": "#CC79A7"}  # L* 61.05, contrast  3.06  (Okabe-Ito purple)
# Globally unique marker per arm, so colour is never the only cue.
MK = {"random": "o", "ancestor": "*", "oracle": "s", "anatomy-v2": "v",
      "envelope": "^", "cover-f021": "D", "anatomy-v1": "P"}
# Historical aliases that appear as keys in some artifacts.
for _src, _dst in (("oracle", "intensity"), ("cover-f021", "cover")):
    COL[_dst] = COL[_src]
    MK[_dst] = MK[_src]

# Race subgroups in the fairness panel are a different variable from the arms,
# so they get their own audited triple rather than borrowing arm colours.
# dL* 0.00 / 31.88 / 45.97; all clear 3:1 on white.
GRP_COL = {"White": "#000000", "Black": "#882255", "Asian": "#0072B2"}
GRP_MK = {"White": "o", "Black": "s", "Asian": "^"}

def num(x, d=4):
    return ("%." + str(d) + "f") % x


def signed(x, d=4):
    return ("%+." + str(d) + "f") % x


def pfmt(p):
    if p < 1e-4:
        return "$<$0.0001"
    return "%.4f" % p


def matched_trajectory_keys(arm, epoch, table):
    """Select the same probe precision for both members of a plotted contrast."""
    if arm not in ("envelope", "oracle", "cover-f021"):
        raise ValueError("No trajectory precision contract for arm: " + arm)
    precision = "fp32" if arm == "cover-f021" else "fp16"
    source = "%s@ep%d@%s" % (arm, epoch, precision)
    baseline = "random@ep%d@%s" % (epoch, precision)
    missing = [key for key in (source, baseline) if key not in table]
    if missing:
        raise ValueError("Missing matched trajectory evidence: " + ", ".join(missing))
    return source, baseline


def main():
    inv = json.load(open(os.path.join(STATS, "p1b_full_inventory.json")))
    st = json.load(open(os.path.join(STATS, "p1c_stats.json")))
    fair = json.load(open(os.path.join(STATS, "p7_fairness.json")))
    gcorr = json.load(open(os.path.join(STATS, "p7_gap_correlation.json")))
    adjusted = json.load(open(SUBGROUP_ADJUSTED, encoding="utf-8"))

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
    # peak-to-endpoint decline: the arm's own best checkpoint against its last
    if "cover-f021@ep100@fp32" in T:
        cands = {k: T[k]["auc"] for k in T if k.startswith("cover-f021@")}
        peak = max(cands, key=cands.get)
        mac("CoverPeakEpoch", str(T[peak]["epoch"]))
        mac("AUCCoverPeak", num(T[peak]["auc"]))
        r = delta("cover-f021@ep100@fp32", peak)
        if r:
            mac("DCoverPeakToHundred", signed(r[0]))
            mac("DCoverPeakToHundredP", pfmt(r[3]))
    if "random@ep75@fp16" in T and "random@ep50@fp16" in T:
        mac("DRandomSelfFiftyToSeventyFive",
            signed(T["random@ep75@fp16"]["auc"] - T["random@ep50@fp16"]["auc"]))
    for ep, epw in ((75, "EpSeventyFive"), (100, "EpHundred")):
        for a, tag in (("envelope", "Envelope"), ("oracle", "Oracle")):
            k = "cover-f021@ep%d@fp32" % ep
            c = next((x for x in ("%s@ep%d@fp32" % (a, ep), "%s@ep%d@fp16" % (a, ep))
                      if x in T), None)
            if k in T and c:
                r = delta(k, c)
                if r:
                    mac("DCover%s%s" % (tag, epw), signed(r[0]))
                    mac("DCover%s%sP" % (tag, epw), pfmt(r[3]))

    # ---- anatomy-v2 at every epoch it has a VALID probe. Phase C replaces the
    # precision-spliced ep75/ep92 runs with clean fp32 continuations, so these
    # macros must appear on their own as soon as those land, without a hand edit.
    AEPW = {30: "EpThirty", 35: "EpThirtyFive", 40: "EpForty", 50: "EpFifty",
            75: "EpSeventyFive", 100: "EpHundred"}
    for ep, w in AEPW.items():
        k = "anatomy-v2@ep%d@fp32" % ep
        if k not in T:
            continue
        mac("AUCAnatomyTwo%s" % w, num(T[k]["auc"]))
        nul = next((c for c in ("random@ep%d@fp32" % ep, "random@ep%d@fp16" % ep)
                    if c in T), None)
        if nul:
            r = delta(k, nul)
            if r:
                mac("DAnatomyTwoRandom%s" % w, signed(r[0]))
                mac("DAnatomyTwoRandom%sCI" % w, "[%s,\\,%s]" % (signed(r[1]), signed(r[2])))
                mac("DAnatomyTwoRandom%sP" % w, pfmt(r[3]))
                mac("DAnatomyTwoRandom%sNullPrec" % w,
                    "fp32" if nul.endswith("fp32") else "fp16")
    # Epochs this arm was never carried to. The blob/anatomy-v2 fp32 continuation
    # was stopped after the epoch-75 milestone: at ep50 the arm was +0.0013 and
    # indistinguishable from the null, by ep75 it was -0.0111 and clearly below
    # it, so ep100 would have deepened an already-established deficit rather than
    # testing anything new. "not run" is the honest rendering; \ph{pending}
    # would imply a result is still coming.
    NOT_RUN = {"EpHundred"}
    for w in ("EpSeventyFive", "EpHundred"):
        if "AUCAnatomyTwo%s" % w not in M:
            filler = "---" if w in NOT_RUN else r"\ph{pending}"
            mac("AUCAnatomyTwo%s" % w, filler)
            mac("DAnatomyTwoRandom%s" % w, filler)

    # The cross-precision dagger must disappear on its own when the matching
    # fp32 null lands, otherwise the paper keeps warning about a confound it no
    # longer has. Emitted as a macro that is either the dagger or empty.
    for ep, w in (("75", "EpSeventyFive"), ("100", "EpHundred")):
        prec = M.get("DCoverRandom%sNullPrec" % w)
        mac("CoverDag%s" % w, "" if prec == "fp32" else r"^{\ddagger}")

    # Emit placeholder-valued macros for any COVER cell not yet measured, so the
    # table always compiles and resolves itself the moment the probe lands. This
    # removes a manual edit at the exact moment a headline number arrives, which
    # is when a hand edit is most likely to go wrong.
    for w in ("EpSeventyFive", "EpHundred"):
        if "AUCCover%s" % w not in M:
            mac("AUCCover%s" % w, r"\ph{pending}")
        if "DCoverRandom%s" % w not in M:
            mac("DCoverRandom%s" % w, r"\ph{pending}")
            mac("DCoverRandom%sCI" % w, r"\ph{pending}")
            mac("DCoverRandom%sP" % w, r"\ph{pending}")

    # ---- table-cell macros -------------------------------------------------
    # A cell must render either a bold math delta or a red pending marker, and
    # \ph{} cannot sit inside $...$ because it carries its own math. So the cell
    # macro carries its own formatting rather than the caller wrapping it, which
    # also guarantees a real math minus rather than a text hyphen.
    def cell(name, src, dagger=""):
        v = M.get(src)
        if v is None or v.startswith(r"\ph{"):
            mac(name, r"\ph{pending}")
        elif v == "---":
            # "not run" is not a quantity; keep it out of math mode so it renders
            # as an em dash rather than three minus signs.
            mac(name, "---")
        else:
            mac(name, r"$\mathbf{%s}%s$" % (v, dagger))

    for w in ("EpFifty", "EpSeventyFive", "EpHundred"):
        cell("TCoverRandom" + w, "DCoverRandom" + w, M.get("CoverDag" + w, ""))
        cell("TAnatomyTwoRandom" + w, "DAnatomyTwoRandom" + w)
        for src, pre in (("AUCCover", "TAUCCover"), ("AUCAnatomyTwo", "TAUCAnatomyTwo")):
            v = M.get(src + w)
            mac(pre + w, v if v else r"\ph{pending}")

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

    # ---- clinical operating points (p8b) -----------------------------------
    p8b = os.path.join(STATS, "p8b_operating_points.json")
    if os.path.exists(p8b):
        op = json.load(open(p8b))
        AW = {"random": "Random", "envelope": "Envelope", "intensity": "Intensity"}
        for arm, w in AW.items():
            a = op["arms"].get(arm)
            if not a:
                continue
            mac("Brier%s" % w, num(a["brier"]))
            mac("ECE%s" % w, num(a["ece_15bin"]))
            for k, kw in (("spec85", "SpecEightyFive"), ("spec90", "SpecNinety")):
                m = a["at"].get(k)
                if not m:
                    continue
                mac("Sens%s%s" % (w, kw), num(m["sensitivity"]))
                mac("SpecAch%s%s" % (w, kw), num(m["specificity"]))
                mac("PPV%s%s" % (w, kw), num(m["ppv"]))
                mac("NPV%s%s" % (w, kw), num(m["npv"]))
        for pair, pw in (("intensity_minus_random", "IntRand"),
                         ("envelope_minus_random", "EnvRand")):
            c = op["contrasts"].get(pair, {})
            for k, kw in (("spec85", "SpecEightyFive"), ("spec90", "SpecNinety")):
                v = c.get(k)
                if not v:
                    continue
                mac("DSens%s%s" % (pw, kw), signed(v["delta_sensitivity"]))
                mac("DSens%s%sCI" % (pw, kw),
                    "[%s,\\,%s]" % (signed(v["ci95_lo"]), signed(v["ci95_hi"])))
        mac("Prevalence", num(op["prevalence"]))

        rows = []
        for arm, w in (("random", "Random"), ("envelope", "Envelope"), ("intensity", "Intensity")):
            a = op["arms"].get(arm)
            if not a:
                continue
            for k, lbl in (("spec85", "0.85"), ("spec90", "0.90")):
                m = a["at"][k]
                rows.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                    ARM_TEX.get(arm, r"\textsc{%s}" % arm),
                    lbl, num(m["sensitivity"]), num(m["specificity"]),
                    num(m["ppv"]), num(m["npv"]), num(a["brier"]), num(a["ece_15bin"])))
        tab = [r"\begin{tabular}{llcccccc}", r"\toprule",
               r"policy & target spec. & sens. & spec. (test) & PPV & NPV & Brier & ECE \\",
               r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(AUTO, "table_operating.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tab) + "\n")
        print("wrote table_operating.tex")

    # ---- paired subgroup CHANGE intervals (p7c) ----------------------------
    p7c = os.path.join(STATS, "p7c_paired_subgroup.json")
    if os.path.exists(p7c):
        pc = json.load(open(p7c))
        ir = pc["contrasts"].get("intensity_minus_random", {})
        NAME = {"severity:mild": "SevMild", "severity:moderate": "SevModerate",
                "severity:severe": "SevSevere", "race:White": "RaceWhite",
                "race:Black": "RaceBlack", "race:Asian": "RaceAsian",
                "sex:Female": "SexFemale", "sex:Male": "SexMale"}
        for gk, w in NAME.items():
            v = ir.get("per_group", {}).get(gk)
            if not v:
                continue
            av = adjusted.get("auc_family", {}).get("contrasts", {}).get(gk)
            if av:
                v = dict(v)
                v["ci95_lo"] = av["simultaneous_ci95_lo"]
                v["ci95_hi"] = av["simultaneous_ci95_hi"]
                v["excludes_zero"] = av["simultaneous_excludes_zero"]
            mac("PD%sDelta" % w, signed(v["delta_auc"], 5))
            mac("PD%sCI" % w, "[%s,\\,%s]" % (signed(v["ci95_lo"], 5), signed(v["ci95_hi"], 5)))
            mac("PD%sSig" % w, "yes" if v["excludes_zero"] else "no")
        for gname, w in (("race", "Race"), ("sex", "Sex")):
            d = ir.get("%s_differential_benefit" % gname)
            if not d:
                continue
            adjusted_name = ("%s:%s-minus-%s"
                             % (gname,
                                "Black" if gname == "race" else "Female",
                                "Asian" if gname == "race" else "Male"))
            av = adjusted.get("auc_family", {}).get("contrasts", {}).get(adjusted_name)
            if av:
                d = dict(d)
                d["ci95_lo"] = av["simultaneous_ci95_lo"]
                d["ci95_hi"] = av["simultaneous_ci95_hi"]
                d["excludes_zero"] = av["simultaneous_excludes_zero"]
            mac("PDDiff%s" % w, signed(d["delta_worst_minus_delta_best"], 5))
            mac("PDDiff%sCI" % w, "[%s,\\,%s]" % (signed(d["ci95_lo"], 5), signed(d["ci95_hi"], 5)))
            mac("PDDiff%sSig" % w, "yes" if d["excludes_zero"] else "no")

        # appendix table
        rows = []
        for gk in ("severity:mild", "severity:moderate", "severity:severe",
                   "race:White", "race:Black", "race:Asian",
                   "sex:Female", "sex:Male"):
            v = ir.get("per_group", {}).get(gk)
            if not v:
                continue
            av = adjusted.get("auc_family", {}).get("contrasts", {}).get(gk)
            if av:
                v = dict(v)
                v["ci95_lo"] = av["simultaneous_ci95_lo"]
                v["ci95_hi"] = av["simultaneous_ci95_hi"]
                v["excludes_zero"] = av["simultaneous_excludes_zero"]
            lbl = gk.replace("severity:", "severity, ").replace("race:", "race, ").replace("sex:", "sex, ")
            rows.append("%s & %d & %s & [%s,\\,%s] & %s \\\\" % (
                lbl, v["n"], signed(v["delta_auc"], 5),
                signed(v["ci95_lo"], 5), signed(v["ci95_hi"], 5),
                "yes" if v["excludes_zero"] else "no"))
        tab = [r"\begin{tabular}{lcccc}", r"\toprule",
               r"stratum & $n$ & $\Delta$ AUC & simultaneous 95\% CI & excludes 0 \\", r"\midrule"] \
            + rows + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(AUTO, "table_paired_subgroup.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tab) + "\n")
        print("wrote table_paired_subgroup.tex")

    # ---- label efficiency ---------------------------------------------------
    # Registered BEFORE auto_numbers.tex is written; a previous version of this
    # script emitted macros after the write and silently dropped 30 of them.
    lep = os.path.join(STATS, "p5_label_efficiency.json")
    if os.path.exists(lep):
        le = json.load(open(lep, encoding="utf-8"))
        FRW = {"0.01": "One", "0.05": "Five", "0.10": "Ten",
               "0.25": "TwentyFive", "1.00": "Hundred"}
        # the arm is called oracle everywhere in the paper; the artifact predates
        # the rename and still says intensity
        ARMLE = {"random": "Random", "intensity": "Oracle",
                 "envelope": "Envelope", "cover": "Cover"}
        rows = []
        for fk in ("0.01", "0.05", "0.10", "0.25", "1.00"):
            cells = []
            ntr = None
            for arm in ("random", "oracle", "envelope", "cover"):
                src = "intensity" if arm == "oracle" else arm
                rec = le["arms"].get(src, {}).get(fk)
                if rec is None:
                    cells.append("--")
                    continue
                ntr = rec["n_train"]
                mac("LE%s%s" % (ARMLE[src], FRW[fk]), "%.4f" % rec["auc_mean"])
                mac("LESD%s%s" % (ARMLE[src], FRW[fk]), "%.4f" % rec["auc_sd"])
                cells.append("%.4f" % rec["auc_mean"])
            if ntr is not None:
                mac("LEN%s" % FRW[fk], "%d" % ntr)
                pct = "%g" % (float(fk) * 100)
                rows.append("%s\\%% & %d & %s \\\\" % (pct, ntr, " & ".join(cells)))
        mac("LERepeats", "%d" % le.get("repeats", 0))
        mac("LEEpoch", "%d" % le.get("epoch", 0))
        # headline gap at the sparsest fraction that is not dominated by noise
        try:
            g5 = (le["arms"]["intensity"]["0.05"]["auc_mean"]
                  - le["arms"]["random"]["0.05"]["auc_mean"])
            g100 = (le["arms"]["intensity"]["1.00"]["auc_mean"]
                    - le["arms"]["random"]["1.00"]["auc_mean"])
            mac("LEGapFive", "%+.4f" % g5)
            mac("LEGapHundred", "%+.4f" % g100)
        except KeyError:
            pass
        tab = [r"\begin{tabular}{llcccc}", r"\toprule",
               r"labels & $n$ & \textsc{random} & \ArmBest{} & "
               r"\textsc{envelope} & \textsc{cover} \\", r"\midrule"] \
            + rows + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(AUTO, "table_labeleff.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tab) + "\n")
        print("wrote table_labeleff.tex")

    # ---- subgroup operating points -----------------------------------------
    sop = os.path.join(STATS, "p16_subgroup_operating.json")
    if os.path.exists(sop):
        so = json.load(open(sop, encoding="utf-8"))
        GW = {"White": "White", "Black": "Black", "Asian": "Asian",
              "Female": "Female", "Male": "Male"}
        mac("SOPSpecTarget", "%.2f" % so["target_specificity"])
        mac("SOPThreshold", "%.4f" % so["threshold"])
        for arm, key in (("random", "Random"), ("intensity", "Oracle"),
                         ("envelope", "Envelope")):
            ov = so["arms"].get(arm, {}).get("overall")
            if ov:
                mac("SOPSens%s" % key, "%.4f" % ov["sensitivity"])
                mac("SOPSpec%s" % key, "%.4f" % ov["specificity"])
        rows = []
        for gname in ("race", "sex"):
            for v, s in sorted(so.get("delta_random_to_intensity", {}).get(gname, {}).items()):
                if v not in GW:
                    continue
                av = (adjusted.get("sensitivity_family", {}).get("contrasts", {})
                      .get("%s:%s" % (gname, v)))
                if av:
                    s = dict(s)
                    s["ci95_lo"] = av["simultaneous_ci95_lo"]
                    s["ci95_hi"] = av["simultaneous_ci95_hi"]
                    s["excludes_zero"] = av["simultaneous_excludes_zero"]
                mac("SOPD%s" % GW[v], "%+.4f" % s["d_sensitivity"])
                mac("SOPD%sCI" % GW[v],
                    "[%+.4f,\\,%+.4f]" % (s["ci95_lo"], s["ci95_hi"]))
                mac("SOPD%sSig" % GW[v], "yes" if s["excludes_zero"] else "no")
                mac("SOPN%sPos" % GW[v], "%d" % s["n_pos"])
                rows.append("%s & %d & $%+.4f$ & $[%+.4f,\\,%+.4f]$ & %s \\\\"
                            % (v, s["n_pos"], s["d_sensitivity"],
                               s["ci95_lo"], s["ci95_hi"],
                               "yes" if s["excludes_zero"] else "no"))
        tab = [r"\begin{tabular}{lcccc}", r"\toprule",
               r"stratum & positives & $\Delta$ sensitivity & simultaneous 95\% CI & excludes 0 \\",
               r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(AUTO, "table_subgroup_operating.tex"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(tab) + "\n")
        print("wrote table_subgroup_operating.tex")

    # ---- corrected COVER slot -----------------------------------------------
    # A collaborator is expected to supply a COVER run with truncation-aware
    # placement, restarted from the shared epoch-25 ancestor. Until that run
    # exists these render "---" (not run) rather than \ph{pending}: the paper
    # must not imply a result is imminent, and it must never state the
    # over-coverage conclusion before the evidence for it exists. Drop the probe
    # at runs/frozen_meanpool_coverfix_ep<N>/ and rerun refresh_all; the
    # inventory picks it up and these resolve automatically.
    cfix = os.path.join(STATS, "p19_cover_fixed.json")
    have_cfix = os.path.exists(cfix)
    if have_cfix:
        cf = json.load(open(cfix, encoding="utf-8"))
        for w, ep in (("EpFifty", 50), ("EpSeventyFive", 75), ("EpHundred", 100)):
            rec = cf.get("aucs", {}).get(str(ep))
            mac("AUCCoverFixed%s" % w, num(rec) if rec is not None else "---")
        mac("CoverFixedHidden", "%.1f" % cf["pct_anat_hid"])
        mac("CoverFixedStatus", "measured")
    else:
        for w in ("EpFifty", "EpSeventyFive", "EpHundred"):
            mac("AUCCoverFixed%s" % w, "---")
        mac("CoverFixedHidden", "---")
        mac("CoverFixedStatus", "not run")

    # ---- direct anatomy-vs-envelope contrast (H2 as literally stated) -------
    # H2 predicts the anatomy arms EXCEED envelope. An earlier draft argued this
    # transitively - "an arm that does not separate from the null cannot exceed
    # one that beats it" - which is the difference-in-significance fallacy and
    # does not test H2 at all. The direct paired contrast exists; use it.
    for r in st.get("contrasts", []):
        if r.get("a") == "anatomy-v2@ep50@fp32" and r.get("b") == "envelope@ep50@fp32":
            mac("DAnatomyTwoEnvelopeEpFifty", signed(r["delta_a_minus_b"]))
            mac("DAnatomyTwoEnvelopeEpFiftyCI",
                "[%s,\\,%s]" % (signed(r["delong_ci95"][0]), signed(r["delong_ci95"][1])))
            mac("DAnatomyTwoEnvelopeEpFiftyP", pfmt(r["delong_p"]))
            mac("DAnatomyTwoEnvelopeEpFiftyQ", pfmt(r["delong_q_bh"]))
        # anatomy-v1 at its only epoch runs the OTHER way, and the paper must say
        # so: a blanket "anatomical precision does not help" is contradicted by
        # this matched contrast.
        if r.get("a") == "anatomy-v1@ep30@fp32" and r.get("b") == "envelope@ep30@fp32":
            mac("DAnatomyOneEnvelopeEpThirty", signed(r["delta_a_minus_b"]))
            mac("DAnatomyOneEnvelopeEpThirtyCI",
                "[%s,\\,%s]" % (signed(r["delong_ci95"][0]), signed(r["delong_ci95"][1])))
            mac("DAnatomyOneEnvelopeEpThirtyP", pfmt(r["delong_p"]))

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
    ax.plot([25], [anc["auc"]], marker=MK["ancestor"], ls="none", mfc="white",
            color=COL["ancestor"], ms=9, mew=1.2, zorder=5)
    ax.annotate("shared ancestor (ep25)", (25, anc["auc"]), textcoords="offset points",
                xytext=(14, 26), fontsize=8, color=COL["ancestor"],
                arrowprops=dict(arrowstyle="-", color=COL["ancestor"], lw=0.7))

    # Solid line = fp16 family, dashed = fp32 family: line style carries
    # PRECISION here, so arm identity is carried redundantly by a unique marker.
    for arm in ("random", "oracle", "envelope"):
        pts = sorted([t for t in st["table"] if t["arm"] == arm and t["precision"] == "fp16"],
                     key=lambda t: t["epoch"])
        if not pts:
            continue
        ax.plot([25] + [p["epoch"] for p in pts], [anc["auc"]] + [p["auc"] for p in pts],
                ls="-", marker=MK[arm], color=COL[arm], label=ARM_PLOT[arm], lw=2.2, ms=5.5)

    for arm in ("anatomy-v2", "cover-f021", "anatomy-v1"):
        pts = sorted([t for t in st["table"] if t["arm"] == arm], key=lambda t: t["epoch"])
        if not pts:
            continue
        ax.plot([25] + [p["epoch"] for p in pts], [anc["auc"]] + [p["auc"] for p in pts],
                ls="--", marker=MK[arm], color=COL[arm], label=ARM_PLOT[arm] + " (fp32)",
                lw=1.3, ms=4.5, alpha=0.9)

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
            src, nul = matched_trajectory_keys(arm, ep, T)
            r = delta(src, nul)
            if r is None:
                raise ValueError("Missing paired trajectory contrast: %s minus %s" % (src, nul))
            d, l, h, p, c = r
            xs.append(i + off[arm])
            ds.append(d)
            lo.append(d - l)
            hi.append(h - d)
        if not xs:
            continue
        ax.errorbar(xs, ds, yerr=[lo, hi], fmt=MK[arm], color=COL[arm], capsize=4,
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
    # Dot-and-interval, not bars. The effects here are 0.006-0.012 AUC, and a
    # bar read from a non-zero baseline encodes those as a length ratio that
    # exaggerates them. A point encodes POSITION, so the axis may be scaled to
    # the data without deceiving; the interval is drawn and named.
    _all_v, _all_lo, _all_hi = [], [], []
    for gi, g in enumerate(groups):
        vals = [fair["arms"][k]["groups"]["race"]["per_group"][g]["auc"] for k in keys]
        errlo = [vals[i] - fair["arms"][k]["groups"]["race"]["per_group"][g]["auc_ci95_lo"]
                 for i, k in enumerate(keys)]
        errhi = [fair["arms"][k]["groups"]["race"]["per_group"][g]["auc_ci95_hi"] - vals[i]
                 for i, k in enumerate(keys)]
        ax.errorbar(np.arange(len(keys)) + (gi - 1) * w, vals, yerr=[errlo, errhi],
                    fmt=GRP_MK[g], color=GRP_COL[g], capsize=3, ms=6.5, lw=0,
                    elinewidth=1.4, markeredgecolor="white", markeredgewidth=0.6,
                    label="%s (n=%d)" % (g, fair["arms"][keys[0]]["groups"]["race"]["per_group"][g]["n"]))
        _all_v += vals
        _all_lo += errlo
        _all_hi += errhi
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([ARM_PLOT[k.split("@")[0]] for k in keys])
    ax.set_xlim(-0.5, len(keys) - 0.5)
    # Headroom for the key. Widening a point-encoding axis cannot exaggerate an
    # effect; it can only understate one, so it is the conservative direction.
    _lo = min(v - e for v, e in zip(_all_v, _all_lo))
    _hi = max(v + e for v, e in zip(_all_v, _all_hi))
    _rng = _hi - _lo
    ax.set_ylim(_lo - 0.08 * _rng, _hi + 0.30 * _rng)
    ax.set_ylabel("test AUC")
    ax.set_title("Race-stratified AUC at epoch 100\n"
                 "vertical bars: 95%% percentile bootstrap CI (%s resamples)"
                 % "{:,}".format(fair["n_bootstrap"]), fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25, ls=":")

    ax = axes[1]
    xs, ys = [], []
    seen = []
    for k, e in fair["arms"].items():
        s = e["groups"]["race"]["summary"]
        if not s:
            continue
        xs.append(e["overall_auc"])
        ys.append(s["auc_gap"])
        ax.scatter(e["overall_auc"], s["auc_gap"], color=COL.get(e["arm"], "#333"),
                   marker=MK.get(e["arm"], "o"), s=30, edgecolor="white", linewidth=0.4)
        if e["arm"] not in seen:
            seen.append(e["arm"])
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 50)
    ax.plot(xr, np.polyval(z, xr), "k--", lw=1)
    # Proxy handles only: the scatter above stays one call per probe so the
    # plotted values are untouched, and the key is built separately. Labels come
    # from ARM_PLOT, so the artifact name "oracle" can never reach the canvas.
    handles = [Line2D([], [], ls="none", marker=MK.get(a, "o"),
                      color=COL.get(a, "#333"), ms=5.5,
                      markeredgecolor="white", markeredgewidth=0.4,
                      label=ARM_PLOT.get(a, a))
               for a in seen]
    handles.append(Line2D([], [], ls="--", color="k", lw=1,
                          label="least-squares fit"))
    # Reserve a band above the data so the key cannot occlude a point.
    _r = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.08 * _r, max(ys) + 0.40 * _r)
    ax.legend(handles=handles, fontsize=6.5, ncol=4, loc="upper center",
              frameon=False, borderpad=0.3, handletextpad=0.3,
              columnspacing=0.9)
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
        # All three curves are fp16 at epoch 100, so line style is free to stay
        # solid; arm identity is carried redundantly by sparse markers using the
        # same global marker map as every other figure here.
        ax.plot(fpr, tpr, color=COL[arm], lw=1.8, marker=MK[arm],
                markevery=max(len(fpr) // 12, 1), ms=4.5,
                markeredgecolor="white", markeredgewidth=0.5,
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

    # ------------------------------------------------------- label efficiency
    # The strongest clinical result in the paper deserves a figure, not just a
    # table: the point is the SHAPE of the curve, that the arms converge as
    # labels are added and separate sharply as they are removed.
    lep2 = os.path.join(STATS, "p5_label_efficiency.json")
    if os.path.exists(lep2):
        le2 = json.load(open(lep2, encoding="utf-8"))
        order = [("random", "random"), ("intensity", "centroid"),
                 ("envelope", "envelope"), ("cover", "cover")]
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        for key, lab in order:
            col, mk = COL[key], MK[key]
            arm = le2["arms"].get(key)
            if not arm:
                continue
            fr, mu, sd = [], [], []
            for fk in sorted(arm, key=float):
                fr.append(float(fk) * 100)
                mu.append(arm[fk]["auc_mean"])
                sd.append(arm[fk]["auc_sd"])
            mu, sd = np.array(mu), np.array(sd)
            ax.plot(fr, mu, marker=mk, ms=4.2, lw=1.6, color=col, label=lab,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.fill_between(fr, mu - sd, mu + sd, color=col, alpha=0.13, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("labelled training set used (%, log scale)")
        ax.set_ylabel("test AUC")
        ax.legend(fontsize=7.5, loc="lower right", ncol=2)
        ax.grid(alpha=0.25, ls=":")
        fig.tight_layout()
        fig.savefig(os.path.join(AUTO, "fig_labeleff.png"), dpi=200)
        plt.close(fig)
        print("wrote fig_labeleff.png")

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
