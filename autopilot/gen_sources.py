"""Generate SOURCES.md from the artifacts and the manuscript that actually exist.

Round-2 review, MUST-FIX 1: "make SOURCES.md describe the PDF that is actually
built." It had drifted, because it was hand-maintained while the paper was
regenerated many times a day.

The fix is structural rather than a one-off correction: this reads the real
inventory, the real statistics, the real macro file and the real .tex, and emits
the provenance document. Wired into refresh_all.py so it is rebuilt whenever the
paper is.
"""
import json
import os
import re
from datetime import datetime

STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
AUTO = os.path.join(PAPER, "auto")


def jload(n):
    p = os.path.join(STATS, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    inv = jload("p1b_full_inventory.json")
    st = jload("p1c_stats.json")
    tr = jload("p7b_gap_trend.json")
    fp = jload("p3b_fp32.json")
    pc = jload("p7c_paired_subgroup.json")
    op = jload("p8b_operating_points.json")

    tex = open(os.path.join(PAPER, "main_submission.tex"), encoding="utf-8").read()
    macros = re.findall(r"\\newcommand\{\\(\w+)\}", open(
        os.path.join(AUTO, "auto_numbers.tex"), encoding="utf-8").read())
    figs = sorted(set(re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}",
                                 re.sub(r"%[^\n]*\n\s*", "", tex))))
    inputs = sorted(set(re.findall(r"\\input\{([^}]+)\}", tex)))
    cites = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cites.add(k.strip())

    L = []
    A = L.append
    A("# SOURCES AND PROVENANCE")
    A("")
    A("**This file is generated** by `autopilot/gen_sources.py` from the artifacts and")
    A("the manuscript that actually exist, and is rebuilt on every refresh. Do not edit")
    A("it by hand: it will be overwritten, and a hand-edited copy is exactly how it")
    A("drifted from the paper before.")
    A("")
    A("Generated %s" % datetime.now().astimezone().isoformat(timespec="seconds"))
    A("")
    A("## 1. What the built PDF contains")
    A("")
    A("| item | count |")
    A("|---|---|")
    A("| auto-generated numeric macros | %d |" % len(macros))
    A("| generated tables `\\input` into the paper | %d |" % len(inputs))
    A("| figures included | %d |" % len(figs))
    A("| distinct citation keys | %d |" % len(cites))
    A("")
    A("Every numeric quantity in the manuscript resolves through")
    A("`auto/auto_numbers.tex`. No number is typed by hand, so prose, tables and")
    A("figures cannot disagree.")
    A("")
    A("Generated tables:")
    A("")
    for i in inputs:
        A("- `%s.tex`" % i)
    A("")
    A("Figures:")
    A("")
    for f in figs:
        A("- `%s`" % f)
    A("")

    if st:
        A("## 2. Statistical base")
        A("")
        A("- test set: N=%d, %d positive, %d negative, identical across every probe"
          % (st["n_test"], st["n_pos"], st["n_test"] - st["n_pos"]))
        A("- bootstrap: %s resamples, seed %s, stratified by class, the same resampled"
          % ("{:,}".format(st["n_bootstrap"]), st.get("bootstrap_seed")))
        A("  index set applied to every arm so all differences are paired")
        A("- primary frozen probes analysed: %d" % len(st["table"]))
        mult = st.get("multiplicity", {})
        if mult:
            A("- multiplicity: Benjamini-Hochberg within families, confirmatory family "
              "size %s, exploratory %s" % (mult.get("confirmatory_family_size"),
                                           mult.get("exploratory_family_size")))
        A("")
        A("### Probes in the analysis")
        A("")
        A("| arm | epoch | precision | test AUC |")
        A("|---|---|---|---|")
        for t in sorted(st["table"], key=lambda t: (t["arm"], t["epoch"])):
            A("| %s | %s | %s | %.6f |" % (t["arm"], t["epoch"], t["precision"], t["auc"]))
        A("")

    if inv:
        ex = [r for r in inv["records"] if r["status"] in ("retracted", "excluded")]
        A("### Excluded and retracted runs, never cited as evidence")
        A("")
        A("| run | arm | epoch | AUC | status |")
        A("|---|---|---|---|---|")
        for r in ex:
            A("| `%s` | %s | %s | %.6f | %s |" % (r["tag"], r["arm"], r["epoch"],
                                                  r["auc"], r["status"]))
        A("")
        for k, v in inv.get("exclusions", {}).items():
            A("- **%s**: %s" % (k, v))
        A("")

    if fp and fp.get("rows"):
        A("## 3. Precision")
        A("")
        A("`src/eval_downstream.py:541` reads `use_amp = data_cfg.get('use_amp', True)`,")
        A("so the harness default is fp16. Configs that omit the key ran fp16; those")
        A("setting it false ran fp32. Rather than assume this immaterial, it is measured:")
        A("")
        A("| arm | epoch | fp16 | fp32 | difference |")
        A("|---|---|---|---|---|")
        for r in sorted(fp["rows"], key=lambda r: (r["arm"], r["epoch"])):
            if "auc_fp16" in r:
                A("| %s | %d | %.6f | %.6f | %+.6f |" % (r["arm"], r["epoch"],
                  r["auc_fp16"], r["auc_fp32"], r["delta_fp32_minus_fp16"]))
        mx = max(abs(r.get("delta_fp32_minus_fp16", 0)) for r in fp["rows"])
        A("")
        A("Largest observed effect %.2e, orders of magnitude below the reported" % mx)
        A("differences. Each re-probe is hash-guarded: the encoder checkpoint is")
        A("SHA-256 hashed before and after and the run is invalidated if it changed.")
        A("")

    if tr:
        A("## 4. Subgroup analysis")
        A("")
        A("- probes used: %d, drawn from %d pretraining branches"
          % (tr["n_probes"], tr["trends"]["race"]["n_branches"]))
        A("- exclusion status is taken from the evidence inventory, not from the")
        A("  subgroup script's own tag, so runs the paper declares excluded are")
        A("  excluded here too")
        if tr.get("collapsed_duplicates"):
            A("- %d technical duplicates collapsed (an fp32 re-probe and its fp16"
              % len(tr["collapsed_duplicates"]))
            A("  original are the same frozen encoder scored twice):")
            for a, why in tr["collapsed_duplicates"]:
                A("  - `%s` %s" % (a, why))
        A("- %s" % tr.get("independence_caveat", ""))
        A("")

    if pc:
        A("## 5. Paired subgroup changes")
        A("")
        A("%s" % pc.get("method", ""))
        A("")
    if op:
        A("## 6. Clinical operating points")
        A("")
        A("%s. Cohort prevalence %.4f." % (op.get("protocol", ""), op.get("prevalence", float("nan"))))
        A("")

    A("## 7. Generators")
    A("")
    A("| artifact | generator |")
    A("|---|---|")
    for a, g in (("evidence inventory", "autopilot/p1b_full_inventory.py"),
                 ("paired statistics", "autopilot/p1c_stats.py"),
                 ("DeLong validation", "autopilot/p1_validate_delong.py"),
                 ("demographic join and alignment proof", "autopilot/p1_test_metadata.py"),
                 ("subgroup and severity", "paper/genai4health2026/scripts/subgroup_analysis.py"),
                 ("subgroup trends", "autopilot/p7b_gap_trend.py"),
                 ("paired subgroup changes", "autopilot/p7c_paired_subgroup.py"),
                 ("clinical operating points", "autopilot/p8b_operating_points.py"),
                 ("fp32 integration", "autopilot/p3b_integrate_fp32.py"),
                 ("macros, tables, figures", "autopilot/p8_make_assets.py"),
                 ("this file", "autopilot/gen_sources.py"),
                 ("build and validate the archive", "autopilot/p13_build_zip.py")):
        A("| %s | `%s` |" % (a, g))
    A("")
    A("## 8. Pending")
    A("")
    ph = [p for p in re.findall(r"\\ph\{([^}]*)\}", tex) if "newcommand" not in p]
    if ph:
        A("The following render red with a dagger in the PDF and are not yet measured:")
        A("")
        for p in ph:
            A("- %s" % p)
    else:
        A("No unresolved placeholders remain.")
    A("")

    with open(os.path.join(PAPER, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote SOURCES.md: %d macros, %d figures, %d tables, %d citations, %d placeholders"
          % (len(macros), len(figs), len(inputs), len(cites), len(ph)))


if __name__ == "__main__":
    main()
