"""Self-check the manuscript for the error classes that actually bit us.

Every one of these checks exists because the corresponding mistake was made and
caught during this run:

  1 undefined macro          - a \\newcommand emitted after auto_numbers.tex was
                               written, so 30 macros silently vanished
  2 duplicate macro          - two code paths defined \\AUCCoverEpFifty, which is
                               a hard LaTeX error
  3 hand-typed result number - the original draft had transcription errors
                               between prose, tables and figures
  4 dangling citation        - a \\cite key with no bib entry
  5 dangling reference       - a \\ref with no matching \\label
  6 stale probe count        - the paper claimed 21 probes while the artifacts
                               held 28
  7 contradiction phrases    - the Conclusion once contradicted the Abstract on
                               whether subgroup gains reached the worst group
  8 overclaiming phrases     - "pre-specified" on a repeatedly-inspected split,
                               "direct signature" for an unvalidated mechanism

Exit code is non-zero if any hard check fails, so refresh_all can gate on it.
"""
import json
import os
import re
import sys

PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
AUTO = os.path.join(PAPER, "auto")
STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"

# phrases that were factually wrong at some point and must not reappear
BANNED = [
    ("pre-specified", "the test split was inspected repeatedly; not defensible"),
    ("prespecified", "same"),
    ("direct signature", "asserts an unvalidated mechanism for the OD/OS clusters"),
    ("do not reach the groups", "contradicts the measured subgroup gains"),
    ("does not close any gap", "contradicts the measured subgroup gains"),
    ("identical throughout", "precision is not identical across probes"),
]


def main():
    tex = open(os.path.join(PAPER, "main_submission.tex"), encoding="utf-8").read()
    # Strip TeX comments, but NOT escaped percents: "43.7\%" is a literal
    # percent sign, not a comment. A naive %[^\n]* deleted the rest of any line
    # containing a percentage, which blinded every check below to the
    # percent-dense geometry table and could hide an undefined macro there.
    body = re.sub(r"(?<!\\)%[^\n]*", "", tex)
    auto = open(os.path.join(AUTO, "auto_numbers.tex"), encoding="utf-8").read()

    fails, warns = [], []

    # ---- 1 & 2: macros
    defined = re.findall(r"\\newcommand\{\\(\w+)\}", auto)
    dupes = sorted({m for m in defined if defined.count(m) > 1})
    defset = set(defined) | set(re.findall(r"\\newcommand\{\\(\w+)\}", body))
    used = set(re.findall(r"\\([A-Z][A-Za-z]{3,})\b", body))
    known_tex = {"LaTeX", "TeX", "Delta", "Nboot", "Ntest", "Npos", "Nneg"}
    undefined = sorted(u for u in used
                       if u not in defset and u not in known_tex
                       and (u.startswith(("AUC", "D", "Sub", "Sev", "Race", "PD",
                                          "Sens", "Spec", "Brier", "ECE", "FT",
                                          "CI", "Nprobes", "Nbranches", "Prev"))))
    if dupes:
        fails.append("duplicate macro definitions: %s" % dupes)
    if undefined:
        fails.append("macros used but never defined: %s" % undefined)

    unused = sorted(set(defined) - used)
    if unused:
        warns.append("%d generated macros are unused (harmless): %s%s"
                     % (len(unused), ", ".join(unused[:6]),
                        " ..." if len(unused) > 6 else ""))

    # ---- 3: hand-typed result numbers in the argued sections
    seg = body
    for marker in ("\\section{Discussion", "\\appendix"):
        i = seg.find(marker)
        if i > 0:
            seg = seg[:i]
    lits = []
    for m in re.finditer(r"(?<![\w.])0\.8[0-9]{3,}|(?<![\w.])[+-]0\.0[0-9]{3,}", seg):
        ctx = seg[max(0, m.start() - 60):m.start()]
        if "\\newcommand" in ctx:
            continue
        line = seg[:m.start()].count("\n") + 1
        lits.append("%s (line ~%d)" % (m.group(0), line))
    if lits:
        warns.append("hand-typed numeric literals before Discussion: %s" % lits)

    # ---- 4: citations
    bib = open(os.path.join(PAPER, "references.bib"), encoding="utf-8").read()
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    cited = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            if k.strip():
                cited.add(k.strip())
    missing = sorted(cited - keys)
    if missing:
        fails.append("cited but absent from references.bib: %s" % missing)

    # ---- 5: labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", body))
    dangling = sorted(refs - labels)
    if dangling:
        fails.append("refs with no matching label: %s" % dangling)

    # ---- 6: probe counts must match the artifacts
    stp = os.path.join(STATS, "p1c_stats.json")
    trp = os.path.join(STATS, "p7b_gap_trend.json")
    if os.path.exists(stp) and os.path.exists(trp):
        st = json.load(open(stp))
        tr = json.load(open(trp))
        m = re.search(r"\\newcommand\{\\NprobesSub\}\{(\d+)\}", auto)
        if m and int(m.group(1)) != tr["n_probes"]:
            fails.append("NprobesSub macro %s but p7b has %d"
                         % (m.group(1), tr["n_probes"]))
        for bad in re.findall(r"\b(\d{1,2}) (?:frozen )?probes\b", body):
            if int(bad) not in (len(st["table"]), tr["n_probes"],
                                tr["trends"]["race"]["n_branches"]):
                warns.append("literal '%s probes' in text; artifacts say %d analysed "
                             "and %d in the subgroup set"
                             % (bad, len(st["table"]), tr["n_probes"]))

    # ---- 7 & 8: banned phrases
    for phrase, why in BANNED:
        if phrase.lower() in body.lower():
            fails.append("banned phrase '%s' reappeared (%s)" % (phrase, why))

    print("=" * 72)
    print("MANUSCRIPT CONSISTENCY CHECK")
    print("=" * 72)
    print("macros defined %d | used %d | duplicates %d | undefined %d"
          % (len(set(defined)), len(used & defset), len(dupes), len(undefined)))
    print("citations %d cited, %d in bib, %d missing" % (len(cited), len(keys), len(missing)))
    print("labels %d, refs %d, dangling %d" % (len(labels), len(refs), len(dangling)))
    print()
    if fails:
        print("FAILURES (%d):" % len(fails))
        for f in fails:
            print("  FAIL  %s" % f)
    else:
        print("no hard failures")
    if warns:
        print("\nwarnings (%d):" % len(warns))
        for w in warns:
            print("  warn  %s" % w)
    print()
    print("RESULT:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
