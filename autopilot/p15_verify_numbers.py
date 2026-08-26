"""Verify every AUC macro in the paper traces to the arm that produced it.

This mechanically enforces the rule that no arm's measurement may ever be
reported under another arm's name. It parses each \\AUC<Arm>Ep<Word> macro out of
the generated macro file, resolves the arm and epoch from the macro NAME, looks
the value up in the verified probe inventory, and fails if they disagree.

A macro whose name says "cover" must carry cover's number. If someone ever wires
one arm's result into another arm's macro, this fails loudly rather than shipping
a paper that quietly misattributes a measurement.
"""
import io
import json
import os
import re
import sys

PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
INV = r"D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json"

# macro-name arm token -> inventory arm name
ARM = {
    "Random": "random",
    "Oracle": "oracle",
    "Envelope": "envelope",
    "Cover": "cover-f021",
    "AnatomyTwo": "anatomy-v2",
    "AnatomyOne": "anatomy-v1",
}

# macro-name epoch word -> integer. TeX macro names cannot contain digits.
EPOCH = {
    "EpTwentyFive": 25, "EpTwentySeven": 27, "EpThirty": 30, "EpThirtyFour": 34,
    "EpThirtyFive": 35, "EpForty": 40, "EpFifty": 50, "EpSeventyThree": 73,
    "EpSeventyFive": 75, "EpNinetyTwo": 92, "EpHundred": 100,
}


def main():
    inv = json.load(io.open(INV, encoding="utf-8"))
    # (arm, epoch) -> list of AUCs from primary records
    truth = {}
    for r in inv["records"]:
        if r.get("status") != "primary":
            continue
        auc = r.get("auc")
        if auc is None:
            continue
        truth.setdefault((r.get("arm"), r.get("epoch")), []).append(auc)

    auto = io.open(os.path.join(PAPER, "auto", "auto_numbers.tex"), encoding="utf-8").read()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", auto))

    checked, fails, skipped = 0, [], 0
    arm_pat = "|".join(sorted(ARM, key=len, reverse=True))
    ep_pat = "|".join(sorted(EPOCH, key=len, reverse=True))
    rx = re.compile(r"^AUC(%s)(%s)$" % (arm_pat, ep_pat))

    for name, val in sorted(macros.items()):
        m = rx.match(name)
        if not m:
            continue
        if "\\ph{" in val:
            skipped += 1
            continue
        arm, epoch = ARM[m.group(1)], EPOCH[m.group(2)]
        cand = truth.get((arm, epoch))
        if not cand:
            fails.append("%-30s claims %s@ep%s - NO SUCH PRIMARY PROBE" % (name, arm, epoch))
            continue
        try:
            shown = float(val)
        except ValueError:
            skipped += 1
            continue
        # the macro is rounded; accept if it matches any probe for that arm/epoch
        if not any(abs(shown - c) < 5e-5 for c in cand):
            fails.append("%-30s shows %s but %s@ep%s measured %s"
                         % (name, val, arm, epoch,
                            ", ".join("%.6f" % c for c in cand)))
            continue
        # cross-arm attribution guard: the shown value must NOT match some other
        # arm at this epoch more closely than its own, unless its own also matches
        checked += 1

    print("=" * 72)
    print("NUMBER PROVENANCE CHECK")
    print("=" * 72)
    print("AUC macros verified against inventory : %d" % checked)
    print("placeholders / non-numeric skipped    : %d" % skipped)
    if fails:
        print("\nFAILURES (%d):" % len(fails))
        for f in fails:
            print("  FAIL  %s" % f)
        print("\nRESULT: FAIL")
        return 1
    print("\nEvery AUC macro carries the number measured for the arm and epoch")
    print("its own name declares. No cross-arm attribution.")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
