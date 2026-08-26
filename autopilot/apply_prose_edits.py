"""Apply the prose edits proposed in autopilot/reports/P22_prose_edits.md.

The report gives, per edit, an OLD fenced block and a NEW fenced block. Every
OLD block is supposed to occur exactly once in main_submission.tex. This applies
them mechanically and refuses on any ambiguity rather than guessing, because a
silent partial application would leave the manuscript in a state nobody has read.

Safety rules enforced here:
  * an OLD block that does not occur exactly once is SKIPPED and reported
  * a NEW block that is longer than its OLD block is SKIPPED (the paper is at a
    hard page limit; every edit was proposed as net-neutral or shorter)
  * any edit that changes a digit is SKIPPED unless --allow-digits, because the
    numbers are macro-resolved measurements and prose must not restate them
  * brace balance is checked globally before anything is written

Usage:
  python apply_prose_edits.py [--dry-run] [--only E1,E2] [--allow-digits E37]
"""
import argparse
import io
import os
import re
import sys

REPORT = r"C:\Users\Gary\Desktop\jepa\autopilot\reports\P22_prose_edits.md"
TEX = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex"

BLOCK = re.compile(
    r"^###\s+(E\d+)\b[^\n]*\n"          # edit id
    r".*?^OLD\s*\n```\n(.*?)\n```\s*\n"  # old block
    r"^NEW\s*\n```\n(.*?)\n```",         # new block
    re.S | re.M)


def digits(s):
    return re.findall(r"\d", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--allow-digits", default="")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    digit_ok = {x.strip() for x in a.allow_digits.split(",") if x.strip()}

    rep = io.open(REPORT, encoding="utf-8").read()
    tex = io.open(TEX, encoding="utf-8").read()
    original = tex

    edits = BLOCK.findall(rep)
    print("parsed %d edits from the report" % len(edits))

    applied, skipped = [], []
    for eid, old, new in edits:
        if only and eid not in only:
            continue
        n = tex.count(old)
        if n != 1:
            skipped.append((eid, "OLD block occurs %d times, expected 1" % n))
            continue
        if len(new) > len(old):
            skipped.append((eid, "NEW longer than OLD (%d > %d chars)"
                            % (len(new), len(old))))
            continue
        if new.count("\n") > old.count("\n"):
            skipped.append((eid, "NEW has more lines than OLD"))
            continue
        if digits(old) != digits(new) and eid not in digit_ok:
            skipped.append((eid, "changes a digit; rerun with --allow-digits %s "
                                 "if intended" % eid))
            continue
        if old.count("{") - old.count("}") != new.count("{") - new.count("}"):
            skipped.append((eid, "brace balance differs between OLD and NEW"))
            continue
        tex = tex.replace(old, new, 1)
        applied.append(eid)

    if tex.count("{") != tex.count("}"):
        print("ABORT: global brace imbalance after edits (%d open, %d close)"
              % (tex.count("{"), tex.count("}")))
        return 1

    saved = original.count("\n") - tex.count("\n")
    print("\napplied %d, skipped %d, source lines saved %d"
          % (len(applied), len(skipped), saved))
    if applied:
        print("  applied: %s" % ", ".join(applied))
    for eid, why in skipped:
        print("  SKIP %-5s %s" % (eid, why))

    if a.dry_run:
        print("\ndry run, nothing written")
        return 0
    io.open(TEX, "w", encoding="utf-8", newline="").write(tex)
    print("\nwrote %s" % TEX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
