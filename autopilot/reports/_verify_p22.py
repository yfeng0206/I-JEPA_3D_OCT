"""Check the P22 prose edits against main_submission.tex.

Usage (from the repo root):
    python autopilot\\reports\\_verify_p22.py

Before applying: every OLD block should be found exactly once.
After applying:  every OLD block should be found zero times and every NEW block once.
"""

import io
import re
from collections import Counter

TEX = r"paper\genai4health2026\main_submission.tex"
MD = r"autopilot\reports\P22_prose_edits.md"

tex = io.open(TEX, encoding="utf-8").read().replace("\r\n", "\n")
md = io.open(MD, encoding="utf-8").read().replace("\r\n", "\n")

pairs = re.findall(r"\nOLD\n```\n(.*?)\n```\nNEW\n```\n(.*?)\n```\n", md, re.S)
print("edit blocks in report:", len(pairs))

pending = applied = problems = 0
for i, (old, new) in enumerate(pairs, 1):
    n_old, n_new = tex.count(old), tex.count(new)
    if n_old == 1 and n_new == 0:
        pending += 1
    elif n_old == 0 and n_new == 1:
        applied += 1
    else:
        problems += 1
        print("BLOCK %d: old=%d new=%d :: %s" % (i, n_old, n_new, old.split("\n")[0][:70]))

    issues = []
    if len(new) >= len(old):
        issues.append("NEW not shorter (%d -> %d chars)" % (len(old), len(new)))
    if new.count("\n") > old.count("\n"):
        issues.append("NEW uses more lines")
    if (old.count("{") - old.count("}")) != (new.count("{") - new.count("}")):
        issues.append("brace surplus differs")
    digits_old = sorted(re.findall(r"\d+(?:[.,]\d+)?", old))
    digits_new = sorted(re.findall(r"\d+(?:[.,]\d+)?", new))
    if digits_old != digits_new:
        issues.append("digits %s -> %s" % (digits_old, digits_new))
    if issues:
        print("BLOCK %d WARN: %s" % (i, "; ".join(issues)))

print("pending: %d   already applied: %d   problems: %d" % (pending, applied, problems))
print("source lines saved if all applied:", sum(o.count("\n") - n.count("\n") for o, n in pairs))

# Simulated application: global structural audit.
out = tex
for old, new in pairs:
    if out.count(old) == 1:
        out = out.replace(old, new)
print("brace balance  before: %d   after: %d" % (tex.count("{") - tex.count("}"),
                                                 out.count("{") - out.count("}")))
before, after = Counter(re.findall(r"\\[A-Za-z]+", tex)), Counter(re.findall(r"\\[A-Za-z]+", out))
changed = {k: (before[k], after[k]) for k in set(before) | set(after) if before[k] != after[k]}
print("macro count changes (expected: -4 emph, -1 textbf, +1 ref):", changed)
