"""List every prose semicolon with the context needed to retype it.

Reports, for each: the clause before, the clause after, and whether the next
token can be capitalised directly.  A next token that is a LaTeX macro such
as \\textsc{random} cannot simply be upper-cased, so those are flagged and
handled separately rather than by a blanket rule.
"""

import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

STRIP_ENVS = ["tikzpicture", "tabular"]

src = open("paper/genai4health2026/main_submission.tex", encoding="utf-8").read()

# Blank out code environments in place, preserving offsets, so reported
# positions still index the real file.
masked = list(src)
for env in STRIP_ENVS:
    for m in re.finditer(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                         src, flags=re.S):
        for i in range(m.start(), m.end()):
            masked[i] = " "
masked = "".join(masked)

rows = []
for m in re.finditer(r";", masked):
    p = m.start()
    before = re.sub(r"\s+", " ", src[max(0, p - 78):p]).strip()
    after = re.sub(r"\s+", " ", src[p + 1:p + 74]).strip()
    nxt = after.split(" ")[0] if after else ""
    if nxt.startswith("\\"):
        kind = "MACRO"
    elif re.match(r"^[a-z]", nxt):
        kind = "lower"
    elif re.match(r"^[A-Z]", nxt):
        kind = "Upper"
    elif nxt.startswith("$"):
        kind = "MATH"
    else:
        kind = "other"
    rows.append((p, kind, before, after, nxt))

print(f"prose semicolons: {len(rows)}\n")
for i, (p, kind, before, after, nxt) in enumerate(rows):
    print(f"[{i:02d}] @{p} {kind}")
    print(f"     ...{before[-72:]}")
    print(f"  ;  {after[:72]}...")
