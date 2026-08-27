"""Count semicolons in prose only, excluding LaTeX code that legitimately uses them.

The style research measured 156 semicolons in main_submission.tex and compared
that with a 1.74/1kw field median.  That count is taken over the raw source,
which includes the TikZ schematic, where every path statement ends in a
semicolon as a matter of syntax rather than style.  Comparing a raw-source
count against a prose median would overstate the problem, so this strips the
code environments first and reports both numbers.
"""

import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

STRIP_ENVS = ["tikzpicture", "tabular", "table", "figure", "equation", "align"]

src = open("paper/genai4health2026/main_submission.tex", encoding="utf-8").read()
raw = src.count(";")

body = src
for env in STRIP_ENVS:
    body = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                  " ", body, flags=re.S)
body = re.sub(r"^\s*%.*$", "", body, flags=re.M)          # comments
body = re.sub(r"\\(newcommand|usepackage|documentclass|label|graphicspath)"
              r"\{[^}]*\}(\{[^}]*\})?", " ", body)

prose_semis = body.count(";")
words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", body))

print(f"raw source semicolons      : {raw}")
print(f"prose-only semicolons      : {prose_semis}")
print(f"  of which in code/floats  : {raw - prose_semis}")
print(f"prose words                : {words}")
print(f"PROSE DENSITY              : {prose_semis / words * 1000:.2f} per 1,000 words")
print(f"field median (15 landmarks): 1.74 per 1,000 words")
print(f"ratio to field median      : {prose_semis / words * 1000 / 1.74:.1f}x")
target = int(round(1.74 * words / 1000))
print(f"\nto reach the field median  : about {target} semicolons "
      f"({prose_semis - target} to remove)")
print(f"to reach 2x the median     : about {target * 2} "
      f"({max(0, prose_semis - target * 2)} to remove)")
