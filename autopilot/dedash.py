"""Replace prose em dashes in the submission with conventional punctuation.

Operates on the position of each ``---`` token rather than on surrounding
prose, so no multi-line source string ever has to be matched and no regex
group reference can be interpreted inside replacement text.  A previous
attempt at this class of edit with PowerShell ``-replace`` corrupted the
manuscript, because ``$0.784$`` in the replacement was read as a group
reference.

Each occurrence is assigned one of:
    ("pre", tok)   token attaches to the following word   e.g. "("
    ("post", tok)  token attaches to the preceding word   e.g. ")" "," ":"
    ("keep", None) left alone -- table cells, where an em dash is the
                   conventional "not applicable" marker
"""

import re
import shutil
import sys

TEX = "paper/genai4health2026/main_submission.tex"

# Index -> action. Indices are occurrence order of the ``---`` token.
PLAN = {
    0:  ("pre", "("),    1:  ("post", ")"),
    2:  ("post", ":"),
    3:  ("post", ":"),
    4:  ("post", ","),
    5:  ("pre", "("),    6:  ("post", ")"),
    7:  ("pre", "("),    8:  ("post", "),"),
    9:  ("post", ":"),
    10: ("keep", None), 11: ("keep", None), 12: ("keep", None),   # Table 1 cells
    13: ("pre", "("),   14: ("post", ")"),
    15: ("post", ","),
    16: ("post", ":"),                                            # subsection title
    17: ("pre", "("),   18: ("post", "),"),
    19: ("pre", "("),   20: ("post", ")"),
    21: ("pre", "("),   22: ("post", ")"),
    23: ("pre", "("),   24: ("post", ")"),
    25: ("pre", "("),   26: ("post", ")"),
    27: ("pre", "("),   28: ("post", ")"),
    29: ("pre", "("),   30: ("post", ")"),
    31: ("pre", "("),   32: ("post", "),"),
    33: ("pre", "("),   34: ("post", "),"),
    35: ("post", ":"),
    36: ("post", ","),
    37: ("post", ","),
    38: ("pre", "("),   39: ("post", ")"),
    40: ("post", ","),
    41: ("post", ","),
    42: ("post", ","),
    43: ("post", ":"),
    44: ("pre", "("),   45: ("post", ")"),
    46: ("post", ":"),
    47: ("pre", "("),   48: ("post", "),"),
    49: ("post", ":"),
    50: ("post", ","),
    51: ("pre", "("),   52: ("post", ")"),
    53: ("pre", "("),   54: ("post", ")"),
    55: ("pre", "("),   56: ("post", ")"),
    57: ("pre", "("),   58: ("post", "),"),
    59: ("pre", "("),   60: ("post", "),"),
    61: ("pre", "("),   62: ("post", "),"),
    63: ("post", ":"),
    64: ("post", ","),
    65: ("post", ":"),
}

# Follow-up wording fixes that the mechanical pass cannot express, applied
# after it.  Each must match exactly once.
FOLLOWUPS = [
    ("reproduce (if a paired difference changes sign",
     "reproduce (a paired difference changes sign"),
]


def main():
    src = open(TEX, encoding="utf-8").read()
    spans = list(re.finditer(r"([ \t\n]*)(?<!-)---(?!-)([ \t\n]*)", src))
    if len(spans) != len(PLAN):
        sys.exit(f"expected {len(PLAN)} occurrences, found {len(spans)}")

    shutil.copyfile(TEX, TEX + ".predash")

    out, cursor, changed = [], 0, 0
    for i, m in enumerate(spans):
        kind, tok = PLAN[i]
        out.append(src[cursor:m.start()])
        left, right = m.group(1), m.group(2)
        if kind == "keep":
            out.append(m.group(0))
        else:
            nl = "\n" in left or "\n" in right
            sep = "\n" if nl else " "
            out.append(sep + tok if kind == "pre" else tok + sep)
            changed += 1
        cursor = m.end()
    out.append(src[cursor:])
    new = "".join(out)

    for old, rep in FOLLOWUPS:
        if new.count(old) != 1:
            sys.exit(f"follow-up matched {new.count(old)} times: {old!r}")
        new = new.replace(old, rep)

    open(TEX, "w", encoding="utf-8", newline="").write(new)

    remaining = len(re.findall(r"(?<!-)---(?!-)", new))
    print(f"rewrote {changed} em dashes; {remaining} remain (table cells)")
    print(f"length {len(src)} -> {len(new)} chars ({len(new)-len(src):+d})")


if __name__ == "__main__":
    main()
