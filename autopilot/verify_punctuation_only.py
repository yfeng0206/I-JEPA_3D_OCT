"""Prove the semicolon pass changed punctuation only (line-based, fast).

Compares the committed manuscript with the working copy line by line.  For each
changed line, normalising away the expected transformation (";" becomes "." and
the capitalisation that follows) must make the two lines identical.  Anything
left over is reported.  A prose pass on a paper whose numbers are all
macro-generated must not touch a digit, a macro name, a citation key or a
reference, so those are also compared as whole-document token streams.

A character-level difflib comparison was tried first and is quadratic on a
117,000-character file, so it never finished.  Line alignment is sound here
precisely because a pure punctuation pass cannot reflow lines.
"""

import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

PATH = "paper/genai4health2026/main_submission.tex"

KNOWN = [  # made by hand earlier this session, before the semicolon pass
    "significantly", "interval excluding zero", "additionally", "also bridges",
    "also differs",
]

old = subprocess.run(["git", "show", "HEAD:" + PATH],
                     capture_output=True, text=True, encoding="utf-8").stdout
new = open(PATH, encoding="utf-8").read()

o, n = old.split("\n"), new.split("\n")
print(f"lines: {len(o)} -> {len(n)}")
if len(o) != len(n):
    print("LINE COUNT CHANGED - a pure punctuation pass should not reflow lines")


def canon(s):
    """Erase the expected edit: drop case and treat ; and . as the same mark."""
    return re.sub(r"[;.]", ".", s).lower()


semi = cap = 0
unexpected = []
for i, (a, b) in enumerate(zip(o, n)):
    if a == b:
        continue
    if canon(a) == canon(b):
        semi += a.count(";") - b.count(";")
        cap += sum(1 for x, y in zip(a, b) if x != y and x.lower() == y.lower())
    elif any(k in a or k in b for k in KNOWN):
        continue
    else:
        unexpected.append((i + 1, a.strip()[:96], b.strip()[:96]))

print(f"semicolons removed on changed lines : {semi}")
print(f"case-only character changes         : {cap}")
print(f"UNEXPECTED changed lines            : {len(unexpected)}")
for ln, a, b in unexpected[:25]:
    print(f"\n  L{ln}\n    - {a}\n    + {b}")


def toks(t):
    # A decimal point counts only when digits follow it, or converting
    # "Table 2;" to "Table 2." reads as a changed numeric literal.
    return (re.findall(r"\\[A-Za-z]+", t),
            re.findall(r"\d+(?:\.\d+)?", t),
            re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t),
            re.findall(r"\\(?:ref|label)\{([^}]*)\}", t))


print()
allsame = True
for name, a, b in zip(["macro names", "numeric literals", "citation keys",
                       "labels and refs"], toks(old), toks(new)):
    same = a == b
    allsame &= same
    print(f"{name:18s}: {len(a):5d} -> {len(b):5d}  "
          f"{'IDENTICAL' if same else 'CHANGED'}")
    if not same:
        sa, sb = set(a), set(b)
        print(f"    only in old: {sorted(sa - sb)[:8]}")
        print(f"    only in new: {sorted(sb - sa)[:8]}")

print()
print("VERDICT:", "PUNCTUATION ONLY" if not unexpected and allsame
      else "REVIEW REQUIRED")
