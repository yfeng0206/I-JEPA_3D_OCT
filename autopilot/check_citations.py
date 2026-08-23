"""Citation integrity check.

Verifies that every \\cite key in the manuscript resolves to an entry in
references.bib, and that no bib entry is structurally malformed. Fabricated or
dangling citations are a real failure mode when a bibliography is assembled
quickly, and BibTeX reports them only as warnings that are easy to miss.
"""
import re
import os

PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"

tex = open(os.path.join(PAPER, "main_submission.tex"), encoding="utf-8").read()
bib = open(os.path.join(PAPER, "references.bib"), encoding="utf-8").read()

keys = set()
for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        if k.strip():
            keys.add(k.strip())

entries = {}
for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", bib):
    entries[m.group(2).strip()] = m.group(1).lower()

missing = sorted(k for k in keys if k not in entries)
unused = sorted(k for k in entries if k not in keys)

print("cited keys      : %d" % len(keys))
print("bib entries     : %d" % len(entries))
print("MISSING from bib: %s" % (missing if missing else "none"))
print("unused entries  : %d" % len(unused))
if unused:
    print("   ", ", ".join(unused[:15]), "..." if len(unused) > 15 else "")

# structural sanity on every cited entry
blocks = re.split(r"\n(?=@)", bib)
bad = []
for b in blocks:
    m = re.match(r"@(\w+)\s*\{\s*([^,]+),", b)
    if not m:
        continue
    key, typ = m.group(2).strip(), m.group(1).lower()
    if key not in keys:
        continue
    low = b.lower()
    problems = []
    if "year" not in low:
        problems.append("no year")
    if "author" not in low and "organization" not in low:
        problems.append("no author")
    if "title" not in low:
        problems.append("no title")
    if typ == "article" and "journal" not in low:
        problems.append("article without journal")
    if typ == "inproceedings" and "booktitle" not in low:
        problems.append("inproceedings without booktitle")
    if problems:
        bad.append((key, problems))

print("\nstructurally incomplete CITED entries: %d" % len(bad))
for k, p in bad:
    print("   %-28s %s" % (k, ", ".join(p)))

print("\nAll cited keys resolve:", len(missing) == 0)
