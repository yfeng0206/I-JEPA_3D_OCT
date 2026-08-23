"""Merge newly verified BibTeX entries into references.bib.

Only entries whose key is absent are appended, so hand-curated entries already
in the file are never overwritten. Every appended entry came from the P2-02
literature agent's VERIFIED list; its "UNVERIFIED - do not cite" section is not
in that file and is therefore never merged.
"""
import re
import os
import shutil

PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
NEW = r"C:\Users\Gary\Desktop\jepa\autopilot\reports\P2-02_references.bib"
DST = os.path.join(PAPER, "references.bib")

cur = open(DST, encoding="utf-8").read()
new = open(NEW, encoding="utf-8").read()

cur_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", cur))

blocks = []
for m in re.finditer(r"(@\w+\s*\{\s*([^,]+),)", new):
    start = m.start()
    depth = 0
    i = new.index("{", start)
    for j in range(i, len(new)):
        if new[j] == "{":
            depth += 1
        elif new[j] == "}":
            depth -= 1
            if depth == 0:
                blocks.append((m.group(2).strip(), new[start:j + 1]))
                break

added, skipped = [], []
out = [cur.rstrip(), "",
       "% ---------------------------------------------------------------",
       "% Appended from the verified P2-02 literature search (2026-08-22).",
       "% Every entry below was checked against a primary source by the",
       "% research agent; unverified candidates were excluded at source.",
       "% ---------------------------------------------------------------", ""]
for key, block in blocks:
    if key in cur_keys:
        skipped.append(key)
        continue
    out.append(block)
    out.append("")
    added.append(key)
    cur_keys.add(key)

if added:
    shutil.copy(DST, DST + ".bak_premerge")
    with open(DST, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

print("new bib entries parsed : %d" % len(blocks))
print("appended (new keys)    : %d" % len(added))
for k in added:
    print("   +", k)
print("skipped (already present): %d" % len(skipped))
print("   ", ", ".join(skipped))
