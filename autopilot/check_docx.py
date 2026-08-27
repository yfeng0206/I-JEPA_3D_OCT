"""Check the generated Word file is actually usable.

A .docx that silently dropped every generated number, or left "\\AUCRandomEpFifty"
as literal text, would look fine by file size and be worthless.  This reads the
document text out of the .docx zip and checks the things most likely to have
broken in a LaTeX to Word conversion.
"""

import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

DOCX = "paper/genai4health2026/main_submission.docx"

z = zipfile.ZipFile(DOCX)
xml = z.read("word/document.xml").decode("utf-8", "replace")
text = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))

media = [n for n in z.namelist() if n.startswith("word/media/")]
words = len(text.split())

print(f"words in document      : {words:,}")
print(f"embedded images        : {len(media)}")
print(f"document.xml size      : {len(xml):,} bytes")

# 1. Generated numbers must be present as values, not macro names.
leaked = sorted(set(re.findall(r"\\?(?:AUC|Delta|Nprobes|Ntest|Nboot)[A-Za-z]{2,}", text)))
print(f"\nleaked macro names     : {len(leaked)}")
for m in leaked[:8]:
    print("   ", m)

checks = {
    "headline AUC 0.8855": "0.8855",
    "envelope delta +0.0120": "0.0120",
    "test set size 3,000": "3,000",
    "banner present": "Working copy for comment",
    "abstract text": "retinal OCT",
    "a results table value": "0.8641",
}
print()
for label, needle in checks.items():
    print(f"  {'OK ' if needle in text else 'MISSING'}  {label}")

# 2. Citations should be rendered, not left as \citep keys.
raw_cites = len(re.findall(r"\\cite[a-z]*\{", text))
bracket_cites = len(re.findall(r"\((?:[A-Z][A-Za-z\-]+ (?:et al\.|and [A-Z][A-Za-z\-]+),? )\d{4}\)", text))
print(f"\nunrendered \\cite commands: {raw_cites}")
print(f"author-date citations    : {bracket_cites}")

# 3. Headings survived, so the document is navigable.
heads = re.findall(r'w:val="Heading\d"', xml)
print(f"styled headings          : {len(heads)}")

print("\nfirst 320 characters of body text:")
print("  " + " ".join(text.split())[:320])
