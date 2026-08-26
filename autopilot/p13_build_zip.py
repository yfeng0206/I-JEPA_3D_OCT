"""P13: build and VALIDATE the Overleaf submission ZIP.

Validation is the point of this script. It does not just archive files; it
extracts the archive to a scratch directory, compiles it there with no access to
the working tree, and refuses to declare success unless:

  1. the archive compiles standalone (catches missing \\input or figure files)
  2. main content fits the 9-page limit (references/appendix excluded)
  3. there are no undefined citations or references
  4. no author name, affiliation, or identifying URL appears (double-blind)
  5. no unresolved \\ph{} placeholder remains, unless --allow-placeholders
  6. every \\includegraphics target is present in the archive

Usage:
  python p13_build_zip.py [--allow-placeholders] [--out PATH]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime

PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
SCRATCH = r"D:\jepa_phase0\autopilot_out\zip_validate"
TECTONIC = r"D:\jepa_phase0\tools\tectonic\tectonic.exe"
PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
PAGE_LIMIT = 9

# Terms that would break double-blind review if present in the compiled PDF.
IDENTIFYING = [
    "yfeng", "Feng", "Gary", "Microsoft", "garyfeng",
    "github.com/yfeng0206", "huggingface.co/yfeng0206",
    "I-JEPA_3D_OCT", "ijepa-3d-oct-checkpoints",
]


def used_graphics(tex_text):
    """All \\includegraphics targets.

    The options and the filename may be split across lines with a trailing `%`
    line-continuation, so comment-continuations are folded away before matching.
    A naive regex silently misses those and produces a false PASS on the
    "all graphics present" check.
    """
    folded = re.sub(r"(?<!\\)%[^\n]*\n\s*", "", tex_text)
    return set(re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", folded))


def build(out_zip, allow_ph):
    src = os.path.join(PAPER, "main_submission.tex")
    tex = open(src, encoding="utf-8").read()

    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    stage = os.path.join(SCRATCH, "stage")
    os.makedirs(stage, exist_ok=True)

    # ---- assemble
    shutil.copy(src, os.path.join(stage, "main.tex"))
    for f in ("neurips_2026.sty", "references.bib"):
        shutil.copy(os.path.join(PAPER, f), os.path.join(stage, f))
    # auto/ is small and fully generated, so it ships whole
    s = os.path.join(PAPER, "auto")
    if os.path.isdir(s):
        shutil.copytree(s, os.path.join(stage, "auto"),
                        ignore=shutil.ignore_patterns("*.aux", "*.log"))
    # figures/ holds many images from earlier drafts; ship only what is cited,
    # otherwise the archive carries several MB Overleaf will never render.
    wanted_raw = used_graphics(tex)
    os.makedirs(os.path.join(stage, "figures"), exist_ok=True)
    for g in wanted_raw:
        base = os.path.basename(g)
        for ext in ("", ".png", ".pdf", ".jpg"):
            cand = os.path.join(PAPER, "figures", base + ext)
            if os.path.exists(cand):
                shutil.copy(cand, os.path.join(stage, "figures", base + ext))
                break

    # ---- 6. every referenced graphic must exist
    wanted = used_graphics(tex)
    missing_gfx = []
    for g in wanted:
        found = False
        for sub in ("figures", "auto", ""):
            for ext in ("", ".png", ".pdf", ".jpg"):
                if os.path.exists(os.path.join(stage, sub, g + ext)):
                    found = True
                    break
            if found:
                break
        if not found:
            missing_gfx.append(g)

    # ---- 5. placeholders
    # A \ph{} almost never appears literally in the manuscript: it lives in a
    # \newcommand inside auto/auto_numbers.tex and reaches the page through a
    # macro. Scanning only main_submission.tex therefore returned zero and
    # passed this check while Table 1 was visibly rendering red placeholder
    # cells. Resolve macros first, then report only those actually referenced.
    placeholders = [p for p in re.findall(r"\\ph\{([^}]*)\}", tex)
                    if "newcommand" not in p]
    auto_p = os.path.join(PAPER, "auto", "auto_numbers.tex")
    if os.path.exists(auto_p):
        auto_txt = open(auto_p, encoding="utf-8").read()
        for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*\\ph\{[^}]*\}[^}]*)\}",
                                    auto_txt):
            if re.search(r"\\%s\b" % name, tex):
                placeholders.append("%s (macro used in manuscript)" % name)

    # ---- 1. standalone compile
    rc = subprocess.call([TECTONIC, "-X", "compile", "main.tex", "--keep-logs",
                          "--keep-intermediates"],
                         cwd=stage, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    pdf = os.path.join(stage, "main.pdf")
    compiled = rc == 0 and os.path.exists(pdf)

    # ---- 2/4. page budget and anonymity, from the compiled PDF
    pages = refs_page = refs_y = None
    found_ident = []
    if compiled:
        try:
            import fitz
            d = fitz.open(pdf)
            pages = d.page_count
            texts = [d.load_page(i).get_text() or "" for i in range(pages)]
            for i in range(pages):
                hits = d.load_page(i).search_for("References")
                if hits and refs_page is None:
                    refs_page = i + 1
                    refs_y = hits[0].y0
            body = "\n".join(texts[: (refs_page or pages)])
            for term in IDENTIFYING:
                if term.lower() in body.lower():
                    found_ident.append(term)
        except Exception as e:
            print("[warn] pdf inspection failed:", e)

    # Main content occupies pages 1..refs_page, EXCEPT when the References
    # heading sits at the very top of its page, in which case the last page of
    # main content is refs_page - 1. The NeurIPS text block starts at y=72pt
    # (1 inch top margin), so a heading within ~20pt of that is page-topping.
    if refs_page is not None:
        main_pages = refs_page - 1 if (refs_y is not None and refs_y < 95) else refs_page
    else:
        main_pages = pages

    # ---- 3. undefined refs
    logp = os.path.join(stage, "main.log")
    undefined = []
    if os.path.exists(logp):
        log = open(logp, encoding="utf-8", errors="replace").read()
        undefined = re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", log)

    checks = {
        "1_compiles_standalone": compiled,
        "2_main_content_within_%d_pages" % PAGE_LIMIT:
            (main_pages is not None and main_pages <= PAGE_LIMIT),
        "3_no_undefined_refs": len(undefined) == 0,
        "4_anonymous": len(found_ident) == 0,
        "5_no_placeholders": (len(placeholders) == 0) or allow_ph,
        "6_all_graphics_present": len(missing_gfx) == 0,
    }
    ok = all(checks.values())

    # ---- write the archive only from the staged (validated) tree
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for fn in files:
                if fn.endswith((".aux", ".log", ".out", ".blg", ".xdv", ".bbl")):
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, stage))
        z.writestr("README_OVERLEAF.txt",
                   "Upload this zip to Overleaf and set main.tex as the root document.\n"
                   "Compiler: XeLaTeX or pdfLaTeX. Bibliography: BibTeX.\n"
                   "All numeric quantities resolve through auto/auto_numbers.tex,\n"
                   "which is generated from stored per-case predictions.\n"
                   "Do not edit auto/ by hand.\n")

    report = {"built": datetime.now().astimezone().isoformat(timespec="seconds"),
              "zip": out_zip, "zip_bytes": os.path.getsize(out_zip),
              "total_pages": pages, "references_start_page": refs_page, "references_heading_y": refs_y,
              "main_content_pages": main_pages, "page_limit": PAGE_LIMIT,
              "undefined_refs": undefined, "identifying_terms_found": found_ident,
              "unresolved_placeholders": placeholders,
              "missing_graphics": missing_gfx,
              "checks": checks, "ALL_PASS": ok}
    with open(os.path.join(SCRATCH, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print("\n=== ZIP VALIDATION ===")
    for k, v in checks.items():
        print("  %-40s %s" % (k, "PASS" if v else "FAIL"))
    print("\n  total pages          : %s" % pages)
    print("  references start page: %s (heading y=%s)" % (refs_page, refs_y));print("  main content pages   : %s (limit %d)" % (main_pages, PAGE_LIMIT))
    if undefined:
        print("  undefined refs       : %s" % undefined[:8])
    if found_ident:
        print("  IDENTIFYING TERMS    : %s" % found_ident)
    if placeholders:
        print("  placeholders         : %s" % placeholders)
    if missing_gfx:
        print("  missing graphics     : %s" % missing_gfx)
    print("\n  archive: %s (%.2f MB)" % (out_zip, os.path.getsize(out_zip) / 1e6))

    # Publish the validated PDF back into the repo. Without this the ZIP and
    # paper/main_submission.pdf drift apart, because this script compiles in a
    # scratch directory. A mock reviewer read the stale repo PDF and reported a
    # missing appendix and a page-limit violation that the real artifact did not
    # have, so the staleness cost a review round.
    if compiled:
        repo_pdf = os.path.join(PAPER, "main_submission.pdf")
        try:
            shutil.copy(pdf, repo_pdf)
            print("  published validated PDF -> %s" % repo_pdf)
        except Exception as e:
            print("  [warn] could not publish PDF: %s" % e)

    print("  ALL_PASS = %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--out", default=r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip")
    a = ap.parse_args()
    sys.exit(build(a.out, a.allow_placeholders))
