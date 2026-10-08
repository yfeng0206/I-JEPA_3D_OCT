"""Headless PNG preview and build report for the poster (no viewer is opened).

Usage: python make_preview.py PRINT_PDF --design-log poster.log --png out.png [--dpi 50]

Renders page 1 of the print PDF with PyMuPDF, then reports page count, physical
size, file size, fonts, Overfull/Underfull boxes, the tikzposter fit check,
unresolved placeholders and the figure files the design pulled in. Exits 1 if
the design overflows the page, a figure outside the allow-list is used, or the
PDF does not have exactly one page.
"""
import argparse
import collections
import os
import pathlib
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import pymupdf as fitz  # PyMuPDF

# Only figures built from the CC BY 4.0 OCTDL crop (no FairVision pixels).
ALLOWED_FIGURES = {
    "fig_oct_jepa_comparison_final_cr.pdf",
    "fig_oct_all_strategies.pdf",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--design-log", required=True)
    ap.add_argument("--png", required=True)
    ap.add_argument("--dpi", type=int, default=50)
    a = ap.parse_args()

    problems = []
    pdf_path = pathlib.Path(a.pdf)
    doc = fitz.open(pdf_path)
    page = doc[0]
    w_in, h_in = page.rect.width / 72.0, page.rect.height / 72.0
    pix = page.get_pixmap(dpi=a.dpi, alpha=False)
    pix.save(a.png)
    print(f"PDF            {pdf_path.name}: {doc.page_count} page(s), "
          f"{w_in:.2f} x {h_in:.2f} in ({w_in * 25.4:.1f} x {h_in * 25.4:.1f} mm), "
          f"{pdf_path.stat().st_size / 1e6:.2f} MB")
    print(f"PNG preview    {a.png}: {pix.width} x {pix.height} px at {a.dpi} dpi, "
          f"{pathlib.Path(a.png).stat().st_size / 1e6:.2f} MB")
    if doc.page_count != 1:
        problems.append(f"expected 1 page, found {doc.page_count}")

    fonts = sorted({f[3] for f in page.get_fonts(full=True)})
    print(f"Fonts          {len(fonts)}: {', '.join(fonts)}")
    if any(f[2] == "Type3" for f in page.get_fonts(full=True)):
        print("               note: Type3 fonts present")

    raw = pathlib.Path(a.design_log).read_text(encoding="utf-8", errors="replace")
    # TeX hard-wraps log lines at 79 characters; re-join them before matching.
    log = raw
    while True:
        joined = re.sub(r"^(.{79})\n", r"\1", log, flags=re.M)
        if joined == log:
            break
        log = joined
    overfull = re.findall(r"^Overfull \\[hv]box .*$", log, re.M)
    underfull = re.findall(r"^Underfull \\[hv]box .*$", log, re.M)
    print(f"Overfull boxes {len(overfull)}")
    for line in overfull:
        print(f"               {line}")
    print(f"Underfull      {len(underfull)}")
    for line in re.findall(r"POSTER-SIZE: .*", log):
        print("Size           " + " ".join(line.split()))
    for line in re.findall(r"POSTER-FIT .*", log):
        print(f"Fit            {line}")
    overflow = re.findall(r"OVERFLOW \(.*", log)
    for line in overflow:
        print(f"OVERFLOW       {line}")
        problems.append(line)
    holders = collections.Counter(re.findall(r"POSTER-PLACEHOLDER: ([^\n]+)", log))
    print("Placeholders   " + (", ".join(f"{k} x{v}" for k, v in sorted(holders.items()))
                               or "none"))
    figs = sorted(set(re.findall(r"(fig_[A-Za-z0-9_]+\.(?:pdf|png|jpe?g))", log)))
    print(f"Figures        {', '.join(figs) or 'none found in log'}")
    bad = [f for f in figs if f not in ALLOWED_FIGURES]
    if bad:
        problems.append(f"figures outside the allow-list: {bad}")
    if problems:
        print("PROBLEMS       " + "; ".join(problems))
        return 1
    print("Checks         OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
