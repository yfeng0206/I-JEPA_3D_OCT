"""Build an editable Word version of the submission for collaborators.

The .tex remains authoritative.  This exists so people who do not write LaTeX
can read and comment, and it carries a banner saying so, because the moment a
number is edited in Word it loses the provenance guarantee the whole build is
designed around: every figure in the manuscript is a macro generated from a
stored artifact, and a hand-typed number is invisible to the gates.

Three things have to happen before pandoc can see the document properly.

1.  Flatten \\input.  Pandoc does not follow it, so the 400 generated number
    macros and the seven generated tables would otherwise vanish, leaving
    "\\AUCRandomEpFifty" as literal text or nothing at all.

2.  Render the TikZ schematic.  Pandoc cannot execute TikZ, so Figure 1(a)
    would be dropped silently.  It is compiled standalone and substituted as
    an image, which means the Word version keeps the pipeline diagram.

3.  Stamp provenance.  The banner records the commit and the build time so a
    stale copy is identifiable on sight.

Usage:
    python autopilot/make_docx.py
"""

import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

PAPER = "paper/genai4health2026"
MAIN = os.path.join(PAPER, "main_submission.tex")
OUT = os.path.join(PAPER, "main_submission.docx")
TECTONIC = r"D:\jepa_phase0\tools\tectonic\tectonic.exe"


def flatten(path, seen=None):
    """Inline every \\input recursively, so pandoc sees one document."""
    seen = seen or set()
    text = open(path, encoding="utf-8").read()

    def sub(m):
        rel = m.group(1).strip()
        cand = os.path.join(PAPER, rel)
        for p in (cand, cand + ".tex"):
            if os.path.exists(p) and p not in seen:
                seen.add(p)
                return flatten(p, seen)
        return ""

    return re.sub(r"\\input\{([^}]*)\}", sub, text)


def render_tikz(body, workdir):
    """Compile the schematic on its own and return a PNG path, or None."""
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", body, re.S)
    if not m:
        return None, body
    src = r"""\documentclass[border=3pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{amsmath}
\newcommand{\ArmBest}{\textsc{centroid}}
\begin{document}
""" + m.group(0) + "\n\\end{document}\n"

    tex = os.path.join(workdir, "schematic.tex")
    open(tex, "w", encoding="utf-8").write(src)
    r = subprocess.run([TECTONIC, "-X", "compile", tex, "--outdir", workdir],
                       capture_output=True, text=True)
    pdf = os.path.join(workdir, "schematic.pdf")
    if r.returncode != 0 or not os.path.exists(pdf):
        print("  schematic did not compile; leaving it out")
        print("  ", (r.stderr or "")[-300:])
        return None, body[:m.start()] + body[m.end():]

    import fitz
    doc = fitz.open(pdf)
    png = os.path.join(PAPER, "figures", "fig_pipeline_schematic.png")
    doc[0].get_pixmap(dpi=300).save(png)
    doc.close()
    print(f"  rendered schematic -> {png}")
    repl = r"\includegraphics[width=\linewidth]{fig_pipeline_schematic.png}"
    return png, body[:m.start()] + repl + body[m.end():]


def main():
    import pypandoc

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    print("flattening \\input ...")
    body = flatten(MAIN)
    print(f"  {len(body):,} characters after flattening")

    # A "\\" inside \title is a typesetting line break. Pandoc joins the parts
    # with no space, giving "Representationson Retinal OCT", so turn the breaks
    # into spaces before conversion.
    def fix_title(m):
        inner = re.sub(r"\\\\\s*", " ", m.group(1))
        return "\\title{" + re.sub(r"\s+", " ", inner).strip() + "}"

    body, n = re.subn(r"\\title\{(.*?)\}", fix_title, body, count=1, flags=re.S)
    print(f"  normalised {n} title line break(s)")

    work = tempfile.mkdtemp(prefix="docx_")
    try:
        print("rendering the TikZ schematic ...")
        _, body = render_tikz(body, work)

        banner = (
            r"\begin{center}\textbf{Working copy for comment and editing.}"
            "\\\\\n"
            r"The LaTeX source is authoritative. Generated from commit "
            + (commit or "unknown") + " on " + stamp + r". "
            r"Every number here was produced by a macro traced to a stored "
            r"artifact; if you edit one in Word it loses that link, so please "
            r"raise numeric changes as comments rather than edits."
            r"\end{center}" "\n\n"
        )
        body = body.replace(r"\begin{document}", r"\begin{document}" + "\n" + banner, 1)

        flat = os.path.join(work, "flat.tex")
        open(flat, "w", encoding="utf-8").write(body)

        print("converting with pandoc ...")
        pypandoc.convert_file(
            flat, "docx", format="latex", outputfile=OUT,
            extra_args=[
                "--citeproc",
                "--bibliography=" + os.path.join(PAPER, "references.bib"),
                "--resource-path=" + os.pathsep.join(
                    [PAPER,
                     os.path.join(PAPER, "figures"),
                     os.path.join(PAPER, "auto")]),
                "--toc",
                "--toc-depth=2",
            ])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    size = os.path.getsize(OUT)
    print(f"\nwrote {OUT} ({size:,} bytes)")


if __name__ == "__main__":
    main()
