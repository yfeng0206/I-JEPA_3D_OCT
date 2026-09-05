"""Generate a checked Word candidate without overwriting collaborator edits.

The current compiled aux supplies every cross-reference, including appendix
letters. Source hashes (not mtimes) and a DOCX hash record each successful build.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

try:
    from . import release_assets as assets
except ImportError:
    import release_assets as assets

PAPER = assets.PAPER
TECTONIC = r"D:\jepa_phase0\tools\tectonic\tectonic.exe"


def flatten(path, seen=None):
    path = Path(path)
    return assets.source_tree(path.parent, path.name)[0]


def resolve_references(body, aux):
    labels = assets.aux_labels(aux)
    references = assets.source_refs(body, labels)
    iterator = iter(references)

    def replace(match):
        item = next(iterator)
        number = item["number"]
        if match[0].startswith(r"\eqref"):
            number = "(" + number + ")"
        if match[0].startswith(r"\autoref"):
            prefix = item["key"].split(":")[0]
            number = {"fig": "Figure", "tab": "Table", "app": "Appendix",
                      "sec": "Section", "eq": "Equation"}.get(prefix, "Section") + "~" + number
        return r"\hypertarget{%s}{%s}" % (item["bookmark"], number)
    return re.sub(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", replace, body), references


def render_tikz(body, workdir):
    """Render all TikZ images into the private work directory; failures are fatal."""
    workdir = Path(workdir)
    generated = []
    definitions = "\n".join(r"\newcommand{\%s}{%s}" % (name, value)
                            for name, value, _, _ in assets.macros(body)
                            if name.startswith("Arm") and "#" not in value)

    def render(match):
        stem = "schematic_%02d" % len(generated)
        tex = workdir / (stem + ".tex")
        tex.write_text(r"\documentclass[border=3pt]{standalone}" "\n"
                       r"\usepackage{tikz}\usetikzlibrary{arrows.meta}" "\n"
                       r"\usepackage{amsmath}" "\n" + definitions +
                       "\n" r"\begin{document}" + match[0] + r"\end{document}", encoding="utf-8")
        result = subprocess.run([TECTONIC, "-X", "compile", str(tex), "--outdir", str(workdir)],
                                capture_output=True, text=True)
        pdf = tex.with_suffix(".pdf")
        if result.returncode or not pdf.is_file():
            raise RuntimeError("TikZ compile failed; refusing an incomplete Word copy: "
                               + (result.stderr or "")[-400:])
        import fitz
        png = tex.with_suffix(".png")
        with fitz.open(pdf) as doc:
            doc[0].get_pixmap(dpi=300).save(png)
        generated.append(png)
        return r"\includegraphics[width=\linewidth]{%s}" % png.name
    return generated, re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", render, body, flags=re.S)


def compile_aux(paper, work, snapshot=None):
    stage = work / "source"
    snapshot = snapshot if snapshot is not None else assets.input_hashes(paper)
    assets.copy_snapshot(paper, stage, snapshot)
    main = "main_submission.tex" if (stage / "main_submission.tex").exists() else "main.tex"
    result = subprocess.run([TECTONIC, "-X", "compile", main, "--keep-intermediates", "--keep-logs"],
                            cwd=stage, capture_output=True, text=True)
    (work / "compiler.txt").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    aux = (stage / main).with_suffix(".aux")
    if result.returncode or not aux.exists():
        raise ValueError("fresh source compile failed; cannot generate trustworthy Word cross-references")
    assets.assert_unchanged(stage, snapshot)
    return aux


def build(paper, out, aux=None, staging_root=None, expected_docx_sha256=None):
    import pypandoc
    paper, out = Path(paper).resolve(), Path(out).resolve()
    receipt = out.with_suffix(".docx.provenance.json")
    old_hash = assets.check_word_conflict(out, expected_docx_sha256, receipt)
    snapshot = assets.input_hashes(paper)
    work = assets.unique_work(staging_root, "docx")
    source_stage = work / "source"
    if aux:
        original_aux = Path(aux).resolve()
        assets.copy_snapshot(paper, source_stage, snapshot)
        aux = source_stage / "compiled.aux"
        assets.copy_verified(original_aux, aux, assets.sha256(original_aux))
    else:
        aux = compile_aux(paper, work, snapshot)
    body, _ = assets.source_tree(source_stage)
    body, references = resolve_references(body, aux)
    body = re.sub(r"\\title\{(.*?)\}",
                  lambda m: "\\title{" + re.sub(r"\\\\\s*", " ", m[1]).strip() + "}",
                  body, count=1, flags=re.S)
    generated, body = render_tikz(body, work)
    banner = (r"\begin{center}\textbf{Working copy for comment and editing.}\\"
              r"The LaTeX source is authoritative. Generated "
              + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
              + r". Numeric evidence coverage is recorded separately; conversion is not "
              r"scientific validation. Please raise numeric changes as comments."
              r"\end{center}" "\n")
    body = body.replace(r"\begin{document}", "\\begin{document}\n" + banner, 1)
    flat = work / "flat.tex"
    flat.write_text(body, encoding="utf-8")
    candidate = work / "candidate.docx"
    pypandoc.convert_file(str(flat), "docx", format="latex", outputfile=str(candidate),
                         extra_args=["--citeproc", "--bibliography=" + str(source_stage / "references.bib"),
                                     "--metadata=reference-section-title:References",
                                     "--resource-path=" + ";".join(map(str, (work, source_stage,
                                                                  source_stage / "figures", source_stage / "auto"))),
                                     "--toc", "--toc-depth=2"])
    try:
        from .check_docx import check
    except ImportError:
        from check_docx import check
    result = check(candidate, source_stage, aux, generated_images=generated)
    assets.write_json(work / "check_docx.json", result)
    if not result["ALL_PASS"]:
        raise ValueError("Word validation failed: " + "; ".join(result["errors"][:8]))
    assets.assert_unchanged(paper, snapshot)
    assets.assert_unchanged(source_stage, snapshot)
    if (assets.sha256(out) if out.exists() else None) != old_hash:
        raise ValueError("Word copy changed during build; preserving edits")
    provenance = work / "provenance.json"
    assets.write_json(provenance, {"docx_sha256": assets.sha256(candidate), "inputs": snapshot,
                                  "aux_sha256": assets.sha256(aux), "aux_path": str(aux), "references": references,
                                  "generated_images": {path.name: assets.sha256(path) for path in generated},
                                  "check": result})
    assets.promote([(candidate, out), (provenance, receipt)], expected_current={out: old_hash})
    print("Checked Word candidate:", out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default=str(PAPER))
    parser.add_argument("--out")
    parser.add_argument("--aux")
    parser.add_argument("--staging-root")
    parser.add_argument("--expected-docx-sha256")
    args = parser.parse_args(argv)
    paper = Path(args.paper_dir)
    out = args.out or paper / "main_submission.docx"
    aux = args.aux
    try:
        return build(paper, out, aux, args.staging_root, args.expected_docx_sha256)
    except Exception as exc:
        print("RESULT: FAIL:", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
