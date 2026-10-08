"""Build in a unique stage; publish ZIP/PDF/DOCX only after all release gates.

The manifest is published last and records the exact validated source tree.
An interrupted multi-file promotion is detectable by its hashes. Failed gates
leave the previous deliverables and release manifest untouched.

Submission mode (default) enforces anonymity and the 9-page review limit.
--camera-ready replaces the anonymity gate with authors_present (fails while
the author/acknowledgment placeholders remain or the reviewed
camera_ready_authors.json is unconfirmed), keeps a local-path hygiene gate,
adds a workshop-template gate and allows 10 main-text pages, counted up to the
first of the acknowledgments or References heading.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

try:
    from . import release_assets as assets
except ImportError:
    import release_assets as assets

PAPER = str(assets.PAPER)
TECTONIC = r"D:\jepa_phase0\tools\tectonic\tectonic.exe"
PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
PAGE_LIMIT = 9
IDENTIFYING = ["yfeng", "Gary Feng", "Microsoft", "garyfeng",
              "github.com/yfeng0206", "huggingface.co/yfeng0206",
              "I-JEPA_3D_OCT", "ijepa-3d-oct-checkpoints"]
used_graphics = assets.graphics

# Camera-ready mode (--camera-ready): authors are visible, so the anonymity gate
# is replaced by an authors-present gate, a local-path hygiene gate and a
# template gate. Acknowledgments, references and appendices are outside the
# 10-page main-text limit.
CAMERA_READY_PAGE_LIMIT = 10
DEFAULT_OUT = r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip"
CAMERA_READY_DEFAULT_OUT = r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_camera_ready.zip"
AUTHORS_RECORD = "camera_ready_authors.json"
AUTHOR_PLACEHOLDERS = ("CRAuthorsPending", "CRAcknowledgmentsPending")
ACK_HEADING = "Acknowledgments and Disclosure of Funding"
WORKSHOP_NOTICE = "GenAI for Health Workshop @ NeurIPS 2026"
SUBMISSION_NOTICE = ("Submitted to", "Do not distribute")
# Official kit bytes (neurips_2026.sty from the NeurIPS 2026 author kit;
# genai4health_2026.sty from the workshop camera-ready starter, 2026-10-05).
OFFICIAL_STYLES = {
    "neurips_2026.sty": "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a",
    "genai4health_2026.sty": "fb8e525a0f800dd3e0fdee860156f0259902c3b117a1a5caae509322578c493e",
}
LINE_NUMBER_MARGIN_X = 105  # lineno prints digits left of the 108 pt text block


def normalized(text):
    return assets.plain_name(text)


def identifying_text_hits(text):
    folded = re.sub(r"\s+", " ", text).casefold()
    hits = [term for term in IDENTIFYING if term.casefold() in folded]
    hits.extend(local_path_hits(text))
    return hits


def local_path_hits(text):
    if re.search(r"\b(?:[a-z]:[\\/]+)?users[\\/]+gary(?=[\\/]|\s|$)", text, re.I):
        return ["author-specific local path"]
    return []


def inspect_pdf(pdf, mode="submission"):
    import fitz
    camera_ready = mode == "camera_ready"
    with fitz.open(pdf) as doc:
        headings, identifying, local_paths, acks = [], [], [], []
        first_page_text, margin_numbers = "", 0
        for i, page in enumerate(doc):
            text = page.get_text()
            if i == 0:
                first_page_text = text
            for term in identifying_text_hits(text):
                identifying.append({"page": i + 1, "term": term})
            for term in local_path_hits(text):
                local_paths.append({"page": i + 1, "term": term})
            # A prose mention of references is not the References heading.
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    content = "".join(span["text"] for span in line["spans"]).strip()
                    if content == "References":
                        headings.append((i + 1, line["bbox"][1]))
                    elif content == ACK_HEADING:
                        acks.append((i + 1, line["bbox"][1]))
                    elif i == 0 and content.isdigit() and line["bbox"][2] <= LINE_NUMBER_MARGIN_X:
                        margin_numbers += 1
        metadata = {key: value for key, value in doc.metadata.items()
                    if key in ("author", "subject", "keywords") and value}
        refs, y = headings[0] if len(headings) == 1 else (None, None)
        main_pages = refs - 1 if refs and y < 95 else refs
        result = {"total_pages": len(doc), "references_start_page": refs,
                  "references_heading_y": y, "main_content_pages": main_pages,
                  "identifying_terms_found": identifying, "identifying_metadata": metadata,
                  "references_headings": headings}
        if camera_ready:
            # The main text ends at whichever unnumbered back-matter heading comes first.
            end = min(([acks[0]] if len(acks) == 1 else []) + ([(refs, y)] if refs else []),
                      default=None)
            result.update({
                "mode": mode, "acknowledgments_headings": acks,
                "main_content_pages": (end[0] - 1 if end[1] < 95 else end[0]) if end else None,
                "main_text_end": end, "local_paths_found": local_paths,
                "first_page_margin_line_numbers": margin_numbers,
                "first_page_text": first_page_text})
        return result


def authors_record(paper):
    path = Path(paper) / AUTHORS_RECORD
    if not path.is_file():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), assets.sha256(path)


def authors_present(text, inspection, record):
    """Fail while placeholders remain; require recorded names on page 1."""
    problems = []
    body = assets.uncomment(text)
    for name in AUTHOR_PLACEHOLDERS:
        if re.search(r"\\" + name + r"(?![A-Za-z])", body):
            problems.append("placeholder \\%s is still present" % name)
    source_names = assets.author_names(text)
    if not source_names:
        problems.append("\\author block has no author names")
    page = normalized(inspection.get("first_page_text", ""))
    for marker in ("anonymous author", "authors pending"):
        if marker in page:
            problems.append("first page shows '%s'" % marker)
    if record is None:
        problems.append("missing " + AUTHORS_RECORD)
    else:
        names = [entry.get("name", "") if isinstance(entry, dict) else str(entry)
                 for entry in record.get("authors", [])]
        if record.get("status") != "confirmed":
            problems.append("%s status is %r, not 'confirmed'" % (AUTHORS_RECORD, record.get("status")))
        if not names or not all(name.strip() for name in names):
            problems.append(AUTHORS_RECORD + " lists no authors")
        elif [normalized(n) for n in names] != [normalized(n) for n in source_names]:
            problems.append("recorded authors differ from the \\author block (names or order)")
        for name in names:
            if name.strip() and normalized(name) not in page:
                problems.append("author not on first page: " + name)
    return problems


def camera_ready_template(stage, text, inspection):
    problems = []
    body = assets.uncomment(text)
    if not re.search(r"\\usepackage\s*\{genai4health_2026\}", body):
        problems.append("source does not load genai4health_2026 without options")
    if re.search(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\]\s*)?\{[^}]*\bneurips_2026\b", body):
        problems.append("source loads neurips_2026 directly")
    if re.search(r"\\PassOptionsToPackage\s*\{[^}]*\}\s*\{neurips_2026\}", body):
        problems.append("source passes options to neurips_2026")
    if not re.search(r"\\maketitle\b", body):
        problems.append("\\maketitle is missing")
    for name, digest in OFFICIAL_STYLES.items():
        path = Path(stage) / name
        if not path.is_file():
            problems.append("missing " + name)
        elif assets.sha256(path) != digest:
            problems.append(name + " differs from the official kit")
    page = normalized(inspection.get("first_page_text", ""))
    if normalized(WORKSHOP_NOTICE) not in page:
        problems.append("workshop notice missing from page 1")
    for marker in SUBMISSION_NOTICE:
        if normalized(marker) in page:
            problems.append("submission notice '%s' on page 1" % marker)
    if inspection.get("first_page_margin_line_numbers"):
        problems.append("review line numbers on page 1")
    return problems


def camera_ready_pdf_checks(stage, text, inspection, paper):
    record, _ = authors_record(paper)
    return {"authors_present": authors_present(text, inspection, record),
            "no_local_paths": [hit["term"] for hit in inspection.get("local_paths_found", [])],
            "camera_ready_template": camera_ready_template(stage, text, inspection)}


def command_gate(script, arguments, cwd):
    result = subprocess.run([PY, str(Path(__file__).with_name(script)), *arguments],
                            cwd=cwd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print("\n".join(output.splitlines()[-12:]))
    return result.returncode == 0


def build(out_zip, allow_ph=False, mark_uploaded=False, *, paper_dir=None,
          staging_root=None, pdf_out=None, docx_out=None, manifest_out=None,
          expected_docx_sha256=None, stats_dir=None, citation_record=None, review_file=None,
          camera_ready=False):
    if allow_ph or mark_uploaded:
        print("Release bypass/upload marking is unsupported; validate and sync explicitly.")
        return 1
    mode = "camera_ready" if camera_ready else "submission"
    paper = Path(paper_dir or PAPER).resolve()
    out_zip = Path(out_zip).resolve()
    citation_record = Path(citation_record).resolve() if citation_record else None
    stats_dir = Path(stats_dir).resolve() if stats_dir else None
    pdf_out = Path(pdf_out) if pdf_out else paper / "main_submission.pdf"
    docx_out = Path(docx_out) if docx_out else paper / "main_submission.docx"
    manifest_out = Path(manifest_out) if manifest_out else out_zip.with_suffix(".release.json")
    work = assets.unique_work(staging_root)
    stage = work / "stage"
    stage.mkdir()
    report = {"built": datetime.now(timezone.utc).isoformat(), "ALL_PASS": False,
              "mode": mode, "stage": str(stage), "checks": {}}
    checks = report["checks"]
    try:
        snapshot = assets.input_hashes(paper)
        review_receipt = assets.stage_numeric_review(paper, work, review_file)
        report["numeric_review"] = review_receipt
        destinations = [path.resolve() for path in
                        (out_zip, pdf_out, docx_out, manifest_out, docx_out.with_suffix(".docx.provenance.json"))]
        if len(destinations) != len(set(destinations)):
            raise ValueError("release destinations must be distinct")
        protected_inputs = {assets.safe_path(paper, rel) for rel in snapshot}
        if camera_ready:
            # Release-control input: checked against page 1, never shipped in the ZIP.
            protected_inputs.add((paper / AUTHORS_RECORD).resolve())
        if review_receipt is not None:
            protected_inputs.add(Path(review_receipt["source"]).resolve())
        if citation_record is not None:
            protected_inputs.add(citation_record)
        if stats_dir is not None and stats_dir.is_dir():
            protected_inputs.update(path.resolve() for path in stats_dir.iterdir() if path.is_file())
        if set(destinations) & protected_inputs:
            raise ValueError("release destination would overwrite a source input (including numeric review input)")
        report["input_hashes"] = snapshot
        # Compile exactly the inputs identified recursively, not all historical assets.
        staged_snapshot = assets.copy_snapshot(paper, stage, snapshot, rename_main=True)
        source_map = {}
        for rel, digest in snapshot.items():
            remote = "main.tex" if rel == "main_submission.tex" else rel
            source_map[remote] = {"source": rel, "sha256": digest,
                                  "kind": "compiled_input"}
        asset_manifest = assets.asset_inventory(stage)
        report["assets"] = asset_manifest
        checks["immutable_figure_inputs"] = asset_manifest["ALL_PASS"]
        text, _ = assets.source_tree(stage, "main.tex")
        checks["no_placeholders"] = not re.search(r"\\ph\b", assets.expanded_body(text))
        checks["all_graphics_present"] = all(assets.resolve_graphic(stage, name).is_file()
                                            for name in assets.graphics(text))
        gate_args = ["--paper-dir", str(stage)]
        if stats_dir:
            gate_args += ["--stats-dir", str(stats_dir)]
        checks["manuscript"] = command_gate("check_manuscript.py", gate_args, stage)
        numeric_report = work / "numeric_coverage.json"
        report["evidence_paths"] = {"numbers": str(numeric_report)}
        numeric_args = gate_args + ["--report", str(numeric_report)]
        if review_receipt:
            numeric_args += ["--review-file", review_receipt["archived"]]
        checks["numeric_evidence"] = command_gate(
            "p15_verify_numbers.py", numeric_args, stage)
        checks["numeric_review_input"] = False
        assets.verify_numeric_review(review_receipt)
        if checks["numeric_evidence"]:
            numeric_data = json.loads(numeric_report.read_text(encoding="utf-8"))
            assets.verify_numeric_review(review_receipt, numeric_data.get("review_sha256"))
            checks["numeric_evidence"] = (
                numeric_data.get("ALL_PASS") is True and numeric_data.get("checked_auc", 0) > 0
                and bool(numeric_data.get("items"))
                and numeric_data.get("input_hashes") == assets.input_hashes(stage))
        checks["numeric_review_input"] = True
        citation_report = work / "citation_validation.json"
        report["evidence_paths"]["citations"] = str(citation_report)
        citation_args = ["--paper-dir", str(stage), "--report", str(citation_report)]
        if citation_record:
            citation_args += ["--record", str(citation_record)]
        checks["citation_metadata"] = command_gate("verify_citations.py", citation_args, stage)
        if checks["citation_metadata"]:
            citation_data = json.loads(citation_report.read_text(encoding="utf-8"))
            checks["citation_metadata"] = (
                citation_data.get("ALL_PASS") is True and bool(citation_data.get("items"))
                and all(item.get("status") == "matched" for item in citation_data["items"])
                and citation_data.get("source_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
                and citation_data.get("bib_sha256") == hashlib.sha256(
                    (stage / "references.bib").read_text(encoding="utf-8").encode("utf-8")).hexdigest())
        if not all(checks.values()):
            raise ValueError("source/evidence gate failed; no deliverable replaced")
        result = subprocess.run(
            [TECTONIC, "-X", "compile", "main.tex", "--keep-logs", "--keep-intermediates"],
            cwd=stage, capture_output=True, text=True)
        (work / "compiler.txt").write_text((result.stdout or "") + (result.stderr or ""),
                                          encoding="utf-8")
        pdf = stage / "main.pdf"
        checks["compiles_standalone"] = result.returncode == 0 and pdf.is_file()
        if not checks["compiles_standalone"]:
            raise ValueError("standalone compiler failed")
        inspection = inspect_pdf(pdf, mode) if camera_ready else inspect_pdf(pdf)
        report.update({key: value for key, value in inspection.items() if key != "first_page_text"})
        limit = CAMERA_READY_PAGE_LIMIT if camera_ready else PAGE_LIMIT
        report["page_limit"] = limit
        checks["page_limit"] = (inspection["main_content_pages"] is not None
                                and inspection["main_content_pages"] <= limit)
        if camera_ready:
            record, record_hash = authors_record(paper)
            report["authors_record"] = {"path": str(paper / AUTHORS_RECORD), "sha256": record_hash}
            problems = camera_ready_pdf_checks(stage, text, inspection, paper)
            report["camera_ready_problems"] = problems
            for name, found in problems.items():
                checks[name] = not found
        else:
            checks["anonymous"] = not (inspection["identifying_terms_found"]
                                       or inspection["identifying_metadata"])
        log_path = stage / "main.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        checks["no_undefined_refs"] = log_path.is_file() and not re.search(
            r"(?:citation|reference).*undefined|undefined (?:citation|reference)|"
            r"There were undefined|Label\(s\) may have changed", log, re.I)
        if not all(checks.values()):
            raise ValueError("PDF validation failed")
        receipt = docx_out.with_suffix(".docx.provenance.json")
        old_word_hash = assets.check_word_conflict(docx_out, expected_docx_sha256, receipt)
        docx = work / "main_submission.docx"
        checks["docx_generated"] = command_gate(
            "make_docx.py", ["--paper-dir", str(stage), "--aux", str(stage / "main.aux"),
                             "--out", str(docx), "--staging-root", str(work)], stage)
        checks["docx_complete"] = checks["docx_generated"] and command_gate(
            "check_docx.py", ["--paper-dir", str(stage), "--aux", str(stage / "main.aux"),
                              "--docx", str(docx)], stage)
        if not all(checks.values()):
            raise ValueError("Word validation failed")
        assets.assert_unchanged(stage, staged_snapshot)
        candidate_zip = work / "release.zip"
        with zipfile.ZipFile(candidate_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in sorted(source_map):
                archive.write(assets.safe_path(stage, rel), rel)
            archive.write(pdf, "main.pdf")
            archive.writestr("README_OVERLEAF.txt",
                             "Set main.tex as root. XeLaTeX/pdfLaTeX and BibTeX.\n"
                             "See the release manifest for evidence coverage and input hashes.\n"
                             "Hash identity and citation metadata are not claim validation.\n")
        with zipfile.ZipFile(candidate_zip) as archive:
            for rel, digest in staged_snapshot.items():
                if hashlib.sha256(archive.read(rel)).hexdigest() != digest:
                    raise ValueError("archived input differs from captured snapshot: " + rel)
        # Frozen attachments are separately checked, never collected by a second glob.
        attachments = {"main_editable.docx": {
            "source": "main_submission.docx", "sha256": assets.sha256(docx),
            "kind": "word_attachment", "check": "check_docx.py"}}
        docx_receipt = work / "docx.provenance.json"
        generated_receipt = docx.with_suffix(".docx.provenance.json")
        receipt_data = json.loads(generated_receipt.read_text(encoding="utf-8")) if generated_receipt.exists() else {}
        assets.write_json(docx_receipt, {**receipt_data, "docx_sha256": assets.sha256(docx),
                                        "inputs": snapshot, "aux_sha256": assets.sha256(stage / "main.aux"),
                                        "aux_path": str(stage / "main.aux")})
        assets.assert_unchanged(paper, snapshot)
        assets.assert_unchanged(stage, staged_snapshot)
        assets.verify_numeric_review(review_receipt)
        if camera_ready and authors_record(paper)[1] != report["authors_record"]["sha256"]:
            raise ValueError("camera-ready author record changed during validation; rebuild required")
        if {key: value["sha256"] for key, value in source_map.items()} != staged_snapshot:
            raise ValueError("staged source manifest differs from captured snapshot")
        if (assets.sha256(docx_out) if docx_out.exists() else None) != old_word_hash:
            raise ValueError("Word copy changed during build; preserving collaborator edits")
        report.update({"ALL_PASS": True, "source_files": source_map,
                       "attachments": attachments,
                       "assets": asset_manifest,
                       "aux": {"path": str(stage / "main.aux"),
                               "sha256": assets.sha256(stage / "main.aux")},
                       "limits": ["Metadata existence/title matching is not citation claim support.",
                                  "Figure identities include baseline-pinned external inputs; hashes "
                                  "do not validate plotted values.",
                                  ("Author checks compare the reviewed author record with selectable "
                                   "first-page text; they do not verify affiliations or e-mail addresses."
                                   if camera_ready else
                                   "Anonymity scanning covers selectable PDF text and selected metadata; "
                                   "it is not an OCR/pixel-identity audit.")],
                       "evidence": {"numbers": json.loads(numeric_report.read_text(encoding="utf-8")),
                                    "citations": json.loads(citation_report.read_text(encoding="utf-8"))},
                       "artifacts": {
                           "zip": {"path": str(out_zip), "sha256": assets.sha256(candidate_zip)},
                           "pdf": {"path": str(pdf_out), "sha256": assets.sha256(pdf)},
                           "docx": {"path": str(docx_out), "sha256": assets.sha256(docx)}}})
        candidate_manifest = work / "release.json"
        assets.write_json(candidate_manifest, report)
        assets.promote([(candidate_zip, out_zip), (pdf, pdf_out), (docx, docx_out),
                        (docx_receipt, receipt), (candidate_manifest, manifest_out)],
                       expected_current={docx_out: old_word_hash})
    except Exception as exc:
        report["ALL_PASS"] = False
        report["error"] = str(exc)
        print("FAIL:", exc)
    assets.write_json(work / "validation.json", report)
    for name, passed in checks.items():
        print("  %s: %s" % (name, "PASS" if passed else "FAIL"))
    print("  evidence:", work / "validation.json")
    print("  ALL_PASS =", report["ALL_PASS"])
    return 0 if report["ALL_PASS"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="ZIP path (default depends on --camera-ready)")
    parser.add_argument("--camera-ready", action="store_true",
                        help="camera-ready gates: authors present, 10 main pages, workshop template")
    parser.add_argument("--paper-dir", default=PAPER)
    parser.add_argument("--staging-root")
    parser.add_argument("--pdf-out")
    parser.add_argument("--docx-out")
    parser.add_argument("--manifest-out")
    parser.add_argument("--expected-docx-sha256")
    parser.add_argument("--stats-dir")
    parser.add_argument("--citation-record")
    parser.add_argument("--review-file", help="Numeric binding/review input; archived as private QA evidence")
    args = vars(parser.parse_args())
    out = args.pop("out") or (CAMERA_READY_DEFAULT_OUT if args["camera_ready"] else DEFAULT_OUT)
    sys.exit(build(out, **args))
