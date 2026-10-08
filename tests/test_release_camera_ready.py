"""Camera-ready release mode: authors gate, 10-page limit, template staging, Word authors."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from autopilot import p13_build_zip as release
from autopilot import release_assets as assets


def passing_gate_report(script, args, cwd):
    if "--report" not in args:
        return
    text, _ = assets.source_tree(cwd)
    if script == "p15_verify_numbers.py":
        report = {"ALL_PASS": True, "checked_auc": 1, "items": [{"status": "verified"}],
                  "input_hashes": assets.input_hashes(cwd), "review_sha256": None}
    else:
        report = {"ALL_PASS": True, "items": [{"status": "matched"}],
                  "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "bib_sha256": hashlib.sha256((Path(cwd) / "references.bib").read_text(
                      encoding="utf-8").encode("utf-8")).hexdigest()}
    assets.write_json(args[args.index("--report") + 1], report)


REPO_PAPER = Path(__file__).resolve().parents[1] / "paper" / "genai4health2026"
AUTHOR_BLOCK = (r"\author{Ann B. Smith\thanks{Corresponding author.} \\ Dept, Inst \\ \texttt{ann@example.org}"
                r" \And Jos\'e Doe \\ Inst \AND Third Person \\ Inst}")
NAMES = ["Ann B. Smith", "Jose Doe", "Third Person"]
PAGE_ONE = ("Title\nAnn B. Smith\u2217 Dept, Inst ann@example.org Jos\u00e9 Doe Inst Third Person Inst\n"
            "GenAI for Health Workshop @ NeurIPS 2026, Sydney.")


def official_styles(folder):
    for name in release.OFFICIAL_STYLES:
        shutil.copyfile(REPO_PAPER / name, folder / name)


@pytest.fixture
def cr_paper(tmp_path):
    folder = tmp_path / "paper"
    (folder / "auto").mkdir(parents=True)
    (folder / "main_submission.tex").write_text(
        r"\documentclass{article}\usepackage{genai4health_2026}\input{auto/auto_numbers}"
        r"\title{T}" + AUTHOR_BLOCK +
        r"\begin{document}\maketitle\AUCRandomEpFifty\cite{x}\begin{ack}Funding.\end{ack}\end{document}",
        encoding="utf-8")
    (folder / "auto" / "auto_numbers.tex").write_text(
        r"\newcommand{\AUCRandomEpFifty}{0.8641}", encoding="utf-8")
    (folder / "references.bib").write_text("@article{x,title={Example},year={2020}}", encoding="utf-8")
    official_styles(folder)
    assets.write_json(folder / release.AUTHORS_RECORD,
                      {"version": 1, "status": "confirmed", "authors": [{"name": n} for n in NAMES]})
    return folder


def test_local_style_chain_is_staged(cr_paper):
    snapshot = assets.input_hashes(cr_paper)
    assert {"genai4health_2026.sty", "neurips_2026.sty"} <= set(snapshot)
    # A style that is present but not loaded is not part of the release tree.
    (cr_paper / "unused.sty").write_text("% unused")
    assert "unused.sty" not in assets.input_hashes(cr_paper)


def test_submission_inputs_keep_only_neurips_style(tmp_path):
    folder = tmp_path / "paper"
    folder.mkdir()
    (folder / "main_submission.tex").write_text(r"\usepackage{neurips_2026}\begin{document}\end{document}")
    (folder / "references.bib").write_text("")
    official_styles(folder)
    assert set(assets.input_hashes(folder)) == {"main_submission.tex", "references.bib", "neurips_2026.sty"}


def test_author_names_drop_affiliations_thanks_and_placeholders(cr_paper):
    source = (cr_paper / "main_submission.tex").read_text(encoding="utf-8")
    assert assets.author_names(source) == NAMES
    assert assets.author_entries(source)[1] == r"Jos\'e Doe"
    assert assets.author_names(r"\author{\CRAuthorsPending}") == []
    assert assets.author_names(r"\author{}") == []
    assert assets.author_names(r"% \author{Commented Out}") == []


def test_authors_gate_fails_on_placeholder_and_unconfirmed_record(cr_paper):
    source = (cr_paper / "main_submission.tex").read_text(encoding="utf-8")
    record = json.loads((cr_paper / release.AUTHORS_RECORD).read_text())
    assert release.authors_present(source, {"first_page_text": PAGE_ONE}, record) == []
    pending = source.replace(AUTHOR_BLOCK, r"\newcommand{\CRAuthorsPending}{AUTHORS PENDING}\author{\CRAuthorsPending}")
    problems = release.authors_present(pending, {"first_page_text": "AUTHORS PENDING"}, record)
    assert any("CRAuthorsPending" in p for p in problems)
    assert any("authors pending" in p for p in problems)
    assert release.authors_present(source, {"first_page_text": PAGE_ONE}, {**record, "status": "pending"})
    assert release.authors_present(source, {"first_page_text": PAGE_ONE}, None)
    reordered = {**record, "authors": list(reversed(record["authors"]))}
    assert any("differ" in p for p in release.authors_present(source, {"first_page_text": PAGE_ONE}, reordered))
    missing = PAGE_ONE.replace("Third Person", "")
    assert any("Third Person" in p for p in release.authors_present(source, {"first_page_text": missing}, record))
    assert release.authors_present(source, {"first_page_text": PAGE_ONE + " Anonymous Author(s)"}, record)


def test_template_gate_requires_official_styles_notice_and_no_line_numbers(cr_paper):
    source = (cr_paper / "main_submission.tex").read_text(encoding="utf-8")
    page = {"first_page_text": PAGE_ONE, "first_page_margin_line_numbers": 0}
    assert release.camera_ready_template(cr_paper, source, page) == []
    assert release.camera_ready_template(cr_paper, source.replace("genai4health_2026", "neurips_2026"), page)
    assert release.camera_ready_template(
        cr_paper, source + r"\PassOptionsToPackage{preprint}{neurips_2026}", page)
    assert release.camera_ready_template(cr_paper, source, {**page, "first_page_margin_line_numbers": 3})
    submitted = {**page, "first_page_text": "Submitted to 40th Conference. Do not distribute."}
    assert len(release.camera_ready_template(cr_paper, source, submitted)) >= 2
    (cr_paper / "neurips_2026.sty").write_text("% edited style")
    assert any("official" in p for p in release.camera_ready_template(cr_paper, source, page))


def test_inspect_pdf_ends_main_text_at_acknowledgments_and_counts_line_numbers(tmp_path):
    import fitz
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as document:
        first = document.new_page()
        for i in range(5):
            first.insert_text((94, 100 + 12 * i), str(i + 1))
        first.insert_text((108, 700), "GenAI for Health Workshop @ NeurIPS 2026, Sydney.")
        document.new_page().insert_text((108, 300), "Body text")
        document.new_page().insert_text((108, 400), "Acknowledgments and Disclosure of Funding")
        document.new_page().insert_text((108, 80), "References")
        document.save(pdf)
    submission = release.inspect_pdf(pdf)
    camera = release.inspect_pdf(pdf, "camera_ready")
    assert submission["main_content_pages"] == 3
    assert "first_page_text" not in submission
    assert camera["main_content_pages"] == 3
    assert camera["main_text_end"][0] == 3
    assert camera["first_page_margin_line_numbers"] == 5


def fake_compile(*args, **kwargs):
    stage = Path(kwargs["cwd"])
    (stage / "main.pdf").write_bytes(b"checked-new-pdf")
    (stage / "main.aux").write_text("")
    (stage / "main.log").write_text("complete")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def camera_gate(script, args, cwd):
    passing_gate_report(script, args, cwd)
    if script == "make_docx.py":
        Path(args[args.index("--out") + 1]).write_bytes(b"checked-new-word")
    return True


@pytest.mark.parametrize("pages,authors,passed", [(10, True, True), (11, True, False), (9, False, False)])
def test_camera_ready_build_gates(cr_paper, tmp_path, monkeypatch, pages, authors, passed):
    out = tmp_path / "release.zip"
    word = cr_paper / "main_submission.docx"
    word.write_bytes(b"reviewed-old-word")
    if not authors:
        assets.write_json(cr_paper / release.AUTHORS_RECORD, {"version": 1, "status": "pending_user_input",
                                                               "authors": []})
    monkeypatch.setattr(release, "command_gate", camera_gate)
    monkeypatch.setattr(release.subprocess, "run", fake_compile)
    monkeypatch.setattr(release, "inspect_pdf", lambda *a: {
        "main_content_pages": pages, "identifying_terms_found": [{"page": 1, "term": "author name"}],
        "identifying_metadata": {"author": "Ann B. Smith"}, "first_page_text": PAGE_ONE,
        "first_page_margin_line_numbers": 0, "local_paths_found": []})
    code = release.build(out, paper_dir=cr_paper, staging_root=tmp_path / "work",
                         expected_docx_sha256=assets.sha256(word), camera_ready=True)
    assert (code == 0) is passed
    report = json.loads(next((tmp_path / "work").glob("release-*/validation.json")).read_text())
    assert report["mode"] == "camera_ready" and report["page_limit"] == 10
    assert "anonymous" not in report["checks"]
    assert report["checks"]["page_limit"] is (pages <= 10)
    assert report["checks"]["authors_present"] is authors
    assert "first_page_text" not in report
    if passed:
        manifest = json.loads(out.with_suffix(".release.json").read_text())
        assert manifest["mode"] == "camera_ready"
        assert manifest["authors_record"]["sha256"] == assets.sha256(cr_paper / release.AUTHORS_RECORD)
        assert {"genai4health_2026.sty", "neurips_2026.sty"} <= set(manifest["source_files"])
        with zipfile.ZipFile(out) as archive:
            assert release.AUTHORS_RECORD not in archive.namelist()
    else:
        assert word.read_bytes() == b"reviewed-old-word"


def test_submission_mode_still_enforces_anonymity_and_nine_pages(cr_paper, tmp_path, monkeypatch):
    out = tmp_path / "release.zip"
    monkeypatch.setattr(release, "command_gate", camera_gate)
    monkeypatch.setattr(release.subprocess, "run", fake_compile)
    monkeypatch.setattr(release, "inspect_pdf", lambda *a: {
        "main_content_pages": 10, "identifying_terms_found": [], "identifying_metadata": {}})
    assert release.build(out, paper_dir=cr_paper, staging_root=tmp_path / "work") == 1
    report = json.loads(next((tmp_path / "work").glob("release-*/validation.json")).read_text())
    assert report["mode"] == "submission"
    assert report["checks"]["page_limit"] is False and report["checks"]["anonymous"] is True
    assert "authors_present" not in report["checks"]


def test_sync_requires_mode_specific_gates():
    from scripts import sync_overleaf as sync
    assert sync.FILE_MAP["genai4health_2026.sty"] == "genai4health_2026.sty"
    assert sync.managed_remote("genai4health_2026.sty")
    base = {"ALL_PASS": True, "checks": {name: True for name in (
        "immutable_figure_inputs", "no_placeholders", "all_graphics_present", "manuscript", "numeric_evidence",
        "numeric_review_input", "citation_metadata", "compiles_standalone", "page_limit",
        "no_undefined_refs", "docx_generated", "docx_complete")}}
    with pytest.raises(ValueError, match="gates are missing"):
        sync.verify_local_release(Path("."), {**base, "mode": "camera_ready"})
    with pytest.raises(ValueError, match="gates are missing"):
        sync.verify_local_release(Path("."), base)
    with pytest.raises(ValueError, match="unknown release mode"):
        sync.verify_local_release(Path("."), {**base, "mode": "draft"})
    camera = {**base, "mode": "camera_ready",
              "checks": {**base["checks"], "authors_present": True, "no_local_paths": True,
                         "camera_ready_template": True}}
    with pytest.raises(ValueError, match="empty or incomplete validated source tree"):
        sync.verify_local_release(Path("."), camera)


def test_refresh_forwards_camera_ready_flag(monkeypatch):
    import sys
    from autopilot import refresh_all
    commands = []
    monkeypatch.setattr(sys, "argv", ["refresh_all.py", "--fast", "--camera-ready"])
    monkeypatch.setattr(refresh_all, "run", lambda label, command, **kwargs: commands.append(command) or 0)
    assert refresh_all.main() == 0
    build = next(command for command in commands if str(command[1]).endswith("p13_build_zip.py"))
    assert "--camera-ready" in build
    assert build[build.index("--out") + 1].endswith("_camera_ready.zip")


def test_camera_ready_macros_are_current_and_bound():
    from autopilot import make_cr_numbers as cr
    if not all((cr.REPO / path).exists() for path in cr.SOURCES.values()):
        pytest.skip("camera-ready evidence not installed")
    values, macros, hashes = cr.build()
    assert cr.OUT.read_text(encoding="utf-8") == cr.render(values, hashes)
    reviews = json.loads(cr.REVIEWS.read_text(encoding="utf-8"))
    assert {name: spec for name, spec in reviews["macros"].items() if name.startswith("CR")} == macros
    for name in cr.OWNED_SOURCES:
        assert reviews["sources"][name]["sha256"] == hashes[name]


def test_fp32_table_p_display_must_match_evidence(tmp_path):
    from autopilot import numeric_bindings as binding
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "p3b_fp32.json").write_text(json.dumps({"rows": [
        {"arm": "random", "epoch": 100, "auc_fp16": 0.874581, "auc_fp32": 0.874485,
         "delta_fp32_minus_fp16": -0.000096, "delong_p": 8.027e-05},
        {"arm": "random", "epoch": 75, "auc_fp16": 0.872302, "auc_fp32": 0.872302,
         "delta_fp32_minus_fp16": -0.000001, "delong_p": 0.962}]}))
    evidence = binding.Evidence(tmp_path, stats)
    small = r"\textsc{random} & 100 & 0.874581 & 0.874485 & -0.000096 & %s \\" + "\n"
    large = r"\textsc{random} & 75 & 0.872302 & 0.872302 & -0.000001 & %s \\" + "\n"
    for display in (r"$<$0.001", "0.000"):  # current generator and the v9 rounding
        cells, errors = binding.table_bindings("auto/table_fp32.tex", (small % display) + (large % "0.962"), evidence)
        assert not errors and all(x["status"] == "verified" for x in cells.values()), errors
    _, errors = binding.table_bindings("auto/table_fp32.tex", (small % "0.000") + (large % r"$<$0.001"), evidence)
    assert any("'<' display" in error for error in errors)
    cells, _ = binding.table_bindings("auto/table_fp32.tex", (small % "0.010") + (large % "0.962"), evidence)
    assert any(x["status"] == "mismatch" for x in cells.values())


def test_word_author_metadata_matches_source_names(tmp_path):
    from autopilot import check_docx, make_docx
    paper = tmp_path / "paper"
    (paper / "auto").mkdir(parents=True)
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\AUCRandomEpFifty}{0.8641}")
    (paper / "references.bib").write_text(
        "@article{a,\n title={Deep learning for retinal OCT classification},\n author={Smith, A},\n"
        " year={2020},\n doi={10.1234/a},\n}\n", encoding="utf-8")
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\title{T}" + AUTHOR_BLOCK +
        r"\begin{document}\maketitle Result \AUCRandomEpFifty. \cite{a}"
        r"\bibliography{references}\end{document}", encoding="utf-8")
    aux = paper / "main_submission.aux"
    aux.write_text("")
    out = tmp_path / "authors.docx"
    assert make_docx.build(paper, out, aux, staging_root=tmp_path / "work") == 0
    result = check_docx.check(out, paper, aux)
    assert result["ALL_PASS"], result["errors"]
    with zipfile.ZipFile(out) as archive:
        core = archive.read("docProps/core.xml").decode("utf-8")
        entries = {name: archive.read(name) for name in archive.namelist()}
    assert "Ann B. Smith; Jos" in core and "Dept" not in core and "example.org" not in core
    # Any other creator (for example an editor's name) is rejected.
    entries["docProps/core.xml"] = core.replace("Third Person", "Someone Else").encode("utf-8")
    damaged = tmp_path / "damaged.docx"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    errors = check_docx.check(damaged, paper, aux)["errors"]
    assert any("creator metadata differs" in error for error in errors)


def test_anonymous_word_still_rejects_any_creator(tmp_path):
    from autopilot import check_docx, make_docx
    paper = tmp_path / "paper"
    (paper / "auto").mkdir(parents=True)
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\AUCRandomEpFifty}{0.8641}")
    (paper / "references.bib").write_text(
        "@article{a,\n title={Deep learning for retinal OCT classification},\n author={Smith, A},\n"
        " year={2020},\n doi={10.1234/a},\n}\n", encoding="utf-8")
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\title{T}\author{}"
        r"\begin{document}\maketitle Result \AUCRandomEpFifty. \cite{a}"
        r"\bibliography{references}\end{document}", encoding="utf-8")
    aux = paper / "main_submission.aux"
    aux.write_text("")
    out = tmp_path / "anonymous.docx"
    assert make_docx.build(paper, out, aux, staging_root=tmp_path / "work") == 0
    assert check_docx.check(out, paper, aux)["ALL_PASS"]
    with zipfile.ZipFile(out) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    core = entries["docProps/core.xml"].decode("utf-8")
    if "<dc:creator>" not in core:
        core = core.replace("</cp:coreProperties>", "<dc:creator>Ann B. Smith</dc:creator></cp:coreProperties>")
    else:
        core = core.replace("<dc:creator>", "<dc:creator>Ann B. Smith", 1)
    entries["docProps/core.xml"] = core.encode("utf-8")
    damaged = tmp_path / "named.docx"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    errors = check_docx.check(damaged, paper, aux)["errors"]
    assert "nonempty creator metadata" in errors
