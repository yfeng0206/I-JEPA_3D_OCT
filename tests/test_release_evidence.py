import json
from pathlib import Path

import pytest

from autopilot import p15_verify_numbers as numbers
from autopilot import verify_citations as citations
from autopilot import check_manuscript
from autopilot import check_docx
from autopilot import make_docx


@pytest.fixture
def evidence_paper(tmp_path):
    paper = tmp_path / "paper"
    (paper / "auto").mkdir(parents=True)
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}"
        r"\begin{document}\AUCRandomEpFifty\cite{a}\end{document}", encoding="utf-8")
    (paper / "auto" / "auto_numbers.tex").write_text(
        r"\newcommand{\AUCRandomEpFifty}{0.8641}", encoding="utf-8")
    (paper / "references.bib").write_text(
        "@article{a,\n title={Deep learning for retinal OCT classification},\n"
        " author={Smith, A},\n year={2020},\n doi={10.1234/a},\n}\n", encoding="utf-8")
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "p1b_full_inventory.json").write_text(json.dumps({"records": [
        {"arm": "random", "epoch": 50, "precision": "fp16", "status": "primary", "auc": 0.8641}]}))
    (stats / "p1c_stats.json").write_text(json.dumps({"table": [
        {"key": "random@ep50@fp16", "arm": "random", "epoch": 50, "precision": "fp16",
         "auc": 0.8641, "ci95_lo": 0.85, "ci95_hi": 0.88}], "contrasts": []}))
    return paper, stats


def test_number_required_coverage_is_nonempty(evidence_paper):
    paper, stats = evidence_paper
    assert numbers.audit(paper, stats)["ALL_PASS"]
    (paper / "auto" / "auto_numbers.tex").write_text("")
    result = numbers.audit(paper, stats)
    assert not result["ALL_PASS"]
    assert result["checked_auc"] == 0


@pytest.mark.parametrize("value,passed", [(r"\mathbf{0.8641}", True),
                                        (r"\unknown{0.8641}", False), ("---", False),
                                        ("0.0000", False)])
def test_wrapped_and_absent_required_values_are_never_skipped(evidence_paper, value, passed):
    paper, stats = evidence_paper
    (paper / "auto" / "auto_numbers.tex").write_text(
        r"\newcommand{\AUCRandomEpFifty}{" + value + "}")
    result = numbers.audit(paper, stats)
    assert result["ALL_PASS"] is passed


def test_true_zero_matches_zero_statistic(evidence_paper):
    paper, stats = evidence_paper
    path = stats / "p1c_stats.json"
    data = json.loads(path.read_text())
    data["table"][0]["auc"] = 0
    path.write_text(json.dumps(data))
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\AUCRandomEpFifty}{0.0000}")
    assert numbers.audit(paper, stats)["ALL_PASS"]


def test_unbound_table_literals_and_macros_are_explicit(evidence_paper):
    paper, stats = evidence_paper
    with (paper / "main_submission.tex").open("a") as stream:
        stream.write(r"\input{auto/table}")
    (paper / "auto" / "table.tex").write_text(r"\newcommand{\Mystery}{0.9123}\Mystery 123")
    result = numbers.audit(paper, stats)
    assert not result["ALL_PASS"]
    assert any(item["id"] == "Mystery" and item["status"] == "unresolved" for item in result["items"])
    assert any(item["kind"] == "literal" and item["value"] == "123" for item in result["items"])


def test_manuscript_gate_follows_placeholder_through_table(evidence_paper):
    paper, stats = evidence_paper
    (paper / "main_submission.tex").write_text(r"\input{auto/auto_numbers}\input{auto/table}\cite{a}")
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\Hidden}{\ph{pending}}")
    (paper / "auto" / "table.tex").write_text(r"\Hidden")
    assert check_manuscript.main(["--paper-dir", str(paper), "--stats-dir", str(stats)]) == 1


def test_citations_missing_empty_and_unresolved_fail(evidence_paper):
    paper, _ = evidence_paper
    bib = (paper / "references.bib").read_text()
    assert not citations.verify("", bib)["ALL_PASS"]
    assert not citations.verify(r"\cite{missing}", bib)["ALL_PASS"]
    assert not citations.verify(r"\cite{a}", bib)["ALL_PASS"]
    assert not citations.verify(r"\cite{a}", bib, expected_keys=["b"])["ALL_PASS"]


def test_citation_strict_title_match_and_persisted_record(evidence_paper, monkeypatch):
    paper, _ = evidence_paper
    bib = (paper / "references.bib").read_text()
    monkeypatch.setattr(citations.time, "sleep", lambda *args: None)
    monkeypatch.setattr(citations, "resolve_doi", lambda *args:
                        "Deep learning for chest X ray classification")
    bad = citations.verify(r"\cite{a}", bib, online=True)
    assert not bad["ALL_PASS"]
    assert bad["items"][0]["status"] == "title_mismatch"
    monkeypatch.setattr(citations, "resolve_doi", lambda *args:
                        "Deep Learning for Retinal OCT Classification")
    good = citations.verify(r"\cite{a}", bib, online=True)
    assert good["ALL_PASS"]
    assert citations.verify(r"\cite{a}", bib, record=good)["ALL_PASS"]
    changed = bib.replace("2020", "2021")
    assert not citations.verify(r"\cite{a}", changed, record=good)["ALL_PASS"]
    assert good["items"][0]["claim_support"] == "not_assessed"


def test_no_identifier_title_search_cannot_skip(evidence_paper, monkeypatch):
    paper, _ = evidence_paper
    bib = (paper / "references.bib").read_text().replace("doi={10.1234/a},", "")
    monkeypatch.setattr(citations, "search_arxiv_title", lambda *args: None)
    monkeypatch.setattr(citations.time, "sleep", lambda *args: None)
    assert not citations.verify(r"\cite{a}", bib, online=True)["ALL_PASS"]


def test_bib_nested_title_parsing():
    assert citations.field("title={{OCT}: a {Retinal} Study},", "title") == "{OCT}: a {Retinal} Study"


def test_versioned_eprint_is_not_discarded_by_unversioned_arxiv_doi():
    body = "doi={10.48550/arXiv.2511.17354},\neprint={2511.17354v4},\n"
    assert citations.identifier(body) == ("arxiv", "2511.17354v4")
    published = "doi={10.1000/published-paper},\neprint={2511.17354v4},\n"
    assert citations.identifier(published) == ("doi", "10.1000/published-paper")


def test_single_tikz_figure_and_title_are_source_derived(evidence_paper, tmp_path):
    if not Path(make_docx.TECTONIC).is_file():
        pytest.skip("project Tectonic compiler is not installed")
    paper, _ = evidence_paper
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\usepackage{tikz}\input{auto/auto_numbers}"
        r"\title{Anatomy-Guided Masking for I-JEPA\\Representation Learning on Retinal OCT}"
        r"\begin{document}\maketitle Result \AUCRandomEpFifty. \cite{a}"
        r"\begin{figure}\begin{tikzpicture}\node[draw] {Pipeline};\end{tikzpicture}"
        r"\caption{Single pipeline}\end{figure}\bibliography{references}\end{document}")
    aux = paper / "main_submission.aux"
    aux.write_text("")
    out = tmp_path / "single-pipeline.docx"
    make_docx.build(paper, out, aux, staging_root=tmp_path / "work")
    result = check_docx.check(out, paper, aux)
    assert result["ALL_PASS"], result["errors"]
    assert result["counts"]["media"] == 1
    assert result["counts"]["figure_captions"] == 1
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(out) as archive:
        text = check_docx.node_text(ET.fromstring(archive.read("word/document.xml")))
    assert "Anatomy-Guided Masking for I-JEPA Representation Learning on Retinal OCT" in text


def test_docx_actual_aux_letters_and_completeness(evidence_paper, tmp_path):
    paper, _ = evidence_paper
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\begin{document}"
        r"See Appendix~\ref{app:test}. Result \AUCRandomEpFifty. \cite{a}"
        r"\begin{table}\caption{Measured AUC}\begin{tabular}{lr}"
        r"Arm & AUC\\Random & \AUCRandomEpFifty\\\end{tabular}\end{table}"
        r"\appendix\section{Test}\label{app:test}\bibliography{references}\end{document}")
    aux = paper / "main_submission.aux"
    aux.write_text(r"\newlabel{app:test}{{B}{2}{Test}{appendix.B}{}}")
    out = tmp_path / "candidate.docx"
    assert make_docx.build(paper, out, aux, staging_root=tmp_path / "work") == 0
    result = check_docx.check(out, paper, aux)
    assert result["ALL_PASS"], result["errors"]
    aux.write_text(r"\newlabel{app:test}{{C}{2}{Test}{appendix.C}{}}")
    assert not check_docx.check(out, paper, aux)["ALL_PASS"]
    assert check_docx.main(["--docx", str(out), "--paper-dir", str(paper), "--aux", str(aux)]) == 1


@pytest.mark.parametrize("damage", ["table", "heading", "bibliography", "macro", "caption", "figure"])
def test_docx_content_damage_fails_without_relying_on_receipt(evidence_paper, tmp_path, damage):
    import zipfile
    import xml.etree.ElementTree as ET
    paper, _ = evidence_paper
    figure = ""
    if damage == "figure":
        import fitz
        image = fitz.Pixmap(fitz.csRGB, (0, 0, 2, 2), False)
        image.clear_with(255)
        image.save(paper / "tiny.png")
        figure = r"\begin{figure}\includegraphics{tiny.png}\caption{Imaging result}\end{figure}"
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\begin{document}"
        r"Result \AUCRandomEpFifty. \cite{a}"
        r"\begin{table}\caption{Measured AUC}\begin{tabular}{lr}"
        r"Arm & AUC\\Random & \AUCRandomEpFifty\\\end{tabular}\end{table}"
        + figure + r"\bibliography{references}\end{document}")
    aux = paper / "main_submission.aux"
    aux.write_text("")
    out = tmp_path / "candidate.docx"
    make_docx.build(paper, out, aux, staging_root=tmp_path / "work")
    with zipfile.ZipFile(out) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(entries["word/document.xml"])
    if damage == "table":
        table = root.find(".//" + check_docx.W + "tbl")
        for node in table.iter():
            if node.tag in (check_docx.W + "t", check_docx.M + "t") and node.text == "0.8641":
                node.text = "0.1234"
    elif damage == "heading":
        for node in root.iter(check_docx.W + "t"):
            if node.text == "References":
                node.text = "Removed heading"
    elif damage == "bibliography":
        for paragraph in root.iter(check_docx.W + "p"):
            style = paragraph.find(check_docx.W + "pPr/" + check_docx.W + "pStyle")
            if style is not None and style.get(check_docx.W + "val") == "Bibliography":
                style.set(check_docx.W + "val", "Normal")
    elif damage == "macro":
        root.find(".//" + check_docx.W + "t").text += r" \AUCRandomEpFifty"
    elif damage == "caption":
        for node in root.iter(check_docx.W + "t"):
            if node.text == "Measured AUC":
                node.text = "Deleted caption words"
    elif damage == "figure":
        name = next(name for name in entries if name.startswith("word/media/"))
        entries[name] = b"wrong image payload"
    entries["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    damaged = tmp_path / "damaged.docx"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    assert not check_docx.check(damaged, paper, aux)["ALL_PASS"]


def test_real_standalone_release_then_numeric_failure_preserves_outputs(evidence_paper, tmp_path, monkeypatch):
    from autopilot import p13_build_zip as release
    from autopilot import release_assets as assets
    if not Path(release.TECTONIC).is_file():
        pytest.skip("project Tectonic compiler is not installed")
    paper, stats = evidence_paper
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\begin{document}"
        r"Measured AUC \AUCRandomEpFifty. \cite{a}."
        r"\bibliographystyle{plain}\bibliography{references}\end{document}")
    monkeypatch.setattr(citations, "resolve_doi", lambda value: "Deep learning for retinal OCT classification")
    monkeypatch.setattr(citations.time, "sleep", lambda *args: None)
    record = citations.verify(r"\cite{a}", (paper / "references.bib").read_text(), online=True)
    authority = tmp_path / "authorities.json"
    assets.write_json(authority, record)
    out = tmp_path / "release.zip"
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "release-work",
                         stats_dir=stats, citation_record=authority) == 0
    targets = [out, out.with_suffix(".release.json"), paper / "main_submission.pdf",
               paper / "main_submission.docx", paper / "main_submission.docx.provenance.json"]
    before = {path: assets.sha256(path) for path in targets}
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\AUCRandomEpFifty}{0.1111}")
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "release-work",
                         stats_dir=stats, citation_record=authority) == 1
    assert {path: assets.sha256(path) for path in targets} == before
