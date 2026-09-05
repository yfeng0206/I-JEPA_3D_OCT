import xml.etree.ElementTree as ET
import zipfile

import pytest

from autopilot import check_docx, make_docx
from autopilot import release_assets as assets


@pytest.fixture
def word_table(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main_submission.tex").write_text(
        r"\documentclass{article}\begin{document}Results \cite{a}."
        r"\begin{table}\caption{Measured values}\begin{tabular}{lrr}"
        r"Arm & First & Second\\Alpha & 0.1234 & 0.5678\\"
        r"Beta & 0.2345 & 0.6789\\\end{tabular}\end{table}"
        r"\bibliography{references}\end{document}")
    (paper / "references.bib").write_text(
        "@article{a, title={A fixture paper}, author={Smith, A}, year={2020}}\n")
    aux = paper / "main_submission.aux"
    aux.write_text("")
    word = tmp_path / "candidate.docx"
    make_docx.build(paper, word, aux, staging_root=tmp_path / "work")
    return paper, aux, word


@pytest.mark.parametrize("mutation", ["surplus", "cells", "rows", "row_labels", "caption", "extra_table"])
def test_ordered_word_cells_reject_surplus_and_permutation(word_table, tmp_path, mutation):
    paper, aux, word = word_table
    with zipfile.ZipFile(word) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(files["word/document.xml"])
    table = root.find(".//" + check_docx.W + "tbl")
    rows = table.findall(check_docx.W + "tr")
    if mutation == "surplus":
        rows[1].findall(check_docx.W + "tc")[1].find(".//" + check_docx.W + "t").text += " 0.9999"
    elif mutation == "cells":
        cell = rows[1].findall(check_docx.W + "tc")[1]
        rows[1].remove(cell)
        rows[1].append(cell)
    elif mutation == "rows":
        table.remove(rows[1])
        table.append(rows[1])
    elif mutation == "row_labels":
        left = rows[1].findall(check_docx.W + "tc")[0].find(".//" + check_docx.W + "t")
        right = rows[2].findall(check_docx.W + "tc")[0].find(".//" + check_docx.W + "t")
        left.text, right.text = right.text, left.text
    elif mutation == "caption":
        for paragraph in root.findall(".//" + check_docx.W + "p"):
            style = paragraph.find(check_docx.W + "pPr/" + check_docx.W + "pStyle")
            if style is not None and style.get(check_docx.W + "val") == "TableCaption":
                paragraph.find(".//" + check_docx.W + "t").text += " 0.9999"
    else:
        import copy
        root.find(check_docx.W + "body").append(copy.deepcopy(table))
    files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    damaged = tmp_path / "damaged.docx"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    result = check_docx.check(damaged, paper, aux)
    assert not result["ALL_PASS"], result
    assert any("numeric" in error or "data-table" in error for error in result["errors"])


def test_editor_update_after_initial_guard_cannot_become_the_approved_hash(word_table, tmp_path, monkeypatch):
    paper, aux, word = word_table
    guard = assets.check_word_conflict

    def update_after_guard(target, *args, **kwargs):
        approved = guard(target, *args, **kwargs)
        if target == word:
            word.write_bytes(b"new collaborator version")
        return approved
    monkeypatch.setattr(assets, "check_word_conflict", update_after_guard)
    with pytest.raises(ValueError, match="Word copy changed"):
        make_docx.build(paper, word, aux, staging_root=tmp_path / "new-work")
    assert word.read_bytes() == b"new collaborator version"
