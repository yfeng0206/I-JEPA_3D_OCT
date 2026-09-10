"""Enforce source-derived Word completeness and actual-aux cross references."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

try:
    from . import release_assets as assets
    from . import verify_citations as citations
except ImportError:
    import release_assets as assets
    import verify_citations as citations

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def node_text(node):
    return "".join(element.text or "" for element in node.iter()
                   if element.tag in (W + "t", M + "t"))


def bookmark_values(root):
    active, values, ids = {}, {}, {}
    for node in root.iter():
        if node.tag == W + "bookmarkStart":
            name, identity = node.get(W + "name"), node.get(W + "id")
            ids[identity] = name
            active[name] = ""
        elif node.tag == W + "bookmarkEnd":
            name = ids.get(node.get(W + "id"))
            if name in active:
                values[name] = active.pop(name)
        elif node.tag in (W + "t", M + "t"):
            for name in active:
                active[name] += node.text or ""
    return values


def numeric_sequence(text):
    """Normalize grouping commas, TeX grouping and equivalent sign glyphs only."""
    text = text.replace("−", "-").replace("–", "-").replace("{,}", "").replace(",", "")
    text = re.sub(r"\\[A-Za-z]+", " ", text).replace("{", " ").replace("}", " ")
    text = re.sub(r"(?<=\d)(?=[+-]\d)", " ", text)
    return tuple(match[0].lstrip("+") for match in assets.NUMBER.finditer(text))


def numeric_tokens(text):
    return Counter(numeric_sequence(text))


def cell_words(text):
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    return tuple(re.findall(r"[a-z]+[0-9]*", text.lower()))


def numeric_xml_text(node):
    if node.tag in (W + "t", M + "t"):
        return node.text or ""
    # Fractions and scripts have semantic boundaries that adjacent runs do not.
    separator = " " if node.tag in (M + "f", M + "sSup", M + "sSub", M + "sSubSup") else ""
    return separator.join(numeric_xml_text(child) for child in node)


def docx_table_rows(table):
    rows = table.findall(W + "tr")
    positions = []
    for row in rows:
        cells, column = [], 0
        for cell in row.findall(W + "tc"):
            span = cell.find(W + "tcPr/" + W + "gridSpan")
            colspan = int(span.get(W + "val")) if span is not None else 1
            merge = cell.find(W + "tcPr/" + W + "vMerge")
            cells.append((column, colspan, cell, merge))
            column += colspan
        positions.append(cells)
    result = []
    for row_index, cells in enumerate(positions):
        normalized, has_content = [], False
        for column, colspan, cell, merge in cells:
            if merge is not None and merge.get(W + "val") != "restart":
                continue
            rowspan = 1
            if merge is not None:
                for following in positions[row_index + 1:]:
                    continuation = next((item for item in following if item[0] == column), None)
                    if (not continuation or continuation[3] is None
                            or continuation[3].get(W + "val") == "restart"):
                        break
                    rowspan += 1
            text = "\n".join(numeric_xml_text(paragraph) for paragraph in cell.findall(W + "p"))
            has_content |= bool(text.strip())
            normalized.append((rowspan, colspan, numeric_sequence(text), cell_words(text)))
        # Empty booktabs/spacing rows are presentation, not data rows.
        if has_content:
            result.append(tuple(normalized))
    return tuple(result)


def source_structure(source, bibliography=None):
    import pypandoc
    # Pandoc expands scalar macros itself. Reading its AST excludes TeX layout
    # arguments (column widths, font sizes) from table-cell expectations.
    source = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", source, flags=re.S)
    options = ["--citeproc", "--bibliography=" + str(bibliography)] if bibliography else []
    document = json.loads(pypandoc.convert_text(source, "json", format="latex", extra_args=options))
    tables, captions = [], []

    def inline_text(node, prose_only=False):
        if isinstance(node, list):
            return "".join(inline_text(item, prose_only) for item in node)
        if not isinstance(node, dict):
            return ""
        kind, value = node.get("t"), node.get("c")
        if kind == "Str":
            return value
        if kind in ("Space", "SoftBreak", "LineBreak"):
            return " "
        if kind in ("Plain", "Para"):
            return inline_text(value, prose_only) + "\n"
        if kind in ("Math", "Code"):
            return "" if prose_only else value[-1]
        if kind == "Cite":
            return "" if prose_only else inline_text(value[1], prose_only)
        if kind == "Note":
            return ""
        return inline_text(value, prose_only)

    def table_rows(contents):
        rows = list(contents[3][1])
        for table_body in contents[4]:
            rows.extend(table_body[2])
            rows.extend(table_body[3])
        rows.extend(contents[5][1])
        result = []
        for _, cells in rows:
            normalized, has_content = [], False
            for cell in cells:
                text = inline_text(cell[4])
                has_content |= bool(text.strip())
                normalized.append((cell[2], cell[3], numeric_sequence(text), cell_words(text)))
            if has_content:
                result.append(tuple(normalized))
        return tuple(result)

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            if node.get("t") in ("Table", "Figure"):
                contents = node["c"]
                captions.append({"kind": node["t"],
                                 "words": re.findall(r"[a-z]+[0-9]*", inline_text(contents[1], True).lower()),
                                 "numbers": numeric_sequence(inline_text(contents[1]))})
                if node["t"] == "Table":
                    tables.append(table_rows(contents))
                else:
                    visit(contents[2:])
            else:
                visit(node.get("c"))
    visit(document.get("blocks", []))
    return {"tables": tables, "captions": captions}


def source_table_numbers(source):
    return source_structure(source)["tables"]


def is_subsequence(expected, actual):
    iterator = iter(actual)
    return all(any(candidate == wanted for candidate in iterator) for wanted in expected)


def check(docx, paper, aux, *, generated_images=()):
    docx, paper = Path(docx), Path(paper).resolve()
    errors = []
    snapshot = assets.input_hashes(paper)
    source, _ = assets.source_tree(paper)
    labels = assets.aux_labels(aux)
    references = assets.source_refs(source, labels)
    with zipfile.ZipFile(docx) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        media = {name: hashlib.sha256(archive.read(name)).hexdigest()
                 for name in archive.namelist() if name.startswith("word/media/")}
        core = ET.fromstring(archive.read("docProps/core.xml")) if "docProps/core.xml" in archive.namelist() else None
    receipt_path = docx.with_suffix(".docx.provenance.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("docx_sha256") != assets.sha256(docx):
            errors.append("Word content differs from its generation receipt (possible collaborator edits)")
        if receipt.get("inputs") != snapshot:
            errors.append("Word source inputs changed since generation")
        if receipt.get("aux_sha256") != assets.sha256(aux):
            errors.append("Word compiled cross-reference source changed")
        for digest in receipt.get("generated_images", {}).values():
            if digest not in media.values():
                errors.append("generated TikZ payload differs from its receipt")
    paragraphs = root.findall(".//" + W + "p")
    text = "\n".join(node_text(paragraph) for paragraph in paragraphs)
    styles = Counter()
    bibliography = []
    actual_captions = {"Table": [], "Figure": []}
    reference_headings, first_bibliography = [], None
    for index, paragraph in enumerate(paragraphs):
        style = paragraph.find(W + "pPr/" + W + "pStyle")
        if style is not None:
            name = style.get(W + "val")
            styles[name] += 1
            if name == "Bibliography":
                bibliography.append(node_text(paragraph))
                if first_bibliography is None:
                    first_bibliography = index
            if name.startswith("Heading") and node_text(paragraph).strip() == "References":
                reference_headings.append(index)
            if name in ("TableCaption", "ImageCaption"):
                actual_captions["Table" if name == "TableCaption" else "Figure"].append(paragraph)
    if (len(reference_headings) != 1 or first_bibliography is None
            or reference_headings[0] >= first_bibliography):
        errors.append("missing unique styled References heading before bibliography")
    if "Working copy for comment and editing." not in text:
        errors.append("missing source-authority banner")
    definitions = {name: value for name, value, _, _ in assets.macros(source)}
    leaks = [name for name in definitions if re.search(r"(?<![A-Za-z])\\?" + re.escape(name) + r"\b", text)]
    if leaks:
        errors.append("unexpanded source macros: " + ", ".join(sorted(leaks)[:12]))
    if re.search(r"\\(?:cite[a-z]*|ref|autoref)\s*\{|\\ph\b", text) or "??" in text:
        errors.append("unresolved citation/reference/placeholder in Word")
    actual_refs = bookmark_values(root)
    for reference in references:
        actual = actual_refs.get(reference["bookmark"])
        expected = reference["number"]
        if reference["command"].startswith(r"\eqref"):
            expected = "(" + expected + ")"
        if actual is None or not re.search(r"(?<![\w.])" + re.escape(expected) + r"(?![\w.])", actual):
            errors.append("cross-reference %s expected %s, found %r" %
                          (reference["key"], expected, actual))
    expected_tables = len(re.findall(r"\\begin\{table\*?\}", source))
    expected_figures = len(re.findall(r"\\begin\{figure\*?\}", source))
    if styles["TableCaption"] != expected_tables:
        errors.append("table captions: expected %d, found %d" % (expected_tables, styles["TableCaption"]))
    if styles["ImageCaption"] != expected_figures:
        errors.append("figure captions: expected %d, found %d" % (expected_figures, styles["ImageCaption"]))
    all_tables = root.findall(".//" + W + "tbl")
    actual_tables = []
    for table in all_tables:
        rows = docx_table_rows(table)
        has_drawing = table.find(".//" + W + "drawing") is not None
        has_numbers = any(cell[2] for row in rows for cell in row)
        # Pandoc may use a drawing-only table for side-by-side figure panels.
        # A numeric addition stops qualifying as layout and is rejected below.
        if not (has_drawing and not has_numbers):
            actual_tables.append(rows)
    try:
        from .make_docx import resolve_references
    except ImportError:
        from make_docx import resolve_references
    resolved_source, _ = resolve_references(source, aux)
    structure = source_structure(resolved_source, paper / "references.bib")
    expected_cells = structure["tables"]
    caption_index = Counter()
    for caption in structure["captions"]:
        kind, index = caption["kind"], caption_index[caption["kind"]]
        caption_index[kind] += 1
        if index >= len(actual_captions[kind]):
            errors.append("%s caption %d missing" % (kind, index + 1))
            continue
        paragraph = actual_captions[kind][index]
        actual = node_text(paragraph)
        if not is_subsequence(caption["words"], re.findall(r"[a-z]+[0-9]*", actual.lower())):
            errors.append("%s caption %d lost source prose" % (kind, index + 1))
        # Read math boundaries without losing fraction/script separation.
        if caption["numbers"] != numeric_sequence(numeric_xml_text(paragraph)):
            errors.append("%s caption %d numeric sequence differs from source" % (kind, index + 1))
    if len(actual_tables) != len(expected_cells):
        errors.append("data-table count: expected %d, found %d" % (len(expected_cells), len(actual_tables)))
    for index, (expected, actual) in enumerate(zip(expected_cells, actual_tables), 1):
        if expected != actual:
            errors.append("table %d ordered row/cell numeric content differs from source" % index)
    expected_images = [assets.resolve_graphic(paper, name) for name in assets.graphics(source)]
    expected_hashes = Counter(assets.sha256(image) for image in expected_images)
    found_hashes = Counter(media.values())
    if expected_hashes - found_hashes:
        errors.append("source figure payload missing or changed")
    tikz_count = len(re.findall(r"\\begin\{tikzpicture\}", source))
    expected_media_count = len(expected_images) + tikz_count
    if len(media) != expected_media_count:
        errors.append("media count: expected %d, found %d" % (expected_media_count, len(media)))
    for image in generated_images:
        if assets.sha256(image) not in found_hashes:
            errors.append("generated TikZ payload missing")
    keys = citations.cited_keys(source)
    entries = citations.bib_entries((paper / "references.bib").read_text(encoding="utf-8"))
    if not keys or len(bibliography) != len(keys):
        errors.append("bibliography coverage: expected %d, found %d" % (len(keys), len(bibliography)))
    bib_text = citations.norm(" ".join(bibliography))
    for key in sorted(keys):
        title = citations.field(entries.get(key, ("", ""))[1], "title")
        if not title or citations.norm(title) not in bib_text:
            errors.append("bibliography title absent: " + key)
    if core is not None:
        identifying = [element.text for element in core.iter()
                       if element.tag.rsplit("}", 1)[-1] in ("creator", "lastModifiedBy")
                       and element.text and element.text.strip()]
        if identifying:
            errors.append("nonempty creator/lastModifiedBy metadata")
    if assets.input_hashes(paper) != snapshot:
        errors.append("source inputs changed during Word validation")
    return {"ALL_PASS": not errors, "errors": errors,
            "docx_sha256": assets.sha256(docx), "input_hashes": snapshot,
            "aux_sha256": assets.sha256(aux), "counts": {
        "references": len(references), "table_captions": styles["TableCaption"],
        "figure_captions": styles["ImageCaption"], "tables": len(all_tables),
        "data_tables": len(actual_tables),
        "media": len(media), "bibliography": len(bibliography)},
        "limits": ["Conversion completeness does not establish scientific validity.",
                   "Data tables require exact ordered row/cell numeric sequences and spans; "
                   "only empty spacer rows and drawing-only layout tables are ignored.",
                   "Numeric normalization equates grouping commas, TeX grouping, optional unary plus "
                   "and minus glyphs, not changed values or precision."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx")
    parser.add_argument("--paper-dir", default=str(assets.PAPER))
    parser.add_argument("--aux")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    paper = Path(args.paper_dir)
    try:
        docx = Path(args.docx) if args.docx else paper / "main_submission.docx"
        receipt_path = docx.with_suffix(".docx.provenance.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        result = check(docx, paper, args.aux or receipt.get("aux_path") or paper / "main_submission.aux")
    except Exception as exc:
        result = {"ALL_PASS": False, "errors": [str(exc)]}
    if args.report:
        assets.write_json(args.report, result)
    print("Word completeness:", result.get("counts", {}))
    for error in result["errors"][:15]:
        print("FAIL:", error)
    print("RESULT:", "PASS" if result["ALL_PASS"] else "FAIL")
    return 0 if result["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
