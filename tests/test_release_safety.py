"""Release regressions use a project-local --basetemp and no network/compiler."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopilot import p13_build_zip as release
from autopilot import refresh_all
from autopilot import release_assets as assets


def passing_gate_report(script, args, cwd):
    if "--report" not in args:
        return
    text, _ = assets.source_tree(cwd)
    if script == "p15_verify_numbers.py":
        report = {"ALL_PASS": True, "checked_auc": 1, "items": [{"status": "verified"}],
                  "input_hashes": assets.input_hashes(cwd)}
        report["review_sha256"] = (assets.sha256(args[args.index("--review-file") + 1])
                                    if "--review-file" in args else None)
    else:
        report = {"ALL_PASS": True, "items": [{"status": "matched"}],
                  "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "bib_sha256": hashlib.sha256((Path(cwd) / "references.bib").read_text(
                      encoding="utf-8").encode("utf-8")).hexdigest()}
    assets.write_json(args[args.index("--report") + 1], report)


@pytest.fixture
def paper(tmp_path):
    folder = tmp_path / "paper"
    (folder / "auto").mkdir(parents=True)
    (folder / "main_submission.tex").write_text(
        r"\documentclass{article}\input{auto/auto_numbers}\begin{document}"
        r"\AUCRandomEpFifty\cite{x}\end{document}", encoding="utf-8")
    (folder / "auto" / "auto_numbers.tex").write_text(
        r"\newcommand{\AUCRandomEpFifty}{0.8641}", encoding="utf-8")
    (folder / "references.bib").write_text("@article{x,title={Example},year={2020}}", encoding="utf-8")
    (folder / "neurips_2026.sty").write_text("% style", encoding="utf-8")
    return folder


def test_refresh_stops_on_mandatory_child_failure(monkeypatch):
    monkeypatch.setattr(refresh_all.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=7, stdout="", stderr="failure"))
    with pytest.raises(SystemExit, match="refresh stopped"):
        refresh_all.run("mandatory gate", ["unused"])


def test_anonymity_distinguishes_cited_names_from_author_identifiers(tmp_path):
    import fitz
    pdf = tmp_path / "citations.pdf"
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), "Gary S. Collins is a cited author.")
        document.new_page().insert_text((72, 72), "References")
        document.save(pdf)
    assert not release.inspect_pdf(pdf)["identifying_terms_found"]
    assert release.identifying_text_hits("Gary\nFeng")
    assert release.identifying_text_hits(r"C:\Users\Gary\Desktop\jepa")
    assert release.identifying_text_hits("github.com/yfeng0206")


@pytest.mark.parametrize("failure", ["compiler", "page", "numeric", "empty-numeric", "word"])
def test_failed_build_preserves_every_old_artifact(tmp_path, paper, monkeypatch, failure):
    out = tmp_path / "release.zip"
    manifest = out.with_suffix(".release.json")
    targets = [out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx",
               tmp_path / ".overleaf_sync.json"]
    for target in targets:
        target.write_bytes(b"previous-good-" + target.suffix.encode())
    before = {path: path.read_bytes() for path in targets}

    def gate(script, args, cwd):
        passing_gate_report(script, args, cwd)
        if failure == "empty-numeric" and script == "p15_verify_numbers.py":
            path = Path(args[args.index("--report") + 1])
            data = json.loads(path.read_text())
            data["checked_auc"] = 0
            assets.write_json(path, data)
        return not ((failure == "numeric" and script == "p15_verify_numbers.py")
                    or (failure == "word" and script == "make_docx.py"))
    monkeypatch.setattr(release, "command_gate", gate)

    def compile_fake(*args, **kwargs):
        stage = Path(kwargs["cwd"])
        if failure != "compiler":
            (stage / "main.pdf").write_bytes(b"new-pdf")
            (stage / "main.log").write_text("no warnings")
        return SimpleNamespace(returncode=7 if failure == "compiler" else 0,
                               stdout="", stderr="")
    monkeypatch.setattr(release.subprocess, "run", compile_fake)
    monkeypatch.setattr(release, "inspect_pdf", lambda *a: {
        "main_content_pages": 10 if failure == "page" else 9,
        "identifying_terms_found": [], "identifying_metadata": {}})
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "work",
                         expected_docx_sha256=assets.sha256(paper / "main_submission.docx")) == 1
    assert {path: path.read_bytes() for path in targets} == before
    report = json.loads(next((tmp_path / "work").glob("release-*/validation.json")).read_text())
    failed_gate = {"compiler": "compiles_standalone", "page": "page_limit",
                   "numeric": "numeric_evidence", "empty-numeric": "numeric_evidence",
                   "word": "docx_generated"}[failure]
    assert report["checks"][failed_gate] is False


def test_placeholder_in_recursive_input_is_visible(paper):
    (paper / "main_submission.tex").write_text(r"\input{auto/auto_numbers}\input{auto/table}")
    (paper / "auto" / "auto_numbers.tex").write_text(r"\newcommand{\Hidden}{\ph{pending}}")
    (paper / "auto" / "table.tex").write_text(r"\Hidden")
    body, files = assets.source_tree(paper)
    assert "auto/table.tex" in files
    assert r"\ph{pending}" in assets.expanded_body(body)


def test_missing_input_and_cyclic_inputs_fail(paper):
    (paper / "main_submission.tex").write_text(r"\input{missing}")
    with pytest.raises(FileNotFoundError):
        assets.source_tree(paper)
    (paper / "main_submission.tex").write_text(r"\input{main_submission}")
    with pytest.raises(ValueError, match="cyclic"):
        assets.source_tree(paper)


def test_word_conflict_is_content_not_mtime(tmp_path):
    word = tmp_path / "paper.docx"
    word.write_bytes(b"collaborator edits")
    with pytest.raises(ValueError, match="Word edit conflict"):
        assets.check_word_conflict(word, "hash-of-prior-generated-word")
    assets.check_word_conflict(word, assets.sha256(word))


def test_promotion_rolls_back_replace_error(tmp_path, monkeypatch):
    old, new = tmp_path / "old.pdf", tmp_path / "new.pdf"
    manifest, candidate = tmp_path / "release.json", tmp_path / "candidate.json"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    manifest.write_bytes(b"old-manifest")
    candidate.write_bytes(b"new-manifest")
    replace = assets._rename_locked

    def fail_marker(source, target):
        if Path(target) == manifest and source.path.name.endswith(".pending"):
            raise OSError("injected disk failure")
        replace(source, target)
    monkeypatch.setattr(assets, "_rename_locked", fail_marker)
    with pytest.raises(assets.PromotionError, match="injected"):
        assets.promote([(new, old), (candidate, manifest)])
    assert old.read_bytes() == b"old"
    assert manifest.read_bytes() == b"old-manifest"


@pytest.fixture
def successful_release(tmp_path, paper, monkeypatch):
    out = tmp_path / "release.zip"
    old_word = paper / "main_submission.docx"
    old_word.write_bytes(b"reviewed-old-word")

    def gate(script, args, cwd):
        passing_gate_report(script, args, cwd)
        if script == "make_docx.py":
            Path(args[args.index("--out") + 1]).write_bytes(b"checked-new-word")
        return True
    monkeypatch.setattr(release, "command_gate", gate)

    def compile_fake(*args, **kwargs):
        stage = Path(kwargs["cwd"])
        (stage / "main.pdf").write_bytes(b"checked-new-pdf")
        (stage / "main.aux").write_text("")
        (stage / "main.log").write_text("complete")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(release.subprocess, "run", compile_fake)
    monkeypatch.setattr(release, "inspect_pdf", lambda *a: {
        "main_content_pages": 9, "identifying_terms_found": [], "identifying_metadata": {}})
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "work",
                         expected_docx_sha256=assets.sha256(old_word)) == 0
    return paper, out, out.with_suffix(".release.json")


def test_success_publishes_exact_hashed_manifest(successful_release):
    paper, out, path = successful_release
    report = json.loads(path.read_text())
    assert report["ALL_PASS"]
    assert report["artifacts"]["zip"]["sha256"] == assets.sha256(out)
    assert report["artifacts"]["pdf"]["sha256"] == assets.sha256(paper / "main_submission.pdf")
    assert report["artifacts"]["docx"]["sha256"] == assets.sha256(paper / "main_submission.docx")
    assert "main.tex" in report["source_files"]
    assert report["attachments"]["main_editable.docx"]["check"] == "check_docx.py"
    assert all(report["checks"].values())


def test_sync_consumes_only_validated_tree_and_checked_word(successful_release, tmp_path, monkeypatch):
    from scripts import sync_overleaf as sync
    paper, out, manifest_path = successful_release
    (paper / "figures").mkdir()
    (paper / "figures" / "unused.png").write_bytes(b"not part of release")
    monkeypatch.setattr(sync, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    work = tmp_path / "sync"
    work.mkdir()
    items, report = sync.validate(paper, manifest_path, work)
    paths = {rel for _, rel in items}
    assert paths == set(report["source_files"]) | {"main_editable.docx"}
    assert "figures/unused.png" not in paths
    assert all(Path(source).is_relative_to(work) for source, _ in items)
    # Editing an input after its stage was approved is detected before push.
    (paper / "auto" / "auto_numbers.tex").write_text("changed")
    with pytest.raises(ValueError, match="inputs changed"):
        sync.verify_local_release(paper, report)


def test_sync_rejects_changed_word_even_with_new_mtime(successful_release):
    from scripts import sync_overleaf as sync
    paper, _, manifest_path = successful_release
    report = json.loads(manifest_path.read_text())
    (paper / "main_submission.docx").write_bytes(b"new collaborator comment")
    with pytest.raises(ValueError, match="artifact changed"):
        sync.verify_local_release(paper, report)


def test_fixed_external_figure_identity_is_pinned(paper, tmp_path):
    figure = paper / "fixed.png"
    figure.write_bytes(b"baseline image")
    (paper / "main_submission.tex").write_text(r"\includegraphics{fixed.png}")
    registry = tmp_path / "assets.json"
    assets.write_json(registry, {"producers": {}, "fixed_inputs": {"fixed.png": assets.sha256(figure)}})
    assert assets.asset_inventory(paper, registry)["ALL_PASS"]
    figure.write_bytes(b"changed image")
    result = assets.asset_inventory(paper, registry)
    assert not result["ALL_PASS"]
    assert result["items"][0]["identity_status"] == "pinned_mismatch"


def test_reviewed_illustration_export_is_pinned_without_claiming_chart_verification(paper, tmp_path):
    figure = paper / "illustration.png"
    figure.write_bytes(b"reviewed illustration")
    (paper / "main_submission.tex").write_text(r"\includegraphics{illustration.png}")
    registry = tmp_path / "illustration-assets.json"
    assets.write_json(registry, {"fixed_inputs": {}, "producers": {"illustration.png": {
        "script": "scripts/illustration.py", "arguments": ["--maps-only"],
        "expected_output_sha256": assets.sha256(figure), "rendered_metrics": [],
        "evidence_scope": "selected example, not a frequency estimate"}}})
    result = assets.asset_inventory(paper, registry)
    assert result["ALL_PASS"]
    assert result["items"][0]["rendered_metrics"] == []
    assert result["items"][0]["kind"] == "available_generator"
    figure.write_bytes(b"unreviewed changed illustration")
    assert not assets.asset_inventory(paper, registry)["ALL_PASS"]


def test_retired_legacy_inputs_are_not_required_and_cannot_be_silently_reintroduced(paper, tmp_path):
    old = paper / "retired.png"
    old.write_bytes(b"historical evidence")
    registry = tmp_path / "retired-assets.json"
    assets.write_json(registry, {"producers": {}, "fixed_inputs": {},
                                 "retired_inputs": {"retired.png": assets.sha256(old)}})
    assert assets.asset_inventory(paper, registry)["ALL_PASS"]
    assert old.read_bytes() == b"historical evidence"
    (paper / "main_submission.tex").write_text(r"\includegraphics{retired.png}")
    result = assets.asset_inventory(paper, registry)
    assert not result["ALL_PASS"]
    assert result["items"][0]["identity_status"] == "undeclared"
    assert old.read_bytes() == b"historical evidence"


def test_source_mutation_after_snapshot_preserves_release(successful_release, tmp_path, monkeypatch):
    paper, out, manifest = successful_release
    targets = [out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx"]
    before = {path: assets.sha256(path) for path in targets}
    original_gate = release.command_gate

    def race(script, args, cwd):
        result = original_gate(script, args, cwd)
        if script == "check_docx.py":
            (paper / "main_submission.tex").write_text("changed in editor after validation")
        return result
    monkeypatch.setattr(release, "command_gate", race)
    work = tmp_path / "raced"
    assert release.build(out, paper_dir=paper, staging_root=work) == 1
    assert {path: assets.sha256(path) for path in targets} == before
    report = json.loads(next(work.glob("release-*/validation.json")).read_text())
    assert "inputs changed" in report["error"]


@pytest.mark.parametrize("builder", ["release", "word", "word-default"])
def test_transient_source_change_during_copy_cannot_revert_past_snapshot(paper, tmp_path, monkeypatch, builder):
    from autopilot import make_docx
    source = paper / "main_submission.tex"
    original = source.read_bytes()
    copy = assets.shutil.copyfile
    hits = []

    def transient(from_path, to_path, *args, **kwargs):
        if Path(from_path) == source:
            source.write_bytes(original + b"\nTRANSIENT COPY CONTENT\n")
            try:
                result = copy(from_path, to_path, *args, **kwargs)
                hits.append(str(to_path))
                return result
            finally:
                source.write_bytes(original)
        return copy(from_path, to_path, *args, **kwargs)
    monkeypatch.setattr(assets.shutil, "copyfile", transient)
    monkeypatch.setattr(release, "command_gate", lambda *a: pytest.fail("copied B must fail before gates"))
    if builder == "release":
        out = tmp_path / "existing.zip"
        out.write_bytes(b"old release")
        assert release.build(out, paper_dir=paper, staging_root=tmp_path / "work") == 1
        assert out.read_bytes() == b"old release"
    else:
        out = tmp_path / "existing.docx"
        out.write_bytes(b"old Word")
        aux = paper / "main_submission.aux"
        aux.write_text("")
        with pytest.raises(ValueError, match="copied input differs"):
            make_docx.build(paper, out, aux if builder == "word" else None,
                            staging_root=tmp_path / "word",
                            expected_docx_sha256=assets.sha256(out))
        assert out.read_bytes() == b"old Word"
    assert hits and source.read_bytes() == original


def test_staged_mutation_after_gate_preserves_release(successful_release, tmp_path, monkeypatch):
    paper, out, manifest = successful_release
    targets = [out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx"]
    before = {path: assets.sha256(path) for path in targets}
    gate = release.command_gate

    def mutate(script, args, cwd):
        result = gate(script, args, cwd)
        if script == "check_docx.py":
            (Path(cwd) / "main.tex").write_text("changed after gates")
        return result
    monkeypatch.setattr(release, "command_gate", mutate)
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "changed-stage") == 1
    assert {path: assets.sha256(path) for path in targets} == before


def test_archive_copy_aba_cannot_bypass_source_manifest(successful_release, tmp_path, monkeypatch):
    import zipfile
    paper, out, manifest = successful_release
    targets = [out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx"]
    before = {path: assets.sha256(path) for path in targets}
    write = zipfile.ZipFile.write

    def transient(archive, filename, arcname=None, *args, **kwargs):
        source = Path(filename)
        if arcname == "main.tex":
            original = source.read_bytes()
            try:
                source.write_bytes(original + b"\nTRANSIENT ARCHIVED CONTENT")
                return write(archive, filename, arcname, *args, **kwargs)
            finally:
                source.write_bytes(original)
        return write(archive, filename, arcname, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "write", transient)
    work = tmp_path / "archive-race"
    assert release.build(out, paper_dir=paper, staging_root=work) == 1
    assert {path: assets.sha256(path) for path in targets} == before
    report = json.loads(next(work.glob("release-*/validation.json")).read_text())
    assert "archived input differs" in report["error"]


def test_noncooperating_word_update_at_last_check_is_preserved(tmp_path, monkeypatch):
    import subprocess
    import sys
    word, new = tmp_path / "word.docx", tmp_path / "candidate.docx"
    word.write_bytes(b"old")
    new.write_bytes(b"new")
    expected = assets.sha256(word)
    opener = assets.ExclusiveFile
    changed = []

    def update_before_claim(path):
        if Path(path) == word and not changed:
            subprocess.run([sys.executable, "-c",
                            "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'editor update')",
                            str(word)], check=True)
            changed.append(True)
        return opener(path)
    monkeypatch.setattr(assets, "ExclusiveFile", update_before_claim)
    with pytest.raises(assets.PromotionError, match="destination changed"):
        assets.promote([(new, word)], expected_current={word: expected})
    assert word.read_bytes() == b"editor update"


def test_editor_creation_between_claim_and_install_preserves_both_versions(tmp_path, monkeypatch):
    import subprocess
    import sys
    word, new = tmp_path / "word.docx", tmp_path / "candidate.docx"
    word.write_bytes(b"reviewed prior Word")
    new.write_bytes(b"generated Word")
    expected = assets.sha256(word)
    rename = assets._rename_locked

    def create_during_gap(file, target):
        if Path(target) == word and file.path.name.endswith(".pending"):
            subprocess.run([sys.executable, "-c",
                            "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'editor replacement')",
                            str(word)], check=True)
        rename(file, target)
    monkeypatch.setattr(assets, "_rename_locked", create_during_gap)
    with pytest.raises(assets.PromotionError) as failure:
        assets.promote([(new, word)], expected_current={word: expected})
    assert word.read_bytes() == b"editor replacement"
    previous = next(tmp_path.glob(".release-recovery-*/word.docx.previous"))
    assert previous.read_bytes() == b"reviewed prior Word"
    assert str(previous.parent) in str(failure.value)


def test_installed_word_is_exclusively_held_until_manifest(tmp_path, monkeypatch):
    import subprocess
    import sys
    word, new = tmp_path / "word.docx", tmp_path / "candidate.docx"
    manifest, candidate = tmp_path / "release.json", tmp_path / "candidate.json"
    word.write_bytes(b"old")
    new.write_bytes(b"new")
    candidate.write_bytes(b"manifest")
    rename = assets._rename_locked
    attempts = []

    def competing_writer(file, target):
        if Path(target) == manifest:
            result = subprocess.run([sys.executable, "-c",
                                     "from pathlib import Path; import sys\n"
                                     "try: Path(sys.argv[1]).write_bytes(b'writer')\n"
                                     "except PermissionError: sys.exit(0)\n"
                                     "sys.exit(9)", str(word)], capture_output=True)
            attempts.append(result.returncode)
        rename(file, target)
    monkeypatch.setattr(assets, "_rename_locked", competing_writer)
    assets.promote([(new, word), (candidate, manifest)],
                   expected_current={word: assets.sha256(word), manifest: None})
    assert attempts == [0]
    assert word.read_bytes() == b"new"


def test_rollback_failure_keeps_backup_and_attempts_other_restorations(tmp_path, monkeypatch):
    pdf, word, manifest = (tmp_path / name for name in ("paper.pdf", "word.docx", "release.json"))
    pairs = []
    for target in (pdf, word, manifest):
        target.write_bytes(b"old-" + target.name.encode())
        source = target.with_suffix(target.suffix + ".candidate")
        source.write_bytes(b"new-" + target.name.encode())
        pairs.append((source, target))
    rename = assets._rename_locked
    restored = []

    def failures(file, target):
        target = Path(target)
        if target == manifest and file.path.name.endswith(".pending"):
            raise OSError("publish failure")
        if file.path.name.endswith(".previous"):
            restored.append(target)
            if target == word:
                raise OSError("restore failure")
        rename(file, target)
    monkeypatch.setattr(assets, "_rename_locked", failures)
    with pytest.raises(assets.PromotionError, match="rollback failures") as failure:
        assets.promote(pairs)
    assert pdf in restored and word in restored and manifest in restored
    assert pdf.read_bytes() == b"old-paper.pdf"
    assert manifest.read_bytes() == b"old-release.json"
    backup = next(tmp_path.glob(".release-recovery-*/word.docx.previous"))
    assert backup.read_bytes() == b"old-word.docx"
    assert failure.value.recovery_paths and failure.value.rollback_errors


@pytest.mark.parametrize("explicit", [False, True])
def test_numeric_review_is_staged_pinned_and_archived_as_private_qa(successful_release, tmp_path, explicit):
    import base64
    import zipfile
    from scripts import sync_overleaf as sync
    paper, out, manifest = successful_release
    review = tmp_path / "delivered_bindings.json" if explicit else paper / "numeric_reviews.json"
    content = b'{\r\n"version": 1, "scope": "synthetic routing test, not scientific approval"\r\n}\r\n'
    review.write_bytes(content)
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "review-work",
                         review_file=review if explicit else None) == 0
    report = json.loads(manifest.read_text())
    receipt = report["numeric_review"]
    assert Path(receipt["source"]) == review
    assert Path(receipt["archived"]).read_bytes() == content
    assert base64.b64decode(receipt["content"]) == content
    assert receipt["sha256"] == report["evidence"]["numbers"]["review_sha256"]
    with zipfile.ZipFile(out) as archive:
        assert not any("numeric_reviews" in name for name in archive.namelist())
    assert "numeric_reviews.json" not in report["source_files"]
    sync.verify_local_release(paper, report)
    review.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="numeric review input changed"):
        sync.verify_local_release(paper, report)


def test_numeric_review_hash_mismatch_preserves_release(successful_release, tmp_path, monkeypatch):
    paper, out, manifest = successful_release
    targets = [out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx"]
    before = {path: assets.sha256(path) for path in targets}
    review = tmp_path / "reviews.json"
    review.write_text('{"version":1}')
    gate = release.command_gate

    def wrong_report(script, args, cwd):
        result = gate(script, args, cwd)
        if script == "p15_verify_numbers.py":
            path = Path(args[args.index("--report") + 1])
            value = json.loads(path.read_text())
            value["review_sha256"] = None
            assets.write_json(path, value)
        return result
    monkeypatch.setattr(release, "command_gate", wrong_report)
    work = tmp_path / "wrong-review"
    assert release.build(out, paper_dir=paper, staging_root=work, review_file=review) == 1
    assert {path: assets.sha256(path) for path in targets} == before
    report = json.loads(next(work.glob("release-*/validation.json")).read_text())
    assert "review hash differs" in report["error"]


def test_numeric_review_change_after_gate_preserves_release(successful_release, tmp_path, monkeypatch):
    paper, out, manifest = successful_release
    before = {path: assets.sha256(path) for path in
              (out, manifest, paper / "main_submission.pdf", paper / "main_submission.docx")}
    review = tmp_path / "reviews.json"
    review.write_text('{"version":1}')
    gate = release.command_gate

    def edit_review(script, args, cwd):
        result = gate(script, args, cwd)
        if script == "check_docx.py":
            review.write_text('{"version":1,"scope":"changed after numeric gate"}')
        return result
    monkeypatch.setattr(release, "command_gate", edit_review)
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "edited-review", review_file=review) == 1
    assert {path: assets.sha256(path) for path in before} == before


def test_review_copy_aba_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "reviews.json"
    original = b'{"version":1}'
    source.write_bytes(original)
    copy = assets.shutil.copyfile

    def transient(path, target, *args, **kwargs):
        if Path(path) == source:
            source.write_bytes(b'{"version":1,"scope":"transient"}')
            try:
                return copy(path, target, *args, **kwargs)
            finally:
                source.write_bytes(original)
        return copy(path, target, *args, **kwargs)
    monkeypatch.setattr(assets.shutil, "copyfile", transient)
    with pytest.raises(ValueError, match="copied input differs"):
        assets.stage_numeric_review(tmp_path, tmp_path / "work", source)
    assert source.read_bytes() == original


def test_refresh_forwards_review_input_to_both_numeric_and_release_gates(tmp_path, monkeypatch):
    import sys
    review = tmp_path / "reviews.json"
    commands = []
    monkeypatch.setattr(sys, "argv", ["refresh_all.py", "--fast", "--review-file", str(review)])
    monkeypatch.setattr(refresh_all, "run", lambda label, command, **kwargs: commands.append(command) or 0)
    assert refresh_all.main() == 0
    consumers = [command for command in commands if any(
        str(argument).endswith(("p15_verify_numbers.py", "p13_build_zip.py")) for argument in command)]
    assert len(consumers) == 2
    assert all(command[command.index("--review-file") + 1] == str(review) for command in consumers)


@pytest.mark.parametrize("review_kind", ["explicit", "default"])
@pytest.mark.parametrize("destination", ["zip", "pdf", "docx", "docx_receipt", "release_manifest"])
def test_every_release_destination_preserves_numeric_review_input(paper, tmp_path, monkeypatch,
                                                                  review_kind, destination):
    products = tmp_path / "products"
    products.mkdir()
    out_zip = products / "submission.zip"
    kwargs = {"pdf_out": products / "submission.pdf",
              "docx_out": products / "submission.docx",
              "manifest_out": products / "release.json"}
    receipt = kwargs["docx_out"].with_suffix(".docx.provenance.json")
    review = (tmp_path / "explicit-review.json" if review_kind == "explicit"
              else paper / "numeric_reviews.json")
    resolved_alias = None
    if destination == "docx_receipt":
        if review_kind == "explicit":
            review = receipt
        else:
            # The default basename cannot textually equal a DOCX receipt.
            # Exercise the resolved file-link alias without requiring elevated
            # Windows symlink privileges.
            resolved_alias = (review, receipt)
    elif destination == "zip":
        out_zip = review
    elif destination == "pdf":
        kwargs["pdf_out"] = review
    elif destination == "docx":
        kwargs["docx_out"] = review
    else:
        kwargs["manifest_out"] = review
    output_paths = [out_zip, kwargs["pdf_out"], kwargs["docx_out"], kwargs["manifest_out"],
                    kwargs["docx_out"].with_suffix(".docx.provenance.json")]
    state = tmp_path / ".overleaf_sync.json"
    for path in set(output_paths + [state]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous-" + path.name.encode())
    review_bytes = b'{"version":1,"scope":"review-input collision regression"}'
    review.write_bytes(review_bytes)
    if resolved_alias:
        receipt.write_bytes(review_bytes)
        resolve = Path.resolve

        def resolve_review_alias(path, *args, **kw):
            return resolve(receipt if path == review else path, *args, **kw)
        monkeypatch.setattr(Path, "resolve", resolve_review_alias)
    source_paths = [assets.safe_path(paper, rel) for rel in assets.input_hashes(paper)]
    watched = set(output_paths + source_paths + [review, state])
    before = {path: path.read_bytes() for path in watched}
    monkeypatch.setattr(release, "command_gate",
                        lambda *args: pytest.fail("collision must stop before validation/compilation"))
    work = tmp_path / "work"
    assert release.build(out_zip, paper_dir=paper, staging_root=work,
                         review_file=review if review_kind == "explicit" else None, **kwargs) == 1
    assert {path: path.read_bytes() for path in watched} == before
    report = json.loads(next(work.glob("release-*/validation.json")).read_text())
    assert "overwrite a source input" in report["error"]


def test_archive_excludes_withdrawn_companions_and_referenced_private_fulltexts(successful_release, tmp_path):
    import base64
    import zipfile
    paper, out, manifest = successful_release
    withdrawn = (
        "fig_masking_policies", "fig_precision_paradox", "interp_04_window_occlusion_W7",
        "interp_14_odos_mirror_test", "interp_heatmap_grid",
        "interp_slice_contribution_by_outcome", "interp_slice_contribution_curves",
    )
    figures = paper / "figures"
    figures.mkdir()
    legacy = []
    for stem in withdrawn:
        for suffix in (".png", ".pdf", ".svg"):
            path = figures / (stem + suffix)
            path.write_bytes(b"synthetic withdrawn fixture " + path.name.encode())
            legacy.append(path)
    local_sources = paper / "local-only-sources"
    local_sources.mkdir()
    local_documents = [local_sources / "paper.pdf", local_sources / "fullFairVisionREADME.txt"]
    for path in local_documents:
        path.write_bytes(b"synthetic full-document fixture, never a redistributable source")
    review = tmp_path / "reviews.json"
    assets.write_json(review, {"version": 1, "sources": {
        "document_%d" % index: {"root": "paper",
                                "path": path.relative_to(paper).as_posix(),
                                "sha256": assets.sha256(path)}
        for index, path in enumerate(local_documents)}})
    before = {path: assets.sha256(path) for path in legacy + local_documents}
    assert release.build(out, paper_dir=paper, staging_root=tmp_path / "archive-scope",
                         review_file=review) == 0
    report = json.loads(manifest.read_text())
    with zipfile.ZipFile(out) as archive:
        members = set(archive.namelist())
        assert members == set(report["source_files"]) | {"main.pdf", "README_OVERLEAF.txt"}
        assert not any(stem in member for member in members for stem in withdrawn)
        assert not any(path.name in members for path in local_documents)
    # The QA archive contains the selected JSON references, not their payloads.
    assert base64.b64decode(report["numeric_review"]["content"]) == review.read_bytes()
    assert {path: assets.sha256(path) for path in before} == before
