import sys
from pathlib import Path

from scripts import download_weights, upload_weights


def test_list_handles_explicit_and_directory_sources(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upload_weights, "RUNS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["upload_weights.py", "--list"])

    assert upload_weights.main() == 0
    output = capsys.readouterr().out
    for arm in upload_weights.ARMS:
        assert arm in output
    assert "explicit checkpoint paths" in output
    assert "[SOURCE CHECKPOINT MISSING]" in output


def test_list_recognizes_present_explicit_sources(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upload_weights, "RUNS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["upload_weights.py", "--list"])
    for source, _ in upload_weights.ARMS["anatomy-v2"]["explicit"]:
        path = tmp_path / Path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")

    assert upload_weights.main() == 0
    line = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("anatomy-v2")
    )
    assert "MISSING" not in line


def test_download_listing_distinguishes_legacy_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["download_weights.py", "--list"])
    download_weights.main()
    output = capsys.readouterr().out
    assert "--arm NAME for a paper arm" in output
    assert "--encoder and --all select legacy weights" in output
