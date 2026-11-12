"""Active packages remain importable after experimental code is archived."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_evaluation_package_import_does_not_require_archived_imagenet_helpers():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import src.evaluation",
         str(root)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_archived_feature_extractor_reports_actionable_error(monkeypatch):
    from src import helper

    monkeypatch.setattr(helper, "FrozenFeatureExtractor", None)
    with pytest.raises(RuntimeError, match="archive/slice_level/feature_extractor.py"):
        helper.init_feature_extractor("cpu")


def test_archived_slice_probe_fails_before_creating_outputs(monkeypatch, tmp_path):
    from src import eval_downstream

    monkeypatch.setattr(eval_downstream, "FrozenFeatureExtractor", None)
    output_dir = tmp_path / "probe"
    config = {"logging": {"output_dir": str(output_dir)}}
    with pytest.raises(RuntimeError, match="archive/slice_level/feature_extractor.py"):
        eval_downstream.run_slice_downstream(config, "cpu")
    assert not output_dir.exists()
