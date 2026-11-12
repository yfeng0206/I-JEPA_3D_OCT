import io
import subprocess
import zipfile

import pytest

from autopilot import release_assets as assets


@pytest.fixture
def historical_numeric_inputs(tmp_path):
    """Keep stale-approval regressions independent of later manuscript edits."""
    manuscript_commit = "b941814184d4c70008fa239aa9ebf23c75a9f9fa"
    approval_commit = "b4e5f35e4d7df16a6dd5c48f108560ecffb1a188"
    paper_relative = "paper/genai4health2026"
    snapshot = tmp_path / "historical_numeric_inputs"
    archive = subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "-c", "core.eol=crlf",
         "archive", "--format=zip", manuscript_commit, "--", paper_relative],
        cwd=assets.REPO)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        bundle.extractall(snapshot)

    receipts = {}
    for name in ("literal_review_candidate_current.json", "paper_release_numeric_candidate.json"):
        relative = "autopilot/investigations/delivered_task/evidence/" + name
        approved = subprocess.check_output(
            ["git", "show", f"{approval_commit}:{relative}"], cwd=assets.REPO)
        path = snapshot / name
        path.write_bytes(approved)
        receipts[name] = path
    return snapshot / paper_relative, receipts
