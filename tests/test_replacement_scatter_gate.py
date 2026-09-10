import json
from pathlib import Path
import shutil

import pytest

from autopilot import numeric_bindings as n
from autopilot import numeric_plot_review as plots
from autopilot import p15_verify_numbers as numbers
from autopilot import release_assets as assets


@pytest.fixture
def scatter(tmp_path):
    stats = Path(numbers.INV).parent
    base = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "legacy_figure_reviews"
    source = base / "replacements" / "fig_purity_auc_ep50_fp32.png"
    if not source.exists() or not (stats / "p1b_full_inventory.json").exists():
        pytest.skip("reviewed replacement evidence not installed")
    paper = tmp_path / "paper"
    (paper / "figures").mkdir(parents=True)
    target = paper / "figures" / source.name
    shutil.copyfile(source, target)
    evidence = n.Evidence(paper, stats, n.BUILTIN_SOURCES)
    item = {"path": "figures/" + source.name, "sha256": assets.sha256(target)}
    return paper, evidence, item


def test_registered_replacement_checks_sources_and_actual_png(scatter):
    paper, evidence, item = scatter
    result = plots.verify_local_plots(paper, evidence, {}, [item])
    row = result[item["path"]]
    assert row["status"] == "programmatically_verified_plotted_values", row
    assert row["mathematically_verified"]
    assert row["validation"]["rendered_sha256"] == item["sha256"]
    records = row["validation"]["observed_series"]
    assert [r["arm"] for r in records] == ["random", "oracle", "envelope", "anatomy-v2", "cover-f021"]
    assert all(r["precision"] == "fp32" and r["epoch"] == 50 for r in records)
    assert all(r["x_expression"]["source"] == "geometry42" for r in records)
    assert all(r["y_expression"]["source"] == "p1b_full_inventory.json" for r in records)
    assert row["validation"]["artist_checks"]["line_or_interval_count"] == 0
    json.dumps(result, allow_nan=False)


def test_scatter_registration_does_not_run_private_token_builder(scatter, monkeypatch):
    paper, evidence, item = scatter
    original = plots._load_scatter_module

    def load(path, name):
        module = original(path, name)

        def forbidden(*args, **kwargs):
            raise AssertionError("private token fixture path must never run in scatter validation")

        if hasattr(module, "frozen_token_data"):
            module.frozen_token_data = forbidden
        if hasattr(module, "verify_token_artists"):
            module.verify_token_artists = forbidden
        return module

    monkeypatch.setattr(plots, "_load_scatter_module", load)
    token_item = {"path": "figures/fig_policy_family_token_maps.png", "sha256": "0" * 64}
    result = plots.verify_local_plots(paper, evidence, {}, [item, token_item])
    assert set(result) == {item["path"]}
    assert result[item["path"]]["status"] == "programmatically_verified_plotted_values"


def test_same_registered_png_cannot_hide_changed_data_artist(scatter, monkeypatch):
    paper, evidence, item = scatter
    original = plots._load_scatter_module

    def load(path, name):
        module = original(path, name)
        if hasattr(module, "build_scatter"):
            build = module.build_scatter

            def altered(*args, **kwargs):
                fig = build(*args, **kwargs)
                point = fig.axes[0].collections[0]
                offsets = point.get_offsets().copy()
                offsets[0, 0] += 1
                point.set_offsets(offsets)
                return fig

            module.build_scatter = altered
        return module

    monkeypatch.setattr(plots, "_load_scatter_module", load)
    row = plots.verify_local_plots(paper, evidence, {}, [item])[item["path"]]
    assert row["status"] == "mismatch", row
    assert "exact fp32 point" in row["action"]
    json.dumps(row, allow_nan=False)


def test_changed_output_bytes_fail_even_with_fresh_item_hash(scatter):
    paper, evidence, item = scatter
    path = assets.safe_path(paper, item["path"])
    path.write_bytes(path.read_bytes() + b"unreviewed")
    item["sha256"] = assets.sha256(path)
    row = plots.verify_local_plots(paper, evidence, {}, [item])[item["path"]]
    assert row["status"] == "unresolved"
    assert "registered source-linked export" in row["action"]


def test_changed_source_pin_fails(scatter):
    paper, evidence, item = scatter
    evidence.specs["geometry42"] = {**evidence.specs["geometry42"], "sha256": "0" * 64}
    row = plots.verify_local_plots(paper, evidence, {}, [item])[item["path"]]
    assert row["status"] == "unresolved"
    assert "hash mismatch" in row["action"]


def test_complete_numeric_candidate_passes_under_staged_root_rename(tmp_path):
    stats = Path(numbers.INV).parent
    candidate = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "paper_release_numeric_candidate.json"
    if not candidate.exists() or not (stats / "p1c_stats.json").exists():
        pytest.skip("complete scientific release candidate not installed")
    paper = tmp_path / "paper"
    for relative in assets.input_hashes(assets.PAPER):
        source = assets.safe_path(assets.PAPER, relative)
        target = assets.safe_path(paper, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.suffix == ".png":
            for extension in (".pdf", ".svg"):
                companion = source.with_suffix(extension)
                if companion.exists():
                    shutil.copyfile(companion, target.with_suffix(extension))
    (paper / "main_submission.tex").rename(paper / "main.tex")
    private_review = tmp_path / "immutable_numeric_review.json"
    shutil.copyfile(candidate, private_review)
    report = numbers.audit(paper, stats, private_review)
    assert report["ALL_PASS"], (report["errors"], report["unresolved_ids"])
    assert report["source_aliases"] == {"main.tex": "main_submission.tex"}
    assert "main.tex" in report["input_hashes"] and "main_submission.tex" not in report["input_hashes"]
    assert private_review.read_bytes() == candidate.read_bytes()
    assert report["review_sha256"] == assets.sha256(candidate)
    json.dumps(report, allow_nan=False)
