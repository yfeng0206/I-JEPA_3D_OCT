import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopilot import numeric_bindings as binding
from autopilot import numeric_plot_review as plots


def replay_subset(tmp_path, monkeypatch, names, *, wrong_hash=False):
    from matplotlib.figure import Figure

    fig = Figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    expected = io.BytesIO()
    fig.savefig(expected, format="png")
    digest = hashlib.sha256(expected.getvalue()).hexdigest()
    evidence = binding.Evidence(tmp_path, tmp_path)
    evidence.paths["p17_subgroup_multiplicity.json"] = tmp_path / "multiplicity.json"
    observed = []

    def artists(figure, name, source):
        assert figure is fig
        observed.append(name)
        return [{"checked": name}]

    def main():
        for name in names:
            fig.savefig(Path(name))

    module = SimpleNamespace(main=main)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda candidate: None))
    monkeypatch.setattr(plots.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(plots.importlib.util, "module_from_spec", lambda candidate: module)
    monkeypatch.setattr(plots, "p8_artists", artists)
    report = plots.verify_local_plots(tmp_path, evidence, {}, [{
        "path": "auto/fig_trajectories_ci.png",
        "sha256": "0" * 64 if wrong_hash else digest,
    }])
    return report, observed


def test_p8_replay_subset_validates_only_referenced_registered_figure(tmp_path, monkeypatch):
    report, observed = replay_subset(tmp_path, monkeypatch, [
        "fig_trajectories_ci.png", "fig_fairness.png", "fig_roc.png", "fig_labeleff.png"])
    assert set(report) == {"auto/fig_trajectories_ci.png"}
    assert report["auto/fig_trajectories_ci.png"]["status"] == "programmatically_verified_plotted_values"
    assert observed == ["fig_trajectories_ci.png"]


@pytest.mark.parametrize("names,wrong_hash,message", [
    (["fig_trajectories_ci.png", "unknown.png"], False, "unregistered P8 output"),
    (["fig_fairness.png"], False, "registered P8 output not produced"),
    (["fig_trajectories_ci.png", "fig_fairness.png"], True, "delivered raster differs"),
])
def test_p8_replay_subset_remains_fail_closed(tmp_path, monkeypatch, names, wrong_hash, message):
    report, _ = replay_subset(tmp_path, monkeypatch, names, wrong_hash=wrong_hash)
    row = report["auto/fig_trajectories_ci.png"]
    assert row["status"] == "unresolved"
    assert message in row["action"]
