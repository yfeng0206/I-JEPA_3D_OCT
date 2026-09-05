import json
from pathlib import Path

import pytest

from autopilot import numeric_bindings as binding
from autopilot import p15_verify_numbers as numbers
from autopilot import release_assets as assets
from autopilot import numeric_plot_review as plots


@pytest.fixture
def paper(tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    stats = tmp_path / "stats"
    stats.mkdir()
    (root / "references.bib").write_text("")
    (root / "main_submission.tex").write_text(
        r"\documentclass{article}\newcommand{\AUCRandomEpFifty}{0.8641}"
        r"\begin{document}\AUCRandomEpFifty\end{document}")
    (stats / "p1c_stats.json").write_text(json.dumps({
        "n_test": 3000, "n_pos": 1466, "table": [
            {"key": "random@ep50@fp16", "arm": "random", "epoch": 50,
             "precision": "fp16", "auc": 0.8641, "ci95_lo": 0.851, "ci95_hi": 0.877}
        ], "contrasts": []}))
    (stats / "p1b_full_inventory.json").write_text('{"records": []}')
    return root, stats


def append_body(root, text):
    path = root / "main_submission.tex"
    path.write_text(path.read_text().replace(r"\end{document}", text + r"\end{document}"))


def source_spec(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    return {"root": "paper", "path": name, "sha256": assets.sha256(path)}


def test_custom_diagnostic_macro_exact_pointer_and_hash(paper):
    root, stats = paper
    spec = source_spec(root, "diagnostic.json", {"background": {"auc": .5515}, "other": {"auc": .5515}})
    append_body(root, r"\newcommand{\BackgroundAUC}{0.5515}\BackgroundAUC")
    review = {"version": 1, "sources": {"diagnostic": spec}, "macros": {
        "BackgroundAUC": {"expression": binding.fmt("%.4f", binding.ref("diagnostic", "background", "auc"))}}}
    (root / "numeric_reviews.json").write_text(json.dumps(review))
    report = numbers.audit(root, stats)
    assert report["ALL_PASS"], report["errors"]
    row = next(r for r in report["items"] if r.get("id") == "BackgroundAUC")
    assert row["expression"]["args"][0]["pointer"] == "/background/auc"
    assert row["source_hashes"]["diagnostic"] == spec["sha256"]
    path = root / "diagnostic.json"
    path.write_text(json.dumps({"background": {"auc": .42}, "other": {"auc": .5515}}))
    assert not numbers.audit(root, stats)["ALL_PASS"]


def test_missing_custom_macro_definition_is_not_silently_skipped(paper):
    root, stats = paper
    spec = source_spec(root, "diagnostic.json", {"auc": .55})
    append_body(root, r"\BackgroundDiagnostic")
    reviews = {"version": 1, "sources": {"diagnostic": spec}, "macros": {
        "BackgroundDiagnostic": {"expression": binding.fmt("%.4f", binding.ref("diagnostic", "auc"))}}}
    (root / "numeric_reviews.json").write_text(json.dumps(reviews))
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert next(r for r in report["items"] if r.get("id") == "BackgroundDiagnostic")["status"] == "undefined"


@pytest.mark.parametrize("expression", [
    binding.operation("literal", value=.9123),
    binding.operation("format", binding.ref("diagnostic", "auc"), format="0.9123%.0s"),
    binding.operation("format", format="0.9123"),
])
def test_no_manufactured_numeric_format_evidence(paper, expression):
    root, stats = paper
    spec = source_spec(root, "diagnostic.json", {"auc": .55})
    evidence = binding.Evidence(root, stats, {"diagnostic": spec})
    with pytest.raises((ValueError, TypeError)):
        evidence.binding(expression)


def test_source_derived_scientific_display(paper):
    root, stats = paper
    spec = source_spec(root, "precision.json", {"difference": .000009782})
    evidence = binding.Evidence(root, stats, {"precision": spec})
    result = evidence.binding(binding.operation("tex_scientific", binding.ref("precision", "difference"), digits=2))
    assert result["expected"] == r"9.8\times10^{-6}"


def test_duplicate_source_keys_fail_closed(paper):
    root, stats = paper
    path = stats / "p1c_stats.json"
    data = json.loads(path.read_text())
    data["table"].append(dict(data["table"][0]))
    path.write_text(json.dumps(data))
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert "ambiguous duplicate primary statistic keys" in report["errors"]


def test_semantic_tex_layout_preserves_node_and_caption_claims():
    source = (r"\begin{document}\setlength{\tabcolsep}{4pt}"
              r"\begin{tikzpicture}[scale=1.4,x=1mm]"
              r"\node at (12,32) {AUC 0.9123};\draw (0,0) -- (1,1);"
              r"\end{tikzpicture}\caption{Result 0.9345}"
              r"\begin{tabular}{p{2.1cm}c}\multicolumn{2}{c}{95\% CI}"
              r"\cmidrule{2-3}\label{fig:2026}\texttt{abcdef1234567890abcdef1234567890}"
              r"\end{document}")
    rows = list(binding.literals(source, "main_submission.tex"))
    remaining = [r["value"] for r in rows if r["status"] != "structural"]
    assert remaining == ["0.9123", "0.9345", "95"]
    assert any(r["value"] == "1.4" and r["status"] == "structural" for r in rows)


def test_physical_measurement_is_not_layout():
    rows = list(binding.literals("Measured width was 5mm.", "main_submission.tex"))
    assert rows[0]["status"] == "unresolved"


def test_thousands_and_adjacent_ci_values_are_separate():
    values = [r["value"] for r in binding.literals(r"2{,}000 cases; [0.8024, 0.9342].", "x.tex")]
    assert values == ["2{,}000", "0.8024", "0.9342"]


def literal_entry(root, stats, value):
    report = numbers.audit(root, stats)
    row = next(r for r in report["items"] if r["kind"] == "literal" and r["value"] == value and r["status"] == "unresolved")
    return {key: row[key] for key in ("file", "context_sha256", "token_index", "value")}


def test_context_bound_protocol_review_and_mutation(paper):
    root, stats = paper
    append_body(root, "\n\nProtocol seed 42.\n\n")
    spec = source_spec(root, "protocol.json", {"seed": 42})
    entry = literal_entry(root, stats, "42")
    entry["review"] = {"kind": "protocol", "reviewer": "test reviewer",
                       "rationale": "Recorded configured seed, not a measured outcome.",
                       "evidence": [{"source": "protocol", "pointer": "/seed"}]}
    reviews = {"version": 1, "sources": {"protocol": spec}, "literals": [entry]}
    (root / "numeric_reviews.json").write_text(json.dumps(reviews))
    assert numbers.audit(root, stats)["ALL_PASS"]
    path = root / "main_submission.tex"
    path.write_text("\n\n" + path.read_text())
    assert numbers.audit(root, stats)["ALL_PASS"], "line motion is not semantic change"
    path.write_text(path.read_text().replace("seed 42", "seed 43"))
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert any("stale" in e for e in report["errors"])


def staged_root_fixture(paper):
    root, stats = paper
    append_body(root, "\n\nProtocol seed 42.\n\n"
                + r"$N{=}3000$ volumes (1466 positive / 1534 negative)." + "\n\n"
                + "\\label{tab:finetuned}\n\\begin{tabular}{lcc}\n"
                + r"\textsc{random} & mean-pool & 0.886756 \\" + "\n\\end{tabular}\n")
    (stats / "p1b_full_inventory.json").write_text(json.dumps({"records": [
        {"family": "finetune", "arm": "random", "tag": "finetune_random/mean_pool", "auc": .8867558176556489}]}))
    spec = source_spec(root, "protocol.json", {"seed": 42})
    entry = literal_entry(root, stats, "42")
    entry["review"] = {"kind": "protocol", "reviewer": "test reviewer",
                       "rationale": "Explicit protocol seed, not an empirical result.",
                       "evidence": [{"source": "protocol", "pointer": "/seed"}]}
    review = {"version": 1, "sources": {"protocol": spec}, "literals": [entry]}
    path = root / "numeric_reviews.json"
    path.write_text(json.dumps(review, indent=3))
    return path, review


def test_staged_main_alias_preserves_bindings_receipts_and_physical_hashes(paper):
    root, stats = paper
    review_file, _ = staged_root_fixture(paper)
    approved = review_file.read_bytes()
    original = numbers.audit(root, stats)
    assert original["ALL_PASS"], original["errors"]
    main_hash = original["input_hashes"]["main_submission.tex"]
    (root / "main_submission.tex").rename(root / "main.tex")
    private_review = root.parent / "immutable_review_copy.json"
    private_review.write_bytes(approved)
    staged = numbers.audit(root, stats, private_review)
    assert staged["ALL_PASS"], staged["errors"]
    assert staged["counts"] == original["counts"]
    assert staged["input_hashes"]["main.tex"] == main_hash
    assert "main_submission.tex" not in staged["input_hashes"]
    assert staged["source_aliases"] == {"main.tex": "main_submission.tex"}
    assert private_review.read_bytes() == review_file.read_bytes() == approved
    assert staged["review_sha256"] == original["review_sha256"] == assets.sha256(private_review)
    rows = [r for r in staged["items"] if r["kind"] == "literal"]
    assert all(r["file"] == "main_submission.tex" and r["source_file"] == "main.tex" for r in rows)
    assert next(r for r in rows if r["value"] == "42")["status"] == "reviewed_protocol"
    assert all(next(r for r in rows if r["value"] == value)["status"] == "verified"
               for value in ("3000", "1466", "1534", "0.886756"))


@pytest.mark.parametrize("before,after,status", [
    ("3000", "3001", "mismatch"),
    ("0.886756", "0.886000", "mismatch"),
    ("seed 42", "seed 43", "unresolved"),
    ("Protocol seed", "Changed seed", "unresolved"),
])
def test_staged_root_alias_never_bypasses_changed_values_or_context(paper, before, after, status):
    root, stats = paper
    review_file, _ = staged_root_fixture(paper)
    approved = review_file.read_bytes()
    (root / "main_submission.tex").rename(root / "main.tex")
    path = root / "main.tex"
    path.write_text(path.read_text().replace(before, after))
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert any(r["kind"] == "literal" and r["status"] == status for r in report["items"])
    assert review_file.read_bytes() == approved
    if status == "unresolved":
        assert any("stale or inapplicable" in error for error in report["errors"])


@pytest.mark.parametrize("included", ["main.tex", "appendix/main.tex"])
def test_main_alias_does_not_relabel_included_files(paper, included):
    root, stats = paper
    staged_root_fixture(paper)
    path = root / "main_submission.tex"
    path.write_text(path.read_text().replace("Protocol seed 42.", r"\input{" + included + "}"))
    extra = root / included
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("Protocol seed 42.\n")
    report = numbers.audit(root, stats)
    assert report["source_aliases"] == {}
    assert not report["ALL_PASS"]
    row = next(r for r in report["items"] if r["kind"] == "literal" and r["file"] == included and r["value"] == "42")
    assert row["status"] == "unresolved"


def test_staged_alias_rejects_duplicate_logical_review_keys(paper):
    root, stats = paper
    review_file, review = staged_root_fixture(paper)
    review["literals"].append({**review["literals"][0], "file": "main.tex"})
    review_file.write_text(json.dumps(review))
    (root / "main_submission.tex").rename(root / "main.tex")
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert any("duplicate literal review" in error for error in report["errors"])


def test_unpinned_review_and_new_literal_fail(paper):
    root, stats = paper
    append_body(root, "\n\nProtocol seed 42.\n\n")
    entry = literal_entry(root, stats, "42")
    entry["review"] = {"kind": "protocol", "reviewer": "test", "rationale": "seed",
                       "evidence": [{"source": "p1c_stats.json", "pointer": "/n_test"}]}
    (root / "numeric_reviews.json").write_text(json.dumps({"version": 1, "literals": [entry]}))
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert any("pin its source hash" in r.get("action", "") for r in report["items"])


def test_citation_review_requires_immutable_and_page_locator(paper):
    root, stats = paper
    spec = source_spec(root, "published.json", {"table3": {"accuracy": 84.9}})
    evidence = binding.Evidence(root, stats, {"paper": spec})
    row = {"value": "84.9"}
    entry = {"value": "84.9", "review": {"kind": "citation", "reviewer": "test",
             "rationale": "Exact published ablation cell", "evidence": [{"source": "paper", "pointer": "/table3/accuracy"}]}}
    with pytest.raises(ValueError, match="locator"):
        binding.review_literal(row, entry, evidence)
    entry["review"].update(immutable_locator="retained publication sha256:" + spec["sha256"], locator="Table 3, accuracy column")
    assert binding.review_literal(row, entry, evidence)["status"] == "reviewed_citation"


def test_retained_binary_publication_supports_explicit_human_review(paper):
    root, stats = paper
    path = root / "publication.pdf"
    path.write_bytes(b"%PDF-1.7\n\x80\xff")
    sha = assets.sha256(path)
    evidence = binding.Evidence(root, stats, {"publication": {"root": "paper", "path": path.name, "sha256": sha}})
    entry = {"value": "84.9", "review": {"kind": "citation", "reviewer": "test reviewer",
             "rationale": "Illustrative test of a retained-source human receipt, not PDF extraction.",
             "immutable_locator": "sha256:" + sha, "locator": "Table 3, row random, accuracy",
             "evidence": [{"source": "publication", "pointer": "/sha256"}]}}
    result = binding.review_literal({"value": "84.9"}, entry, evidence)
    assert result["status"] == "reviewed_citation"
    assert evidence.load("publication")["kind"] == "binary_identity_only"


def test_missing_or_wrong_named_macro_cannot_borrow_equal_value(paper):
    root, stats = paper
    append_body(root, r"\newcommand{\AUCOracleEpFifty}{0.8641}\AUCOracleEpFifty")
    report = numbers.audit(root, stats)
    assert not report["ALL_PASS"]
    assert next(r for r in report["items"] if r.get("id") == "AUCOracleEpFifty")["status"] == "unresolved"


def test_explicit_no_result_wrapper_rejected_if_result_exists(paper):
    root, stats = paper
    append_body(root, r"\newcommand{\TAUCAnatomyTwoEpHundred}{---}\TAUCAnatomyTwoEpHundred")
    report = numbers.audit(root, stats)
    assert report["ALL_PASS"]
    assert next(r for r in report["items"] if r.get("id") == "TAUCAnatomyTwoEpHundred")["status"] == "explicit_no_result"
    path = stats / "p1c_stats.json"
    data = json.loads(path.read_text())
    data["table"].append({"key": "anatomy-v2@ep100@fp32", "arm": "anatomy-v2", "epoch": 100, "precision": "fp32", "auc": .85})
    path.write_text(json.dumps(data))
    assert not numbers.audit(root, stats)["ALL_PASS"]


def test_fp32_cell_and_missing_row_fail(paper):
    root, stats = paper
    data = {"rows": [{"arm": "random", "epoch": 50, "auc_fp16": .864097, "auc_fp32": .864121,
                      "delta_fp32_minus_fp16": .000024, "delong_p": .128}]}
    (stats / "p3b_fp32.json").write_text(json.dumps(data))
    evidence = binding.Evidence(root, stats)
    source = r"\textsc{random} & 50 & 0.864097 & 0.864121 & +0.000024 & 0.128 \\" + "\n"
    cells, errors = binding.table_bindings("auto/table_fp32.tex", source, evidence)
    assert not errors and all(x["status"] == "verified" for x in cells.values())
    cells, _ = binding.table_bindings("auto/table_fp32.tex", source.replace("0.864121", "0.864097"), evidence)
    assert any(x["status"] == "mismatch" for x in cells.values())
    _, errors = binding.table_bindings("auto/table_fp32.tex", "", evidence)
    assert errors and "missing" in errors[0]


def test_primary_contrast_header_is_not_a_data_macro(paper):
    root, stats = paper
    source = "\\label{tab:contrasts}\n" + r"contrast & epoch & $\Delta$ AUC & 95\% CI & $p$ & $q$ \\" + "\n"
    cells, errors = binding.table_bindings("main_submission.tex", source, binding.Evidence(root, stats))
    assert not cells and not errors


def test_paired_subgroup_uses_adjusted_not_marginal_ci(paper):
    root, stats = paper
    (stats / "p7c_paired_subgroup.json").write_text(json.dumps({"contrasts": {
        "intensity_minus_random": {"per_group": {"race:Black": {"delta_auc": .01, "ci95_lo": -.01, "ci95_hi": .03}}}}}))
    (stats / "adjusted.json").write_text(json.dumps({"auc_family": {"contrasts": {
        "race:Black": {"simultaneous_ci95_lo": -.02, "simultaneous_ci95_hi": .04}}}}))
    evidence = binding.Evidence(root, stats, {"p17_subgroup_multiplicity.json": {"root": "stats", "path": "adjusted.json"}})
    out = binding.extended_bindings(evidence, {})
    assert out["PDRaceBlackCI"]["expected"] == r"[-0.02000,\,+0.04000]"
    assert out["PDRaceBlackCI"]["expression"]["args"][0]["source"] == "p17_subgroup_multiplicity.json"


def receipt(evidence, item, caption):
    evidence.load("figure-data.json")
    return {"path": item["path"], "sha256": item["sha256"],
            "caption_sha256": binding.digest_text(caption),
            "inputs": [{"source": "figure-data.json", "pointer": "/values",
                        "sha256": evidence.hashes["figure-data.json"]}],
            "validation": {"method": "reviewed_historical_illustration", "reviewer": "test reviewer",
                           "limitations": "Historical illustration; no pixel-level measurement reconstruction.",
                           "quantitative_scope": "illustrative_only"}}


@pytest.mark.parametrize("method", ["reviewed_historical_illustration", "reviewed_source_illustration"])
def test_figure_identity_alone_is_not_numeric_verification(paper, method):
    root, stats = paper
    (stats / "figure-data.json").write_text('{"values": [1, 2]}')
    evidence = binding.Evidence(root, stats)
    item = {"path": "plot.png", "sha256": "a" * 64}
    caption = "Historical example, not measurement evidence."
    with pytest.raises(ValueError):
        binding.figure_receipt(item, {"sha256": item["sha256"], "caption_sha256": binding.digest_text(caption)}, evidence, caption)
    entry = receipt(evidence, item, caption)
    entry["validation"]["method"] = method
    out = binding.figure_receipt(item, entry, evidence, caption)
    assert out["status"] == method
    assert not out["mathematically_verified"]
    with pytest.raises(ValueError, match="changed"):
        binding.figure_receipt({**item, "sha256": "b" * 64}, entry, evidence, caption)
    with pytest.raises(ValueError, match="changed"):
        binding.figure_receipt(item, entry, evidence, caption + " AUC 0.99")
    entry["inputs"][0]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="hash"):
        binding.figure_receipt(item, entry, evidence, caption)


def test_svg_values_independently_checked_and_uncovered_series_fail(paper):
    root, stats = paper
    (stats / "figure-data.json").write_text('{"values": [1, 2], "x": [0, 1]}')
    path = root / "plot.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><polyline id="result" points="0,10 10,20"/></svg>')
    evidence = binding.Evidence(root, stats)
    item = {"path": "plot.svg", "sha256": assets.sha256(path)}
    entry = receipt(evidence, item, "Plot.")
    entry["validation"] = {"method": "svg_coordinates", "series": [{
        "element_id": "result", "x": binding.ref("figure-data.json", "x"),
        "y": binding.ref("figure-data.json", "values"), "x_scale": 10, "x_offset": 0, "y_scale": 10, "y_offset": 0}]}
    assert binding.figure_receipt(item, entry, evidence, "Plot.")["mathematically_verified"]
    path.write_text(path.read_text().replace("10,20", "10,25"))
    item["sha256"] = entry["sha256"] = assets.sha256(path)
    with pytest.raises(ValueError, match="coordinates"):
        binding.figure_receipt(item, entry, evidence, "Plot.")
    path.write_text('<svg><polyline id="result" points="0,10 10,20"/><polyline id="unknown" points="2,3"/></svg>')
    item["sha256"] = entry["sha256"] = assets.sha256(path)
    with pytest.raises(ValueError, match="uncovered"):
        binding.figure_receipt(item, entry, evidence, "Plot.")


def test_matplotlib_artist_mutation_is_not_hidden_by_matching_source(paper):
    import matplotlib.pyplot as plt
    root, stats = paper
    evidence = binding.Evidence(root, stats, binding.BUILTIN_SOURCES)
    data = evidence.load("geometry42")
    fields = ("hidden_frac_of_grid", "ctx_frac_of_grid", "n_slots_mean", "hidden_share_of_all_anat")
    keys = ("random", "oracle", "envelope", "cover", "anatomy")
    labels = ("random", "oracle", "envelope", "cover-f0.21", "anatomy-v2")
    fig, axes = plt.subplots(1, 4)
    for ax, field in zip(axes, fields):
        values = [data[key][field] * (100 if field.endswith("frac_of_grid") else 1) for key in keys]
        ax.bar(range(5), values)
        ax.set_xticks(range(5), labels)
        for i, value in enumerate(values):
            ax.text(i, value, "%.1f" % value)
        ax.set_ylim(0, max(values) * 1.2)
    fig.text(.5, .01, "600 FairVision Training slices (24 volumes x 25), 16x16 grid, seed 42, COVER floor f=0.21; means, no interval drawn.")
    assert len(plots.geometry_artists(fig, evidence)) == 4
    axes[0].patches[0].set_height(99)
    with pytest.raises(ValueError, match="plotted values differ"):
        plots.geometry_artists(fig, evidence)
    plt.close(fig)


def test_matplotlib_replay_requires_actual_delivered_raster_identity(paper):
    root, stats = paper
    evidence = binding.Evidence(root, stats, binding.BUILTIN_SOURCES)
    report = plots.verify_local_plots(root, evidence, {}, [
        {"path": "figures/fig_geometry_panel.png", "sha256": "0" * 64}])
    assert report["figures/fig_geometry_panel.png"]["status"] == "unresolved"
    assert "raster" in report["figures/fig_geometry_panel.png"]["action"]
    json.dumps(report, allow_nan=False)


def test_mismatched_artist_detail_records_remain_json_serializable(monkeypatch):
    import numpy as np
    stats = Path(numbers.INV).parent
    if not stats.is_dir():
        pytest.skip("real local evidence not installed")
    original_equal = plots.equal

    def altered_observation(actual, expected, label):
        if label == "ancestor epoch":
            actual = np.asarray(actual, dtype=np.int64) + 1
        return original_equal(actual, expected, label)

    monkeypatch.setattr(plots, "equal", altered_observation)
    report = numbers.audit(assets.PAPER, stats)
    row = next(r for r in report["items"] if r["id"] == "auto/fig_trajectories_ci.png")
    assert row["status"] == "mismatch"
    assert "observed=[26.0]; expected=[25.0]" in row["action"]
    assert not report["ALL_PASS"]
    json.dumps(report, allow_nan=False)


def test_real_used_p8_macros_have_no_unknown_numeric_bindings():
    stats = Path(numbers.INV).parent
    if not stats.is_dir():
        pytest.skip("real local evidence not installed")
    report = numbers.audit(assets.PAPER, stats)
    unresolved = [r["id"] for r in report["items"] if r["kind"] == "macro" and r["status"] in ("unresolved", "mismatch", "undefined")]
    # New diagnostic macros require the documented custom-binding file. This
    # test covers only definitions actually produced by the P8 source artifact.
    p8_names = {name for name, _, _, _ in assets.macros((assets.PAPER / "auto" / "auto_numbers.tex").read_text())}
    assert not (set(unresolved) & p8_names)
    assert report["checked_auc"] > 0
    assert not [e for e in report["errors"] if e != "asset identity declaration failed"]
    json.dumps(report, allow_nan=False)


def test_new_delivered_diagnostic_macros_use_typed_source_bindings():
    stats = Path(numbers.INV).parent
    reviews = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "delivered_numeric_bindings.json"
    if not stats.is_dir() or not reviews.exists():
        pytest.skip("parent diagnostic evidence not installed")
    report = numbers.audit(assets.PAPER, stats, reviews)
    names = set(json.loads(reviews.read_text())["macros"])
    rows = [r for r in report["items"] if r["kind"] == "macro" and r["id"] in names and r["rendered"]]
    assert rows and all(r["status"] == "verified" and r["source_hashes"] for r in rows)


def test_selected_maps_only_review_is_scoped_and_source_pinned():
    review_file = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "delivered_illustration_review.json"
    if not review_file.exists():
        pytest.skip("selected-case review not installed")
    evidence = binding.Evidence(assets.PAPER, assets.REPO)
    reviews = binding.read_reviews(assets.PAPER, evidence, review_file)
    entry = reviews["figures"][0]
    text, _ = assets.source_tree(assets.PAPER)
    caption = binding.figure_context(text, entry["path"])
    item = {"path": entry["path"], "sha256": assets.sha256(assets.safe_path(assets.PAPER, entry["path"]))}
    result = binding.figure_receipt(item, entry, evidence, caption)
    assert result["status"] == "reviewed_source_illustration"
    assert not result["mathematically_verified"]
    assert evidence.load("delivered_map_manifest")["rendered_metrics"] == []
    selected = []
    for name in ("delivered_map_legacy_records", "delivered_map_guard_records"):
        rows = [json.loads(line) for line in evidence.load(name).splitlines()]
        matches = [r for r in rows if r["ordinal"] == 94]
        assert len(matches) == 1
        selected.append(matches[0])
    assert selected[0]["crop_tensor_sha256"] == selected[1]["crop_tensor_sha256"]
    assert selected[0]["guide_sha256"] == selected[1]["guide_sha256"]
    assert selected[0]["context_tissue"] == 0
    assert selected[1]["context_tissue"] == 16


@pytest.mark.parametrize("value", [True, 3.5, "3"])
def test_integer_binding_cannot_truncate_or_coerce(paper, value):
    root, stats = paper
    spec = source_spec(root, "counts.json", {"n": value})
    evidence = binding.Evidence(root, stats, {"count": spec})
    with pytest.raises(ValueError):
        evidence.binding(binding.fmt("%d", binding.ref("count", "n")))
