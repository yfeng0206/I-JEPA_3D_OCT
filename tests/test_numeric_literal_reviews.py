import json
from pathlib import Path

import pytest

from autopilot import numeric_bindings as n
from autopilot import release_assets as assets


@pytest.fixture
def evidence(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"agreement": .0008977946002479698,
                                "spread": .027082803431451863, "fp32": .00019209869604108754,
                                "p": 7.074136299452872e-20, "delta": -9.782803965130427e-6}))
    return n.Evidence(tmp_path, tmp_path, {"source": {"root": "stats", "path": path.name,
                                                    "sha256": assets.sha256(path)}})


def assertion(value, field, bound, relation="le", component=0):
    return {"value": value, "assertion": {"expression": n.ref("source", field), "relation": relation,
                                         "bound": bound, "component": component}}


def test_bound_is_not_rounded_into_success(evidence):
    entry = assertion("0.027", "spread", "0.027")
    result = n.review_literal({"value": "0.027"}, entry, evidence)
    assert result["status"] == "mismatch"
    assert result["computed_value"] > result["bound_value"]
    assert "does not satisfy" in result["action"]


def test_explicit_rounded_point_is_distinct_from_exact_bound(evidence):
    point = {"value": "0.027", "expression": n.fmt("%.3f", n.ref("source", "spread"))}
    assert n.review_literal({"value": "0.027"}, point, evidence)["status"] == "verified"
    assert n.review_literal({"value": "0.027"}, assertion("0.027", "spread", "0.027"), evidence)["status"] == "mismatch"


def test_true_agreement_bound_passes(evidence):
    entry = assertion("0.0009", "agreement", "0.0009")
    result = n.review_literal({"value": "0.0009"}, entry, evidence)
    assert result["status"] == "verified"
    assert result["verification"] == "exact_source_inequality"
    assert result["source_hashes"]


@pytest.mark.parametrize("component,value", [(0, "2"), (1, "10"), (2, "-4")])
def test_scientific_bound_components_share_one_checked_quantity(evidence, component, value):
    entry = assertion(value, "fp32", r"2\times10^{-4}", "lt", component)
    result = n.review_literal({"value": value}, entry, evidence)
    assert result["status"] == "verified"
    assert result["bound_value"] == .0002


def test_power_of_ten_probability_bound(evidence):
    entry = assertion("-10", "p", r"10^{-10}", "lt", 1)
    assert n.review_literal({"value": "-10"}, entry, evidence)["status"] == "verified"


@pytest.mark.parametrize("component,value", [(0, "9.8"), (1, "10"), (2, "-6")])
def test_exact_formatter_components(evidence, component, value):
    entry = {"value": value, "expression": n.operation("tex_scientific",
             n.operation("abs", n.ref("source", "delta")), digits=2), "component": component}
    result = n.review_literal({"value": value}, entry, evidence)
    assert result["status"] == "verified"
    assert result["expected_display"] == r"9.8\times10^{-6}"


def test_cannot_change_bound_while_keeping_old_printed_token(evidence):
    with pytest.raises(ValueError, match="does not match"):
        n.review_literal({"value": "0.027"}, assertion("0.027", "spread", "0.028"), evidence)


@pytest.mark.parametrize("bound,component", [(r"10^{9999}", 0), ("not-a-number", 0), ("0.027", -1), ("0.027", True)])
def test_invalid_assertion_displays_fail_closed(evidence, bound, component):
    with pytest.raises(ValueError):
        n.review_literal({"value": "0.027"}, assertion("0.027", "spread", bound, component=component), evidence)


def test_assertion_cannot_be_combined_with_other_approval_modes(evidence):
    entry = assertion("0.027", "spread", "0.027")
    entry["expression"] = n.fmt("%.3f", n.ref("source", "spread"))
    with pytest.raises(ValueError, match="exactly one"):
        n.review_literal({"value": "0.027"}, entry, evidence)


def test_real_candidate_conforms_to_schema():
    import jsonschema
    root = assets.REPO / "autopilot"
    path = root / "investigations" / "delivered_task" / "evidence" / "literal_review_candidate_61d.json"
    if not path.exists():
        pytest.skip("literal candidate not installed")
    schema = json.loads((root / "numeric_reviews.schema.json").read_text())
    candidate = json.loads(path.read_text())
    jsonschema.Draft202012Validator(schema).validate(candidate)
    assert len(candidate["literals"]) == 382
    assert all(r.get("review", {}).get("reviewer") for r in candidate["literals"] if "review" in r)


def test_snapshot_keeps_net_rank_change_distinct_from_disagreement_count():
    path = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "literal_sources" / "fixed_head_reproduction.json"
    if not path.exists():
        pytest.skip("fixed-head validation snapshot not installed")
    data = json.loads(path.read_text())
    assert data["strict_order_flips"] + data["tie_status_changes"] == data["all_pair_order_disagreements"]
    assert abs(data["net_concordance_pair_equivalents"]) == pytest.approx(abs(data["delta"] * data["pair_count"]), abs=1e-8)
    assert data["all_pair_order_disagreements"] != abs(data["net_concordance_pair_equivalents"])


def test_revised_current_candidate_has_no_unresolved_text_quantities():
    from autopilot import p15_verify_numbers as numbers
    path = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "literal_review_candidate_current.json"
    stats = Path(numbers.INV).parent
    if not path.exists() or not stats.is_dir():
        pytest.skip("local reviewed source/evidence not installed")
    report = numbers.audit(assets.PAPER, stats, path)
    blockers = [item for item in report["items"] if item["kind"] in ("literal", "macro")
                and item["status"] in ("unresolved", "mismatch", "undefined")]
    assert not blockers, [(item["id"], item["value"]) for item in blockers]
    assert not report["errors"]
