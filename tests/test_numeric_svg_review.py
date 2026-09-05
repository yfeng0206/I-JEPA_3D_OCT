import json
import xml.etree.ElementTree as ET

import pytest

from autopilot import numeric_bindings as n
from autopilot import numeric_svg_review as svg
from autopilot import release_assets as assets


@pytest.fixture
def bar_receipt(tmp_path):
    import matplotlib.pyplot as plt

    paper = tmp_path / "paper"
    paper.mkdir()
    source = paper / "means.json"
    # Fixed test data exercise the actual Matplotlib exporter, not a fabricated
    # receipt for the manuscript's PNG/PDF.
    rows = [
        {"policy": "Legacy", "stage": "Scored", "mean": 78.50782221211611, "n": 573},
        {"policy": "Legacy", "stage": "Delivered", "mean": 74.5704824283693, "n": 573},
        {"policy": "V2 + guard", "stage": "Scored", "mean": 78.5604925466725, "n": 573},
        {"policy": "V2 + guard", "stage": "Delivered", "mean": 78.56049256799109, "n": 573},
    ]
    source.write_text(json.dumps({"metrics": rows}))
    path = paper / "bars.svg"
    fig, ax = plt.subplots(figsize=(6, 3))
    positions = [3, 2, 1, 0]
    bars = ax.barh(positions, [r["mean"] for r in rows], height=.62)
    labels = [r["policy"] + "\n" + r["stage"].lower() for r in rows]
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.7, 3.7)
    ax.set_xlabel("Guide mass percent")
    ax.set_title("573 valid views").set_gid("sample-count")
    for i, (bar, row, y) in enumerate(zip(bars, rows, positions)):
        bar.set_gid("mean-%d" % i)
        ax.text(row["mean"] + 1, y, "%.1f" % row["mean"]).set_gid("value-%d" % i)
    zero = ax.transData.transform((0, 0))
    one = ax.transData.transform((1, 1))
    factor = 72 / fig.dpi
    x_scale, x_offset = (one[0] - zero[0]) * factor, zero[0] * factor
    y_scale, y_offset = -(one[1] - zero[1]) * factor, fig.get_figheight() * 72 - zero[1] * factor
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    root = ET.fromstring(path.read_bytes())
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    decorations = []
    for gid, element in ids.items():
        kind = None
        if gid.startswith("patch_"):
            primitive = next(e for e in element if svg.tag(e) == "path")
            _, closed = svg.points_of(primitive)
            kind = "background" if closed else "axis_frame"
        elif gid.startswith("matplotlib.axis_"):
            kind = "axis_layout"
        if kind:
            item = {"element_id": gid, "sha256": svg.subtree_sha256(element), "kind": kind,
                    "reviewer": "test-fixture", "rationale": "Explicit exporter-layout classification for this test."}
            if gid == "matplotlib.axis_1":
                item.update(dimension="x", scale=float(x_scale), offset=float(x_offset))
            decorations.append(item)
    series = [
        {"element_id": "mean-%d" % i, "value": n.ref("means", "metrics", i, "mean"),
         "orientation": "horizontal", "baseline": 0, "category_interval": [y - .31, y + .31],
         "category_label": labels[i], "axis_id": "matplotlib.axis_1",
         "x_scale": float(x_scale), "x_offset": float(x_offset),
         "y_scale": float(y_scale), "y_offset": float(y_offset)}
        for i, y in enumerate(positions)
    ]
    binding_labels = {"value-%d" % i: n.fmt("%.1f", n.ref("means", "metrics", i, "mean")) for i in range(4)}
    binding_labels["sample-count"] = n.fmt("%d valid views", n.ref("means", "metrics", 0, "n"))
    evidence = n.Evidence(paper, paper, {"means": {"root": "paper", "path": source.name, "sha256": assets.sha256(source)}})
    entry = {"path": path.name, "sha256": assets.sha256(path), "caption_sha256": n.digest_text("Measured cohort means."),
             "inputs": [{"source": "means", "pointer": "/metrics", "sha256": assets.sha256(source)}],
             "validation": {"method": "svg_bars", "series": series, "labels": binding_labels, "decorations": decorations}}
    return path, evidence, entry


def verify(fixture):
    path, evidence, entry = fixture
    return n.figure_receipt({"path": path.name, "sha256": assets.sha256(path)}, entry, evidence, "Measured cohort means.")


def rewrite(fixture, mutate, refresh_reviews=False):
    path, _, entry = fixture
    root = ET.fromstring(path.read_bytes())
    mutate(root)
    path.write_bytes(ET.tostring(root))
    entry["sha256"] = assets.sha256(path)
    if refresh_reviews:
        ids = {e.get("id"): e for e in root.iter() if e.get("id")}
        for item in entry["validation"]["decorations"]:
            item["sha256"] = svg.subtree_sha256(ids[item["element_id"]])


def test_actual_matplotlib_bar_paths_and_group_gids_pass(bar_receipt):
    result = verify(bar_receipt)
    assert result["status"] == "programmatically_verified_plotted_values"
    assert result["mathematically_verified"]
    assert len(result["verified_series"]) == 4
    assert result["verified_series"][0]["value"] == 78.50782221211611


@pytest.mark.parametrize("kind", ["value", "missing_series", "wrong_label", "forged_scale", "source_not_in_inputs"])
def test_bar_numeric_mutations_fail(bar_receipt, kind):
    _, _, entry = bar_receipt
    series = entry["validation"]["series"][0]
    if kind == "value":
        series["value"] = n.ref("means", "metrics", 1, "mean")
    elif kind == "missing_series":
        entry["validation"]["series"].pop()
    elif kind == "wrong_label":
        series["category_label"] = "V2 + guard delivered"
    elif kind == "forged_scale":
        series["x_scale"] *= 1.01
    else:
        entry["inputs"] = [{"source": "means", "pointer": "/metrics", "sha256": "0" * 64}]
    with pytest.raises(ValueError):
        verify(bar_receipt)


def test_rehashed_svg_wrong_bar_still_fails(bar_receipt):
    def mutate(root):
        element = next(e for e in root.iter() if e.get("id") == "mean-0")
        primitive = next(e for e in element if svg.tag(e) == "path")
        points, _ = svg.points_of(primitive)
        maximum = max(x for x, y in points)
        points = [(x + 2 if x == maximum else x, y) for x, y in points[:-1]]
        primitive.set("d", "M " + " L ".join("%g %g" % p for p in points) + " z")
    rewrite(bar_receipt, mutate)
    with pytest.raises(ValueError, match="coordinates differ"):
        verify(bar_receipt)


def test_even_rehashed_axis_receipt_cannot_forge_tick_values(bar_receipt):
    def mutate(root):
        tick = next(e for e in root.iter() if e.get("id") == "xtick_2")
        text = next(e for e in tick.iter() if svg.tag(e) == "text")
        text.text = "21"
    rewrite(bar_receipt, mutate, refresh_reviews=True)
    with pytest.raises(ValueError, match="tick labels contradict"):
        verify(bar_receipt)


def test_missing_numeric_annotation_binding_fails(bar_receipt):
    bar_receipt[2]["validation"]["labels"].pop("sample-count")
    with pytest.raises(ValueError, match="uncovered"):
        verify(bar_receipt)


def test_svg_hash_must_match_actual_file_even_for_direct_api(bar_receipt):
    path, evidence, entry = bar_receipt
    old_hash = entry["sha256"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="asset inventory"):
        n.figure_receipt({"path": path.name, "sha256": old_hash}, entry, evidence, "Measured cohort means.")


def test_inventory_and_affine_helpers_do_not_approve_evidence(bar_receipt):
    path, _, entry = bar_receipt
    result = svg.inventory(path)
    assert result["status"] == "inventory_only_not_numeric_verification"
    assert result["sha256"] == entry["sha256"]
    bounds = next(iter(result["clip_boxes"].values()))
    affine = svg.affine_from_bounds(bounds, [0, 100], [-.7, 3.7])
    for key, value in affine.items():
        assert value == pytest.approx(entry["validation"]["series"][0][key], abs=1e-6)


@pytest.mark.parametrize("mutation", ["transform", "hidden", "hidden_attribute", "extra_path", "duplicate_gid"])
def test_uncovered_or_hidden_svg_geometry_fails(bar_receipt, mutation):
    namespace = "{http://www.w3.org/2000/svg}"
    def mutate(root):
        element = next(e for e in root.iter() if e.get("id") == "mean-0")
        if mutation == "transform":
            element.set("transform", "translate(1,0)")
        elif mutation == "hidden":
            element.set("style", "opacity:0.0")
        elif mutation == "hidden_attribute":
            element.set("opacity", "0")
        elif mutation == "extra_path":
            ET.SubElement(element, namespace + "path", {"d": "M 0 0 L 1 1"})
        else:
            ET.SubElement(root, namespace + "g", {"id": "mean-0"})
    rewrite(bar_receipt, mutate)
    with pytest.raises(ValueError):
        verify(bar_receipt)


def test_scope_cannot_substitute_svg_proof_for_png(bar_receipt):
    path, evidence, entry = bar_receipt
    png = path.with_suffix(".png")
    png.write_bytes(b"not-an-svg")
    entry["path"], entry["sha256"] = png.name, assets.sha256(png)
    with pytest.raises(ValueError, match="requires SVG"):
        n.figure_receipt({"path": png.name, "sha256": entry["sha256"]}, entry, evidence, "Measured cohort means.")


def test_bar_cannot_be_reclassified_as_background(bar_receipt):
    path, _, entry = bar_receipt
    root = ET.fromstring(path.read_bytes())
    item = entry["validation"]["series"].pop()
    element = next(e for e in root.iter() if e.get("id") == item["element_id"])
    entry["validation"]["decorations"].append({"element_id": item["element_id"], "sha256": svg.subtree_sha256(element),
        "kind": "background", "reviewer": "test", "rationale": "Incorrect classification must fail."})
    with pytest.raises(ValueError, match="clipped data artist"):
        verify(bar_receipt)


def test_bar_cannot_be_reclassified_as_illustration(bar_receipt):
    path, _, entry = bar_receipt
    root = ET.fromstring(path.read_bytes())
    item = entry["validation"]["series"].pop()
    element = next(e for e in root.iter() if e.get("id") == item["element_id"])
    entry["validation"]["decorations"].append({"element_id": item["element_id"], "sha256": svg.subtree_sha256(element),
        "kind": "reviewed_illustration", "reviewer": "test", "rationale": "Incorrect classification must fail.",
        "limitations": "No quantitative inference", "quantitative_scope": "illustrative_only"})
    with pytest.raises(ValueError, match="unbound bars"):
        verify(bar_receipt)


def test_separate_illustration_keeps_mixed_status_explicit(bar_receipt):
    namespace = "{http://www.w3.org/2000/svg}"
    def mutate(root):
        group = ET.SubElement(root, namespace + "g", {"id": "separate-map"})
        ET.SubElement(group, namespace + "path", {"d": "M 1 1 L 2 1 L 2 2 L 1 2 z"})
    rewrite(bar_receipt, mutate)
    path, _, entry = bar_receipt
    root = ET.fromstring(path.read_bytes())
    group = next(e for e in root.iter() if e.get("id") == "separate-map")
    entry["validation"]["decorations"].append({"element_id": "separate-map", "sha256": svg.subtree_sha256(group),
        "kind": "reviewed_illustration", "reviewer": "test", "rationale": "Separate conceptual test region.",
        "limitations": "This map is not mathematically verified.", "quantitative_scope": "illustrative_only"})
    result = verify(bar_receipt)
    assert result["status"] == "programmatically_verified_plotted_values_with_reviewed_illustration"
    assert not result["mathematically_verified"]
    assert len(result["verified_series"]) == 4
    assert all(row["mathematically_verified"] for row in result["verified_series"])


@pytest.mark.parametrize("data", ["M 0 0 C 1 2 3 4 5 6 z", "M 0 0 L 1", "M 0 0 z M 2 2 z", "M 0 0 L NaN 3 z"])
def test_unsupported_or_malformed_svg_paths_fail(data):
    with pytest.raises(ValueError):
        svg.path_vertices(data)


def test_relative_line_path_is_parsed_without_guessing():
    points, closed = svg.path_vertices("m 10 20 l 3 0 l 0 4 l -3 0 z")
    assert closed
    assert points == [(10, 20), (13, 20), (13, 24), (10, 24), (10, 20)]


def test_generic_coordinates_accept_closed_group_paths(tmp_path):
    path = tmp_path / "shape.svg"
    path.write_text('<svg><g id="shape"><path d="M 0 0 L 2 0 L 2 3 L 0 3 z"/></g></svg>')
    data = tmp_path / "data.json"
    data.write_text('{"x":[0,2,2,0],"y":[0,0,3,3]}')
    evidence = n.Evidence(tmp_path, tmp_path)
    validation = {"method": "svg_coordinates", "series": [{"element_id": "shape", "x": n.ref("data.json", "x"),
        "y": n.ref("data.json", "y"), "x_scale": 1, "x_offset": 0, "y_scale": 1, "y_offset": 0}]}
    result = svg.verify(path, validation, evidence, [{"source": "data.json", "pointer": "/", "sha256": assets.sha256(data)}])
    assert result["mathematically_verified"]
