"""Regression cases for the figures_v2 layout guard (V13 P2-1).

Run: python -m pytest autopilot/investigations/camera_ready_20261008/figures_v2/lane_b/test_layout_guard.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle_v2 as st  # noqa: E402


def test_selftest_detects_every_injected_defect():
    cases = st.guard_selftest()
    assert set(cases) == {"clean_layout_passes", "numeral_collision_detected", "title_numeral_collision_detected",
                          "rotated_label_collision_detected", "out_of_canvas_label_detected",
                          "token_digit_collision_detected"}
    assert all(cases.values())


def test_guard_collects_texts_the_first_guard_skipped():
    fig = plt.figure(figsize=(2, 1))
    try:
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        ax.set_title("title")
        ax.set_xlabel("x label")
        st.draw_token(ax, 0.5, 0.5, 0, 0.1, 7, 0.4)
        fig.legend([plt.Line2D([0], [0])], ["legend entry"], loc="upper right")
        texts = {t.get_text() for t in st.visible_texts(fig)}
        assert {"title", "x label", "1", "legend entry"} <= texts
    finally:
        plt.close(fig)


def test_assert_layout_raises_on_collision():
    fig = plt.figure(figsize=(2, 1))
    try:
        fig.text(0.5, 0.5, "A", fontsize=7)
        fig.text(0.51, 0.5, "B", fontsize=7)
        with pytest.raises(AssertionError):
            st.assert_layout(fig)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stem,expected", [("fig_oct_strategies_v2", 61), ("fig_oct_jepa_volume_v2", 69)])
def test_delivered_figures_record_a_complete_clean_guard(stem, expected):
    evidence = json.loads((HERE.parent / (stem + ".evidence.json")).read_text(encoding="utf-8"))
    for variant in ("paper", "poster"):
        guard = evidence["layout_guard"][variant]
        assert guard["texts_checked"] == expected
        assert guard["overlaps"] == [] and guard["outside_canvas"] == []
    assert all(evidence["layout_guard"]["selftest"].values())
    assert evidence["typography"]["pdf_text_min_pt"] >= 6.0


def test_fig3_evidence_records_a_complete_clean_guard():
    evidence = json.loads((HERE / "fig_trajectories_ci.evidence.json").read_text(encoding="utf-8"))
    guard = evidence["layout_guard"]
    assert guard["texts_checked"] == len(evidence["texts"]) and guard["overlaps"] == []
    assert guard["outside_canvas"] == []
    assert evidence["typography"]["min_text_pt"] >= 7.0