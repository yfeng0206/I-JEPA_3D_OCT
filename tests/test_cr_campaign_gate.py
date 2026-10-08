"""G1 gate checker on synthetic training logs."""
import json
import math

import pytest

from scripts import cr_gate_check as gate

REF = gate.load_reference()


def ref_val(arm, kind, e):
    return REF["arms"][arm][kind][str(e)]


def fp_mean(key):
    s = REF["envelope_mask_fingerprint"]["stats"][key]
    return (s["min"] + s["max"]) / 2.0


def epoch_block(e, train, val, horizon=100, mirage=None, n_mirage=187):
    lines = []
    for it in range(1, 4):
        lines.append("  [Epoch %d/%d | Iter %d/9375] loss=%.4f  lr=2.0e-04  wd=0.1000  ema=0.99700  gpu=9000MB"
                     % (e, horizon, it * 50, train if math.isfinite(train) else float("nan")))
    if mirage:
        for _ in range(n_mirage):
            lines.append("    [MIRAGE] " + "  ".join("%s=%.3f" % (k, v) for k, v in mirage.items()))
    val_s = "  val_loss=%.4f" % val if val is not None else ""
    lines.append("Epoch %d/%d  (3600s)  train_loss=%.4f%s" % (e, horizon, train, val_s))
    return "\n".join(lines) + "\n"


def write_run(tmp_path, arm, epochs, offsets=None, nan_epoch=None, horizon=100, mirage=None,
              attempt=0, start=26, curve_arm=None):
    run = tmp_path / ("run_" + arm)
    run.mkdir(exist_ok=True)
    text = ""
    for e in range(start, start + epochs):
        tr = ref_val(curve_arm or arm, "train", e) + (offsets or {}).get(e, 0.0)
        va = ref_val(curve_arm or arm, "val", e) + (offsets or {}).get(e, 0.0)
        if e == nan_epoch:
            tr = float("nan")
        text += epoch_block(e, tr, va, horizon, mirage(e) if mirage else None)
    (run / ("train_a%d.log" % attempt)).write_text(text, encoding="utf-8")
    return run


def test_on_reference_passes(tmp_path):
    run = write_run(tmp_path, "random", 25)
    res = gate.check_run(run, "random")
    assert res["latest_epoch"] == 50
    assert res["worst_status"] == "PASS", res["summary"]
    assert res["epochs"]["35"]["checks"]["identity"]["status"] == "PASS"


def test_warn_band(tmp_path):
    run = write_run(tmp_path, "random", 3, offsets={27: 0.004})
    res = gate.check_run(run, "random")
    assert res["epochs"]["27"]["status"] == "WARN"
    assert res["epochs"]["26"]["status"] == "PASS"
    assert res["worst_status"] == "WARN"


def test_single_hold_excursion_is_only_warn(tmp_path):
    run = write_run(tmp_path, "centroid", 4, offsets={27: 0.006})
    res = gate.check_run(run, "centroid")
    assert res["epochs"]["27"]["status"] == "WARN"
    assert res["worst_status"] == "WARN"


def test_two_consecutive_over_hold_is_hold(tmp_path):
    run = write_run(tmp_path, "centroid", 4, offsets={27: 0.006, 28: -0.0055})
    res = gate.check_run(run, "centroid")
    assert res["epochs"]["27"]["status"] == "WARN"
    assert res["epochs"]["28"]["status"] == "HOLD"
    assert res["latest_status"] == "PASS" and res["worst_status"] == "HOLD"


def test_non_consecutive_over_hold_is_not_hold(tmp_path):
    run = write_run(tmp_path, "centroid", 5, offsets={27: 0.006, 29: 0.006})
    res = gate.check_run(run, "centroid")
    assert res["worst_status"] == "WARN"


def test_nan_is_stop(tmp_path):
    run = write_run(tmp_path, "envelope", 3, nan_epoch=28)
    res = gate.check_run(run, "envelope")
    assert res["epochs"]["28"]["status"] == "STOP"
    assert res["latest_status"] == "STOP"
    assert gate.main(["--run-dir", str(run), "--arm", "envelope"]) == 4


def test_wrong_horizon_and_start_are_stop(tmp_path):
    run = write_run(tmp_path, "random", 2, horizon=50)
    assert gate.check_run(run, "random")["worst_status"] == "STOP"
    run2 = write_run(tmp_path, "centroid", 2, start=27)
    res = gate.check_run(run2, "centroid")
    assert res["worst_status"] == "STOP" and "first logged epoch" in res["summary"]


def test_identity_check_detects_wrong_policy(tmp_path):
    # A "CENTROID" run whose losses follow the ENVELOPE curve from epoch 26.
    run = write_run(tmp_path, "centroid", 10, curve_arm="envelope")
    res = gate.check_run(run, "centroid")
    ident = res["epochs"]["35"]["checks"]["identity"]
    assert ident["closest"] == "envelope" and ident["status"] == "HOLD"
    assert "30" in res["epochs"] and "identity" not in res["epochs"]["30"]["checks"]


def test_identity_passes_for_own_curve(tmp_path):
    for arm in ("random", "centroid", "envelope"):
        res = gate.check_run(write_run(tmp_path, arm, 10), arm)
        assert res["epochs"]["35"]["checks"]["identity"]["status"] == "PASS", arm


def test_envelope_mask_fingerprint(tmp_path):
    good = {k: fp_mean(k) for k in gate.FINGERPRINT_KEYS}
    good.update({"background": 0.5, "unbiased": 0})
    run = write_run(tmp_path, "envelope", 7, mirage=lambda e: good)
    res = gate.check_run(run, "envelope")
    assert res["epochs"]["31"]["checks"]["mask"]["status"] == "PASS"
    assert res["epochs"]["32"]["checks"]["mask"]["n_lines"] == 187
    assert "mask" not in res["epochs"]["30"]["checks"]


def test_envelope_fix_a_signature_holds(tmp_path):
    fixa = {k: fp_mean(k) for k in gate.FINGERPRINT_KEYS}
    fixa.update({"unique_targets": 132.82, "context": 99.81, "accept": 0.244, "retina_visible": 0.211,
                 "background": 0.5, "unbiased": 0})
    run = write_run(tmp_path, "envelope", 7, mirage=lambda e: fixa)
    res = gate.check_run(run, "envelope")
    m = res["epochs"]["31"]["checks"]["mask"]
    assert m["status"] == "HOLD" and m["stats"]["unique_targets"]["status"] == "HOLD"


def test_envelope_slightly_outside_is_warn(tmp_path):
    s = REF["envelope_mask_fingerprint"]["stats"]["unique_targets"]
    vals = {k: fp_mean(k) for k in gate.FINGERPRINT_KEYS}
    vals.update({"unique_targets": s["max"] + 0.2 * (s["max"] - s["min"]), "background": 0.5, "unbiased": 0})
    run = write_run(tmp_path, "envelope", 6, mirage=lambda e: vals)
    assert gate.check_run(run, "envelope")["epochs"]["31"]["checks"]["mask"]["status"] == "WARN"


def test_missing_mirage_lines_hold(tmp_path):
    run = write_run(tmp_path, "envelope", 6)
    assert gate.check_run(run, "envelope")["epochs"]["31"]["checks"]["mask"]["status"] == "HOLD"


def test_later_attempt_overrides_and_csv_dedup(tmp_path):
    run = write_run(tmp_path, "random", 4, offsets={28: 0.02})
    text = epoch_block(28, ref_val("random", "train", 28), ref_val("random", "val", 28))
    text += epoch_block(29, ref_val("random", "train", 29), ref_val("random", "val", 29))
    (run / "train_a1.log").write_text(text, encoding="utf-8")
    res = gate.check_run(run, "random")
    assert res["epochs"]["28"]["attempt_log"] == "train_a1.log"
    assert res["worst_status"] == "PASS"
    csv_path = run / "x-log.csv"
    csv_path.write_text("epoch,iteration,loss,lr,wd,ema,data_time_ms,forward_time_ms,backward_time_ms,gpu_mem_mb\n"
                        "28,1,0.5,0,0,0,0,0,0,0\n28,2,0.5,0,0,0,0,0,0,0\n"
                        "28,1,0.1,0,0,0,0,0,0,0\n28,2,0.3,0,0,0,0,0,0,0\n", encoding="utf-8")
    means = gate.csv_epoch_means(csv_path)
    assert means[28]["n"] == 2 and abs(means[28]["mean"] - 0.2) < 1e-12


def test_random_cb_is_informational(tmp_path):
    run = write_run(tmp_path, "random_cb", 8, curve_arm="random")
    res = gate.check_run(run, "random_cb")
    assert res["epochs"]["31"]["checks"]["identity"]["status"] == "INFO"
    assert res["worst_status"] == "INFO"


def test_cli_writes_json(tmp_path):
    run = write_run(tmp_path, "random", 2)
    out = tmp_path / "g.json"
    assert gate.main(["--run-dir", str(run), "--arm", "random", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["latest_epoch"] == 27


def test_reference_has_full_ep25_50_curves():
    for arm in ("random", "centroid", "envelope"):
        for kind in ("train", "val"):
            assert set(REF["arms"][arm][kind]) == {str(e) for e in range(25, 51)}, (arm, kind)
    assert REF["arms"]["random"]["train"]["50"] == pytest.approx(0.1413)
    assert REF["arms"]["centroid"]["train"]["50"] == pytest.approx(0.1316)
    assert REF["arms"]["envelope"]["train"]["50"] == pytest.approx(0.1216)
    fp = REF["envelope_mask_fingerprint"]["stats"]["unique_targets"]
    assert fp["min"] == pytest.approx(119.27, abs=0.01) and fp["max"] == pytest.approx(121.14, abs=0.01)


def test_secondary_mask_stat_far_outside_is_only_warn(tmp_path):
    vals = {k: fp_mean(k) for k in gate.FINGERPRINT_KEYS}
    vals.update({"on_region": 0.40, "background": 0.5, "unbiased": 0})
    run = write_run(tmp_path, "envelope", 6, mirage=lambda e: vals)
    m = gate.check_run(run, "envelope")["epochs"]["31"]["checks"]["mask"]
    assert m["stats"]["on_region"]["status"] == "WARN" and not m["stats"]["on_region"]["primary"]
    assert m["status"] == "WARN"
