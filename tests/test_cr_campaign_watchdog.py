"""Watchdog: sample rows and alert triggers with mocked readings (nothing real is touched)."""
import csv
import json

from scripts import cr_watchdog as wd


def make(tmp_path, **over):
    vals = dict(ram=8000.0, commit_used=30000.0, commit_limit=40000.0, gpu_mem=11000.0, temp=70.0,
                alive=True, log_size=100, log_mtime=0.0, now=1000.0, disk_c=20.0, disk_d=500.0,
                forbidden=[], suspended=[])
    vals.update(over)

    readers = {
        "memory": lambda: {"ram_avail_mb": vals["ram"], "commit_used_mb": vals["commit_used"],
                           "commit_limit_mb": vals["commit_limit"], "pagefile_used_mb": 10.0,
                           "pagefile_total_mb": 8000.0},
        "gpu": lambda: {"gpu_mem_used_mb": vals["gpu_mem"], "gpu_mem_total_mb": 24576.0, "gpu_util_pct": 99.0,
                        "gpu_temp_c": vals["temp"], "gpu_power_w": 300.0, "gpu_power_limit_w": 315.0},
        "gpu_forbidden": lambda: vals["forbidden"],
        "disk": lambda d: vals["disk_c"] if d.startswith("C") else vals["disk_d"],
        "alive": lambda pid, ct: vals["alive"],
        "progress": lambda log: {"latest_epoch": 27, "latest_iter": "28:150/9375",
                                 "log_size": vals["log_size"], "log_mtime": vals["log_mtime"]},
        "now": lambda: vals["now"],
        "suspended": lambda pid, ct: vals["suspended"],
    }
    (tmp_path / "state.json").write_text(json.dumps({"current": {
        "step": "R1:train", "pid": 4242, "create_time": 1.0, "log": str(tmp_path / "train_a0.log")}}))
    w = wd.Watchdog(tmp_path, readers=readers, realert_s=3600)
    return w, vals


def kinds(alerts):
    return sorted(k for k, _ in alerts)


def test_sample_row_has_all_columns(tmp_path):
    w, _ = make(tmp_path)
    row = w.sample()
    assert set(row) == set(wd.COLUMNS)
    assert row["commit_headroom_mb"] == 10000.0
    assert row["trainer_alive"] is True and row["trainer_pid"] == 4242
    assert row["latest_iter"] == "28:150/9375" and row["latest_epoch"] == 27
    assert row["disk_free_c_gib"] == 20.0 and row["disk_free_d_gib"] == 500.0
    assert row["errors"] == ""
    assert w.evaluate(row) == []


def test_csv_header_written_once(tmp_path):
    w, _ = make(tmp_path)
    w.step()
    w.step()
    rows = list(csv.DictReader(open(tmp_path / "watchdog.csv", newline="")))
    assert len(rows) == 2 and list(rows[0]) == wd.COLUMNS


def test_ram_low_needs_five_consecutive(tmp_path):
    w, v = make(tmp_path, ram=600.0)
    for _ in range(4):
        assert "RAM_LOW" not in kinds(w.evaluate(w.sample()))
    assert "RAM_LOW" in kinds(w.evaluate(w.sample()))
    v["ram"] = 900.0
    w.evaluate(w.sample())
    v["ram"] = 600.0
    for _ in range(4):
        assert "RAM_LOW" not in kinds(w.evaluate(w.sample()))


def test_threshold_alerts(tmp_path):
    w, v = make(tmp_path, commit_used=39000.0, gpu_mem=23.8 * 1024, temp=87.0, disk_c=5.5, disk_d=39.0)
    assert kinds(w.evaluate(w.sample())) == ["COMMIT", "DISK_C", "DISK_D", "GPU_MEM", "GPU_TEMP"]


def test_thresholds_not_triggered_at_safe_edges(tmp_path):
    w, v = make(tmp_path, commit_used=40000.0 - 1600.0, gpu_mem=23.5 * 1024, temp=86.0, disk_c=8.0, disk_d=40.0)
    assert w.evaluate(w.sample()) == []


def test_normal_training_gpu_memory_is_quiet(tmp_path):
    w, _ = make(tmp_path, gpu_mem=21.7 * 1024)
    assert w.evaluate(w.sample()) == []


def test_commit_alert_needs_low_headroom_and_low_c_free(tmp_path):
    # The system-managed pagefile on C: grows the limit on demand, so low headroom alone is normal.
    w, v = make(tmp_path, commit_used=48000.0, commit_limit=49000.0, disk_c=20.0)
    row = w.sample()
    assert row["commit_headroom_mb"] == 1000.0 and row["disk_free_c_gib"] == 20.0
    assert w.evaluate(row) == []
    v["disk_c"] = 5.9
    assert kinds(w.evaluate(w.sample())) == ["COMMIT", "DISK_C"]
    v["commit_used"] = 47000.0  # headroom 2000 MB with C: still low -> only DISK_C (already alerted)
    assert w.evaluate(w.sample()) == []
    w2, v2 = make(tmp_path, commit_used=48000.0, commit_limit=49000.0, disk_c=7.0)
    assert kinds(w2.evaluate(w2.sample())) == ["DISK_C"]


def test_log_stall_only_while_alive(tmp_path):
    w, v = make(tmp_path, now=1000.0, log_mtime=1000.0)
    assert w.evaluate(w.sample()) == []
    v["now"] = 1000.0 + 31 * 60
    alerts = w.evaluate(w.sample())
    assert kinds(alerts) == ["LOG_STALL"]
    v["alive"] = False
    w2, v2 = make(tmp_path, now=1000.0 + 31 * 60, log_mtime=1000.0, alive=False)
    assert w2.evaluate(w2.sample()) == []


def test_log_size_change_counts_as_activity(tmp_path):
    # NTFS may not refresh mtime while the writer holds the file open.
    w, v = make(tmp_path, now=1000.0, log_mtime=0.0, log_size=100)
    w.sample()
    v["now"], v["log_size"] = 1000.0 + 20 * 60, 200
    row = w.sample()
    assert row["seconds_since_log"] == 0.0
    v["now"] = 1000.0 + 20 * 60 + 31 * 60
    assert kinds(w.evaluate(w.sample())) == ["LOG_STALL"]


def test_realert_suppressed_until_cleared(tmp_path):
    w, v = make(tmp_path, temp=90.0)
    assert kinds(w.evaluate(w.sample())) == ["GPU_TEMP"]
    assert w.evaluate(w.sample()) == []
    v["temp"] = 60.0
    assert w.evaluate(w.sample()) == []
    v["temp"] = 90.0
    assert kinds(w.evaluate(w.sample())) == ["GPU_TEMP"]


def test_step_writes_alert_files(tmp_path):
    w, v = make(tmp_path, disk_d=10.0, forbidden=["League of Legends.exe"])
    row, paths = w.step()
    names = sorted(p.split("\\")[-1].split("_")[1] for p in paths)
    assert names == ["DISK", "GPU"]
    assert all((tmp_path / p.split("\\")[-1]).exists() for p in paths)


def test_reader_errors_are_recorded_not_raised(tmp_path):
    w, _ = make(tmp_path)

    def boom():
        raise RuntimeError("nvidia-smi missing")
    w.readers["gpu"] = boom
    row = w.sample()
    assert "nvidia-smi missing" in row["errors"] and row["gpu_temp_c"] is None


def test_trainer_suspended_two_consecutive(tmp_path):
    w, v = make(tmp_path, suspended=[4243])
    row = w.sample()
    assert row["trainer_suspended"] == "4243"
    assert w.evaluate(row) == []
    assert kinds(w.evaluate(w.sample())) == ["TRAINER_SUSPENDED"]
    v["suspended"] = []
    assert w.evaluate(w.sample()) == []


def test_v3_first_sample_with_deferred_mtime_is_not_a_stall(tmp_path):
    # V3 P2-1: a growing, open NTFS log can carry an old mtime; the first sample must not alert.
    w, v = make(tmp_path, now=10000.0, log_mtime=1.0, log_size=100000)
    row = w.sample()
    assert row["seconds_since_log"] == 0.0
    assert "LOG_STALL" not in kinds(w.evaluate(row))
    v["now"] = 10000.0 + 31 * 60  # size unchanged for an observed 31 min -> confirmed stall
    assert kinds(w.evaluate(w.sample())) == ["LOG_STALL"]
