#!/usr/bin/env python
"""End-to-end smoke test for the G1 replication configs.

Launches real training from the locked epoch-25 ancestor with a replication
config, but redirected to a throwaway output folder, waits until the trainer
has written a handful of real iterations to its CSV log, checks the losses are
finite, then stops the process it started and deletes nothing else.

This exists because the six-run chain costs about six days of GPU time.  The
failure modes it catches are all startup-time and all fatal: a missing slice
cache, a missing or mixed guide directory, an ancestor that does not load, a
curriculum mode that no longer exists, or an out-of-memory at the configured
batch size.

Usage:
    python scripts/smoke_replication.py --config configs/replication/rep_centroid_s1234.yaml
    python scripts/smoke_replication.py --all
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
SMOKE_ROOT = pathlib.Path(r"D:\jepa_phase0\runs\_smoke")
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")


def make_smoke_cfg(cfg_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    out = SMOKE_ROOT / cfg_path.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["folder"] = str(out)
    tag = "smoke"
    cfg["logging"]["write_tag"] = tag
    CAMP.mkdir(parents=True, exist_ok=True)
    p = CAMP / f"smoke_{cfg_path.stem}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p, out, tag


def read_rows(csv_path: pathlib.Path):
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", newline="", errors="ignore") as fh:
            return list(csv.DictReader(fh))
    except Exception:  # noqa: BLE001
        return []


def smoke_one(cfg_path: pathlib.Path, want_rows: int, timeout_s: int) -> bool:
    print("=" * 70, flush=True)
    print(f"SMOKE {cfg_path.name}", flush=True)
    smoke_cfg, out, tag = make_smoke_cfg(cfg_path)
    log_path = out / "smoke_stdout.log"
    csv_path = out / f"{tag}-log.csv"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [PY, "-u", "src/train_patch.py", "--config", str(smoke_cfg)],
            cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT, env=env)
    ok = False
    reason = "timed out before any iteration was logged"
    t0 = time.time()
    try:
        while time.time() - t0 < timeout_s:
            if proc.poll() is not None:
                reason = f"trainer exited early rc={proc.returncode}"
                break
            rows = read_rows(csv_path)
            if len(rows) >= want_rows:
                losses = [float(r["loss"]) for r in rows[:want_rows]]
                finite = all(l == l and abs(l) < 1e6 for l in losses)
                positive = all(l > 0 for l in losses)
                ok = finite and positive
                reason = (f"{len(rows)} iterations logged, first losses "
                          f"{['%.5f' % l for l in losses[:5]]}")
                if not ok:
                    reason = "non-finite or non-positive loss: " + reason
                break
            time.sleep(10)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except Exception:  # noqa: BLE001
                proc.kill()

    tail = log_path.read_text(errors="ignore").splitlines()
    for line in tail[:32]:
        print("  | " + line, flush=True)
    if not ok:
        print("  ... tail ...", flush=True)
        for line in tail[-25:]:
            print("  | " + line, flush=True)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} -- {reason}", flush=True)
    print(f"  elapsed {time.time() - t0:.0f}s", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rows", type=int, default=12,
                    help="iterations that must be logged before declaring PASS")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()

    if a.all:
        cfgs = [REPO / "configs" / "replication" / f"rep_{p}_s1234.yaml"
                for p in ("random", "envelope", "centroid")]
    elif a.config:
        cfgs = [pathlib.Path(a.config)]
    else:
        ap.error("pass --config or --all")

    results = {}
    for c in cfgs:
        results[c.name] = smoke_one(c, a.rows, a.timeout)
    print("=" * 70, flush=True)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
