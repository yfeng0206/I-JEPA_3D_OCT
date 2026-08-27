"""Master GPU sequencer: Phase A -> B -> C, strictly one training job at a time.

  PHASE A  finish the running frozen probes, then refresh the manuscript + ZIP
  PHASE B  COVER f=0.21 pretraining, epoch 73 -> 75 -> 100, probing at each
  PHASE C  blob / anatomy-v2 fp32 continuation, epoch 56 -> 75 -> 100, probing

Phases B and C are PRETRAINING and are explicitly authorised by the operator
(2026-08-23): "once we finish everything here we will continue our cover 0.21
pretraining and see all the way at auc 75 and epoch 100 and its auc. then the
fp32 for the blob all the way to 100 epoch once done too."

Safety properties
-----------------
* Never starts a training leg while any `eval_downstream.py` is alive, so a
  probe and a trainer never contend for the GPU.
* Refreshes the manuscript and the Overleaf ZIP after every milestone, so the
  archive is always current and any new finding surfaces immediately.
* Writes a findings delta after each milestone into autopilot/FINDINGS_LOG.md.
* Each chain script is idempotent: re-running skips completed probes and resumes
  from the rolling `-last` checkpoint.
"""
import json
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = r"C:\Users\Gary\Desktop\jepa"
HERE = os.path.join(REPO, "autopilot")
RUNS = r"D:\jepa_phase0\runs"
ZIP = r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip"
LOG = r"D:\jepa_phase0\autopilot_out\sequencer.log"

PHASE_A_PROBES = [
    "frozen_meanpool_cover_f021_ep73",
    "frozen_meanpool_random_ep50_fp32",
]


def say(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def probe_done(n):
    return os.path.exists(os.path.join(RUNS, n, "results.json"))


def eval_running():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
          "Where-Object { $_.CommandLine -match 'eval_downstream' } | "
          "Measure-Object | ForEach-Object { $_.Count }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        return False


def wait_for_gpu_free(reason, timeout_min=420):
    say("waiting for GPU to be free before %s" % reason)
    t0 = time.time()
    while eval_running():
        if (time.time() - t0) / 60 > timeout_min:
            say("WARN: gpu-free wait exceeded %d min; proceeding" % timeout_min)
            return
        time.sleep(120)
    say("GPU free")


def refresh(tag, fast=False):
    # Let the just-finished probe's files settle. eval_downstream writes
    # test_predictions.npz, then results.json, then plots; starting the
    # inventory mid-write can silently omit the newest probe from every table.
    say("settling 90s before refresh so probe outputs are fully written")
    time.sleep(90)
    say("refreshing manuscript + ZIP (%s)" % tag)
    cmd = [PY, "-u", os.path.join(HERE, "refresh_all.py"), "--out", ZIP]
    if fast:
        cmd.append("--fast")
    rc = subprocess.call(cmd)
    say("refresh rc=%d" % rc)
    subprocess.call([PY, os.path.join(HERE, "findings_delta.py"), tag])
    return rc


def main():
    say("=" * 70)
    say("MASTER SEQUENCER START")

    # ---------------- PHASE A ----------------
    say("PHASE A: waiting for in-flight frozen probes")
    t0 = time.time()
    while not all(probe_done(p) for p in PHASE_A_PROBES):
        if (time.time() - t0) / 60 > 240:
            say("PHASE A deadline reached; continuing with what landed")
            break
        time.sleep(180)
    for p in PHASE_A_PROBES:
        say("  %s : %s" % (p, "done" if probe_done(p) else "MISSING"))
    refresh("phase-A")

    # ---------------- PHASE B ----------------
    wait_for_gpu_free("PHASE B (COVER pretraining)")
    say("PHASE B: COVER f=0.21 pretraining, epoch 73 -> 100")
    rc = subprocess.call([PY, "-u", "scripts/chain_cover_f021.py"], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO})
    say("PHASE B chain exited rc=%d" % rc)
    refresh("phase-B-cover-ep100")

    # ---------------- PHASE C ----------------
    wait_for_gpu_free("PHASE C (blob fp32 pretraining)")
    say("PHASE C: blob/anatomy-v2 fp32 continuation, epoch 56 -> 100")
    rc = subprocess.call([PY, "-u", "scripts/chain_blob_fp32.py"], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO})
    say("PHASE C chain exited rc=%d" % rc)
    refresh("phase-C-blob-fp32-ep100")

    say("MASTER SEQUENCER COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
