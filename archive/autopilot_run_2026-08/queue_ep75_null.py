"""Queue the epoch-75 fp32 null, to run after Phase B releases the GPU.

Why this reverses an earlier decision
-------------------------------------
On 2026-08-23 I cancelled the epoch-75 fp32 re-probes with the reasoning that no
fp32 arm existed at epoch 75, so there was nothing to precision-match against.
That was correct at the time. It stopped being correct when COVER f=0.21 was
probed at epoch 75 (fp32, AUC 0.863858): the manuscript now reports
cover minus random at epoch 75, and that contrast crosses the precision
boundary.

The measured precision effect is at most 2e-4 across six fp16/fp32 pairs, some
forty times smaller than the -0.0084 gap being reported, so the conclusion is
very unlikely to turn on it. But "unlikely to matter" is not the same as
"matched", and every other headline contrast in the paper is matched. This
closes the last one.

Waits for the GPU to be free so it never contends with training.
"""
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = r"D:\jepa_phase0\runs"
RND = r"D:\jepa_phase0\checkpoints_hf\random-posfix-100ep"
LOG = r"D:\jepa_phase0\autopilot_out\ep75_null.log"

TARGET = ("meanpool_random_ep75_fp32", os.path.join(RND, "jepa_patch-ep075.pth.tar"))


def say(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_busy():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
          "Where-Object { $_.CommandLine -match 'train_patch|main_distributed|eval_downstream' } | "
          "Measure-Object | ForEach-Object { $_.Count }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        return True          # fail safe: assume busy


def main():
    name, ckpt = TARGET
    if os.path.exists(os.path.join(RUNS, "frozen_" + name, "results.json")):
        say("%s already done" % name)
        return 0
    if not os.path.exists(ckpt):
        say("ABORT: checkpoint missing %s" % ckpt)
        return 2

    say("waiting for the GPU to be free before the epoch-75 fp32 null")
    t0 = time.time()
    # Poll fast. The only window in which this probe can start is between Phase B
    # releasing the GPU and Phase C claiming it, which is the length of one
    # refresh (about 4-5 minutes). A 300 s poll can miss that window entirely and
    # then sit behind a two-day training run.
    while gpu_busy():
        if (time.time() - t0) / 3600 > 40:
            say("ABORT: waited 40 h, giving up rather than contending")
            return 1
        time.sleep(20)
    say("GPU free, launching %s" % name)
    rc = subprocess.call([PY, "-u", os.path.join(HERE, "run_guarded_probe.py"), name, ckpt])
    say("%s rc=%d" % (name, rc))
    if rc == 0:
        say("refreshing so the epoch-75 contrast becomes precision-matched")
        subprocess.call([PY, "-u", os.path.join(HERE, "refresh_all.py"), "--fast",
                         "--out", r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
