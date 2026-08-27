"""Trim the unnecessary ep75 fp32 probes from queue 2.

Rationale
---------
Every cross-precision (confounded) contrast in the manuscript is at epoch 50:
anatomy-v2 and cover-f021 exist only at fp32, and at ep50 they are compared
against an fp16 null. Re-probing random / intensity / envelope at ep50 in fp32
makes those six contrasts precision-matched, which is a real confound removal.

At epochs 75 and 100 only random, intensity and envelope exist and all three are
already fp16, so there is nothing to match against; an fp32 re-probe there only
re-confirms a precision effect already measured at <= 2e-4 by the completed
ep100 trio. That is about 2.5 GPU hours for no claim.

This watcher waits for the last JUSTIFIED queue-2 probe (random ep50) to write
its results.json, then stops the queue-2 driver before it starts the ep75 pair.
Queue 1 is left alone: its remaining item is the COVER epoch-73 probe, which is
new evidence rather than a replication.
"""
import json
import os
import subprocess
import sys
import time

RUNS = r"D:\jepa_phase0\runs"
KEEP = ["frozen_meanpool_oracle_ep50_fp32", "frozen_meanpool_random_ep50_fp32"]
DRIVER_MATCH = "gpu_queue2.py"


def done(name):
    return os.path.exists(os.path.join(RUNS, name, "results.json"))


def driver_pids():
    """PIDs whose command line runs the queue-2 driver (not its probe children)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
          "Where-Object { $_.CommandLine -match 'gpu_queue2\\.py' } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=60).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def main():
    print("[trim] waiting for justified queue-2 probes to finish:", KEEP, flush=True)
    while not all(done(n) for n in KEEP):
        pending = [n for n in KEEP if not done(n)]
        print("[trim] %s still pending: %s" % (time.strftime("%H:%M:%S"), pending), flush=True)
        time.sleep(120)

    print("[trim] both ep50 fp32 probes complete. Stopping queue-2 driver "
          "before it starts the unnecessary ep75 pair.", flush=True)
    for pid in driver_pids():
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Stop-Process -Id %d -Force" % pid],
                           capture_output=True, text=True, timeout=60)
            print("[trim] stopped driver pid %d" % pid, flush=True)
        except Exception as e:
            print("[trim] could not stop pid %d: %s" % (pid, e), flush=True)

    # any probe child still attached to the driver is also stopped
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
          "Where-Object { $_.CommandLine -match 'oracle_ep75_fp32|random_ep75_fp32' } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=60).stdout
    for x in out.split():
        if x.strip().isdigit():
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Stop-Process -Id %s -Force" % x.strip()],
                           capture_output=True, text=True, timeout=60)
            print("[trim] stopped stray ep75 child pid %s" % x.strip(), flush=True)

    print("[trim] done. Queue 1 continues to the COVER ep73 probe.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
