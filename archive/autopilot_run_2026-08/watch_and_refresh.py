"""Wait for the outstanding probes, then refresh everything automatically.

Watches for the probes that still change the manuscript:
  - frozen_meanpool_envelope_fp32_ep75   (robustness triple completes)
  - frozen_meanpool_oracle_ep50_fp32     (removes the H2 cross-precision confound)
  - frozen_meanpool_random_ep50_fp32     (removes the H2 cross-precision confound)
  - frozen_meanpool_cover_f021_ep73      (fills the COVER placeholder in Table 1)

When all four exist, runs the FULL refresh (including the slow subgroup re-run,
because COVER ep73 adds a probe to the subgroup family) and rebuilds the ZIP.

If some probes are still missing after the deadline, it refreshes with whatever
has landed rather than waiting forever, so there is always a current archive.
"""
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = r"D:\jepa_phase0\runs"
OUT = r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip"

WANT = [
    "frozen_meanpool_envelope_fp32_ep75",
    "frozen_meanpool_oracle_ep50_fp32",
    "frozen_meanpool_random_ep50_fp32",
    "frozen_meanpool_cover_f021_ep73",
]
MAX_WAIT_MIN = 300          # hard stop; refresh with what exists
POLL_SEC = 180


def done(n):
    return os.path.exists(os.path.join(RUNS, n, "results.json"))


def main():
    t0 = time.time()
    last = None
    while True:
        have = [n for n in WANT if done(n)]
        missing = [n for n in WANT if not done(n)]
        state = tuple(have)
        if state != last:
            print("[%s] landed %d/%d: %s" % (time.strftime("%H:%M:%S"), len(have), len(WANT),
                                             [n.replace("frozen_meanpool_", "") for n in have]),
                  flush=True)
            last = state
        if not missing:
            print("[watch] all probes landed", flush=True)
            break
        if (time.time() - t0) / 60.0 > MAX_WAIT_MIN:
            print("[watch] deadline reached; refreshing with %d/%d" % (len(have), len(WANT)), flush=True)
            break
        time.sleep(POLL_SEC)

    # GPU should now be idle; a full refresh including the subgroup re-run is safe
    print("[watch] starting FULL refresh", flush=True)
    rc = subprocess.call([PY, "-u", os.path.join(HERE, "refresh_all.py"), "--out", OUT])
    print("[watch] refresh rc=%d" % rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
