"""Serialize the permitted GPU probe queue.

Order matters:
  1. envelope fp32 re-probes  - fixes the precision confound under the headline claim
  2. COVER-0.21 epoch 73      - the newest evidence for the in-progress arm

Strictly one GPU job at a time. Nothing here trains a representation encoder.
"""
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_guarded_probe.py")
ENV = r"D:\jepa_phase0\runs\patch_mirage_envelope"
COV = r"D:\jepa_phase0\runs\cover_f021_ep25"

QUEUE = [
    ("meanpool_envelope_fp32_ep100", os.path.join(ENV, "jepa_patch_mirage-ep100.pth.tar")),
    ("meanpool_envelope_fp32_ep50",  os.path.join(ENV, "jepa_patch_mirage-ep50.pth.tar")),
    ("meanpool_envelope_fp32_ep75",  os.path.join(ENV, "jepa_patch_mirage-ep75.pth.tar")),
    # pinned copy of `-last` (epoch 73) so the rolling file cannot mutate under us
    ("meanpool_cover_f021_ep73",     os.path.join(COV, "jepa_patch_cover_f021-ep73-pinned.pth.tar")),
]

if __name__ == "__main__":
    t0 = time.time()
    for i, (name, ckpt) in enumerate(QUEUE, 1):
        print("\n" + "=" * 78, flush=True)
        print("[queue %d/%d] %s   (+%.1f min elapsed)" % (i, len(QUEUE), name, (time.time() - t0) / 60), flush=True)
        print("=" * 78, flush=True)
        rc = subprocess.call([PY, "-u", RUNNER, name, ckpt])
        print("[queue] %s finished rc=%d" % (name, rc), flush=True)
        if rc == 3:
            print("[queue] ABORT: encoder hash changed. Halting queue.", flush=True)
            sys.exit(3)
    print("\n[queue] ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)
