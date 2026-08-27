"""GPU queue 2: complete the fp32 family for the three H1-critical arms.

Rationale
---------
The paper's primary comparison (random / oracle / envelope) was probed in fp16,
because `eval_downstream.py:541` defaults `use_amp` to True and those configs
omit the key. The anatomy-v2 and cover arms were probed in fp32. Table 1
therefore mixes precisions while claiming a single shared protocol.

Rather than merely disclosing that, we close it: every H1 arm is re-probed at
fp32 under the exact protocol the fp32 arms already used. The fp16 numbers are
retained as a measured robustness check rather than discarded.

All encoders are local:
  random   -> checkpoints_hf/random-posfix-100ep/jepa_patch-ep{050,075,100}.pth.tar
  oracle   -> checkpoints_hf/oracle-anatomical-100ep/jepa_patch_oracle-ep{050,075,100}.pth.tar
  envelope -> runs/patch_mirage_envelope/jepa_patch_mirage-ep{50,75,100}.pth.tar  (queue 1)

`meanpool_oracle_ep100_fp32` deliberately reuses the existing output dir, whose
fp32 feature cache (3 GB, encoder key 52d1a1812356) is already complete from an
earlier run that was interrupted at probe epoch 44/50. That saves ~60 minutes.

Nothing here trains a representation encoder.
"""
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_guarded_probe.py")
ORC = r"D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep"
RND = r"D:\jepa_phase0\checkpoints_hf\random-posfix-100ep"

QUEUE = [
    # epoch 100 first: completes the headline trio
    ("meanpool_oracle_ep100_fp32", os.path.join(ORC, "jepa_patch_oracle-ep100.pth.tar")),
    ("meanpool_random_ep100_fp32", os.path.join(RND, "jepa_patch-ep100.pth.tar")),
    # epoch 50: the matched-epoch primary comparison
    ("meanpool_oracle_ep50_fp32",  os.path.join(ORC, "jepa_patch_oracle-ep050.pth.tar")),
    ("meanpool_random_ep50_fp32",  os.path.join(RND, "jepa_patch-ep050.pth.tar")),
    # epoch 75: completes the trajectory
    ("meanpool_oracle_ep75_fp32",  os.path.join(ORC, "jepa_patch_oracle-ep075.pth.tar")),
    ("meanpool_random_ep75_fp32",  os.path.join(RND, "jepa_patch-ep075.pth.tar")),
]

if __name__ == "__main__":
    t0 = time.time()
    for i, (name, ckpt) in enumerate(QUEUE, 1):
        print("\n" + "=" * 78, flush=True)
        print("[q2 %d/%d] %s   (+%.1f min)" % (i, len(QUEUE), name, (time.time() - t0) / 60), flush=True)
        print("=" * 78, flush=True)
        rc = subprocess.call([PY, "-u", RUNNER, name, ckpt])
        print("[q2] %s rc=%d" % (name, rc), flush=True)
        if rc == 3:
            print("[q2] ABORT: encoder hash changed.", flush=True)
            sys.exit(3)
    print("\n[q2] ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)
