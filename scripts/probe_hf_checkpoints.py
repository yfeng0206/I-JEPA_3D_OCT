"""Re-probe the HF oracle/random checkpoints under the LOCAL fp32 protocol.

WHY THIS EXISTS
---------------
The six oracle/random AUCs in the master table came from a remote sweep on
2026-06-07.  Those runs had no `use_amp` key, and `src/eval_downstream.py:541`
defaults it to True, so the frozen encoder ran under AMP fp16.  All fourteen
local arm probes set `use_amp: False` and ran in true fp32.  The cross-family
comparison is therefore confounded by feature precision.

The stored numbers themselves are NOT in doubt: recomputing roc_auc_score from
the saved *_test_predictions.npz reproduces every one of the six to delta
0.000e+00.  What is in doubt is whether they are COMPARABLE to the local arms.

This script answers that by re-probing the same encoder checkpoints with a
config byte-identical to scripts/probe_blob_epochs.py:make_probe_cfg -- the
same builder that produced every local arm number.  The ONLY thing that
differs from the June runs is precision (fp32) and encode_chunk_size (100 vs
50, a pure batching knob).

No head weights were ever recovered from the remote runs (they were written to
/tmp/ijepa_outputs/... and never copied back), so each probe necessarily
retrains the 2,305-parameter linear head.  The encoder stays frozen throughout.

ORDERING
--------
TARGETS is ordered by importance.  oracle ep100 and random ep100 are the two
numbers the paper's headline claims rest on, so they run first.  Results are
written incrementally, so killing this script after any probe keeps the ones
already finished.  Re-running skips completed probes.

COST: ~60 min and ~2.5 GB of feature cache per checkpoint.  Feature caches are
keyed by encoder checkpoint hash, so nothing is shared between targets.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import yaml

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = pathlib.Path(r"D:\jepa_phase0\runs")
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")
HF = pathlib.Path(r"D:\jepa_phase0\checkpoints_hf")
RESULTS = CAMP / "probe_hf_results.json"
LOG = CAMP / "probe_hf.log"

# (name, checkpoint, remote fp16 AUC to compare against)
TARGETS = [
    ("meanpool_oracle_ep100_fp32",
     HF / "oracle-anatomical-100ep" / "jepa_patch_oracle-ep100.pth.tar",
     0.8854851648),
    ("meanpool_random_ep100_fp32",
     HF / "random-posfix-100ep" / "jepa_patch-ep100.pth.tar",
     0.8745808958),
    ("meanpool_oracle_ep75_fp32",
     HF / "oracle-anatomical-100ep" / "jepa_patch_oracle-ep075.pth.tar",
     0.8836355479),
    ("meanpool_random_ep75_fp32",
     HF / "random-posfix-100ep" / "jepa_patch-ep075.pth.tar",
     0.8723021695),
    ("meanpool_oracle_ep50_fp32",
     HF / "oracle-anatomical-100ep" / "jepa_patch_oracle-ep050.pth.tar",
     0.8740299461),
    ("meanpool_random_ep50_fp32",
     HF / "random-posfix-100ep" / "jepa_patch-ep050.pth.tar",
     0.8640970650),
]


def say(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    CAMP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def make_probe_cfg(ckpt, out_dir, name):
    """Byte-identical to scripts/probe_blob_epochs.py:make_probe_cfg.

    Do not change any value here.  Every local arm AUC was produced with
    exactly this config; altering a knob makes the new number incomparable to
    the fourteen already in the master table.
    """
    cfg = {
        "mode": "patch",
        "data": {
            "data_dir": r"D:\jepa_phase0\fairvision-glaucoma\data",
            "num_slices": 100, "slice_size": 256, "batch_size": 256,
            "num_workers": 4, "encode_chunk_size": 100, "use_amp": False,
        },
        "model": {
            "encoder_checkpoint": str(ckpt), "encoder_name": "vit_base",
            "patch_size": 16, "crop_size": 256, "freeze_encoder": True,
            "probe_type": "mean_pool", "head_type": "linear",
        },
        "training": {
            "lr_probe": 0.0004, "lr_head": 0.0004, "weight_decay": 0.05,
            "dropout": 0.2, "epochs": 50, "patience": 15,
            "warmup_epochs": 5, "seed": 42,
        },
        "logging": {"output_dir": str(out_dir)},
    }
    CAMP.mkdir(parents=True, exist_ok=True)
    p = CAMP / ("probe_%s.yaml" % name)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def record(name, payload):
    data = {}
    if RESULTS.exists():
        try:
            data = json.loads(RESULTS.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    data[name] = payload
    RESULTS.write_text(json.dumps(data, indent=2))


def run_probe(ckpt, name, remote_auc):
    out_dir = RUNS / ("frozen_%s" % name)
    res = out_dir / "results.json"
    if res.exists():
        auc = json.loads(res.read_text()).get("test_auc")
        say("  %s: already done (test_auc=%s)" % (name, auc))
        return auc
    if not ckpt.exists():
        say("  %s: SKIP, checkpoint missing: %s" % (name, ckpt))
        return None

    cfg = make_probe_cfg(ckpt, out_dir, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    say("  %s: launching -> %s" % (name, out_dir))
    t0 = time.time()
    with open(out_dir / "eval.log", "w") as lf:
        rc = subprocess.call(
            [PY, "-u", "src/eval_downstream.py", "--config", str(cfg)],
            cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO)})
    dt = time.time() - t0

    if rc != 0 or not res.exists():
        say("  %s: FAILED rc=%s after %.0fs (see %s)"
            % (name, rc, dt, out_dir / "eval.log"))
        record(name, {"status": "failed", "rc": rc, "wall_s": round(dt)})
        return None

    j = json.loads(res.read_text())
    auc = j.get("test_auc")
    delta = auc - remote_auc
    say("  %s: fp32 test_auc=%.10f  remote fp16=%.10f  delta=%+.10f  (%.0f min)"
        % (name, auc, remote_auc, delta, dt / 60.0))
    record(name, {
        "status": "ok",
        "checkpoint": str(ckpt),
        "fp32_test_auc": auc,
        "remote_fp16_test_auc": remote_auc,
        "delta_fp32_minus_fp16": delta,
        "best_val_auc": j.get("best_val_auc"),
        "best_epoch": j.get("best_epoch"),
        "sensitivity": j.get("sensitivity"),
        "specificity": j.get("specificity"),
        "test_loss": j.get("test_loss"),
        "wall_s": round(dt),
    })
    return auc


def main():
    say("=" * 70)
    say("HF checkpoint re-probe under LOCAL fp32 protocol")
    say("targets: %d" % len(TARGETS))
    for name, ckpt, remote in TARGETS:
        say("-" * 70)
        run_probe(ckpt, name, remote)
    say("=" * 70)
    say("all targets attempted; results -> %s" % RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
