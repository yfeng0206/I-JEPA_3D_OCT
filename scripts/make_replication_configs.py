#!/usr/bin/env python
"""Materialise the six G1 replication configs (3 policies x 2 new seeds).

The Area Chair's decisive objection is that every masking-policy contrast rests
on ONE post-fork continuation per policy, so policy is confounded with
post-fork optimisation noise.  This script writes the configs for two
ADDITIONAL independently randomised continuations per policy, all starting from
the same locked epoch-25 ancestor and all stopping at the same locked endpoint
(epoch 50, the only epoch at which every existing arm already has a frozen
MeanPool probe).

Design rules enforced here, mechanically rather than by hand-editing YAML:

  * EVERY field except ``mask.curriculum`` is identical across the three
    policies.  A hand-maintained trio of YAML files drifts; a generator cannot.
  * The two seeds are the SAME two values in all three policies, so the design
    is paired and a continuation-level paired analysis is available.
  * ``optimization.epochs`` stays 100.  It drives the cosine LR/WD/EMA
    schedules; shortening it to 50 would anneal the LR to ``final_lr`` at the
    endpoint and make these continuations incomparable with the originals.
    The endpoint is enforced by the supervisor's ``--stop_after_epoch 50``.
  * ``meta.amp_target`` is false and ``mask.enc_truncate`` is left at the stock
    ``prefix``: random, oracle/centroid and envelope were all trained that way,
    and at the ~0.002-0.005 AUC effect sizes in play a single arm must not
    differ in how its regression targets are computed.

Policy definitions are copied verbatim from the archived arm configs:
  RANDOM   -- stock uniform multiblock (curriculum disabled), the null that
              configs/patch_vitb16_ep100.yaml ran from epoch 0 to 100.
  ENVELOPE -- configs/patch_mirage_envelope.yaml, mode ``mirage_envelope``.
  CENTROID -- configs/patch_oracle_anatomical.yaml, mode ``anatomical_prior``
              (the arm the paper calls CENTROID; test AUC 0.8854852 at ep100).

Usage:
    python scripts/make_replication_configs.py
    python scripts/make_replication_configs.py --check   # verify, write nothing
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "configs" / "replication"

# Locked epoch-25 ancestor: the common fork point of random, oracle/centroid,
# envelope and cover.  This is the CANONICAL LOCAL FILE, not a re-download.
# Confirmed as the shared fork point by, in order of directness:
#   * configs/patch_cover_random_ep25.yaml -- "The common ancestor of random,
#     oracle, envelope AND cover" written directly above this same path;
#   * configs/patch_mirage_envelope.yaml -- names this path and this SHA-256 as
#     "the verified ep25 fork point" to restart the envelope arm from;
#   * configs/frozen_meanpool_fork_ep25.yaml -- the paper's shared-ancestor
#     probe row is computed on this exact file;
#   * autopilot/TIMELINE_AND_CRITICAL_PATH.md -- "ancestor (shared fork) |
#     fairvision-glaucoma/checkpoint-ep25 (random_posfix) | ep25";
#   * checkpoint-ep25/README.md ties it to training run
#     patch_vit_base_ps16_ep100_bs64_lr0.00025_20260411_063607, which is the
#     blob prefix docs/experiments/pretraining/random_100ep.md records for the
#     random-posfix 100-epoch run that every arm forks from.
# The Hugging Face artifact random-posfix-100ep/jepa_patch-ep025.pth.tar hashes
# to the SAME SHA-256, so the "ep025" and "ep25" names denote identical bytes.
ANCESTOR = (r"D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25"
            r"\jepa_patch-random_posfix-ep25.pth.tar")
ANCESTOR_SHA256 = "e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b"

SEEDS = [1234, 5678]
RUN_ROOT = r"D:\jepa_phase0\runs"

BASE = {
    "data": {
        "batch_size": 64,
        "crop_size": 256,
        "crop_scale": [0.3, 1.0],
        # 6 workers, not more.  32 GB RAM, ~4 GB per worker; 6 was measured at
        # 90-98% RAM.  Raising it pages and makes training slower.
        "num_workers": 6,
        # Validation spawns its own workers on top of the training loader's.
        # 6 + 6 exhausted the Windows commit limit (error 1455) and killed a run.
        "val_num_workers": 2,
        "pin_mem": True,
        "prefetch_factor": 2,
        "data_dir": r"D:\jepa_phase0\fairvision-glaucoma\data",
        # Active slice cache on the SSD.
        "slice_cache_dir": r"C:\jepa_data\slice_cache",
        "num_slices": 100,
        "color_jitter_strength": 0.0,
        "use_color_distortion": False,
        "use_gaussian_blur": False,
        "use_horizontal_flip": False,
    },
    "mask": {
        "patch_size": 16,
        "num_enc_masks": 1,
        "num_pred_masks": 4,
        "enc_mask_scale": [0.85, 1.0],
        "pred_mask_scale": [0.15, 0.2],
        "aspect_ratio": [0.75, 1.5],
        "allow_overlap": False,
        "min_keep": 10,
    },
    "meta": {
        "model_name": "vit_base",
        "pred_depth": 6,
        "pred_emb_dim": 384,
        "use_bfloat16": False,
        "amp_target": False,
        "load_checkpoint": True,
        "read_checkpoint": ANCESTOR,
        "seed": None,
    },
    "optimization": {
        "epochs": 100,
        "lr": 0.00025,
        "start_lr": 0.0001,
        "final_lr": 1.0e-06,
        "warmup": 5,
        "weight_decay": 0.04,
        "final_weight_decay": 0.4,
        "ema": [0.996, 1.0],
        "ipe_scale": 1.0,
        # 64 x 1 GPU x 8 accumulation = 512 effective, matching the original
        # 64 x 4 T4s x 2 that every archived arm used.
        "accum_steps": 8,
        # Milestone checkpoints.  A rolling `<tag>-last.pth.tar` is written
        # EVERY epoch by train_patch.py regardless of this value, so a crash
        # resumes at most one epoch back.
        "save_every": 5,
        "patience": 9999,
    },
    "logging": {"folder": None, "write_tag": None},
}

# ---------------------------------------------------------------------------
# Policy definitions.  These are the ONLY differences between the three arms.
# ---------------------------------------------------------------------------
POLICIES = {
    # Unguided null: stock I-JEPA uniform multiblock via MaskCollator.
    "random": {"enabled": False},
    # Rectangles of the same shape, size and count, rejection-sampled onto the
    # MIRAGE retinal envelope.  Verbatim from configs/patch_mirage_envelope.yaml.
    "envelope": {
        "enabled": True,
        "mode": "mirage_envelope",
        "T_warm": 25,
        "T_total": 30,
        "r_max": 1.0,
        "ramp_shape": "linear",
        "mirage_guide_dir": r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides",
        "mirage_dilate_patches": 0,
        "mirage_min_block_fill": 0.40,
        "mirage_min_retina_visible": 0.25,
        "mirage_max_attempts": 30,
        "mirage_occupancy_threshold": 0.25,
        "mirage_spread": True,
        "mirage_overlap_tolerance": 0.25,
    },
    # Segmentation-free band located by a per-column intensity centroid.
    # Verbatim from configs/patch_oracle_anatomical.yaml.
    "centroid": {
        "enabled": True,
        "mode": "anatomical_prior",
        "T_warm": 25,
        "T_total": 30,
        "r_max": 1.0,
        "ramp_shape": "linear",
        "oracle_region_frac": 0.28,
        "oracle_lateral_frac": 0.6,
        "oracle_row_offset": 0.0,
        "oracle_min_band_rows": 3,
    },
}

HEADER = """# G1 REPLICATION -- {policy_upper}, seed {seed}
#
# GENERATED by scripts/make_replication_configs.py.  Do not hand-edit: the
# three policies are byte-identical outside `mask.curriculum`, and editing one
# file breaks that guarantee silently.
#
# Ancestor : {ancestor}
#            sha256 {sha}
#            The canonical LOCAL fork point shared by random, oracle/centroid,
#            envelope and cover.  Re-verified before every launch.
# Endpoint : epoch 50 (enforced by campaign_supervisor.py --stop_after_epoch 50;
#            optimization.epochs stays 100 so the cosine schedules are unchanged)
# Seed     : {seed}  (meta.seed, read at src/train_patch.py:149; seeds
#            random/numpy/torch/torch.cuda, so crop draws, mask draws, dropout
#            and the DataLoader worker streams all differ between seeds)
"""


def build(policy: str, seed: int) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["mask"]["curriculum"] = copy.deepcopy(POLICIES[policy])
    cfg["meta"]["seed"] = seed
    name = f"rep_{policy}_s{seed}"
    cfg["logging"]["folder"] = f"{RUN_ROOT}\\{name}"
    cfg["logging"]["write_tag"] = f"jepa_patch_{name}"
    return cfg


def path_for(policy: str, seed: int) -> pathlib.Path:
    return OUT_DIR / f"rep_{policy}_s{seed}.yaml"


def all_specs():
    for seed in SEEDS:
        for policy in ("random", "envelope", "centroid"):
            yield policy, seed


def render(policy: str, seed: int) -> str:
    head = HEADER.format(policy_upper=policy.upper(), seed=seed,
                         ancestor=ANCESTOR, sha=ANCESTOR_SHA256)
    return head + yaml.safe_dump(build(policy, seed), sort_keys=False,
                                 default_flow_style=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify on-disk configs match this generator")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for policy, seed in all_specs():
        p = path_for(policy, seed)
        text = render(policy, seed)
        if a.check:
            if not p.exists() or p.read_text() != text:
                print(f"DRIFT: {p}")
                bad += 1
            else:
                print(f"ok:    {p}")
        else:
            p.write_text(text)
            print(f"wrote: {p}")

    # Cross-check the pairing invariant: outside mask.curriculum, all six
    # configs differ only in meta.seed and logging.*.
    ref = build("random", SEEDS[0])
    for policy, seed in all_specs():
        cfg = build(policy, seed)
        for section in ("data", "optimization"):
            if cfg[section] != ref[section]:
                print(f"INVARIANT VIOLATED: {section} differs for {policy}/{seed}")
                bad += 1
        meta = dict(cfg["meta"]); meta.pop("seed")
        rmeta = dict(ref["meta"]); rmeta.pop("seed")
        if meta != rmeta:
            print(f"INVARIANT VIOLATED: meta differs for {policy}/{seed}")
            bad += 1
        mask = dict(cfg["mask"]); mask.pop("curriculum")
        rmask = dict(ref["mask"]); rmask.pop("curriculum")
        if mask != rmask:
            print(f"INVARIANT VIOLATED: mask differs for {policy}/{seed}")
            bad += 1
    print("INVARIANTS OK" if bad == 0 else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
