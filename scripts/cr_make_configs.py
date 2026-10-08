#!/usr/bin/env python
"""Generate the cr_seed_v1 continuation configs (PLAN Stage 2/3, A2 section 7.1).

    configs/cr_seed_v1/cr_seed_v1_<arm>_s<seed>.yaml   arms random/centroid/envelope, seeds 1234/5678,
                                                       plus random_cb (E9 RANDOM-CB) at seed 1234 only
    configs/cr_seed_v1/config_diffs.json               every key that differs from the arm's
                                                       ARCHIVED run config (must be declared)

Each config starts from the archived configuration that actually produced the
arm's original run (A1 section 1.1), never from a stale template:

    RANDOM    results/pretraining/pretrain_random_posfix/config.yaml (runtime dump)
    CENTROID  results/pretraining/pretrain_oracle_anatomical/config.yaml (runtime dump,
              cross-checked against configs/patch_oracle_anatomical.yaml)
    ENVELOPE  configs/patch_mirage_envelope.yaml, with the shared epoch-25 ancestor
              instead of its resume-ep27 checkpoint
    RANDOM-CB the CENTROID base; differs from the CENTROID config only in
              mask.curriculum.mode/matching and logging folder/tag

and changes ONLY the declared keys below.  Anything else that differs aborts
generation (exit 2).  Seed-5678 configs differ from seed-1234 configs only in
meta.seed, logging.folder and logging.write_tag; the arms differ from each other
only inside mask.curriculum (plus folder/tag/seed).

Usage:
    python scripts/cr_make_configs.py            # (re)write configs + config_diffs.json
    python scripts/cr_make_configs.py --check    # verify files on disk == regenerated; exit 1 if not
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = "cr_seed_v1"
OUT_DIR = REPO / "configs" / CAMPAIGN

SEEDS = (1234, 5678)
ARMS = ("random", "centroid", "envelope")
# E9 RANDOM-CB: the CENTROID config with only mask.curriculum.mode/matching changed;
# pre-registered at seed 1234 only (PREREGISTRATION_cr_seed_v1.md section 1).
CB_ARM = "random_cb"
CB_SEEDS = (1234,)
CB_MODE = "centroid_budget_random"
# strict = match U, L, D, C and the full hidden union H; single-thread collation of a
# real 64-image batch measured at 0.12 s mean / 0.37 s max (impl_i7_random_cb.md).
CB_MATCHING = "strict"
CB_CURRICULUM_KEYS = {"mask.curriculum.mode", "mask.curriculum.matching"}
ALL_ARMS = ARMS + (CB_ARM,)
ARM_SEEDS = {**{arm: SEEDS for arm in ARMS}, CB_ARM: CB_SEEDS}
PLACEHOLDER_ARMS = ()

# One constant for the training loader workers. Coordinator decision 2026-10-08 03:44 (GPU
# benchmark, CR_CAMPAIGN_NOTES.md): 4 workers for all runs (RAM/commit-bound box; 0.343 s/iter).
NUM_WORKERS = 4
VAL_NUM_WORKERS = 2
PREFETCH_FACTOR = 2

ANCESTOR = r"D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\jepa_patch-random_posfix-ep25.pth.tar"
ANCESTOR_SHA256 = "e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b"
ANCESTOR_SIZE = 1507519602
FORK_START_EPOCH = 25
STOP_EPOCH = 50
RUN_ROOT = r"D:\jepa_phase0\runs"
DATA_DIR = r"D:\jepa_phase0\fairvision-glaucoma\data"
SLICE_CACHE_DIR = r"C:\jepa_data\slice_cache"
MIRAGE_GUIDE_DIR = r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides"
ENVELOPE_OVERLAP_FALLBACK = "legacy_uniform_v1"

ARCHIVED = {
    "random": "results/pretraining/pretrain_random_posfix/config.yaml",
    "centroid": "results/pretraining/pretrain_oracle_anatomical/config.yaml",
    "envelope": "configs/patch_mirage_envelope.yaml",
    "random_cb": "results/pretraining/pretrain_oracle_anatomical/config.yaml",
}
# Must be identical to ARCHIVED except for these keys.
CROSSCHECK = {
    "centroid": ("configs/patch_oracle_anatomical.yaml", {"logging.folder"}),
    "random_cb": ("configs/patch_oracle_anatomical.yaml", {"logging.folder"}),
}

# Mask/crop parameters that must be byte-for-byte the archived values.
FROZEN_KEYS = (
    "data.crop_size", "data.crop_scale", "data.num_slices", "data.color_jitter_strength",
    "data.use_color_distortion", "data.use_gaussian_blur", "data.use_horizontal_flip",
    "mask.patch_size", "mask.num_enc_masks", "mask.num_pred_masks", "mask.enc_mask_scale",
    "mask.pred_mask_scale", "mask.aspect_ratio", "mask.allow_overlap", "mask.min_keep",
    "meta.model_name", "meta.pred_depth", "meta.pred_emb_dim",
    "optimization.start_lr", "optimization.final_lr", "optimization.weight_decay",
    "optimization.final_weight_decay",
)


def run_name(arm: str, seed: int) -> str:
    return "%s_%s_s%d" % (CAMPAIGN, arm, seed)


def run_folder(arm: str, seed: int, run_root: str = RUN_ROOT) -> str:
    return os.path.join(str(run_root), run_name(arm, seed))


def write_tag(arm: str, seed: int) -> str:
    return "jepa_patch_" + run_name(arm, seed)


def config_path(arm: str, seed: int) -> Path:
    return OUT_DIR / ("%s.yaml" % run_name(arm, seed))


def common_overrides(arm: str, seed: int) -> dict:
    return {
        "data.batch_size": 64,
        "data.num_workers": NUM_WORKERS,
        "data.val_num_workers": VAL_NUM_WORKERS,
        "data.prefetch_factor": PREFETCH_FACTOR,
        "data.pin_mem": True,
        "data.data_dir": DATA_DIR,
        "data.slice_cache_dir": SLICE_CACHE_DIR,
        "meta.use_bfloat16": False,
        "meta.amp_target": False,
        "meta.seed": seed,
        "meta.load_checkpoint": True,
        "meta.read_checkpoint": ANCESTOR,
        "meta.resume_policy": "fork",
        "meta.fork_start_epoch": FORK_START_EPOCH,
        "optimization.epochs": 100,  # NEVER 50: the schedules depend on the horizon (A2 3.3)
        "optimization.lr": 0.00025,
        "optimization.warmup": 5,
        "optimization.ema": [0.996, 1.0],
        "optimization.ipe_scale": 1.0,
        "optimization.accum_steps": 8,
        "optimization.save_every": 5,
        "optimization.patience": 9999,
        "logging.folder": run_folder(arm, seed),
        "logging.write_tag": write_tag(arm, seed),
    }


ARM_OVERRIDES = {
    "random": {"mask.curriculum": {"enabled": False}},
    "centroid": {
        "mask.curriculum.enc_truncate": "prefix",
        "mask.curriculum.oracle_lateral_frac": 0.6,
    },
    "envelope": {
        "mask.curriculum.enc_truncate": "prefix",
        "mask.curriculum.mirage_guide_dir": MIRAGE_GUIDE_DIR,
        "mask.curriculum.mirage_overlap_fallback": ENVELOPE_OVERLAP_FALLBACK,
    },
}
ARM_OVERRIDES[CB_ARM] = dict(ARM_OVERRIDES["centroid"], **{
    "mask.curriculum.mode": CB_MODE,
    "mask.curriculum.matching": CB_MATCHING,
})
# Keys removed from the archived config, with the reason.
ARM_REMOVALS = {
    # null in the archived RANDOM config and read with default None by the trainer;
    # dropped so the three arms are identical outside mask.curriculum.
    "random": {"meta.pretrained_encoder": "null-valued, no effect; keeps arms identical"},
    "centroid": {},
    "envelope": {},
    "random_cb": {},
}


# ---------------------------------------------------------------------------
# dotted-key helpers
# ---------------------------------------------------------------------------

def flatten(d, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        key = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def set_key(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = copy.deepcopy(value)


def del_key(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    del cur[parts[-1]]


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def diff_flat(old: dict, new: dict) -> dict:
    fo, fn = flatten(old), flatten(new)
    changed = {k: [fo[k], fn[k]] for k in fo if k in fn and _norm(fo[k]) != _norm(fn[k])}
    added = {k: fn[k] for k in fn if k not in fo}
    removed = {k: fo[k] for k in fo if k not in fn}
    return {"changed": changed, "added": added, "removed": removed}


def load_yaml(rel_or_abs) -> dict:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = REPO / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def file_sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def declared_keys(arm: str, seed: int) -> dict:
    ov = dict(common_overrides(arm, seed))
    ov.update(ARM_OVERRIDES[arm])
    return ov


def build_config(arm: str, seed: int, archived: dict | None = None) -> dict:
    if arm in PLACEHOLDER_ARMS:
        raise NotImplementedError("%s is a placeholder until the E9 sampler config keys exist" % arm)
    if arm not in ALL_ARMS:
        raise ValueError("unknown arm %r" % arm)
    if seed not in ARM_SEEDS[arm]:
        raise ValueError("%s is pre-registered only for seeds %s, not %r" % (arm, ARM_SEEDS[arm], seed))
    cfg = copy.deepcopy(archived if archived is not None else load_yaml(ARCHIVED[arm]))
    for key in ARM_REMOVALS[arm]:
        if key in flatten(cfg) or _has_key(cfg, key):
            del_key(cfg, key)
    for key, value in declared_keys(arm, seed).items():
        set_key(cfg, key, value)
    validate_run_config(cfg, arm, seed)
    return cfg


def _has_key(d, dotted):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True


def validate_run_config(cfg: dict, arm: str, seed: int, *, expect_fork: bool = True,
                        run_root: str = RUN_ROOT, ancestor: str = ANCESTOR) -> None:
    """Hard invariants of a cr_seed_v1 training config. Raises ValueError on any violation."""
    errs = []

    def need(key, want, path=False):
        cur = cfg
        for p in key.split("."):
            if not isinstance(cur, dict) or p not in cur:
                errs.append("%s missing (want %r)" % (key, want))
                return
            cur = cur[p]
        if path:
            ok = isinstance(cur, str) and (os.path.normcase(os.path.normpath(cur))
                                           == os.path.normcase(os.path.normpath(str(want))))
        else:
            ok = _norm(cur) == _norm(want)
        if not ok:
            errs.append("%s = %r (want %r)" % (key, cur, want))

    need("optimization.epochs", 100)
    need("optimization.accum_steps", 8)
    need("optimization.ipe_scale", 1.0)
    need("data.batch_size", 64)
    need("data.num_workers", NUM_WORKERS)
    need("meta.amp_target", False)
    need("meta.use_bfloat16", False)
    need("meta.seed", seed)
    need("meta.load_checkpoint", True)
    need("logging.folder", run_folder(arm, seed, run_root), path=True)
    need("logging.write_tag", write_tag(arm, seed))
    if expect_fork:
        need("meta.resume_policy", "fork")
        need("meta.fork_start_epoch", FORK_START_EPOCH)
        need("meta.read_checkpoint", ancestor, path=True)
    else:
        need("meta.resume_policy", "exact")
        if "fork_start_epoch" in cfg.get("meta", {}):
            errs.append("exact resume config must not contain meta.fork_start_epoch")
    if "pred_target_k" in cfg.get("mask", {}):
        errs.append("mask.pred_target_k must be absent")
    if "enc_truncate" in cfg.get("mask", {}):
        errs.append("enc_truncate belongs under mask.curriculum, not mask")
    cur = cfg.get("mask", {}).get("curriculum") or {}
    if arm == "random":
        if cur.get("enabled", False):
            errs.append("RANDOM must not enable a curriculum")
    elif arm == "centroid":
        need("mask.curriculum.enabled", True)
        need("mask.curriculum.mode", "anatomical_prior")
        need("mask.curriculum.enc_truncate", "prefix")
        need("mask.curriculum.oracle_lateral_frac", 0.6)
    elif arm == "envelope":
        need("mask.curriculum.enabled", True)
        need("mask.curriculum.mode", "mirage_envelope")
        need("mask.curriculum.enc_truncate", "prefix")
        need("mask.curriculum.mirage_guide_dir", MIRAGE_GUIDE_DIR)
        need("mask.curriculum.mirage_overlap_fallback", ENVELOPE_OVERLAP_FALLBACK)
    elif arm == CB_ARM:
        # The shadow is production CENTROID: every ramp/oracle knob is pinned.
        need("mask.curriculum.enabled", True)
        need("mask.curriculum.mode", CB_MODE)
        need("mask.curriculum.matching", CB_MATCHING)
        need("mask.curriculum.enc_truncate", "prefix")
        need("mask.curriculum.oracle_lateral_frac", 0.6)
        need("mask.curriculum.oracle_region_frac", 0.28)
        need("mask.curriculum.oracle_row_offset", 0.0)
        need("mask.curriculum.oracle_min_band_rows", 3)
        need("mask.curriculum.T_warm", 25)
        need("mask.curriculum.T_total", 30)
        need("mask.curriculum.r_max", 1.0)
        need("mask.curriculum.ramp_shape", "linear")
        if seed not in CB_SEEDS:
            errs.append("random_cb is pre-registered only for seeds %s" % (CB_SEEDS,))
    else:
        errs.append("unknown arm %r" % arm)
    if errs:
        raise ValueError("invalid %s config for %s s%d: %s" % (CAMPAIGN, arm, seed, "; ".join(errs)))


def make_resume_config(fork_cfg: dict, last_checkpoint: str) -> dict:
    """Exact-resume twin of a fork config: same everything, read -last, no fork fields."""
    cfg = copy.deepcopy(fork_cfg)
    meta = cfg["meta"]
    meta["load_checkpoint"] = True
    meta["read_checkpoint"] = str(last_checkpoint)
    meta["resume_policy"] = "exact"
    meta.pop("fork_start_epoch", None)
    return cfg


def render_yaml(cfg: dict, arm: str, seed: int) -> str:
    header = (
        "# %s  arm=%s  seed=%d\n"
        "# GENERATED by scripts/cr_make_configs.py -- do not hand-edit.\n"
        "# Base: %s (archived run config); diffs vs base in configs/%s/config_diffs.json.\n"
        "# Fork from the shared epoch-25 ancestor (sha256 %s), horizon 100 epochs,\n"
        "# stopped after the verified epoch-%d save by `--stop-after-epoch %d`.\n"
        % (run_name(arm, seed), arm, seed, ARCHIVED[arm], CAMPAIGN, ANCESTOR_SHA256,
           STOP_EPOCH, STOP_EPOCH))
    if arm == CB_ARM:
        header += ("# RANDOM-CB (E9): identical to %s.yaml except mask.curriculum.mode/matching and\n"
                   "# logging folder/tag; uniform placement at the CENTROID shadow's exact budgets.\n"
                   % run_name("centroid", seed))
    body = yaml.dump(cfg, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=120)
    return header + body


class _Dumper(yaml.SafeDumper):
    """Block-style mappings, flow-style scalar lists ([0.3, 1.0])."""


def _repr_list(dumper, data):
    flow = all(not isinstance(x, (list, dict)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def check_declared(arm: str, seed: int, archived: dict, cfg: dict) -> dict:
    d = diff_flat(archived, cfg)
    decl = {k: v for k, v in flatten_decl(declared_keys(arm, seed)).items()}
    removals = set(ARM_REMOVALS[arm])
    undeclared = []
    for k in list(d["changed"]) + list(d["added"]):
        if k not in decl:
            undeclared.append(k)
    for k in d["removed"]:
        if k not in removals:
            undeclared.append(k)
    touched = set(d["changed"]) | set(d["added"]) | set(d["removed"])
    unchanged_declared = sorted(k for k in decl if k not in touched)
    return {"diff": d, "undeclared": sorted(undeclared),
            "declared_but_equal_to_archived": unchanged_declared}


def flatten_decl(decl: dict) -> dict:
    out = {}
    for k, v in decl.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, k))
        else:
            out[k] = v
    return out


def cross_checks(cfgs: dict) -> dict:
    res = {"seed_pairs": {}, "arm_pairs": {}, "frozen_keys": {}, "errors": []}
    for arm in ARMS:
        a, b = cfgs.get((arm, SEEDS[0])), cfgs.get((arm, SEEDS[1]))
        if a is None or b is None:
            continue
        d = diff_flat(a, b)
        keys = sorted(set(d["changed"]) | set(d["added"]) | set(d["removed"]))
        res["seed_pairs"][arm] = keys
        if set(keys) != {"meta.seed", "logging.folder", "logging.write_tag"}:
            res["errors"].append("%s seeds differ in %s" % (arm, keys))
    for seed in SEEDS:
        present = [arm for arm in ALL_ARMS if (arm, seed) in cfgs]
        for i, x in enumerate(present):
            for y in present[i + 1:]:
                a, b = cfgs[(x, seed)], cfgs[(y, seed)]
                d = diff_flat(a, b)
                keys = sorted(set(d["changed"]) | set(d["added"]) | set(d["removed"]))
                res["arm_pairs"]["%s_vs_%s_s%d" % (x, y, seed)] = keys
                bad = [k for k in keys if not (k.startswith("mask.curriculum")
                                               or k in ("logging.folder", "logging.write_tag"))]
                if bad:
                    res["errors"].append("%s vs %s (s%d) differ outside mask.curriculum: %s"
                                         % (x, y, seed, bad))
                if {x, y} == {"centroid", CB_ARM}:
                    want = CB_CURRICULUM_KEYS | {"logging.folder", "logging.write_tag"}
                    if set(keys) != want:
                        res["errors"].append("random_cb vs centroid (s%d) differ in %s, want exactly %s"
                                             % (seed, keys, sorted(want)))
    return res


def generate(write: bool = True) -> tuple[dict, dict]:
    report = {"campaign": CAMPAIGN, "generator": "scripts/cr_make_configs.py",
              "ancestor": {"path": ANCESTOR, "sha256": ANCESTOR_SHA256},
              "num_workers": NUM_WORKERS, "configs": {}, "crosscheck": {}, "errors": []}
    texts = {}
    cfgs = {}
    for arm in ALL_ARMS:
        archived = load_yaml(ARCHIVED[arm])
        if arm in CROSSCHECK:
            other_rel, allowed = CROSSCHECK[arm]
            dd = diff_flat(archived, load_yaml(other_rel))
            keys = set(dd["changed"]) | set(dd["added"]) | set(dd["removed"])
            report["crosscheck"][arm] = {"other": other_rel, "differs_in": sorted(keys)}
            if not keys <= allowed:
                report["errors"].append("%s archived config %s and %s differ in %s"
                                        % (arm, ARCHIVED[arm], other_rel, sorted(keys - allowed)))
        for key in FROZEN_KEYS:
            if not _has_key(archived, key):
                report["errors"].append("%s archived config lacks frozen key %s" % (arm, key))
        if "pred_target_k" in archived.get("mask", {}):
            report["errors"].append("%s archived config has pred_target_k" % arm)
        for seed in ARM_SEEDS[arm]:
            cfg = build_config(arm, seed, archived)
            chk = check_declared(arm, seed, archived, cfg)
            text = render_yaml(cfg, arm, seed)
            if yaml.safe_load(text) != cfg:
                report["errors"].append("%s s%d YAML does not round-trip" % (arm, seed))
            fc = flatten(cfg)
            fa = flatten(archived)
            for key in FROZEN_KEYS:
                if _norm(fc.get(key)) != _norm(fa.get(key)):
                    report["errors"].append("%s s%d changed frozen key %s" % (arm, seed, key))
            name = run_name(arm, seed)
            report["configs"][name] = {
                "arm": arm, "seed": seed,
                "file": "configs/%s/%s.yaml" % (CAMPAIGN, name),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "archived": ARCHIVED[arm],
                "archived_sha256": file_sha256(REPO / ARCHIVED[arm]),
                "changed": chk["diff"]["changed"], "added": chk["diff"]["added"],
                "removed": chk["diff"]["removed"],
                "removal_reasons": ARM_REMOVALS[arm],
                "declared_but_equal_to_archived": chk["declared_but_equal_to_archived"],
                "undeclared": chk["undeclared"],
            }
            if chk["undeclared"]:
                report["errors"].append("%s: undeclared differences %s" % (name, chk["undeclared"]))
            texts[config_path(arm, seed)] = text
            cfgs[(arm, seed)] = cfg
    xc = cross_checks(cfgs)
    report["seed_pair_diffs"] = xc["seed_pairs"]
    report["arm_pair_diffs"] = xc["arm_pairs"]
    report["errors"].extend(xc["errors"])
    report["placeholders"] = {a: "not generated: E9 matched-budget sampler pending"
                              for a in PLACEHOLDER_ARMS}
    report["arm_seeds"] = {arm: list(seeds) for arm, seeds in ARM_SEEDS.items()}
    report_text = json.dumps(report, indent=2, sort_keys=False, default=str) + "\n"
    texts[OUT_DIR / "config_diffs.json"] = report_text
    if write and not report["errors"]:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for p, t in texts.items():
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(t)
    return report, texts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and compare with the files on disk")
    args = ap.parse_args(argv)
    report, texts = generate(write=not args.check)
    if report["errors"]:
        for e in report["errors"]:
            print("ERROR:", e)
        return 2
    if args.check:
        bad = []
        for p, t in texts.items():
            if not p.exists() or p.read_text(encoding="utf-8") != t:
                bad.append(str(p))
        if bad:
            print("STALE or missing generated files:\n  " + "\n  ".join(bad))
            return 1
        print("OK: %d generated files match" % len(texts))
        return 0
    for name, c in report["configs"].items():
        print("%-32s changed=%d added=%d removed=%d undeclared=%d"
              % (name, len(c["changed"]), len(c["added"]), len(c["removed"]), len(c["undeclared"])))
    print("wrote %d files to %s" % (len(texts), OUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
