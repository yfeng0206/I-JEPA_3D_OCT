#!/usr/bin/env python3
"""Measure what each masking policy actually shows and hides, on real data.

For every arm we report, per image:

    context tokens         how many patches reach the encoder
      - on anatomy         ... of those, how many sit on retinal tissue
      - background         ... how many are off-tissue (black)
    hidden tokens (union)  how many distinct patches are removed as targets
      - on anatomy         ... of those, how many sit on tissue
      - background         ... how many are background
    predictor slots        target cells actually fed to the loss, duplicates
                           included (pred_target_k pads short targets WITH
                           REPLACEMENT, so slots > unique hidden cells)

All arms are scored against the SAME anatomy reference so the comparison is
about masking policy alone.  On FairVision that reference is the MIRAGE guide
occupancy (fraction of the patch covered by retina, thresholded at the
production 0.25).  On GOALS it is the ground-truth segmentation.

Camera-ready protocol (``audit`` command, the default)
-----------------------------------------------------
* Every arm is built from its TRAINED configuration, read explicitly from the
  run config rather than from code defaults:

    random    stock src.masks.multiblock.MaskCollator
    centroid  configs/patch_oracle_anatomical.yaml (anatomical_prior, region
              0.28, lateral 0.6, offset 0, min rows 3)
    envelope  configs/patch_mirage_envelope.yaml, placed on the repaired HARD
              MIRAGE envelope (mirage_guides) with the pre-fc49f61
              ``legacy_uniform_v1`` overlap fallback that trained the run
    anatomy   configs/patch_anatomy_v2.yaml on the soft guide product
    cover     configs/patch_cover_f021_ep25.yaml on the soft guide product

* Sampler sources are executed from ONE pinned git revision (default HEAD),
  not from a possibly half-edited work tree.  Their SHA-256s, the commit and
  the legacy-fallback method are written to the output.
* Crops and every rectangle size are paired across arms, and every arm is
  reseeded identically before placement.  Each audit seed draws
  ``--budget-draws`` independent batches of ``--batch-size`` views.
* Decomposition arms rerun the published (v9) settings on the same crops and
  sizes (CENTROID code-default lateral 0.8; ENVELOPE on the soft guide and/or
  the least-overlap HEAD fallback), so the old-to-new change can be
  attributed.

``reproduce-published`` re-executes the historical probe (548f4d6) and the
pre-camera-ready probe (HEAD blob) in memory and checks the archived table.
``combine`` merges per-seed ``audit`` outputs into one summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

GRID = 16
NPATCH = GRID * GRID
OCC_THRESHOLD = 0.25          # production mirage_occupancy_threshold
CROP = 256
PATCH = 16

BASE_KW = dict(
    input_size=(CROP, CROP),
    patch_size=PATCH,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    nenc=1,
    npred=4,
    min_keep=10,
    allow_overlap=False,
)

CURR_COMMON = dict(T_warm=25, T_total=30, r_max=1.0, ramp_shape="linear")

DATA_ROOT = pathlib.Path(r"D:\jepa_phase0\fairvision-glaucoma")
DATA_DIR = DATA_ROOT / "data"
SLICE_CACHE = DATA_ROOT / "slice_cache"
SOFT_GUIDE_DIR = (DATA_ROOT / "mirage_soft_guides"
                  / "base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy")
HARD_GUIDE_DIR = DATA_ROOT / "mirage_guides"
ARCHIVED_TABLE = (ROOT / "results" / "masking" / "table2_geometry"
                  / "mask_geometry_600slices_bs{bs}_coverf021_seed42.json")
HISTORICAL_PROBE_REV = "548f4d6"
DEFAULT_OUT_DIR = (ROOT / "autopilot" / "investigations" / "camera_ready_20261008"
                   / "mask_audit")

TRAINED_CONFIGS = {
    "centroid": "configs/patch_oracle_anatomical.yaml",
    "envelope": "configs/patch_mirage_envelope.yaml",
    "anatomy": "configs/patch_anatomy_v2.yaml",
    "cover": "configs/patch_cover_f021_ep25.yaml",
}

# name -> how the arm is built.  "guide" is the product used for PLACEMENT;
# scoring always uses the common proxies (soft primary, hard secondary).
ARM_SPECS = {
    "random": dict(kind="random", label="RANDOM", role="trained"),
    "centroid": dict(kind="centroid", config="centroid", label="CENTROID",
                     role="trained"),
    "envelope": dict(kind="guided", config="envelope", guide="hard",
                     fallback="legacy_uniform_v1", label="ENVELOPE",
                     role="trained"),
    "anatomy": dict(kind="guided", config="anatomy", guide="soft",
                    fallback="default", label="ANATOMY-V2", role="trained"),
    "cover": dict(kind="guided", config="cover", guide="soft",
                  fallback="default", label="COVER", role="trained"),
    "centroid_lat08_published": dict(
        kind="centroid", config="centroid", overrides={"oracle_lateral_frac": 0.8},
        label="CENTROID, published audit lateral 0.8 (code default)",
        role="decomposition"),
    "envelope_soft_leastoverlap_published": dict(
        kind="guided", config="envelope", guide="soft", fallback="least_overlap_v2",
        label="ENVELOPE, published audit (soft guide, least-overlap HEAD fallback)",
        role="decomposition"),
    "envelope_hard_leastoverlap": dict(
        kind="guided", config="envelope", guide="hard", fallback="least_overlap_v2",
        label="ENVELOPE, hard guide, least-overlap HEAD fallback",
        role="decomposition"),
    "envelope_soft_legacy": dict(
        kind="guided", config="envelope", guide="soft", fallback="legacy_uniform_v1",
        label="ENVELOPE, soft guide, legacy fallback", role="decomposition"),
}
DEFAULT_ARMS = list(ARM_SPECS)
GAP_REFERENCE = "random"

# --------------------------------------------------------------------------
# Pinned sampler sources
# --------------------------------------------------------------------------
MASK_MODULES = ("utils", "multiblock", "anatomy", "cover", "curriculum")
FIXA_LINE = b"if least.any():"
FIXA_DISABLED = b"if False and least.any():"
# A1's audited pre-FIX-A source (a1_scratch/tree_HEAD_noFixA/src/masks/
# curriculum.py, CRLF checkout).  It disables exactly the same line.
A1_NOFIXA_SHA256_CRLF = "e51cee23d5ea26df62d7c20420679c454461b2d692b25c85bae21739e5ca2e50"

_SAMPLERS = None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return _sha(pathlib.Path(path).read_bytes())


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _crlf(data):
    return _lf(data).replace(b"\n", b"\r\n")


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def install_sampler_sources(rev="HEAD"):
    """Execute ``src/masks`` from one git revision and register it as the
    ``src.masks`` package of this process.

    ``rev="worktree"`` executes the files on disk instead (recorded as such).
    The legacy ENVELOPE fallback uses the native ``mirage_overlap_fallback``
    key when the pinned source has it; otherwise the single FIX-A line is
    disabled in an in-memory copy (A1's reconstruction).
    """
    global _SAMPLERS
    if _SAMPLERS is not None:
        if _SAMPLERS["requested"] != rev:
            raise RuntimeError("Sampler sources already pinned to %r" % _SAMPLERS["requested"])
        return _SAMPLERS
    loaded = sorted(n for n in sys.modules if n == "src.masks" or n.startswith("src.masks."))
    if loaded:
        raise RuntimeError("src.masks was imported before pinning: %s" % loaded)
    head = _git("rev-parse", "HEAD")
    if rev == "worktree":
        commit, origin = head, "worktree"
        dirty = bool(_git("status", "--porcelain", "--", "src/masks"))

        def read(name):
            return (ROOT / "src" / "masks" / f"{name}.py").read_bytes()
    else:
        commit = _git("rev-parse", f"{rev}^{{commit}}")
        origin, dirty = commit, None

        def read(name):
            return subprocess.check_output(
                ["git", "show", f"{commit}:src/masks/{name}.py"], cwd=ROOT)
    import src
    package = types.ModuleType("src.masks")
    package.__path__ = [str(ROOT / "src" / "masks")]
    package.__package__ = "src.masks"
    package.__file__ = f"{origin}:src/masks/__init__.py"
    sys.modules["src.masks"] = package
    src.masks = package
    sources, files = {}, {}
    for name in MASK_MODULES:
        raw = read(name)
        sources[name] = raw
        module = types.ModuleType(f"src.masks.{name}")
        module.__file__ = f"{origin}:src/masks/{name}.py"
        module.__package__ = "src.masks"
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(raw.decode("utf-8"), module.__file__, "exec"), module.__dict__)
        files[f"src/masks/{name}.py"] = dict(
            sha256_exact_bytes=_sha(raw), sha256_lf=_sha(_lf(raw)),
            sha256_crlf_checkout=_sha(_crlf(raw)), bytes=len(raw))
    package.apply_masks = package.utils.apply_masks
    package.MaskCollator = package.multiblock.MaskCollator
    curriculum = package.curriculum
    raw = sources["curriculum"]
    if b"mirage_overlap_fallback" in raw:
        legacy_module = curriculum
        legacy_cfg = {"mirage_overlap_fallback": "legacy_uniform_v1"}
        head_cfg = {"mirage_overlap_fallback": "least_overlap_v2"}
        legacy_record = dict(
            method="native config key mirage_overlap_fallback=legacy_uniform_v1")
    else:
        if raw.count(FIXA_LINE) != 1:
            raise RuntimeError("Cannot locate the single FIX-A fallback line")
        patched = raw.replace(FIXA_LINE, FIXA_DISABLED)
        legacy_module = types.ModuleType("src.masks.curriculum_legacy_uniform_v1")
        legacy_module.__file__ = f"{origin}:src/masks/curriculum.py[FIX-A disabled]"
        legacy_module.__package__ = "src.masks"
        sys.modules[legacy_module.__name__] = legacy_module
        exec(compile(patched.decode("utf-8"), legacy_module.__file__, "exec"),
             legacy_module.__dict__)
        legacy_cfg, head_cfg = {}, {}
        legacy_record = dict(
            method=("in-memory copy of the pinned curriculum.py with the single "
                    "fc49f61 least-overlap line disabled: `if least.any():` -> "
                    "`if False and least.any():`"),
            patched_sha256_lf=_sha(_lf(patched)),
            patched_sha256_crlf_checkout=_sha(_crlf(patched)),
            a1_noFixA_reference_sha256_crlf=A1_NOFIXA_SHA256_CRLF,
            equals_a1_noFixA_reference=_sha(_crlf(patched)) == A1_NOFIXA_SHA256_CRLF)
    _SAMPLERS = dict(
        requested=rev, commit=commit, head=head, origin=origin,
        worktree_masks_dirty=dirty, files=files, legacy_envelope=legacy_record,
        MaskCollator=package.multiblock.MaskCollator,
        CurriculumMaskGenerator=curriculum.CurriculumMaskGenerator,
        MirageMaskCollator=curriculum.MirageMaskCollator,
        legacy_module=legacy_module, legacy_cfg=legacy_cfg, head_cfg=head_cfg)
    return _SAMPLERS


def sampler_record():
    if _SAMPLERS is None:
        return None
    return {k: _SAMPLERS[k] for k in ("requested", "commit", "head", "origin",
                                      "worktree_masks_dirty", "files",
                                      "legacy_envelope")}


def _classes():
    """Pinned classes if installed, else the work-tree modules (legacy use)."""
    if _SAMPLERS is not None:
        return _SAMPLERS
    from src.masks.curriculum import CurriculumMaskGenerator, MirageMaskCollator
    from src.masks.multiblock import MaskCollator
    return dict(MaskCollator=MaskCollator, CurriculumMaskGenerator=CurriculumMaskGenerator,
                MirageMaskCollator=MirageMaskCollator)


# --------------------------------------------------------------------------
# Trained configurations
# --------------------------------------------------------------------------
_GEOMETRY_KEYS = dict(patch_size="patch_size", num_enc_masks="nenc",
                      num_pred_masks="npred", enc_mask_scale="enc_mask_scale",
                      pred_mask_scale="pred_mask_scale", aspect_ratio="aspect_ratio",
                      allow_overlap="allow_overlap", min_keep="min_keep")


def load_trained_config(arm):
    """Mask/curriculum settings of a trained run, read from its YAML."""
    path = ROOT / TRAINED_CONFIGS[arm]
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    mask, data = cfg["mask"], cfg["data"]
    for key, kw in _GEOMETRY_KEYS.items():
        value, expected = mask[key], BASE_KW[kw]
        if isinstance(expected, tuple):
            value = tuple(value)
        if value != expected:
            raise ValueError(f"{path}: mask.{key}={value!r}, audit uses {expected!r}")
    if (data["crop_size"] != CROP or tuple(data.get("crop_scale", (0.3, 1.0))) != (0.3, 1.0)
            or data.get("use_gaussian_blur") or data.get("use_horizontal_flip")
            or data.get("use_color_distortion") or data.get("color_jitter_strength", 0.)):
        raise ValueError(f"{path}: data augmentation differs from the audit transform")
    curriculum = dict(mask["curriculum"])
    if curriculum.get("mode") in ("mirage_envelope", "mirage_anatomy", "mirage_cover"):
        if (int(curriculum.get("mirage_dilate_patches", 1)) != 0
                or float(curriculum.get("mirage_occupancy_threshold", .5)) != OCC_THRESHOLD):
            raise ValueError(f"{path}: guide dataset settings differ from the audit dataset")
    return dict(path=TRAINED_CONFIGS[arm], sha256_lf=_sha(_lf(raw)),
                sha256_exact_bytes=_sha(raw), curriculum=curriculum,
                pred_target_k=mask.get("pred_target_k"))


def build_audit_arm(name, guide_dirs=None):
    """Instantiate one arm at full guidance (r_t = 1) and describe it."""
    spec = ARM_SPECS[name]
    classes = _classes()
    if spec["kind"] == "random":
        return classes["MaskCollator"](**BASE_KW), dict(
            spec, sampler="src.masks.multiblock.MaskCollator", config=None)
    trained = load_trained_config(spec["config"])
    curriculum = dict(trained["curriculum"])
    original_guide_dir = curriculum.get("mirage_guide_dir")
    if spec.get("guide"):
        curriculum["mirage_guide_dir"] = str((guide_dirs or {}).get(
            spec["guide"], HARD_GUIDE_DIR if spec["guide"] == "hard" else SOFT_GUIDE_DIR))
    curriculum.update(spec.get("overrides", {}))
    cls = classes["CurriculumMaskGenerator"]
    fallback = spec.get("fallback")
    if fallback == "legacy_uniform_v1":
        if _SAMPLERS is None:
            raise RuntimeError("legacy_uniform_v1 requires install_sampler_sources()")
        cls = _SAMPLERS["legacy_module"].CurriculumMaskGenerator
        curriculum.update(_SAMPLERS["legacy_cfg"])
    elif fallback == "least_overlap_v2" and _SAMPLERS is not None:
        curriculum.update(_SAMPLERS["head_cfg"])
    arm = cls(**BASE_KW, pred_target_k=trained["pred_target_k"], curriculum_cfg=curriculum)
    arm.set_epoch(50, 100)
    if arm._r_t != 1.0:
        raise RuntimeError(f"{name}: guidance not at full strength")
    if spec["kind"] == "centroid":
        expected = spec.get("overrides", {}).get("oracle_lateral_frac", 0.6)
        if (arm.oracle_lateral_frac != expected or arm.oracle_region_frac != 0.28
                or arm.oracle_row_offset != 0.0 or arm.oracle_min_band_rows != 3):
            raise RuntimeError(f"{name}: CENTROID band settings are not the intended ones")
    record = dict(spec, sampler=f"{cls.__module__}.{cls.__name__}", config=trained["path"],
                  config_sha256_lf=trained["sha256_lf"],
                  config_mirage_guide_dir=original_guide_dir,
                  curriculum_cfg=curriculum, pred_target_k=trained["pred_target_k"],
                  oracle_lateral_frac=getattr(arm, "oracle_lateral_frac", None)
                  if spec["kind"] == "centroid" else None)
    return arm, record


# --------------------------------------------------------------------------
# Published (v9) audit helpers, kept for reproduction and old callers
# --------------------------------------------------------------------------
def build_arms(guide_dir, cover_floor=0.15):
    """PUBLISHED v9 audit arms -- NOT the trained configuration.

    CENTROID uses the code default lateral fraction 0.8 (trained: 0.6), and
    ENVELOPE is placed on whatever guide the caller passes (the published
    table passed the soft product; trained: the hard envelope, legacy
    fallback).  Use ``build_audit_arm`` for the trained settings.
    """
    classes = _classes()
    MaskCollator = classes["MaskCollator"]
    CurriculumMaskGenerator = classes["CurriculumMaskGenerator"]
    arms = {}
    arms["random"] = MaskCollator(**BASE_KW)

    arms["oracle"] = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(mode="anatomical_prior", **CURR_COMMON),
    )
    arms["envelope"] = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(
            mode="mirage_envelope", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25, **CURR_COMMON,
        ),
    )
    arms["anatomy"] = CurriculumMaskGenerator(
        **BASE_KW,
        pred_target_k=16,
        curriculum_cfg=dict(
            mode="mirage_anatomy", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25,
            anatomy_mass_cap=0.9, anatomy_tau=0.1,
            anatomy_bridge_diagonals=True, **CURR_COMMON,
        ),
    )
    arms["cover"] = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(
            mode="mirage_cover", mirage_guide_dir=guide_dir,
            mirage_occupancy_threshold=OCC_THRESHOLD,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25,
            anatomy_tau=0.1,
            cover_leave_frac=cover_floor, cover_min_visible_frac=cover_floor,
            cover_min_visible_cells=4, cover_transition=True, **CURR_COMMON,
        ),
    )
    for name, a in arms.items():
        if hasattr(a, "set_epoch"):
            a.set_epoch(50, 100)          # well past T_total -> r_t = r_max = 1
    return arms


def run_arm(name, arm, images, guides, valid, block_sizes=None):
    """Return (list_of_context_index_sets, list_of_target_index_lists)."""
    B = images.size(0)
    if name == "random":
        _, m_enc, m_pred = arm([images[i] for i in range(B)], block_sizes=block_sizes)
    elif name == "oracle":
        m_enc, m_pred = arm.generate(batch_size=B, imgs_cpu=images, block_sizes=block_sizes)
    else:
        m_enc, m_pred = arm.generate(
            batch_size=B, guide_grids=guides, guide_valid=valid, block_sizes=block_sizes
        )
    ctx = [set(m_enc[0][b].tolist()) for b in range(B)]
    blocks = []
    for b in range(B):
        blocks.append([g[b].tolist() for g in m_pred])
    return ctx, blocks


def score_image(ctx, blocks, on_anat_flat):
    """Per-mask and per-image splits of context / masked tokens."""
    ci = np.fromiter(ctx, dtype=int, count=len(ctx))
    c_on = int(on_anat_flat[ci].sum()) if ci.size else 0

    # ---- per individual target mask (npred of them) ----
    per_tok, per_on, per_uniq, per_uniq_on = [], [], [], []
    for blk in blocks:
        a = np.asarray(blk, dtype=int)
        u = np.unique(a)
        per_tok.append(a.size)
        per_on.append(int(on_anat_flat[a].sum()))
        per_uniq.append(u.size)
        per_uniq_on.append(int(on_anat_flat[u].sum()))

    slots = np.concatenate([np.asarray(b, dtype=int) for b in blocks])
    hid = np.unique(slots)
    h_on = int(on_anat_flat[hid].sum()) if hid.size else 0

    return dict(
        # context
        n_ctx=int(ci.size),
        ctx_on_anat=c_on,
        ctx_bg=int(ci.size) - c_on,
        # PER MASK (one target block)
        mask_tokens=float(np.mean(per_tok)),
        mask_on_anat=float(np.mean(per_on)),
        mask_bg=float(np.mean(per_tok) - np.mean(per_on)),
        mask_uniq_cells=float(np.mean(per_uniq)),
        mask_uniq_on_anat=float(np.mean(per_uniq_on)),
        mask_uniq_bg=float(np.mean(per_uniq) - np.mean(per_uniq_on)),
        # per image, union over the npred masks
        n_hidden=int(hid.size),
        hid_on_anat=h_on,
        hid_bg=int(hid.size) - h_on,
        n_slots=int(slots.size),
        slot_dupes=int(slots.size - hid.size),
        anat_cells=int(on_anat_flat.sum()),
    )


def aggregate(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    out = {"n_images": len(rows)}
    for k, v in arr.items():
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sd"] = float(v.std())
        out[f"{k}_quantiles"] = np.quantile(v, [0, .05, .5, .95, 1]).tolist()
    # Derived rates, computed from the means so they read as percentages.
    ctx, hid = arr["n_ctx"].mean(), arr["n_hidden"].mean()
    out["ctx_frac_of_grid"] = float(ctx / NPATCH)
    out["hidden_frac_of_grid"] = float(hid / NPATCH)
    out["ctx_pct_on_anat"] = float(arr["ctx_on_anat"].mean() / ctx * 100) if ctx else 0.0
    out["hidden_pct_on_anat"] = float(arr["hid_on_anat"].mean() / hid * 100) if hid else 0.0
    # What share of all anatomy cells in the image does each set cover?
    anat = arr["anat_cells"].mean()
    out["anat_cells_mean"] = float(anat)
    out["ctx_share_of_all_anat"] = float(arr["ctx_on_anat"].mean() / anat * 100) if anat else 0.0
    out["hidden_share_of_all_anat"] = float(arr["hid_on_anat"].mean() / anat * 100) if anat else 0.0
    return out


# --------------------------------------------------------------------------
# Camera-ready audit
# --------------------------------------------------------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def make_dataset(guide_dir, split="Training"):
    return GuidedOCTSliceDataset(
        data_dir=str(DATA_DIR / split), guide_dir=str(pathlib.Path(guide_dir) / split),
        num_slices=100, slice_size=CROP,
        transform=make_paired_transforms(
            crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
            horizontal_flip=False, color_distortion=False, color_jitter=0.0),
        patch_size=PATCH, dilate_patches=0, occupancy_threshold=OCC_THRESHOLD,
        slice_cache=str(SLICE_CACHE / split))


def select_views(n_files, volumes, slices_per_volume, selection_seed=42, num_slices=100):
    vols = sorted(random.Random(selection_seed).sample(range(n_files), volumes))
    step = max(1, num_slices // slices_per_volume)
    views = [v * num_slices + s for v in vols for s in range(0, num_slices, step)]
    return vols, views[:volumes * slices_per_volume]


def deliver(spec, arm, images, guides, valid, sizes):
    """Delivered (context [B, C], targets [npred, B, K]) for one batch."""
    if spec["kind"] == "random":
        _, enc, pred = arm(list(images), block_sizes=sizes)
    elif spec["kind"] == "centroid":
        enc, pred = arm.generate(len(images), imgs_cpu=images, block_sizes=sizes)
    else:
        product = spec["guide"]
        enc, pred = arm.generate(len(images), guide_grids=guides[product],
                                 guide_valid=valid[product], block_sizes=sizes)
    if len(enc) != BASE_KW["nenc"] or len(pred) != BASE_KW["npred"]:
        raise ValueError("Wrong context/target group count")
    context = enc[0].cpu()
    targets = torch.stack([p.cpu() for p in pred])
    B = len(images)
    if context.shape[0] != B or targets.shape[1] != B or context.dtype != torch.long:
        raise ValueError("Invalid delivered tensor contract")
    if int(context.min()) < 0 or int(context.max()) >= NPATCH:
        raise ValueError("Out-of-range context index")
    if int(targets.min()) < 0 or int(targets.max()) >= NPATCH:
        raise ValueError("Out-of-range target index")
    return context.numpy(), targets.numpy()


def batch_metrics(context, targets, tissue):
    """Per-image delivered statistics, vectorised (same definitions as score_image)."""
    B = context.shape[0]
    rows = np.arange(B)[:, None]
    ctx_hot = np.zeros((B, NPATCH), bool)
    ctx_hot[rows, context] = True
    if not np.all(ctx_hot.sum(1) == context.shape[1]):
        raise ValueError("Repeated context index")
    flat = targets.transpose(1, 0, 2).reshape(B, -1)
    tgt_hot = np.zeros((B, NPATCH), bool)
    tgt_hot[rows, flat] = True
    if np.any(ctx_hot & tgt_hot):
        raise ValueError("Delivered context-target overlap")
    out = dict(n_ctx=np.full(B, context.shape[1]), n_slots=np.full(B, flat.shape[1]),
               n_hidden=tgt_hot.sum(1))
    out["slot_dupes"] = out["n_slots"] - out["n_hidden"]
    for proxy, t in tissue.items():
        out[f"ctx_on_{proxy}"] = (ctx_hot & t).sum(1)
        out[f"hid_on_{proxy}"] = (tgt_hot & t).sum(1)
        out[f"anat_{proxy}"] = t.sum(1)
    return out


SUM_KEYS = ("n_ctx", "n_slots", "n_hidden", "slot_dupes", "ctx_on_soft", "hid_on_soft",
            "anat_soft", "ctx_on_hard", "hid_on_hard", "anat_hard")


def summarize_arm(batches):
    """Seed-level summary from per-batch sums (images weighted equally)."""
    n = np.array([b["n"] for b in batches], float)
    total = {k: float(sum(b["sum"][k] for b in batches)) for k in SUM_KEYS}
    N = n.sum()
    mean = {k: total[k] / N for k in SUM_KEYS}
    ctx_b = np.array([b["C"] for b in batches], float)
    k_b = np.array([b["K"] for b in batches], float)
    nb = len(batches)
    out = dict(
        n_images=int(N), n_batches=nb,
        context_tokens=mean["n_ctx"], context_pct_grid=100 * mean["n_ctx"] / NPATCH,
        context_pct_grid_batch_se=(100 * ctx_b.std(ddof=1) / math.sqrt(nb) / NPATCH)
        if nb > 1 else None,
        unique_targets_U=mean["n_hidden"], loss_slots_L=mean["n_slots"],
        duplicate_slots_D=mean["slot_dupes"],
        mask_ratio_pct=100 * mean["n_hidden"] / NPATCH,
        purity_soft_pct=100 * mean["hid_on_soft"] / mean["n_hidden"],
        purity_hard_pct=100 * mean["hid_on_hard"] / mean["n_hidden"],
        anatomy_hidden_soft_pct=100 * mean["hid_on_soft"] / mean["anat_soft"],
        context_tissue_share_soft_pct=100 * mean["ctx_on_soft"] / mean["anat_soft"],
        context_on_tissue_soft_pct=100 * mean["ctx_on_soft"] / mean["n_ctx"],
        tissue_cells_soft=mean["anat_soft"], tissue_cells_hard=mean["anat_hard"],
        delivered_context_C_quantiles=np.quantile(ctx_b, [0, .05, .5, .95, 1]).tolist(),
        delivered_target_K_quantiles=np.quantile(k_b, [0, .05, .5, .95, 1]).tolist(),
    )
    return out


def paired_gap(batches, ref_batches):
    """Per-batch paired context difference (arm - reference), % of grid."""
    d = np.array([(b["C"] - r["C"]) * 100 / NPATCH for b, r in zip(batches, ref_batches)])
    w = np.array([b["n"] for b in batches], float)
    return dict(gap_pct_grid=float((d * w).sum() / w.sum()),
                batch_se=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
                batches_arm_greater=int((d > 0).sum()), batches_equal=int((d == 0).sum()),
                batches_arm_smaller=int((d < 0).sum()), n_batches=len(d),
                per_batch=d.tolist())


def run_seed(seed, args, arms, records, soft_ds, hard_ds, views, save_dir):
    order = list(views)
    if args.batch_order == "shuffled":
        random.Random(seed * 7919 + 17).shuffle(order)
    bs = args.batch_size
    n_batches = args.budget_draws or math.ceil(len(order) / bs)
    if n_batches * bs > len(order) and args.budget_draws:
        raise ValueError(f"{n_batches} draws x {bs} views exceed the {len(order)}-view scope")
    size_rng = torch.Generator().manual_seed(seed)
    sizer = arms["random"] if "random" in arms else _classes()["MaskCollator"](**BASE_KW)
    per_arm = {name: [] for name in arms}
    saved = {name: dict(enc=[], pred=[], C=[], K=[]) for name in arms}
    meta = dict(view_index=[], batch_id=[], tissue_soft=[], tissue_hard=[],
                crop_sha16=[], valid_soft=[], valid_hard=[], sizes=[])
    started = time.perf_counter()
    crop_checks = 0
    for b in range(n_batches):
        start = b * bs
        chunk = order[start:start + bs]
        soft_items, hard_items = [], []
        for index in chunk:
            seed_all(seed + index)
            soft = soft_ds[index]
            seed_all(seed + index)
            hard = hard_ds[index]
            if not torch.equal(soft[0], hard[0]):
                raise AssertionError("Soft/hard loaders produced different crops")
            crop_checks += 1
            soft_items.append(soft)
            hard_items.append(hard)
        images = torch.stack([x[0] for x in soft_items])
        guides = dict(soft=torch.stack([x[1] for x in soft_items]),
                      hard=torch.stack([x[1] for x in hard_items]))
        valid = dict(soft=torch.stack([x[2] for x in soft_items]),
                     hard=torch.stack([x[2] for x in hard_items]))
        tissue = {p: (guides[p][:, 0].reshape(len(chunk), -1).numpy() >= OCC_THRESHOLD)
                  for p in ("soft", "hard")}
        sizes = dict(
            pred=[sizer._sample_block_size(BASE_KW["pred_mask_scale"], size_rng)
                  for _ in range(BASE_KW["npred"])],
            enc=[sizer._sample_block_size(BASE_KW["enc_mask_scale"], size_rng)
                 for _ in range(BASE_KW["nenc"])])
        for name, arm in arms.items():
            seed_all(seed + start)
            context, targets = deliver(records[name], arm, images, guides, valid, sizes)
            m = batch_metrics(context, targets, tissue)
            per_arm[name].append(dict(n=len(chunk), C=int(context.shape[1]),
                                      K=int(targets.shape[2]),
                                      sum={k: int(m[k].sum()) for k in SUM_KEYS}))
            if save_dir is not None:
                saved[name]["enc"].append(context.astype(np.uint8).ravel())
                saved[name]["pred"].append(targets.astype(np.uint8).ravel())
                saved[name]["C"].append(context.shape[1])
                saved[name]["K"].append(targets.shape[2])
        meta["view_index"].extend(chunk)
        meta["batch_id"].extend([b] * len(chunk))
        meta["tissue_soft"].append(tissue["soft"])
        meta["tissue_hard"].append(tissue["hard"])
        meta["valid_soft"].append(valid["soft"].numpy())
        meta["valid_hard"].append(valid["hard"].numpy())
        meta["sizes"].append([list(s) for s in sizes["pred"] + sizes["enc"]])
        meta["crop_sha16"].extend(
            hashlib.sha256(images[i].numpy().tobytes()).hexdigest()[:16]
            for i in range(len(chunk)))
        if (b + 1) % 10 == 0 or b + 1 == n_batches:
            print(f"  seed {seed}: batch {b + 1}/{n_batches} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    result = dict(
        seed=seed, n_batches=n_batches, crop_identity_assertions=crop_checks,
        hard_guide_valid=int(np.concatenate(meta["valid_hard"]).sum()),
        soft_guide_valid=int(np.concatenate(meta["valid_soft"]).sum()),
        size_draws=meta["sizes"],
        metrics={name: summarize_arm(per_arm[name]) for name in arms},
        per_batch_context_C={name: [x["C"] for x in per_arm[name]] for name in arms},
        per_batch_target_K={name: [x["K"] for x in per_arm[name]] for name in arms},
        per_batch_n=[x["n"] for x in per_arm[next(iter(arms))]],
        elapsed_seconds=time.perf_counter() - started)
    if GAP_REFERENCE in arms:
        result["paired_context_gaps_vs_random"] = {
            name: paired_gap(per_arm[name], per_arm[GAP_REFERENCE])
            for name in arms if name != GAP_REFERENCE}
    if save_dir is not None:
        arrays = dict(
            view_index=np.asarray(meta["view_index"], np.int64),
            batch_id=np.asarray(meta["batch_id"], np.int32),
            tissue_soft=np.concatenate(meta["tissue_soft"]),
            tissue_hard=np.concatenate(meta["tissue_hard"]),
            valid_soft=np.concatenate(meta["valid_soft"]),
            valid_hard=np.concatenate(meta["valid_hard"]),
            crop_sha16=np.asarray(meta["crop_sha16"]),
            sizes=np.asarray(meta["sizes"], np.int16))
        for name in arms:
            arrays[f"{name}__enc"] = np.concatenate(saved[name]["enc"])
            arrays[f"{name}__pred"] = np.concatenate(saved[name]["pred"])
            arrays[f"{name}__C"] = np.asarray(saved[name]["C"], np.int16)
            arrays[f"{name}__K"] = np.asarray(saved[name]["K"], np.int16)
        path = pathlib.Path(save_dir) / f"delivered_masks_seed{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        result["saved_masks"] = dict(file=str(path), sha256=sha_file(path),
                                     layout=("per arm: <arm>__enc uint8 = concatenated "
                                             "batch contexts [n_b, C_b] (row-major); "
                                             "<arm>__pred uint8 = concatenated batch "
                                             "targets [npred, n_b, K_b]; <arm>__C/K per batch"))
    return result


def seed_summary(seed_results, arms):
    """Mean +- SD across audit seeds, and pooled paired gaps."""
    keys = ("context_pct_grid", "context_tokens", "unique_targets_U", "loss_slots_L",
            "duplicate_slots_D", "mask_ratio_pct", "purity_soft_pct", "purity_hard_pct",
            "anatomy_hidden_soft_pct", "context_tissue_share_soft_pct",
            "context_on_tissue_soft_pct")
    seeds = [r["seed"] for r in seed_results]
    summary = {}
    for name in arms:
        summary[name] = {}
        for key in keys:
            values = [r["metrics"][name][key] for r in seed_results]
            summary[name][key] = dict(
                mean=float(np.mean(values)),
                seed_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                per_seed=dict(zip(map(str, seeds), values)))
    gaps = {}
    if all("paired_context_gaps_vs_random" in r for r in seed_results):
        for name in seed_results[0]["paired_context_gaps_vs_random"]:
            per_seed = [r["paired_context_gaps_vs_random"][name]["gap_pct_grid"]
                        for r in seed_results]
            pooled = np.concatenate([r["paired_context_gaps_vs_random"][name]["per_batch"]
                                     for r in seed_results])
            weights = np.concatenate([r["per_batch_n"] for r in seed_results]).astype(float)
            se = float(pooled.std(ddof=1) / math.sqrt(len(pooled)))
            mean = float((pooled * weights).sum() / weights.sum())
            gaps[name] = dict(
                per_seed=dict(zip(map(str, seeds), per_seed)),
                mean_of_seeds=float(np.mean(per_seed)),
                seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                pooled_batches=len(pooled), pooled_mean=mean, pooled_batch_se=se,
                pooled_normal_95ci=[mean - 1.96 * se, mean + 1.96 * se],
                batches_arm_greater=int((pooled > 0).sum()),
                batches_arm_smaller=int((pooled < 0).sum()),
                batches_equal=int((pooled == 0).sum()))
    return summary, gaps


def environment_record(extra_files=()):
    files = [pathlib.Path(__file__), ROOT / "src" / "datasets" / "oct_slices_guided.py",
             ROOT / "src" / "datasets" / "oct_slices.py", ROOT / "src" / "transforms.py",
             ROOT / "src" / "guides" / "mirage_envelope.py", *extra_files]
    return dict(
        git_head=_git("rev-parse", "HEAD"),
        git_status_short=_git("status", "--short", "--untracked-files=no"),
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        torch=torch.__version__, torch_threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        worktree_file_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                              sha_file(p) for p in files})


def audit(args):
    samplers = install_sampler_sources(args.sampler_rev)
    torch.set_num_threads(args.threads)
    environment = environment_record([ROOT / p for p in TRAINED_CONFIGS.values()])
    arms_requested = args.arms
    guide_dirs = dict(soft=pathlib.Path(args.soft_guide_dir), hard=pathlib.Path(args.hard_guide_dir))
    arms, records = {}, {}
    for name in arms_requested:
        arms[name], records[name] = build_audit_arm(name, guide_dirs)
    soft_ds = make_dataset(guide_dirs["soft"])
    hard_ds = make_dataset(guide_dirs["hard"])
    if soft_ds.file_paths != hard_ds.file_paths:
        raise AssertionError("Soft/hard datasets enumerate different volumes")
    vols, views = select_views(len(soft_ds.file_paths), args.volumes, args.slices_per_volume)
    print(f"audit: {len(views)} views from {len(vols)} volumes; seeds {args.audit_seeds}; "
          f"{args.budget_draws or 'one pass of'} budget draws x {args.batch_size}; "
          f"order {args.batch_order}; samplers {samplers['origin']}", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in args.audit_seeds:
        seed_results.append(run_seed(seed, args, arms, records, soft_ds, hard_ds, views,
                                     args.save_masks))
        brief = {n: round(float(seed_results[-1]["metrics"][n]["context_pct_grid"]), 3)
                 for n in arms}
        print(f"seed {seed} context % of grid: {brief}", flush=True)
    summary, gaps = seed_summary(seed_results, arms)
    soft_meta = pathlib.Path(args.soft_guide_dir) / "cache_meta.json"
    result = dict(
        protocol=dict(
            name="camera_ready_delivered_mask_audit_v1",
            split="Training", volumes=len(vols), slices_per_volume=args.slices_per_volume,
            views=len(views), volume_selection="sorted(random.Random(42).sample(range(6000), volumes))",
            view_rule="slices 0,4,...,96 of each selected volume (num_slices=100 grid)",
            batch_size=args.batch_size, budget_draws_per_seed=args.budget_draws or None,
            batch_order=args.batch_order,
            audit_seeds=list(args.audit_seeds),
            pairing=("Within each audit seed, every arm sees the identical crop of every "
                     "view (seed_all(seed + view) before each loader call; soft and hard "
                     "loaders asserted equal), the identical batch composition, the "
                     "identical 4 target + 1 context rectangle sizes per batch "
                     "(torch.Generator(seed)), and the identical placement seed "
                     "(seed_all(seed + batch_start)) before each arm."),
            guidance="set_epoch(50, 100): r_t = 1 for every guided arm",
            tissue_proxies=dict(
                soft=("PRIMARY common scoring proxy: patch occupancy >= 0.25 of the "
                      "schema-2 MIRAGE soft-guide envelope (summed P_inner+P_choroid >= "
                      "0.5), same proxy as the published tables: " + str(guide_dirs["soft"])),
                hard=("secondary: patch occupancy >= 0.25 of the repaired hard MIRAGE "
                      "envelope (ENVELOPE's placement product): " + str(guide_dirs["hard"]))),
            definitions=dict(
                context_pct_grid="delivered encoder context tokens C_b / 256 (after batch equalisation), image mean",
                unique_targets_U="distinct delivered target cells per image",
                loss_slots_L="delivered target index slots per image (npred x K_b), duplicates included",
                duplicate_slots_D="L - U",
                mask_ratio_pct="U / 256",
                purity_pct="tissue-proxy cells among the U unique delivered targets (ratio of means)",
                anatomy_hidden_pct="tissue-proxy cells hidden as targets / all tissue-proxy cells",
                context_tissue_share_pct="tissue-proxy cells in delivered context / all tissue-proxy cells"),
            warning=("Mask-audit seeds are sampler replays, not pretraining seeds; the "
                     "historical training masks were not recorded.")),
        samplers=sampler_record(), arms=records,
        soft_guide_cache_meta={k: v for k, v in json.loads(soft_meta.read_text()).items()
                               if k != "slice_indices"} if soft_meta.is_file() else None,
        environment=environment,
        seeds=seed_results, summary=summary, paired_context_gaps_vs_random=gaps)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", flush=True)
    print_summary(summary, gaps)
    return result


def print_summary(summary, gaps):
    print(f"{'arm':38s} {'ctx%':>14s} {'U':>7s} {'L':>7s} {'D':>6s} {'mask%':>6s} "
          f"{'pur_soft':>8s} {'pur_hard':>8s}")
    for name, s in summary.items():
        sd = s["context_pct_grid"]["seed_sd"]
        print(f"{name:38s} {s['context_pct_grid']['mean']:7.3f}+-{(sd or 0):5.3f} "
              f"{s['unique_targets_U']['mean']:7.2f} {s['loss_slots_L']['mean']:7.2f} "
              f"{s['duplicate_slots_D']['mean']:6.2f} {s['mask_ratio_pct']['mean']:6.2f} "
              f"{s['purity_soft_pct']['mean']:8.2f} {s['purity_hard_pct']['mean']:8.2f}")
    for name, g in gaps.items():
        print(f"gap {name:34s} {g['mean_of_seeds']:+7.3f} (seed sd {(g['seed_sd'] or 0):.3f}; "
              f"pooled {g['pooled_mean']:+.3f} +- {g['pooled_batch_se']:.3f} SE) "
              f"per seed {[round(v, 3) for v in g['per_seed'].values()]}")


def combine(args):
    parts = [json.loads(pathlib.Path(p).read_text()) for p in args.inputs]
    first = parts[0]
    for part in parts[1:]:
        for key in ("samplers", "arms"):
            if part[key] != first[key]:
                raise ValueError(f"Inputs differ in {key}")
        a = {k: v for k, v in part["protocol"].items() if k != "audit_seeds"}
        b = {k: v for k, v in first["protocol"].items() if k != "audit_seeds"}
        if a != b:
            raise ValueError("Inputs differ in protocol")
    seed_results = sorted((s for p in parts for s in p["seeds"]), key=lambda s: s["seed"])
    if len({s["seed"] for s in seed_results}) != len(seed_results):
        raise ValueError("Duplicate audit seed")
    arms = list(first["arms"])
    summary, gaps = seed_summary(seed_results, arms)
    result = dict(first)
    result["protocol"] = dict(first["protocol"], audit_seeds=[s["seed"] for s in seed_results])
    result.update(seeds=seed_results, summary=summary, paired_context_gaps_vs_random=gaps,
                  combined_from=[dict(file=str(p), sha256=sha_file(p)) for p in args.inputs],
                  environments=[p["environment"] for p in parts])
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    print_summary(summary, gaps)


# --------------------------------------------------------------------------
# Reproduce the published (v9) table
# --------------------------------------------------------------------------
REPRO_ARMS = ("random", "oracle", "envelope", "anatomy", "cover")


def reproduce_published(args):
    samplers = install_sampler_sources(args.sampler_rev)
    torch.set_num_threads(args.threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(samplers=sampler_record(), environment=environment_record(), runs={})
    for label, rev in (("published_v9_probe_548f4d6", HISTORICAL_PROBE_REV),
                       ("pre_camera_ready_probe_HEAD", samplers["commit"])):
        code = subprocess.check_output(
            ["git", "show", f"{rev}:scripts/mask_composition_probe.py"], cwd=ROOT)
        module = types.ModuleType("probe_" + label)
        module.__file__ = str(HERE / "mask_composition_probe.py")
        exec(compile(code.decode("utf-8"), f"{rev}:scripts/mask_composition_probe.py", "exec"),
             module.__dict__)
        target = out_dir / f"{label}_bs{args.batch_size}.json"
        argv = sys.argv
        sys.argv = [module.__file__, "--volumes", "24", "--slices_per_volume", "25",
                    "--batch_size", str(args.batch_size), "--seed", "42",
                    "--cover_floor", "0.21", "--guide_dir", str(SOFT_GUIDE_DIR),
                    "--data_dir", str(DATA_DIR), "--slice_cache", str(SLICE_CACHE),
                    "--out", str(target)]
        started = time.perf_counter()
        try:
            module.main()
        finally:
            sys.argv = argv
        result = json.loads(target.read_text())
        entry = dict(script_rev=rev, script_sha256_lf=_sha(_lf(code)), output=str(target),
                     output_sha256=sha_file(target), elapsed_seconds=time.perf_counter() - started,
                     settings=("CENTROID code-default lateral 0.8; ENVELOPE/ANATOMY/COVER on "
                               "the common soft guide; ENVELOPE least-overlap HEAD fallback; "
                               "600 views from 24 volumes, seed 42"),
                     protocol=("unpaired: sequential shared RNG, crops depend on mask draws"
                               if rev == HISTORICAL_PROBE_REV else
                               "fixed crops + shared sizes; torch/random reseeded per arm "
                               "(numpy not reseeded)"),
                     metrics={a: {k: result[a][k] for k in (
                         "n_ctx_mean", "ctx_frac_of_grid", "n_hidden_mean", "n_slots_mean",
                         "slot_dupes_mean", "hidden_frac_of_grid", "hidden_pct_on_anat",
                         "hidden_share_of_all_anat", "ctx_share_of_all_anat")}
                         for a in REPRO_ARMS})
        archived_path = pathlib.Path(str(ARCHIVED_TABLE).format(bs=args.batch_size))
        if rev == HISTORICAL_PROBE_REV and archived_path.is_file():
            archived = json.loads(archived_path.read_text())
            diffs, compared = {}, 0
            for a in REPRO_ARMS:
                for key, value in archived[a].items():
                    if isinstance(value, (int, float)):
                        compared += 1
                        diffs[f"{a}.{key}"] = abs(result[a][key] - value)
            worst = max(diffs.values())
            entry["archived_table"] = dict(file=str(archived_path.relative_to(ROOT)),
                                           sha256=sha_file(archived_path),
                                           numeric_fields_compared=compared,
                                           max_abs_difference=worst,
                                           exact=worst == 0.0)
            if worst != 0.0 and not args.allow_mismatch:
                raise AssertionError(f"Archived table not reproduced: max diff {worst}")
        report["runs"][label] = entry
        print(label, {a: round(100 * entry["metrics"][a]["ctx_frac_of_grid"], 4)
                      for a in REPRO_ARMS}, flush=True)
    path = out_dir / f"reproduce_published_bs{args.batch_size}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")


def _fmt(summary, key, digits=2):
    s = summary[key]
    sd = s.get("seed_sd")
    return f"{s['mean']:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{s['mean']:.{digits}f}"


def report(args):
    """Old-vs-new delivered-context table from the saved artifacts."""
    primary = json.loads(pathlib.Path(args.primary).read_text())
    repro = json.loads(pathlib.Path(args.reproduction).read_text())
    a3 = json.loads(pathlib.Path(args.a3_scope).read_text()) if args.a3_scope else None
    S = primary["summary"]
    G = primary["paired_context_gaps_vs_random"]
    pub = repro["runs"]["published_v9_probe_548f4d6"]["metrics"]
    pre = repro["runs"]["pre_camera_ready_probe_HEAD"]["metrics"]
    pct = lambda m: 100 * m["ctx_frac_of_grid"]  # noqa: E731
    lines = ["# Delivered-mask audit (E12): old vs new", "",
             f"Primary: `{args.primary}` (sha256 {sha_file(args.primary)[:16]}); samplers "
             f"{primary['samplers']['origin']}; seeds {primary['protocol']['audit_seeds']}; "
             f"{primary['protocol']['budget_draws_per_seed']} budget draws x "
             f"{primary['protocol']['batch_size']} views per seed; {primary['protocol']['views']} "
             f"views from {primary['protocol']['volumes']} volumes; batch order "
             f"{primary['protocol']['batch_order']}.", "",
             "## Trained configuration (mean ± SD across audit seeds)", "",
             "| Arm | Context % grid | per seed | U | L | D | Mask ratio % | Purity % (soft proxy) | Purity % (hard proxy) | Tissue hidden % | Context share of tissue % |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in [a for a in primary["arms"] if ARM_SPECS[a]["role"] == "trained"]:
        s = S[name]
        per = " / ".join(f"{v:.2f}" for v in s["context_pct_grid"]["per_seed"].values())
        lines.append(f"| {ARM_SPECS[name]['label']} | {_fmt(s, 'context_pct_grid')} | {per} | "
                     f"{_fmt(s, 'unique_targets_U', 1)} | {_fmt(s, 'loss_slots_L', 1)} | "
                     f"{_fmt(s, 'duplicate_slots_D', 1)} | {_fmt(s, 'mask_ratio_pct', 1)} | "
                     f"{_fmt(s, 'purity_soft_pct', 1)} | {_fmt(s, 'purity_hard_pct', 1)} | "
                     f"{_fmt(s, 'anatomy_hidden_soft_pct', 1)} | {_fmt(s, 'context_tissue_share_soft_pct', 1)} |")
    lines += ["", "## Paired context gaps vs RANDOM (percentage points of the 256-cell grid)", "",
              "| Arm − RANDOM | per seed | mean ± seed SD | pooled ± batch SE (95% CI) | batches arm>R / = / < |",
              "|---|---|---:|---:|---:|"]
    for name, g in G.items():
        per = " / ".join(f"{v:+.2f}" for v in g["per_seed"].values())
        lo, hi = g["pooled_normal_95ci"]
        lines.append(f"| {ARM_SPECS[name]['label']} | {per} | {g['mean_of_seeds']:+.2f} ± "
                     f"{g['seed_sd']:.2f} | {g['pooled_mean']:+.2f} ± {g['pooled_batch_se']:.2f} "
                     f"({lo:+.2f}, {hi:+.2f}) | {g['batches_arm_greater']} / {g['batches_equal']} / "
                     f"{g['batches_arm_smaller']} |")
    rows = [("Published v9 (548f4d6; unpaired; 600 views, seed 42, 10 draws; C lateral 0.8; E soft guide + HEAD fallback)",
             pct(pub["random"]), pct(pub["oracle"]), pct(pub["envelope"]), ""),
            ("Pre-camera-ready probe (HEAD; paired crops+sizes; same 600 views/seed/settings)",
             pct(pre["random"]), pct(pre["oracle"]), pct(pre["envelope"]), "")]
    if a3:
        s = a3["summary"]
        rows.append(("A3-scope replay, trained settings (600 views x 3 seeds, 10 draws/seed, sequential)",
                     s["random"]["context_pct_grid"]["mean"], s["centroid"]["context_pct_grid"]["mean"],
                     s["envelope"]["context_pct_grid"]["mean"],
                     " / ".join(f"SD {s[a]['context_pct_grid']['seed_sd']:.2f}"
                                for a in ("random", "centroid", "envelope"))))
    if "centroid_lat08_published" in S and "envelope_soft_leastoverlap_published" in S:
        rows.append(("New protocol, OLD settings (C lateral 0.8; E soft guide + least-overlap)",
                     S["random"]["context_pct_grid"]["mean"],
                     S["centroid_lat08_published"]["context_pct_grid"]["mean"],
                     S["envelope_soft_leastoverlap_published"]["context_pct_grid"]["mean"], ""))
    if "envelope_hard_leastoverlap" in S:
        rows.append(("New protocol, C lateral 0.6; E hard guide + least-overlap (HEAD)",
                     S["random"]["context_pct_grid"]["mean"], S["centroid"]["context_pct_grid"]["mean"],
                     S["envelope_hard_leastoverlap"]["context_pct_grid"]["mean"], ""))
    rows.append(("**New protocol, TRAINED settings (C 0.6; E hard guide + legacy fallback)**",
                 S["random"]["context_pct_grid"]["mean"], S["centroid"]["context_pct_grid"]["mean"],
                 S["envelope"]["context_pct_grid"]["mean"],
                 " / ".join(f"SD {S[a]['context_pct_grid']['seed_sd']:.2f}"
                            for a in ("random", "centroid", "envelope"))))
    lines += ["", "## Old vs new: delivered context, % of grid (decomposition ladder)", "",
              "| Step | RANDOM | CENTROID | ENVELOPE | C − R | E − R | note |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for label, r, c, e, note in rows:
        lines.append(f"| {label} | {r:.2f} | {c:.2f} | {e:.2f} | {c - r:+.2f} | {e - r:+.2f} | {note} |")
    if "envelope_soft_legacy" in S:
        lines += ["", "ENVELOPE 2x2 (guide product x fallback), context % grid: "
                  + ", ".join(f"{ARM_SPECS[a]['label']}: {S[a]['context_pct_grid']['mean']:.2f}"
                              for a in ("envelope_soft_leastoverlap_published",
                                        "envelope_soft_legacy", "envelope_hard_leastoverlap",
                                        "envelope") if a in S)]
    text = "\n".join(lines) + "\n"
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command")
    a = sub.add_parser("audit", help="camera-ready multi-seed delivered-mask audit (default)")
    a.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARM_SPECS))
    a.add_argument("--audit-seeds", type=int, nargs="+", default=[42, 1234, 5678])
    a.add_argument("--budget-draws", type=int, default=100,
                   help="batches (independent size/budget draws) per audit seed; "
                        "0 = one pass over the views with a partial last batch")
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--volumes", type=int, default=256)
    a.add_argument("--slices-per-volume", type=int, default=25)
    a.add_argument("--batch-order", choices=("shuffled", "sequential"), default="shuffled")
    a.add_argument("--soft-guide-dir", default=str(SOFT_GUIDE_DIR))
    a.add_argument("--hard-guide-dir", default=str(HARD_GUIDE_DIR))
    a.add_argument("--sampler-rev", default="HEAD",
                   help="git revision whose src/masks is executed, or 'worktree'")
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--save-masks", default=None, help="directory for delivered-mask .npz files")
    a.add_argument("--out", default=str(DEFAULT_OUT_DIR / "delivered_mask_audit.json"))
    c = sub.add_parser("combine", help="merge per-seed audit outputs")
    c.add_argument("inputs", nargs="+")
    c.add_argument("--out", required=True)
    r = sub.add_parser("reproduce-published", help="re-execute the v9 probe and check the archive")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--sampler-rev", default="HEAD")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--allow-mismatch", action="store_true")
    r.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "published_reproduction"))
    t = sub.add_parser("report", help="old-vs-new markdown table from audit artifacts")
    t.add_argument("--primary", required=True)
    t.add_argument("--reproduction", required=True)
    t.add_argument("--a3-scope", default=None)
    t.add_argument("--out", required=True)
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("audit", "combine", "reproduce-published", "report",
                                   "-h", "--help"):
        argv = ["audit", *argv]
    args = parser.parse_args(argv)
    {"audit": audit, "combine": combine, "reproduce-published": reproduce_published,
     "report": report}[args.command](args)


if __name__ == "__main__":
    main()
