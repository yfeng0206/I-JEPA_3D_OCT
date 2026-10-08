"""Bounded CPU replay of dataset -> paired crop -> worker -> delivered masks.

Only the existing table2_geometry Training scope is eligible. Real pixels and
case names are never written; row IDs are ordinal, deidentified observations.

Arms use the TRAINED settings explicitly: CENTROID (``oracle``) takes its band
parameters from configs/patch_oracle_anatomical.yaml (lateral 0.6, not the
code default 0.8); ENVELOPE is placed on the repaired HARD MIRAGE envelope
with the pre-fc49f61 ``legacy_uniform_v1`` overlap fallback that trained it,
while every arm is still scored on the common soft-guide proxy.  The replay
executes ``src/masks`` from a pinned git revision (``--sampler-rev``) and
records the commit and source SHA-256s.

``--verify-saved-masks`` recomputes the camera-ready audit statistics from the
index tensors saved by ``scripts/mask_composition_probe.py audit`` with
separate, per-image set arithmetic, as an independent check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
from src.masks.curriculum import CurriculumMaskGenerator, MirageMaskCollator
from src.masks.multiblock import MaskCollator
from src.transforms import make_paired_transforms
from scripts.mask_composition_probe import (
    A1_NOFIXA_SHA256_CRLF, FIXA_DISABLED, FIXA_LINE, HARD_GUIDE_DIR,
    load_trained_config, _crlf, _lf, _sha,
)

BASELINE = ROOT / "results" / "masking" / "table2_geometry" / "mask_geometry_600slices_bs64_coverf021_seed42.json"
DEFAULT_OUT = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence" / "mask_replay_v2"
BASE = dict(input_size=(256, 256), patch_size=16, enc_mask_scale=(.85, 1.),
            pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5),
            nenc=1, npred=4, min_keep=10, allow_overlap=False)
RANDOM_SOURCES = {"uniform", "unguided", "unbiased_by_ramp", "random",
                  "random_legal", "fallback_invalid", "infeasible",
                  "infeasible_uniform"}
# Trained placement that is not expressed by arm_kwargs alone: which guide
# product the sampler sees, and which ENVELOPE overlap fallback trained it.
TRAINED_PLACEMENT = {"envelope": dict(guide="hard", fallback="legacy_uniform_v1")}
MASK_MODULE_NAMES = ("utils", "anatomy", "cover", "multiblock", "curriculum")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def drawn_sizes(group):
    """Size draws do not consume any policy's placement RNG."""
    sizer = MaskCollator(**BASE)
    rng = torch.Generator().manual_seed(3107 + group)
    return {
        "pred": [sizer._sample_block_size(BASE["pred_mask_scale"], rng) for _ in range(4)],
        "enc": [sizer._sample_block_size(BASE["enc_mask_scale"], rng)],
    }


def arm_kwargs(name):
    cfg = dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.,
               mirage_occupancy_threshold=.25, mirage_min_block_fill=.4,
               mirage_min_retina_visible=.25, mirage_max_attempts=30,
               mirage_spread=True, mirage_overlap_tolerance=.25,
               anatomy_tau=.1, cover_leave_frac=.21, cover_min_visible_frac=.21,
               cover_min_visible_cells=4, cover_fill="random_legal",
               enc_truncate="prefix", audit_masks=True)
    extra = {}
    if name == "oracle":
        cfg["mode"] = "anatomical_prior"
        # Trained band (region .28, lateral .6, offset 0, min rows 3), never
        # the code default lateral .8 the published audits silently used.
        cfg.update({key: value for key, value in
                    load_trained_config("centroid")["curriculum"].items()
                    if key.startswith("oracle_")})
    elif name == "envelope":
        cfg["mode"] = "mirage_envelope"
    elif name == "anatomy":
        cfg.update(mode="mirage_anatomy", anatomy_mass_cap=.9,
                   anatomy_bridge_diagonals=True)
        extra["pred_target_k"] = 16
    elif name.startswith("cover_v2"):
        cfg["cover_algorithm"] = "delivered_v2"
        cfg["cover_context_guard"] = name == "cover_v2_guard"
    return dict(BASE, curriculum_cfg=cfg, **extra)


class FixedCropDataset(Dataset):
    """One fixed crop per scoped image.

    With ``placement_dataset`` (e.g. the hard ENVELOPE guide product), items are
    ``(image, placement_guide, placement_valid, ordinal, scoring_guide)``: the
    sampler sees the placement product while every arm is scored on the same
    proxy.  Both loaders draw the same crop, which is asserted.
    """

    def __init__(self, dataset, indices, placement_dataset=None):
        self.dataset, self.indices = dataset, indices
        self.placement_dataset = placement_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ordinal):
        # One fixed crop per scoped image, independent of policy/batch/workers.
        seed_all(91009 + ordinal * 997)
        item = self.dataset[self.indices[ordinal]]
        if self.placement_dataset is None:
            return (*item, ordinal)
        seed_all(91009 + ordinal * 997)
        placed = self.placement_dataset[self.indices[ordinal]]
        if not torch.equal(item[0], placed[0]):
            raise AssertionError("Placement and scoring loaders drew different crops")
        return (placed[0], placed[1], placed[2], ordinal, item[1])


def validate_delivered(enc, pred, *, batch_size, npred=4, nenc=1, patches=256):
    if len(pred) != npred or len(enc) != nenc:
        raise ValueError("Wrong target/context group count")
    target_k = pred[0].shape[1] if pred else 0
    for group in enc + pred:
        if (group.dtype != torch.long or group.ndim != 2
                or group.shape[0] != batch_size or group.shape[1] < 1):
            raise ValueError("Invalid delivered tensor contract")
        if int(group.min()) < 0 or int(group.max()) >= patches:
            raise ValueError("Out-of-bounds delivered index")
    if any(g.shape[1] != target_k for g in pred):
        raise ValueError("Unequal target group budgets")
    if any(g.shape[1] != enc[0].shape[1] for g in enc):
        raise ValueError("Unequal context group budgets")
    for b in range(batch_size):
        union = set(torch.cat([g[b] for g in pred]).tolist())
        if any(union.intersection(g[b].tolist()) for g in enc):
            raise ValueError("Delivered context-target overlap")


def measure(row, guide):
    tissue = (guide[0].flatten().numpy() >= .25)
    before = row["intended_targets"]
    after = row["targets"]
    slots_before = [i for group in before for i in group]
    slots = [i for group in after for i in group]
    union, old_union = set(slots), set(slots_before)
    context, old_context = row["context"][0], row["context_before_collation"][0]
    sources = row["target_sources"]
    if len(sources) != 4 or len(after) != 4:
        raise ValueError("Wrong target count/source bookkeeping")
    random_ids = [i for source, group in zip(sources, after)
                  if source in RANDOM_SOURCES for i in group]
    row.update(
        guide_channels=int(guide.shape[0]),
        guide_occupancy_mass=float(guide[0].sum()),
        tissue_cells=int(tissue.sum()), intended_loss_slots=len(slots_before),
        delivered_loss_slots=len(slots), unique_target_union=len(union),
        duplicate_loss_slots=len(slots) - len(union),
        intended_target_tissue_unique=int(tissue[list(old_union)].sum()),
        delivered_target_tissue_unique=int(tissue[list(union)].sum()),
        intended_target_tissue_slots=int(tissue[slots_before].sum()),
        target_tissue_slots=int(tissue[slots].sum()),
        target_background_slots=int((~tissue[slots]).sum()),
        context_tokens=len(context), context_tissue=int(tissue[context].sum()),
        context_tissue_before_collation=int(tissue[old_context].sum()),
        context_tokens_before_collation=len(old_context),
        complement_tissue=int(tissue.sum()) - int(tissue[list(union)].sum()),
        random_loss_slots=len(random_ids),
        random_tissue_slots=int(tissue[random_ids].sum()),
        random_background_slots=int((~tissue[random_ids]).sum()),
        per_target=[dict(source=source, intended_slots=len(old), delivered_slots=len(new),
                         unique_cells=len(set(new)), tissue_slots=int(tissue[new].sum()),
                         background_slots=int((~tissue[new]).sum()))
                    for source, old, new in zip(sources, before, after)],
    )
    if guide.shape[0] >= 4:
        scores = guide[2:4].sum(0).numpy()
        soft_support = (scores > .1).ravel()
        row["guide_score_channel_means"] = [float(x.mean()) for x in guide[2:4]]
        row["guide_soft_support_cells"] = int(soft_support.sum())
        row["guide_soft_vs_occupancy_disagreement"] = int((soft_support != tissue).sum())
    if row.get("policy_info"):
        score = guide[2:].sum(0).numpy() if guide.shape[0] >= 4 else guide[0].numpy()
        supported_mass = np.where(score > .1, score, 0).ravel()
        total = supported_mass.sum()
        actual = float(supported_mass[list(union)].sum() / total) if total else 0.
        row["scored_hidden_mass_fraction"] = row["policy_info"]["covered_frac"]
        row["delivered_hidden_mass_fraction"] = actual
    return row


class AuditCollator:
    def __init__(self, name, sampler_rev=None):
        self.name = name
        self.sampler_rev = sampler_rev
        self.collator = None

    def _build(self):
        if self.sampler_rev:
            pinned = pinned_policy_modules(self.sampler_rev)
            curriculum = pinned["modules"]["curriculum"]
            collator_cls = pinned["modules"]["multiblock"].MaskCollator
        else:
            pinned, curriculum, collator_cls = None, None, MaskCollator
        generator_cls = curriculum.CurriculumMaskGenerator if curriculum else CurriculumMaskGenerator
        mirage_cls = curriculum.MirageMaskCollator if curriculum else MirageMaskCollator
        kwargs = arm_kwargs(self.name) if self.name != "random" else None
        if TRAINED_PLACEMENT.get(self.name, {}).get("fallback") == "legacy_uniform_v1":
            if pinned is None:
                raise RuntimeError("The legacy ENVELOPE fallback needs --sampler-rev")
            mirage_cls = pinned["legacy_module"].MirageMaskCollator
            kwargs["curriculum_cfg"].update(pinned["legacy_cfg"])
        if self.name == "random":
            self.collator = collator_cls(**BASE, audit_masks=True)
        elif self.name == "oracle":
            self.collator = generator_cls(**kwargs)
            self.collator.set_epoch(50, 100)
        else:
            self.collator = mirage_cls(**kwargs)
            self.collator.set_epoch(50, 100)

    def __call__(self, batch):
        torch.set_num_threads(1)
        ordinal = int(batch[0][3])
        sizes = drawn_sizes(ordinal // 64)
        seed_all(7103 + ordinal)
        images = torch.stack([it[0] for it in batch])
        guides = torch.stack([it[1] for it in batch])
        valid = torch.stack([it[2] for it in batch])
        scoring = torch.stack([it[4] if len(it) > 4 else it[1] for it in batch])
        if self.collator is None:
            self._build()
        if self.name == "random":
            _, enc, pred = self.collator(list(images), block_sizes=sizes)
            rows = self.collator.last_mask_audit
        elif self.name == "oracle":
            enc, pred = self.collator.generate(
                len(batch), imgs_cpu=images, guide_grids=guides,
                guide_valid=valid, block_sizes=sizes)
            rows = self.collator.last_mask_audit
        else:
            _, enc, pred, stats = self.collator(batch, block_sizes=sizes)
            rows = stats["delivered_audit"]
        validate_delivered(enc, pred, batch_size=len(batch))
        worker = torch.utils.data.get_worker_info()
        for b, row in enumerate(rows):
            row.update(ordinal=int(batch[b][3]), arm=self.name, batch_size=len(batch),
                       guide_valid=bool(valid[b]), worker_id=worker.id if worker else -1,
                       crop_tensor_sha256=hashlib.sha256(images[b].numpy().tobytes()).hexdigest(),
                       guide_sha256=hashlib.sha256(scoring[b].numpy().tobytes()).hexdigest(),
                       placement_guide_sha256=hashlib.sha256(guides[b].numpy().tobytes()).hexdigest(),
                       drawn_sizes=sizes)
            measure(row, scoring[b])
        return rows


def summarize(rows):
    keys = ["intended_loss_slots", "delivered_loss_slots", "unique_target_union",
            "duplicate_loss_slots", "intended_target_tissue_unique",
            "delivered_target_tissue_unique", "target_tissue_slots",
            "target_background_slots", "context_tokens", "context_tissue",
            "context_tissue_before_collation", "complement_tissue",
            "random_loss_slots", "random_tissue_slots", "random_background_slots"]
    result = {"n": len(rows)}
    for key in keys:
        values = np.asarray([r[key] for r in rows], float)
        result[key] = dict(mean=float(values.mean()),
                           quantiles=dict(zip(["min", "p05", "p50", "p95", "max"],
                                              np.quantile(values, [0, .05, .5, .95, 1]).tolist())))
    result["zero_tissue_context"] = sum(r["context_tissue"] == 0 for r in rows)
    result["context_floor_deficit"] = sum(
        r["context_tissue"] < max(int(np.ceil(.21 * r["tissue_cells"])),
                                 min(4, r["tissue_cells"])) for r in rows)
    result["collation_removed_tissue"] = sum(
        r["context_tissue"] < r["context_tissue_before_collation"] for r in rows)
    result["target_truncation_removed_tissue"] = sum(
        r["delivered_target_tissue_unique"] < r["intended_target_tissue_unique"] for r in rows)
    result["no_random_slots"] = sum(r["random_loss_slots"] == 0 for r in rows)
    result["guide_invalid"] = sum(not r["guide_valid"] for r in rows)
    result["source_blocks"] = dict(Counter(s for r in rows for s in r["target_sources"]))
    result["floor_status"] = dict(Counter(
        r.get("context_floor", {}).get("status", "historical_no_final_guard") for r in rows))
    return result


def synthetic_fixture(out):
    guides = torch.zeros(2, 4, 16, 16)
    guides[:, :2, 8:11] = 1
    guides[:, 2, 8:10] = 1
    guides[:, 3, 10:11] = 1
    codes = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    image = codes.repeat_interleave(16, 0).repeat_interleave(16, 1) / 255
    images = torch.stack([image.repeat(3, 1, 1), image.flip(1).repeat(3, 1, 1)])
    valid = torch.ones(2, dtype=torch.bool)
    collator = MirageMaskCollator(**arm_kwargs("cover_v2_guard"))
    collator.set_epoch(50, 100)
    seed_all(120)
    _, enc, pred, _ = collator(
        list(zip(images, guides, valid)), block_sizes=drawn_sizes(0))
    validate_delivered(enc, pred, batch_size=2)
    fixture = dict(schema_version=1, images=images, masks_enc=enc, masks_pred=pred,
                   guides=guides, guide_valid=valid,
                   metadata=dict(source="synthetic_coordinate_codes",
                                 policy="cover_v2_guard", seed=120))
    torch.save(fixture, out / "synthetic_final_masks.pt")
    corruptions = {}
    for name, bad in [("wrong_target_count", pred[:-1]),
                      ("out_of_bounds", [torch.full_like(pred[0], 256)] + pred[1:]),
                      ("context_target_overlap", [enc[0][:, :1].repeat(1, pred[0].shape[1])] + pred[1:])]:
        try:
            validate_delivered(enc, bad, batch_size=2)
        except ValueError:
            corruptions[name] = "detected"
        else:
            raise AssertionError(name)
    return corruptions


def _load_mask_modules(rev, legacy_curriculum_source=None):
    """Bind ``src/masks`` at ``rev`` with its own dependencies, then restore imports.

    ``legacy_curriculum_source`` optionally adds a ``curriculum_legacy`` module
    executed from that text against the same pinned dependencies.
    """
    names = MASK_MODULE_NAMES
    sources = {
        name: subprocess.check_output(
            ["git", "show", f"{rev}:src/masks/{name}.py"], cwd=ROOT, text=True,
            encoding="utf-8")
        for name in names
    }
    old = {name: types.ModuleType(f"src.masks.{name}") for name in names}
    missing = object()
    package = sys.modules["src.masks"]
    previous = {name: sys.modules.get(f"src.masks.{name}", missing) for name in names}
    attributes = {name: getattr(package, name, missing) for name in names}
    try:
        for name in names:
            sys.modules[f"src.masks.{name}"] = old[name]
            setattr(package, name, old[name])
        for name in names:
            exec(compile(sources[name], f"{rev}:{name}.py", "exec"), old[name].__dict__)
        if legacy_curriculum_source is not None:
            old["curriculum_legacy"] = types.ModuleType("src.masks.curriculum_legacy_uniform_v1")
            exec(compile(legacy_curriculum_source, f"{rev}:curriculum.py[FIX-A disabled]", "exec"),
                 old["curriculum_legacy"].__dict__)
    finally:
        for name in names:
            qualified = f"src.masks.{name}"
            if previous[name] is missing:
                sys.modules.pop(qualified, None)
            else:
                sys.modules[qualified] = previous[name]
            if attributes[name] is missing:
                delattr(package, name)
            else:
                setattr(package, name, attributes[name])
    old["_sources"] = sources
    return old


def _load_historical_mask_modules():
    """Load baseline samplers with baseline mask dependencies, then restore imports."""
    old = _load_mask_modules("de145d7")
    previous = {name: sys.modules[f"src.masks.{name}"] for name in ("cover", "curriculum")}
    assert old["curriculum"].cover_build_targets is old["cover"].build_targets
    assert old["curriculum"].cover_is_viable is old["cover"].is_viable
    assert old["curriculum"].cover_build_targets is not previous["cover"].build_targets
    assert old["curriculum"].cover_build_targets is not previous["curriculum"].cover_build_targets
    assert old["curriculum"].anatomy_build_targets is old["anatomy"].build_targets
    assert old["curriculum"].resample_to_k is old["utils"].resample_to_k
    return old


_PINNED = {}


def pinned_policy_modules(rev="HEAD"):
    """Pinned sampler modules plus the trained (legacy) ENVELOPE fallback."""
    if rev in _PINNED:
        return _PINNED[rev]
    commit = subprocess.check_output(["git", "rev-parse", f"{rev}^{{commit}}"],
                                     cwd=ROOT, text=True).strip()
    current = subprocess.check_output(["git", "show", f"{commit}:src/masks/curriculum.py"],
                                      cwd=ROOT, text=True, encoding="utf-8")
    if "mirage_overlap_fallback" in current:
        modules = _load_mask_modules(commit)
        legacy_module = modules["curriculum"]
        legacy_cfg = {"mirage_overlap_fallback": "legacy_uniform_v1"}
        legacy = dict(method="native config key mirage_overlap_fallback=legacy_uniform_v1")
    else:
        line, disabled = FIXA_LINE.decode(), FIXA_DISABLED.decode()
        if current.count(line) != 1:
            raise RuntimeError("Cannot locate the single FIX-A fallback line")
        patched = current.replace(line, disabled)
        modules = _load_mask_modules(commit, legacy_curriculum_source=patched)
        legacy_module = modules["curriculum_legacy"]
        legacy_cfg = {}
        legacy = dict(method="in-memory copy with `if least.any():` disabled",
                      patched_sha256_lf=_sha(_lf(patched.encode())),
                      equals_a1_noFixA_reference=(
                          _sha(_crlf(patched.encode())) == A1_NOFIXA_SHA256_CRLF))
    record = dict(rev=rev, commit=commit, legacy_envelope=legacy, files={
        f"src/masks/{name}.py": dict(sha256_lf=_sha(_lf(text.encode())),
                                     sha256_crlf_checkout=_sha(_crlf(text.encode())))
        for name, text in modules["_sources"].items()})
    _PINNED[rev] = dict(modules=modules, legacy_module=legacy_module,
                        legacy_cfg=legacy_cfg, record=record)
    return _PINNED[rev]


def historical_replay_controls():
    """Compare default tensors with independently bound baseline mask modules."""
    old = _load_historical_mask_modules()
    guide = torch.zeros(2, 4, 16, 16)
    guide[:, :2, 8:11] = 1
    guide[:, 2, 8:10] = 1
    guide[:, 3, 10:11] = 1
    images = torch.zeros(2, 3, 256, 256)
    checks = []
    for name in ["random", "oracle", "envelope", "anatomy", "cover_legacy"]:
        for seed in [0, 7, 42]:
            outputs = []
            for historical in [True, False]:
                seed_all(seed)
                if name == "random":
                    cls = old["multiblock"].MaskCollator if historical else MaskCollator
                    _, enc, pred = cls(**BASE)(list(images))
                else:
                    cls = old["curriculum"].CurriculumMaskGenerator if historical else CurriculumMaskGenerator
                    kwargs = arm_kwargs(name)
                    kwargs["curriculum_cfg"].pop("audit_masks")
                    obj = cls(**kwargs)
                    obj.set_epoch(50, 100)
                    enc, pred = obj.generate(
                        2, imgs_cpu=images, guide_grids=guide,
                        guide_valid=torch.ones(2, dtype=torch.bool))
                outputs.append(enc + pred)
            match = all(torch.equal(a, b) for a, b in zip(*outputs))
            if not match:
                raise AssertionError(f"Historical default masks changed: {name}, seed {seed}")
            checks.append(dict(arm=name, seed=seed, tensors_bitwise_equal=match,
                               baseline_mask_dependencies_isolated=True))
    return checks


def verify_saved_masks(audit_path, out_dir):
    """Recompute an audit's delivered statistics from its saved index tensors.

    Independent of ``mask_composition_probe``'s vectorised scorer: per-image
    Python set arithmetic over the stored uint8 indices and tissue proxies.
    """
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    npatch = 256
    report = dict(audit=str(audit_path),
                  audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  method=("per image: context set and target slot list rebuilt from the "
                          "saved uint8 arrays; U=len(set(slots)), L=len(slots), "
                          "purity=sum tissue[U]/sum U; context %=mean len(context)/256"),
                  seeds={}, max_abs_difference={})
    worst = {}
    for seed_result in audit["seeds"]:
        saved = seed_result["saved_masks"]
        path = Path(saved["file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
            raise AssertionError(f"Saved masks changed since the audit: {path}")
        seed_report = {}
        with np.load(path, allow_pickle=False) as z:
            n_per_batch = np.bincount(z["batch_id"])
            tissue = dict(soft=z["tissue_soft"], hard=z["tissue_hard"])
            for arm in audit["arms"]:
                enc, pred = z[f"{arm}__enc"], z[f"{arm}__pred"]
                widths_c, widths_k = z[f"{arm}__C"], z[f"{arm}__K"]
                e = p = image = 0
                acc = dict(ctx=0, slots=0, unique=0, on_soft=0, on_hard=0, batches=[])
                for b, n in enumerate(n_per_batch):
                    c, k = int(widths_c[b]), int(widths_k[b])
                    contexts = enc[e:e + n * c].reshape(n, c)
                    targets = pred[p:p + 4 * n * k].reshape(4, n, k)
                    e, p = e + n * c, p + 4 * n * k
                    for i in range(n):
                        context = set(int(x) for x in contexts[i])
                        slots = [int(x) for g in range(4) for x in targets[g, i]]
                        union = set(slots)
                        if len(context) != c or context & union:
                            raise AssertionError(f"{arm} seed {seed_result['seed']}: invalid masks")
                        acc["ctx"] += len(context)
                        acc["slots"] += len(slots)
                        acc["unique"] += len(union)
                        acc["on_soft"] += sum(bool(tissue["soft"][image, j]) for j in union)
                        acc["on_hard"] += sum(bool(tissue["hard"][image, j]) for j in union)
                        image += 1
                    acc["batches"].append(c)
                if e != enc.size or p != pred.size or image != len(z["batch_id"]):
                    raise AssertionError(f"{arm}: saved array lengths do not tile the batches")
                recomputed = dict(
                    context_pct_grid=100 * acc["ctx"] / image / npatch,
                    unique_targets_U=acc["unique"] / image, loss_slots_L=acc["slots"] / image,
                    duplicate_slots_D=(acc["slots"] - acc["unique"]) / image,
                    mask_ratio_pct=100 * acc["unique"] / image / npatch,
                    purity_soft_pct=100 * acc["on_soft"] / acc["unique"],
                    purity_hard_pct=100 * acc["on_hard"] / acc["unique"])
                reported = seed_result["metrics"][arm]
                diffs = {key: abs(value - reported[key]) for key, value in recomputed.items()}
                if acc["batches"] != seed_result["per_batch_context_C"][arm]:
                    raise AssertionError(f"{arm}: per-batch context widths disagree")
                seed_report[arm] = dict(recomputed=recomputed, abs_difference=diffs,
                                        images=image)
                for key, value in diffs.items():
                    worst[key] = max(worst.get(key, 0.0), value)
            if "random" in audit["arms"]:
                ref = z["random__C"].astype(float)
                for arm in audit["arms"]:
                    if arm == "random":
                        continue
                    gap = float(np.average((z[f"{arm}__C"] - ref) * 100 / npatch,
                                           weights=n_per_batch))
                    reported = seed_result["paired_context_gaps_vs_random"][arm]["gap_pct_grid"]
                    seed_report[arm]["paired_gap_vs_random"] = dict(
                        recomputed=gap, abs_difference=abs(gap - reported))
                    worst["paired_gap"] = max(worst.get("paired_gap", 0.0), abs(gap - reported))
        report["seeds"][str(seed_result["seed"])] = seed_report
    report["max_abs_difference"] = worst
    report["agrees_within_1e-9"] = all(v <= 1e-9 for v in worst.values())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{audit_path.stem}__saved_mask_verification.json"
    target.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(max_abs_difference=worst, agrees=report["agrees_within_1e-9"],
                          out=str(target))))
    if not report["agrees_within_1e-9"]:
        raise AssertionError("Saved-mask recomputation disagrees with the audit")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 64])
    parser.add_argument("--workers", type=int, choices=[0, 1], default=1)
    parser.add_argument("--arms", nargs="+", default=[
        "random", "oracle", "envelope", "anatomy", "cover_legacy", "cover_v2", "cover_v2_guard"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--sampler-rev", default="HEAD",
                        help="git revision whose src/masks drives the replay")
    parser.add_argument("--soft-guide-dir", type=Path, default=None,
                        help="common scoring guide; default: archived path, else its relocation")
    parser.add_argument("--hard-guide-dir", type=Path, default=HARD_GUIDE_DIR,
                        help="ENVELOPE placement guide (trained product)")
    parser.add_argument("--slice-cache-dir", type=Path, default=None)
    parser.add_argument("--verify-saved-masks", type=Path, default=None,
                        help="recompute an audit JSON from its saved masks and exit")
    args = parser.parse_args()
    if args.verify_saved_masks:
        verify_saved_masks(args.verify_saved_masks, args.out)
        return
    if not 1 <= args.count <= 600:
        parser.error("--count must stay inside the existing 600-slice scope")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    pinned = pinned_policy_modules(args.sampler_rev)
    code_sha256 = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [r"src\masks\cover.py", r"src\masks\curriculum.py",
                     r"src\masks\multiblock.py", r"scripts\delivered_mask_audit.py",
                     r"scripts\mask_composition_probe.py"]}
    controls = synthetic_fixture(args.out)
    historical = historical_replay_controls()
    (args.out / "historical_replay_controls.json").write_text(json.dumps(historical, indent=2))
    if args.synthetic_only:
        print(json.dumps(controls))
        return
    baseline = json.loads(BASELINE.read_text())
    metadata = baseline["_meta"]
    config = yaml.safe_load((ROOT / "configs" / "patch_cover_f021_ep25.yaml").read_text())
    data = config["data"]
    soft_dir = args.soft_guide_dir or Path(metadata["guide_dir"])
    if not soft_dir.is_dir():
        from scripts.mask_composition_probe import SOFT_GUIDE_DIR
        if SOFT_GUIDE_DIR.name != soft_dir.name:
            raise FileNotFoundError(soft_dir)
        soft_dir = SOFT_GUIDE_DIR
    cache_dir = args.slice_cache_dir or Path(data["slice_cache_dir"])
    if not cache_dir.is_dir():
        cache_dir = Path(data["data_dir"]).parent / "slice_cache"

    def guided(guide_dir):
        return GuidedOCTSliceDataset(
            data_dir=os.path.join(data["data_dir"], "Training"),
            guide_dir=os.path.join(guide_dir, "Training"),
            num_slices=100, slice_size=256, patch_size=16, dilate_patches=0,
            occupancy_threshold=.25, transform=make_paired_transforms(),
            slice_cache=os.path.join(cache_dir, "Training"))
    dataset = guided(soft_dir)
    vols = sorted(random.Random(metadata["seed"]).sample(
        range(len(dataset.file_paths)), metadata["volumes"]))
    indices = [v * 100 + s for v in vols for s in range(0, 100, 4)][:args.count]
    frozen = FixedCropDataset(dataset, indices)
    placement = {}
    for arm in args.arms:
        product = TRAINED_PLACEMENT.get(arm, {}).get("guide")
        if product == "hard":
            if "hard" not in placement:
                hard = guided(args.hard_guide_dir)
                if hard.file_paths != dataset.file_paths:
                    raise AssertionError("Hard and soft guide datasets enumerate different volumes")
                placement["hard"] = FixedCropDataset(dataset, indices, placement_dataset=hard)
    cache_checks = []
    for ordinal in sorted({0, min(25, len(indices) - 1)}):
        vi, si = divmod(indices[ordinal], 100)
        with np.load(dataset.file_paths[vi], allow_pickle=False) as source:
            native = source["oct_bscans"][dataset.slice_indices[si]]
        cached = dataset.read_slice(vi, si)
        max_error = int(np.abs(native.astype(int) - cached.astype(int)).max())
        if max_error:
            raise AssertionError("Native slice cache does not match the source volume")
        cache_checks.append(dict(ordinal=ordinal, max_absolute_byte_error=max_error))
    summary = dict(
        baseline="de145d7", baseline_measurement_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        historical_scope=dict(volumes=24, slices=600, split="Training", selection_seed=42),
        replay_count=args.count, controls=controls, workers=args.workers,
        historical_default_controls=historical,
        native_slice_cache_checks=cache_checks,
        pairing="Exact per-image crop hashes and injected sizes; placement RNG draws are NOT paired across policies.",
        crop_seed_rule="91009 + ordinal * 997; new fixed crops, not reconstructed historical crops",
        samplers=pinned["record"],
        trained_settings=dict(
            oracle={k: v for k, v in arm_kwargs("oracle")["curriculum_cfg"].items()
                    if k.startswith("oracle_")},
            oracle_config=load_trained_config("centroid")["path"],
            envelope=dict(TRAINED_PLACEMENT["envelope"], placement_guide_dir=str(args.hard_guide_dir))),
        scoring_guide_dir=str(soft_dir),
        code_sha256_at_start=code_sha256,
        metrics={})
    hashes = {}
    guard_pairs = {}
    verified_guard_pairs = 0
    for bs in args.batch_sizes:
        for arm in args.arms:
            product = TRAINED_PLACEMENT.get(arm, {}).get("guide")
            loader = DataLoader(placement.get(product, frozen), batch_size=bs, shuffle=False,
                                num_workers=args.workers,
                                collate_fn=AuditCollator(arm, args.sampler_rev))
            rows = [row for batch in loader for row in batch]
            for row in rows:
                pair = (row["crop_tensor_sha256"], row["guide_sha256"])
                if row["ordinal"] in hashes:
                    assert hashes[row["ordinal"]] == pair, "Cross-policy crop mismatch"
                hashes[row["ordinal"]] = pair
                if arm in ("cover_v2", "cover_v2_guard"):
                    signature = (row["targets"], row["context_before_collation"],
                                 row["context_tokens"])
                    key = (bs, row["ordinal"])
                    if key in guard_pairs:
                        assert guard_pairs[key] == signature, "Guard ablation changed targets/budget"
                        verified_guard_pairs += 1
                    else:
                        guard_pairs[key] = signature
            key = f"{arm}_bs{bs}"
            with (args.out / f"{key}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            summary["metrics"][key] = summarize(rows)
            print(key, json.dumps({k: summary["metrics"][key][k] for k in
                  ["n", "zero_tissue_context", "context_floor_deficit",
                   "target_truncation_removed_tissue", "no_random_slots"]}), flush=True)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["code_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [r"src\masks\cover.py", r"src\masks\curriculum.py",
                     r"src\masks\multiblock.py", r"scripts\delivered_mask_audit.py"]}
    summary["python"] = sys.version
    summary["torch"] = torch.__version__
    summary["verified_exact_guard_pairs"] = verified_guard_pairs
    summary["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
