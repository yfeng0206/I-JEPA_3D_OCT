"""Bounded CPU replay of dataset -> paired crop -> worker -> delivered masks.

Only the existing table2_geometry Training scope is eligible. Real pixels and
case names are never written; row IDs are ordinal, deidentified observations.
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

BASELINE = ROOT / "results" / "masking" / "table2_geometry" / "mask_geometry_600slices_bs64_coverf021_seed42.json"
DEFAULT_OUT = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence" / "mask_replay_v2"
BASE = dict(input_size=(256, 256), patch_size=16, enc_mask_scale=(.85, 1.),
            pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5),
            nenc=1, npred=4, min_keep=10, allow_overlap=False)
RANDOM_SOURCES = {"uniform", "unguided", "unbiased_by_ramp", "random",
                  "random_legal", "fallback_invalid", "infeasible",
                  "infeasible_uniform"}


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
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ordinal):
        # One fixed crop per scoped image, independent of policy/batch/workers.
        seed_all(91009 + ordinal * 997)
        return (*self.dataset[self.indices[ordinal]], ordinal)


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
    def __init__(self, name):
        self.name = name
        self.collator = None

    def __call__(self, batch):
        torch.set_num_threads(1)
        ordinal = int(batch[0][3])
        sizes = drawn_sizes(ordinal // 64)
        seed_all(7103 + ordinal)
        images = torch.stack([it[0] for it in batch])
        guides = torch.stack([it[1] for it in batch])
        valid = torch.stack([it[2] for it in batch])
        if self.collator is None:
            if self.name == "random":
                self.collator = MaskCollator(**BASE, audit_masks=True)
            elif self.name == "oracle":
                self.collator = CurriculumMaskGenerator(**arm_kwargs(self.name))
                self.collator.set_epoch(50, 100)
            else:
                self.collator = MirageMaskCollator(**arm_kwargs(self.name))
                self.collator.set_epoch(50, 100)
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
                       guide_sha256=hashlib.sha256(guides[b].numpy().tobytes()).hexdigest(),
                       drawn_sizes=sizes)
            measure(row, guides[b])
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


def _load_historical_mask_modules():
    """Load baseline samplers with baseline mask dependencies, then restore imports."""
    names = ("utils", "anatomy", "cover", "multiblock", "curriculum")
    sources = {
        name: subprocess.check_output(
            ["git", "show", f"de145d7:src/masks/{name}.py"], cwd=ROOT, text=True)
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
            exec(compile(sources[name], f"de145d7:{name}.py", "exec"), old[name].__dict__)
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
    assert old["curriculum"].cover_build_targets is old["cover"].build_targets
    assert old["curriculum"].cover_is_viable is old["cover"].is_viable
    assert old["curriculum"].cover_build_targets is not previous["cover"].build_targets
    assert old["curriculum"].cover_build_targets is not previous["curriculum"].cover_build_targets
    assert old["curriculum"].anatomy_build_targets is old["anatomy"].build_targets
    assert old["curriculum"].resample_to_k is old["utils"].resample_to_k
    return old


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 64])
    parser.add_argument("--workers", type=int, choices=[0, 1], default=1)
    parser.add_argument("--arms", nargs="+", default=[
        "random", "oracle", "envelope", "anatomy", "cover_legacy", "cover_v2", "cover_v2_guard"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 600:
        parser.error("--count must stay inside the existing 600-slice scope")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
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
    dataset = GuidedOCTSliceDataset(
        data_dir=os.path.join(data["data_dir"], "Training"),
        guide_dir=os.path.join(metadata["guide_dir"], "Training"),
        num_slices=100, slice_size=256, patch_size=16, dilate_patches=0,
        occupancy_threshold=.25, transform=make_paired_transforms(),
        slice_cache=os.path.join(data["slice_cache_dir"], "Training"))
    vols = sorted(random.Random(metadata["seed"]).sample(
        range(len(dataset.file_paths)), metadata["volumes"]))
    indices = [v * 100 + s for v in vols for s in range(0, 100, 4)][:args.count]
    frozen = FixedCropDataset(dataset, indices)
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
        metrics={})
    hashes = {}
    guard_pairs = {}
    verified_guard_pairs = 0
    for bs in args.batch_sizes:
        for arm in args.arms:
            loader = DataLoader(frozen, batch_size=bs, shuffle=False,
                                num_workers=args.workers, collate_fn=AuditCollator(arm))
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
