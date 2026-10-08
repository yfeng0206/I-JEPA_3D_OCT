"""Bounded historical-guide and checkpoint diagnostics; no optimizer updates."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

ROOT = Path(r"C:\Users\Gary\Desktop\jepa")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scripts.delivered_mask_audit import (
    BASELINE, FixedCropDataset, arm_kwargs, drawn_sizes, measure, seed_all,
    validate_delivered,
)
from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
from src.helper import init_patch_model
from src.masks.curriculum import MirageMaskCollator
from src.masks.utils import apply_masks
from src.train_patch import jepa_forward_loss
from src.transforms import make_paired_transforms

OUT = ROOT / r"autopilot\investigations\masking_family_20260910"
PRIVATE = ROOT / r".audit\masking_family_20260910"
REPLAY = ROOT / r"autopilot\investigations\delivered_task\evidence\mask_replay600_v2"
ARMS = ("random", "oracle", "envelope", "anatomy", "cover_legacy", "cover_v2_guard")
ORDINALS = list(range(0, 576, 25))
CHECKPOINTS = {
    "ancestor25": r"D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\jepa_patch-random_posfix-ep25.pth.tar",
    "envelope50": r"D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep50.pth.tar",
    "envelope100": r"D:\jepa_phase0\runs\patch_mirage_envelope\jepa_patch_mirage-ep100.pth.tar",
    "anatomy_v1_30": r"D:\jepa_phase0\runs\patch_mirage_anatomy\jepa_patch_mirage-ep30.pth.tar",
    "anatomy40": r"D:\jepa_phase0\runs\anatomy_v2_ep25\jepa_patch_mirage-ep40.pth.tar",
    "anatomy50": r"D:\jepa_phase0\runs\anatomy_v2_ep25\jepa_patch_mirage-ep50.pth.tar",
    "anatomy75": r"D:\jepa_phase0\runs\blob_fp32_ep56\jepa_patch_blob_fp32-ep75.pth.tar",
    "cover50": r"D:\jepa_phase0\runs\cover_f021_ep25\jepa_patch_cover_f021-ep50.pth.tar",
    "cover75": r"D:\jepa_phase0\runs\cover_f021_ep25\jepa_patch_cover_f021-ep75.pth.tar",
    "cover100": r"D:\jepa_phase0\runs\cover_f021_ep25\jepa_patch_cover_f021-ep100.pth.tar",
    "oracle100": r"D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep\jepa_patch_oracle-ep100.pth.tar",
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_dataset(guide_dir):
    cfg = yaml.safe_load((ROOT / r"configs\patch_cover_f021_ep25.yaml").read_text())["data"]
    return GuidedOCTSliceDataset(
        data_dir=str(Path(cfg["data_dir"]) / "Training"), guide_dir=str(guide_dir),
        num_slices=100, slice_size=256, patch_size=16, dilate_patches=0,
        occupancy_threshold=.25, transform=make_paired_transforms(),
        slice_cache=str(Path(cfg["slice_cache_dir"]) / "Training"),
    )


def load_rows(arm):
    with (REPLAY / f"{arm}_bs64.jsonl").open() as stream:
        return {r["ordinal"]: r for line in stream if (r := json.loads(line))}


def pooled_geometry(rows):
    total_tissue = sum(r["tissue_cells"] for r in rows)
    tissue_targets = sum(r["delivered_target_tissue_unique"] for r in rows)
    total_targets = sum(r["unique_target_union"] for r in rows)
    return {
        "n": len(rows),
        "guide_invalid": sum(not r["guide_valid"] for r in rows),
        "mean_guide_tissue_cells": total_tissue / len(rows),
        "mean_context_tokens": np.mean([r["context_tokens"] for r in rows]).item(),
        "mean_context_tissue": np.mean([r["context_tissue"] for r in rows]).item(),
        "tissue_hidden_pct": 100 * tissue_targets / total_tissue,
        "tissue_visible_pct": 100 * sum(r["context_tissue"] for r in rows) / total_tissue,
        "image_hidden_pct": 100 * total_targets / (256 * len(rows)),
        "target_tissue_purity_pct": 100 * tissue_targets / total_targets,
        "zero_tissue_contexts": sum(r["context_tissue"] == 0 for r in rows),
    }


def prepare():
    if (PRIVATE / "fixed_inputs.pt").exists():
        raise FileExistsError("Prepared fixture already exists")
    new_ds = make_dataset(Path(json.loads(BASELINE.read_text())["_meta"]["guide_dir"]) / "Training")
    old_ds = make_dataset(Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training"))
    if new_ds.file_paths != old_ds.file_paths:
        raise ValueError("Guide variants have different image manifests")
    volumes = sorted(random.Random(42).sample(range(len(new_ds.file_paths)), 24))
    indices = [v * 100 + s for v in volumes for s in range(0, 100, 4)]
    new_fixed, old_fixed = FixedCropDataset(new_ds, indices), FixedCropDataset(old_ds, indices)
    saved = {arm: load_rows(arm) for arm in ARMS}
    selected, old_rows, agreement = {}, [], []
    collator = MirageMaskCollator(**arm_kwargs("envelope"))
    collator.set_epoch(50, 100)
    for start in range(0, 576, 64):
        batch = []
        guides_new = []
        for ordinal in range(start, start + 64):
            x, guide, valid, _ = new_fixed[ordinal]
            old_x, old_guide, old_valid, _ = old_fixed[ordinal]
            if not torch.equal(x, old_x):
                raise ValueError("Changing guide product changed input pixels")
            row = saved["envelope"][ordinal]
            if tensor_sha(x) != row["crop_tensor_sha256"] or tensor_sha(guide) != row["guide_sha256"]:
                raise ValueError("Input does not match the fixed saved replay")
            a, b = guide[0] >= .25, old_guide[0] >= .25
            inter, union = int((a & b).sum()), int((a | b).sum())
            agreement.append({
                "ordinal": ordinal, "new_tissue": int(a.sum()), "old_tissue": int(b.sum()),
                "intersection": inter, "union": union,
                "new_guide_valid": bool(valid), "old_guide_valid": bool(old_valid),
            })
            if ordinal in ORDINALS:
                selected[ordinal] = (x, guide, old_guide, valid, old_valid)
            batch.append((old_x, old_guide, old_valid, ordinal))
            guides_new.append(guide)
        seed_all(7103 + start)
        _, enc, pred, stats = collator(batch, block_sizes=drawn_sizes(start // 64))
        validate_delivered(enc, pred, batch_size=64)
        for offset, row in enumerate(stats["delivered_audit"]):
            ordinal = start + offset
            row["ordinal"] = ordinal
            measure(row, batch[offset][1])
            new_tissue = guides_new[offset][0].flatten() >= .25
            row["context_tissue_using_new_guide"] = int(new_tissue[row["context"][0]].sum())
            old_rows.append(row)
        print("Compared historical/new guides", start + 64, flush=True)
    by_old = {r["ordinal"]: r for r in old_rows}
    cases = {}
    for arm in ARMS + ("envelope_original_guide",):
        source = by_old if arm == "envelope_original_guide" else saved[arm]
        cases[arm] = [source[o] for o in ORDINALS]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    fixture = {
        "images": torch.stack([selected[o][0] for o in ORDINALS]),
        "guides": torch.stack([selected[o][1] for o in ORDINALS]),
        "old_guides": torch.stack([selected[o][2] for o in ORDINALS]),
        "ordinals": ORDINALS, "cases": cases,
    }
    torch.save(fixture, PRIVATE / "fixed_inputs.pt")
    payload = {
        "scope": "576 fixed Training views, 24 original sampled volumes; no Test or new training",
        "selection_for_gpu": "One predeclared first view per each of 24 volumes: ordinals 0,25,...575",
        "guide_comparison": agreement,
        "pooled_guide_iou": sum(r["intersection"] for r in agreement) / sum(r["union"] for r in agreement),
        "old_guide_envelope_geometry": pooled_geometry(old_rows),
        "new_guide_envelope_geometry": pooled_geometry([saved["envelope"][i] for i in range(576)]),
        "old_envelope_context_new_definition_mean": np.mean(
            [r["context_tissue_using_new_guide"] for r in old_rows]).item(),
        "limitations": [
            "Neither segmentation product is clinical ground truth.",
            "Guide variants may consume different placement RNG; not identically realized placements.",
            "Current guided loader preserves historical schema-1 behavior; historical launch source not retained for every run.",
        ],
        "fixture_sha256": sha(PRIVATE / "fixed_inputs.pt"),
        "script_sha256": sha(__file__),
    }
    write_json(OUT / "guide_product_comparison.json", payload)
    print("GUIDE_SUMMARY", json.dumps({k: v for k, v in payload.items() if k != "guide_comparison"}))


def state_metadata(path, state):
    opt = state["opt"]
    steps = [float(v["step"]) for v in opt["state"].values() if "step" in v]
    h = hashlib.sha256()
    for component in ("encoder", "predictor", "target_encoder"):
        for name, value in state[component].items():
            h.update((component + ":" + name).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {
        "path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        "model_components_sha256": h.hexdigest(),
        "epoch": state["epoch"], "lr": state["lr"], "loss": state["loss"],
        "batch_size": state["batch_size"], "scaler": state["scaler"],
        "adam_step_min": min(steps), "adam_step_max": max(steps),
        "optimizer_groups": [
            {k: g[k] for k in ("lr", "weight_decay", "eps", "betas")}
            for g in opt["param_groups"]
        ],
    }


def feature_stats(features):
    x = features.double()
    pooled = x.mean(1)
    centered = pooled - pooled.mean(0, keepdim=True)
    values = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    if not float(values.sum()) > 0:
        raise ValueError("Zero between-image pooled feature variance")
    probabilities = values / values.sum()
    nz = probabilities > 0
    return {
        "pooled_centered_rms": float(centered.square().mean().sqrt()),
        "pooled_effective_rank": float(torch.exp(-(probabilities[nz] * probabilities[nz].log()).sum())),
        "pooled_top_eigenvalue_fraction": float(probabilities.max()),
        "same_position_between_image_rms": float((x - x.mean(0, keepdim=True)).square().mean().sqrt()),
        "within_image_spatial_rms": float((x - pooled[:, None]).square().mean().sqrt()),
        "mean_feature_norm": float(torch.linalg.vector_norm(x, dim=-1).mean()),
        "n_images": len(x), "rank_ceiling": len(x) - 1,
    }


def grouped_masks(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["ordinal"] // 64].append(index)
    for indices in groups.values():
        enc = [torch.tensor([rows[i]["context"][0] for i in indices], device="cuda")]
        pred = [
            torch.tensor([rows[i]["targets"][p] for i in indices], device="cuda")
            for p in range(4)
        ]
        validate_delivered(enc, pred, batch_size=len(indices))
        yield indices, enc, pred


@torch.no_grad()
def cross_task(models, images, h_full, tissue, rows):
    encoder, predictor, _ = models
    permutation = torch.roll(torch.arange(len(images), device="cuda"), len(images) // 2)
    sums = defaultdict(float)
    teacher_sum = h_full.sum(0, keepdim=True)
    for indices, enc, pred in grouped_masks(rows):
        ii = torch.tensor(indices, device="cuda")
        target = apply_masks(h_full[ii], pred)
        normal = predictor(encoder(images[ii], enc), enc, pred)
        shuffled = predictor(encoder(images[permutation[ii]], enc), enc, pred)
        template = apply_masks((teacher_sum - h_full[ii]) / (len(images) - 1), pred)
        losses = F.smooth_l1_loss(normal, target, reduction="none").mean(-1)
        shuffled_losses = F.smooth_l1_loss(shuffled, target, reduction="none").mean(-1)
        template_losses = F.smooth_l1_loss(template, target, reduction="none").mean(-1)
        tissue_slots = apply_masks(tissue[ii, :, None].float(), pred).squeeze(-1).bool()
        if not all(torch.isfinite(v).all() for v in (normal, target, losses, shuffled_losses)):
            raise ValueError("Nonfinite checkpoint diagnostic")
        for name, value in (("loss", losses), ("shuffled_loss", shuffled_losses),
                            ("position_template_loss", template_losses)):
            sums[name] += value.double().sum().item()
        sums["slots"] += losses.numel()
        sums["prediction_shuffle_delta_sq"] += (normal - shuffled).double().square().sum().item()
        sums["prediction_elements"] += normal.numel()
        sums["tissue_loss_sum"] += losses[tissue_slots].double().sum().item()
        sums["tissue_slots"] += int(tissue_slots.sum())
        sums["background_loss_sum"] += losses[~tissue_slots].double().sum().item()
        sums["background_slots"] += int((~tissue_slots).sum())
    return {
        "normal_loss": sums["loss"] / sums["slots"],
        "wrong_image_context_loss": sums["shuffled_loss"] / sums["slots"],
        "wrong_image_penalty": (sums["shuffled_loss"] - sums["loss"]) / sums["slots"],
        "leave_one_image_out_position_template_loss": sums["position_template_loss"] / sums["slots"],
        "prediction_change_rms": (sums["prediction_shuffle_delta_sq"] / sums["prediction_elements"]) ** .5,
        "target_slots": int(sums["slots"]),
        "tissue_target_slots": int(sums["tissue_slots"]),
        "tissue_loss": sums["tissue_loss_sum"] / sums["tissue_slots"] if sums["tissue_slots"] else None,
        "background_loss": sums["background_loss_sum"] / sums["background_slots"] if sums["background_slots"] else None,
    }


def backward_check(models, images, h_full, rows, amp):
    encoder, predictor, teacher = models
    indices, enc, pred = next(grouped_masks(rows))
    encoder.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    teacher.zero_grad(set_to_none=True)
    encoder.train()
    predictor.train()
    loss, _, _ = jepa_forward_loss(
        encoder, predictor, teacher, images[indices], enc, pred,
        use_amp=amp, amp_target=False, h_full=h_full[indices],
    )
    scale = 128.0 if amp else 1.0
    (loss * scale).backward()
    result = {"loss": loss.item(), "amp": amp, "diagnostic_loss_scale": scale,
              "images": len(indices), "target_k": pred[0].shape[1]}
    for label, model in (("encoder", encoder), ("predictor", predictor)):
        norms = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Missing/nonfinite gradient in {label}.{name}")
            norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().float() / scale))
        result[label + "_grad_norm"] = sum(n * n for n in norms.values()) ** .5
        result[label + "_zero_grad_parameters"] = [n for n, v in norms.items() if v == 0]
    if any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("Teacher received a gradient")
    encoder.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    encoder.eval()
    predictor.eval()
    return result


def historical_model_parity(models, state, images, rows):
    source = subprocess.check_output(
        ["git", "show", "804c639:src/models/vision_transformer.py"], cwd=ROOT, text=True,
    )
    module = types.ModuleType("historical_vit_804c639")
    exec(compile(source, "historical_vit_804c639", "exec"), module.__dict__)
    old_encoder = module.VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12).cuda().eval()
    old_predictor = module.VisionTransformerPredictor(
        num_patches=256, embed_dim=768, predictor_embed_dim=384, depth=6, num_heads=12).cuda().eval()
    old_encoder.load_state_dict(state["encoder"])
    old_predictor.load_state_dict(state["predictor"])
    indices, enc, pred = next(grouped_masks(rows))
    result = {}
    with torch.no_grad():
        for amp in (False, True):
            with torch.autocast("cuda", enabled=amp, dtype=torch.float16):
                x = models[0](images[indices], enc)
                y = models[1](x, enc, pred)
                old_x = old_encoder(images[indices], enc)
                old_y = old_predictor(old_x, enc, pred)
            result["amp" if amp else "fp32"] = {
                "encoder_max_abs_difference": float((x - old_x).abs().max()),
                "predictor_max_abs_difference": float((y - old_y).abs().max()),
            }
    del old_encoder, old_predictor
    return result


def gpu():
    if (OUT / "checkpoint_diagnostics.json").exists():
        raise FileExistsError("Checkpoint diagnostic results already exist")
    receipt = json.loads((OUT / "guide_product_comparison.json").read_text())
    if sha(PRIVATE / "fixed_inputs.pt") != receipt["fixture_sha256"]:
        raise ValueError("Private fixture changed after preparation")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True,
    )
    if any("python" in line.lower() and not line.startswith(str(os.getpid()) + ",") for line in active.splitlines()):
        raise RuntimeError("Conflicting Python GPU process")
    manifest = {
        "pid": os.getpid(), "status": "running", "optimizer_updates": 0,
        "max_checkpoints": 11, "max_images": 24, "max_backward_calls": 12,
        "purpose": "Fixed-input historical checkpoint sensitivity/feature and actual ANATOMY mask-to-loss diagnosis",
        "fixture_sha256": receipt["fixture_sha256"], "script_sha256": sha(__file__),
        "torch_version": torch.__version__, "cuda": torch.version.cuda,
        "gpu_preflight": active, "code_sha256": {
            path: sha(ROOT / path) for path in (
                r"src\models\vision_transformer.py", r"src\train_patch.py",
                r"src\masks\curriculum.py", r"src\datasets\oct_slices_guided.py",
            )
        },
    }
    write_json(OUT / "gpu_lease.json", manifest)
    fixture = torch.load(PRIVATE / "fixed_inputs.pt", weights_only=True)
    images = fixture["images"].cuda()
    tissue = (fixture["guides"][:, 0].flatten(1) >= .25).cuda()
    encoder, predictor = init_patch_model(
        torch.device("cuda"), model_name="vit_base", crop_size=256, patch_size=16,
        pred_depth=6, pred_emb_dim=384,
    )
    teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
    models = encoder.eval(), predictor.eval(), teacher
    results, backward_calls = {}, 0
    started = time.perf_counter()
    try:
        for label, checkpoint in CHECKPOINTS.items():
            path = Path(checkpoint)
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            meta = state_metadata(path, state)
            for model, key in zip(models, ("encoder", "predictor", "target_encoder")):
                model.load_state_dict(state[key], strict=True)
                model.eval()
            with torch.no_grad():
                raw = torch.cat([teacher(batch) for batch in images.split(8)])
                h_full = F.layer_norm(raw.float(), (raw.shape[-1],))
            stats = feature_stats(raw.cpu())
            tasks = {
                arm: cross_task(models, images, h_full, tissue, rows)
                for arm, rows in fixture["cases"].items()
            }
            checks = []
            if label in ("anatomy40", "anatomy50", "anatomy75", "cover50", "cover100", "envelope100"):
                arm = "anatomy" if label.startswith("anatomy") else (
                    "cover_legacy" if label.startswith("cover") else "envelope_original_guide")
                for amp in (False, True):
                    checks.append(backward_check(models, images, h_full, fixture["cases"][arm], amp))
                    backward_calls += 1
            parity = historical_model_parity(
                models, state, images, fixture["cases"]["anatomy"]) if label == "ancestor25" else None
            results[label] = {
                "checkpoint": meta, "features": stats, "cross_task": tasks,
                "backward": checks, "historical_transformer_parity": parity,
            }
            write_json(OUT / "checkpoint_diagnostics.partial.json", results)
            print(label, json.dumps({
                "features": stats, "own_task": tasks.get(
                    "anatomy" if label.startswith("anatomy") else (
                        "cover_legacy" if label.startswith("cover") else "envelope_original_guide")),
                "backward": checks, "parity": parity,
            }), flush=True)
            del state, raw, h_full
            gc.collect()
        if backward_calls != 12 or len(results) != 11:
            raise ValueError("Diagnostic scope incomplete")
        write_json(OUT / "checkpoint_diagnostics.json", {
            "manifest": manifest, "checkpoints": results,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "backward_calls": backward_calls, "optimizer_updates": 0,
            "limitations": [
                "24 fixed Training images from 24 selected volume files; no clinical labels or Test metrics.",
                "Production batch64 masks preserved, but model forwards use smaller batches; no batch normalization exists.",
                "Wrong-image perturbation is a diagnostic distribution shift, not a counterfactual pretraining campaign.",
                "Feature rank cannot establish clinical usefulness or rule out all representation degeneration.",
                "Mask policies other than original-guide ENVELOPE use the common new guide in the earlier replay.",
                "Model-component digest binds encoder/predictor/teacher weights, not the whole optimizer file.",
                "No optimizer steps, EMA updates, new checkpoints, or sustained pretraining.",
            ],
        })
    finally:
        manifest["status"] = "completed" if (OUT / "checkpoint_diagnostics.json").exists() else "failed"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["backward_calls"] = backward_calls
        write_json(OUT / "gpu_lease.json", manifest)
    print("CHECKPOINT DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "gpu"))
    args = parser.parse_args()
    prepare() if args.phase == "prepare" else gpu()
