#!/usr/bin/env python
"""Benchmark I-JEPA microbatch sizes on a single GPU.

The MIRAGE run must keep the original effective batch of 512.  On four T4s that
was 64 x 4 GPUs x 2 accumulation; on one RTX 3090 the equivalent settings are
64 x 1 x 8, or 128 x 1 x 4 if the larger microbatch fits.  This script measures
peak memory and throughput for each candidate so the choice is made from
measurement rather than assumption.

Example:
    python scripts/benchmark_microbatch.py --batch-sizes 64 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.helper import init_patch_model, init_opt  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402
from src.utils.tensors import repeat_interleave_batch  # noqa: E402

EFFECTIVE_BATCH = 512


def build(device, crop_size, patch_size):
    encoder, predictor = init_patch_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name="vit_base",
        pred_depth=6,
        pred_emb_dim=384,
    )
    target_encoder = init_patch_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name="vit_base",
        pred_depth=6,
        pred_emb_dim=384,
    )[0]
    target_encoder.load_state_dict(encoder.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad = False
    return encoder, predictor, target_encoder


def make_generator(crop_size, patch_size):
    return CurriculumMaskGenerator(
        input_size=(crop_size, crop_size),
        patch_size=patch_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg={
            "mode": "mirage_envelope",
            "T_warm": 25,
            "T_total": 30,
            "r_max": 1.0,
        },
    )


def band_guides(batch, grid):
    guide = torch.zeros(batch, 2, grid, grid)
    guide[:, 0, grid // 2 - 2 : grid // 2 + 2, :] = 1.0
    guide[:, 1, grid // 2 - 3 : grid // 2 + 3, :] = 1.0
    return guide


def measure(batch_size, steps, crop_size, patch_size, device):
    accum = EFFECTIVE_BATCH // batch_size
    encoder, predictor, target_encoder = build(device, crop_size, patch_size)
    optimizer, scaler, _sched, _wd = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=100,
        start_lr=1e-4,
        ref_lr=2.5e-4,
        warmup=5,
        num_epochs=100,
        wd=0.04,
        final_wd=0.4,
        final_lr=1e-6,
        use_bfloat16=False,
        ipe_scale=1.0,
    )
    generator = make_generator(crop_size, patch_size)
    generator.set_epoch(30)
    grid = crop_size // patch_size

    imgs_cpu = torch.randn(batch_size, 3, crop_size, crop_size)
    guides = band_guides(batch_size, grid)
    valid = torch.ones(batch_size, dtype=torch.bool)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def one_microbatch(index):
        imgs = imgs_cpu.to(device, non_blocking=True)
        masks_enc, masks_pred = generator.generate(
            batch_size=batch_size,
            imgs_cpu=imgs_cpu,
            guide_grids=guides,
            guide_valid=valid,
        )
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]
        if index % accum == 0:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            h = target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            h_pred = apply_masks(h, masks_pred)
            h_rep = repeat_interleave_batch(h_pred, batch_size, repeat=len(masks_enc))
        with torch.cuda.amp.autocast():
            z = encoder(imgs, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            loss = F.smooth_l1_loss(z, h_rep) / accum
        scaler.scale(loss).backward()
        if (index + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
        return float(loss.item()) * accum

    losses = [one_microbatch(i) for i in range(accum)]  # warm-up window
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for index in range(steps * accum):
        losses.append(one_microbatch(index))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    images = steps * accum * batch_size
    result = {
        "microbatch": batch_size,
        "accum_steps": accum,
        "effective_batch": batch_size * accum,
        "optimizer_steps": steps,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_optimizer_step": round(elapsed / steps, 3),
        "images_per_second": round(images / elapsed, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "finite_loss": bool(all(l == l for l in losses)),
    }
    del encoder, predictor, target_encoder, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "total_memory_gib": round(total, 2),
                "target_effective_batch": EFFECTIVE_BATCH,
            }
        ),
        flush=True,
    )

    results = []
    for batch_size in args.batch_sizes:
        if EFFECTIVE_BATCH % batch_size:
            print(f"skip {batch_size}: does not divide {EFFECTIVE_BATCH}", flush=True)
            continue
        try:
            result = measure(
                batch_size, args.steps, args.crop_size, args.patch_size, device
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            result = {
                "microbatch": batch_size,
                "accum_steps": EFFECTIVE_BATCH // batch_size,
                "status": "oom",
                "error": str(error)[:200],
            }
        results.append(result)
        print(json.dumps(result), flush=True)

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
