"""Measure ramp-zero task changes and conditional mask-placement diversity."""
import json
from pathlib import Path
import random

import numpy as np
import torch

import masking_family_diagnostic as d
from scripts.delivered_mask_audit import BASE, arm_kwargs, drawn_sizes, measure, seed_all
from src.masks.curriculum import CurriculumMaskGenerator, MirageMaskCollator
from src.masks.multiblock import MaskCollator

ARMS = ("random", "oracle", "envelope", "anatomy", "cover_legacy")
TRIALS = 16


def make_sampler(arm, epoch):
    if arm == "random":
        return MaskCollator(**BASE, audit_masks=True)
    if arm == "oracle":
        sampler = CurriculumMaskGenerator(**arm_kwargs(arm))
    else:
        sampler = MirageMaskCollator(**arm_kwargs(arm))
    sampler.set_epoch(epoch, 100)
    return sampler


def generate(sampler, arm, batch, seed, size_draw=0):
    seed_all(seed)
    images = torch.stack([item[0] for item in batch])
    guides = torch.stack([item[1] for item in batch])
    valid = torch.stack([item[2] for item in batch])
    sizes = drawn_sizes(size_draw)
    if arm == "random":
        _, enc, pred = sampler(list(images), block_sizes=sizes)
        rows = sampler.last_mask_audit
    elif arm == "oracle":
        enc, pred = sampler.generate(
            len(batch), imgs_cpu=images, guide_grids=guides,
            guide_valid=valid, block_sizes=sizes)
        rows = sampler.last_mask_audit
    else:
        _, enc, pred, stats = sampler(batch, block_sizes=sizes)
        rows = stats["delivered_audit"]
    d.validate_delivered(enc, pred, batch_size=len(batch))
    for index, row in enumerate(rows):
        row["guide_valid"] = bool(valid[index])
        measure(row, guides[index])
    return rows


def entropy_of_frequencies(frequency):
    result = np.zeros_like(frequency)
    varying = (frequency > 0) & (frequency < 1)
    p = frequency[varying]
    result[varying] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return result


def main():
    output = d.OUT / "placement_diversity_and_ramp_v2.json"
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(d.BASELINE.read_text())
    ds = d.make_dataset(Path(meta["_meta"]["guide_dir"]) / "Training")
    volumes = sorted(random.Random(42).sample(range(len(ds.file_paths)), 24))
    indices = [v * 100 + s for v in volumes for s in range(0, 100, 4)]
    fixed = d.FixedCropDataset(ds, indices)
    batch = [fixed[i] for i in range(64)]
    saved = d.load_rows("envelope")
    for i, item in enumerate(batch):
        if d.tensor_sha(item[0]) != saved[i]["crop_tensor_sha256"]:
            raise ValueError("Fixed crop mismatch")
    tissue = np.stack([item[1][0].numpy().ravel() >= .25 for item in batch])
    results, raw_masks = {}, {}
    for arm in ARMS:
        zero_rows = generate(make_sampler(arm, 25), arm, batch, 50000)
        sampler = make_sampler(arm, 30)
        masks = np.zeros((TRIALS, 64, 256), bool)
        varying_size_masks = np.zeros_like(masks)
        tissue_slots, background_slots = [], []
        for trial in range(TRIALS):
            rows = generate(sampler, arm, batch, 50000 + trial)
            for index, row in enumerate(rows):
                union = {i for group in row["targets"] for i in group}
                masks[trial, index, list(union)] = True
            tissue_slots.extend(r["target_tissue_slots"] for r in rows)
            background_slots.extend(r["target_background_slots"] for r in rows)
            varied_rows = generate(sampler, arm, batch, 50000 + trial, size_draw=trial)
            for index, row in enumerate(varied_rows):
                union = {i for group in row["targets"] for i in group}
                varying_size_masks[trial, index, list(union)] = True
        pair_jaccard = []
        for first in range(TRIALS):
            for second in range(first + 1, TRIALS):
                intersection = (masks[first] & masks[second]).sum(1)
                union = (masks[first] | masks[second]).sum(1)
                if not np.all(union > 0):
                    raise ValueError("Empty target union")
                pair_jaccard.append(intersection / union)
        counts = [
            len({np.packbits(masks[trial, sample]).tobytes() for trial in range(TRIALS)})
            for sample in range(64)
        ]
        entropy = entropy_of_frequencies(masks.mean(0))
        frequency = masks.mean(0)
        varying_frequency = varying_size_masks.mean(0)
        varied_entropy = entropy_of_frequencies(varying_frequency)
        raw_masks[arm] = {
            "fixed_sizes": torch.from_numpy(masks),
            "varied_sizes": torch.from_numpy(varying_size_masks),
        }
        results[arm] = {
            "ramp_zero": {
                "internal_epoch": 25,
                "mean_target_slots": float(np.mean([r["delivered_loss_slots"] for r in zero_rows])),
                "mean_context_tokens": float(np.mean([r["context_tokens"] for r in zero_rows])),
                "mean_tissue_target_slots": float(np.mean([r["target_tissue_slots"] for r in zero_rows])),
                "mean_context_tissue": float(np.mean([r["context_tissue"] for r in zero_rows])),
            },
            "full_guidance": {
                "trials": TRIALS, "images": 64,
                "unique_target_unions_per_image": counts,
                "mean_unique_target_unions": float(np.mean(counts)),
                "images_with_identical_union_all_trials": sum(n == 1 for n in counts),
                "mean_pairwise_target_jaccard": float(np.mean(pair_jaccard)),
                "mean_target_inclusion_entropy_bits_per_cell": float(entropy.mean()),
                "mean_target_inclusion_entropy_bits_per_tissue_cell": float(entropy[tissue].mean()),
                "tissue_cells_changing_target_membership_pct": float(
                    100 * ((frequency[tissue] > 0) & (frequency[tissue] < 1)).mean()),
                "with_sizes_also_varied_tissue_cells_changing_membership_pct": float(
                    100 * ((varying_frequency[tissue] > 0) & (varying_frequency[tissue] < 1)).mean()),
                "with_sizes_also_varied_tissue_entropy": float(varied_entropy[tissue].mean()),
                "mean_tissue_loss_slots": float(np.mean(tissue_slots)),
                "mean_background_loss_slots": float(np.mean(background_slots)),
            },
        }
        print(arm, json.dumps(results[arm]), flush=True)
    private_masks = d.PRIVATE / "placement_replay_masks.pt"
    if private_masks.exists():
        raise FileExistsError(private_masks)
    torch.save({"tissue": torch.from_numpy(tissue), "masks": raw_masks}, private_masks)
    d.write_json(output, {
        "scope": "First 64 fixed Training replay views, production mask batch size64",
        "fixed": "Images, guide product, crop, drawn context/target sizes, full-guidance probability",
        "varied": "16 placement RNG seeds 50000..50015; second condition also uses 16 common independently drawn size sets",
        "source": str(d.REPLAY),
        "script_sha256": d.sha(__file__),
        "private_masks_sha256": d.sha(private_masks),
        "results": results,
        "limitations": [
            "Conditional placement diversity is not total training diversity; random crops and block sizes also vary in training.",
            "ANATOMY uses its historical 64-slot budget, others160 slots in this batch; not a budget-matched causal estimate.",
            "No optimal entropy, downstream performance or causal AUC claim follows from these descriptive statistics.",
            "ENVELOPE uses the common new guide here, not its historical guide product.",
        ],
    })


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
