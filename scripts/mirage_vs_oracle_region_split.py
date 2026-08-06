#!/usr/bin/env python
"""Split ORACLE vs MIRAGE target placement into inner retina vs choroid.

The frozen MeanPool sweep produced a paradox: MIRAGE masks the retina more
purely than the hand-crafted oracle band (0.632 vs 0.560 purity over 1,000
volumes, `mirage_guided_masking.md`) yet its encoder is *not* better downstream
(0.8807 vs 0.8855 Test AUC at ep100, `frozen/mirage_meanpool_sweep.md`).

The standing hypothesis to test is that the two priors claim *different tissue*,
not just different amounts of it: that MIRAGE, which segments the choroid
explicitly and gets a thick confident band there, ends up putting less of its
masked area on the **inner retina** (RNFL + GCIPL — the layers glaucoma
actually thins) and more on the **choroid** than the oracle band does.

Nothing in the pipeline measures that, because both the policy sweep and
`oracle_failure_cases.py` score against a single binary "is this retina" truth
which merges all three MIRAGE classes.  This script re-scores the same
placements against the *class* map instead:

    inner  = CLASS_RNFL (1) + CLASS_GCIPL (2)
    choroid= CLASS_CHOROID (3)
    fill   = inside the repaired envelope but unlabelled by MIRAGE
             (the mid-retina the repair closes over: INL/OPL/ONL/RPE)

Both arms are built exactly as the training run builds them:

  * oracle   `CurriculumMaskGenerator._anatomical_prior_weight_grid_for_image`
             driven by `configs/patch_oracle_anatomical.yaml`, same as
             `scripts/oracle_failure_cases.py`
  * MIRAGE   `GuidedOCTSliceDataset` placement channel at the shipped policy
             (occupancy >= 0.25, dilation 0) plus
             `CurriculumMaskGenerator(mode=mirage_envelope)`, same as
             `scripts/compare_mirage_vs_oracle.py`

and both are sampled with the *same* RNG seed per slice, so the four block
sizes are identical across arms and only placement moves.

It also measures placement freedom (hypothesis "B2": the driver is masked-area
entropy, not purity).  For each arm and each of the four sampled block sizes it
counts how many top-left positions the sampler could have chosen, and reports
log2 of that count — the entropy of a uniform draw over admissible windows.
The oracle's true sampler is multinomial over block-summed weights, so its
exact Shannon entropy is reported as well.

No MIRAGE inference is run: the cached hard label maps under ``--mask-dir`` are
read directly, so this needs only the repo venv.

    python scripts/mirage_vs_oracle_region_split.py --volumes 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PIL import Image  # noqa: E402

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.guides.mirage_envelope import (  # noqa: E402
    CLASS_CHOROID,
    CLASS_GCIPL,
    CLASS_RNFL,
    patch_occupancy,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

CROP, PATCH, GRID = 256, 16, 16
NATIVE = 200
INNER_CLASSES = (CLASS_RNFL, CLASS_GCIPL)
MASK_DIR = r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training"


class RegionSplitDataset(GuidedOCTSliceDataset):
    """``GuidedOCTSliceDataset`` that also returns the aligned MIRAGE classes.

    The class map has to travel through the *same* RandomResizedCrop as the
    image and the envelope or it points at the wrong anatomy.  Rather than
    reimplementing the crop, the paired transform is called twice with the torch
    RNG state restored in between: ``RandomResizedCrop.get_params`` draws from
    the torch generator, so the second call reproduces the first call's exact
    rectangle.  The image tensor returned is the one from the first call, i.e.
    bit-identical to what training sees.
    """

    def __init__(self, mask_dir, **kwargs):
        super().__init__(**kwargs)
        self.mask_dir = mask_dir

    def _load_labels(self, file_index, slice_within):
        name = os.path.basename(self.file_paths[file_index])
        path = os.path.join(self.mask_dir, name)
        with np.load(path, allow_pickle=False) as cache:
            source = str(cache["source_filename"].item())
            if source != name:
                raise RuntimeError("Mask %s was built from %s" % (path, source))
            if not np.array_equal(cache["slice_indices"], self.slice_indices):
                raise RuntimeError("Mask %s slice indices do not match" % path)
            return np.asarray(cache["hard_masks"][slice_within], dtype=np.uint8)

    def __getitem__(self, idx):
        file_idx = idx // self.num_slices
        slice_within = idx % self.num_slices

        slice_2d = self.read_slice(file_idx, slice_within)
        image = (
            Image.fromarray(slice_2d, mode="L")
            .resize((self.slice_size, self.slice_size), Image.BILINEAR)
            .convert("RGB")
        )
        envelope, valid = self._load_guide(file_idx, slice_within)
        if envelope is None:
            raise FileNotFoundError("No guide for %s" % self.file_paths[file_idx])
        guide_image = Image.fromarray(
            envelope.astype(np.uint8) * 255, mode="L"
        ).resize((self.slice_size, self.slice_size), Image.NEAREST)
        label_image = Image.fromarray(
            self._load_labels(file_idx, slice_within), mode="L"
        ).resize((self.slice_size, self.slice_size), Image.NEAREST)

        state = torch.get_rng_state()
        tensor, cropped_guide = self.transform(image, guide_image)
        torch.set_rng_state(state)
        _, cropped_labels = self.transform(image, label_image)

        grid = patch_occupancy(np.asarray(cropped_guide) > 127, patch_size=self.patch_size)
        placement = grid >= self.occupancy_threshold
        labels = np.asarray(cropped_labels)
        inner = patch_occupancy(np.isin(labels, INNER_CLASSES), patch_size=self.patch_size)
        choroid = patch_occupancy(labels == CLASS_CHOROID, patch_size=self.patch_size)
        return (
            tensor,
            torch.from_numpy(
                np.ascontiguousarray(
                    np.stack([grid.astype(np.float32), placement.astype(np.float32)], 0)
                )
            ),
            torch.tensor(bool(valid), dtype=torch.bool),
            inner,
            choroid,
        )


def _seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))


def _make_generator(mode_cfg, mask_cfg):
    return CurriculumMaskGenerator(
        input_size=(CROP, CROP),
        patch_size=PATCH,
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        allow_overlap=mask_cfg["allow_overlap"],
        curriculum_cfg=mode_cfg,
    )


def _target_union(generator, kind, img_t, guide, seed):
    """Union of the four target blocks for one arm, as a (GRID, GRID) bool."""
    _seed_all(seed)
    if kind == "oracle":
        _enc, pred = generator.generate(batch_size=1, imgs_cpu=img_t.unsqueeze(0))
    else:
        _enc, pred = generator.generate(
            batch_size=1,
            guide_grids=guide,
            guide_valid=torch.ones(1, dtype=torch.bool),
        )
    flat = np.zeros(GRID * GRID, dtype=bool)
    for block in pred:
        flat[block[0].numpy()] = True
    return flat.reshape(GRID, GRID)


def _window_counts(region, block_h, block_w):
    """Number of (block_h, block_w) windows whose fill clears each threshold."""
    padded = np.zeros((GRID + 1, GRID + 1), dtype=np.float64)
    padded[1:, 1:] = region
    sat = padded.cumsum(0).cumsum(1)
    n_top, n_left = GRID - block_h + 1, GRID - block_w + 1
    if n_top <= 0 or n_left <= 0:
        return None
    sums = (
        sat[block_h:, block_w:]
        - sat[:n_top, block_w:]
        - sat[block_h:, :n_left]
        + sat[:n_top, :n_left]
    )
    return sums / float(block_h * block_w)


def _entropy(weights):
    total = weights.sum()
    if total <= 0:
        return float(np.log2(weights.size))
    p = weights.ravel() / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _split(occ_inner, occ_choroid, cells):
    """Mean per-cell pixel fraction that is inner retina / choroid."""
    n = int(cells.sum())
    if n == 0:
        return 0.0, 0.0, 0
    return (
        float(occ_inner[cells].sum() / n),
        float(occ_choroid[cells].sum() / n),
        n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/patch_mirage_envelope.yaml")
    ap.add_argument("--oracle-config", default="configs/patch_oracle_anatomical.yaml")
    ap.add_argument("--mask-dir", default=MASK_DIR)
    ap.add_argument("--volumes", type=int, default=400)
    ap.add_argument("--slice-stride", type=int, default=17)
    ap.add_argument("--max-slices", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument(
        "--spread",
        choices=("config", "on", "off"),
        default="config",
        help="override mirage_spread; 'off' reproduces the policy sweep, which "
             "measured MIRAGE with spread disabled while the shipped config "
             "trains with it enabled",
    )
    ap.add_argument("--out", default="results/masking/region_split.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    oracle_curr = yaml.safe_load(open(args.oracle_config))["mask"]["curriculum"]
    d, m, curr = cfg["data"], cfg["mask"], dict(cfg["mask"]["curriculum"])
    if args.spread != "config":
        curr["mirage_spread"] = args.spread == "on"

    ds = RegionSplitDataset(
        mask_dir=args.mask_dir,
        data_dir=os.path.join(d["data_dir"], "Training"),
        guide_dir=os.path.join(curr["mirage_guide_dir"], "Training"),
        num_slices=d["num_slices"],
        slice_size=CROP,
        transform=make_paired_transforms(
            crop_size=CROP,
            crop_scale=tuple(d["crop_scale"]),
            horizontal_flip=False,
            gaussian_blur=False,
            color_distortion=False,
            color_jitter=0.0,
        ),
        patch_size=PATCH,
        dilate_patches=int(curr["mirage_dilate_patches"]),
        occupancy_threshold=float(curr["mirage_occupancy_threshold"]),
        slice_cache=os.path.join(d["slice_cache_dir"], "Training"),
    )

    ocfg = dict(curr)
    ocfg.update(
        {k: v for k, v in oracle_curr.items() if k == "mode" or k.startswith("oracle_")}
    )
    oracle_gen = _make_generator(ocfg, m)
    mirage_gen = _make_generator(curr, m)
    oracle_gen.set_epoch(60, cfg["optimization"]["epochs"])
    mirage_gen.set_epoch(60, cfg["optimization"]["epochs"])
    min_fill = float(curr["mirage_min_block_fill"])

    rng = np.random.default_rng(args.seed)
    n_vol = min(args.volumes, len(ds.file_paths))
    vol_ids = rng.choice(len(ds.file_paths), size=n_vol, replace=False)

    acc = {
        arm: {k: [] for k in ("region_inner", "region_choroid", "region_area",
                              "target_inner", "target_choroid", "target_area",
                              "windows", "bits")}
        for arm in ("oracle", "mirage")
    }
    oracle_true_bits = []
    scanned = 0

    for vi in vol_ids:
        for sl in range(0, ds.num_slices, args.slice_stride):
            idx = int(vi) * ds.num_slices + sl
            _seed_all(idx)
            img_t, guide_t, valid_t, inner, choroid = ds[idx]
            if not bool(valid_t):
                continue
            occ = guide_t[0].numpy()
            mirage_region = guide_t[1].numpy() > 0
            if mirage_region.sum() < 4:
                continue
            oracle_region = (
                oracle_gen._anatomical_prior_weight_grid_for_image(img_t).numpy() > 0
            )
            if oracle_region.sum() == 0:
                continue
            scanned += 1

            guide = torch.from_numpy(
                np.stack([occ, mirage_region.astype(np.float32)], 0)
            ).unsqueeze(0)
            seed = 77 + (idx % 1000)

            regions = {"oracle": oracle_region, "mirage": mirage_region}
            for arm, region in regions.items():
                r_in, r_ch, r_n = _split(inner, choroid, region)
                acc[arm]["region_inner"].append(r_in)
                acc[arm]["region_choroid"].append(r_ch)
                acc[arm]["region_area"].append(r_n / float(GRID * GRID))

                gen = oracle_gen if arm == "oracle" else mirage_gen
                union = _target_union(gen, arm, img_t, guide, seed)
                t_in, t_ch, t_n = _split(inner, choroid, union)
                acc[arm]["target_inner"].append(t_in)
                acc[arm]["target_choroid"].append(t_ch)
                acc[arm]["target_area"].append(t_n / float(GRID * GRID))

            # Placement freedom, measured on the SAME four block sizes both
            # arms would draw for this slice.
            gen_t = torch.Generator()
            gen_t.manual_seed(seed)
            sizes = [
                mirage_gen._sample_block_size(tuple(m["pred_mask_scale"]), gen_t)
                for _ in range(m["num_pred_masks"])
            ]
            for arm, region in regions.items():
                counts, bits = [], []
                for bh, bw in sizes:
                    fill = _window_counts(region.astype(np.float64), bh, bw)
                    if fill is None:
                        continue
                    admissible = int((fill >= min_fill).sum())
                    counts.append(admissible)
                    bits.append(float(np.log2(admissible)) if admissible else 0.0)
                if counts:
                    acc[arm]["windows"].append(float(np.mean(counts)))
                    acc[arm]["bits"].append(float(np.mean(bits)))
            true_bits = []
            for bh, bw in sizes:
                fill = _window_counts(oracle_region.astype(np.float64), bh, bw)
                if fill is not None:
                    true_bits.append(_entropy(np.clip(fill * bh * bw, 1e-9, None)))
            if true_bits:
                oracle_true_bits.append(float(np.mean(true_bits)))

        if scanned >= args.max_slices:
            break

    def mean(arm, key):
        return float(np.mean(acc[arm][key]))

    report = {
        "slices": scanned,
        "volumes": int(n_vol),
        "occupancy_threshold": float(curr["mirage_occupancy_threshold"]),
        "dilate_patches": int(curr["mirage_dilate_patches"]),
        "min_block_fill": min_fill,
        "mirage_spread": bool(curr.get("mirage_spread", True)),
    }
    for arm in ("oracle", "mirage"):
        report[arm] = {k: mean(arm, k) for k in acc[arm]}
        tissue = report[arm]["target_inner"] + report[arm]["target_choroid"]
        report[arm]["target_inner_share_of_tissue"] = (
            report[arm]["target_inner"] / tissue if tissue else float("nan")
        )
        rtissue = report[arm]["region_inner"] + report[arm]["region_choroid"]
        report[arm]["region_inner_share_of_tissue"] = (
            report[arm]["region_inner"] / rtissue if rtissue else float("nan")
        )
    report["oracle_true_placement_bits"] = float(np.mean(oracle_true_bits))

    paired = {
        "slices_mirage_less_inner": float(
            np.mean(
                np.asarray(acc["mirage"]["target_inner"])
                < np.asarray(acc["oracle"]["target_inner"])
            )
        ),
        "slices_mirage_more_choroid": float(
            np.mean(
                np.asarray(acc["mirage"]["target_choroid"])
                > np.asarray(acc["oracle"]["target_choroid"])
            )
        ),
    }
    report["paired"] = paired

    print("scanned %d slices from %d volumes" % (scanned, n_vol))
    print()
    print("ADMISSIBLE REGION  (cells a target block may be drawn from)")
    print("%-34s %9s %9s" % ("", "ORACLE", "MIRAGE"))
    for label, key in (
        ("area (fraction of frame)", "region_area"),
        ("inner retina (RNFL+GCIPL) frac", "region_inner"),
        ("choroid frac", "region_choroid"),
        ("inner share of labelled tissue", "region_inner_share_of_tissue"),
    ):
        print("%-34s %9.3f %9.3f" % (label, report["oracle"][key], report["mirage"][key]))
    print()
    print("PLACED TARGET BLOCKS  (union of the four blocks)")
    print("%-34s %9s %9s" % ("", "ORACLE", "MIRAGE"))
    for label, key in (
        ("area (fraction of frame)", "target_area"),
        ("inner retina (RNFL+GCIPL) frac", "target_inner"),
        ("choroid frac", "target_choroid"),
        ("inner share of labelled tissue", "target_inner_share_of_tissue"),
    ):
        print("%-34s %9.3f %9.3f" % (label, report["oracle"][key], report["mirage"][key]))
    print()
    print("ratio MIRAGE/ORACLE  inner %.3f   choroid %.3f"
          % (report["mirage"]["target_inner"] / max(report["oracle"]["target_inner"], 1e-9),
             report["mirage"]["target_choroid"] / max(report["oracle"]["target_choroid"], 1e-9)))
    print("slices where MIRAGE places LESS inner retina : %.1f%%"
          % (100 * paired["slices_mirage_less_inner"]))
    print("slices where MIRAGE places MORE choroid      : %.1f%%"
          % (100 * paired["slices_mirage_more_choroid"]))
    print()
    print("PLACEMENT FREEDOM  (per block size, fill >= %.2f)" % min_fill)
    print("%-34s %9s %9s" % ("", "ORACLE", "MIRAGE"))
    print("%-34s %9.1f %9.1f" % ("admissible windows",
                                 report["oracle"]["windows"], report["mirage"]["windows"]))
    print("%-34s %9.2f %9.2f" % ("bits (log2 admissible windows)",
                                 report["oracle"]["bits"], report["mirage"]["bits"]))
    print("%-34s %9.2f %9s" % ("oracle sampler true entropy (bits)",
                               report["oracle_true_placement_bits"], "n/a"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
