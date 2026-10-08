"""All-strategy target-selection panel on the licensed external OCTDL crop.

RANDOM, CENTROID, ENVELOPE, ANATOMY-V2 and COVER on the same real external
OCT crop as Figure 1 (Kulyabin et al., OCTDL, CC BY 4.0; no FairVision pixels).
Every arm uses its production sampler and TRAINED configuration, built by
``scripts/mask_composition_probe.build_audit_arm`` from the run configs:

* CENTROID: configs/patch_oracle_anatomical.yaml (lateral 0.6), guide computed
  from this crop by the sampler itself;
* ENVELOPE: configs/patch_mirage_envelope.yaml, legacy_uniform_v1 fallback,
  on the repaired HARD MIRAGE envelope;
* ANATOMY-V2 / COVER: configs/patch_anatomy_v2.yaml / patch_cover_f021_ep25.yaml
  on the soft MIRAGE guide product.

The MIRAGE guides come from ``produce_external_guides.py`` (production guide
pipelines run on this crop, CPU).  They are routed through the production
dataset code (GuidedOCTSliceDataset.__getitem__ without a crop), so the patch
grids, validity re-check and placement threshold are the training ones.  If
the guides are unavailable, ``--fallback-guide intensity`` substitutes a
clearly labelled intensity-threshold proxy; the default refuses.

Same block-size draws (seed 3107) and placement seed (42) as Figure 1; batch
size 1; samplers from a pinned git revision.  Overlays show exact delivered
target indices; encoder context is omitted, as in Figure 1.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.mask_composition_probe import (  # noqa: E402
    BASE_KW, TRAINED_CONFIGS, build_audit_arm, install_sampler_sources, sampler_record,
    sha_file)
from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.guides.mirage_envelope import DEFAULT_REPAIR  # noqa: E402

SAMPLERS = install_sampler_sources("HEAD")

STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
STEM = "fig_oct_all_strategies"
SOURCE = HERE / "octdl_figure1_original.png"
LICENSE = HERE / "octdl_primary_license.html"
CROP = (3200, 245, 4480, 1525)
ARMS = [("random", "RANDOM"), ("centroid", "CENTROID"), ("envelope", "ENVELOPE"),
        ("anatomy", "ANATOMY-V2"), ("cover", "COVER")]
GUIDE_LABEL = {"random": "No guide", "centroid": "Intensity-centroid ribbon",
               "envelope": "Hard MIRAGE envelope", "anatomy": "Soft MIRAGE scores",
               "cover": "Soft MIRAGE scores"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


class ExternalGuideSlice(GuidedOCTSliceDataset):
    """Production guide-grid code path for one external, uncropped image."""

    def __init__(self, image, envelope=None, envelope_valid=False, soft=None):
        self._image, self._envelope, self._valid, self._soft = image, envelope, envelope_valid, soft
        self.slice_size, self.patch_size, self.grid_size = 256, 16, 16
        self.num_slices, self.transform, self.require_guides = 1, None, True
        self.dilate_patches, self.occupancy_threshold = 0, 0.25
        self.repair_params = DEFAULT_REPAIR
        self.file_paths = ["octdl_external_crop"]
        self.guide_dir = ""

    def read_slice(self, file_idx, slice_within):
        return self._image

    def _load_soft_guide(self, file_index, slice_within):
        return None if self._soft is None else self._soft.astype(np.float32) / 255.0

    def _load_guide(self, file_index, slice_within):
        if self._soft is not None:
            return super()._load_guide(file_index, slice_within)
        return self._envelope, self._valid


def intensity_proxy(native):
    """Labelled fallback only: brightest 40% of pixels as a 200x200 envelope."""
    small = np.asarray(Image.fromarray(native).resize((200, 200), Image.BILINEAR), np.float32)
    threshold = np.quantile(small, 0.6)
    return small >= threshold


def boundary_segments(cells):
    segments = []
    for y, x in zip(*np.nonzero(cells)):
        for dy, dx, seg in ((-1, 0, [(x, y), (x + 1, y)]), (1, 0, [(x, y + 1), (x + 1, y + 1)]),
                            (0, -1, [(x, y), (x, y + 1)]), (0, 1, [(x + 1, y), (x + 1, y + 1)])):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < 16 and 0 <= nx < 16 and cells[ny, nx]):
                segments.append(seg)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guides", type=Path, default=STAGING / "external_guides" / "octdl_crop_guides.npz")
    parser.add_argument("--fallback-guide", choices=("none", "intensity"), default="none")
    parser.add_argument("--out-dir", type=Path, default=STAGING)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert "creativecommons.org/licenses/by/4.0" in LICENSE.read_text(encoding="utf-8")
    published = json.loads((HERE / "evidence.json").read_text())
    original = Image.open(SOURCE)
    assert original.size == (4768, 1764) and sha(SOURCE) == published["source_image_sha256"]
    native = np.asarray(original.crop(CROP).convert("L")).copy()
    image = np.asarray(Image.fromarray(native).resize((256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == published["input_tensor_sha256"]

    guide_record = None
    if args.guides.is_file():
        record_path = args.guides.with_suffix(".json")
        guide_record = json.loads(record_path.read_text())
        assert guide_record["output"]["sha256"] == sha(args.guides)
        with np.load(args.guides) as z:
            envelope, envelope_valid, soft = z["envelope"], bool(z["envelope_valid"]), z["soft_scores"]
        guide_source = "production MIRAGE guide pipelines run on this crop (produce_external_guides.py)"
    elif args.fallback_guide == "intensity":
        envelope = intensity_proxy(native)
        envelope_valid, soft = True, None
        guide_source = ("FALLBACK: brightness-quantile proxy; MIRAGE guides unavailable for "
                        "this external image. ENVELOPE/ANATOMY-V2/COVER panels are NOT the "
                        "production guide product.")
    else:
        raise FileNotFoundError(f"{args.guides} missing; run produce_external_guides.py "
                                "or pass --fallback-guide intensity")

    hard_item = ExternalGuideSlice(image, envelope=envelope, envelope_valid=envelope_valid)[0]
    if soft is not None:
        soft_item = ExternalGuideSlice(image, soft=soft)[0]
    else:
        soft_item = hard_item
    for item in (hard_item, soft_item):
        assert torch.equal(item[0], tensor), "production image path changed the pixels"
    guides = {"hard": hard_item[1][None], "soft": soft_item[1][None]}
    valid = {"hard": hard_item[2][None], "soft": soft_item[2][None]}

    sizer = SAMPLERS["MaskCollator"](**BASE_KW)
    size_rng = torch.Generator().manual_seed(3107)
    sizes = {"pred": [sizer._sample_block_size(BASE_KW["pred_mask_scale"], size_rng) for _ in range(4)],
             "enc": [sizer._sample_block_size(BASE_KW["enc_mask_scale"], size_rng)]}
    assert [list(s) for s in sizes["pred"]] == published["block_sizes"]["pred"]

    masks, records, overlays = {}, {}, {}
    for name, _ in ARMS:
        arm, records[name] = build_audit_arm(name)
        spec = records[name]
        seed()
        if spec["kind"] == "random":
            _, enc, pred = arm([tensor], block_sizes=sizes)
            overlays[name] = np.zeros((16, 16), bool)
        elif spec["kind"] == "centroid":
            enc, pred = arm.generate(1, imgs_cpu=tensor[None], block_sizes=sizes)
            overlays[name] = arm._anatomical_prior_weight_grid_for_image(tensor).numpy() > 0
        else:
            product = spec["guide"]
            enc, pred = arm.generate(1, guide_grids=guides[product], guide_valid=valid[product],
                                     block_sizes=sizes)
            overlays[name] = guides[product][0, 1].numpy() > 0
        targets = [p[0].tolist() for p in pred]
        context = [e[0].tolist() for e in enc]
        union = {i for group in targets for i in group}
        assert union and not union.intersection(context[0])
        assert all(0 <= i < 256 for i in union | set(context[0]))
        masks[name] = {"targets": targets, "context": context}
    staged_figure = out / "fig_oct_target_selection.evidence.json"
    consistency = {}
    if staged_figure.is_file():
        staged = json.loads(staged_figure.read_text())
        consistency = {"random_equals_staged_figure1": staged["masks"]["RANDOM"] == masks["random"],
                       "centroid_equals_staged_figure1": staged["masks"]["CENTROID"] == masks["centroid"],
                       "centroid_guide_equals_staged_figure1":
                           np.asarray(staged["guide"], bool).tolist() == overlays["centroid"].tolist()}

    fallback = soft is None
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": STEM}):
        fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.75))
        fig.subplots_adjust(left=.02, right=.98, bottom=.13, top=.94, wspace=.08, hspace=.30)
        panels = [("(a) OCT B-scan", "External OCTDL crop", None)] + [
            (f"({chr(98 + j)}) {label}", GUIDE_LABEL[name] + (" (fallback proxy)" if fallback and name in ("envelope", "anatomy", "cover") else ""), name)
            for j, (name, label) in enumerate(ARMS)]
        for ax, (title, subtitle, name) in zip(axes.ravel(), panels):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none", extent=(0, 16, 16, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
            ax.text(.5, -.05, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=7)
            if name is not None:
                for i in sorted({i for group in masks[name]["targets"] for i in group}):
                    y, x = divmod(i, 16)
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=(1., .62, .08, .42),
                                           edgecolor="#FFD17A", linewidth=.35))
                for (x0, y0), (x1, y1) in boundary_segments(overlays[name]):
                    ax.plot([x0, x1], [y0, y1], color="#3FE0EE", linewidth=.9, solid_capstyle="butt")
            ax.set_xlim(0, 16)
            ax.set_ylim(16, 0)
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
        fig.legend(handles=[Patch(facecolor="#EAAF54", edgecolor="#A86100", label="Final target union"),
                            Patch(facecolor="none", edgecolor="#22B8C8", label="Placement guide outline")],
                   loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, fontsize=7.3, frameon=False)
        footer = ("OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0), cropped. Illustration only; not FairVision."
                  if not fallback else
                  "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0). MIRAGE guides unavailable: proxy guide shown.")
        fig.text(.5, .008, footer, ha="center", fontsize=6.5)
        for suffix in ("png", "pdf", "svg"):
            meta = {"Creator": "Matplotlib", "Title": "All target-selection strategies on an external OCT crop"}
            if suffix == "pdf":
                meta.update(Author="", CreationDate=None, ModDate=None)
            elif suffix == "svg":
                meta["Date"] = None
            fig.savefig(out / f"{STEM}.{suffix}", dpi=300, facecolor="white", transparent=False, metadata=meta)
        plt.close(fig)
    caption = (
        "All five target-selection strategies on one external OCT crop (Kulyabin et al., OCTDL, "
        "Fig. 1, arXiv:2312.08255v3, CC BY 4.0), not FairVision data. Each panel shows the final "
        "target union of the production sampler at its trained configuration, with matched "
        "block-size draws and a fixed seed; cyan outlines mark the placement guide that strategy "
        "uses. CENTROID's ribbon is computed from the image; the MIRAGE envelope (ENVELOPE) and "
        "soft scores (ANATOMY-V2, COVER) were produced by running the training guide pipelines "
        "on this crop, on which MIRAGE was not trained. Encoder context is omitted. A single "
        "illustrative replay, not a segmentation, historical training batch or performance comparison."
        if not fallback else
        "FALLBACK VERSION: MIRAGE guides could not be produced; ENVELOPE/ANATOMY-V2/COVER use a "
        "brightness proxy guide and do not show the production guide product.")
    evidence = {
        "status": "real_external_OCT_CC_BY_4_0_all_strategy_illustration" + ("_FALLBACK_GUIDES" if fallback else ""),
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "source_image_url": published["source_image_url"], "source_image_file": SOURCE.name,
        "source_image_sha256": sha(SOURCE), "primary_license_url": published["primary_license_url"],
        "primary_license_sha256": sha(LICENSE), "license": "CC BY 4.0",
        "attribution": published["attribution"], "crop_xyxy": list(CROP),
        "input_tensor_sha256": published["input_tensor_sha256"],
        "guide_source": guide_source,
        "guide_npz": {"file": str(args.guides), "sha256": sha(args.guides)} if args.guides.is_file() else None,
        "guide_record": guide_record,
        "guide_grids_sha256": {p: hashlib.sha256(g.numpy().tobytes()).hexdigest() for p, g in guides.items()},
        "guide_valid_after_dataset_qc": {p: bool(v[0]) for p, v in valid.items()},
        "guide_cells": {name: int(overlays[name].sum()) for name, _ in ARMS},
        "arms": {name: {k: v for k, v in records[name].items() if k not in ("curriculum_cfg",)} |
                 {"curriculum_cfg": records[name].get("curriculum_cfg")} for name, _ in ARMS},
        "configs_sha256": {k: sha_file(ROOT / v) for k, v in TRAINED_CONFIGS.items()},
        "samplers": sampler_record(),
        "seed": 42, "block_size_seed": 3107, "block_sizes": sizes, "batch_size": 1,
        "masks": masks, "guide_overlays": {k: v.astype(int).tolist() for k, v in overlays.items()},
        "consistency_with_staged_figure1": consistency,
        "clinical_pixels": "Published external OCTDL crop only; no FairVision clinical pixels",
        "identifiers_or_patient_metadata_exported": False, "numerical_readouts": False,
        "caption": caption,
        "alt_text": ("Six panels show the same real external OCT crop: the plain crop, then the final "
                     "targets of RANDOM, CENTROID, ENVELOPE, ANATOMY-V2 and COVER as orange "
                     "translucent cells, with each strategy's placement guide outlined in cyan."),
        "figure_inches": [6.4, 4.75], "png_pixels": [1920, 1425],
        "outputs": {s: {"file": f"{STEM}.{s}", "sha256": sha(out / f"{STEM}.{s}")} for s in ("png", "pdf", "svg")},
        "out_dir": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
    }
    (out / f"{STEM}.evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"guide_valid": evidence["guide_valid_after_dataset_qc"],
                      "guide_cells": evidence["guide_cells"], "consistency": consistency,
                      "outputs": evidence["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
