"""Real, CC-BY external OCT example with exact production-sampler targets.

Camera-ready (E13): CENTROID uses the TRAINED band, lateral fraction 0.6
(configs/patch_oracle_anatomical.yaml; v9 used the code default 0.8), with
everything else unchanged.  Samplers run from a pinned git revision.  Outputs
go to the camera-ready staging folder by default, never over the paper's
figures unless ``--out-dir`` names it explicitly.
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
    install_sampler_sources, load_trained_config, sampler_record)

SAMPLERS = install_sampler_sources("HEAD")
MaskCollator = SAMPLERS["MaskCollator"]
CurriculumMaskGenerator = SAMPLERS["CurriculumMaskGenerator"]

PAPER_FIGURES = ROOT / "paper" / "genai4health2026" / "figures"
STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
STEM = "fig_oct_target_selection"
SOURCE = HERE / "octdl_figure1_original.png"
LICENSE = HERE / "octdl_primary_license.html"
CROP = (3200, 245, 4480, 1525)
BASE = dict(input_size=(256, 256), patch_size=16, enc_mask_scale=(.85, 1.),
            pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5),
            nenc=1, npred=4, min_keep=10, allow_overlap=False)
# Trained CENTROID band (configs/patch_oracle_anatomical.yaml); v9 used .8.
CFG = dict(mode="anatomical_prior", T_warm=0, T_total=1, r_max=1.,
           oracle_region_frac=.28, oracle_lateral_frac=.6,
           oracle_row_offset=0., oracle_min_band_rows=3, audit_masks=True)
CAPTION = (
    "Target-selection intuition on an external OCT example, not FairVision data. "
    "(a) One annotation-free crop from the real OCT B-scan in Kulyabin et al., "
    "OCTDL, Fig. 1 (arXiv:2312.08255v3), adapted under CC BY 4.0. "
    "(b) RANDOM final target union. (c) CENTROID's intensity-derived guide. "
    "(d) CENTROID final target union. The same crop is downsampled to 256 by 256 "
    "pixels without contrast enhancement; masks use the production samplers, "
    "matched block-size draws and fixed seed. Overlays show exact delivered "
    "indices; encoder context is omitted. This single illustrative replay is "
    "not a segmentation annotation, historical training batch, evaluation sample "
    "or performance comparison. Source and license: "
    "https://arxiv.org/abs/2312.08255v3 and https://creativecommons.org/licenses/by/4.0/."
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=STAGING)
    args = parser.parse_args()
    DEST = args.out_dir.resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    trained = {k: v for k, v in load_trained_config("centroid")["curriculum"].items()
               if k.startswith("oracle_")}
    assert trained == {k: v for k, v in CFG.items() if k.startswith("oracle_")}, trained
    assert "creativecommons.org/licenses/by/4.0" in LICENSE.read_text(encoding="utf-8")
    original = Image.open(SOURCE)
    assert original.size == (4768, 1764)
    image = np.asarray(original.crop(CROP).convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    random_sampler = MaskCollator(**BASE, audit_masks=True)
    size_rng = torch.Generator().manual_seed(3107)
    sizes = {"pred": [random_sampler._sample_block_size(BASE["pred_mask_scale"], size_rng)
                      for _ in range(4)],
             "enc": [random_sampler._sample_block_size(BASE["enc_mask_scale"], size_rng)]}
    seed()
    _, enc_r, pred_r = random_sampler([tensor], block_sizes=sizes)
    guided = CurriculumMaskGenerator(**BASE, curriculum_cfg=CFG)
    guided.set_epoch(2, 3)
    assert guided._r_t == 1.
    guide = guided._anatomical_prior_weight_grid_for_image(tensor).numpy()
    seed()
    enc_c, pred_c = guided.generate(1, imgs_cpu=tensor[None], block_sizes=sizes)
    masks = {}
    for name, enc, pred in [("RANDOM", enc_r, pred_r), ("CENTROID", enc_c, pred_c)]:
        targets = [x[0].tolist() for x in pred]
        context = [x[0].tolist() for x in enc]
        union = {i for group in targets for i in group}
        assert union and not union.intersection(context[0])
        assert all(0 <= i < 256 for i in union)
        masks[name] = {"targets": targets, "context": context}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": STEM}):
        fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.0))
        fig.subplots_adjust(left=.017, right=.985, bottom=.25, top=.86, wspace=.15)
        titles = ["(a) OCT B-scan", "(b) RANDOM", "(c) Intensity guide", "(d) CENTROID"]
        labels = ["External OCTDL crop", "Unguided targets", "Computed from this crop",
                  "Intensity-guided targets"]
        for j, ax in enumerate(axes):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255,
                      interpolation="none", extent=(0, 16, 16, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[j], loc="left", fontsize=8, fontweight="bold", pad=7)
            ax.text(.5, -.08, labels[j], transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.2)
            if j in (1, 3):
                name = "RANDOM" if j == 1 else "CENTROID"
                for i in sorted({i for group in masks[name]["targets"] for i in group}):
                    y, x = divmod(i, 16)
                    ax.add_patch(Rectangle((x, y), 1, 1,
                                           facecolor=(1., .62, .08, .42),
                                           edgecolor="#FFD17A", linewidth=.35))
            elif j == 2:
                for y, x in zip(*np.nonzero(guide)):
                    ax.add_patch(Rectangle((x, y), 1, 1,
                                           facecolor=(.1, .75, .85, .36),
                                           edgecolor="#82EBF2", linewidth=.35))
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
        fig.legend(handles=[Patch(facecolor="#EAAF54", edgecolor="#A86100",
                                  label="Final target union"),
                            Patch(facecolor="#5EC6D0", edgecolor="#167B87",
                                  label="Intensity-derived guide")],
                   loc="lower center", bbox_to_anchor=(.5, .05),
                   ncol=2, fontsize=7.3, frameon=False)
        fig.text(.5, .012, "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0), cropped. Illustration only; not FairVision.",
                 ha="center", fontsize=6.5)
        for suffix in ("png", "pdf", "svg"):
            meta = {"Creator": "Matplotlib", "Title": "External OCT target-selection illustration"}
            if suffix == "pdf":
                meta.update(Author="", CreationDate=None, ModDate=None)
            elif suffix == "svg":
                meta["Date"] = None
            fig.savefig(DEST / f"{STEM}.{suffix}", dpi=300, facecolor="white",
                        transparent=False, metadata=meta)
        plt.close(fig)
    evidence = {
        "status": "real_external_OCT_CC_BY_4_0_production_sampler_illustration",
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "source_image_url": "https://arxiv.org/html/2312.08255v3/images/figure_1.png",
        "source_image_file": SOURCE.name, "source_image_sha256": sha(SOURCE),
        "primary_license_url": "https://arxiv.org/abs/2312.08255v3",
        "primary_license_file": LICENSE.name, "primary_license_sha256": sha(LICENSE),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Mikhail Kulyabin et al., OCTDL, Fig. 1, arXiv:2312.08255v3 (2024)",
        "source_original_pixels": list(original.size),
        "crop_xyxy": list(CROP), "crop_rationale": "Right-side annotation-free square within the published B-scan pane.",
        "transformations": ["crop specified rectangle", "PIL grayscale conversion",
                            "whole-crop bilinear downsample 1280x1280 to 256x256",
                            "no contrast enhancement", "semi-transparent categorical overlays"],
        "sampler_config": BASE, "curriculum_config": CFG,
        "centroid_setting": ("trained configuration: configs/patch_oracle_anatomical.yaml "
                             "oracle_* keys (region 0.28, lateral 0.6, offset 0, min rows 3); "
                             "the v9 figure used the code-default lateral 0.8"),
        "seed": 42, "block_size_seed": 3107, "block_sizes": sizes, "batch_size": 1,
        "samplers": sampler_record(),
        "source_code_sha256": {f"src\\masks\\{Path(k).name}": v["sha256_crlf_checkout"]
                               for k, v in sampler_record()["files"].items()},
        "masks": masks, "guide": guide.astype(int).tolist(),
        "input_tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "guide_sha256": hashlib.sha256(guide.tobytes()).hexdigest(),
        "figure_inches": [6.4, 2.0], "png_pixels": [1920, 600],
        "clinical_pixels": "Published external OCTDL crop only; no FairVision clinical pixels",
        "identifiers_or_patient_metadata_exported": False,
        "numerical_readouts": False,
        "limits": "Illustration only; no segmentation ground truth, cohort inference, historical batch reconstruction or performance claim.",
        "caption": CAPTION,
        "alt_text": "Four panels show the same real external OCT crop, RANDOM final targets, "
                    "an intensity-derived centroid ribbon, and CENTROID final targets. "
                    "Orange outlined translucent cells mark targets; cyan cells mark the guide. "
                    "The example does not come from FairVision and shows no performance results.",
        "outputs": {s: {"file": f"{STEM}.{s}", "sha256": sha(DEST / f"{STEM}.{s}")}
                    for s in ("png", "pdf", "svg")},
        "out_dir": str(DEST.relative_to(ROOT)) if DEST.is_relative_to(ROOT) else str(DEST),
        "validation": {"indices_bounded": True, "context_target_disjoint": True,
                       "same_crop_and_matched_block_draws": True,
                       "source_license_primary_verified": True,
                       "source_original_visually_reviewed": True,
                       "source_crop_excludes_author_annotation": True},
    }
    published = json.loads((HERE / "evidence.json").read_text())
    if published["input_tensor_sha256"] == evidence["input_tensor_sha256"]:
        old_guide = np.asarray(published["guide"], dtype=bool)
        evidence["change_vs_published_v9"] = {
            "published_evidence_sha256": sha(HERE / "evidence.json"),
            "published_lateral_frac": published["curriculum_config"]["oracle_lateral_frac"],
            "guide_cells": {"published": int(old_guide.sum()), "camera_ready": int(guide.sum())},
            "random_masks_identical": published["masks"]["RANDOM"] == masks["RANDOM"],
            "centroid_target_union_cells": {
                "published": len({i for g in published["masks"]["CENTROID"]["targets"] for i in g}),
                "camera_ready": len({i for g in masks["CENTROID"]["targets"] for i in g})},
            "centroid_masks_identical": published["masks"]["CENTROID"] == masks["CENTROID"],
        }
    target = HERE / "evidence.json" if DEST == PAPER_FIGURES.resolve() else DEST / f"{STEM}.evidence.json"
    target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ["generator_sha256", "source_image_sha256",
                      "primary_license_sha256", "outputs"]}, indent=2))


if __name__ == "__main__":
    main()
