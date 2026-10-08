"""Symmetric I-JEPA comparison; reuses licensed pixels and reviewed exact masks.

Camera-ready (E13): reads the masks from the regenerated Figure-1 evidence
(trained CENTROID lateral 0.6) and writes to the staging folder by default.
"""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "figure"
FRAMEWORK = HERE.parent / "framework"
PAPER_FIGURES = ROOT / "paper" / "genai4health2026" / "figures"
STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
OUT = PAPER_FIGURES
STEM = "fig_oct_jepa_comparison"
COLORS = ["#F5AE46", "#51C8D4", "#DA8CCC", "#B6DB78"]
CAPTION = (
    "Uniform and anatomy-guided target selection within the same I-JEPA framework. "
    "In both panels, the context encoder and predictor are trainable; the EMA "
    "target encoder receives the full image, and normalized target embeddings "
    "are selected afterward with gradients stopped. The predictor uses visible "
    "context and target-position queries to predict each target group in parallel. "
    "Only the target-selection strategy changes, with its resulting context shown "
    "explicitly. The guided panel illustrates CENTROID, not a ranking or "
    "sequential-prediction method. Its intensity-derived guide feeds only the mask "
    "sampler, not the encoder. The image and exact masks reuse the licensed example "
    "in Figure 1; colours identify target groups, not relevance. Gray regions "
    "visualize omitted patches rather than zero-filled encoder input. Target "
    "outlines do not hide pixels from the teacher. The shared footer shows "
    "frozen-encoder evaluation."
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-evidence", type=Path,
                        default=STAGING / "fig_oct_target_selection.evidence.json")
    parser.add_argument("--out-dir", type=Path, default=STAGING)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    prior_evidence = args.figure_evidence.resolve()
    prior = json.loads(prior_evidence.read_text())
    prior_out = ROOT / prior["out_dir"] if "out_dir" in prior else PAPER_FIGURES
    old_framework = json.loads((FRAMEWORK / "evidence.json").read_text())
    preserved = {p: sha(p) for p in [
        prior_evidence, PRIOR / "generate_external_target_selection.py",
        FRAMEWORK / "evidence.json", FRAMEWORK / "generate_framework.py",
        *[prior_out / d["file"] for d in prior["outputs"].values()],
        *[PAPER_FIGURES / d["file"] for d in old_framework["outputs"].values()]]}
    source = PRIOR / prior["source_image_file"]
    assert sha(source) == prior["source_image_sha256"]
    image = np.asarray(Image.open(source).crop(prior["crop_xyxy"]).convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == prior["input_tensor_sha256"]
    guide = np.asarray(prior["guide"], dtype=bool)
    contract = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": STEM}):
        fig = plt.figure(figsize=(6.4, 3.6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 6.4), ylim=(0, 3.6))
        ax.axis("off")

        def text(x, y, label, **kw):
            return ax.text(x, y, label, ha="center", va="center", fontsize=8, **kw)

        def box(x, y, w, h, label, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.035",
                         facecolor=color, edgecolor="#41505C", linewidth=.65))
            text(x + w / 2, y + h / 2, label)

        def arrow(start, end, dashed=False):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                         mutation_scale=8, linewidth=.7,
                         linestyle="--" if dashed else "-", color="#35434E"))

        ax.plot([3.2, 3.2], [.48, 3.5], color="#BDC4CB", linewidth=.65)
        for p, arm in enumerate(("RANDOM", "CENTROID")):
            o = p * 3.2 + .02
            masks = prior["masks"][arm]
            targets = masks["targets"]
            context = set(masks["context"][0])
            union = {i for group in targets for i in group}
            assert len(targets) == 4 and not context.intersection(union)
            assert all(0 <= i < 256 for i in context | union)
            visible = np.zeros((16, 16), dtype=bool)
            for i in context:
                visible[divmod(i, 16)] = True
            visible_pixels = np.repeat(np.repeat(visible, 16, axis=0), 16, axis=1)
            student = np.repeat(image[:, :, None], 3, axis=2)
            student[~visible_pixels] = [212, 216, 220]
            assert np.array_equal(student[visible_pixels, 0], image[visible_pixels])
            contract[arm] = {"masks": masks,
                             "visible_pixels": int(visible_pixels.sum()),
                             "context_display_sha256": hashlib.sha256(student.tobytes()).hexdigest()}
            text(o + 1.55, 3.47, "(a) Uniform I-JEPA" if p == 0
                 else "(b) Anatomy-guided I-JEPA", fontweight="bold")
            text(o + 1.55, 3.28, "Uniform sampler" if p == 0 else "CENTROID example")
            box(o + 1.02, 2.92, 1.08, .25, "Smooth-L1", "#F9EAD5")
            text(o + .63, 2.82, r"$z_m$; stop-grad")
            text(o + 2.53, 2.82, r"$\hat{z}_m$")
            arrow((o + .80, 2.87), (o + 1.02, 3.015))
            arrow((o + 2.36, 2.87), (o + 2.10, 3.015))
            box(o + .12, 2.28, 1.02, .38,
                "per-token LN\nselect " + r"$T_m$", "#E4EDF5")
            arrow((o + .63, 2.68), (o + .63, 2.74))
            box(o + 2.02, 2.28, 1.02, .38,
                r"Trainable $g_\phi$" + "\nparallel targets", "#E6E1F3")
            arrow((o + 2.53, 2.68), (o + 2.53, 2.74))
            box(o + .12, 1.61, 1.02, .51,
                r"Teacher $f_{\bar{\theta}}$" + "\nEMA; no gradients", "#DDEBF6")
            box(o + 2.02, 1.61, 1.02, .51,
                r"Context $f_\theta$" + "\ntrainable", "#DDEFE6")
            arrow((o + .63, 2.14), (o + .63, 2.26))
            arrow((o + 2.53, 2.14), (o + 2.53, 2.26))
            text(o + 2.88, 2.20, r"$h_C$")
            text(o + 1.57, 2.48, r"$p_C,\;p_{T_m}$" + "\n+ mask tokens")
            arrow((o + 1.92, 2.47), (o + 2.01, 2.47))
            arrow((o + 2.00, 1.84), (o + 1.16, 1.84), dashed=True)
            text(o + 1.57, 2.01, "EMA weights")
            full_x, context_x, bottom, size = o + .22, o + 2.14, .65, .80
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                      extent=(full_x, full_x + size, bottom, bottom + size))
            ax.imshow(student, interpolation="none",
                      extent=(context_x, context_x + size, bottom, bottom + size))
            for group, color in zip(targets, COLORS):
                group_index = targets.index(group)
                inset = group_index * .0009
                for i in group:
                    y, x = divmod(i, 16)
                    ax.add_patch(Rectangle((full_x + x * size / 16 + inset,
                                            bottom + size - (y + 1) * size / 16 + inset),
                                           size / 16 - 2 * inset, size / 16 - 2 * inset,
                                           facecolor="none", edgecolor=color, linewidth=.34))
            arrow((o + .63, 1.47), (o + .63, 1.59))
            arrow((o + 2.53, 1.47), (o + 2.53, 1.59))
            text(o + .62, .54, r"Full $x$ + $T_m$ outlines")
            text(o + 2.54, .54, r"Visible $C$ only")
            box(o + 1.16, .73, .82, .29,
                ("Uniform" if p == 0 else "Guided") + r" sampler" + "\n" + r"$C,\;T_m$",
                "#F1EEE1")
            arrow((o + 1.03, .87), (o + 1.15, .87))
            arrow((o + 2.00, .87), (o + 2.12, .87))
            if p == 0:
                text(o + 1.57, 1.35, "No guide")
            else:
                gleft, gbottom, gs = o + 1.31, 1.12, .52
                ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                          extent=(gleft, gleft + gs, gbottom, gbottom + gs))
                for y, x in zip(*np.nonzero(guide)):
                    ax.add_patch(Rectangle((gleft + x * gs / 16,
                                            gbottom + gs - (y + 1) * gs / 16),
                                           gs / 16, gs / 16,
                                           facecolor=(.25, .8, .6, .48),
                                           edgecolor="none"))
                text(o + 1.57, 1.72, r"$A(x)$")
                arrow((o + 1.57, 1.11), (o + 1.57, 1.04))
        text(1.95, .39, "Target groups:")
        for j, color in enumerate(COLORS):
            x = 2.65 + j * .44
            ax.add_patch(Rectangle((x, .345), .12, .085,
                                  facecolor=color, edgecolor="#41505C", linewidth=.45))
            text(x + .245, .39, r"$T_" + str(j + 1) + "$")
        text(3.2, .215,
             r"Evaluation: freeze EMA encoder $\rightarrow$ pool patches / B-scans $\rightarrow$ linear classifier")
        text(3.2, .055, "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0); gray = omitted patches, not zero input.")
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{STEM}.{suffix}"
            if path.exists():
                raise FileExistsError("Final comparison exports are write-once: " + str(path))
            metadata = {"Creator": "Matplotlib", "Title": "Uniform and anatomy-guided I-JEPA comparison"}
            if suffix == "pdf":
                metadata.update(Author="", CreationDate=None, ModDate=None)
            elif suffix == "svg":
                metadata["Date"] = None
            fig.savefig(path, dpi=300, facecolor="white", transparent=False, metadata=metadata)
        plt.close(fig)
    assert all(sha(p) == expected for p, expected in preserved.items())
    evidence = {
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "prior_evidence_sha256": sha(prior_evidence),
        "prior_evidence_path": rel(prior_evidence),
        "source_image_url": prior["source_image_url"],
        "source_image_sha256": prior["source_image_sha256"],
        "primary_license_url": prior["primary_license_url"],
        "primary_license_sha256": prior["primary_license_sha256"],
        "license": prior["license"], "attribution": prior["attribution"],
        "crop_xyxy": prior["crop_xyxy"], "input_tensor_sha256": prior["input_tensor_sha256"],
        "guide": prior["guide"], "mask_contract": contract,
        "sampling": {"seed": prior["seed"], "block_size_seed": prior["block_size_seed"],
                     "block_sizes": prior["block_sizes"],
                     "sampler_config": prior["sampler_config"],
                     "curriculum_config": prior["curriculum_config"]},
        "source_code": old_framework["source_code"],
        "caption": CAPTION,
        "alt_text": "Two symmetric panels show identical I-JEPA teacher, context encoder, predictor "
                    "and Smooth-L1 pathways. Uniform masks appear left and CENTROID masks right, "
                    "on the same real external OCT crop. Each teacher sees the full image, with "
                    "categorical target outlines; each context thumbnail shows only its exact "
                    "visible patch set. The right intensity guide feeds only its sampler. "
                    "EMA arrows update teacher weights; all targets are predicted in parallel. "
                    "The shared footer describes frozen-encoder evaluation.",
        "figure_inches": [6.4, 3.6], "png_pixels": [1920, 1080], "base_font_points": 8,
        "target_group_colors": COLORS,
        "encoding": "Categorical target groups; slight nested token-border insets reveal overlap. No relevance scale.",
        "privacy": {"identifiers": False, "FairVision_pixels": False,
                    "new_clinical_source": False, "numerical_results": False},
        "limits": "Illustrative external OCT example, not cohort evaluation, ranking, or sequential prediction.",
        "preserved_v1_v2_files": {rel(p): h for p, h in preserved.items()},
        "validation": {"preserved_files_unchanged": True,
                       "same_input_tensor_hash": True,
                       "exact_prior_context_and_target_indices": True,
                       "context_target_disjoint": True,
                       "visible_context_pixels_unchanged": True,
                       "guide_only_feeds_sampler": True,
                       "teacher_full_image_unoccluded": True},
        "outputs": {s: {"file": f"{STEM}.{s}", "sha256": sha(OUT / f"{STEM}.{s}")}
                    for s in ("png", "pdf", "svg")},
        "out_dir": rel(OUT),
    }
    target = HERE / "evidence.json" if OUT == PAPER_FIGURES.resolve() else OUT / f"{STEM}.evidence.json"
    target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generator_sha256": evidence["generator_sha256"],
                      "manifest_sha256": sha(target),
                      "outputs": evidence["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
