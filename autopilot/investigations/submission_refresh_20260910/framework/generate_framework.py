"""Illustrate the production JEPA path using the already-cleared OCTDL crop."""
from pathlib import Path
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
PREVIOUS = HERE.parent / "figure"
DEST = ROOT / "paper" / "genai4health2026" / "figures"
STEM = "fig_oct_jepa_framework"
CAPTION = (
    "OCT-JEPA pretraining and the target-selection intervention. The full image "
    "enters the EMA teacher; per-token layer normalization precedes gathering "
    "the target indices, and the target representations are stop-gradient. "
    "The trainable online encoder retains only the delivered context tokens. "
    "The trainable predictor combines context representations and positional "
    "embeddings with mask-token queries at target positions, predicting target "
    "groups in parallel. Smooth-L1 compares predictions with teacher targets; "
    "teacher weights follow the online encoder by EMA rather than gradients. "
    "Uniform, intensity-derived and segmentation-guided target selection are "
    "alternative interventions, not sequential stages. Orange outlines mark "
    "targets on the full teacher input; gray student regions visualize omitted "
    "tokens, not pathology or a pixel-inpainting objective. Colored feature bars "
    "are schematic embeddings. The external OCTDL crop and exact CENTROID "
    "context/target indices are shared with the target-selection illustration; "
    "they are not FairVision evaluation data. "
    "For downstream evaluation, the frozen EMA encoder supplies pooled patch "
    "and B-scan representations to a linear classifier. "
    "OCT adapted from Kulyabin et al., OCTDL, Fig. 1, arXiv:2312.08255v3, CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/)."
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    previous = json.loads((PREVIOUS / "evidence.json").read_text())
    source = PREVIOUS / previous["source_image_file"]
    assert sha(source) == previous["source_image_sha256"]
    image = np.asarray(Image.open(source).crop(previous["crop_xyxy"]).convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == previous["input_tensor_sha256"]
    masks = previous["masks"]["CENTROID"]
    context = set(masks["context"][0])
    targets = {i for group in masks["targets"] for i in group}
    assert context and targets and not context.intersection(targets)
    visible = np.zeros((16, 16), dtype=bool)
    for i in context:
        visible[divmod(i, 16)] = True
    pixels_visible = np.repeat(np.repeat(visible, 16, axis=0), 16, axis=1)
    student = np.repeat(image[:, :, None], 3, axis=2)
    student[~pixels_visible] = [211, 215, 219]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": STEM}):
        fig = plt.figure(figsize=(6.4, 3))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 6.4), ylim=(0, 3))
        ax.axis("off")

        def text(x, y, s, **kw):
            return ax.text(x, y, s, ha="center", va="center", fontsize=8, **kw)

        def box(x, y, w, h, label, color="#EDF2F7"):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.015,rounding_size=0.045",
                         facecolor=color, edgecolor="#324150", linewidth=.75))
            text(x + w / 2, y + h / 2, label)

        def arrow(start, end, dashed=False):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                         mutation_scale=9, linewidth=.85,
                         linestyle="--" if dashed else "-", color="#35434E"))

        def features(x, y, label, color):
            for j in range(4):
                ax.add_patch(Rectangle((x + j * .10, y), .075, .28,
                                      facecolor=color, edgecolor="#324150", linewidth=.45))
            text(x + .1875, y + .43, label)

        box(.09, 2.52, 6.22, .39,
            "Target selection varies: uniform / intensity / segmentation guidance\n"
            r"Guide $A(x)$, when used $\rightarrow$ sampler $\rightarrow$ delivered $C,\;T_m$",
            "#F2F0E7")
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                  extent=(.10, .94, 1.44, 2.28))
        ax.imshow(student, interpolation="none", extent=(.10, .94, .39, 1.23))
        for i in sorted(targets):
            y, x = divmod(i, 16)
            ax.add_patch(Rectangle((.10 + x * .84 / 16, 2.28 - (y + 1) * .84 / 16),
                                  .84 / 16, .84 / 16, facecolor="none",
                                  edgecolor="#FFC65A", linewidth=.35))
        text(.52, 2.39, r"Full image $x$")
        text(.65, 1.33, r"Visible patches $C$ only")
        box(1.14, 1.62, 1.12, .57, "EMA teacher\n" + r"$f_{\bar{\theta}}$" + "\nno gradients",
            "#E1EDF6")
        box(1.14, .50, 1.12, .57, "Online encoder\n" + r"$f_\theta$" + "\ntrainable",
            "#E0F1E9")
        arrow((.96, 1.87), (1.12, 1.87))
        arrow((.96, .79), (1.12, .79))
        arrow((1.70, 1.09), (1.70, 1.60), dashed=True)
        text(2.20, 1.33, "EMA\nweights")
        box(2.52, 1.63, .86, .54, "Full tokens\nper-token LN")
        arrow((2.28, 1.90), (2.50, 1.90))
        box(3.60, 1.67, .63, .46, "Gather\n" + r"$T_m$")
        arrow((3.40, 1.90), (3.58, 1.90))
        features(4.47, 1.76, r"$z_m$" + "\nstop-grad", "#76A9CE")
        arrow((4.25, 1.90), (4.44, 1.90))
        features(2.57, .65, r"$h_C$", "#7DBA9B")
        arrow((2.28, .79), (2.54, .79))
        box(3.18, .50, 1.12, .57, "Predictor " + r"$g_\phi$" + "\ntrainable\nparallel targets",
            "#E0F1E9")
        arrow((2.99, .79), (3.16, .79))
        text(3.78, 1.36, r"$p_C,\;p_{T_m}$" + "\n+ mask tokens")
        arrow((3.78, 1.18), (3.78, 1.09))
        features(4.53, .65, r"$\hat{z}_m$", "#EAAF54")
        arrow((4.32, .79), (4.50, .79))
        box(5.33, 1.07, .95, .62, "Smooth-L1\n" + r"$\hat{z}_m$ vs. $z_m$",
            "#FAECD9")
        arrow((4.90, 1.90), (5.34, 1.57))
        arrow((4.96, .79), (5.35, 1.17))
        text(5.62, .73, "Schematic\nembeddings")
        box(1.16, .345, 5.10, .13,
            r"Evaluation: frozen EMA $\rightarrow$ pool patches / B-scans $\rightarrow$ linear classifier",
            "#F4F4F4")
        text(3.2, .27, "Orange: target outlines, not hidden. Gray: omitted patches, not zero-filled input.")
        text(3.2, .10, "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0), cropped; not FairVision.")
        for suffix in ("png", "pdf", "svg"):
            meta = {"Creator": "Matplotlib", "Title": "OCT-JEPA framework with target-selection intervention"}
            if suffix == "pdf":
                meta.update(Author="", CreationDate=None, ModDate=None)
            if suffix == "svg":
                meta["Date"] = None
            fig.savefig(DEST / f"{STEM}.{suffix}", dpi=300, facecolor="white",
                        transparent=False, metadata=meta)
        plt.close(fig)
    evidence = {
        "status": "production_path_illustration_with_licensed_external_OCT",
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "prior_figure_evidence": str((PREVIOUS / "evidence.json").relative_to(ROOT)),
        "prior_figure_evidence_sha256": sha(PREVIOUS / "evidence.json"),
        "source_image_url": previous["source_image_url"],
        "source_image_sha256": sha(source),
        "primary_license_url": previous["primary_license_url"],
        "primary_license_sha256": previous["primary_license_sha256"],
        "license": "CC BY 4.0",
        "attribution": previous["attribution"],
        "crop_xyxy": previous["crop_xyxy"],
        "input_tensor_sha256": previous["input_tensor_sha256"],
        "masks": masks, "sampler": "previously reviewed CENTROID exact delivered indices",
        "source_code": [
            {"path": "src\\train_patch.py", "lines": [129, 142],
             "sha256": sha(ROOT / "src" / "train_patch.py"),
             "claim": "no_grad teacher, per-token LN then gather, predictor+online, SmoothL1"},
            {"path": "src\\models\\vision_transformer.py", "lines": [447, 461],
             "sha256": sha(ROOT / "src" / "models" / "vision_transformer.py"),
             "claim": "context patch tokens selected before encoder transformer blocks"},
            {"path": "src\\models\\vision_transformer.py", "lines": [576, 611],
             "sha256": sha(ROOT / "src" / "models" / "vision_transformer.py"),
             "claim": "context position addition, mask tokens plus target positions, parallel target groups"},
            {"path": "src\\helper.py", "lines": [244, 248],
             "sha256": sha(ROOT / "src" / "helper.py"),
             "claim": "EMA update from detached online parameters; no teacher gradients"},
            {"path": "src\\eval_downstream.py", "lines": [93, 111, 189, 470, 481, 619, 623],
             "sha256": sha(ROOT / "src" / "eval_downstream.py"),
             "claim": "frozen target encoder, patch means, mean-pool option, linear classifier"},
        ],
        "caption": CAPTION,
        "alt_text": "Real external OCT images anchor two JEPA branches. The full-image EMA "
                    "teacher produces normalized target representations without gradients. "
                    "The online encoder sees only delivered context; its trainable predictor "
                    "uses positional mask-token queries to predict parallel target groups. "
                    "Smooth-L1 compares the branches. A dashed arrow updates teacher weights "
                    "by EMA. Colored bars are schematic, gray input tiles are omitted tokens.",
        "privacy": {"patient_identifiers": False, "numerical_readouts": False,
                    "FairVision_pixels": False, "new_source_image": False},
        "transformations": previous["transformations"][:4] + [
            "teacher image unchanged; target outlines are visual annotations only",
            "student pixels outside delivered context rendered neutral gray for illustration",
            "no neural features computed; bars are explicitly schematic"],
        "figure_inches": [6.4, 3.0], "png_pixels": [1920, 900],
        "font_points": 8,
        "validation": {"input_tensor_hash_matches_prior_figure": True,
                       "context_target_disjoint": True,
                       "gray_occlusion_exactly_complements_context": True,
                       "teacher_pixels_not_occluded": True},
        "outputs": {s: {"file": f"{STEM}.{s}", "sha256": sha(DEST / f"{STEM}.{s}")}
                    for s in ("png", "pdf", "svg")},
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generator_sha256": evidence["generator_sha256"],
                      "evidence_sha256": sha(HERE / "evidence.json"),
                      "outputs": evidence["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
