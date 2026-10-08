"""Clean symmetric comparison with isolated index-derived predictor queries.

Camera-ready (E13): chains from the regenerated Figure-1 and comparison
evidence (trained CENTROID lateral 0.6) and writes to the staging folder by
default.
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
FIRST = HERE.parent / "figure"
PREVIOUS = HERE.parent / "comparison"
PAPER_FIGURES = ROOT / "paper" / "genai4health2026" / "figures"
STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
OUT = PAPER_FIGURES
STEM = "fig_oct_jepa_comparison_final"
COLORS = ["#F5AE46", "#51C8D4", "#DA8CCC", "#B6DB78"]


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
    parser.add_argument("--comparison-evidence", type=Path,
                        default=STAGING / "fig_oct_jepa_comparison.evidence.json")
    parser.add_argument("--out-dir", type=Path, default=STAGING)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    first_evidence = args.figure_evidence.resolve()
    previous_evidence = args.comparison_evidence.resolve()
    first = json.loads(first_evidence.read_text())
    previous = json.loads(previous_evidence.read_text())
    previous_out = ROOT / previous["out_dir"] if "out_dir" in previous else PAPER_FIGURES
    preserved = {ROOT / k: h for k, h in previous["preserved_v1_v2_files"].items()}
    preserved.update({p: sha(p) for p in [previous_evidence,
        PREVIOUS / "generate_comparison.py",
        *[previous_out / item["file"] for item in previous["outputs"].values()]]})
    assert all(sha(p) == h for p, h in preserved.items())
    source = FIRST / first["source_image_file"]
    assert sha(source) == first["source_image_sha256"]
    image = np.asarray(Image.open(source).crop(first["crop_xyxy"]).convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == first["input_tensor_sha256"]
    guide = np.asarray(first["guide"], dtype=bool)
    edges = [("full_image", "teacher"), ("teacher", "LN_then_select"),
             ("LN_then_select", "z_m"), ("z_m", "loss"),
             ("visible_C", "online_encoder"), ("online_encoder", "h_C"),
             ("h_C", "predictor"), ("mask_indices", "position_queries"),
             ("position_queries", "predictor"), ("predictor", "prediction"),
             ("prediction", "loss"), ("full_image", "sampler"),
             ("guide", "sampler"), ("sampler", "visible_C")]
    assert set(a for a, b in edges if b == "predictor") == {"h_C", "position_queries"}
    assert [b for a, b in edges if a == "z_m"] == ["loss"]
    contracts = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.4,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": STEM}):
        fig = plt.figure(figsize=(6.4, 3.6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 6.4), ylim=(0, 3.6))
        ax.axis("off")

        def text(x, y, s, size=8.4, **kw):
            return ax.text(x, y, s, ha="center", va="center", fontsize=size, **kw)

        def box(x, y, w, h, s, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.010,rounding_size=0.035",
                         facecolor=color, edgecolor="#465463", linewidth=.72))
            if s:
                text(x + w / 2, y + h / 2, s)

        def arrow(start, end, dashed=False):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                         mutation_scale=8, linewidth=.75,
                         linestyle="--" if dashed else "-", color="#35434E"))

        ax.plot([3.2, 3.2], [.48, 3.51], color="#C1C7CD", linewidth=.6)
        for panel, arm in enumerate(("RANDOM", "CENTROID")):
            o = panel * 3.2 + .02
            masks = first["masks"][arm]
            assert masks == previous["mask_contract"][arm]["masks"]
            context = set(masks["context"][0])
            targets = masks["targets"]
            target_union = {i for group in targets for i in group}
            assert len(targets) == 4 and not context.intersection(target_union)
            visible = np.zeros((16, 16), dtype=bool)
            for i in context:
                visible[divmod(i, 16)] = True
            pixels = np.repeat(np.repeat(visible, 16, axis=0), 16, axis=1)
            student = np.repeat(image[:, :, None], 3, axis=2)
            student[~pixels] = [212, 216, 220]
            student_hash = hashlib.sha256(student.tobytes()).hexdigest()
            assert student_hash == previous["mask_contract"][arm]["context_display_sha256"]
            contracts[arm] = {"masks": masks, "context_display_sha256": student_hash}
            text(o + 1.55, 3.47, "(a) Uniform I-JEPA" if not panel
                 else "(b) Anatomy-guided I-JEPA", size=9, fontweight="bold")
            text(o + 1.55, 3.28, "Uniform sampler" if not panel else "CENTROID example")
            box(o + 1.02, 2.93, 1.08, .25, "Smooth-L1", "#FAEBD6")
            text(o + .63, 2.82, r"$z_m$; stop-grad")
            text(o + 2.53, 2.82, r"$\hat{z}_m$")
            arrow((o + .80, 2.88), (o + 1.02, 3.02))
            arrow((o + 2.36, 2.88), (o + 2.10, 3.02))
            box(o + .12, 2.28, 1.02, .38,
                "per-token LN\nselect " + r"$T_m$", "#E2ECF5")
            box(o + 2.04, 2.28, 1.00, .38,
                r"Trainable $g_\phi$" + "\nparallel targets", "#E8E2F5")
            arrow((o + .63, 2.68), (o + .63, 2.74))
            arrow((o + 2.53, 2.68), (o + 2.53, 2.74))
            box(o + 1.21, 2.17, .73, .61, "", "#F5F6F8")
            text(o + 1.575, 2.655, "Index queries", size=8)
            text(o + 1.575, 2.495, r"$p_C,\;p_{T_m}$", size=9)
            text(o + 1.575, 2.335, "+ mask\ntokens", size=8)
            arrow((o + 1.96, 2.47), (o + 2.025, 2.47))
            for x, title, symbol, state, color in [
                (.12, "EMA teacher", r"$f_{\bar{\theta}}$", "no gradients", "#DEECF7"),
                (2.02, "Online encoder", r"$f_\theta$", "trainable", "#DFF0E7"),
            ]:
                box(o + x, 1.60, 1.02, .55, "", color)
                text(o + x + .51, 2.025, title)
                text(o + x + .51, 1.855, symbol, size=11)
                text(o + x + .51, 1.685, state)
            arrow((o + .63, 2.17), (o + .63, 2.26))
            arrow((o + 2.53, 2.17), (o + 2.53, 2.26))
            text(o + 2.89, 2.215, r"$h_C$")
            arrow((o + 2.00, 1.84), (o + 1.16, 1.84), dashed=True)
            text(o + 1.57, 2.01, "EMA weights")
            full_x, context_x, bottom, size = o + .22, o + 2.14, .65, .80
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                      extent=(full_x, full_x + size, bottom, bottom + size))
            ax.imshow(student, interpolation="none",
                      extent=(context_x, context_x + size, bottom, bottom + size))
            for group_index, (group, color) in enumerate(zip(targets, COLORS)):
                inset = group_index * .0009
                for i in group:
                    y, x = divmod(i, 16)
                    ax.add_patch(Rectangle((full_x + x * size / 16 + inset,
                                            bottom + size - (y + 1) * size / 16 + inset),
                                           size / 16 - 2 * inset, size / 16 - 2 * inset,
                                           facecolor="none", edgecolor=color, linewidth=.34))
            arrow((o + .63, 1.47), (o + .63, 1.58))
            arrow((o + 2.53, 1.47), (o + 2.53, 1.58))
            text(o + .62, .54, r"Full $x$ + $T_m$ outlines", size=8.2)
            text(o + 2.54, .54, r"Visible $C$ only")
            box(o + 1.16, .73, .82, .29, "Sampler\n" + r"$C,\;T_m$", "#F2EEE0")
            arrow((o + 1.03, .87), (o + 1.15, .87))
            arrow((o + 2.00, .87), (o + 2.12, .87))
            if not panel:
                text(o + 1.57, 1.35, "No guide")
            else:
                gx, gy, gs = o + 1.31, 1.12, .52
                ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                          extent=(gx, gx + gs, gy, gy + gs))
                for y, x in zip(*np.nonzero(guide)):
                    ax.add_patch(Rectangle((gx + x * gs / 16, gy + gs - (y + 1) * gs / 16),
                                           gs / 16, gs / 16, edgecolor="none",
                                           facecolor=(.25, .8, .6, .48)))
                text(o + 1.57, 1.72, r"$A(x)$")
                arrow((o + 1.57, 1.11), (o + 1.57, 1.04))
        text(1.95, .39, "Target groups:")
        for j, color in enumerate(COLORS):
            x = 2.65 + j * .44
            ax.add_patch(Rectangle((x, .345), .12, .085, facecolor=color,
                                  edgecolor="#465463", linewidth=.45))
            text(x + .245, .39, r"$T_" + str(j + 1) + "$")
        text(3.2, .215,
             r"Evaluation: freeze EMA encoder $\rightarrow$ pool patches / B-scans $\rightarrow$ linear classifier",
             size=8)
        text(3.2, .055,
             "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0); gray = omitted patches, not zero input.",
             size=8)
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{STEM}.{suffix}"
            if path.exists():
                raise FileExistsError("Final exports are write-once: " + str(path))
            metadata = {"Creator": "Matplotlib", "Title": "I-JEPA comparison with isolated index queries"}
            if suffix == "pdf":
                metadata.update(Author="", CreationDate=None, ModDate=None)
            elif suffix == "svg":
                metadata["Date"] = None
            fig.savefig(path, dpi=300, facecolor="white", transparent=False, metadata=metadata)
        plt.close(fig)
    assert all(sha(p) == h for p, h in preserved.items())
    evidence = {
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "prior_comparison_evidence_sha256": sha(previous_evidence),
        "prior_comparison_evidence_path": rel(previous_evidence),
        "figure_evidence_path": rel(first_evidence),
        "figure_evidence_sha256": sha(first_evidence),
        "source_image_url": first["source_image_url"],
        "source_image_sha256": first["source_image_sha256"],
        "primary_license_url": first["primary_license_url"],
        "primary_license_sha256": first["primary_license_sha256"],
        "license": first["license"], "attribution": first["attribution"],
        "input_tensor_sha256": first["input_tensor_sha256"], "crop_xyxy": first["crop_xyxy"],
        "guide": first["guide"], "mask_contract": contracts,
        "source_code": previous["source_code"], "sampling": previous["sampling"],
        "caption": previous["caption"],
        "alt_text": previous["alt_text"] + " A separate Index queries box supplies predictor "
                    "positions and mask tokens. Teacher target embeddings connect only to the loss, "
                    "not to that box or predictor. Teacher barred theta is enlarged and distinct "
                    "from online theta.",
        "semantic_forward_edges": edges,
        "parameter_update_edges": [["online_encoder", "teacher", "EMA; no gradient"]],
        "correctness": {"predictor_inputs_only_context_and_index_queries": True,
                        "teacher_targets_only_flow_to_loss": True,
                        "query_box_is_separate_from_teacher_LN": True,
                        "query_source": "mask indices C and T_m; production positional lookup"},
        "style_reference": "User-described clean large-text symmetric pastel comparison; attachment pixels not used.",
        "attachment_pixels_used": False,
        "figure_inches": [6.4, 3.6], "png_pixels": [1920, 1080],
        "label_font_points": {"base": 8.4, "minimum_nonmath": 8, "encoder_symbols": 11, "titles": 9},
        "target_group_colors": COLORS,
        "privacy": previous["privacy"], "limits": previous["limits"],
        "preserved_prior_files": {rel(p): h for p, h in preserved.items()},
        "validation": {"prior_files_unchanged": True, "source_crop_hash_matches": True,
                       "both_context_displays_byte_identical_to_prior": True,
                       "exact_prior_mask_indices": True, "one_final_export": True},
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
