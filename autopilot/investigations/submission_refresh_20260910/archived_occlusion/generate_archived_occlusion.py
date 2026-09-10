"""Re-layout all six archived pairs without changing a single panel pixel."""
from pathlib import Path
import hashlib
import io
import json
import fitz
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "summary" / "heatmap_grid.png"
OUT = ROOT / "paper" / "genai4health2026" / "figures"
STEM = "fig_archived_occlusion_examples"
Y = [190, 594, 997, 1400, 1804, 2207]
X = [63, 382]
CAPTION = (
    "Selected historical per-patch occlusion overlays from fine-tuned classifiers, "
    "shown only as examples of the archived visualization. All six archived pairs "
    "are retained; A and B are selection labels, not diagnoses or matched subjects "
    "across columns. The original and overlay panels are copied without pixel "
    "alteration. The documented producer omits one patch from a patch mean, using "
    "the remaining P-1 patches, and displays baseline-minus-alternative logit "
    "differences with per-image symmetric scaling, linear interpolation, RdBu_r "
    "and alpha blending. Red and blue indicate positive and negative differences "
    "under that convention, not comparable cross-image magnitudes. Original "
    "attribution arrays and exact historical-run provenance are unavailable; "
    "sub-patch detail, disease localization, population-level attribution and "
    "masking-policy mechanisms must not be inferred."
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data = SOURCE.read_bytes()
    assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == \
        "ad7374313235bbb05239bd84005b62c7c37d4e08"
    source_hash = sha(SOURCE)
    source = Image.open(io.BytesIO(data))
    assert source.size == (731, 2513) and source.mode == "RGBA"
    canvas = Image.new("RGBA", (1920, 960), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = font_manager.findfont("DejaVu Sans")
    normal = ImageFont.truetype(font_path, 34)
    heading = ImageFont.truetype(font_path, 39)

    def label(x, y, text, font=normal):
        draw.text((x, y), text, fill="#202830", font=font, anchor="mm")

    mapping = []
    for column, model in enumerate(("MeanPool", "CrossAttn", "Attentive")):
        left = 38 + column * 628
        label(left + 294, 32, model, heading)
        label(left + 144, 82, "Original")
        label(left + 444, 82, "Archived overlay")
        for row, example in enumerate(("A", "B")):
            source_row = column * 2 + row
            top = 150 + row * 370
            for side, source_x in enumerate(X):
                crop_box = [source_x, Y[source_row], source_x + 288, Y[source_row] + 288]
                crop = source.crop(crop_box)
                destination = [left + side * 300, top, left + side * 300 + 288, top + 288]
                canvas.paste(crop, destination[:2])
                assert np.array_equal(np.asarray(canvas.crop(destination)), np.asarray(crop))
                mapping.append({"column": model, "example": example,
                                "archived_row_one_based": source_row + 1,
                                "panel": "original" if side == 0 else "overlay",
                                "source_crop_xyxy": crop_box, "destination_xyxy": destination,
                                "pixel_sha256": hashlib.sha256(np.asarray(crop).tobytes()).hexdigest()})
    label(960, 125, "Selected example A")
    label(960, 495, "Selected example B")
    label(960, 859, "Archived illustrations only; per-image colour scales are not comparable.")
    label(960, 910, "All six pairs retained. No attribution rerun, clinical validation or pixel alteration.")
    png, pdf = OUT / f"{STEM}.png", OUT / f"{STEM}.pdf"
    assert not png.exists() and not pdf.exists()
    canvas.save(png, dpi=(300, 300))
    reopened = Image.open(png)
    for m in mapping:
        assert np.array_equal(np.asarray(reopened.crop(m["destination_xyxy"])),
                              np.asarray(source.crop(m["source_crop_xyxy"])))
    document = fitz.open()
    page = document.new_page(width=6.4 * 72, height=3.2 * 72)
    page.insert_image(page.rect, stream=png.read_bytes())
    document.set_metadata({"title": "Selected historical occlusion illustrations",
                           "author": "", "subject": "Unaltered archived image panels; qualitative illustration only"})
    document.save(pdf)
    document.close()
    assert sha(SOURCE) == source_hash
    evidence = {
        "generator": Path(__file__).name, "generator_sha256": sha(Path(__file__)),
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "source_git_blob": "ad7374313235bbb05239bd84005b62c7c37d4e08",
        "source_original_preserved": True,
        "authorization": "User explicitly requested reuse of their archived composite; this is not a new FairVision license/deidentification determination.",
        "review_path": "autopilot\\investigations\\delivered_task\\evidence\\legacy_figure_reviews\\REVIEW.md",
        "review_lines": [184, 230],
        "review_sha256": sha(ROOT / "autopilot" / "investigations" / "delivered_task" /
                             "evidence" / "legacy_figure_reviews" / "REVIEW.md"),
        "conditional_scope": "Newly cropped historical illustration only; not a receipt for original empirical/clinical headings.",
        "source_code": [
            {"path": "scripts\\assemble_heatmap_grid.py",
             "sha256": sha(ROOT / "scripts" / "assemble_heatmap_grid.py"),
             "claim": "Fixed six-row mapping: meanpool first pair, crossattn second pair, d1 third pair."},
            {"path": "scripts\\interpretability.py", "lines": [286, 375],
             "sha256": sha(ROOT / "scripts" / "interpretability.py"),
             "claim": "Documented P-1 patch-mean omission; baseline-minus-alt logit; per-image symmetric scale; linear zoom; RdBu_r; alpha blend."}],
        "mapping": mapping,
        "rows_are_not_matched_subjects_across_models": True,
        "removed_outside_panels": ["representativeness heading", "volume and slice identifiers",
                                  "diagnostic labels", "prediction probabilities",
                                  "signed slice contributions", "unreviewed clinical interpretations"],
        "transformations": ["Exact integer pixel crops of all twelve image panels",
                            "Native-resolution paste into new two-row three-column layout",
                            "New neutral external headings; no panel resizing, colour or contrast changes"],
        "caption": CAPTION,
        "alt_text": "Two rows, Selected example A and B, and three columns, MeanPool, CrossAttn "
                    "and Attentive. Each cell pairs an unchanged archived grayscale OCT panel "
                    "with its unchanged red-blue overlay. These are six independently selected "
                    "historical illustrations, not matched subjects, diagnosis labels or quantitative comparisons.",
        "limits": "No recovery of raw attribution arrays or rerun; RGB cannot establish historical executable provenance or numerical accuracy.",
        "figure_inches": [6.4, 3.2], "png_pixels": [1920, 960], "panel_pixels": [288, 288],
        "validation": {"source_blob_matches": True, "all_six_pairs_retained": True,
                       "twelve_exported_panels_pixel_identical": True,
                       "no_panel_resampling": True, "source_unchanged": True},
        "outputs": {p.suffix[1:]: {"file": p.name, "sha256": sha(p)} for p in (png, pdf)},
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generator": evidence["generator_sha256"], "manifest": sha(HERE / "evidence.json"),
                      "outputs": evidence["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
