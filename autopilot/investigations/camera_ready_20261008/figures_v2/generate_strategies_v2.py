"""Strategy-uniqueness figure (camera-ready v2, I13): five target-selection
strategies on one external OCT crop, in an I-JEPA-style encoding.

Row (a) shows each strategy's placement guide A_s(x) on the 16x16 patch grid
(none for RANDOM; the intensity-centroid band for CENTROID; the repaired hard
MIRAGE envelope cells for ENVELOPE; the summed soft MIRAGE class scores for
ANATOMY-V2 and COVER). Row (b) shows what one I-JEPA training view delivers:
visible context C (image), every target group T1..T4 in its own Okabe-Ito
colour, and cells that are neither (hidden from the context encoder but not
predicted). Repeated indices inside a target group are marked.

Content is not redrawn by hand. Masks are the exact delivered indices of the
staged camera-ready replay (fig_oct_all_strategies.evidence.json: trained
sampler configurations, matched block-size draws seed 3107, placement seed 42,
batch size 1, samplers pinned to commit 86ae882). The guide grids are rebuilt
through the production dataset code path (GuidedOCTSliceDataset.__getitem__)
from the staged guide products and must hash-match the staged replay. With
``--replay`` the masks are recomputed from the pinned sampler sources and must
equal the staged indices.

OCT pixels: Kulyabin et al., OCTDL Fig. 1 (arXiv:2312.08255v3), CC BY 4.0;
no FairVision pixels. Illustration only; not a segmentation, historical batch
or performance comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle_v2 as st  # noqa: E402

ROOT = st.ROOT
sys.path.insert(0, str(ROOT))

STEM = "fig_oct_strategies_v2"
POSTER_STEM = "fig_oct_strategies_v2_poster"
STAGED_EVIDENCE = st.STAGING / "fig_oct_all_strategies.evidence.json"
STAGED_EVIDENCE_SHA256 = "57b725073338f4b3fa54e6f82b6b772d24d1c3b00d85a75c08273dd2be8e03a7"
GUIDES_NPZ = st.STAGING / "external_guides" / "octdl_crop_guides.npz"
GUIDES_JSON = st.STAGING / "external_guides" / "octdl_crop_guides.json"
GUIDES_NPZ_SHA256 = "2dccb3c7ac3cd2b8c9a5b952922f0dc8d8e7ba3fdc50ef096d97c1630dd1f9b0"
GUIDES_JSON_SHA256 = "f9456a3f0fa806bc0891c765b934b903c7649b7d7d2d524a14bcedaafc19cc2b"
GUIDED_DATASET = "src/datasets/oct_slices_guided.py"

ARMS = [("random", "RANDOM", "no guide"),
        ("centroid", "CENTROID", "intensity band"),
        ("envelope", "ENVELOPE", "MIRAGE envelope"),
        ("anatomy", "ANATOMY-V2", "soft MIRAGE scores"),
        ("cover", "COVER", "soft MIRAGE scores")]
GUIDE_KIND = {"random": None, "centroid": "band", "envelope": "hard", "anatomy": "soft",
              "cover": "soft"}
# Target rule printed under row (b): (target shape, placement) cells of the paper's Table
# tab:strategies, checked against the manuscript by table_rules().
TABLE_ROW = {"random": r"\textsc{random}", "centroid": r"\ArmBest{}", "envelope": r"\textsc{envelope}",
             "anatomy": r"\textsc{anatomy-v1/v2}", "cover": r"\textsc{cover}"}
RULES = {"random": ("rectangles", "uniform"),
         "centroid": ("rectangles", "weighted by band overlap"),
         "envelope": ("rectangles", "rejection-sampled on envelope"),
         "anatomy": ("connected components", "grown on scores"),
         "cover": ("rectangles", "greedy coverage")}
PAPER_TEX = ROOT / "paper" / "genai4health2026" / "main_submission.tex"


def table_rules():
    """Shape and placement cells of tab:strategies; every RULES entry must equal them."""
    import re
    text = PAPER_TEX.read_text(encoding="utf-8")
    table = re.search(r"\\label\{tab:strategies\}(.*?)\\end\{tabular\}", text, re.S).group(1)
    rows = {}
    for name, head in TABLE_ROW.items():
        line = [l for l in table.splitlines() if l.strip().startswith(head + " &")]
        assert len(line) == 1, name
        cells = [c.strip() for c in line[0].rstrip().rstrip("\\\\").split("&")]
        rows[name] = {"row": line[0].strip(), "target_shape": cells[2], "placement": cells[3]}
        assert (cells[2], cells[3]) == RULES[name], (name, cells)
    return rows


def table_source():
    """Declared manuscript input of table_rules(): the tab:strategies table environment, pinned by
    the SHA-256 of its LF-normalised text (stable across checkouts and unrelated paper edits)."""
    import re
    raw = PAPER_TEX.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    blocks = re.findall(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?\\label\{tab:strategies\}.*?\\end\{table\}",
                        text, re.S)
    assert len(blocks) == 1, len(blocks)
    start = text[:text.index(blocks[0])].count("\n") + 1
    return {"file": str(PAPER_TEX.relative_to(ROOT)).replace("\\", "/"),
            "table_label": "tab:strategies",
            "table_block_lines": [start, start + blocks[0].count("\n")],
            "table_block_sha256_lf": hashlib.sha256(blocks[0].encode("utf-8")).hexdigest(),
            "file_sha256_at_export": hashlib.sha256(raw).hexdigest()}


def guide_grids(image, staged):
    """Hard and soft guide grids through the production dataset code path."""
    import torch
    from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
    from src.guides.mirage_envelope import DEFAULT_REPAIR

    class ExternalGuideSlice(GuidedOCTSliceDataset):
        """Production guide-grid code path for one external, uncropped image
        (identical to generate_all_strategies.py, which produced the staged replay)."""

        def __init__(self, img, envelope=None, envelope_valid=False, soft=None):
            self._image, self._envelope, self._valid, self._soft = img, envelope, envelope_valid, soft
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

    assert st.sha(GUIDES_NPZ) == GUIDES_NPZ_SHA256 and st.sha(GUIDES_JSON) == GUIDES_JSON_SHA256
    with np.load(GUIDES_NPZ) as z:
        envelope, envelope_valid, soft = z["envelope"], bool(z["envelope_valid"]), z["soft_scores"]
    tensor = torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    hard_item = ExternalGuideSlice(image, envelope=envelope, envelope_valid=envelope_valid)[0]
    soft_item = ExternalGuideSlice(image, soft=soft)[0]
    for item in (hard_item, soft_item):
        assert torch.equal(item[0], tensor), "production image path changed the pixels"
    grids = {"hard": hard_item[1][None], "soft": soft_item[1][None]}
    valid = {"hard": hard_item[2][None], "soft": soft_item[2][None]}
    digests = {p: hashlib.sha256(g.numpy().tobytes()).hexdigest() for p, g in grids.items()}
    assert digests == staged["guide_grids_sha256"], "guide grids differ from the staged replay"
    return tensor, grids, valid, digests


def replay_masks(tensor, grids, valid, staged):
    """Recompute every arm's masks from the pinned sampler sources (generate_all_strategies loop)."""
    import torch
    from scripts.mask_composition_probe import (BASE_KW, build_audit_arm, install_sampler_sources,
                                                sampler_record)
    commit = staged["samplers"]["commit"]
    samplers = install_sampler_sources(commit)
    sizer = samplers["MaskCollator"](**BASE_KW)
    size_rng = torch.Generator().manual_seed(staged["block_size_seed"])
    sizes = {"pred": [sizer._sample_block_size(BASE_KW["pred_mask_scale"], size_rng) for _ in range(4)],
             "enc": [sizer._sample_block_size(BASE_KW["enc_mask_scale"], size_rng)]}
    assert [list(s) for s in sizes["pred"]] == staged["block_sizes"]["pred"]
    masks, centroid_overlay = {}, None
    for name, _, _ in ARMS:
        arm, spec = build_audit_arm(name)
        random.seed(staged["seed"])
        np.random.seed(staged["seed"])
        torch.manual_seed(staged["seed"])
        if spec["kind"] == "random":
            _, enc, pred = arm([tensor], block_sizes=sizes)
        elif spec["kind"] == "centroid":
            enc, pred = arm.generate(1, imgs_cpu=tensor[None], block_sizes=sizes)
            centroid_overlay = (arm._anatomical_prior_weight_grid_for_image(tensor).numpy() > 0)
        else:
            product = spec["guide"]
            enc, pred = arm.generate(1, guide_grids=grids[product], guide_valid=valid[product],
                                     block_sizes=sizes)
        masks[name] = {"targets": [p[0].tolist() for p in pred], "context": [e[0].tolist() for e in enc]}
    record = sampler_record()
    return masks, centroid_overlay, {"commit": record["commit"], "requested": record["requested"],
                                     "files": record["files"]}


def describe(masks):
    """Exact plotted cell sets per arm (derived from the indices, recorded in the evidence)."""
    out = {}
    for name, m in masks.items():
        groups = m["targets"]
        union = set().union(*map(set, groups))
        context = set(m["context"][0])
        assert not union & context and all(0 <= i < 256 for i in union | context)
        repeats = {}
        for k, group in enumerate(groups):
            counts = {}
            for i in group:
                counts[i] = counts.get(i, 0) + 1
            rep = {str(i): c for i, c in sorted(counts.items()) if c > 1}
            if rep:
                repeats["T%d" % (k + 1)] = rep
        out[name] = {"target_index_lists": [len(g) for g in groups],
                     "unique_cells_per_group": [len(set(g)) for g in groups],
                     "target_union_cells": len(union), "context_cells": len(context),
                     "hidden_not_target_cells": 256 - len(union) - len(context),
                     "repeated_indices": repeats}
    return out


def render(image, masks, overlays, soft_score, out_dir, stem, poster=False):
    width = 12.0 if poster else 5.5
    fs = 2.25 if poster else 1.0          # font multiplier (paper base 7 pt)
    u = 2.0 if poster else 1.0            # layout multiplier for fixed gaps
    title_pt, sub_pt, label_pt, legend_pt = 8.0 * fs, 6.9 * fs, 7.6 * fs, 6.8 * fs
    num_pt, rule_pt = 7.0 * fs, 6.7 * fs
    lw = 0.8 * u
    left, right, gap_x, gap_y = 0.40 * fs, 0.03, 0.05 * u, 0.045 * u
    size = (width - left - right - 4 * gap_x) / 5
    header = 0.30 * u * (1.05 if poster else 1.0)
    legend_h = 0.52 * u * (1.05 if poster else 1.0)
    rule_line = rule_pt * 1.18 / 72.0
    rule_h = 3 * rule_line + 0.05 * u
    height = header + 2 * size + gap_y + rule_h + legend_h
    with plt.rc_context(st.rc(fs)):
        fig = plt.figure(figsize=(width, height))
        fx = lambda inch: inch / width  # noqa: E731
        fy = lambda inch: inch / height  # noqa: E731
        rows = [height - header - size, height - header - 2 * size - gap_y]
        numerals, wrapped = {}, {}
        for j, (name, label, guide_name) in enumerate(ARMS):
            x = left + j * (size + gap_x)
            fig.text(fx(x + size / 2), fy(height - 0.03 * u), label, ha="center", va="top",
                     fontsize=title_pt, fontweight="bold", color=st.TEXT)
            fig.text(fx(x + size / 2), fy(height - header + 0.035 * u), guide_name, ha="center",
                     va="bottom", fontsize=sub_pt, color=st.MUTED)
            # (a) guide A_s(x)
            ax = fig.add_axes([fx(x), fy(rows[0]), fx(size), fy(size)])
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none", extent=(0, 16, 16, 0))
            kind = GUIDE_KIND[name]
            if kind in ("band", "hard"):
                cells = overlays[name]
                for y, xx in zip(*np.nonzero(cells)):
                    ax.add_patch(Rectangle((xx, y), 1, 1, facecolor=st.rgba(st.GUIDE_COLOR, .42),
                                           edgecolor="none", zorder=3))
                st.draw_outline(ax, cells, st.GUIDE_COLOR, lw * 1.2, zorder=6, halo="black",
                                halo_lw=lw * 2.6)
            elif kind == "soft":
                for y, xx in zip(*np.nonzero(soft_score > 0)):
                    a = float(min(soft_score[y, xx], 1.0)) * 0.8
                    ax.add_patch(Rectangle((xx, y), 1, 1, facecolor=st.rgba(st.GUIDE_COLOR, a),
                                           edgecolor="none", zorder=3))
            else:
                ax.text(8, 15.3, "uniform placement", ha="center", va="bottom", fontsize=sub_pt,
                        color="white", zorder=7,
                        path_effects=[patheffects.withStroke(linewidth=lw * 2.2, foreground="black")])
            # (b) I-JEPA view: context, targets, hidden-not-target
            ax2 = fig.add_axes([fx(x), fy(rows[1]), fx(size), fy(size)])
            rgb, ctx, union = st.ijepa_view(image, masks[name]["targets"], masks[name]["context"][0])
            ax2.imshow(rgb, interpolation="none", extent=(0, 16, 16, 0))
            for k, group in enumerate(masks[name]["targets"]):
                st.draw_outline(ax2, st.cells_mask(group), st.TARGET_COLORS[k], lw * 1.25, zorder=6 + k,
                                halo="white", halo_lw=lw * 2.5)
                counts = {}
                for i in group:
                    counts[i] = counts.get(i, 0) + 1
                for i, c in counts.items():
                    if c > 1:
                        yy, xx = divmod(int(i), 16)
                        ax2.add_patch(Circle((xx + .5, yy + .5), .27, facecolor="white",
                                             edgecolor="black", linewidth=lw * .7, zorder=12))
            placed, leaders, how = st.place_numerals(masks[name]["targets"])
            st.draw_numerals(ax2, placed, leaders, num_pt, halo=2.0 * u, lw=lw)
            numerals[name] = {"T%d" % (k + 1): {"centre_cells": [round(placed[k][0], 3), round(placed[k][1], 3)],
                                                  "rule": how[k],
                                                  "leader_to": None if k not in leaders else list(leaders[k])}
                              for k in sorted(placed)}
            # target rule under row (b), wrapped to the column width
            words = (RULES[name][0] + ", " + RULES[name][1]).split(" ")
            lines, current = [], ""
            for word in words:
                trial = (current + " " + word).strip()
                probe = fig.text(0, 0, trial, fontsize=rule_pt)
                fig.canvas.draw()
                fits = probe.get_window_extent().width / fig.dpi <= size * 0.94
                probe.remove()
                if fits or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
            assert len(lines) <= 3, (name, lines)
            wrapped[name] = lines
            for n, line in enumerate(lines):
                fig.text(fx(x + size / 2), fy(rows[1] - 0.03 * u - (n + 0.5) * rule_line), line,
                         ha="center", va="center", fontsize=rule_pt, color=st.TEXT)
            for a in (ax, ax2):
                a.set_xlim(0, 16)
                a.set_ylim(16, 0)
                a.set_xticks([])
                a.set_yticks([])
                for spine in a.spines.values():
                    spine.set_linewidth(lw * .6)
                    spine.set_color("#333333")
        # Row labels (two lines, rotated)
        for n, word in enumerate(("target", "rule")):
            fig.text(fx(left - 0.03 * u), fy(rows[1] - 0.03 * u - (n + 0.5) * rule_line), word,
                     ha="right", va="center", fontsize=rule_pt, fontweight="bold", color=st.TEXT)
        for r, (head, sub) in zip(rows, (("(a) guide", "$A_s(x)$"), ("(b) I-JEPA view", "$C$, targets 1–4"))):
            fig.text(fx(left * 0.30), fy(r + size / 2), head, rotation=90, ha="center", va="center",
                     fontsize=label_pt, fontweight="bold", color=st.TEXT)
            fig.text(fx(left * 0.68), fy(r + size / 2), sub, rotation=90, ha="center", va="center",
                     fontsize=label_pt, color=st.TEXT)
        # Legend (drawn, so swatches match the panels exactly)
        lax = fig.add_axes([0, 0, 1, fy(legend_h)])
        lax.set_xlim(0, width)
        lax.set_ylim(0, legend_h)
        lax.axis("off")
        sw = 0.115 * u
        y1, y2, ya = legend_h - 0.105 * u, legend_h - 0.275 * u, 0.07 * u
        pad, sep = 0.045 * u, 0.20 * u

        def text_at(xpos, ypos, s, **kw):
            t = lax.text(xpos, ypos, s, ha="left", va="center", fontsize=legend_pt, color=st.TEXT, **kw)
            fig.canvas.draw()
            return t.get_window_extent().transformed(lax.transData.inverted()).x1

        cursor = left
        for k in range(4):
            st.draw_token(lax, cursor + sw / 2 + k * sw * 1.08, y1, k, sw, num_pt * 0.94, lw * .6, z=3)
        cursor = text_at(cursor + 4 * sw * 1.08 + pad, y1, "targets T1–T4 (predicted)") + sep
        crop = image[96:160, 96:160]
        lax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="none",
                   extent=(cursor, cursor + sw, y1 - sw / 2, y1 + sw / 2), zorder=2)
        lax.add_patch(Rectangle((cursor, y1 - sw / 2), sw, sw, facecolor="none", edgecolor="#333333",
                                lw=lw * .6, zorder=3))
        cursor = text_at(cursor + sw + pad, y1, "visible context $C$") + sep
        lax.add_patch(Rectangle((cursor, y1 - sw / 2), sw, sw, facecolor=st.HIDDEN_GREY, edgecolor="#333333",
                                lw=lw * .6))
        text_at(cursor + sw + pad, y1, "hidden, not a target")
        cursor = left
        lax.add_patch(Rectangle((cursor, y2 - sw / 2), sw, sw, facecolor=st.rgba(st.GUIDE_COLOR, .42),
                                edgecolor="black", lw=lw * 1.6, zorder=2))
        lax.add_patch(Rectangle((cursor, y2 - sw / 2), sw, sw, facecolor="none",
                                edgecolor=st.GUIDE_COLOR, lw=lw * .8, zorder=3))
        cursor = text_at(cursor + sw + pad, y2, "hard guide cells") + sep
        for q in range(4):
            lax.add_patch(Rectangle((cursor + q * sw * .6, y2 - sw / 2), sw * .6, sw,
                                    facecolor=st.rgba(st.GUIDE_COLOR, .2 * (q + 1)), edgecolor="none"))
        cursor = text_at(cursor + 4 * sw * .6 + pad, y2, "soft score (opacity)") + sep
        lax.add_patch(Circle((cursor + sw / 2, y2), sw * .3, facecolor="white", edgecolor="black", lw=lw * .7))
        text_at(cursor + sw + pad, y2, "repeated index")
        attribution = lax.text(width / 2, ya, "OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0), cropped; not "
                               "FairVision. One fixed-seed sampler replay.", ha="center", va="center",
                 fontsize=legend_pt * 0.96, color=st.MUTED)
        fig.canvas.draw()
        bb = attribution.get_window_extent()
        assert bb.x0 >= 0.05 * fig.dpi and bb.x1 <= (width - 0.05) * fig.dpi, "attribution touches the edge"
        # Every drawn text (titles, subtitles, rotated row labels, numerals, legend digits and
        # labels, attribution, target rules): no collision and fully inside the canvas.
        guard = st.assert_layout(fig)
        outputs = st.save_all(fig, out_dir, stem, "Five target-selection strategies in I-JEPA view",
                              dpi=300 if poster else 400)
        plt.close(fig)
    return outputs, [width, round(height, 4)], {"numerals": numerals, "rules": wrapped, "layout_guard": guard}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--replay", action="store_true",
                        help="recompute every mask from the pinned sampler sources and require equality")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    assert st.sha(STAGED_EVIDENCE) == STAGED_EVIDENCE_SHA256, "staged replay evidence changed"
    staged = json.loads(STAGED_EVIDENCE.read_text(encoding="utf-8"))
    assert staged["status"] == "real_external_OCT_CC_BY_4_0_all_strategy_illustration"
    assert staged["source_image_sha256"] == st.SOURCE_SHA256
    native, image = st.load_crop()
    tensor, grids, valid, grid_digests = guide_grids(image, staged)
    masks = staged["masks"]
    overlays = {name: np.asarray(staged["guide_overlays"][name], bool) for name, _, _ in ARMS}
    assert np.array_equal(overlays["envelope"], grids["hard"][0, 1].numpy() > 0)
    for name in ("anatomy", "cover"):
        assert np.array_equal(overlays[name], grids["soft"][0, 1].numpy() > 0)
    soft_score = (grids["soft"][0, 2] + grids["soft"][0, 3]).numpy().astype(np.float64)
    replay = None
    if args.replay:
        replayed, centroid_overlay, record = replay_masks(tensor, grids, valid, staged)
        replay = {"sampler_commit": record["commit"], "sampler_files": record["files"],
                  "masks_equal_staged": replayed == masks,
                  "centroid_guide_equal_staged": bool(np.array_equal(centroid_overlay, overlays["centroid"]))}
        assert replay["masks_equal_staged"] and replay["centroid_guide_equal_staged"], replay
    plotted = describe(masks)
    rules_table = table_rules()
    rules_source = table_source()
    selftest = st.guard_selftest()
    with st.math_scripts():
        paper_outputs, paper_inches, layout = render(image, masks, overlays, soft_score, out, STEM)
        poster_outputs, poster_inches, poster_layout = render(image, masks, overlays, soft_score, out / "poster",
                                                              POSTER_STEM, poster=True)
    assert poster_layout["numerals"] == layout["numerals"]
    sizes = st.pdf_text_sizes(out / (STEM + ".pdf"))
    assert min(s for s, _ in sizes) >= 6.0, sorted(set(s for s, _ in sizes))
    positive = soft_score[soft_score > 0]
    v1_omitted = soft_score[(soft_score > 0) & (soft_score <= 0.02)]
    caption = (
        "The five target-selection strategies on one external OCT example, not FairVision data: an "
        "annotation-free B-scan crop from Kulyabin et al., OCTDL, Fig. 1 (arXiv:2312.08255v3), adapted under "
        "CC BY 4.0 and downsampled to 256 by 256 pixels without contrast enhancement. (a) Placement guide "
        "A_s(x); MIRAGE was not trained on OCTDL. (b) One I-JEPA view: visible context C, target groups T_m "
        "(colour and number mark identity, not relevance), hidden non-targets (grey) and repeated indices "
        "(dots); the target rule follows Table tab:strategies. Exact indices from each trained sampler for "
        "this image (matched block sizes, fixed seed); a single illustrative replay, not a segmentation, "
        "training batch, evaluation sample or performance comparison. License: CC BY 4.0 URL.")
    evidence = {
        "status": "real_external_OCT_CC_BY_4_0_strategy_uniqueness_illustration",
        "generator": Path(__file__).name, "generator_sha256": st.sha(Path(__file__)),
        "style_module": "figstyle_v2.py", "style_module_sha256": st.sha(HERE / "figstyle_v2.py"),
        "source_image_file": str(st.SOURCE.relative_to(ROOT)), "source_image_sha256": st.sha(st.SOURCE),
        "primary_license_file": str(st.LICENSE.relative_to(ROOT)), "primary_license_sha256": st.sha(st.LICENSE),
        "license": "CC BY 4.0", "license_source": "https://arxiv.org/abs/2312.08255v3",
        "attribution": st.ATTRIBUTION, "crop_xyxy": list(st.CROP),
        "input_tensor_sha256": st.INPUT_TENSOR_SHA256,
        "mask_source": {"file": str(STAGED_EVIDENCE.relative_to(ROOT)), "sha256": STAGED_EVIDENCE_SHA256,
                        "producer": staged["generator"], "producer_sha256": staged["generator_sha256"],
                        "sampler_commit": staged["samplers"]["commit"], "seed": staged["seed"],
                        "block_size_seed": staged["block_size_seed"], "block_sizes": staged["block_sizes"],
                        "batch_size": staged["batch_size"],
                        "configs_sha256": staged["configs_sha256"]},
        "guide_products": {"npz": {"file": str(GUIDES_NPZ.relative_to(ROOT)), "sha256": GUIDES_NPZ_SHA256},
                           "record": {"file": str(GUIDES_JSON.relative_to(ROOT)), "sha256": GUIDES_JSON_SHA256},
                           "dataset_code": {"file": GUIDED_DATASET, "sha256": st.sha(ROOT / GUIDED_DATASET)},
                           "grid_sha256_equal_staged": grid_digests == staged["guide_grids_sha256"],
                           "grid_sha256": grid_digests},
        "replay": replay,
        "plotted": {
            "row_a": {"random": "none (plain crop)",
                      "centroid": "staged guide_overlays.centroid (intensity band cells > 0)",
                      "envelope": "hard grid channel 1 (placement cells) == staged guide_overlays.envelope",
                      "anatomy": "soft grid channels 2 + 3 (P_inner + P_choroid per patch), opacity 0.8 * min(score, 1) "
                                 "for every cell with score > 0 (no display cutoff)",
                      "cover": "same soft score as anatomy"},
            "row_b": "context = staged masks[arm].context[0]; T_m = staged masks[arm].targets[m-1] "
                     "(tint alpha 0.55 + outline, group order 1..4, bold numeral m with white halo placed by "
                     "figstyle_v2.place_numerals: centre of the first 2x2 square of the group's own cells in "
                     "raster order, i.e. its top-left corner, skipping squares closer than 2.3 cells to another "
                     "numeral or on a repeat dot; groups without a 2x2 square get the nearest target-free 2x2 "
                     "square with a leader line); grey = neither; "
                     "dot = repeated index",
            "numerals": layout["numerals"],
            "target_rules": {"source": "paper Table tab:strategies (target shape, placement); checked by "
                                       "table_rules() against main_submission.tex at export",
                             "manuscript_input": rules_source,
                             "rows": rules_table, "printed_lines": layout["rules"]},
            "display_transformations": {
                "image": "OCTDL crop, PIL L, bilinear 1280 -> 256, no contrast change",
                "hard_guides": "cells > 0 filled yellow alpha 0.42 with outline",
                "soft_guides": "opacity 0.8 * min(score, 1) for all %d cells with score > 0; no cutoff" % positive.size,
                "soft_guides_v1_note": ("the first I13 export omitted the %d cells with 0 < score <= 0.02 "
                                        "(summed score %.7f of %.7f, %.2f%%); removed after V12, display only, "
                                        "sampler input unaffected" % (v1_omitted.size, float(v1_omitted.sum()),
                                                                      float(positive.sum()),
                                                                      100 * float(v1_omitted.sum() / positive.sum()))),
                "targets": "image tinted with the group colour (alpha 0.55), composited in group order"},
            "collation": ("Batch size 1: no cross-image shortening. Target-group collation still applies: the "
                          "rectangle samplers truncate every group to the shortest group (block sizes 7x6, 8x6, "
                          "8x5, 7x6 cells -> 40 indices each) and ANATOMY-V2 resamples every group to K = 16 "
                          "indices; the context excludes the full blocks, so 4-10 cells are neither context nor "
                          "target. Training-batch collation can shorten the context further (Table 2)."),
            "cells": plotted,
            "soft_score_sha256": hashlib.sha256(soft_score.tobytes()).hexdigest(),
            "masks": masks,
            "guide_overlays": {k: v.astype(int).tolist() for k, v in overlays.items()}},
        "palette": {"targets": st.TARGET_COLORS, "guide": st.GUIDE_COLOR, "hidden": st.HIDDEN_GREY,
                    "source": "Okabe-Ito colour-blind-safe set"},
        "clinical_pixels": "Published external OCTDL crop only; no FairVision clinical pixels",
        "identifiers_or_patient_metadata_exported": False, "numerical_readouts": False,
        "limits": ("Single fixed-seed replay at batch size 1 on an external image; MIRAGE was not trained on "
                   "OCTDL, so MIRAGE guides are not a validated segmentation; no cohort statistic, "
                   "historical batch or performance claim. At batch size 1 there is no cross-image "
                   "shortening, but target-group collation still applies (see plotted.collation); "
                   "training-batch collation can shorten the delivered context further (Table 2 of the paper)."),
        "typography": {"pdf_text_min_pt": min(s for s, _ in sizes),
                       "pdf_text_sizes_pt": sorted(set(s for s, _ in sizes)),
                       "math_script_scale": st.MATH_SCRIPT_SCALE,
                       "note": ("sizes at the 5.5 in print width; includes mathtext sub/superscripts, drawn at "
                                "%.1f x their base size (matplotlib default 0.7) so that they print at >= 6 pt"
                                % st.MATH_SCRIPT_SCALE)},
        "layout_guard": {"paper": layout["layout_guard"], "poster": poster_layout["layout_guard"],
                         "scope": ("figstyle_v2.layout_guard: every visible text artist of the figure "
                                   "(figure and axes texts, numerals, legend digits and labels, titles, "
                                   "rotated row labels, attribution, target rules) pairwise without collision "
                                   "(2 pt horizontal / 0.5 pt vertical clearance) and fully inside the canvas "
                                   "(0.5 pt margin); asserted before export"),
                         "selftest": selftest},
        "caption_summary": caption,
        "alt_text": ("Ten panels in two rows over five columns (RANDOM, CENTROID, ENVELOPE, ANATOMY-V2, COVER) "
                     "show the same external OCT crop. Top row: each strategy's placement guide in yellow "
                     "(none for RANDOM). Bottom row: visible context as image, four target groups tinted "
                     "orange, sky blue, green and purple with outlines and numbered badges 1 to 4, grey cells hidden "
                     "but not predicted, white dots on repeated target indices."),
        "outputs": {"paper": {"inches": paper_inches, **paper_outputs},
                    "poster": {"inches": poster_inches, "dir": "poster", **poster_outputs}},
        "out_dir": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
    }
    (out / f"{STEM}.evidence.json").write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"cells": plotted, "replay": None if replay is None else
                      {k: v for k, v in replay.items() if k != "sampler_files"},
                      "outputs": evidence["outputs"]}, indent=1))


if __name__ == "__main__":
    main()
