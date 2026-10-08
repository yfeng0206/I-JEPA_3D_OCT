"""Generate camera-ready numeric macros from retained evidence (auto/cr_numbers.tex).

Every value is computed by the same typed expressions that p15_verify_numbers.py
re-evaluates from SHA-256-pinned sources. With --update-reviews the expressions
and source hashes are written into paper/genai4health2026/numeric_reviews.json
("sources" and "macros"), replacing only the entries this script owns.

Sources:
  * Delivered-mask audit under the trained sampler configuration (I5, E12):
    three audit seeds x 100 batches of 64 views, and a batch-size-1 run.
  * Submitted (v9) geometry measurements: results/masking/table2_geometry.
  * Guide-agreement and envelope characterization (I6): guide_quality/summary.json.
  * Published reference values transcribed with locators: paper_i8/external_reference_values.json.

Usage (repo root):
  python autopilot/make_cr_numbers.py [--update-reviews] [--check]
"""
import argparse
import json
from pathlib import Path
import sys

try:
    from . import numeric_bindings as numeric
    from . import release_assets as assets
except ImportError:
    import numeric_bindings as numeric
    import release_assets as assets

REPO = assets.REPO
PAPER = assets.PAPER
OUT = PAPER / "auto" / "cr_numbers.tex"
REVIEWS = PAPER / "numeric_reviews.json"
CR = "autopilot/investigations/camera_ready_20261008"
SOURCES = {
    "cr_mask_bs64": CR + "/mask_audit/delivered_mask_audit_trained_3seeds_100draws.json",
    "cr_mask_bs1": CR + "/mask_audit/runs/per_image_bs1_trained_3seeds.json",
    "cr_guide": CR + "/guide_quality/summary.json",
    "cr_external": CR + "/paper_i8/external_reference_values.json",
    # Submitted v9 measurements (same files the v9 tables were bound to).
    "lr_geom": "results\\masking\\table2_geometry\\mask_geometry_600slices_bs1_coverf021_seed42.json",
    "lr_geom64": "results\\masking\\table2_geometry\\mask_geometry_600slices_bs64_coverf021_seed42.json",
}
OWNED_SOURCES = ("cr_mask_bs64", "cr_mask_bs1", "cr_guide", "cr_external")
ARMS = {"Random": ("random", "random"), "Centroid": ("centroid", "oracle"),
        "Envelope": ("envelope", "envelope"), "Cover": ("cover", "cover"),
        "AnatomyTwo": ("anatomy", "anatomy")}

ref, fmt, op = numeric.ref, numeric.fmt, numeric.operation


def pct(expression):
    return op("percent", expression)


def macro_specs():
    """(name, expression, thousands) in output order."""
    specs = []

    def add(name, expression, thousands=False):
        specs.append((name, expression, thousands))

    m64, m1, old1, old64 = "cr_mask_bs64", "cr_mask_bs1", "lr_geom", "lr_geom64"
    # Delivered masks at the training batch size (main geometry table).
    for word, (new, old) in ARMS.items():
        summary = ("summary", new)
        add("CRHid" + word, fmt("%.1f", ref(m64, *summary, "anatomy_hidden_soft_pct", "mean")))
        add("CRPur" + word, fmt("%.1f", ref(m64, *summary, "purity_soft_pct", "mean")))
        add("CRUniq" + word, fmt("%.1f", ref(m64, *summary, "unique_targets_U", "mean")))
        add("CRSlots" + word, fmt("%.1f", ref(m64, *summary, "loss_slots_L", "mean")))
        add("CRCtx" + word, fmt("%.1f", ref(m64, *summary, "context_pct_grid", "mean")))
        add("CRCtxSD" + word, fmt("%.1f", ref(m64, *summary, "context_pct_grid", "seed_sd")))
        add("CRCtxOld" + word, fmt("%.1f", pct(ref(old64, old, "ctx_frac_of_grid"))))
        # Per-image geometry before batching (appendix old-versus-new table).
        summary = ("summary", new)
        add("CRImgHid" + word, fmt("%.1f", ref(m1, *summary, "anatomy_hidden_soft_pct", "mean")))
        add("CRImgHidOld" + word, fmt("%.1f", ref(old1, old, "hidden_share_of_all_anat")))
        add("CRImgPur" + word, fmt("%.1f", ref(m1, *summary, "purity_soft_pct", "mean")))
        add("CRImgPurOld" + word, fmt("%.1f", ref(old1, old, "hidden_pct_on_anat")))
        add("CRImgMask" + word, fmt("%.1f", ref(m1, *summary, "mask_ratio_pct", "mean")))
        add("CRImgMaskOld" + word, fmt("%.1f", pct(ref(old1, old, "hidden_frac_of_grid"))))
        add("CRImgCtx" + word, fmt("%.1f", ref(m1, *summary, "context_pct_grid", "mean")))
        add("CRImgCtxOld" + word, fmt("%.1f", pct(ref(old1, old, "ctx_frac_of_grid"))))
        add("CRImgSlots" + word, fmt("%.1f", ref(m1, *summary, "loss_slots_L", "mean")))
        add("CRImgSlotsOld" + word, fmt("%.1f", ref(old1, old, "n_slots_mean")))
    for word, arm in (("Centroid", "centroid"), ("Envelope", "envelope")):
        gap = ("paired_context_gaps_vs_random", arm)
        add("CRCtxGap" + word, fmt("%.1f", ref(m64, *gap, "mean_of_seeds")))
        add("CRCtxGap" + word + "CI", fmt("%.1f--%.1f", ref(m64, *gap, "pooled_normal_95ci", 0),
                                          ref(m64, *gap, "pooled_normal_95ci", 1)))
        add("CRCtxGapSD" + word, fmt("%.1f", ref(m64, *gap, "seed_sd")))
    protocol = ("protocol",)
    add("CRAuditSeedCount", fmt("%d", op("length", ref(m64, *protocol, "audit_seeds"))))
    add("CRAuditDraws", fmt("%d", ref(m64, *protocol, "budget_draws_per_seed")))
    add("CRAuditBatch", fmt("%d", ref(m64, *protocol, "batch_size")))
    add("CRAuditViews", fmt("%d", ref(m64, *protocol, "views")), thousands=True)
    add("CRAuditVolumes", fmt("%d", ref(m64, *protocol, "volumes")))
    add("CRAuditSlices", fmt("%d", ref(m64, *protocol, "slices_per_volume")))
    add("CRImgDraws", fmt("%d", ref(m1, *protocol, "budget_draws_per_seed")), thousands=True)
    add("CRSamplerCommit", ref(m64, "samplers", "commit"))
    add("CROldSlices", fmt("%d", ref(old64, "_meta", "slices")))
    add("CROldVolumes", fmt("%d", ref(old64, "_meta", "volumes")))
    add("CROldSeed", fmt("%d", ref(old64, "_meta", "seed")))
    # CENTROID band geometry of the trained configuration.
    g = "cr_guide"
    ribbon = ("design", "ribbon")
    add("CRBandRows", fmt("%d", ref(g, *ribbon, "band_h")))
    add("CRBandCols", fmt("%d", ref(g, *ribbon, "x_keep")))
    add("CRBandRegion", fmt("%.2f", ref(g, *ribbon, "region_frac")))
    add("CRBandLateral", fmt("%.1f", ref(g, *ribbon, "lateral_frac")))
    # Guide agreement (CENTROID band versus MIRAGE envelope) and envelope flags.
    add("CRGuideVolumes", fmt("%d", ref(g, "design", "n_volumes")))
    add("CRGuideBscans", fmt("%d", ref(g, "design", "n_bscans")), thousands=True)
    add("CRGuidePerStratum", fmt("%d", ref(g, "design", "per_stratum")))
    add("CRCensusVolumes", fmt("%d", ref(g, "census", "n_volumes")), thousands=True)
    f1 = ("rates", "F1")
    add("CRFOne", fmt("%.1f", pct(ref(g, *f1, "overall_popweighted", "rate"))))
    add("CRFOneCI", fmt("%.1f--%.1f", pct(ref(g, *f1, "overall_popweighted", "ci95", 0)),
                        pct(ref(g, *f1, "overall_popweighted", "ci95", 1))))
    add("CRFOneCensus", fmt("%.1f", pct(ref(g, "census", *f1, "overall", "rate"))))
    add("CRFOneNormal", fmt("%.1f", pct(ref(g, *f1, "label=0", "rate"))))
    add("CRFOneGlaucoma", fmt("%.1f", pct(ref(g, *f1, "label=1", "rate"))))
    add("CRFOneLowSNR", fmt("%.1f", pct(ref(g, *f1, "snr_t0", "rate"))))
    add("CRFOneHighSNR", fmt("%.1f", pct(ref(g, *f1, "snr_t2", "rate"))))
    add("CRFTwo", fmt("%.1f", pct(ref(g, "rates", "F2", "overall_popweighted", "rate"))))
    add("CRIoUMedian", fmt("%.2f", ref(g, "continuous", "iou", "all", "median")))
    add("CREnvAnyFlag", fmt("%.1f", pct(ref(g, "rates", "env_any_flag", "overall_popweighted", "rate"))))
    add("CREnvAnyFlagCI", fmt("%.1f--%.1f",
                              pct(ref(g, "rates", "env_any_flag", "overall_popweighted", "ci95", 0)),
                              pct(ref(g, "rates", "env_any_flag", "overall_popweighted", "ci95", 1))))
    add("CREnvFragmented", fmt("%.1f", pct(ref(g, "rates", "fragmented", "overall_popweighted", "rate"))))
    add("CREnvThickness", fmt("%.1f", pct(ref(g, "rates", "thickness_irregular", "overall_popweighted", "rate"))))
    add("CREnvPosition", fmt("%.1f", pct(ref(g, "rates", "position_implausible", "overall_popweighted", "rate"))))
    add("CRMirageDice", fmt("%.3f", ref(g, "mirage_context", "goals_test_evaluation",
                                        "foreground_mean_dice", "value")))
    ext = ("cr_external", "fairvision_glaucoma_3d_resnet")
    add("CRFairVisionAUC", fmt("%.4f", ref(*ext, "overall_auc")))
    add("CRFairVisionSD", fmt("%.4f", ref(*ext, "overall_auc_sd")))
    return specs


def thousands(text):
    """Display grouping only; p15 normalises {,} before comparing."""
    return "{,}".join(reversed([text[::-1][i:i + 3][::-1] for i in range(0, len(text), 3)]))


def build():
    evidence = numeric.Evidence(PAPER, REPO, {name: {"root": "repo", "path": path}
                                              for name, path in SOURCES.items()})
    values, macros = [], {}
    for name, expression, group in macro_specs():
        bound = evidence.binding(expression)
        value = bound["expected"]
        if group:
            value = thousands(value)
        values.append((name, value))
        macros[name] = {"expression": expression}
    hashes = {name: evidence.hashes[name] for name in SOURCES if name in evidence.hashes}
    return values, macros, hashes


def render(values, hashes):
    lines = ["% Generated by autopilot/make_cr_numbers.py from SHA-256-pinned evidence; do not edit values.",
             "% Sources:"]
    lines += ["%%   %s  %s" % (name, hashes[name]) for name in sorted(hashes)]
    lines += [r"\newcommand{\%s}{%s}" % (name, value) for name, value in values]
    return "\n".join(lines) + "\n"


def update_reviews(macros, hashes):
    raw = REVIEWS.read_bytes()
    crlf = b"\r\n" in raw
    data = json.loads(raw.decode("utf-8"))
    for name, path in SOURCES.items():
        if name in OWNED_SOURCES:
            data["sources"][name] = {"root": "repo", "path": path, "sha256": hashes[name]}
        elif data["sources"].get(name, {}).get("sha256") != hashes[name]:
            raise ValueError("shared source hash differs from numeric_reviews.json: " + name)
    data["macros"] = {name: spec for name, spec in data["macros"].items() if not name.startswith("CR")}
    data["macros"].update(macros)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    REVIEWS.write_bytes(text.encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-reviews", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail if auto/cr_numbers.tex is stale")
    args = parser.parse_args(argv)
    values, macros, hashes = build()
    text = render(values, hashes)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        print("RESULT:", "PASS" if current == text else "FAIL (stale cr_numbers.tex)")
        return 0 if current == text else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", OUT, "(%d macros)" % len(values))
    if args.update_reviews:
        update_reviews(macros, hashes)
        print("updated", REVIEWS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
