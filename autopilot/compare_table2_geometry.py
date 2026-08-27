"""Compare Table 2 (tab:geom) printed values against a regenerated measurement.

Primary artifact: scripts/mask_composition_probe.py, 600 slices, batch_size=1
(per-image geometry, i.e. before the collator's batch-minimum truncation),
COVER floor 0.21 (the trained arm's production value).

Only two hand-typed inputs exist below: PRINTED, transcribed from
paper/genai4health2026/main_submission.tex lines 473-477, and AUC_EP50,
transcribed from paper/genai4health2026/auto/auto_numbers.tex.  Everything else
is read from the artifacts.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "results" / "masking" / "table2_geometry"

PRIMARY = ART / "mask_geometry_600slices_bs1_coverf021_seed42.json"
REPLICATES = [
    ART / "mask_geometry_600slices_bs1_coverf021_seed42.json",
    ART / "mask_geometry_600slices_bs1_coverf021_seed1234.json",
    ART / "mask_geometry_600slices_bs1_coverf021_seed2026.json",
]
DELIVERED = ART / "mask_geometry_600slices_bs64_coverf021_seed42.json"

ARMS = [
    ("random", "random"),
    ("centroid", "oracle"),
    ("envelope", "envelope"),
    ("cover", "cover"),
    ("anatomy-v2", "anatomy"),
]

METRICS = [
    ("anatomy hidden", "hidden_share_of_all_anat", 1.0),
    ("purity", "hidden_pct_on_anat", 1.0),
    ("mask ratio", "hidden_frac_of_grid", 100.0),
    ("context kept", "ctx_frac_of_grid", 100.0),
    ("loss slots", "n_slots_mean", 1.0),
]

PRINTED = {
    "random":     {"anatomy hidden": 52.2, "purity": 31.6, "mask ratio": 43.7,
                   "context kept": 42.1, "loss slots": 157.7},
    "centroid":   {"anatomy hidden": 62.2, "purity": 41.1, "mask ratio": 40.0,
                   "context kept": 45.6, "loss slots": 158.4},
    "envelope":   {"anatomy hidden": 76.9, "purity": 43.5, "mask ratio": 46.4,
                   "context kept": 40.5, "loss slots": 159.9},
    "cover":      {"anatomy hidden": 74.1, "purity": 45.3, "mask ratio": 43.3,
                   "context kept": 43.5, "loss slots": 160.0},
    "anatomy-v2": {"anatomy hidden": 80.3, "purity": 97.3, "mask ratio": 21.4,
                   "context kept": 67.9, "loss slots": 64.0},
}

AUC_EP50 = {"random": 0.8641, "centroid": 0.8740, "envelope": 0.8761,
            "cover": 0.8643, "anatomy-v2": 0.8654}

RECTANGLES = ["random", "centroid", "cover", "envelope"]
ROUND_TOL = 0.05 + 1e-9


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    return ({label: {m: d[key][k] * s for m, k, s in METRICS}
             for label, key in ARMS}, d["_meta"])


def order(vals):
    return [a for a, _ in sorted(vals.items(), key=lambda kv: kv[1])]


def main():
    primary, meta = load(PRIMARY)
    reps = [load(p)[0] for p in REPLICATES]
    delivered, meta_d = load(DELIVERED)

    spread = {a: {m: max(r[a][m] for r in reps) - min(r[a][m] for r in reps)
                  for m, _, _ in METRICS} for a, _ in ARMS}

    rank_held = {}
    for m, _, _ in METRICS:
        po = order({a: PRINTED[a][m] for a, _ in ARMS})
        ro = order({a: primary[a][m] for a, _ in ARMS})
        rank_held[m] = {a: po.index(a) == ro.index(a) for a, _ in ARMS}

    out = []
    out.append("| arm | metric | printed | regenerated | abs diff | "
               "seed range (n=3) | verdict |")
    out.append("|---|---|---|---|---|---|---|")
    counts = {"MATCHES": 0, "CLOSE": 0, "DIFFERS": 0}
    for a, _ in ARMS:
        for m, _, _ in METRICS:
            p, r = PRINTED[a][m], primary[a][m]
            d = abs(r - p)
            if d <= ROUND_TOL:
                v = "MATCHES"
            elif rank_held[m][a]:
                v = "CLOSE"
            else:
                v = "DIFFERS"
            counts[v] += 1
            out.append(f"| {a} | {m} | {p:.1f} | {r:.2f} | {d:.2f} | "
                       f"{spread[a][m]:.2f} | {v} |")
    out += ["", f"verdict counts: {counts}", ""]

    inside = sum(1 for a, _ in ARMS for m, _, _ in METRICS
                 if abs(primary[a][m] - PRINTED[a][m])
                 <= max(spread[a][m], ROUND_TOL))
    worst = max((abs(primary[a][m] - PRINTED[a][m]), a, m)
                for a, _ in ARMS for m, _, _ in METRICS)
    out.append(f"cells within the 3-seed range of the measurement: {inside}/25")
    out.append(f"largest absolute difference over all 25 cells: "
               f"{worst[0]:.2f} ({worst[1]} / {worst[2]})")
    out.append("")

    out.append("### Delivered-to-encoder variant (batch_size=64, same slices)")
    out.append("| arm | metric | printed | delivered (bs=64) | abs diff |")
    out.append("|---|---|---|---|---|")
    for a, _ in ARMS:
        for m, _, _ in METRICS:
            p, r = PRINTED[a][m], delivered[a][m]
            out.append(f"| {a} | {m} | {p:.1f} | {r:.2f} | {abs(r - p):.2f} |")
    out.append("")

    out.append("### Rank order")
    for m in ("anatomy hidden", "purity"):
        po = order({a: PRINTED[a][m] for a, _ in ARMS})
        ro = order({a: primary[a][m] for a, _ in ARMS})
        po4 = [a for a in po if a in RECTANGLES]
        ro4 = [a for a in ro if a in RECTANGLES]
        out.append(f"- {m}, 5 arms, printed (low to high): {po}")
        out.append(f"- {m}, 5 arms, regen   (low to high): {ro}")
        out.append(f"- {m}, 5 arms unchanged: {po == ro}")
        out.append(f"- {m}, 4 rectangles, printed: {po4}")
        out.append(f"- {m}, 4 rectangles, regen  : {ro4}")
        out.append(f"- {m}, 4 rectangles unchanged: {po4 == ro4}")
        for i, rep in enumerate(reps):
            ro_i = [a for a in order({a: rep[a][m] for a, _ in ARMS})
                    if a in RECTANGLES]
            out.append(f"  - replicate {i} 4-rectangle order: {ro_i} "
                       f"(same as printed: {ro_i == po4})")
        out.append("")

    out.append("### Spearman rho against AUC @ ep50")
    out.append("| metric | arm set | source | rho | p |")
    out.append("|---|---|---|---|---|")
    for m in ("anatomy hidden", "purity"):
        for sname, arms in (("4 rectangles", RECTANGLES),
                            ("5 arms", [a for a, _ in ARMS])):
            for src, data in (("printed", PRINTED), ("regenerated", primary)):
                rho, p = spearmanr([data[a][m] for a in arms],
                                   [AUC_EP50[a] for a in arms])
                out.append(f"| {m} | {sname} | {src} | {rho:+.4f} | {p:.4f} |")
            for i, rep in enumerate(reps):
                rho, p = spearmanr([rep[a][m] for a in arms],
                                   [AUC_EP50[a] for a in arms])
                out.append(f"| {m} | {sname} | replicate {i} | "
                           f"{rho:+.4f} | {p:.4f} |")
    out.append("")
    out.append("### Replicate values (n=3 draws of 600 slices)")
    seeds = [json.loads(p.read_text())["_meta"]["seed"] for p in REPLICATES]
    out.append("| arm | metric | " + " | ".join(f"seed {s}" for s in seeds)
               + " | mean | stdev |")
    out.append("|---|---|" + "---|" * (len(REPLICATES) + 2))
    for a, _ in ARMS:
        for m, _, _ in METRICS:
            vals = [rep[a][m] for rep in reps]
            out.append(f"| {a} | {m} | " + " | ".join(f"{v:.2f}" for v in vals)
                       + f" | {statistics.mean(vals):.2f} "
                         f"| {statistics.stdev(vals):.2f} |")
    out.append("")
    out.append(f"primary meta: {json.dumps(meta)}")
    out.append(f"delivered meta: {json.dumps(meta_d)}")

    text = "\n".join(out)
    print(text)
    (ART / "table2_comparison.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
