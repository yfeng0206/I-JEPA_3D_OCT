"""E2 distribution check: replay the trained ENVELOPE run's [MIRAGE] fingerprint.

Feeds real post-crop training guides (build_real_guides.py) through
MirageMaskCollator built exactly as src/train_patch.py builds it from a YAML
(production configs/patch_mirage_envelope.yaml with the sampler key injected
as text), and compares each version's own logged statistics with the
historical train.log per-epoch means (A1 envelope_mirage_stats_by_epoch.json).

Usage: python fingerprint_replay.py GUIDES.npz HIST.json OUT.json
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import src.masks.curriculum as NEW  # noqa: E402

torch.set_num_threads(2)
GUIDES, HIST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
B = 64
SEEDS = [0, 1, 2]
FULL_EPOCH0 = 40          # 0-based; r_t = 1.0 (1-based epoch 41)
RAMP_EPOCHS0 = [27, 28, 29]  # r_t 0.4/0.6/0.8 == logged epochs 28/29/30

LOG_KEYS = [  # train.log [MIRAGE] name -> mirage_stats key
    ("patches/block", "patches_per_block"),
    ("unique_targets", "unique_target_patches"),
    ("context", "context_patches"),
    ("on_region", "target_on_region"),
    ("background", "target_background"),
    ("fallbacks", "fallbacks"),
    ("infeasible", "infeasible"),
    ("unbiased", "unbiased_by_ramp"),
    ("accept", "accept_rate"),
    ("fill", "mean_block_fill"),
    ("retina_visible", "retina_visible"),
    ("tries", "mean_attempts"),
]

# The ENVELOPE-replicate YAML: production config plus ONE inserted line.
with open(C.PROD_YAML, "r", encoding="utf-8") as handle:
    prod_text = handle.read()
ANCHOR = "    mirage_overlap_tolerance: 0.25\n"
assert prod_text.count(ANCHOR) == 1


def variant_yaml(version):
    path = os.path.join(C.HERE, "patch_mirage_envelope__%s.yaml" % version)
    text = prod_text.replace(
        ANCHOR, ANCHOR + "    mirage_overlap_fallback: %s\n" % version
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return path, cfg


ref804_path, ref804_sha = C.extract_ref(C.RUN_ERA_COMMIT)
REF804 = C.load_module(ref804_path, "curriculum_ref_804c639")

d = np.load(GUIDES)
G = torch.from_numpy(d["guides"]).float()
V = torch.from_numpy(d["valid"]).bool()
n = (G.shape[0] // B) * B
batches = [(G[i:i + B], V[i:i + B]) for i in range(0, n, B)]


def replay(module, cfg, seed, epoch0, legacy=False):
    # Mirrors train_patch.py: curr_cfg = mask_cfg.get('curriculum')
    curr_cfg = cfg["mask"].get("curriculum", {}) or {}
    kwargs = C.collator_kwargs(cfg, curr_cfg)
    if legacy:
        kwargs = C.legacy_kwargs(kwargs)
    C.seed_all(seed)
    collator = module.MirageMaskCollator(**kwargs)
    if hasattr(collator, "mirage_overlap_fallback"):
        assert collator.mirage_overlap_fallback == curr_cfg.get(
            "mirage_overlap_fallback", "least_overlap_v2")
    collator.set_epoch(epoch0, cfg["optimization"]["epochs"])
    rows = []
    for guides, valid in batches:
        _, enc, pred, ms = collator(C.make_batch(guides, valid))
        row = {name: float(ms[key]) for name, key in LOG_KEYS}
        row["delivered_context_K"] = int(enc[0].shape[1])
        row["delivered_target_K"] = int(pred[0].shape[1])
        rows.append(row)
    gen = collator._generator
    assert gen.mirage_overlap_fallback == curr_cfg.get(
        "mirage_overlap_fallback", "least_overlap_v2") if module is NEW else True
    return rows, float(gen._r_t)


def summarize(rows):
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "sd_batch": float(vals.std(ddof=1)),
                  "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))}
    return out


with open(HIST, "r", encoding="utf-8") as handle:
    hist_doc = json.load(handle)
hist = hist_doc.get("per_epoch", hist_doc)
hist_pooled = hist_doc.get("pooled_ep31_100", {})
full_epochs = [e for e in sorted(hist, key=int) if 31 <= int(e) <= 100]
hist_summary = {}
for name, _ in LOG_KEYS:
    vals = np.array([hist[e][name] for e in full_epochs], dtype=np.float64)
    hist_summary[name] = {"mean": float(vals.mean()), "min": float(vals.min()),
                          "max": float(vals.max()), "sd_epoch": float(vals.std(ddof=1)),
                          "n_epochs": int(vals.size)}

t0 = time.time()
versions = {}
for version in ("legacy_uniform_v1", "least_overlap_v2"):
    path, cfg = variant_yaml(version)
    assert cfg["mask"]["curriculum"]["mirage_overlap_fallback"] == version
    versions[version] = {"yaml": os.path.relpath(path, C.REPO), "cfg": cfg}

results = {}
for label, module, version, legacy in [
    ("legacy_uniform_v1", NEW, "legacy_uniform_v1", False),
    ("least_overlap_v2", NEW, "least_overlap_v2", False),
    ("ref_804c639", REF804, "legacy_uniform_v1", True),
]:
    cfg = versions[version]["cfg"]
    rows_all, per_seed = [], {}
    for seed in SEEDS:
        rows, r_t = replay(module, cfg, seed, FULL_EPOCH0, legacy=legacy)
        assert r_t == 1.0
        rows_all += rows
        per_seed[seed] = {k: v["mean"] for k, v in summarize(rows).items()}
        print("%-18s seed %d  unique_targets %.2f  context %.2f  accept %.3f  (%.0fs)"
              % (label, seed, per_seed[seed]["unique_targets"],
                 per_seed[seed]["context"], per_seed[seed]["accept"],
                 time.time() - t0), flush=True)
    results[label] = {"r_t": 1.0, "epoch0": FULL_EPOCH0, "batches": len(rows_all),
                      "summary": summarize(rows_all), "per_seed_means": per_seed}

ramp = {}
for version in ("legacy_uniform_v1", "least_overlap_v2"):
    cfg = versions[version]["cfg"]
    ramp[version] = {}
    for epoch0 in RAMP_EPOCHS0:
        rows, r_t = replay(NEW, cfg, 0, epoch0)
        ramp[version][str(epoch0 + 1)] = {
            "r_t": r_t, **{k: v["mean"] for k, v in summarize(rows).items()}}
    print("ramp", version, {e: round(v["unique_targets"], 2) for e, v in ramp[version].items()})

comparison = {}
for name, _ in LOG_KEYS:
    h = hist_summary[name]
    row = {"historical_ep31_100": dict(h, sd_batch=hist_pooled.get(name, {}).get("sd_batch"))}
    for label in ("legacy_uniform_v1", "least_overlap_v2", "ref_804c639"):
        m = results[label]["summary"][name]["mean"]
        row[label] = {
            "mean": m,
            "sd_batch": results[label]["summary"][name]["sd_batch"],
            "inside_hist_range": bool(h["min"] <= m <= h["max"]),
            "delta_vs_hist_mean": m - h["mean"],
            "z_vs_hist_epoch_sd": (m - h["mean"]) / h["sd_epoch"] if h["sd_epoch"] > 0 else None,
        }
    comparison[name] = row
placement_keys = ["unique_targets", "context", "on_region", "accept", "fill",
                  "retina_visible", "tries"]
verdict = {
    "legacy_uniform_v1_inside_all_placement_ranges": all(
        comparison[k]["legacy_uniform_v1"]["inside_hist_range"] for k in placement_keys),
    "least_overlap_v2_outside_ranges": [
        k for k in placement_keys
        if not comparison[k]["least_overlap_v2"]["inside_hist_range"]],
    "legacy_uniform_v1_outside_ranges": [
        k for k, _ in LOG_KEYS if not comparison[k]["legacy_uniform_v1"]["inside_hist_range"]],
    "ref_804c639_stats_identical_to_legacy_uniform_v1": all(
        results["ref_804c639"]["summary"][k]["mean"]
        == results["legacy_uniform_v1"]["summary"][k]["mean"] for k, _ in LOG_KEYS),
}
hist_ramp = {e: {k: hist[e][k] for k in ("unique_targets", "context", "accept", "unbiased")}
             for e in ("28", "29", "30") if e in hist}
report = {
    "purpose": "E2 distribution check vs trained ENVELOPE [MIRAGE] fingerprint",
    "command": "python " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    "cwd": os.getcwd(),
    "inputs": {
        "guides": {"path": GUIDES, "sha256": C.sha256_file(GUIDES),
                   "views": int(n), "valid": int(V[:n].sum()),
                   "selection_seed": int(d["selection_seed"]),
                   "crop_seed": int(d["crop_seed"])},
        "historical": {"path": HIST, "sha256": C.sha256_file(HIST),
                       "epochs": "31-100 (r_t=1), per-epoch means of 187 logged batches"},
        "yaml_variants": {v: versions[v]["yaml"] for v in versions},
        "ref_804c639_sha256": ref804_sha,
        "working_tree_curriculum_sha256": C.sha256_file(NEW.__file__),
    },
    "seeds": SEEDS, "batch_size": B,
    "results": results, "ramp_replay_seed0": ramp, "historical_ramp": hist_ramp,
    "comparison": comparison, "verdict": verdict,
    "elapsed_s": round(time.time() - t0, 1),
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1)
print("%-15s %-24s %-10s %-10s %-10s" % ("stat", "hist mean [min,max]", "v1", "v2", "804c639"))
for name, _ in LOG_KEYS:
    c = comparison[name]
    h = c["historical_ep31_100"]
    print("%-15s %7.3f [%7.3f,%7.3f] %9.3f%s %9.3f%s %9.3f"
          % (name, h["mean"], h["min"], h["max"],
             c["legacy_uniform_v1"]["mean"], "*" if c["legacy_uniform_v1"]["inside_hist_range"] else " ",
             c["least_overlap_v2"]["mean"], "*" if c["least_overlap_v2"]["inside_hist_range"] else " ",
             c["ref_804c639"]["mean"]))
print(json.dumps(verdict, indent=1))
print("wrote", OUT)
