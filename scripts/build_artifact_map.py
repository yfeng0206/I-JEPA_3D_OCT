"""Build the artifact map: every downstream run -> its head, predictions, encoder.

Downstream artifacts are split across three places — trained heads and encoders
are too large for git and live on Hugging Face, per-volume predictions are
committed here, and the metrics sit in per-run JSON. Nothing tied those together,
so given a published AUC there was no mechanical way to find the weights that
produced it.

This script scans the local runs and emits that mapping in two forms:

    results/downstream/ARTIFACT_MAP.json  machine-readable, sha256 per file
    results/downstream/ARTIFACT_MAP.md    the same table for humans

Run it after adding a new downstream arm:

    python scripts/build_artifact_map.py
"""
import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = "results/downstream"
HF_REPO = "yfeng0206/ijepa-3d-oct-checkpoints"
HF_URL = f"https://huggingface.co/{HF_REPO}"

# Pretraining encoders, published on Hugging Face.
ENCODERS = {
    "random": "random-posfix-100ep/jepa_patch-ep{ep:03d}.pth.tar",
    "oracle": "oracle-anatomical-100ep/jepa_patch_oracle-ep{ep:03d}.pth.tar",
}

# (run dir, prefix, arm, pretrain epoch, probe, frozen?, HF head path)
RUNS = [
    ("meanpool_sweep_oracle", "ep50",  "oracle",  50, "MeanPool", True,  "frozen-meanpool/oracle-ep50-head.pt"),
    ("meanpool_sweep_oracle", "ep75",  "oracle",  75, "MeanPool", True,  "frozen-meanpool/oracle-ep75-head.pt"),
    ("meanpool_sweep_oracle", "ep100", "oracle", 100, "MeanPool", True,  "frozen-meanpool/oracle-ep100-head.pt"),
    ("meanpool_sweep_random", "ep50",  "random",  50, "MeanPool", True,  "frozen-meanpool/random-ep50-head.pt"),
    ("meanpool_sweep_random", "ep75",  "random",  75, "MeanPool", True,  "frozen-meanpool/random-ep75-head.pt"),
    ("meanpool_sweep_random", "ep100", "random", 100, "MeanPool", True,  "frozen-meanpool/random-ep100-head.pt"),
    ("frozen_random_crossattn", "ep100", "random", 100, "CrossAttnPool", True, "frozen-other/random-ep100-crossattnpool.pt"),
    ("linear_sweep_random_posfix_d1", "ep25",  "random",  25, "Attentive d1", True, "frozen-other/random-ep25-attentive-d1.pt"),
    ("linear_sweep_random_posfix_d1", "ep50",  "random",  50, "Attentive d1", True, "frozen-other/random-ep50-attentive-d1.pt"),
    ("linear_sweep_random_posfix_d1", "ep75",  "random",  75, "Attentive d1", True, "frozen-other/random-ep75-attentive-d1.pt"),
    ("linear_sweep_random_posfix_d1", "ep100", "random", 100, "Attentive d1", True, "frozen-other/random-ep100-attentive-d1.pt"),
    ("finetune_oracle", "meanpool",  "oracle", 100, "MeanPool",      False, "finetuned/oracle-meanpool.pt"),
    ("finetune_oracle", "crossattn", "oracle", 100, "CrossAttnPool", False, "finetuned/oracle-crossattnpool.pt"),
    ("finetune_oracle", "d1",        "oracle", 100, "Attentive d1",  False, "finetuned/oracle-attentive.pt"),
    ("finetune_random", "mean_pool",       "random", 100, "MeanPool",      False, "finetuned/random-meanpool.pt"),
    ("finetune_random", "cross_attn_pool", "random", 100, "CrossAttnPool", False, "finetuned/random-crossattnpool.pt"),
    ("finetune_random", "attentive",       "random", 100, "Attentive d1",  False, "finetuned/random-attentive.pt"),
]

# Some runs name their results file differently.
RESULTS_ALIASES = ["{p}_results.json", "{arm}_{p}.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find(run, prefix, arm, patterns):
    for pat in patterns:
        p = os.path.join(ROOT, run, pat.format(p=prefix, arm=arm))
        if os.path.exists(p):
            return p
    return None


def entry(run, prefix, arm, ep, probe, frozen, hf_head):
    res_p = find(run, prefix, arm, RESULTS_ALIASES)
    res = json.load(open(res_p)) if res_p else {}

    head_p = find(run, prefix, arm, ["{p}_best_model.pt"])
    pred_p = find(run, prefix, arm, ["{p}_test_predictions.npz"])

    rec = {
        "run": f"{run}/{prefix}",
        "arm": arm,
        "probe": probe,
        "encoder_state": "frozen" if frozen else "fine-tuned",
        "pretrain_epoch": ep,
        "test_auc": res.get("test_auc"),
        "val_auc": res.get("best_val_auc"),
        "sensitivity": res.get("sensitivity"),
        "specificity": res.get("specificity"),
        "artifacts": {
            "head_local": head_p.replace("\\", "/") if head_p else None,
            "head_hf": f"downstream-heads/{hf_head}",
            "head_sha256": sha256(head_p) if head_p else None,
            "head_bytes": os.path.getsize(head_p) if head_p else None,
            "predictions_local": pred_p.replace("\\", "/") if pred_p else None,
            "results_json": res_p.replace("\\", "/") if res_p else None,
            # A fine-tuned head bundles its own encoder; a frozen one does not.
            "encoder_hf": None if not frozen else ENCODERS[arm].format(ep=ep),
        },
    }
    return rec


def main():
    entries = [entry(*r) for r in RUNS]
    entries.sort(key=lambda e: -(e["test_auc"] or 0))

    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_artifact_map.py",
        "hf_repo": HF_REPO,
        "test_split": {"name": "FairVision Test", "n": 3000,
                       "positive": 1466, "negative": 1534, "seed": 42},
        "note": ("Frozen entries need encoder_hf + head_hf together. "
                 "Fine-tuned entries bundle their encoder in the head file."),
        "runs": entries,
    }
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "ARTIFACT_MAP.json"), "w") as f:
        json.dump(doc, f, indent=2)

    lines = [
        "# Artifact map",
        "",
        "Generated by `scripts/build_artifact_map.py` — do not edit by hand.",
        "",
        f"Every downstream result below maps to the exact weights that produced it. "
        f"Heads and encoders are published at [`{HF_REPO}`]({HF_URL}); per-volume "
        f"predictions are committed in this directory.",
        "",
        "Split: FairVision **Test**, N=3000 (1466 positive / 1534 negative), `seed: 42`.",
        "",
        "| Test AUC | arm | probe | encoder | head (Hugging Face) | encoder (Hugging Face) |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        a = e["artifacts"]
        auc = f"**{e['test_auc']:.4f}**" if e["test_auc"] else "—"
        enc = f"`{a['encoder_hf']}`" if a["encoder_hf"] else "_bundled in head_"
        lines.append(
            f"| {auc} | {e['arm']} | {e['probe']} | {e['encoder_state']} "
            f"| `{a['head_hf']}` | {enc} |")

    lines += [
        "",
        "## Reproducing a number",
        "",
        "```bash",
        "python scripts/load_classifier.py \\",
        "  --head    results/downstream/meanpool_sweep_oracle/ep100_best_model.pt \\",
        "  --encoder <oracle ep100 .pth.tar> \\",
        "  --data-dir <FairVision root> --split Test \\",
        "  --expect-npz results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz",
        "```",
        "",
        "`ARTIFACT_MAP.json` carries the same data plus sha256 and byte size for "
        "every head, and the local path of each predictions file.",
    ]
    with open(os.path.join(ROOT, "ARTIFACT_MAP.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    missing = [e["run"] for e in entries if not e["artifacts"]["head_local"]]
    print(f"wrote ARTIFACT_MAP.json / .md  ({len(entries)} runs)")
    print(f"  with test_auc     : {sum(1 for e in entries if e['test_auc'])}")
    print(f"  heads found local : {sum(1 for e in entries if e['artifacts']['head_local'])}")
    print(f"  predictions local : {sum(1 for e in entries if e['artifacts']['predictions_local'])}")
    if missing:
        print(f"  MISSING heads     : {missing}")


if __name__ == "__main__":
    main()
