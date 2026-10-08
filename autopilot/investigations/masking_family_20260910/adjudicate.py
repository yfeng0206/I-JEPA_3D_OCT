"""Validate and consolidate the bounded masking-family investigation."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-findings", type=Path, required=True)
    parser.add_argument("--horizon-findings", type=Path, required=True)
    args = parser.parse_args()
    output = HERE / "FINDINGS.json"
    if output.exists():
        raise FileExistsError(output)
    probes = read(args.probe_findings)
    horizons = read(args.horizon_findings)
    checkpoints = read(HERE / "checkpoint_diagnostics.json")
    guides = read(HERE / "guide_product_comparison.json")
    placement = read(HERE / "placement_diversity_and_ramp_v2.json")
    rng = read(HERE / "probe_rng_witness.json")
    literature = read(HERE / "literature.json")
    if probes["status"] != "complete_and_output_validated" or horizons["status"] != "complete_and_validated":
        raise ValueError("Unvalidated probe evidence")
    if len(checkpoints["checkpoints"]) != 11 or checkpoints["optimizer_updates"] != 0:
        raise ValueError("Unexpected GPU scope")
    if len(probes["summary"]) != 7 or horizons["control_verification"]["fits"] != 6:
        raise ValueError("Unexpected head-fit scope")
    endpoints = horizons["endpoints"]
    short_drop = (endpoints["cover100"]["paired50_selected_validation_auc"]
                  - endpoints["cover50"]["paired50_selected_validation_auc"])
    long_drop = (endpoints["cover100"]["paired200_selected_validation_auc"]
                 - endpoints["cover50"]["paired200_selected_validation_auc"])
    reduction = 1 - long_drop / short_drop
    if abs(reduction - horizons["cover_decline_descriptive_reduction_fraction"]) > 1e-12:
        raise ValueError("Head-horizon effect does not reconcile")
    if sha(HERE / "masking_family_diagnostic.py") != checkpoints["manifest"]["script_sha256"]:
        raise ValueError("GPU diagnostic source mismatch")
    if sha(HERE / "mask_placement_replay.py") != placement["script_sha256"]:
        raise ValueError("Placement diagnostic source mismatch")
    if sha(HERE / "probe_rng_witness.py") != rng["script_sha256"]:
        raise ValueError("RNG witness source mismatch")
    evidence_paths = [
        args.probe_findings, args.horizon_findings,
        HERE / "checkpoint_diagnostics.json", HERE / "guide_product_comparison.json",
        HERE / "placement_diversity_and_ramp_v2.json", HERE / "probe_rng_witness.json",
        HERE / "literature.json",
    ]
    finding = {
        "status": "bounded_investigation_complete",
        "complete_historical_auc_cause_established": False,
        "strongest_new_finding": (
            "The original frozen-probe horizon and annealing schedule amplify COVER's "
            "late Validation decline. The encoder features were unchanged while the "
            "same head/AdamW recipe with a longer horizon substantially reduced that decline."
        ),
        "controlled_horizon_diagnostic": {
            "split": "Validation, not Test",
            "training_n": 6000, "validation_n": 1000,
            "seed": 42,
            "held_fixed": [
                "precomputed features", "LayerNorm-linear head", "initial parameter state",
                "paired training-order streams", "AdamW", "peak LR 0.0004",
                "weight decay 0.05", "batch size 256", "warmup five epochs",
                "Validation-AUC patience 15",
            ],
            "changed": "Configured head horizon 50 to 200, with correspondingly stretched cosine schedule",
            "endpoints": endpoints,
            "short_horizon_cover100_minus50": short_drop,
            "long_horizon_cover100_minus50": long_drop,
            "descriptive_decline_reduction_fraction": reduction,
            "limits": horizons["interpretation_limits"],
        },
        "alternate_linear_probe": {
            "protocol": "Predeclared non-affine feature LayerNorm plus LogisticRegression C1, lbfgs, max_iter1000",
            "summary": probes["summary"],
            "conclusion": probes["question_answer"],
            "limits": probes["limits"],
        },
        "confirmed_setup_differences": [
            {
                "finding": "Guide product changed along with the masking method",
                "old": "Historical ENVELOPE repaired hard-envelope cache",
                "new": "ANATOMY-v1 cfg7 soft guide; ANATOMY-v2/COVER encoder-tap adapted soft guide",
                "old_new_guide_pooled_iou_on576_views": guides["pooled_guide_iou"],
                "additional_lineage": "Both retained new guide adapters were taught by the ENVELOPE epoch100 checkpoint.",
                "limits": "Neither guide is clinical ground truth; a changed guide is not itself proof of worse guidance.",
            },
            {
                "finding": "ANATOMY changes target/context budgets even at guidance probability zero",
                "first64_view_ramp_zero": {
                    name: placement["results"][name]["ramp_zero"]
                    for name in ("random", "oracle", "envelope", "anatomy", "cover_legacy")
                },
                "limits": "The configured K16 target path is not an ignored ramp or a COVER-specific failure.",
            },
            {
                "finding": "Default head RNG depends on cache-iterator work before head construction",
                "witness": rng,
                "limits": "Reproduced source-pattern coupling, not a quantified explanation of the historical AUC ordering.",
            },
        ],
        "rejected_or_weakened_explanations": [
            {
                "hypothesis": "A new broken transformer implementation explains the weak families",
                "evidence": checkpoints["checkpoints"]["ancestor25"]["historical_transformer_parity"],
                "limits": "Exact parity in the tested July31/current-source and fixed-ancestor mask cases, not every historical checkout.",
            },
            {
                "hypothesis": "The weak models stopped updating or only output an image-independent template",
                "evidence": "Late AdamW counters progress; twelve actual backwards have finite nonzero gradients; wrong-image context materially increases loss across the eleven checkpoints.",
                "limits": "Does not establish retention of glaucoma-specific signal or rule out all representation degradation.",
            },
            {
                "hypothesis": "COVER is effectively static even with ordinary size variation",
                "tissue_membership_variation_percent": {
                    name: {
                        "fixed_sizes": placement["results"][name]["full_guidance"]["tissue_cells_changing_target_membership_pct"],
                        "varied_sizes": placement["results"][name]["full_guidance"]["with_sizes_also_varied_tissue_cells_changing_membership_pct"],
                    } for name in ("envelope", "anatomy", "cover_legacy")
                },
                "limits": "Fixed crops only. Restoring size variation makes COVER much less static; crop diversity remains unmeasured here.",
            },
            {
                "hypothesis": "ANATOMY-v2 and COVER have one demonstrated common collapse",
                "evidence": "ANATOMY has poor random-mask prediction transfer and a persistent alternate-probe decline; COVER retains ordinary-mask prediction ability and shows much greater readout-budget sensitivity.",
                "limits": "Distinct observations do not prove completely different causes.",
            },
        ],
        "literature": {
            "verified_studies": [item["title"] for item in literature["studies"]],
            "interpretation": "There is genuine positive importance-guided latent-prediction evidence. Neither those studies nor the local percentages establish that COVER must dominate.",
            "source": "literature.json",
        },
        "actions_not_taken": [
            "No sustained pretraining, encoder optimizer step, or new pretraining checkpoint",
            "No new Test-set evaluation or Test-based parameter selection",
            "No modification of historical checkpoints, feature caches, predictions or trained heads",
            "No manuscript, Overleaf, production training-code or main-branch change",
        ],
        "recommended_order": [
            "Make future probe initialization and training-loader RNG independent of cache history.",
            "Fix a common sufficiently trained probe protocol using Training/Validation, across all compared arms; do not assume200 epochs proves convergence.",
            "Separate guide-product, target-budget and delivered-context interventions in any subsequent masking comparison.",
            "Do not infer a complete pretraining mechanism or launch a long retraining campaign from these diagnostics alone.",
        ],
        "evidence": {str(path): sha(path) for path in evidence_paths},
        "adjudication_script_sha256": sha(__file__),
    }
    output.write_text(json.dumps(finding, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Validated FINDINGS.json written.")
    print("Controlled COVER Validation decline:", short_drop, "->", long_drop)
    print("Descriptive reduction:", reduction)


if __name__ == "__main__":
    main()
