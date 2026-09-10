"""Independently aggregate saved row fields; does not re-derive the guide masks."""

import json
import math
from pathlib import Path
import statistics

import yaml


ROOT = Path(__file__).resolve().parents[3]
ROWS = Path(__file__).parent / "evidence" / "mask_replay600_v2"


def main():
    config = yaml.safe_load(
        (ROOT / "configs" / "patch_cover_f021_ep25.yaml").read_text()
    )["mask"]["curriculum"]
    fraction = config["cover_min_visible_frac"]
    minimum = config["cover_min_visible_cells"]
    expected = json.loads((ROWS / "full_batch64_confirmation.json").read_text())
    results = {}
    for policy in ("cover_legacy", "cover_v2", "cover_v2_guard"):
        with (ROWS / f"{policy}_bs64.jsonl").open() as stream:
            rows = [row for line in stream if (row := json.loads(line))["batch_size"] == 64]
        if len(rows) != 576:
            raise ValueError(f"{policy}: expected nine complete 64-view batches")
        valid = [row for row in rows if row["guide_valid"]]
        if not valid:
            raise ValueError(f"{policy}: no valid guides")
        for row in rows:
            target_union = {index for group in row["targets"] for index in group}
            if any(target_union.intersection(group) for group in row["context"]):
                raise ValueError(f"{policy}: target-context overlap")
            if row["delivered_target_tissue_unique"] != row["target_tissue_unique"]:
                raise ValueError(f"{policy}: delivered-tissue counters disagree")
        results[policy] = {
            "full_batch_images": len(rows),
            "valid": len(valid),
            "target_tissue_truncation_losses": sum(
                row["intended_target_tissue_unique"] > row["target_tissue_unique"]
                for row in rows
            ),
            "zero_tissue_context": sum(row["context_tissue"] == 0 for row in rows),
            "valid_floor_misses": sum(
                row["context_tissue"]
                < max(math.ceil(fraction * row["tissue_cells"]), min(minimum, row["tissue_cells"]))
                for row in valid
            ),
            "mean_context_tissue": statistics.mean(row["context_tissue"] for row in rows),
            "mean_scored_mass": statistics.mean(row["scored_hidden_mass_fraction"] for row in valid),
            "mean_delivered_mass": statistics.mean(
                row["delivered_hidden_mass_fraction"] for row in valid
            ),
        }
        for field, value in results[policy].items():
            if not math.isclose(value, expected[policy][field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{policy}/{field}: row aggregation differs from summary")
    report = {
        "status": "passed",
        "scope": "Re-aggregates saved row-level measurements; not a new segmentation truth check",
        "config_fraction": fraction,
        "config_minimum": minimum,
        "results": results,
    }
    (ROWS / "coordinator_aggregate_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
