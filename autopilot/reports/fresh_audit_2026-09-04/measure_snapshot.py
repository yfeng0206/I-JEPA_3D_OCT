"""Read saved prediction artifacts without fitting models or changing results."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("prediction_snapshot.json")


def main():
    entries = []
    status = None
    for line_number, line in enumerate(
        (ROOT / "HANDOFF.md").read_text(encoding="utf-8").splitlines(), 1
    ):
        status_match = re.match(r"### status: (\w+)", line)
        if status_match:
            status = status_match.group(1)
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5 or not re.fullmatch(r"0\.\d+", cells[3]):
            continue
        arm, epoch, precision, recorded_auc, source = cells
        path = Path(source.strip("`"))
        entry = {
            "handoff_line": line_number,
            "status": status,
            "arm": arm,
            "epoch": int(epoch) if epoch.isdigit() else None,
            "precision": precision,
            "path": str(path),
            "exists": path.is_file(),
            "recorded_auc": float(recorded_auc),
        }
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                labels = saved["labels"]
                probabilities = saved["probs"]
                if labels.ndim != 1 or probabilities.shape != labels.shape:
                    raise ValueError(f"Unexpected prediction shape: {path}")
                if not np.all(np.isin(labels, [0, 1])):
                    raise ValueError(f"Non-binary labels: {path}")
                if not np.all(np.isfinite(probabilities)):
                    raise ValueError(f"Non-finite scores: {path}")
                auc = float(roc_auc_score(labels, probabilities))
                entry.update(
                    n=int(labels.size),
                    positives=int(labels.sum()),
                    negatives=int(labels.size - labels.sum()),
                    score_dtype=str(probabilities.dtype),
                    array_keys=saved.files,
                    label_sequence_sha256=hashlib.sha256(
                        labels.astype(np.uint8).tobytes()
                    ).hexdigest(),
                    measured_auc=auc,
                    matches_six_decimal_record=abs(
                        auc - float(recorded_auc)
                    ) <= 0.0000005,
                )
            entry["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(entry)

    if not entries:
        raise ValueError("No prediction entries found; an empty inventory is not success")
    if len(entries) != 43:
        raise ValueError(f"Baseline inventory changed: expected 43 records, found {len(entries)}")
    measured = [e for e in entries if e["exists"]]
    by_status = {
        s: sum(e["status"] == s for e in entries)
        for s in sorted({e["status"] for e in entries})
    }
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source": "HANDOFF.md five-column prediction tables",
        "counts": by_status,
        "total": len(entries),
        "present": len(measured),
        "matching_auc": sum(e["matches_six_decimal_record"] for e in measured),
        "unique_label_sequences": len(
            {e["label_sequence_sha256"] for e in measured}
        ),
        "scope_limit": (
            "Recomputes AUC from stored scores only; does not rerun encoders or "
            "heads. Equal label arrays do not establish subject-identity pairing. "
            "Excluded and retracted records are not restored to valid evidence."
        ),
        "entries": entries,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
    for entry in entries:
        if not entry["exists"] or not entry.get("matches_six_decimal_record", False):
            print("ATTENTION", entry["arm"], entry["epoch"], entry["path"])
    return 0 if len(measured) == 43 and result["matching_auc"] == 43 else 1


if __name__ == "__main__":
    raise SystemExit(main())
