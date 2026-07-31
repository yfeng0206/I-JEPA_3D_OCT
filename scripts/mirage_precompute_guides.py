#!/usr/bin/env python
"""Precompute repaired MIRAGE retinal envelopes for I-JEPA training.

Envelope repair costs about 6.5 ms per slice, which would add roughly a quarter
of an hour of CPU work to *every* epoch if done in the data loader.  This script
runs it once over the cached MIRAGE masks and stores bit-packed native 200x200
envelopes plus per-slice quality control, so the dataset only has to unpack.

Example:
    python scripts/mirage_precompute_guides.py --limit 4     # smoke
    python scripts/mirage_precompute_guides.py               # full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    GUIDE_SCHEMA_VERSION,
    RepairParams,
    build_union,
    pack_guides,
    params_fingerprint,
    repair_union,
    unpack_guides,
)

MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
GUIDE_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training")
MANIFEST = Path(r"D:\jepa_phase0\fairvision-glaucoma\manifests\mirage-guides-complete.json")

SLICES = 100
NATIVE = 200


def validate_guide(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "schema_version",
                "params_fingerprint",
                "packed_envelopes",
                "valid",
                "area_frac",
                "span_frac",
                "source_filename",
            }
            if not required.issubset(cache.files):
                return False
            return (
                int(cache["schema_version"]) == GUIDE_SCHEMA_VERSION
                and str(cache["params_fingerprint"].item()) == fingerprint
                and cache["valid"].shape == (SLICES,)
                and cache["packed_envelopes"].shape[0] == SLICES
            )
    except Exception:
        return False


def process(path: Path, output: Path, params: RepairParams, fingerprint: str) -> dict:
    with np.load(path, allow_pickle=False) as cache:
        masks = cache["hard_masks"]
        slice_indices = cache["slice_indices"]
        label = int(cache["glaucoma"])
    if masks.shape != (SLICES, NATIVE, NATIVE):
        raise RuntimeError(f"{path.name}: unexpected mask shape {masks.shape}")

    envelopes = np.zeros((SLICES, NATIVE, NATIVE), dtype=bool)
    valid = np.zeros(SLICES, dtype=bool)
    area = np.zeros(SLICES, dtype=np.float32)
    span = np.zeros(SLICES, dtype=np.float32)
    for index in range(SLICES):
        envelope, is_valid, stats = repair_union(
            build_union(masks[index]), params=params
        )
        envelopes[index] = envelope
        valid[index] = is_valid
        area[index] = stats["repaired_area_frac"]
        span[index] = stats["span_frac"]

    packed = pack_guides(envelopes)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.stem}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(GUIDE_SCHEMA_VERSION, dtype=np.int16),
                params_fingerprint=np.asarray(fingerprint),
                source_filename=np.asarray(path.name),
                glaucoma=np.asarray(label, dtype=np.int8),
                slice_indices=slice_indices,
                packed_envelopes=packed,
                valid=valid,
                area_frac=area.astype(np.float16),
                span_frac=span.astype(np.float16),
            )
        if not validate_guide(temporary, fingerprint):
            raise RuntimeError(f"Generated guide failed validation: {path.name}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "filename": path.name,
        "valid_slices": int(valid.sum()),
        "mean_area_frac": float(area.mean()),
        "output_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    parser.add_argument("--output", type=Path, default=GUIDE_DIR)
    args = parser.parse_args()

    params = DEFAULT_REPAIR
    fingerprint = params_fingerprint(params)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.mask_dir.glob("data_*.npz"))
    if not sources:
        raise FileNotFoundError(f"No MIRAGE caches under {args.mask_dir}")

    pending = [
        path
        for path in sources
        if not validate_guide(args.output / path.name, fingerprint)
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        json.dumps(
            {
                "sources": len(sources),
                "pending": len(pending),
                "params_fingerprint": fingerprint,
                "output": str(args.output),
            }
        ),
        flush=True,
    )

    started = time.monotonic()
    valid_total, area_total, bytes_total, done = 0, 0.0, 0, 0
    for path in pending:
        record = process(path, args.output / path.name, params, fingerprint)
        valid_total += record["valid_slices"]
        area_total += record["mean_area_frac"]
        bytes_total += record["output_bytes"]
        done += 1
        if done % args.report_every == 0 or done == len(pending):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = done / elapsed
            remaining = (len(pending) - done) / max(rate, 1e-9)
            print(
                f"{done}/{len(pending)} volumes; {rate * 100:.1f} slices/s; "
                f"ETA={remaining / 3600:.2f}h; cache={bytes_total / 1e9:.2f}GB",
                flush=True,
            )

    if args.limit is None:
        complete = [
            path
            for path in sources
            if validate_guide(args.output / path.name, fingerprint)
        ]
        summary = {
            "status": "complete" if len(complete) == len(sources) else "partial",
            "volumes": len(complete),
            "expected": len(sources),
            "params": params.to_dict(),
            "params_fingerprint": fingerprint,
            "guide_root": str(args.output),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "limited_run": True,
                    "processed": done,
                    "mean_valid_slices": valid_total / max(done, 1),
                    "mean_area_frac": area_total / max(done, 1),
                    "mean_bytes": bytes_total / max(done, 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
