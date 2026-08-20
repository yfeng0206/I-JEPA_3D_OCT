"""Deep-diff a candidate run config against the baseline arms.

Purpose: prove that the ONLY differences are the masking knobs we intend to
change. Anything else that shows up here is an unintended deviation and would
confound the comparison, which is exactly how the previous campaign went wrong.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


# Keys that are SUPPOSED to differ: run identity, paths, and the masking method.
EXPECTED = (
    "logging.", "meta.read_checkpoint", "meta.load_checkpoint", "data.",
    "mask.curriculum.mode", "mask.curriculum.cover", "mask.curriculum.anatomy_tau",
    "mask.curriculum.mirage_", "mask.curriculum.enc_truncate",
    "mask.pred_target_k", "optimization.epochs", "meta.seed",
)


def expected(k):
    return any(k.startswith(p) or p in k for p in EXPECTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    a = flat(yaml.safe_load(open(args.candidate)))
    b = flat(yaml.safe_load(open(args.baseline)))

    print(f"candidate : {args.candidate}")
    print(f"baseline  : {args.baseline}\n")

    keys = sorted(set(a) | set(b))
    unexpected, intended = [], []
    for k in keys:
        va, vb = a.get(k, "<ABSENT>"), b.get(k, "<ABSENT>")
        if va == vb:
            continue
        (intended if expected(k) else unexpected).append((k, vb, va))

    print("=" * 78)
    print("INTENDED differences (masking method, paths, run identity)")
    print("=" * 78)
    for k, vb, va in intended:
        print(f"  {k}\n      baseline : {vb}\n      candidate: {va}")

    print("\n" + "=" * 78)
    if unexpected:
        print(f"!!! UNEXPECTED differences: {len(unexpected)} -- REVIEW EACH")
    else:
        print("NO unexpected differences. Everything else is identical.")
    print("=" * 78)
    for k, vb, va in unexpected:
        print(f"  {k}\n      baseline : {vb}\n      candidate: {va}")

    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
