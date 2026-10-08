#!/usr/bin/env python
"""G1 gate checker for cr_seed_v1 training runs (PLAN section 3.4).

Reads a run's trainer stdout logs (train_a<k>.log, one per launch attempt; later
attempts override earlier ones for the same epoch) and, as a cross-check, the
per-iteration CSV, and compares every completed epoch with
configs/cr_seed_v1/reference_curves.json:

  G1 loss bands   |run - own-arm reference| for train and val loss:
                  > 0.003 -> WARN; > 0.005 on two consecutive epochs -> HOLD.
  non-finite      NaN/Inf train or val loss -> STOP.  Wrong horizon (Epoch N/H with
                  H != 100) or a first logged epoch other than 26 -> STOP.
  G1 identity     from epoch 31: cumulative mean |train - ref| over epochs 31..e must be
                  smallest for the run's own arm -> otherwise HOLD.
  ENVELOPE mask   from epoch 31: per-epoch means of the [MIRAGE] log statistics must lie
                  in the historical per-epoch range (ep31-100, A1 section 3.2) -> PASS;
                  within the range widened by 50% of its width on each side -> WARN;
                  beyond -> HOLD for unique_targets/context/accept, WARN for the others.

Severity: PASS < INFO < WARN < HOLD < STOP.  HOLD means "alert the coordinator";
the campaign runner keeps training.  Output: JSON (``--out``) + one summary line.

    python scripts/cr_gate_check.py --run-dir D:\\jepa_phase0\\runs\\cr_seed_v1_random_s1234 --arm random
    python scripts/cr_gate_check.py --build-reference --a1-scratch <dir>   # (re)build reference_curves.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "configs" / "cr_seed_v1" / "reference_curves.json"
SEVERITY = {"PASS": 0, "INFO": 1, "WARN": 2, "HOLD": 3, "STOP": 4}
ARM_REF = {"random": "random", "centroid": "centroid", "envelope": "envelope",
           "random_cb": None}
HORIZON = 100
FIRST_EPOCH = 26

EPOCH_RE = re.compile(
    r"^Epoch (\d+)/(\d+)\s+\((\d+)s\)\s+train_loss=(\S+?)(?:\s+val_loss=(\S+?))?(\s+\*)?\s*$")
ITER_RE = re.compile(r"\[Epoch (\d+)/(\d+) \| Iter (\d+)/(\d+)\]\s+loss=(\S+)")
MIRAGE_RE = re.compile(r"\[MIRAGE\]\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z_/]+)=([-+0-9.eEnaifNI]+)")
ATTEMPT_RE = re.compile(r"train_a(\d+)\.log$")
MIRAGE_KEYS = ("patches/block", "unique_targets", "context", "on_region", "background",
               "fallbacks", "infeasible", "unbiased", "accept", "fill", "retina_visible", "tries")
FINGERPRINT_KEYS = ("unique_targets", "context", "accept", "fill", "retina_visible",
                    "on_region", "tries", "patches/block", "fallbacks", "infeasible")
# Placement-sensitive statistics that separate the legacy sampler from FIX-A (i3/A1); the others
# have narrow historical ranges (on_region spans 0.005) or are integer counts, so they can only WARN.
PRIMARY_MASK_KEYS = ("unique_targets", "context", "accept")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def worst(*statuses):
    s = [x for x in statuses if x]
    return max(s, key=lambda k: SEVERITY[k]) if s else "PASS"


# ---------------------------------------------------------------------------
# log parsing
# ---------------------------------------------------------------------------

def attempt_logs(run_dir) -> list[str]:
    files = glob.glob(os.path.join(str(run_dir), "train_a*.log"))
    def key(p):
        m = ATTEMPT_RE.search(p)
        return int(m.group(1)) if m else -1
    return sorted((p for p in files if ATTEMPT_RE.search(p)), key=key)


def parse_log_text(text: str) -> dict:
    """Parse one attempt's stdout. Returns {'epochs': {e: {...}}, 'order': [...], 'horizons': set}."""
    epochs, order, horizons = {}, [], set()
    mirage = {}
    last_iter = {}
    cur_epoch = None
    for line in text.splitlines():
        line = line.rstrip("\r")
        m = ITER_RE.search(line)
        if m:
            cur_epoch = int(m.group(1))
            horizons.add(int(m.group(2)))
            last_iter[cur_epoch] = (int(m.group(3)), int(m.group(4)))
            continue
        m = MIRAGE_RE.search(line)
        if m and cur_epoch is not None:
            kv = {k: _f(v) for k, v in KV_RE.findall(m.group(1))}
            mirage.setdefault(cur_epoch, []).append(kv)
            continue
        m = EPOCH_RE.match(line.strip())
        if m:
            e = int(m.group(1))
            horizons.add(int(m.group(2)))
            epochs[e] = {"epoch": e, "horizon": int(m.group(2)), "loop_seconds": int(m.group(3)),
                         "train_loss": _f(m.group(4)),
                         "val_loss": _f(m.group(5)) if m.group(5) is not None else None,
                         "improved": bool(m.group(6))}
            order.append(e)
    for e, rec in epochs.items():
        rec["mirage_lines"] = mirage.get(e, [])
    return {"epochs": epochs, "order": order, "horizons": horizons, "last_iter": last_iter}


def parse_run_logs(run_dir) -> dict:
    merged, attempts, problems = {}, [], []
    horizons = set()
    last_iter = None
    for k, path in enumerate(attempt_logs(run_dir)):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            p = parse_log_text(fh.read())
        horizons |= p["horizons"]
        if p["order"] != sorted(p["order"]) or len(set(p["order"])) != len(p["order"]):
            problems.append("non-monotonic epoch sequence in %s: %s" % (os.path.basename(path), p["order"]))
        for e, rec in p["epochs"].items():
            rec["attempt_log"] = os.path.basename(path)
            merged[e] = rec
        if p["last_iter"]:
            e = max(p["last_iter"])
            last_iter = {"epoch": e, "iter": p["last_iter"][e][0], "of": p["last_iter"][e][1],
                         "log": os.path.basename(path)}
        attempts.append({"log": os.path.basename(path), "epochs": p["order"]})
    return {"epochs": merged, "attempts": attempts, "horizons": sorted(horizons),
            "problems": problems, "last_iter": last_iter}


def csv_epoch_means(csv_path) -> dict:
    """Per-epoch mean of the per-iteration loss; duplicate (epoch, iteration) rows from
    appended attempts are resolved to the LAST occurrence."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    rows = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows[(int(float(r["epoch"])), int(float(r["iteration"])))] = float(r["loss"])
            except (KeyError, TypeError, ValueError):
                continue
    by_e = {}
    for (e, _i), v in rows.items():
        by_e.setdefault(e, []).append(v)
    return {e: {"mean": sum(v) / len(v), "n": len(v)} for e, v in by_e.items()}


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------

def load_reference(path=REFERENCE) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ref(ref, arm, kind, e):
    v = ref["arms"].get(arm, {}).get(kind, {}).get(str(e))
    return None if v is None else float(v)


def evaluate(parsed: dict, arm: str, ref: dict, csv_means: dict | None = None) -> dict:
    bands = ref["bands"]
    warn_abs, hold_abs = float(bands["warn_abs"]), float(bands["hold_abs"])
    id_from, mask_from = int(bands["identity_from_epoch"]), int(bands["mask_from_epoch"])
    widen = float(bands["mask_widen_frac"])
    own = ARM_REF.get(arm, arm)
    epochs = parsed["epochs"]
    out_epochs = {}
    global_notes = []
    global_status = "PASS"
    if any(h != HORIZON for h in parsed["horizons"]):
        global_status = "STOP"
        global_notes.append("horizon %s != %d" % (parsed["horizons"], HORIZON))
    if epochs and min(epochs) != FIRST_EPOCH:
        global_status = "STOP"
        global_notes.append("first logged epoch %d != %d (fork start not honoured)"
                            % (min(epochs), FIRST_EPOCH))
    for p in parsed.get("problems", []):
        global_status = "STOP"
        global_notes.append(p)
    prev_out5 = False
    prev_e = None
    for e in sorted(epochs):
        rec = epochs[e]
        notes, checks = [], {}
        st = "PASS"
        tr, va = rec["train_loss"], rec["val_loss"]
        if not math.isfinite(tr) or (va is not None and not math.isfinite(va)):
            st = "STOP"
            notes.append("non-finite loss (train=%s val=%s)" % (tr, va))
        if va is None:
            st = worst(st, "WARN")
            notes.append("no val_loss on epoch line")
        # bands
        if own is not None and math.isfinite(tr):
            rt, rv = _ref(ref, own, "train", e), _ref(ref, own, "val", e)
            dt = None if rt is None else tr - rt
            dv = None if (rv is None or va is None or not math.isfinite(va)) else va - rv
            mags = [abs(x) for x in (dt, dv) if x is not None]
            out5 = bool(mags) and max(mags) > hold_abs
            band = "PASS"
            if mags and max(mags) > warn_abs:
                band = "WARN"
            consecutive = out5 and prev_out5 and prev_e == e - 1
            if consecutive:
                band = "HOLD"
            if rt is None:
                band = "INFO"
                notes.append("no reference for epoch %d" % e)
            checks["bands"] = {"ref_train": rt, "ref_val": rv, "d_train": dt, "d_val": dv,
                               "exceeds_hold_abs": out5, "status": band}
            st = worst(st, band)
            prev_out5 = out5
        elif own is None and math.isfinite(tr):
            info = {}
            for a in ref["arms"]:
                rt = _ref(ref, a, "train", e)
                info[a] = None if rt is None else tr - rt
            checks["bands"] = {"status": "INFO", "d_train_vs": info,
                               "note": "no own-arm reference for %s" % arm}
            st = worst(st, "INFO")
        prev_e = e
        # identity
        if e >= id_from and math.isfinite(tr):
            if own is None:
                checks["identity"] = {"status": "INFO", "note": "not applicable to %s" % arm}
            else:
                dist = {}
                for a in ref["arms"]:
                    diffs = []
                    for k in range(id_from, e + 1):
                        if k in epochs and math.isfinite(epochs[k]["train_loss"]):
                            r = _ref(ref, a, "train", k)
                            if r is not None:
                                diffs.append(abs(epochs[k]["train_loss"] - r))
                    if diffs:
                        dist[a] = sum(diffs) / len(diffs)
                closest = min(dist, key=dist.get) if dist else None
                ok = closest == own and all(dist[a] > dist[own] for a in dist if a != own)
                checks["identity"] = {"mean_abs_train_diff": dist, "closest": closest,
                                      "status": "PASS" if ok else "HOLD"}
                if not ok:
                    notes.append("identity: closest reference is %s, not %s" % (closest, own))
                st = worst(st, checks["identity"]["status"])
        # envelope mask fingerprint
        if arm == "envelope" and e >= mask_from:
            fp = ref["envelope_mask_fingerprint"]["stats"]
            lines = rec.get("mirage_lines", [])
            mres = {"n_lines": len(lines), "stats": {}}
            mst = "PASS"
            if len(lines) < int(ref["envelope_mask_fingerprint"].get("min_lines", 100)):
                mst = "WARN"
                notes.append("only %d [MIRAGE] lines for epoch %d" % (len(lines), e))
            for key in FINGERPRINT_KEYS:
                vals = [ln[key] for ln in lines if key in ln and math.isfinite(ln[key])]
                if not vals:
                    if lines:
                        miss = "HOLD" if key in PRIMARY_MASK_KEYS else "WARN"
                        mst = worst(mst, miss)
                        mres["stats"][key] = {"status": miss, "note": "missing"}
                    continue
                mean = sum(vals) / len(vals)
                lo, hi = float(fp[key]["min"]), float(fp[key]["max"])
                w = (hi - lo) * widen
                if lo <= mean <= hi:
                    s = "PASS"
                elif lo - w <= mean <= hi + w:
                    s = "WARN"
                else:
                    s = "HOLD"
                if s == "HOLD" and key not in PRIMARY_MASK_KEYS:
                    s = "WARN"
                mres["stats"][key] = {"mean": mean, "range": [lo, hi], "status": s,
                                      "primary": key in PRIMARY_MASK_KEYS}
                mst = worst(mst, s)
            if not lines:
                mst = "HOLD"
                notes.append("no [MIRAGE] lines for epoch %d" % e)
            mres["status"] = mst
            checks["mask"] = mres
            st = worst(st, mst)
        if csv_means and e in csv_means and math.isfinite(tr):
            cm = csv_means[e]
            checks["csv"] = {"mean": cm["mean"], "n": cm["n"], "abs_diff_vs_log": abs(cm["mean"] - tr)}
        out_epochs[e] = {"epoch": e, "train_loss": tr, "val_loss": va,
                         "loop_seconds": rec.get("loop_seconds"), "attempt_log": rec.get("attempt_log"),
                         "status": st, "checks": checks, "notes": notes}
    latest = max(out_epochs) if out_epochs else None
    latest_status = worst(global_status, out_epochs[latest]["status"]) if latest else global_status
    worst_status = worst(global_status, *[r["status"] for r in out_epochs.values()])
    res = {"arm": arm, "reference_arm": own, "epochs": {str(k): v for k, v in out_epochs.items()},
           "latest_epoch": latest, "latest_status": latest_status, "worst_status": worst_status,
           "global_notes": global_notes, "attempts": parsed.get("attempts", []),
           "last_iter": parsed.get("last_iter")}
    res["summary"] = summary_line(res)
    return res


def summary_line(res: dict) -> str:
    e = res["latest_epoch"]
    if e is None:
        return "G1 %s: no completed epoch yet (%s)%s" % (
            res["arm"], res["latest_status"],
            ("; " + "; ".join(res["global_notes"])) if res["global_notes"] else "")
    r = res["epochs"][str(e)]
    b = r["checks"].get("bands", {})
    parts = ["G1 %s ep%d: %s" % (res["arm"], e, res["latest_status"]),
             "train=%.4f val=%s" % (r["train_loss"], "%.4f" % r["val_loss"] if r["val_loss"] is not None else "NA")]
    if b.get("d_train") is not None:
        parts.append("dtrain=%+.4f" % b["d_train"])
    if b.get("d_val") is not None:
        parts.append("dval=%+.4f" % b["d_val"])
    if "identity" in r["checks"]:
        parts.append("identity=%s" % r["checks"]["identity"]["status"])
    if "mask" in r["checks"]:
        parts.append("mask=%s" % r["checks"]["mask"]["status"])
    parts.append("worst=%s" % res["worst_status"])
    notes = r["notes"] + res["global_notes"]
    if notes:
        parts.append("notes: " + "; ".join(notes))
    return " | ".join(parts)


def check_run(run_dir, arm, ref_path=REFERENCE, csv_path=None) -> dict:
    ref = load_reference(ref_path)
    parsed = parse_run_logs(run_dir)
    if csv_path is None:
        cands = glob.glob(os.path.join(str(run_dir), "*-log.csv"))
        csv_path = cands[0] if len(cands) == 1 else None
    res = evaluate(parsed, arm, ref, csv_epoch_means(csv_path))
    res["run_dir"] = str(run_dir)
    res["reference"] = {"path": str(ref_path),
                        "sha256": hashlib.sha256(Path(ref_path).read_bytes()).hexdigest()}
    return res


# ---------------------------------------------------------------------------
# reference builder (one-off, from A1's evidence)
# ---------------------------------------------------------------------------

# Exact values from docs/experiments/pretraining/{random_100ep,oracle_100ep}.md (A1 section 3, bold).
DOC_POINTS = {
    "random": {25: (0.1174, 0.1197), 50: (0.1413, 0.1423)},
    "centroid": {26: (0.1186, 0.1202), 30: (0.1197, 0.1242), 35: (0.1232, 0.1310),
                 50: (0.1316, 0.1400)},
}


def build_reference(a1_scratch, out=REFERENCE) -> dict:
    a1 = Path(a1_scratch)
    dig_path = a1 / "digitized_loss.json"
    fp_path = a1 / "envelope_mirage_stats_by_epoch.json"
    env_csv = REPO / "logs" / "pretraining" / "mirage_epoch_summary.csv"
    dig = json.loads(dig_path.read_text(encoding="utf-8"))
    arms = {}
    for arm, key in (("random", "random"), ("centroid", "oracle")):
        tr, va, src = {}, {}, {}
        for e in range(25, 51):
            t = dig[key]["train"].get(str(e))
            v = dig[key]["val"].get(str(e))
            if t is not None:
                tr[str(e)], src[str(e)] = round(float(t), 5), "digitized"
            if v is not None:
                va[str(e)] = round(float(v), 5)
            if e in DOC_POINTS[arm]:
                tr[str(e)], va[str(e)] = DOC_POINTS[arm][e]
                src[str(e)] = "doc"
        arms[arm] = {"train": tr, "val": va, "source": src}
    arms["centroid"]["train"].setdefault("25", DOC_POINTS["random"][25][0])
    arms["centroid"]["val"].setdefault("25", DOC_POINTS["random"][25][1])
    arms["centroid"]["source"].setdefault("25", "doc (shared ancestor)")
    arms["random"]["val_note"] = "original RANDOM val loss is a rank-0 shard (~25% of the val set)"
    tr, va, src = {"25": DOC_POINTS["random"][25][0]}, {"25": DOC_POINTS["random"][25][1]}, {"25": "doc (shared ancestor)"}
    with open(env_csv, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            e = int(r["epoch"])
            if 26 <= e <= 50:
                tr[str(e)], va[str(e)], src[str(e)] = float(r["train_loss"]), float(r["val_loss"]), "log"
    arms["envelope"] = {"train": tr, "val": va, "source": src}
    fpd = json.loads(fp_path.read_text(encoding="utf-8"))
    stats = {}
    for key in FINGERPRINT_KEYS:
        vals = [float(fpd[str(e)][key]) for e in range(31, 101) if str(e) in fpd and key in fpd[str(e)]]
        v3150 = [float(fpd[str(e)][key]) for e in range(31, 51) if str(e) in fpd and key in fpd[str(e)]]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
        stats[key] = {"min": min(vals), "max": max(vals), "mean_31_100": m, "sd_31_100": sd,
                      "mean_31_50": sum(v3150) / len(v3150), "n_epochs": len(vals)}
    ref = {
        "schema": 1,
        "description": "cr_seed_v1 G1 reference curves (ep25-50) and ENVELOPE [MIRAGE] fingerprint",
        "sources": {
            "digitized": {"path": str(dig_path), "sha256": hashlib.sha256(dig_path.read_bytes()).hexdigest(),
                          "note": "A1 digitized train_val_loss.png (|err| <= 2e-4 vs doc points)"},
            "doc_points": "docs/experiments/pretraining/random_100ep.md, oracle_100ep.md (A1 section 3, bold)",
            "envelope_log": {"path": "logs/pretraining/mirage_epoch_summary.csv",
                             "sha256": hashlib.sha256(env_csv.read_bytes()).hexdigest()},
            "envelope_fingerprint": {"path": str(fp_path),
                                     "sha256": hashlib.sha256(fp_path.read_bytes()).hexdigest(),
                                     "note": "per-epoch means of the historical [MIRAGE] lines (187/epoch)"},
        },
        "bands": {"warn_abs": 0.003, "hold_abs": 0.005, "hold_consecutive": 2,
                  "identity_from_epoch": 31, "mask_from_epoch": 31, "mask_widen_frac": 0.5},
        "arms": arms,
        "envelope_mask_fingerprint": {
            "window": "historical per-epoch means, epochs 31-100 (r_t = 1)",
            "min_lines": 100, "stats": stats,
            "per_epoch_28_50": {e: fpd[e] for e in sorted(fpd, key=int) if 28 <= int(e) <= 50},
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(ref, indent=2) + "\n")
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir")
    ap.add_argument("--arm", choices=sorted(ARM_REF))
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--a1-scratch", default=None)
    args = ap.parse_args(argv)
    if args.build_reference:
        if not args.a1_scratch:
            ap.error("--build-reference needs --a1-scratch")
        ref = build_reference(args.a1_scratch, args.reference)
        print("wrote %s (%d arms)" % (args.reference, len(ref["arms"])))
        return 0
    if not args.run_dir or not args.arm:
        ap.error("--run-dir and --arm are required")
    res = check_run(args.run_dir, args.arm, args.reference, args.csv)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    print(res["summary"])
    return {"PASS": 0, "INFO": 0, "WARN": 0, "HOLD": 3, "STOP": 4}[res["worst_status"]]


if __name__ == "__main__":
    sys.exit(main())
