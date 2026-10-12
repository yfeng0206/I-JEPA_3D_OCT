"""Helpers for autopilot/cr_postfreeze.ps1, the post-freeze pipeline of cr_seed_v1.

Subcommands
  plan             preflight from the campaign state: the probe directory of every registered
                   run and anchor, which are ready / finished after the freeze (late) / still in
                   progress / not completed, unseal receipts, RANDOM-CB trainer logs. Reads
                   results.json and the hash-verified sealed-manifest JSON only, never a
                   prediction array. "Late" is cr_stats.completion_eligibility, exactly as
                   cr_stats final decides it (stop file, seal and probe finish times). With
                   --rehearsal every derived path must resolve inside --fixture-root (V8-1).
  input-key        cache key of the analysis inputs and their completion evidence (skips an
                   unchanged final / MD on re-runs; a changed or missing stop file recomputes)
  check-final      final.json must agree with the current completion evidence and the plan,
                   and its labels, contrasts, means and RANDOM range must be what the registered
                   rules give for the receipt-verified per-run AUCs (V10-1)
  contain          rehearsal: paths must resolve inside the fixture root (junctions followed)
  scratch-copy     copy a paper or poster directory for a scratch build
  compile-report   pages, main-text fraction, undefined references, overfull boxes of a compile
  summary          REVIEW.md for the human checkpoint
  rehearsal-mirror / rehearsal-bind / rehearsal-ack
                   synthetic rehearsal only: a mirror repository standing in for the real one,
                   Rep* review bindings for synthetic results (which build --update-reviews
                   refuses), a stand-in acknowledgment. All refuse the real repository.
Exit codes: 0 ok, 2 refused / blocking problem, 3 registered runs still in progress.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_stats  # noqa: E402
import make_cr_results as mk  # noqa: E402
import release_assets as assets  # noqa: E402

PLAN_SCHEMA = "cr_postfreeze_plan_v1"
MIRROR_MARKER = "CR_POSTFREEZE_REHEARSAL_MIRROR.json"
SCRATCH_MARKER = ".cr_postfreeze_scratch"
PAGE_LIMIT = 10


class Refused(RuntimeError):
    pass


def sha256_file(path):
    return cr_stats.sha256_file(path)


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# plan (preflight)
# ---------------------------------------------------------------------------

# Real campaign roots: a rehearsal never reads or unseals anything below them (V8-1).
FORBIDDEN_ROOTS = (r"D:\jepa_phase0\runs", r"D:\jepa_phase0\campaign",
                   r"D:\jepa_phase0\cr_worktrees")
FIXTURE_MARKER = "CR_SYNTHETIC_FIXTURE.json"


def _abs(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _real(path):
    return os.path.normcase(os.path.realpath(str(path)))


def _within(path, root):
    root = root.rstrip("\\/")
    return path == root or path.startswith(root + os.sep)


def _under(path, root):
    return _within(_real(path), _real(root))


def _forbidden(path):
    """Real campaign roots, by string first (no handle is opened on them), then resolved."""
    for root in FORBIDDEN_ROOTS + (str(assets.REPO),):
        if _within(_abs(path), _abs(root)) or _within(_abs(root), _abs(path)):
            return root
    for root in FORBIDDEN_ROOTS + (str(assets.REPO),):
        if _within(_real(path), _abs(root)):
            return root
    return None


class Containment(object):
    """Rehearsal guard: every path derived from a fixture campaign must resolve (junctions and
    symlinks followed) inside the fixture root, never under a real campaign/run/worktree root
    or the real repository. Checked before the path is read."""

    def __init__(self, root):
        if not root:
            raise Refused("rehearsal needs --fixture-root")
        hit = _forbidden(root)
        if hit:
            raise Refused("fixture root %s overlaps the real root %s" % (root, hit))
        self.root = _real(root)

    def check(self, path, what):
        if not path:
            raise Refused("%s: no path" % what)
        if any(_within(_abs(path), _abs(r)) for r in FORBIDDEN_ROOTS + (str(assets.REPO),)):
            raise Refused("%s %s is under a real root" % (what, path))
        real = _real(path)
        # Real roots are compared as strings: no handle is ever opened on them.
        hit = next((r for r in FORBIDDEN_ROOTS + (str(assets.REPO),) if _within(real, _abs(r))),
                   None)
        if hit:
            raise Refused("%s %s resolves under the real root %s" % (what, path, hit))
        if not _within(real, self.root):
            raise Refused("%s %s resolves outside the fixture root (%s)" % (what, path, real))
        return real


def _contained_in(path, directory, what):
    if not _under(path, directory):
        raise Refused("%s %s resolves outside its probe directory %s" % (what, path, directory))


def _probe_check(out_dir, result, arm, seed, role, guard=None):
    """Ready-run checks; returns (record, blocking problems). No prediction array is opened.
    With a rehearsal guard, every path the analysis will read is contained first."""
    rec, problems = {"probe_dir": out_dir}, []
    if guard is not None:
        guard.check(out_dir, "probe directory")
    if not out_dir or not os.path.isdir(out_dir):
        return rec, ["probe directory missing: %s" % out_dir]
    files = (result or {}).get("files") or {}
    for key, name in (("aggregate", "results.json"), ("sealed_manifest", "sealed_manifest.json")):
        path = os.path.join(out_dir, name)
        if guard is not None:
            _contained_in(path, out_dir, name)
        recorded = (files.get(key) or {}).get("sha256")
        if not os.path.isfile(path):
            problems.append("%s missing in %s" % (name, out_dir))
        elif recorded and sha256_file(path) != recorded:
            problems.append("%s changed since the campaign validated it: %s" % (name, out_dir))
        elif not recorded:
            problems.append("campaign state records no hash for %s of %s" % (name, out_dir))
    if problems:
        return rec, problems
    if guard is not None:
        for name in ("sealed_manifest.json.sha256", "unsealed_results.json"):
            _contained_in(os.path.join(out_dir, name), out_dir, name)
    try:
        res = cr_stats.load_run(out_dir)
        if guard is not None:
            if res.get("synthetic") is not True:
                raise Refused("%s: results.json is not marked synthetic (rehearsal)" % out_dir)
            ident0 = res.get("identity") or {}
            if ident0.get("run_provenance"):
                guard.check(ident0["run_provenance"], "run provenance (stop file)")
            for split, kinds in (res.get("cache_provenance") or {}).items():
                for kind, rec_ in (kinds or {}).items():
                    if (rec_ or {}).get("path"):
                        guard.check(rec_["path"], "%s %s feature cache" % (split, kind))
        ident = cr_stats._identity(res)
        manifest, manifest_sha = cr_stats.read_seal_manifest(out_dir)
        if guard is not None:
            for entry in manifest.get("files") or []:
                for field in ("path", "sidecar_path"):
                    target = os.path.join(out_dir, str(entry.get(field) or ""))
                    _contained_in(target, out_dir, "sealed %s" % field)
                    _contained_in(os.path.join(os.path.dirname(target), "best_model.pt"),
                                  out_dir, "head checkpoint")
        cr_stats.check_seal_identity(out_dir, ident, manifest)
    except (cr_stats.IntegrityError, OSError, ValueError, KeyError) as exc:
        return rec, [str(exc)]
    if ident.get("arm") != arm or ident.get("role") != role or \
            (role == "new" and ident.get("train_seed") != seed) or ident.get("epoch") != \
            cr_stats.ENDPOINT_EPOCH:
        problems.append("%s: identity %s does not match the registered run (%s, %s, %s)"
                        % (out_dir, {k: ident.get(k) for k in ("arm", "train_seed", "role",
                                                               "epoch")}, arm, seed, role))
    eligibility = cr_stats.completion_eligibility(out_dir, res, manifest)
    receipt = os.path.join(out_dir, "unsealed_results.json")
    unsealed = os.path.isfile(receipt)
    if unsealed:
        try:
            cr_stats.check_receipt(out_dir, manifest_sha)
        except cr_stats.IntegrityError as exc:
            problems.append(str(exc))
    rec.update({"eligibility": eligibility, "late": not eligibility["eligible"],
                "unsealed": unsealed, "sealed_manifest_sha256": manifest_sha,
                "identity": {k: ident.get(k) for k in cr_stats.IDENTITY_KEYS}})
    return rec, problems


def _run_status(train, probe):
    t = (train or {}).get("status")
    p = (probe or {}).get("status")
    if p == "DONE":
        return "ready"
    if t in ("FAILED", "ABANDONED", "STOPPED") or p in ("FAILED", "ABANDONED", "STOPPED"):
        return "not_completed"
    if t in ("RUNNING", "PAUSED") or p in ("RUNNING", "PAUSED") or \
            (t == "DONE" and p in (None, "PENDING")):
        return "in_progress"
    if t in (None, "PENDING") and not (train or {}).get("attempts"):
        return "not_started"
    return "unknown"


def make_plan(campaign_path, state_path, runs_root=None, rehearsal=False, fixture_root=None,
              inputs=()):
    guard = None
    if rehearsal:
        # V8-1: contain the inputs before reading them; explicit boolean synthetic provenance.
        guard = Containment(fixture_root)
        marker = os.path.join(fixture_root, FIXTURE_MARKER)
        guard.check(marker, "fixture marker")
        for path, what in [(campaign_path, "campaign.json"), (state_path, "state.json")] + \
                [(path, "input") for path in inputs]:
            guard.check(path, what)
        if not os.path.isfile(marker) or read_json(marker).get("synthetic") is not True:
            raise Refused("fixture root has no %s with \"synthetic\": true" % FIXTURE_MARKER)
    campaign, state = read_json(campaign_path), read_json(state_path)
    flags = (campaign.get("synthetic"), state.get("synthetic"))
    if rehearsal and flags != (True, True):
        raise Refused("rehearsal needs campaign.json and state.json with \"synthetic\": true "
                      "(found %r)" % (flags,))
    if not rehearsal and any(flag not in (None, False) for flag in flags):
        raise Refused("synthetic campaign given without rehearsal mode")
    synthetic = rehearsal
    if campaign.get("freeze") not in (None, cr_stats.FREEZE):
        raise Refused("campaign freeze %r differs from the registered %s"
                      % (campaign.get("freeze"), cr_stats.FREEZE))
    runs_root = runs_root or campaign.get("run_root")
    if guard is not None:
        guard.check(runs_root, "runs root")
    states = state.get("runs") or {}
    rows, blocking = [], []
    specs = [(spec["id"], cr_stats.norm_arm(spec["arm"]), spec.get("seed"), "new",
              states.get(spec["id"])) for spec in campaign.get("runs") or []]
    items = (((states.get("ANCHORS") or {}).get("probe") or {}).get("items")) or {}
    specs += [("anchor:" + item["name"], cr_stats.norm_arm(item["arm"]), None, "anchor",
               items.get(item["name"])) for item in (campaign.get("anchors") or {}).get("items", [])]
    for rid, arm, seed, role, entry in specs:
        if role == "anchor":
            train, probe = {"status": "DONE"} if entry else None, entry
        else:
            train, probe = (entry or {}).get("train"), (entry or {}).get("probe")
        status = _run_status(train, probe) if entry else "not_started"
        row = {"id": rid, "arm": arm, "seed": seed, "role": role, "status": status,
               "train_status": (train or {}).get("status"),
               "probe_status": (probe or {}).get("status")}
        if guard is not None:
            for attempt in (probe or {}).get("attempts") or []:
                if attempt.get("out_dir"):
                    guard.check(attempt["out_dir"], "%s probe attempt" % rid)
            for attempt in (train or {}).get("attempts") or []:
                if attempt.get("log"):
                    guard.check(attempt["log"], "%s trainer log" % rid)
            for key in ("stop_file", "pinned"):
                if ((train or {}).get(key) or {}).get("path"):
                    guard.check(train[key]["path"], "%s %s" % (rid, key))
        if status == "ready":
            result = (probe or {}).get("result") or {}
            rec, problems = _probe_check(result.get("out_dir"), result, arm, seed, role, guard)
            row.update(rec)
            if problems:
                row["status"] = "invalid"
                row["problems"] = problems
                blocking += ["%s: %s" % (rid, p) for p in problems]
        elif status == "unknown":
            blocking.append("%s: unrecognised campaign state train=%s probe=%s"
                            % (rid, row["train_status"], row["probe_status"]))
        if status == "in_progress":
            row["attempt_dirs"] = [a.get("out_dir") for a in (probe or {}).get("attempts") or []
                                   if a.get("out_dir")]
        if role == "new" and arm == "random_cb":
            row["train_logs"] = [a["log"] for a in (train or {}).get("attempts") or []
                                 if a.get("log") and os.path.isfile(a["log"])]
        rows.append(row)
    ready = [r for r in rows if r["status"] == "ready"]
    ready_dirs = [os.path.realpath(r["probe_dir"]) for r in ready]
    stray = []
    if runs_root:
        for path in sorted(glob.glob(os.path.join(runs_root, "cr_probe_*"))):
            if os.path.isdir(path) and os.path.realpath(path) not in ready_dirs:
                owner = [r["id"] for r in rows if path in (r.get("attempt_dirs") or [])]
                stray.append({"dir": path, "reason": ("in-progress attempt of %s" % owner[0])
                              if owner else "not a validated result of the campaign state"})
    seeds = sorted(set(r["seed"] for r in rows if r["role"] == "new" and r["seed"] is not None))
    index = {}
    for seed in seeds:
        arms = {r["arm"]: ("late" if r.get("late") else r["status"]) for r in rows
                if r["role"] == "new" and r["seed"] == seed}
        index[str(seed)] = {"arms": arms, "primary_on_time": all(
            arms.get(a) == "ready" for a in cr_stats.PRIMARY_ARMS)}
    cb = [r for r in rows if r["arm"] == "random_cb"]
    now = cr_stats._now()
    return {
        "schema": PLAN_SCHEMA, "synthetic": synthetic,
        "generated": now.isoformat(timespec="seconds"), "freeze": cr_stats.FREEZE,
        "after_freeze": now >= datetime.datetime.fromisoformat(cr_stats.FREEZE),
        "campaign": str(Path(campaign_path).resolve()), "state": str(Path(state_path).resolve()),
        "state_sha256": sha256_file(state_path), "runs_root": runs_root,
        "campaign_status": state.get("status"), "current_step": (state.get("current") or {}).get(
            "step"),
        "runs": rows, "ready_dirs": ready_dirs,
        "to_unseal": [d for r, d in zip(ready, ready_dirs) if not r.get("unsealed")],
        "already_unsealed": [d for r, d in zip(ready, ready_dirs) if r.get("unsealed")],
        "late": [r["id"] for r in ready if r.get("late")],
        "in_progress": [r["id"] for r in rows if r["status"] == "in_progress"],
        "not_completed": [r["id"] for r in rows if r["status"] == "not_completed"],
        "not_started": [r["id"] for r in rows if r["status"] == "not_started"],
        "seed_indices": index, "stray_dirs": stray,
        "cb": {"status": cb[0]["status"] if cb else None, "late": bool(cb and cb[0].get("late")),
               "train_logs": cb[0].get("train_logs", []) if cb else []},
        "blocking": blocking}


def print_plan(plan):
    print("%-26s %-9s %-5s %-13s %-6s %-9s %s" % ("run", "arm", "seed", "status", "late",
                                                  "unsealed", "probe dir"))
    for r in plan["runs"]:
        print("%-26s %-9s %-5s %-13s %-6s %-9s %s" % (
            r["id"], r["arm"], r["seed"] or "-", r["status"],
            "LATE" if r.get("late") else "", "yes" if r.get("unsealed") else "",
            os.path.basename(r.get("probe_dir") or "") or "-"))
        if r.get("late"):
            print("    late because: %s" % "; ".join(r["eligibility"]["reasons"]))
    for seed, info in sorted(plan["seed_indices"].items()):
        print("seed index %s: %s" % (seed, "all three primary runs on time" if
                                     info["primary_on_time"] else "INCOMPLETE (descriptive): %s"
                                     % info["arms"]))
    for s in plan["stray_dirs"]:
        print("excluded probe dir %s (%s)" % (s["dir"], s["reason"]))
    for b in plan["blocking"]:
        print("BLOCKING: %s" % b)
    if plan["in_progress"]:
        print("IN PROGRESS at the freeze: %s" % ", ".join(plan["in_progress"]))


def cmd_plan(a):
    plan = make_plan(a.campaign, a.state, a.runs_root, a.rehearsal, a.fixture_root, a.inputs)
    write_json(a.out, plan)
    print_plan(plan)
    if plan["blocking"]:
        return 2
    if plan["in_progress"] and not a.allow_in_progress:
        print("STOP: registered runs are still in progress. Either wait for them (they are then "
              "reported descriptively as finished after the freeze) or re-run with "
              "-AllowIncomplete to report them as not completed (pre-registration section 7).")
        return 3
    if not plan["ready_dirs"]:
        print("STOP: no probe directory is ready")
        return 2
    return 0


# ---------------------------------------------------------------------------
# input-key, scratch-copy, compile-report, summary
# ---------------------------------------------------------------------------

def eligibility_snapshot(run_dir):
    """Canonical completion evidence of one probe dir (V8-2): stop-file path, existence and
    hash, seal and probe times, and the eligibility cr_stats final would compute now."""
    res = cr_stats.load_run(run_dir)
    manifest, _ = cr_stats.read_seal_manifest(run_dir)
    stop = (res.get("identity") or {}).get("run_provenance")
    exists = bool(stop) and os.path.isfile(stop)
    eligibility = cr_stats.completion_eligibility(run_dir, res, manifest)
    return {"stop_file": stop, "stop_exists": exists,
            "stop_sha256": sha256_file(stop) if exists else None,
            "seal_created": manifest.get("created"), "probe_finished": res.get("finished"),
            "eligibility": json.loads(json.dumps(eligibility, default=str))}


def input_key(plan_path, extra):
    """Cache key of final / MD: input file hashes plus the current completion evidence, so a
    changed or missing stop file forces recomputation."""
    plan = read_json(plan_path)
    digest = hashlib.sha256()
    for run_dir in sorted(plan["ready_dirs"]):
        for name in ("results.json", "sealed_manifest.json", "unsealed_results.json"):
            path = os.path.join(run_dir, name)
            digest.update(("%s|%s|%s\n" % (os.path.basename(run_dir), name, sha256_file(path)
                                           if os.path.isfile(path) else "-")).encode("utf-8"))
        digest.update(("%s|eligibility|%s\n" % (os.path.basename(run_dir), json.dumps(
            eligibility_snapshot(run_dir), sort_keys=True))).encode("utf-8"))
    for item in extra or []:
        digest.update(("%s|%s\n" % (item, sha256_file(item) if os.path.isfile(item)
                                    else item)).encode("utf-8"))
    return digest.hexdigest()


def check_final(plan_path, final_path):
    """Plan (current evidence) and final must agree run by run and on the primary indices."""
    plan, final = read_json(plan_path), read_json(final_path)
    problems = []
    runs = final.get("runs") or {}
    ready = {os.path.basename(r["probe_dir"]): r for r in plan["runs"] if r["status"] == "ready"}
    if set(ready) != set(runs):
        problems.append("final runs %s differ from the ready probe dirs %s"
                        % (sorted(runs), sorted(ready)))
    for name, row in ready.items():
        recorded = (runs.get(name) or {}).get("eligibility")
        now = eligibility_snapshot(row["probe_dir"])["eligibility"]
        if json.loads(json.dumps(recorded, default=str)) != now:
            problems.append("%s: final eligibility %s, current evidence %s" % (name, recorded, now))
    complete = ((final.get("analysis") or {}).get(cr_stats.PRIMARY_VARIANT) or {}).get(
        "complete_seed_indices")
    planned = sorted(int(s) for s, info in plan["seed_indices"].items() if info["primary_on_time"])
    if sorted(complete or []) != planned:
        problems.append("final primary seed indices %s, plan %s" % (complete, planned))
    # V10-1: the analysis itself (labels, contrasts, means, RANDOM range, indices) must be what
    # the registered rules give for the receipt-verified per-run AUCs.
    try:
        mk.verify_final(final, bool(plan.get("synthetic")))
        records, _ = mk.verify_runs(final, [r["probe_dir"] for r in ready.values()],
                                    bool(plan.get("synthetic")))
        mk.verify_analysis(final, records)
    except mk.InputError as exc:
        problems.append(str(exc))
    for p in problems:
        print("STALE: %s" % p)
    print("final agrees with the current completion evidence" if not problems else
          "final.json is stale: recompute it (delete final.json / final.key and re-run)")
    return 0 if not problems else 2


def contain(root, paths):
    guard = Containment(root)
    for path in paths:
        guard.check(path, "path")
    print("%d path(s) resolve inside %s" % (len(paths), root))
    return 0


def scratch_copy(src, dst, kind):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if inside(dst, src) or inside(src, dst):
        raise Refused("scratch copy must not overlap its source")
    if inside(dst, assets.REPO):
        raise Refused("scratch copies stay outside the repository: %s" % dst)
    if dst.exists():
        if not (dst / SCRATCH_MARKER).is_file():
            raise Refused("%s exists and is not a scratch copy made by this tool" % dst)
        shutil.rmtree(dst)
    if kind == "paper":
        # Build outputs live at the top level; figure PDFs (used by the poster) are kept.
        def ignore(directory, names):
            top = Path(directory).resolve() == src
            return [n for n in names if n.startswith("_") or (top and n.lower().endswith(
                (".pdf", ".docx", ".zip")))]
    else:
        ignore = shutil.ignore_patterns("*.pdf", "*_preview.png", "build_logs", "_*", "*.log")
    shutil.copytree(src, dst, ignore=ignore)
    (dst / SCRATCH_MARKER).write_text("scratch copy of %s (%s)\n" % (src, kind), encoding="utf-8")
    return dst


def compile_report(paper_dir, stem, out):
    import p13_build_zip as p13
    paper = Path(paper_dir)
    pdf, log_path = paper / (stem + ".pdf"), paper / (stem + ".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    undefined = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined",
                                      log)))
    overfull = [float(x) for x in re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too", log)]
    errors = [line for line in log.splitlines() if line.startswith("!")]
    report = {"pdf": str(pdf), "pdf_exists": pdf.is_file(), "undefined_refs": undefined,
              "rerun_warning": bool(re.search(r"Label\(s\) may have changed", log)),
              "overfull_boxes": len(overfull), "overfull_max_pt": max(overfull or [0.0]),
              "tex_errors": errors[:10]}
    if pdf.is_file():
        info = p13.inspect_pdf(str(pdf), "camera_ready")
        end = info.get("main_text_end")
        report.update({"total_pages": info["total_pages"],
                       "main_content_pages": info["main_content_pages"],
                       "main_text_end": end,
                       "main_text_fraction": round(end[0] - 1 + (end[1] - 72) / 648.0, 2)
                       if end else None, "page_limit": PAGE_LIMIT})
    report["ok"] = bool(report["pdf_exists"] and not undefined and not errors and
                        report.get("main_content_pages") is not None and
                        report["main_content_pages"] <= PAGE_LIMIT)
    write_json(out, report)
    print("pages %s main (limit %d), main text ends at %s; undefined refs %s; overfull %d (max "
          "%.1f pt); TeX errors %d -> %s" % (
              report.get("main_content_pages"), PAGE_LIMIT, report.get("main_text_fraction"),
              undefined or "none", report["overfull_boxes"], report["overfull_max_pt"],
              len(errors), "OK" if report["ok"] else "PROBLEM"))
    return 0 if report["ok"] else 1


def summary(out_root):
    root = Path(out_root)

    def load(rel):
        path = root / rel
        return read_json(path) if path.is_file() else None

    plan, steps = load("plan.json"), load("steps_review.json") or []
    lines = ["# Post-freeze pipeline: human review checkpoint", "",
             "Generated %s. Output root `%s`." % (
                 datetime.datetime.now().astimezone().isoformat(timespec="seconds"), root), ""]
    if plan:
        lines += ["## Campaign", "",
                  "Ready probe dirs: %d; late (descriptive): %s; in progress at the freeze "
                  "(not completed): %s; not completed: %s; not started: %s." % (
                      len(plan["ready_dirs"]), plan["late"] or "none",
                      plan["in_progress"] or "none", plan["not_completed"] or "none",
                      plan["not_started"] or "none"), ""]
        for seed, info in sorted(plan["seed_indices"].items()):
            lines.append("* seed %s: %s" % (seed, "complete on time" if info["primary_on_time"]
                                            else "incomplete %s" % info["arms"]))
        lines.append("")
    results = root / "results_dir.txt"
    sidecar = None
    if results.is_file():
        path = Path(results.read_text(encoding="utf-8-sig").strip()) / "cr_results.values.json"
        sidecar = read_json(path) if path.is_file() else None
    if sidecar:
        s = sidecar["scenario"]
        lines += ["## Generated results%s" % (" (SYNTHETIC)" if sidecar["synthetic"] else ""), "",
                  "Scenario **%s** (CENTROID %s, ENVELOPE %s); matched control **%s**; complete "
                  "seed indices %s, incomplete %s." % (
                      s["scenario"], s["labels"]["centroid"], s["labels"]["envelope"],
                      s["matched_control"]["label"], s["complete_seed_indices"],
                      s["incomplete_seed_indices"]), "",
                  "TL;DR: %s" % sidecar["tldr"], ""]
        if s.get("reference_missing"):
            lines += ["No registered labels and no registered RANDOM range (V10-2): %s; the "
                      "matched control is descriptive (%s)." % (
                          ", ".join(s["reference_missing"]),
                          s["matched_control"].get("reason")), ""]
        if sidecar.get("warnings"):
            lines += ["Display warnings (check before release):", ""] + \
                ["* %s" % w for w in sidecar["warnings"]] + [""]
    for rel, title in (("compile_scratch.json", "Scratch paper compile"),):
        report = load(rel)
        if report:
            lines += ["## %s" % title, "",
                      "%s main pages (limit %d), main text ends at %s, %s total; undefined refs: "
                      "%s; overfull boxes: %d (max %.1f pt); TeX errors: %d." % (
                          report.get("main_content_pages"), PAGE_LIMIT,
                          report.get("main_text_fraction"), report.get("total_pages"),
                          report["undefined_refs"] or "none", report["overfull_boxes"],
                          report["overfull_max_pt"], len(report["tex_errors"])), ""]
    preview = load("receipts_preview.json")
    if preview:
        lines += ["## Receipt preview (scratch copy, before migration)", "",
                  "Stale receipts %d; migratable by the integration mapping %d; manual (human "
                  "acceptance) %d." % (preview["n_stale"], len(preview["migrations"]),
                                       len(preview["manual"])), ""]
    poster = load("poster_report.json")
    if poster:
        lines += ["## Poster (scratch copy)", "", "build exit code %s; %s" % (
            poster.get("exit_code"), poster.get("note", "")), ""]
    if steps:
        lines += ["## Steps", "", "| step | seconds | outcome |", "|---|---:|---|"]
        lines += ["| %s | %.1f | %s |" % (s["step"], s["seconds"], s["outcome"]) for s in steps]
        lines.append("")
    text = "\n".join(lines) + "\n"
    (root / "REVIEW.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------------------
# rehearsal-only helpers (refuse the real repository)
# ---------------------------------------------------------------------------

def _git_files(repo):
    out = []
    for args in (["ls-files", "-z"], ["ls-files", "-z", "--others", "--exclude-standard"]):
        proc = subprocess.run(["git", "-C", str(repo)] + args, capture_output=True, check=True)
        out += [p for p in proc.stdout.decode("utf-8").split("\0") if p]
    return sorted(set(out))


def rehearsal_mirror(repo, dest):
    """Copy of the working tree (tracked + untracked, non-ignored files) standing in for the
    real repository: its paper, registry and review file absorb the rehearsal's edits."""
    repo, dest = Path(repo).resolve(), Path(dest).resolve()
    if inside(dest, repo) or inside(repo, dest) or str(dest).lower().startswith("d:\\jepa_phase0"):
        raise Refused("mirror must be outside the repository and D:\\jepa_phase0")
    if dest.exists():
        if not (dest / MIRROR_MARKER).is_file():
            raise Refused("%s exists and is not a rehearsal mirror" % dest)
        shutil.rmtree(dest)
    files = [f for f in _git_files(repo) if not f.startswith((".v4/", ".git/"))]
    # Evidence the gates read that git ignores (e.g. licence/readme snapshots): every repo-rooted
    # review source and every registered producer input / evidence manifest.
    reviews = read_json(repo / "paper" / "genai4health2026" / "numeric_reviews.json")
    registry = read_json(repo / "autopilot" / "release_assets.json")
    extra = [spec["path"] for spec in reviews.get("sources", {}).values()
             if spec.get("root") == "repo"]
    for spec in registry.get("producers", {}).values():
        extra += list(spec.get("inputs") or []) + [spec.get("evidence_manifest") or "",
                                                   spec.get("script") or ""]
    have = set(f.replace("\\", "/") for f in files)
    for rel in extra:
        rel = rel.replace("\\", "/").strip("/")
        if rel and rel not in have and (repo / rel).is_file() and inside(repo / rel, repo):
            files.append(rel)
            have.add(rel)
    for rel in files:
        src = repo / rel
        if src.is_file():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    tools = {name: sha256_file(repo / "autopilot" / name) for name in (
        "p13_build_zip.py", "p15_verify_numbers.py", "release_assets.py", "numeric_bindings.py",
        "make_cr_results.py", "cr_stats.py")}
    write_json(dest / MIRROR_MARKER, {"rehearsal_mirror_of": str(repo), "files": len(files),
                                      "created": cr_stats._now().isoformat(timespec="seconds"),
                                      "tool_sha256": tools,
                                      "note": "SYNTHETIC REHEARSAL ONLY; delete after use"})
    print("mirror of %d files at %s" % (len(files), dest))
    return 0


def _require_mirror(path):
    path = Path(path).resolve()
    real = assets.REPO.resolve()
    if inside(path, real):
        raise Refused("rehearsal helpers never write inside the real repository: %s" % path)
    root = next((p for p in [path] + list(path.parents) if (p / MIRROR_MARKER).is_file()), None)
    if root is None:
        raise Refused("%s is not inside a rehearsal mirror" % path)
    return root


def rehearsal_bind(sidecar_path, review_file, repo_root):
    """Rep* bindings for SYNTHETIC results (the role of build --update-reviews, which refuses
    synthetic inputs): rep_* sources rooted in the mirror repository."""
    mirror = _require_mirror(review_file)
    repo_root = Path(repo_root).resolve()
    if repo_root != mirror:
        raise Refused("--repo-root must be the rehearsal mirror %s" % mirror)
    sidecar = read_json(sidecar_path)
    if sidecar.get("synthetic") is not True:
        raise Refused("rehearsal bindings are for synthetic results only")
    raw = Path(review_file).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    data["sources"] = {k: v for k, v in data.get("sources", {}).items() if not k.startswith("rep_")}
    for name, spec in sidecar["inputs"].items():
        staged = Path(spec["staged"]).resolve()
        if not inside(staged, mirror) or sha256_file(staged) != spec["sha256"]:
            raise Refused("staged input %s is not unchanged inside the mirror" % name)
        data["sources"][name] = {"root": "repo", "path": staged.relative_to(mirror).as_posix(),
                                 "sha256": spec["sha256"]}
    data["macros"] = {k: v for k, v in data.get("macros", {}).items() if not k.startswith("Rep")}
    for record in sidecar["macros"]:
        data["macros"][record["name"]] = {"expression": record["expression"]}
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    Path(review_file).write_bytes((text.replace("\n", "\r\n") if b"\r\n" in raw
                                   else text).encode("utf-8"))
    print("bound %d Rep* macros and %d rep_* sources in %s" % (
        len(sidecar["macros"]), len(sidecar["inputs"]), review_file))
    return 0


ACK_DEFINITION = re.compile(r"\\newcommand\{\\CRAcknowledgmentsPending\}\{\\textcolor\{red\}\{"
                            r"ACKNOWLEDGMENTS PENDING:\s+acknowledgments and funding disclosure "
                            r"to be supplied by the authors\.\}\}\n")


def rehearsal_ack(paper_dir):
    """Stand-in acknowledgment so the remaining camera-ready gates can be exercised."""
    _require_mirror(paper_dir)
    path = Path(paper_dir) / "main_submission.tex"
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if len(ACK_DEFINITION.findall(text)) != 1:
        raise Refused("acknowledgment placeholder definition not found exactly once")
    text = ACK_DEFINITION.sub("", text)
    usage = re.findall(r"^\\CRAcknowledgmentsPending\s*$", text, re.M)
    if len(usage) != 1:
        raise Refused("acknowledgment placeholder use not found exactly once")
    text = re.sub(r"^\\CRAcknowledgmentsPending\s*$",
                  "Synthetic rehearsal acknowledgment (stand-in; not the authors' text).", text,
                  count=1, flags=re.M)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("rehearsal acknowledgment written in %s" % path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--campaign", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--runs-root")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-in-progress", action="store_true")
    p.add_argument("--rehearsal", action="store_true")
    p.add_argument("--fixture-root", help="rehearsal: every derived path must resolve inside it")
    p.add_argument("--inputs", nargs="*", default=[],
                   help="rehearsal: further inputs that must resolve inside the fixture root")
    p = sub.add_parser("input-key")
    p.add_argument("--plan", required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p = sub.add_parser("check-final")
    p.add_argument("--plan", required=True)
    p.add_argument("--final", required=True)
    p = sub.add_parser("contain")
    p.add_argument("--root", required=True)
    p.add_argument("--paths", nargs="+", required=True)
    p = sub.add_parser("scratch-copy")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--kind", choices=("paper", "poster"), default="paper")
    p = sub.add_parser("compile-report")
    p.add_argument("--paper-dir", required=True)
    p.add_argument("--stem", default="main_submission")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--out-root", required=True)
    p = sub.add_parser("rehearsal-mirror")
    p.add_argument("--repo", default=str(assets.REPO))
    p.add_argument("--dest", required=True)
    p = sub.add_parser("rehearsal-bind")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--review-file", required=True)
    p.add_argument("--repo-root", required=True)
    p = sub.add_parser("rehearsal-ack")
    p.add_argument("--paper-dir", required=True)
    a = ap.parse_args(argv)
    try:
        if a.command == "plan":
            return cmd_plan(a)
        if a.command == "input-key":
            print(input_key(a.plan, a.extra))
            return 0
        if a.command == "check-final":
            return check_final(a.plan, a.final)
        if a.command == "contain":
            return contain(a.root, a.paths)
        if a.command == "scratch-copy":
            print("scratch copy: %s" % scratch_copy(a.src, a.dst, a.kind))
            return 0
        if a.command == "compile-report":
            return compile_report(a.paper_dir, a.stem, a.out)
        if a.command == "summary":
            return summary(a.out_root)
        if a.command == "rehearsal-mirror":
            return rehearsal_mirror(a.repo, a.dest)
        if a.command == "rehearsal-bind":
            return rehearsal_bind(a.sidecar, a.review_file, a.repo_root)
        if a.command == "rehearsal-ack":
            return rehearsal_ack(a.paper_dir)
    except (Refused, cr_stats.IntegrityError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
