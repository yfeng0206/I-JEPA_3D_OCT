"""E2 golden-mask certification: legacy_uniform_v1 == 804c639, least_overlap_v2 == HEAD.

For every fixture x seed x epoch, the reference collator and the working-tree
collator are run from identical Python/NumPy/torch RNG state on identical
guide grids, NB consecutive batches each (the generator persists across
batches as it does inside a DataLoader worker).  Every delivered encoder and
predictor mask tensor is hashed, plus the numeric [MIRAGE] statistics.

Implementations compared:
  ref_804c639     git show 804c639:src/masks/curriculum.py (ENVELOPE run era)
  ref_head        git show 86ae882:src/masks/curriculum.py (pre-change HEAD)
  new_v1          working tree, mirage_overlap_fallback: legacy_uniform_v1
  new_v2          working tree, mirage_overlap_fallback: least_overlap_v2
  new_default     working tree, key absent
  instr_v1/_v2    instrumented copy of the working tree (counts overlap-
                  infeasible events; must hash-equal new_v1/new_v2)

Usage: python golden_masks.py OUT.json [--skip-synthetic] [real_guides.npz ...]
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

import src.masks.curriculum as NEW  # noqa: E402

torch.set_num_threads(2)
SKIP_SYNTH = "--skip-synthetic" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--skip-synthetic"]
OUT = ARGS[0]
REAL = ARGS[1:]

ref804_path, ref804_sha = C.extract_ref(C.RUN_ERA_COMMIT)
refhead_path, refhead_sha = C.extract_ref(C.PRE_CHANGE_HEAD)
REF804 = C.load_module(ref804_path, "curriculum_ref_804c639")
REFHEAD = C.load_module(refhead_path, "curriculum_ref_head")
INSTR = C.load_module(NEW.__file__, "curriculum_instrumented", patch=C.instrument)

cfg = C.load_prod_yaml()
base_curr = dict(cfg["mask"]["curriculum"])
base_curr.pop("mirage_overlap_fallback", None)


def curr_with(version):
    out = dict(base_curr)
    if version is not None:
        out["mirage_overlap_fallback"] = version
    return out


IMPLS = {
    "ref_804c639": (REF804, None, True),
    "new_v1": (NEW, "legacy_uniform_v1", False),
    "instr_v1": (INSTR, "legacy_uniform_v1", False),
    "ref_head": (REFHEAD, None, False),
    "new_v2": (NEW, "least_overlap_v2", False),
    "new_default": (NEW, None, False),
    "instr_v2": (INSTR, "least_overlap_v2", False),
}
PAIRS = [
    ("ref_804c639", "new_v1"),
    ("new_v1", "instr_v1"),
    ("ref_head", "new_v2"),
    ("ref_head", "new_default"),
    ("new_v2", "instr_v2"),
    ("ref_804c639", "ref_head"),   # expected to DIFFER on guided batches
    ("new_v1", "new_v2"),          # expected to DIFFER on guided batches
]


def run(impl_name, batches, seed, epoch):
    module, version, legacy = IMPLS[impl_name]
    kwargs = C.collator_kwargs(cfg, curr_with(version))
    if legacy:
        kwargs = C.legacy_kwargs(kwargs)
    if module is INSTR:
        INSTR.EVENTS.update(overlap_infeasible=0, least_strict_subset=0)
    C.seed_all(seed)
    collator = module.MirageMaskCollator(**kwargs)
    collator.set_epoch(epoch, cfg["optimization"]["epochs"])
    rows = []
    for guides, valid in batches:
        _, enc, pred, ms = collator(C.make_batch(guides, valid))
        rows.append({
            "enc": C.hash_tensors(enc),
            "pred": C.hash_tensors(pred),
            "stats": C.numeric_stats(ms),
            "r_t": float(collator._generator._r_t),
        })
    events = dict(INSTR.EVENTS) if module is INSTR else None
    return rows, events


def compare(a_rows, b_rows):
    tensors = differ = stats_differ = 0
    for a, b in zip(a_rows, b_rows):
        for key in ("enc", "pred"):
            tensors += 1
            differ += int(a[key] != b[key])
        common = set(a["stats"]) & set(b["stats"])
        stats_differ += int(any(a["stats"][k] != b["stats"][k] for k in common))
    return {"hash_groups": tensors, "differ": differ, "stats_batches_differ": stats_differ}


def suite(name, batches_by_fixture, seeds, epochs):
    results = {"cases": [], "totals": {}}
    totals = {"%s|%s" % p: {"hash_groups": 0, "differ": 0, "stats_batches_differ": 0}
              for p in PAIRS}
    events_tot = {"instr_v1": {"overlap_infeasible": 0, "least_strict_subset": 0},
                  "instr_v2": {"overlap_infeasible": 0, "least_strict_subset": 0}}
    for fixture, batches in batches_by_fixture.items():
        for seed in seeds:
            for epoch in epochs:
                outs, events = {}, {}
                for impl in IMPLS:
                    outs[impl], ev = run(impl, batches, seed, epoch)
                    if ev is not None:
                        events[impl] = ev
                        for k, v in ev.items():
                            events_tot[impl][k] += v
                case = {
                    "fixture": fixture, "seed": seed, "epoch0": epoch,
                    "r_t": outs["new_v1"][0]["r_t"], "batches": len(batches),
                    "events": events, "pairs": {},
                    "golden_v1_pred_hashes": [r["pred"] for r in outs["new_v1"]],
                    "golden_v2_pred_hashes": [r["pred"] for r in outs["new_v2"]],
                }
                for a, b in PAIRS:
                    cmp_ = compare(outs[a], outs[b])
                    case["pairs"]["%s|%s" % (a, b)] = cmp_
                    for k, v in cmp_.items():
                        totals["%s|%s" % (a, b)][k] += v
                results["cases"].append(case)
                print("%-10s %-20s seed %-5d ep %-3d r_t %.1f  804vs_v1 %d  head_vs_v2 %d  "
                      "head_vs_default %d  804_vs_head %d  events v1 %s"
                      % (name, fixture, seed, epoch, case["r_t"],
                         case["pairs"]["ref_804c639|new_v1"]["differ"],
                         case["pairs"]["ref_head|new_v2"]["differ"],
                         case["pairs"]["ref_head|new_default"]["differ"],
                         case["pairs"]["ref_804c639|ref_head"]["differ"],
                         events.get("instr_v1")), flush=True)
    results["totals"] = totals
    results["overlap_events"] = events_tot
    return results


t0 = time.time()
report = {
    "purpose": "E2 golden-mask certification of mask.curriculum.mirage_overlap_fallback",
    "command": "python " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    "cwd": os.getcwd(),
    "refs": {
        "804c639": {"path": os.path.relpath(ref804_path, C.HERE), "sha256": ref804_sha},
        C.PRE_CHANGE_HEAD: {"path": os.path.relpath(refhead_path, C.HERE),
                            "sha256": refhead_sha},
        "working_tree": {"path": os.path.relpath(NEW.__file__, C.REPO),
                         "sha256": C.sha256_file(NEW.__file__)},
    },
    "config": {"yaml": os.path.relpath(C.PROD_YAML, C.REPO), "curriculum": base_curr},
    "torch": torch.__version__, "numpy": np.__version__,
}

synth = {name: [fx] * 4 for name, fx in C.synth_fixtures(64).items()}
if not SKIP_SYNTH:
    report["synthetic"] = suite("synthetic", synth, seeds=[0, 7, 42, 1234],
                                epochs=[25, 26, 27, 29, 30, 40])

for path in REAL:
    d = np.load(path)
    G = torch.from_numpy(d["guides"]).float()
    V = torch.from_numpy(d["valid"]).bool()
    n = (G.shape[0] // 64) * 64
    batches = [(G[i:i + 64], V[i:i + 64]) for i in range(0, min(n, 1280), 64)]
    key = "real:" + os.path.basename(path)
    report[key] = suite("real", {os.path.basename(path): batches}, seeds=[0, 1, 2],
                        epochs=[26, 27, 30, 40])
    report[key]["input"] = {"path": path, "sha256": C.sha256_file(path),
                            "views_used": len(batches) * 64,
                            "valid": int(V[: len(batches) * 64].sum())}

report["elapsed_s"] = round(time.time() - t0, 1)
verdict = {}
for section in [k for k in report if k == "synthetic" or k.startswith("real:")]:
    tot = report[section]["totals"]
    verdict[section] = {
        "legacy_uniform_v1_equals_804c639": tot["ref_804c639|new_v1"]["differ"] == 0
        and tot["ref_804c639|new_v1"]["stats_batches_differ"] == 0,
        "least_overlap_v2_equals_HEAD": tot["ref_head|new_v2"]["differ"] == 0
        and tot["ref_head|new_v2"]["stats_batches_differ"] == 0,
        "default_equals_HEAD": tot["ref_head|new_default"]["differ"] == 0,
        "instrumentation_is_inert": tot["new_v1|instr_v1"]["differ"] == 0
        and tot["new_v2|instr_v2"]["differ"] == 0,
        "fixtures_exercise_fallback": report[section]["overlap_events"]["instr_v1"]["least_strict_subset"] > 0,
        "804c639_differs_from_HEAD_tensors": tot["ref_804c639|ref_head"]["differ"],
    }
report["verdict"] = verdict
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1)
print(json.dumps(verdict, indent=1))
print("wrote", OUT, "%.1fs" % report["elapsed_s"])
