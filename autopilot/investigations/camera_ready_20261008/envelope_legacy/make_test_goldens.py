"""Derive the golden hashes in tests/test_cr_envelope_legacy.py from the
reference sources (804c639 -> legacy_uniform_v1, 86ae882 -> least_overlap_v2)
and cross-check them against the working tree.

Usage: python make_test_goldens.py OUT.json
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.REPO, "tests"))
import test_cr_envelope_legacy as T  # noqa: E402

OUT = sys.argv[1]
refs = {
    "legacy_uniform_v1": C.extract_ref(C.RUN_ERA_COMMIT),
    "least_overlap_v2": C.extract_ref(C.PRE_CHANGE_HEAD),
}
golden, check = {}, {}
for version, (path, sha) in refs.items():
    module = C.load_module(path, "ref_" + version)
    kwargs = dict(T.COMMON, curriculum_cfg=T.curriculum())
    golden[version] = {}
    check[version] = {}
    for seed in T.GOLDEN_SEEDS:
        for epoch in T.GOLDEN_EPOCHS:
            key = "%d:%d" % (seed, epoch)
            ref = T.masks_digest(module, kwargs, seed, epoch)
            new = T.masks_digest(T.current_module(), T.new_kwargs(version), seed, epoch)
            golden[version][key] = ref
            check[version][key] = new == ref
report = {
    "command": "python make_test_goldens.py " + OUT,
    "refs": {v: {"path": os.path.relpath(p, C.HERE), "sha256": s} for v, (p, s) in refs.items()},
    "golden": golden, "working_tree_matches": check,
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1)
print(json.dumps(report, indent=1))
