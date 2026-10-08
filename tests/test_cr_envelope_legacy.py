"""ENVELOPE overlap-fallback sampler version (camera-ready task E2).

``fc49f61`` (2026-08-08) changed what ``mirage_envelope`` does when no
admissible window clears ``mirage_overlap_tolerance``: it now picks among the
least-overlapping windows, whereas the sampler that trained the original
ENVELOPE run (finished 2026-08-04, run-era candidate ``804c639``) picked
uniformly among all admissible windows.  The behaviour is now selected by
``mask.curriculum.mirage_overlap_fallback``:

  * ``least_overlap_v2``  -- current behaviour, the DEFAULT;
  * ``legacy_uniform_v1`` -- bit-identical to ``804c639``.

The golden hashes below were produced by the reference sources themselves
(``git show 804c639:src/masks/curriculum.py`` for v1 and the pre-change HEAD
``86ae882`` for v2) on the fixture in this file; see
autopilot/investigations/camera_ready_20261008/envelope_legacy/ for the
generator script and the larger real-slice certification.
"""

import ast
import hashlib
import importlib.util
import os
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.masks.curriculum import (  # noqa: E402
    DEFAULT_MIRAGE_OVERLAP_FALLBACK,
    MIRAGE_OVERLAP_FALLBACKS,
    CurriculumMaskGenerator,
    MirageMaskCollator,
)

TRAIN_PATCH = os.path.join(REPO_ROOT, "src", "train_patch.py")
MIRAGE_CONFIG = os.path.join(REPO_ROOT, "configs", "patch_mirage_envelope.yaml")

# Production ENVELOPE curriculum (configs/patch_mirage_envelope.yaml).
ENVELOPE_CURRICULUM = dict(
    enabled=True, mode="mirage_envelope", T_warm=25, T_total=30, r_max=1.0,
    ramp_shape="linear", mirage_guide_dir="unused", mirage_dilate_patches=0,
    mirage_min_block_fill=0.40, mirage_min_retina_visible=0.25,
    mirage_max_attempts=30, mirage_occupancy_threshold=0.25,
    mirage_spread=True, mirage_overlap_tolerance=0.25,
)
COMMON = dict(
    input_size=(256, 256), patch_size=16, enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
    min_keep=10, allow_overlap=False,
)
GOLDEN_SEEDS = (0, 1234)
GOLDEN_EPOCHS = (27, 40)   # 0-based: r_t 0.4 (ramp) and 1.0 (full guidance)
GOLDEN_BATCHES = 2
B = 32

# sha256 over every delivered enc/pred tensor of GOLDEN_BATCHES consecutive
# collator calls, keyed "seed:epoch".
GOLDEN = {
    "legacy_uniform_v1": {  # == git show 804c639:src/masks/curriculum.py
        "0:27": "842d70e43a554f178bbe0278db63da1322bdb055544817e5df026ce872635710",
        "0:40": "56e5445aff62b18c19ae1a9940866ac6f6fd8ffa1b8227294d1a1032661af902",
        "1234:27": "a2b4fff7f1e04415cd83ead6e6fb3c5805560c8593ec00ad5eb4793cd70c5b3b",
        "1234:40": "e42b4c4c5fc9bf73472598f3d6aef36cb1477e07610728deb36ce5ad7f26d46e",
    },
    "least_overlap_v2": {  # == git show 86ae882:src/masks/curriculum.py
        "0:27": "4669c675fbc22f568d4a5809981d600ac07aeed1596c28cea5f68dc4e26517bd",
        "0:40": "54ec7b849c9a89205820ecb6439b38f34318e0894240efa8bcd4bcc0ec5ca4a7",
        "1234:27": "22531b1530e929441bfa8a0605badb7d5f7445a2e30029e175af2487d9e2062d",
        "1234:40": "3fc2d443d9c83b3f9a69dcf9a5c612fad01e9fd71663afabeab481e06c4ed973",
    },
}


def curriculum(version=None):
    cfg = dict(ENVELOPE_CURRICULUM)
    if version is not None:
        cfg["mirage_overlap_fallback"] = version
    return cfg


def golden_guides():
    """Thin tilted bands (overlap tolerance usually infeasible), plus invalid,
    empty and too-small guides."""
    rs = np.random.RandomState(654)
    g = np.zeros((B, 16, 16), np.float32)
    for b in range(B):
        centre = rs.uniform(4, 12)
        tilt = rs.uniform(-0.2, 0.2)
        thick = rs.uniform(0.3, 2.5)
        for col in range(16):
            ctr = centre + tilt * (col - 8)
            for row in range(16):
                g[b, row, col] = float(
                    np.clip(1.0 - (abs(row - ctr) - thick / 2.0), 0.0, 1.0)
                )
    occ = torch.from_numpy(g)
    guides = torch.stack([occ, (occ >= 0.25).float()], dim=1)
    valid = torch.ones(B, dtype=torch.bool)
    valid[::9] = False
    guides[4::13] = 0.0
    tiny = torch.zeros(16, 16)
    tiny[7:9, 7:9] = 1.0
    guides[6::11, 0] = tiny
    guides[6::11, 1] = tiny
    return guides, valid


def collator_batch(guides, valid):
    image = torch.zeros(3, 1, 1)
    return [(image, guides[i], valid[i]) for i in range(guides.shape[0])]


def masks_digest(module, kwargs, seed, epoch):
    """Run the collator from a fixed RNG state; hash every delivered mask."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    collator = module.MirageMaskCollator(**kwargs)
    collator.set_epoch(epoch, 100)
    guides, valid = golden_guides()
    digest = hashlib.sha256()
    for _ in range(GOLDEN_BATCHES):
        _, enc, pred, _ = collator(collator_batch(guides, valid))
        for t in list(enc) + list(pred):
            arr = t.contiguous().numpy()
            digest.update(str(arr.dtype).encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
    return digest.hexdigest()


def current_module():
    import src.masks.curriculum as module

    return module


def new_kwargs(version):
    return dict(COMMON, curriculum_cfg=curriculum(version))


# ----------------------------------------------------------------------
# Contract: values, default, validation
# ----------------------------------------------------------------------


def test_versions_and_default():
    assert set(MIRAGE_OVERLAP_FALLBACKS) == {"legacy_uniform_v1", "least_overlap_v2"}
    assert DEFAULT_MIRAGE_OVERLAP_FALLBACK == "least_overlap_v2"
    gen = CurriculumMaskGenerator(**COMMON, curriculum_cfg=curriculum())
    assert gen.mirage_overlap_fallback == "least_overlap_v2"
    assert MirageMaskCollator(**new_kwargs(None)).mirage_overlap_fallback == (
        "least_overlap_v2"
    )


@pytest.mark.parametrize("version", ["legacy_uniform_v1", "least_overlap_v2"])
def test_known_values_accepted(version):
    gen = CurriculumMaskGenerator(**COMMON, curriculum_cfg=curriculum(version))
    assert gen.mirage_overlap_fallback == version
    assert MirageMaskCollator(**new_kwargs(version)).mirage_overlap_fallback == version


@pytest.mark.parametrize(
    "bad", ["legacy", "least_overlap", "LEGACY_UNIFORM_V1", "", None, 1]
)
def test_unknown_value_raises(bad):
    cfg = dict(ENVELOPE_CURRICULUM, mirage_overlap_fallback=bad)
    with pytest.raises(ValueError, match="mirage_overlap_fallback"):
        CurriculumMaskGenerator(**COMMON, curriculum_cfg=cfg)
    # The collator builds its generator lazily inside DataLoader workers, so it
    # must reject a bad value eagerly, at construction in the main process.
    with pytest.raises(ValueError, match="mirage_overlap_fallback"):
        MirageMaskCollator(**COMMON, curriculum_cfg=cfg)


def test_existing_configs_keep_current_default():
    """No shipped config may silently change sampler; any explicit value is valid."""
    with open(MIRAGE_CONFIG, "r", encoding="utf-8") as handle:
        curr = yaml.safe_load(handle)["mask"]["curriculum"]
    assert "mirage_overlap_fallback" not in curr
    for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, "configs")):
        for name in files:
            if not name.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(root, name), "r", encoding="utf-8") as handle:
                try:
                    cfg = yaml.safe_load(handle)
                except yaml.YAMLError:
                    continue
            curr = ((cfg or {}).get("mask") or {}).get("curriculum") or {}
            if "mirage_overlap_fallback" in curr:
                assert curr["mirage_overlap_fallback"] in MIRAGE_OVERLAP_FALLBACKS, name


# ----------------------------------------------------------------------
# Golden masks
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", GOLDEN_SEEDS)
@pytest.mark.parametrize("epoch", GOLDEN_EPOCHS)
@pytest.mark.parametrize("version", ["legacy_uniform_v1", "least_overlap_v2"])
def test_golden_hashes(version, seed, epoch):
    got = masks_digest(current_module(), new_kwargs(version), seed, epoch)
    assert got == GOLDEN[version]["%d:%d" % (seed, epoch)]


@pytest.mark.parametrize("seed", GOLDEN_SEEDS)
@pytest.mark.parametrize("epoch", GOLDEN_EPOCHS)
def test_default_hash_equals_least_overlap_v2(seed, epoch):
    got = masks_digest(current_module(), new_kwargs(None), seed, epoch)
    assert got == GOLDEN["least_overlap_v2"]["%d:%d" % (seed, epoch)]


def test_versions_differ_on_overlap_infeasible_fixture():
    """The fixture really exercises the fallback, so the goldens discriminate."""
    for seed in GOLDEN_SEEDS:
        for epoch in GOLDEN_EPOCHS:
            key = "%d:%d" % (seed, epoch)
            assert GOLDEN["legacy_uniform_v1"][key] != GOLDEN["least_overlap_v2"][key]


def test_ramp_off_versions_identical():
    """At r_t = 0 no block is guided, so the flag cannot matter."""
    assert masks_digest(current_module(), new_kwargs("legacy_uniform_v1"), 0, 25) == (
        masks_digest(current_module(), new_kwargs("least_overlap_v2"), 0, 25)
    )


def _load_git_source(commit, name):
    try:
        blob = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "show", "%s:src/masks/curriculum.py" % commit],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history for %s unavailable" % commit)
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob.decode("utf-8"), "%s:curriculum.py" % commit, "exec"),
         module.__dict__)
    return module


@pytest.mark.parametrize(
    "commit,version",
    [("804c639", "legacy_uniform_v1"), ("86ae882", "least_overlap_v2")],
)
def test_reference_sources_reproduce_goldens(commit, version):
    """Re-derive the goldens from the reference source (skips without git)."""
    module = _load_git_source(commit, "_curriculum_ref_%s" % commit)
    kwargs = dict(COMMON, curriculum_cfg=curriculum())
    for seed in GOLDEN_SEEDS:
        for epoch in GOLDEN_EPOCHS:
            assert masks_digest(module, kwargs, seed, epoch) == (
                GOLDEN[version]["%d:%d" % (seed, epoch)]
            )


# ----------------------------------------------------------------------
# YAML -> trainer -> collator -> worker wiring
# ----------------------------------------------------------------------


def _train_patch_tree():
    with open(TRAIN_PATCH, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=TRAIN_PATCH)


def test_train_patch_forwards_whole_curriculum_dict():
    """train_patch.py must hand mask.curriculum to the collator unfiltered,
    which is what carries mirage_overlap_fallback without a dedicated line."""
    tree = _train_patch_tree()
    assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "curr_cfg" for t in node.targets)
    ]
    assert assigns, "curr_cfg assignment not found"
    assert any("'curriculum'" in ast.unparse(a.value) or '"curriculum"' in ast.unparse(a.value)
               for a in assigns)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in ("MirageMaskCollator", "CurriculumMaskGenerator")
    ]
    assert {c.func.id for c in calls} == {"MirageMaskCollator", "CurriculumMaskGenerator"}
    for call in calls:
        kw = [k for k in call.keywords if k.arg == "curriculum_cfg"]
        assert kw and isinstance(kw[0].value, ast.Name) and kw[0].value.id == "curr_cfg", (
            "%s must receive curriculum_cfg=curr_cfg" % call.func.id
        )


@pytest.mark.parametrize("version", ["legacy_uniform_v1", "least_overlap_v2"])
def test_yaml_key_reaches_worker_generator(version):
    """Production YAML + the key -> collator (as train_patch builds it) ->
    pickled worker copy -> generator."""
    with open(MIRAGE_CONFIG, "r", encoding="utf-8") as handle:
        text = handle.read()
    anchor = "    mirage_overlap_tolerance: 0.25\n"
    assert text.count(anchor) == 1
    cfg = yaml.safe_load(
        text.replace(anchor, anchor + "    mirage_overlap_fallback: %s\n" % version)
    )
    mask_cfg = cfg["mask"]
    curr_cfg = mask_cfg.get("curriculum", {}) or {}   # as in train_patch.py
    crop_size = cfg["data"]["crop_size"]
    collator = MirageMaskCollator(
        input_size=(crop_size, crop_size),
        patch_size=mask_cfg["patch_size"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        allow_overlap=mask_cfg["allow_overlap"],
        pred_target_k=mask_cfg.get("pred_target_k"),
        curriculum_cfg=curr_cfg,
    )
    assert collator.mirage_overlap_fallback == version
    worker_copy = pickle.loads(pickle.dumps(collator))
    worker_copy.set_epoch(40, cfg["optimization"]["epochs"])
    assert worker_copy._get_generator().mirage_overlap_fallback == version
    # The YAML-built worker collator emits the golden masks for its version.
    guides, valid = golden_guides()
    expected = masks_digest(current_module(), new_kwargs(version), 0, 40)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    fresh = pickle.loads(pickle.dumps(collator))
    fresh.set_epoch(40, 100)
    digest = hashlib.sha256()
    for _ in range(GOLDEN_BATCHES):
        _, enc, pred, _ = fresh(collator_batch(guides, valid))
        for t in list(enc) + list(pred):
            arr = t.contiguous().numpy()
            digest.update(str(arr.dtype).encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
    assert digest.hexdigest() == expected == GOLDEN[version]["0:40"]
