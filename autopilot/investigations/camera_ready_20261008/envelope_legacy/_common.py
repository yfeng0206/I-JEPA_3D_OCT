"""Shared helpers for the E2 ENVELOPE sampler-version certification (CPU only).

Reference implementations are extracted from git into ``refs/`` and loaded
under private module names, so the 804c639 (run-era) sampler, the pre-change
HEAD sampler and the working-tree sampler can run side by side in one process
on identical inputs and RNG state.
"""
import hashlib
import importlib.util
import os
import random
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

REFS = os.path.join(HERE, "refs")
RUN_ERA_COMMIT = "804c639"   # ENVELOPE run era (resumed run 07-31 .. 08-04)
FIXA_COMMIT = "fc49f61"      # least-overlap fallback ("FIX A"), 08-08
PRE_CHANGE_HEAD = "86ae882"  # HEAD before this change (curriculum.py == 73e0b55)
PROD_YAML = os.path.join(REPO, "configs", "patch_mirage_envelope.yaml")
GUIDE_DIR = r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides"
DATA_DIR = r"D:\jepa_phase0\fairvision-glaucoma\data"
SLICE_CACHE = r"D:\jepa_phase0\fairvision-glaucoma\slice_cache"


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def extract_ref(commit, relpath="src/masks/curriculum.py"):
    """``git show commit:relpath`` into refs/, returning (path, sha256)."""
    os.makedirs(REFS, exist_ok=True)
    out = os.path.join(REFS, "curriculum_%s.py" % commit)
    blob = subprocess.check_output(
        ["git", "-C", REPO, "show", "%s:%s" % (commit, relpath)]
    )
    with open(out, "wb") as handle:
        handle.write(blob)
    return out, hashlib.sha256(blob).hexdigest()


def load_module(path, name, patch=None):
    """Import a source file under a private name, optionally text-patched."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    if patch is not None:
        source = patch(source)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


INSTRUMENT_ANCHOR = (
    "                            if free.any():\n"
    "                                candidates = free\n"
)


def instrument(source):
    """Count overlap-infeasible placements without touching any RNG.

    ``overlap_infeasible``: a guided block where no admissible window clears
    the overlap tolerance (the branch fc49f61 changed).
    ``least_strict_subset``: of those, cases where the least-overlap set is a
    strict subset of the admissible set, i.e. where the two policies can pick
    differently.
    """
    source = source.replace("\r\n", "\n")
    assert source.count(INSTRUMENT_ANCHOR) == 1, "instrument anchor not unique"
    probe = (
        "                            EVENTS['overlap_infeasible'] += int(not free.any())\n"
        "                            if not free.any():\n"
        "                                _w = np.where(candidates, overlap, np.inf).min()\n"
        "                                EVENTS['least_strict_subset'] += int(\n"
        "                                    int((candidates & (overlap <= _w)).sum())\n"
        "                                    < int(candidates.sum()))\n"
    )
    header = "EVENTS = {'overlap_infeasible': 0, 'least_strict_subset': 0}\n"
    # Keep ``from __future__`` first.
    future = "from __future__ import annotations\n"
    assert source.count(future) == 1
    source = source.replace(future, future + header)
    return source.replace(INSTRUMENT_ANCHOR, probe + INSTRUMENT_ANCHOR)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_prod_yaml():
    import yaml

    with open(PROD_YAML, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collator_kwargs(cfg, curriculum_cfg):
    """Exactly the kwargs src/train_patch.py passes to MirageMaskCollator."""
    mask_cfg = cfg["mask"]
    crop_size = cfg["data"]["crop_size"]
    return dict(
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
        curriculum_cfg=curriculum_cfg,
    )


def legacy_kwargs(kwargs):
    """804c639's collator/generator has no ``pred_target_k``."""
    out = dict(kwargs)
    assert out.pop("pred_target_k", None) is None
    return out


def hash_tensors(tensors):
    h = hashlib.sha256()
    for t in tensors:
        arr = t.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def numeric_stats(ms):
    return {
        k: float(v)
        for k, v in ms.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def synth_guides(B, seed, thickness=(1.5, 4.5)):
    """(B, 2, 16, 16) band-shaped guides (A1 replay_masks.py generator)."""
    rs = np.random.RandomState(seed)
    g = np.zeros((B, 16, 16), np.float32)
    for b in range(B):
        c = rs.uniform(4, 12)
        tilt = rs.uniform(-0.2, 0.2)
        th = rs.uniform(*thickness)
        for col in range(16):
            ctr = c + tilt * (col - 8)
            for row in range(16):
                d = abs(row - ctr)
                g[b, row, col] = float(np.clip(1.0 - (d - th / 2.0), 0.0, 1.0))
    occ = torch.from_numpy(g)
    return torch.stack([occ, (occ >= 0.25).float()], dim=1)


def synth_fixtures(B=64):
    """Named synthetic guide batches covering the fallback's edge cases."""
    fixtures = {}
    fixtures["band"] = (synth_guides(B, 321), torch.ones(B, dtype=torch.bool))
    # Thin bands: few admissible windows, so the overlap tolerance is usually
    # infeasible for the 2nd-4th block -- the case fc49f61 changed.
    fixtures["thin_band"] = (
        synth_guides(B, 654, thickness=(0.3, 1.5)),
        torch.ones(B, dtype=torch.bool),
    )
    # Mixed: invalid guides, empty guides and tiny regions (infeasible).
    g = synth_guides(B, 987)
    valid = torch.ones(B, dtype=torch.bool)
    valid[::7] = False
    g[3::11] = 0.0
    tiny = torch.zeros(16, 16)
    tiny[7:9, 7:9] = 1.0
    g[5::13, 0] = tiny
    g[5::13, 1] = tiny
    fixtures["mixed_invalid_tiny"] = (g, valid)
    return fixtures


def make_batch(guides, valid, image=None):
    """Collator input: (image, guide, valid) tuples; images do not affect masks."""
    if image is None:
        image = torch.zeros(3, 1, 1)
    return [(image, guides[i], valid[i]) for i in range(guides.shape[0])]
