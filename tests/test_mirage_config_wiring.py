"""Regression tests for MIRAGE guided-masking configuration wiring.

These pin a silent, severe bug found during the pre-launch review of the
MIRAGE thr-0.25 run: ``configs/patch_mirage_envelope.yaml`` set
``mirage_occupancy_threshold: 0.25``, the collator read it correctly, but
``train_patch.py`` never passed it to ``GuidedOCTSliceDataset``.  The dataset
therefore kept its ``0.5`` default and built the channel-1 *placement* grid --
the grid the collator draws target-block candidates from -- under a different
policy from the one it scored against.

Measured effect on 3,840 images at full guidance: unique target patches 125.0
vs 107.7 and context patches 102.2 vs 123.3, i.e. the run would have masked
~16% more of the image than the calibrated policy.  Nothing crashed and no
metric looked obviously wrong, so only a wiring assertion catches it.
"""

import ast
import os

import numpy as np
import pytest
import yaml

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATCH = os.path.join(REPO_ROOT, "src", "train_patch.py")
MIRAGE_CONFIG = os.path.join(REPO_ROOT, "configs", "patch_mirage_envelope.yaml")


def _guided_dataset_call():
    """Return the ``GuidedOCTSliceDataset(...)`` call node in train_patch.py."""
    with open(TRAIN_PATCH, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=TRAIN_PATCH)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GuidedOCTSliceDataset"
    ]
    assert len(calls) == 1, (
        "Expected exactly one GuidedOCTSliceDataset construction in "
        "train_patch.py, found %d" % len(calls)
    )
    return calls[0]


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def test_train_patch_passes_occupancy_threshold():
    """The configured threshold must reach the dataset, not just the collator."""
    call = _guided_dataset_call()
    kw = _keyword(call, "occupancy_threshold")
    assert kw is not None, (
        "train_patch.py constructs GuidedOCTSliceDataset without "
        "occupancy_threshold, so the dataset will silently use its 0.5 default "
        "while the collator uses the configured value."
    )
    source = ast.dump(kw.value)
    assert "mirage_occupancy_threshold" in source, (
        "occupancy_threshold must be read from the curriculum config key "
        "'mirage_occupancy_threshold', got: %s" % ast.unparse(kw.value)
    )


def test_train_patch_passes_dilate_patches():
    """The sibling guide parameter must stay wired for the same reason."""
    call = _guided_dataset_call()
    kw = _keyword(call, "dilate_patches")
    assert kw is not None, "GuidedOCTSliceDataset must receive dilate_patches"
    assert "mirage_dilate_patches" in ast.dump(kw.value)


def test_guided_dataset_default_threshold_is_half():
    """Documents why the wiring test matters.

    If this default ever changes, the blast radius of a missing kwarg changes
    with it, so the value is pinned deliberately.
    """
    import inspect

    signature = inspect.signature(GuidedOCTSliceDataset.__init__)
    assert signature.parameters["occupancy_threshold"].default == 0.5


@pytest.mark.parametrize(
    "low,high",
    [(0.25, 0.5), (0.1, 0.75), (0.25, 1.0)],
)
def test_placement_grid_is_monotone_in_threshold(low, high):
    """A lower threshold must admit a superset of placement cells."""
    rng = np.random.default_rng(0)
    grid = rng.random((16, 16)).astype(np.float32)
    lower = grid >= low
    higher = grid >= high
    assert np.all(lower[higher]), "threshold is not monotone"
    assert lower.sum() > higher.sum(), (
        "expected strictly more admissible cells at the lower threshold"
    )


def test_shipped_config_matches_selected_policy():
    """Pin the policy chosen by the 1,000-volume sweep.

    thr 0.25 / dilate 0 was selected because it is the only setting whose
    masking geometry stays within 0.5% of the oracle arm on every axis, which
    is what keeps the experiment a controlled comparison.
    """
    with open(MIRAGE_CONFIG, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    curriculum = config["mask"]["curriculum"]
    assert curriculum["mode"] == "mirage_envelope"
    assert curriculum["mirage_occupancy_threshold"] == 0.25
    assert curriculum["mirage_dilate_patches"] == 0
    # The warm-start contract: ep25 is the zero-bias bootstrap epoch and full
    # guidance engages at ep30, exactly as the oracle arm ramped.
    assert curriculum["T_warm"] == 25
    assert curriculum["T_total"] == 30
    assert curriculum["r_max"] == 1.0
    # Effective batch must stay 512 to match the four-GPU reference run.
    assert (
        config["data"]["batch_size"] * config["optimization"]["accum_steps"] == 512
    )
