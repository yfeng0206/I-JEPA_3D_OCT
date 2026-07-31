"""Regression tests for the OCT slice cache.

The cache exists purely to remove a 200x read amplification (see
``scripts/build_slice_cache.py``).  It must therefore be *bit-identical* to
reading the ``.npz`` directly -- if it is not, it silently changes the training
data.  These tests build a small cache from synthetic volumes and assert exact
equality on both the raw slice reader and the fully-transformed tensor.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.datasets.oct_slices import (
    SLICE_CACHE_ARRAY,
    SLICE_CACHE_MANIFEST,
    OCTSliceDataset,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO_ROOT, "scripts", "build_slice_cache.py")

NUM_VOLUMES = 3
NUM_SLICES = 5


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    """Three synthetic volumes with deterministic, distinguishable content."""
    root = tmp_path_factory.mktemp("oct")
    split = root / "Training"
    split.mkdir()
    rng = np.random.default_rng(1234)
    for v in range(NUM_VOLUMES):
        volume = rng.integers(0, 256, size=(200, 200, 200), dtype=np.uint8)
        np.savez(
            split / ("data_%05d.npz" % v),
            oct_bscans=volume,
            glaucoma=np.int64(v % 2),
        )
    return root


@pytest.fixture(scope="module")
def cache_root(dataset_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    result = subprocess.run(
        [
            sys.executable,
            BUILDER,
            "--data-dir",
            str(dataset_root),
            "--cache-dir",
            str(out),
            "--splits",
            "Training",
            "--num-slices",
            str(NUM_SLICES),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def _datasets(dataset_root, cache_root, transform=None):
    plain = OCTSliceDataset(
        data_dir=str(dataset_root / "Training"),
        num_slices=NUM_SLICES,
        slice_size=64,
        transform=transform,
    )
    cached = OCTSliceDataset(
        data_dir=str(dataset_root / "Training"),
        num_slices=NUM_SLICES,
        slice_size=64,
        transform=transform,
        slice_cache=str(cache_root / "Training"),
    )
    return plain, cached


def test_cache_manifest_describes_the_split(dataset_root, cache_root):
    with open(cache_root / "Training" / SLICE_CACHE_MANIFEST) as handle:
        meta = json.load(handle)
    assert meta["num_slices"] == NUM_SLICES
    assert meta["volumes"] == ["data_%05d.npz" % v for v in range(NUM_VOLUMES)]
    assert meta["slice_indices"] == np.linspace(
        0, 199, num=NUM_SLICES, dtype=np.int64
    ).tolist()
    size = os.path.getsize(cache_root / "Training" / SLICE_CACHE_ARRAY)
    assert size == NUM_VOLUMES * NUM_SLICES * 200 * 200


def test_cached_slices_are_bit_identical(dataset_root, cache_root):
    """The whole point: identical bytes, different I/O cost."""
    plain, cached = _datasets(dataset_root, cache_root)
    assert len(plain) == len(cached) == NUM_VOLUMES * NUM_SLICES
    for file_idx in range(NUM_VOLUMES):
        for slice_within in range(NUM_SLICES):
            a = plain.read_slice(file_idx, slice_within)
            b = cached.read_slice(file_idx, slice_within)
            assert a.dtype == b.dtype == np.uint8
            assert np.array_equal(a, b), (
                "cache differs at volume %d slice %d" % (file_idx, slice_within)
            )


def test_transformed_tensors_are_identical(dataset_root, cache_root):
    """Equality must survive the resize/RGB/tensor path, not just the raw read."""
    plain, cached = _datasets(dataset_root, cache_root)
    for idx in range(len(plain)):
        a = plain[idx]
        b = cached[idx]
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        assert torch.equal(a, b), "tensor mismatch at index %d" % idx


def test_cache_rejects_wrong_num_slices(dataset_root, cache_root):
    with pytest.raises(RuntimeError, match="slices/volume"):
        OCTSliceDataset(
            data_dir=str(dataset_root / "Training"),
            num_slices=NUM_SLICES + 1,
            slice_size=64,
            slice_cache=str(cache_root / "Training"),
        )


def test_cache_rejects_volume_mismatch(dataset_root, cache_root, tmp_path):
    """A cache must never be paired with a split it was not built from."""
    other = tmp_path / "Training"
    other.mkdir()
    rng = np.random.default_rng(7)
    for v in range(NUM_VOLUMES + 1):
        np.savez(
            other / ("data_%05d.npz" % v),
            oct_bscans=rng.integers(0, 256, size=(200, 200, 200), dtype=np.uint8),
            glaucoma=np.int64(0),
        )
    with pytest.raises(RuntimeError, match="do not match"):
        OCTSliceDataset(
            data_dir=str(other),
            num_slices=NUM_SLICES,
            slice_size=64,
            slice_cache=str(cache_root / "Training"),
        )


def test_dataset_pickles_without_the_memmap(dataset_root, cache_root):
    """DataLoader workers re-open the memmap; the handle must not be pickled."""
    import pickle

    _, cached = _datasets(dataset_root, cache_root)
    cached.read_slice(0, 0)  # force the memmap open
    assert cached._cache_mm is not None
    restored = pickle.loads(pickle.dumps(cached))
    assert restored._cache_mm is None
    assert np.array_equal(restored.read_slice(1, 2), cached.read_slice(1, 2))
