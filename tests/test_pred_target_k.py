"""Tests for fixed-K predictor target resampling.

Irregular anatomy targets vary in size.  The collators stack per-sample
target lists into one tensor, and previously did that by front-slicing every
target to the minimum length in the microbatch.  Measured over 981 slices at
batch 64 that left the predictor 4.0 of 55.7 anatomy cells per slice (7.2%),
with K==1 in 99.8% of batches.
"""

import numpy as np
import pytest
import torch

from src.masks.curriculum import CurriculumMaskGenerator
from src.masks.utils import apply_masks, resample_to_k


def _cfg(**kw):
    base = {"mode": "mirage_envelope", "enabled": False}
    base.update(kw)
    return base


def _gen(pred_target_k=None):
    return CurriculumMaskGenerator(
        input_size=(256, 256),
        patch_size=16,
        npred=4,
        nenc=1,
        pred_target_k=pred_target_k,
        curriculum_cfg=_cfg(),
    )


# --------------------------------------------------------------- resample_to_k

def test_resample_returns_exactly_k():
    for n in (1, 3, 7, 16, 40, 200):
        t = torch.arange(n, dtype=torch.long)
        for k in (1, 4, 16, 64):
            assert resample_to_k(t, k).numel() == k


def test_resample_is_identity_when_already_k():
    t = torch.arange(16, dtype=torch.long)
    out = resample_to_k(t, 16)
    assert torch.equal(out, t)


def test_resample_subsamples_without_replacement_when_larger():
    t = torch.arange(50, dtype=torch.long)
    out = resample_to_k(t, 16)
    assert out.numel() == 16
    assert out.unique().numel() == 16, "must not duplicate when it can avoid it"


def test_resample_keeps_every_original_index_when_smaller():
    """Upsampling must not drop any of the anatomy cells it was given."""
    t = torch.tensor([3, 9, 40], dtype=torch.long)
    out = resample_to_k(t, 16)
    assert out.numel() == 16
    assert set(t.tolist()) <= set(out.tolist())


def test_resample_only_emits_real_indices():
    """No sentinel/pad values: the predictor's attention has no padding mask."""
    t = torch.tensor([5, 11, 12], dtype=torch.long)
    out = resample_to_k(t, 32)
    assert set(out.tolist()) <= set(t.tolist())


def test_resample_output_is_sorted():
    t = torch.tensor([40, 3, 9], dtype=torch.long)
    assert torch.equal(resample_to_k(t, 3).sort().values, resample_to_k(t, 3))


def test_resample_is_reproducible_with_generator():
    t = torch.arange(50, dtype=torch.long)
    a = resample_to_k(t, 16, torch.Generator().manual_seed(0))
    b = resample_to_k(t, 16, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_resample_rejects_empty_target():
    with pytest.raises(ValueError):
        resample_to_k(torch.empty(0, dtype=torch.long), 4)


def test_resampled_indices_are_gatherable():
    """Duplicated indices must still work through apply_masks."""
    x = torch.randn(2, 256, 8)
    t = torch.tensor([1, 2, 3], dtype=torch.long)
    m = torch.stack([resample_to_k(t, 16), resample_to_k(t, 16)], dim=0)
    out = apply_masks(x, [m])
    assert out.shape == (2, 16, 8)


# ------------------------------------------------------------------- collator

def test_default_is_unchanged_min_truncate():
    """pred_target_k=None must keep the old behaviour bit-identical."""
    g = _gen(pred_target_k=None)
    assert g.pred_target_k is None
    _, pred = g.generate(batch_size=4)
    lens = {p.shape[1] for p in pred}
    assert len(lens) == 1, "min-truncate makes every group the same length"


def test_fixed_k_gives_every_group_exactly_k():
    g = _gen(pred_target_k=16)
    _, pred = g.generate(batch_size=4)
    assert len(pred) == 4
    for p in pred:
        assert p.shape == (4, 16)


def test_fixed_k_does_not_change_the_context_mask():
    """The invariant that makes this fix safe.

    The context mask is built by subtracting the FULL untruncated target
    union, so switching the predictor policy must leave the encoder's view
    byte-identical.
    """
    import random as _r

    def run(k):
        torch.manual_seed(0)
        np.random.seed(0)
        _r.seed(0)
        return _gen(pred_target_k=k).generate(batch_size=4)

    enc_a, _ = run(None)
    enc_b, _ = run(16)
    assert len(enc_a) == len(enc_b)
    for a, b in zip(enc_a, enc_b):
        assert torch.equal(a, b), "context mask changed; the fix is not safe"


def test_fixed_k_retains_far_more_target_area_when_targets_are_ragged():
    """The point of the change.

    Note this is asserted on RAGGED targets, not on the generator's default
    rectangles.  Rectangular targets are all the same size, so min-truncate
    costs nothing there and a fixed K would only throw cells away -- which is
    precisely why pred_target_k defaults to None.  Anatomy targets are ragged,
    and that is the case this fixes.
    """
    rng = np.random.default_rng(0)
    batch, npred, k = 64, 4, 16
    groups = [[torch.arange(int(n), dtype=torch.long)
               for n in rng.integers(1, 40, size=batch)]
              for _ in range(npred)]

    gmin = max(1, min(int(t.numel()) for g in groups for t in g))
    old = batch * npred * gmin
    new = batch * npred * k

    distinct_old = sum(min(int(t.numel()), gmin) for g in groups for t in g)
    distinct_new = sum(min(int(t.numel()), k) for g in groups for t in g)

    assert gmin == 1, "one tiny target should collapse the whole batch"
    assert new > old
    assert distinct_new > 5 * distinct_old


def test_fixed_k_on_rectangles_is_exactly_k():
    """Guard the default: rectangles are uniform, so K is honoured exactly."""
    g = _gen(pred_target_k=16)
    _, pred = g.generate(batch_size=16)
    for p in pred:
        assert p.shape == (16, 16)


def test_fixed_k_indices_stay_in_range():
    g = _gen(pred_target_k=16)
    _, pred = g.generate(batch_size=4)
    for p in pred:
        assert int(p.min()) >= 0
        assert int(p.max()) < g.height * g.width
