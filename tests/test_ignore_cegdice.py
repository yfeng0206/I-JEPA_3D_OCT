"""Tests for IgnoreAwareCEGDice.

The anchor property is EQUIVALENCE with MIRAGE's own CEGDiceLoss when nothing
is ignored. If that fails, a merged training run changes both the data and the
objective, and no downstream difference can be attributed to the data.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from src.losses.ignore_cegdice import IgnoreAwareCEGDice

MIRAGE = r'D:\jepa_phase0\mirage-goals\MIRAGE'
if MIRAGE not in sys.path:
    sys.path.insert(0, MIRAGE)

try:
    from mutils.gdice import CEGDiceLoss
    HAVE_MIRAGE = True
except Exception:                                    # pragma: no cover
    HAVE_MIRAGE = False


def _logits(b=3, c=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, generator=g, dtype=torch.float64)


@pytest.mark.skipif(not HAVE_MIRAGE, reason='MIRAGE checkout not available')
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_matches_mirage_cegdice_when_nothing_ignored(seed):
    """The anchor: identical to MIRAGE's loss on fully-labelled batches."""
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255).double()
    theirs = CEGDiceLoss().double()
    x = _logits(seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    y = torch.randint(0, 4, (3, 8, 8), generator=g)
    a, b = ours(x, y), theirs(x, y)
    assert torch.allclose(a, b, atol=1e-9), (a.item(), b.item())


@pytest.mark.skipif(not HAVE_MIRAGE, reason='MIRAGE checkout not available')
def test_matches_mirage_with_unbalanced_classes():
    """Equivalence must hold when a class is absent, which is where the
    inverse-square-frequency weights hit infinities."""
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255).double()
    theirs = CEGDiceLoss().double()
    x = _logits(seed=9)
    y = torch.zeros(3, 8, 8, dtype=torch.long)
    y[:, :2] = 1                     # classes 2 and 3 absent entirely
    assert torch.allclose(ours(x, y), theirs(x, y), atol=1e-9)


def test_ignored_pixels_do_not_affect_cross_entropy():
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255,
                              ce_weight=1.0).double()
    x = _logits()
    g = torch.Generator().manual_seed(5)
    y = torch.randint(0, 4, (3, 8, 8), generator=g)
    y2 = y.clone()
    y2[0, 0, :] = 255
    keep = y2 != 255
    want = F.cross_entropy(x, y, reduction='none')[keep].mean()
    assert torch.allclose(ours(x, y2), want, atol=1e-12)


def test_ignoring_a_region_changes_the_dice_term():
    """Sanity: the mask is actually applied, not silently dropped."""
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255,
                              ce_weight=0.0).double()
    x = _logits(seed=11)
    y = torch.randint(0, 4, (3, 8, 8), generator=torch.Generator().manual_seed(6))
    base = ours(x, y)
    y2 = y.clone()
    y2[:, :4] = 255
    assert not torch.allclose(base, ours(x, y2))


def test_all_ignored_returns_zero_with_gradient():
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255).double()
    x = _logits().requires_grad_(True)
    y = torch.full((3, 8, 8), 255)
    out = ours(x, y)
    assert torch.isfinite(out) and out.item() == 0.0
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_perfect_prediction_is_near_zero():
    ours = IgnoreAwareCEGDice(num_classes=3, ignore_index=255).double()
    y = torch.tensor([[[0, 1], [2, 1]]])
    x = F.one_hot(y, 3).permute(0, 3, 1, 2).double() * 60.0
    assert ours(x, y).item() < 1e-6


def test_ignored_pixels_cannot_be_penalised_by_prediction():
    """Whatever the model predicts on ignored pixels must not change the loss."""
    ours = IgnoreAwareCEGDice(num_classes=4, ignore_index=255).double()
    y = torch.randint(0, 4, (2, 6, 6), generator=torch.Generator().manual_seed(3))
    y[:, :2] = 255
    x1 = _logits(b=2, h=6, w=6, seed=21)
    x2 = x1.clone()
    x2[:, :, :2] = torch.randn(2, 4, 2, 6, dtype=torch.float64) * 5
    assert torch.allclose(ours(x1, y), ours(x2, y), atol=1e-9)


def test_rejects_malformed_inputs():
    loss = IgnoreAwareCEGDice(num_classes=4)
    with pytest.raises(ValueError, match='logits'):
        loss(torch.randn(2, 4, 8), torch.zeros(2, 8, dtype=torch.long))
    with pytest.raises(ValueError, match='channels'):
        loss(torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))
    with pytest.raises(ValueError, match='target shape'):
        loss(torch.randn(2, 4, 8, 8), torch.zeros(2, 4, 4, dtype=torch.long))
    with pytest.raises(ValueError, match='ce_weight'):
        IgnoreAwareCEGDice(num_classes=4, ce_weight=-0.1)
