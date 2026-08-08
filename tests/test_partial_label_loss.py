"""Tests for the partial-label segmentation loss.

The critical property is EQUIVALENCE: on hard labels the cross-entropy term
must reduce exactly to ``torch.nn.functional.cross_entropy``.  If it does not,
then merged training is not comparable to the GOALS-only baseline and any
downstream difference is uninterpretable.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses.partial_label import (
    IDX_CHOROID,
    IDX_ELSEWHERE,
    IDX_GCIPL,
    IDX_IGNORE,
    IDX_INNER_RETINA,
    IDX_RNFL,
    PartialLabelCEGDice,
)


def _logits(b=2, c=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, generator=g, dtype=torch.float64)


def test_hard_labels_match_cross_entropy_exactly():
    """CE-only mode on hard labels == F.cross_entropy. The anchor property."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits()
    g = torch.Generator().manual_seed(1)
    y = torch.randint(0, 4, (2, 8, 8), generator=g)
    got = loss(x, y)
    want = F.cross_entropy(x, y)
    assert torch.allclose(got, want, atol=1e-12), (got.item(), want.item())


def test_ignore_pixels_excluded_from_cross_entropy():
    """Ignored pixels must not shift the loss at all."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits()
    g = torch.Generator().manual_seed(2)
    y = torch.randint(0, 4, (2, 8, 8), generator=g)
    ref = loss(x, y)

    y2 = y.clone()
    y2[0, 0, :] = IDX_IGNORE          # blank a row
    got = loss(x, y2)
    keep = y2 != IDX_IGNORE
    want = F.cross_entropy(x, y, reduction='none')[keep].mean()
    assert torch.allclose(got, want, atol=1e-12)
    assert not torch.allclose(got, ref)   # it really did change something


def test_all_ignore_returns_zero_with_gradient():
    """A batch with no supervision must not produce NaN, and must stay
    connected to the graph so DDP still sees a gradient."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits().requires_grad_(True)
    y = torch.full((2, 8, 8), IDX_IGNORE)
    out = loss(x, y)
    assert torch.isfinite(out) and out.item() == 0.0
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_superclass_equals_neg_log_sum_of_members():
    """InnerRetina must cost -log(p_RNFL + p_GCIPL)."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits(b=1, h=2, w=2, seed=3)
    y = torch.full((1, 2, 2), IDX_INNER_RETINA)
    got = loss(x, y)
    p = F.softmax(x, dim=1)
    want = -(p[:, IDX_RNFL] + p[:, IDX_GCIPL]).log().mean()
    assert torch.allclose(got, want, atol=1e-12)


def test_superclass_is_never_worse_than_either_member():
    """The set objective must be <= the cost of committing to either member,
    since the permitted mass includes both."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits(b=1, h=4, w=4, seed=4)
    soft = loss(x, torch.full((1, 4, 4), IDX_INNER_RETINA))
    as_rnfl = loss(x, torch.full((1, 4, 4), IDX_RNFL))
    as_gcipl = loss(x, torch.full((1, 4, 4), IDX_GCIPL))
    assert soft < as_rnfl and soft < as_gcipl


def test_superclass_indifferent_to_split_between_members():
    """Two logit fields with the same RNFL+GCIPL mass but different splits must
    cost the same. This is what 'we do not know which' means."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    a = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
    a[0, IDX_RNFL, 0, 0] = 2.0
    a[0, IDX_GCIPL, 0, 0] = 0.0
    b = a.clone()
    b[0, IDX_RNFL, 0, 0] = 0.0
    b[0, IDX_GCIPL, 0, 0] = 2.0
    y = torch.full((1, 1, 1), IDX_INNER_RETINA)
    assert torch.allclose(loss(a, y), loss(b, y), atol=1e-12)


def test_superclass_gradient_pushes_mass_into_the_set():
    """Gradient must increase permitted-class logits and decrease others."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = torch.zeros(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
    loss(x, torch.full((1, 1, 1), IDX_INNER_RETINA)).backward()
    g = x.grad[0, :, 0, 0]
    assert g[IDX_RNFL] < 0 and g[IDX_GCIPL] < 0        # push up
    assert g[IDX_ELSEWHERE] > 0 and g[IDX_CHOROID] > 0  # push down


def test_perfect_hard_prediction_drives_loss_to_zero():
    """Confident correct predictions on hard labels -> ~0 for CE and Dice."""
    loss = PartialLabelCEGDice(ce_weight=0.5).double()
    y = torch.tensor([[[IDX_ELSEWHERE, IDX_RNFL], [IDX_GCIPL, IDX_CHOROID]]])
    x = F.one_hot(y, 4).permute(0, 3, 1, 2).double() * 60.0
    assert loss(x, y).item() < 1e-6


def test_dice_ignores_superclass_pixels():
    """Dice has no definite target for a superclass pixel, so changing only
    those pixels must not move the Dice term."""
    loss = PartialLabelCEGDice(ce_weight=0.0).double()   # Dice only
    x = _logits(b=1, h=4, w=4, seed=7)
    y = torch.full((1, 4, 4), IDX_RNFL)
    y[0, 0, :] = IDX_GCIPL
    base = loss(x, y)
    y2 = y.clone()
    y2[0, 2, :] = IDX_INNER_RETINA
    y3 = y.clone()
    y3[0, 2, :] = IDX_IGNORE
    # Superclass and ignore must be treated identically by the Dice term.
    assert torch.allclose(loss(x, y2), loss(x, y3), atol=1e-12)
    assert not torch.allclose(loss(x, y2), base)


def test_rejects_malformed_inputs():
    loss = PartialLabelCEGDice()
    with pytest.raises(ValueError, match='logits'):
        loss(torch.randn(2, 4, 8), torch.zeros(2, 8, dtype=torch.long))
    with pytest.raises(ValueError, match='channels'):
        loss(torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))
    with pytest.raises(ValueError, match='target shape'):
        loss(torch.randn(2, 4, 8, 8), torch.zeros(2, 4, 4, dtype=torch.long))


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match='ce_weight'):
        PartialLabelCEGDice(ce_weight=1.5)
    with pytest.raises(ValueError, match='outside'):
        PartialLabelCEGDice(num_classes=4, allowed={0: (0,), 1: (9,)})
    with pytest.raises(ValueError, match='no classes'):
        PartialLabelCEGDice(allowed={0: ()})
    with pytest.raises(ValueError, match='ignore_index'):
        PartialLabelCEGDice(allowed={0: (0,), IDX_IGNORE: (1,)})


def test_out_of_range_labels_are_ignored_not_crashing():
    """A stray label index must be dropped, never indexed out of bounds."""
    loss = PartialLabelCEGDice(ce_weight=1.0).double()
    x = _logits()
    y = torch.randint(0, 4, (2, 8, 8))
    y[0, 0, 0] = 99
    out = loss(x, y)
    assert torch.isfinite(out)


def test_is_partial_flags_only_the_superclass():
    loss = PartialLabelCEGDice()
    flags = loss.is_partial()
    assert not flags[IDX_RNFL] and not flags[IDX_CHOROID]
    assert flags[IDX_INNER_RETINA]
    assert not flags[IDX_IGNORE]


def test_float32_and_amp_dtype_stability():
    """The real run uses fp16 autocast; the loss must stay finite."""
    loss = PartialLabelCEGDice(ce_weight=0.5)
    x = torch.randn(2, 4, 16, 16).half() * 10
    y = torch.randint(0, 6, (2, 16, 16))
    out = loss(x.float(), y)
    assert torch.isfinite(out)
