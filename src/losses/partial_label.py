"""Partial-label segmentation loss for merging datasets with unequal taxonomies.

Motivation
----------
We fine-tune a 4-class retinal layer model (Elsewhere / RNFL / GCIPL / Choroid)
on a merge of GOALS, Duke DME and AROI.  The sources do not agree on what they
annotate:

  * GOALS labels all four classes.
  * Duke DME separates RNFL and GCIPL but its deepest class runs to the image
    bottom and so mixes choroid with sclera -> those pixels are unusable.
  * AROI annotates the inner retina as ONE merged band (``ILM-IPL/INL``) that
    cannot be split into RNFL vs GCIPL, and its sub-BM class has the same
    sclera problem as Duke.

Discarding AROI's inner band would throw away our largest source (1,105 of
1,315 images).  Instead its pixels carry a SUPERCLASS label meaning "this is
RNFL or GCIPL, but we do not know which".

Formulation
-----------
Every pixel carries a SET of permitted classes:

    hard label c        -> {c}
    InnerRetina         -> {RNFL, GCIPL}
    ignore              -> {} (excluded)

and the cross-entropy term is the negative log of the total probability mass on
that set::

    L = -log( sum_{c in allowed} p_c )

For a singleton set this reduces exactly to ordinary cross-entropy, so hard and
partial labels are handled by one expression with no special-casing and no
relative weighting to tune.  It is computed as ``-logsumexp`` over the allowed
log-probabilities, which is numerically stable and never forms ``p`` explicitly.

This is the standard partial-label / "superset label" objective; the model is
free to choose either member of the set, and is penalised only for putting mass
outside it.

The Dice term is computed on hard-labelled pixels only: a generalized Dice
needs a definite per-class target, which a superclass pixel does not have.

Why not just use ``CEGDiceLoss``
--------------------------------
MIRAGE's ``CEGDiceLoss`` has no ``ignore_index`` at all (GOALS never needed one,
since all four of its classes are real).  Merged training ignores 50-63% of
pixels in the Duke and AROI images, so an ignore-aware criterion is required
regardless of the superclass work.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Label-tensor indices produced by scripts/build_seg_merged.py via INFO.json.
# 0-3 are real model outputs; 4 and 5 exist only in the target.
IDX_ELSEWHERE, IDX_RNFL, IDX_GCIPL, IDX_CHOROID = 0, 1, 2, 3
IDX_INNER_RETINA = 4
IDX_IGNORE = 5

#: Which model classes each label index permits.
DEFAULT_ALLOWED: Dict[int, Sequence[int]] = {
    IDX_ELSEWHERE: (IDX_ELSEWHERE,),
    IDX_RNFL: (IDX_RNFL,),
    IDX_GCIPL: (IDX_GCIPL,),
    IDX_CHOROID: (IDX_CHOROID,),
    IDX_INNER_RETINA: (IDX_RNFL, IDX_GCIPL),
}


class PartialLabelCEGDice(nn.Module):
    """Cross-entropy over permitted class sets, plus generalized Dice.

    Args:
        num_classes: number of model logits (4 here).
        allowed: label index -> permitted model classes. Any index absent from
            this mapping is treated as ignore.
        ignore_index: label index excluded from every term.
        ce_weight: weight on the cross-entropy term; Dice gets ``1 - ce_weight``.
            Defaults to 0.5, matching MIRAGE's ``CEGDiceLoss``.
        include_background: whether class 0 participates in the Dice term.
    """

    def __init__(
        self,
        num_classes: int = 4,
        allowed: Optional[Dict[int, Sequence[int]]] = None,
        ignore_index: int = IDX_IGNORE,
        ce_weight: float = 0.5,
        include_background: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 <= ce_weight <= 1.0:
            raise ValueError('ce_weight must be in [0, 1], got %r' % ce_weight)
        self.num_classes = int(num_classes)
        self.allowed = dict(DEFAULT_ALLOWED if allowed is None else allowed)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

        for idx, classes in self.allowed.items():
            if not len(classes):
                raise ValueError('label %d permits no classes' % idx)
            for c in classes:
                if not 0 <= c < self.num_classes:
                    raise ValueError(
                        'label %d permits class %d outside [0, %d)'
                        % (idx, c, self.num_classes))
        if self.ignore_index in self.allowed:
            raise ValueError('ignore_index %d must not appear in `allowed`'
                             % self.ignore_index)

        # (max_label + 1, num_classes) boolean membership table, registered so
        # it follows the module across devices.
        size = max(max(self.allowed) + 1, self.ignore_index + 1)
        table = torch.zeros(size, self.num_classes, dtype=torch.bool)
        for idx, classes in self.allowed.items():
            for c in classes:
                table[idx, c] = True
        self.register_buffer('allowed_table', table, persistent=False)

    def is_partial(self) -> torch.Tensor:
        """True for label indices that permit more than one class."""
        return self.allowed_table.sum(dim=1) > 1

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``(B, num_classes, H, W)``.
            target: ``(B, H, W)`` of label indices.
        """
        if logits.dim() != 4:
            raise ValueError('expected (B, C, H, W) logits, got %r'
                             % (tuple(logits.shape),))
        if logits.shape[1] != self.num_classes:
            raise ValueError('logits have %d channels, expected %d'
                             % (logits.shape[1], self.num_classes))
        if target.shape != (logits.shape[0],) + logits.shape[2:]:
            raise ValueError('target shape %r does not match logits %r'
                             % (tuple(target.shape), tuple(logits.shape)))

        target = target.long()
        table = self.allowed_table.to(logits.device)
        # Any label outside the table, and the ignore index itself, is dropped.
        known = (target >= 0) & (target < table.shape[0])
        safe = torch.where(known, target, torch.zeros_like(target))
        permits = table[safe]                       # (B, H, W, C)
        valid = known & permits.any(dim=-1)         # ignore has an all-False row

        if not bool(valid.any()):
            # No supervised pixel in this batch; keep the graph connected so
            # every parameter still receives a (zero) gradient.
            return logits.sum() * 0.0

        log_p = F.log_softmax(logits, dim=1)                      # (B, C, H, W)
        log_p_last = log_p.permute(0, 2, 3, 1)                    # (B, H, W, C)
        neg_inf = torch.finfo(log_p_last.dtype).min
        masked = log_p_last.masked_fill(~permits, neg_inf)
        # -log sum_{c in allowed} p_c, stable and exact for singleton sets.
        ce_map = -torch.logsumexp(masked, dim=-1)
        ce = ce_map[valid].mean()

        loss = self.ce_weight * ce
        if self.ce_weight >= 1.0:
            return loss

        dice = self._generalized_dice(log_p, permits, valid)
        return loss + (1.0 - self.ce_weight) * dice

    def _generalized_dice(
        self,
        log_p: torch.Tensor,
        permits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Generalized Dice over hard-labelled pixels only.

        Superclass pixels have no definite per-class target, so they cannot
        contribute a Dice numerator; including them would require inventing an
        assignment. They still train the model through the cross-entropy term.
        """
        hard = valid & (permits.sum(dim=-1) == 1)
        if not bool(hard.any()):
            return log_p.sum() * 0.0

        probs = log_p.exp().permute(0, 2, 3, 1)          # (B, H, W, C)
        cls = permits.float().argmax(dim=-1)             # singleton -> its class
        onehot = F.one_hot(cls, self.num_classes).to(probs.dtype)

        m = hard.unsqueeze(-1).to(probs.dtype)
        probs = probs * m
        onehot = onehot * m

        dims = (0, 1, 2)                                  # reduce B, H, W
        start = 0 if self.include_background else 1
        probs = probs[..., start:]
        onehot = onehot[..., start:]

        intersection = (probs * onehot).sum(dims)
        ground_o = onehot.sum(dims)
        pred_o = probs.sum(dims)

        # Inverse-square-frequency weighting, as in the generalized Dice paper.
        # Classes absent from the batch get zero weight rather than infinity.
        w = torch.where(ground_o > 0, 1.0 / (ground_o * ground_o),
                        torch.zeros_like(ground_o))
        if not bool((w > 0).any()):
            return log_p.sum() * 0.0
        w = torch.where(torch.isinf(w), torch.zeros_like(w), w)

        num = 2.0 * (intersection * w).sum() + self.smooth
        den = ((ground_o + pred_o) * w).sum() + self.smooth
        return 1.0 - num / den
