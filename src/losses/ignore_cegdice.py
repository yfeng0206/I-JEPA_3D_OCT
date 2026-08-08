"""Ignore-aware CE + Generalized Dice, numerically matching MIRAGE's CEGDiceLoss.

Why this exists
---------------
MIRAGE's ``mutils.gdice.CEGDiceLoss`` has no ``ignore_index``.  GOALS never
needed one because all of its classes are real.  The merged dataset ignores
50-55% of pixels in the Duke and AROI images (their sub-Bruch's class mixes
choroid with sclera, and their lesion classes have no counterpart in our
taxonomy), so an ignore-aware criterion is mandatory.

The hard requirement is EQUIVALENCE: with no ignored pixels this must equal
``CEGDiceLoss`` to floating-point tolerance.  Otherwise a merged run differs
from the GOALS baseline in two ways at once -- the data AND the objective --
and any difference in the result is uninterpretable.

Matching MIRAGE exactly therefore means reproducing its reductions, not merely
computing "a generalized Dice":

* cross-entropy: mean over non-ignored pixels;
* Dice: computed PER IMAGE over spatial dims, summed across channels, then
  averaged over the batch -- not pooled across the batch;
* the inverse-square-frequency weights use MIRAGE's own infinity handling
  (``w[isinf] = 0`` then ``w[isinf] = max(w)``), applied per image;
* ``0.5 * CE + 0.5 * Dice``.

Ignored pixels are removed by zeroing them in both the probability map and the
one-hot target, so they contribute nothing to intersection, ground truth volume
or predicted volume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IgnoreAwareCEGDice(nn.Module):
    """CE + generalized Dice with an ignore index.

    Args:
        num_classes: number of model logits.
        ignore_index: target value excluded from both terms.
        ce_weight: weight on cross-entropy; Dice gets ``1 - ce_weight``.
        include_background: whether class 0 participates in the Dice term.
            MIRAGE's default is ``True``.
        smooth: numerical smoothing, matching MIRAGE's ``1e-5``.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        include_background: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 <= ce_weight <= 1.0:
            raise ValueError('ce_weight must be in [0, 1], got %r' % ce_weight)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.include_background = bool(include_background)
        self.smooth = float(smooth)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        valid = target != self.ignore_index
        if not bool(valid.any()):
            # Nothing supervised in this batch: return a connected zero so every
            # parameter still receives a gradient.
            return logits.sum() * 0.0

        ce = self.ce(logits, target)
        loss = self.ce_weight * ce
        if self.ce_weight >= 1.0:
            return loss
        return loss + (1.0 - self.ce_weight) * self._dice(logits, target, valid)

    def _dice(self, logits, target, valid):
        probs = torch.softmax(logits, dim=1)                     # (B, C, H, W)
        safe = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe, self.num_classes)               # (B, H, W, C)
        onehot = onehot.permute(0, 3, 1, 2).to(probs.dtype)      # (B, C, H, W)

        m = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * m
        onehot = onehot * m

        if not self.include_background:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]

        reduce_axis = list(range(2, probs.dim()))                # spatial only
        intersection = torch.sum(onehot * probs, reduce_axis)    # (B, C)
        ground_o = torch.sum(onehot, reduce_axis)
        pred_o = torch.sum(probs, reduce_axis)
        denominator = ground_o + pred_o

        w = torch.reciprocal(ground_o.float() * ground_o.float())
        # MIRAGE's own infinity handling, per image (mutils/gdice.py).
        w = w.clone()
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f = 1.0 - (2.0 * (intersection * w).sum(1) + self.smooth) \
            / ((denominator * w).sum(1) + self.smooth)
        return torch.mean(f)
