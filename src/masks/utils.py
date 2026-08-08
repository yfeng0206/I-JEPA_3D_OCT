"""Mask utilities for I-JEPA."""

import torch


def resample_to_k(indices, k, generator=None):
    """Return exactly ``k`` token indices drawn from ``indices``.

    Irregular anatomy targets vary in size, and the collators stack the
    per-sample target lists into one tensor.  Stacking previously took the
    MINIMUM length across the whole microbatch and front-sliced everything to
    it, so a single one-cell target anywhere in the batch truncated all of
    them.  Measured over 981 slices at batch 64, that left the predictor with
    4.0 of 55.7 anatomy cells per slice -- 7.2%, with K==1 in 99.8% of
    batches.

    This gives every target the same length without discarding a sample:

        len(indices) > k   subsample k distinct indices
        len(indices) < k   sample the shortfall WITH replacement

    Every slot therefore holds a real token index, which matters because the
    predictor's attention has no padding mask and would otherwise attend to
    pad positions as if they were real.

    This does NOT change which patches are hidden.  Both collators build the
    context mask by subtracting the FULL, untruncated target union before
    this runs, so the encoder sees exactly what it saw before; only the number
    of hidden cells the predictor is asked to reconstruct changes.

    Args:
        indices: 1-D LongTensor of token indices for one target.
        k: Desired number of indices.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        1-D LongTensor of length ``k``, sorted ascending.
    """
    n = int(indices.numel())
    if n == 0:
        raise ValueError("cannot resample an empty target to k=%d" % k)
    if n == k:
        return indices.sort().values
    if n > k:
        sel = torch.randperm(n, generator=generator)[:k]
        return indices[sel].sort().values
    extra = torch.randint(0, n, (k - n,), generator=generator)
    return torch.cat([indices, indices[extra]]).sort().values


def apply_masks(x, masks):
    """Apply binary masks to select token subsets from a sequence.

    Args:
        x: Tensor of shape (B, N, D) — full sequence of tokens.
        masks: List of index tensors, each of shape (B, num_keep) containing
            the integer indices of tokens to keep.  When multiple masks are
            provided the results are concatenated along the batch dimension
            (each mask yields one copy of the batch).

    Returns:
        Tensor of shape (B_total, num_keep, D) where B_total = B * len(masks).
    """
    all_x = []
    for m in masks:
        # m: (B, num_keep) — integer indices into the N dimension
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, num_keep, D)
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
