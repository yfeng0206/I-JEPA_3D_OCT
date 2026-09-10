"""Multi-block masking collator for patch-level I-JEPA on OCT data.

Adapted from the original I-JEPA implementation for a 16x16 patch grid
(256 patches from 256x256 images with patch_size=16).
"""

import math
import random
from typing import List, Tuple

import torch


class MaskCollator:
    """Generates context (encoder) and target (predictor) masks for I-JEPA.

    For each image in a batch the collator:
      1. Samples ``npred`` target blocks (using ``pred_mask_scale`` and
         ``aspect_ratio``).
      2. Samples one context block (using ``enc_mask_scale``) that avoids
         overlap with the target blocks (unless ``allow_overlap=True``).

    Block sizes are determined by a seeded generator so they stay
    consistent within a batch; block *locations* are random per image.

    Args:
        input_size: Spatial resolution of the input image (H, W).
        patch_size: Size of each non-overlapping patch.
        enc_mask_scale: (min, max) fraction of total patches for context.
        pred_mask_scale: (min, max) fraction of total patches for targets.
        aspect_ratio: (min, max) aspect ratio for sampled blocks.
        nenc: Number of context masks per image.
        npred: Number of target masks per image.
        min_keep: Minimum number of context patches to keep.
        allow_overlap: If True, context and target masks may overlap.
    """

    def __init__(
        self,
        input_size=(256, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        audit_masks=False,
    ):
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size  # grid rows
        self.width = input_size[1] // patch_size   # grid cols
        self.num_patches = self.height * self.width  # 256 for 16x16

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.audit_masks = bool(audit_masks)
        self.last_mask_audit = []

        # Seeded generator for block *sizes* (consistent within a batch).
        self._size_gen = torch.Generator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_block_size(self, scale, generator):
        # type: (Tuple[float, float], torch.Generator) -> Tuple[int, int]
        """Return (block_h, block_w) in patch units."""
        min_s, max_s = scale
        num_target = int(self.num_patches * (
            min_s + (max_s - min_s) * torch.rand(1, generator=generator).item()
        ))
        num_target = max(num_target, 1)

        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + (max_ar - min_ar) * torch.rand(1, generator=generator).item()

        # block_h * block_w ~= num_target  and  block_h / block_w ~= ar
        block_h = int(round(math.sqrt(num_target * ar)))
        block_w = int(round(math.sqrt(num_target / ar)))
        block_h = max(1, min(block_h, self.height))
        block_w = max(1, min(block_w, self.width))
        return block_h, block_w

    @staticmethod
    def _sample_block_location(block_h, block_w, grid_h, grid_w):
        # type: (int, int, int, int) -> Tuple[int, int]
        """Return random top-left corner (row, col) for a block."""
        top = random.randint(0, grid_h - block_h)
        left = random.randint(0, grid_w - block_w)
        return top, left

    def _block_to_indices(self, top, left, block_h, block_w):
        # type: (int, int, int, int) -> List[int]
        """Convert block coordinates to a sorted list of patch indices."""
        indices = []
        for r in range(top, top + block_h):
            for c in range(left, left + block_w):
                indices.append(r * self.width + c)
        return sorted(indices)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, batch, *, block_sizes=None):
        """Collate a batch and generate encoder / predictor masks.

        Args:
            batch: list of image tensors, each (C, H, W).

        Returns:
            collated_batch: (B, C, H, W) tensor.
            collated_masks_enc: list of ``nenc`` tensors, each (B, num_keep).
            collated_masks_pred: list of ``npred`` tensors, each (B, num_keep).
        """
        B = len(batch)
        collated_batch = torch.stack(batch, dim=0)

        # Seed the size generator so block *sizes* are identical for every
        # image in this batch (locations still differ).
        if block_sizes is None:
            seed = random.randint(0, 2 ** 31)
            self._size_gen.manual_seed(seed)
            pred_sizes = [
                self._sample_block_size(self.pred_mask_scale, self._size_gen)
                for _ in range(self.npred)
            ]
            enc_sizes = [
                self._sample_block_size(self.enc_mask_scale, self._size_gen)
                for _ in range(self.nenc)
            ]
        else:
            pred_sizes, enc_sizes = block_sizes["pred"], block_sizes["enc"]
            if len(pred_sizes) != self.npred or len(enc_sizes) != self.nenc:
                raise ValueError("Injected sizes must match npred and nenc")
            if any(not (1 <= h <= self.height and 1 <= w <= self.width)
                   for h, w in list(pred_sizes) + list(enc_sizes)):
                raise ValueError("Injected block sizes are out of bounds")

        # Per-image mask generation.
        masks_enc = [[] for _ in range(self.nenc)]   # nenc lists of B entries
        masks_pred = [[] for _ in range(self.npred)]  # npred lists of B entries

        for _ in range(B):
            # --- Target (predictor) blocks ---
            pred_indices_union = set()  # type: set
            per_image_pred = []
            for p in range(self.npred):
                bh, bw = pred_sizes[p]
                top, left = self._sample_block_location(
                    bh, bw, self.height, self.width
                )
                indices = self._block_to_indices(top, left, bh, bw)
                per_image_pred.append(indices)
                pred_indices_union.update(indices)

            # --- Context (encoder) block(s) ---
            for e in range(self.nenc):
                bh, bw = enc_sizes[e]
                # Try up to 50 random placements to honour min_keep after
                # removing target overlap.
                best_indices = None  # type: list
                for _attempt in range(50):
                    top, left = self._sample_block_location(
                        bh, bw, self.height, self.width
                    )
                    indices = self._block_to_indices(top, left, bh, bw)
                    if self.allow_overlap:
                        kept = indices
                    else:
                        kept = [i for i in indices if i not in pred_indices_union]
                    if len(kept) >= self.min_keep:
                        best_indices = kept
                        break
                # Fallback: if we never found a valid placement, use all
                # non-target patches.
                if best_indices is None:
                    all_patches = list(range(self.num_patches))
                    if self.allow_overlap:
                        best_indices = all_patches
                    else:
                        best_indices = [
                            i for i in all_patches if i not in pred_indices_union
                        ]
                    # A smaller context is preferable to silently exposing
                    # target tokens. With no context, the task is infeasible.
                    if not best_indices:
                        raise ValueError("No non-target context exists for sampled masks")
                masks_enc[e].append(
                    torch.tensor(sorted(best_indices), dtype=torch.long)
                )

            # Store predictor indices.
            for p in range(self.npred):
                masks_pred[p].append(
                    torch.tensor(per_image_pred[p], dtype=torch.long)
                )

        # apply_masks concatenates groups along the batch dimension, so all
        # context groups need the same token count (also when nenc > 1).
        collated_masks_enc = self._truncate_and_stack(masks_enc)

        # Truncate predictor masks: GLOBAL min across all pred groups and
        # all samples, so apply_masks can torch.cat along dim=0.
        global_min_pred = min(
            t.numel() for group in masks_pred for t in group
        )
        global_min_pred = max(global_min_pred, 1)
        collated_masks_pred = []
        for group in masks_pred:
            collated_masks_pred.append(torch.stack(
                [t[:global_min_pred] for t in group], dim=0
            ))

        if self.audit_masks:
            self.last_mask_audit = [
                {
                    "intended_targets": [g[b].tolist() for g in masks_pred],
                    "targets": [g[b].tolist() for g in collated_masks_pred],
                    "context_before_collation": [g[b].tolist() for g in masks_enc],
                    "context": [g[b].tolist() for g in collated_masks_enc],
                    "target_sources": ["uniform"] * self.npred,
                } for b in range(B)
            ]

        return collated_batch, collated_masks_enc, collated_masks_pred

    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_and_stack(mask_groups):
        # type: (List[List[torch.Tensor]]) -> List[torch.Tensor]
        """Truncate all groups to their shared minimum length and stack.

        Returns tensors of shape (B, min_len), with one K across all groups.
        """
        result = []
        min_len = max(1, min(t.numel() for group in mask_groups for t in group))
        for group in mask_groups:
            result.append(torch.stack(
                [t[:min_len] for t in group], dim=0
            ))
        return result
