"""Guided slice dataset: OCT B-scans paired with MIRAGE retinal envelopes.

Extends :class:`~src.datasets.oct_slices.OCTSliceDataset` without modifying it.
Each item is the usual normalised image tensor plus the spatially aligned
retinal-envelope patch grid used to bias I-JEPA target-block placement.

The guide is a *precomputed* repaired envelope (see
``scripts/mirage_precompute_guides.py``) stored bit-packed at the native
200x200 label resolution.  Because the image and guide must share one random
crop, both are transformed together by
:class:`~src.transforms.PairedRandomResizedCrop`; the guide is only reduced to
a patch grid afterwards, so it always describes the pixels the model actually
sees.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.oct_slices import OCTSliceDataset
from src.guides.mirage_envelope import (
    DEFAULT_REPAIR,
    GUIDE_SCHEMA_VERSION,
    RepairParams,
    dilate_patch_grid,
    occupancy_is_valid,
    params_fingerprint,
    patch_occupancy,
    unpack_guides,
)

NATIVE_SIZE = 200


class GuidedOCTSliceDataset(OCTSliceDataset):
    """OCT slices paired with their MIRAGE retinal-envelope guide.

    Args:
        data_dir: Split directory holding the ``.npz`` OCT volumes.
        guide_dir: Directory holding the precomputed envelope guides.  File
            names must match the volume file names exactly.
        num_slices: Slices sampled per volume (must match the guide cache).
        slice_size: Output spatial resolution.
        transform: A ``PairedRandomResizedCrop``-style callable taking
            ``(image, guide)`` and returning ``(tensor, guide_image)``.
        patch_size: Patch size of the ViT, used to build the guide grid.
        dilate_patches: Whole-patch tolerance added around the region before
            block placement, absorbing MIRAGE boundary error.
        repair_params: Thresholds used to validate the post-crop guide.
        require_guides: When True a missing guide file is an error; otherwise
            the sample falls back to an all-zero (invalid) guide so training
            can continue with uniform random placement.

    Returns:
        ``(image, guide, guide_valid)`` where ``guide`` is a ``(2, grid, grid)``
        float tensor holding the true per-patch retinal occupancy in channel 0
        and the dilated placement region in channel 1, and ``guide_valid`` is a
        scalar bool tensor.

        The two channels are kept separate on purpose: the dilated region grants
        blocks tolerance for MIRAGE's boundary error, but retina-visibility
        accounting must still be measured against the true segmentation, or
        dilation would inflate the very metric that protects the encoder's
        context.
    """

    def __init__(
        self,
        data_dir,                    # type: str
        guide_dir,                   # type: str
        num_slices=100,              # type: int
        slice_size=256,              # type: int
        transform=None,              # type: Optional[object]
        patch_size=16,               # type: int
        dilate_patches=1,            # type: int
        occupancy_threshold=0.5,     # type: float
        repair_params=DEFAULT_REPAIR,  # type: RepairParams
        require_guides=True,         # type: bool
        slice_cache=None,            # type: Optional[str]
    ):
        super(GuidedOCTSliceDataset, self).__init__(
            data_dir=data_dir,
            num_slices=num_slices,
            slice_size=slice_size,
            transform=transform,
            slice_cache=slice_cache,
        )
        self.guide_dir = guide_dir
        self.patch_size = patch_size
        self.dilate_patches = int(dilate_patches)
        self.occupancy_threshold = float(occupancy_threshold)
        self.repair_params = repair_params
        self.require_guides = require_guides
        self.expected_fingerprint = params_fingerprint(repair_params)
        if slice_size % patch_size:
            raise ValueError(
                "slice_size %d is not divisible by patch_size %d"
                % (slice_size, patch_size)
            )
        self.grid_size = slice_size // patch_size

        if require_guides:
            missing = [
                os.path.basename(path)
                for path in self.file_paths
                if not os.path.isfile(
                    os.path.join(guide_dir, os.path.basename(path))
                )
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing %d MIRAGE guide files under %s (first: %s)"
                    % (len(missing), guide_dir, missing[:3])
                )

    # ------------------------------------------------------------------

    def _load_soft_guide(self, file_index, slice_within):
        # type: (int, int) -> Optional[np.ndarray]
        """Return (2, NATIVE, NATIVE) float32 class scores, or None.

        Schema 2 caches P_inner and P_choroid POST-softmax at native
        resolution, which is the only form that survives the paired random
        crop: the guide is cropped with the image and pooled to the token grid
        afterwards, so the softmax must already have happened.  Schema 1 stored
        a single 1-bit envelope, which merges inner retina and choroid into one
        class; `build_targets` was validated on the two-class form.
        """
        name = os.path.basename(self.file_paths[file_index])
        path = os.path.join(self.guide_dir, name)
        if not os.path.isfile(path):
            return None
        with np.load(path, allow_pickle=False) as cache:
            if "soft_scores" not in cache:
                return None
            source = str(cache["source_filename"].item())
            if source != name:
                raise RuntimeError("Guide %s was built from %s" % (path, source))
            if int(cache["schema_version"]) != 2:
                raise RuntimeError(
                    "Guide %s has schema %d, expected 2"
                    % (path, int(cache["schema_version"]))
                )
            if not np.array_equal(cache["slice_indices"], self.slice_indices):
                raise RuntimeError(
                    "Guide %s slice indices do not match the dataset" % path
                )
            soft = cache["soft_scores"][slice_within]
        if soft.shape != (2, NATIVE_SIZE, NATIVE_SIZE):
            raise RuntimeError(
                "Guide %s has soft shape %s, expected %s"
                % (path, soft.shape, (2, NATIVE_SIZE, NATIVE_SIZE))
            )
        return soft.astype(np.float32) / 255.0

    def _load_guide(self, file_index, slice_within):
        # type: (int, int) -> Tuple[Optional[np.ndarray], bool]
        """Return the native-resolution envelope for one slice."""
        name = os.path.basename(self.file_paths[file_index])
        guide_path = os.path.join(self.guide_dir, name)
        if not os.path.isfile(guide_path):
            return None, False
        soft = self._load_soft_guide(file_index, slice_within)
        if soft is not None:
            # A schema-2 cache has no packed envelope and no repair
            # fingerprint; the envelope is simply where the summed class score
            # says anatomy.  Validity is still re-checked after cropping by the
            # caller, which is the check that actually matters.
            return (soft.sum(axis=0) >= 0.5), True
        with np.load(guide_path, allow_pickle=False) as cache:
            # The guide must describe the same volume, the same slice ordering
            # and the same repair logic as the image, otherwise the mask points
            # at the wrong anatomy -- fail loudly rather than train on a silent
            # mismatch or on a stale cache from earlier repair parameters.
            source = str(cache["source_filename"].item())
            if source != name:
                raise RuntimeError(
                    "Guide %s was built from %s" % (guide_path, source)
                )
            schema = int(cache["schema_version"])
            if schema != GUIDE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Guide %s has schema %d, expected %d"
                    % (guide_path, schema, GUIDE_SCHEMA_VERSION)
                )
            fingerprint = str(cache["params_fingerprint"].item())
            if fingerprint != self.expected_fingerprint:
                raise RuntimeError(
                    "Guide %s was built with repair params %s, expected %s"
                    % (guide_path, fingerprint, self.expected_fingerprint)
                )
            if not np.array_equal(cache["slice_indices"], self.slice_indices):
                raise RuntimeError(
                    "Guide %s slice indices do not match the dataset" % guide_path
                )
            packed_all = cache["packed_envelopes"]
            expected_bytes = (NATIVE_SIZE * NATIVE_SIZE + 7) // 8
            if packed_all.shape != (self.num_slices, expected_bytes):
                raise RuntimeError(
                    "Guide %s has packed shape %s, expected %s"
                    % (
                        guide_path,
                        packed_all.shape,
                        (self.num_slices, expected_bytes),
                    )
                )
            packed = packed_all[slice_within : slice_within + 1]
            valid = bool(cache["valid"][slice_within])
        envelope = unpack_guides(packed, (NATIVE_SIZE, NATIVE_SIZE))[0]
        return envelope, valid

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        file_idx = idx // self.num_slices
        slice_within = idx % self.num_slices

        slice_2d = self.read_slice(file_idx, slice_within)

        image = Image.fromarray(slice_2d, mode="L").resize(
            (self.slice_size, self.slice_size), Image.BILINEAR
        ).convert("RGB")

        envelope, valid = self._load_guide(file_idx, slice_within)
        if envelope is None:
            if self.require_guides:
                raise FileNotFoundError(
                    "No guide for %s" % self.file_paths[file_idx]
                )
            envelope = np.zeros((NATIVE_SIZE, NATIVE_SIZE), dtype=bool)
            valid = False

        # Schema 2 additionally carries per-class soft scores.  They must go
        # through the SAME crop as the image, so they ride along in the R and G
        # channels of an RGB image and are split out afterwards.  Bilinear here,
        # unlike the envelope's nearest, because these are continuous scores.
        soft = self._load_soft_guide(file_idx, slice_within)
        soft_image = None
        if soft is not None:
            rgb = np.zeros((NATIVE_SIZE, NATIVE_SIZE, 3), np.uint8)
            rgb[..., 0] = np.clip(soft[0] * 255, 0, 255).astype(np.uint8)
            rgb[..., 1] = np.clip(soft[1] * 255, 0, 255).astype(np.uint8)
            soft_image = Image.fromarray(rgb, mode="RGB").resize(
                (self.slice_size, self.slice_size), Image.BILINEAR
            )

        # Nearest resize keeps the hard envelope hard; the paired transform then
        # applies the SAME crop to image and guide.
        guide_image = Image.fromarray(
            envelope.astype(np.uint8) * 255, mode="L"
        ).resize((self.slice_size, self.slice_size), Image.NEAREST)

        if self.transform is None:
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
            cropped_guide = guide_image
            cropped_soft = soft_image
        elif soft_image is not None:
            # One call, one crop draw: the soft scores MUST ride along with the
            # image and envelope. Cropping them in a second call would draw a
            # different rectangle and silently point the guide at the wrong
            # anatomy.
            tensor, cropped_guide, cropped_soft = self.transform(
                image, guide_image, soft_image
            )
        else:
            tensor, cropped_guide = self.transform(image, guide_image)
            cropped_soft = None

        guide_mask = np.asarray(cropped_guide) > 127
        grid = patch_occupancy(guide_mask, patch_size=self.patch_size)
        # Validity is re-checked AFTER cropping: a healthy guide can still land
        # off the retina, and such a sample must fall back to random placement.
        valid = bool(valid and occupancy_is_valid(grid, self.repair_params))
        placement = grid >= self.occupancy_threshold
        if self.dilate_patches > 0:
            placement = dilate_patch_grid(placement, self.dilate_patches)
        channels = [grid.astype(np.float32), placement.astype(np.float32)]
        if cropped_soft is not None:
            arr = np.asarray(cropped_soft, np.float32) / 255.0
            p = self.patch_size
            g = self.grid_size
            for c in (0, 1):
                channels.append(
                    arr[..., c].reshape(g, p, g, p).mean(axis=(1, 3))
                )
        guide = np.stack(channels, axis=0)
        return (
            tensor,
            torch.from_numpy(np.ascontiguousarray(guide)),
            torch.tensor(valid, dtype=torch.bool),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def collate(batch):
        """Stack ``(image, guide, valid)`` triples for the training loader."""
        images = torch.stack([item[0] for item in batch], dim=0)
        guides = torch.stack([item[1] for item in batch], dim=0)
        valid = torch.stack([item[2] for item in batch], dim=0)
        return images, guides, valid
