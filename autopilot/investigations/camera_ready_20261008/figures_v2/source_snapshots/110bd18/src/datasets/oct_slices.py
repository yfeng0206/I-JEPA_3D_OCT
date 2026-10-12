"""Dataset for patch-level I-JEPA: loads individual 2-D OCT B-scans.

Each FairVision ``.npz`` volume has key ``oct_bscans`` with shape
(200, 200, 200) (uint8) and a ``glaucoma`` label.  This dataset
uniformly samples ``num_slices`` axial slices from each volume and
exposes every slice as an independent sample.

Total dataset size: ``num_volumes * num_slices``.

Intended for self-supervised pretraining -- no labels are returned.
"""

import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SLICE_CACHE_MANIFEST = "slice_cache.json"
SLICE_CACHE_ARRAY = "slice_cache.u8"


class OCTSliceDataset(Dataset):
    """Individual-slice dataset for patch-level I-JEPA pretraining.

    Args:
        data_dir: Path to the split directory (e.g. ``Training/`` or
            ``Validation/``) that contains ``.npz`` files.
        num_slices: Number of slices to uniformly sample from each volume.
        slice_size: Target spatial resolution (square) of each slice.
        transform: Optional ``torchvision.transforms`` applied to the
            PIL image *before* final tensor conversion.
        slice_cache: Optional directory holding a prebuilt slice cache (see
            ``scripts/build_slice_cache.py``).  Reading a slice from the
            ``.npz`` costs a full 8 MB volume decode to obtain one 40 KB
            slice -- a 200x read amplification that makes training disk-bound
            on spinning media.  The cache stores exactly the sampled slices so
            a sample costs one 40 KB read.  The bytes returned are identical;
            this is purely an I/O layout change.
    """

    def __init__(
        self,
        data_dir,          # type: str
        num_slices=32,     # type: int
        slice_size=256,    # type: int
        transform=None,    # type: Optional[object]
        slice_cache=None,  # type: Optional[str]
    ):
        super(OCTSliceDataset, self).__init__()

        self.data_dir = data_dir
        self.num_slices = num_slices
        self.slice_size = slice_size
        self.transform = transform

        # Discover all .npz files.
        pattern = os.path.join(data_dir, "*.npz")
        self.file_paths = sorted(glob.glob(pattern))
        if len(self.file_paths) == 0:
            raise RuntimeError(
                "No .npz files found in {!r}. Check the data_dir path.".format(
                    data_dir
                )
            )

        # Pre-compute the slice indices (integers into the depth axis).
        # np.linspace gives evenly-spaced indices across the 200-slice volume.
        self.slice_indices = np.linspace(
            0, 199, num=num_slices, dtype=np.int64
        )

        self.slice_cache = slice_cache
        self._cache_meta = None
        self._cache_mm = None
        if slice_cache:
            self._cache_meta = self._load_cache_manifest(slice_cache)

    # ------------------------------------------------------------------

    def _load_cache_manifest(self, cache_dir):
        """Validate that a cache describes exactly this dataset."""
        manifest_path = os.path.join(cache_dir, SLICE_CACHE_MANIFEST)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "No slice cache manifest at %s" % manifest_path
            )
        with open(manifest_path, "r") as handle:
            meta = json.load(handle)
        if int(meta["num_slices"]) != int(self.num_slices):
            raise RuntimeError(
                "Slice cache %s holds %d slices/volume, dataset wants %d"
                % (cache_dir, meta["num_slices"], self.num_slices)
            )
        if not np.array_equal(
            np.asarray(meta["slice_indices"], dtype=np.int64), self.slice_indices
        ):
            raise RuntimeError(
                "Slice cache %s was built with different slice indices" % cache_dir
            )
        names = [os.path.basename(p) for p in self.file_paths]
        if list(meta["volumes"]) != names:
            raise RuntimeError(
                "Slice cache %s covers %d volumes that do not match the %d "
                "volumes found in %s"
                % (cache_dir, len(meta["volumes"]), len(names), self.data_dir)
            )
        return meta

    def _memmap(self):
        """Open the cache lazily so each DataLoader worker gets its own view."""
        if self._cache_mm is None:
            meta = self._cache_meta
            self._cache_mm = np.memmap(
                os.path.join(self.slice_cache, SLICE_CACHE_ARRAY),
                dtype=np.uint8,
                mode="r",
                shape=(
                    len(meta["volumes"]),
                    int(meta["num_slices"]),
                    int(meta["height"]),
                    int(meta["width"]),
                ),
            )
        return self._cache_mm

    def __getstate__(self):
        # np.memmap holds an OS handle that must not cross a process boundary.
        state = self.__dict__.copy()
        state["_cache_mm"] = None
        return state

    def read_slice(self, file_idx, slice_within):
        # type: (int, int) -> np.ndarray
        """Return one native-resolution slice as a (H, W) uint8 array."""
        if self._cache_meta is not None:
            return np.asarray(self._memmap()[file_idx, slice_within])
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        try:
            # NOTE: npz member access decodes the WHOLE (200, 200, 200) array.
            volume = data["oct_bscans"]
            depth_idx = int(self.slice_indices[slice_within])
            return np.array(volume[depth_idx])
        finally:
            data.close()

    # ------------------------------------------------------------------

    def __len__(self):
        # type: () -> int
        return len(self.file_paths) * self.num_slices

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # type: (int) -> torch.Tensor
        """Return a single slice as a (3, slice_size, slice_size) tensor."""
        file_idx = idx // self.num_slices
        slice_within = idx % self.num_slices

        slice_2d = self.read_slice(file_idx, slice_within)  # (200, 200) uint8

        # Resize to target resolution using PIL (bilinear).
        pil_img = Image.fromarray(slice_2d, mode="L")
        pil_img = pil_img.resize(
            (self.slice_size, self.slice_size), Image.BILINEAR
        )

        # Convert to 3-channel RGB (duplicate the grayscale channel).
        pil_img = pil_img.convert("RGB")

        # Apply user-supplied transforms (e.g. RandomResizedCrop, etc.).
        if self.transform is not None:
            pil_img = self.transform(pil_img)
            # If the transform already produces a tensor, return as-is.
            if isinstance(pil_img, torch.Tensor):
                return pil_img

        # Fallback: manual conversion to tensor normalised to [0, 1].
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        return tensor
