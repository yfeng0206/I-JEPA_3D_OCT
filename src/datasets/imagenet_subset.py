"""Deterministic ImageNet class-subset loading.

The class order is defined by an explicit WNID manifest, never by directory
sorting. Raw ImageNet files remain outside the repository under their original
access terms.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


_WNID_PATTERN = re.compile(r"^n[0-9]{8}$")
_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def load_wnids(path):
    """Load and validate an ordered WNID manifest."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    wnids = [line.strip() for line in lines if line.strip()]
    if not wnids:
        raise ValueError("WNID manifest is empty: %s" % path)
    invalid = [wnid for wnid in wnids if not _WNID_PATTERN.match(wnid)]
    if invalid:
        raise ValueError("invalid ImageNet WNIDs: %s" % invalid)
    if len(set(wnids)) != len(wnids):
        raise ValueError("WNID manifest contains duplicate classes")
    return tuple(wnids)


def canonical_wnid_bytes(wnids):
    """Return canonical UTF-8/LF bytes used for manifest hashing."""
    return ("\n".join(wnids) + "\n").encode("utf-8")


def wnid_manifest_sha256(path):
    """Hash normalized manifest content, independent of platform line endings."""
    return hashlib.sha256(canonical_wnid_bytes(load_wnids(path))).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    """Return a streaming SHA-256 for one local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atlas_selection_score(wnid, image_sha256, namespace="phase0-atlas-v1"):
    """Stable score for selecting review images without inspecting model output."""
    payload = (
        str(namespace)
        + "\0"
        + str(wnid)
        + "\0"
        + str(image_sha256)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_class_names(path, wnids=None):
    """Load WNID-to-label metadata and optionally validate exact coverage."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("class metadata must contain a nonempty classes object")
    names = {}
    for wnid, value in classes.items():
        if not _WNID_PATTERN.match(str(wnid)):
            raise ValueError("invalid class metadata WNID %r" % wnid)
        if not isinstance(value, dict):
            raise ValueError("class metadata values must be objects")
        label = value.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("class metadata labels must be nonempty strings")
        names[str(wnid)] = label.strip()
    if wnids is not None:
        expected = tuple(wnids)
        missing = [wnid for wnid in expected if wnid not in names]
        extra = [wnid for wnid in names if wnid not in set(expected)]
        if missing or extra:
            raise ValueError(
                "class metadata coverage mismatch: %d missing, %d extra"
                % (len(missing), len(extra))
            )
        manifest_hash = payload.get("wnid_manifest_sha256")
        expected_hash = hashlib.sha256(
            canonical_wnid_bytes(expected)
        ).hexdigest()
        if manifest_hash is not None and manifest_hash != expected_hash:
            raise ValueError("class metadata WNID manifest hash mismatch")
    return names


class ImageNetSubsetDataset(Dataset):
    """ImageNet subset whose labels follow an explicit WNID manifest.

    Expected layout::

        root/
          train/
            n01440764/*.JPEG
          val/
            n01440764/*.JPEG

    Args:
        root: Subset root containing ``train`` and ``val`` directories.
        split: ``train`` or ``val``.
        wnid_manifest: Ordered WNID text file.
        transform: Optional callable applied to the PIL image.
        strict: Require every WNID directory to exist and contain images.

    Returns:
        ``(image, label, sample_id)`` where ``sample_id`` is a stable relative
        path and ``label`` follows manifest order.
    """

    def __init__(
        self,
        root,
        split,
        wnid_manifest,
        transform=None,
        strict=True,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        self.wnids = load_wnids(wnid_manifest)
        self.class_to_idx = {
            wnid: index for index, wnid in enumerate(self.wnids)
        }
        self.transform = transform
        self.samples = []

        if not self.split_root.is_dir():
            raise FileNotFoundError(
                "ImageNet split directory does not exist: %s"
                % self.split_root
            )

        for wnid in self.wnids:
            class_dir = self.split_root / wnid
            if not class_dir.is_dir():
                if strict:
                    raise FileNotFoundError(
                        "missing ImageNet class directory: %s" % class_dir
                    )
                continue
            files = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in _IMAGE_SUFFIXES
            )
            if strict and not files:
                raise RuntimeError(
                    "ImageNet class contains no decodable image files: %s"
                    % class_dir
                )
            label = self.class_to_idx[wnid]
            for path in files:
                relative = path.relative_to(self.root).as_posix()
                self.samples.append((path, label, relative, wnid))

        if not self.samples:
            raise RuntimeError(
                "no ImageNet subset images found under %s" % self.split_root
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, sample_id, _ = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, sample_id

    def class_counts(self):
        """Return counts in manifest order."""
        counts = [0 for _ in self.wnids]
        for _, label, _, _ in self.samples:
            counts[label] += 1
        return {
            wnid: counts[index]
            for index, wnid in enumerate(self.wnids)
        }

    def records(self):
        """Return private local records used to build integrity manifests."""
        return tuple(
            {
                "path": os.fspath(path),
                "label": label,
                "sample_id": sample_id,
                "wnid": wnid,
            }
            for path, label, sample_id, wnid in self.samples
        )
