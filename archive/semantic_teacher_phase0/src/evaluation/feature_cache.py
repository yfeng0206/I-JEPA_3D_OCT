"""Resumable, integrity-checked frozen feature caches."""

import hashlib
import json
import os
import shutil
import tempfile

import numpy as np


_SCHEMA_VERSION = 1


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp-",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def dataset_snapshot(dataset, output_path, overwrite=False):
    """Write a content-addressed manifest for one local dataset split."""
    records = dataset.records()
    expected = [
        {
            "sample_id": record["sample_id"],
            "label": int(record["label"]),
            "wnid": record["wnid"],
        }
        for record in records
    ]
    identity_hash = stable_json_sha256(expected)
    if os.path.isfile(output_path) and not overwrite:
        with open(output_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("identity_sha256") != identity_hash:
            raise RuntimeError(
                "existing dataset snapshot does not match dataset order: %s"
                % output_path
            )
        if len(existing.get("files", ())) != len(records):
            raise RuntimeError("existing dataset snapshot has the wrong count")
        for record, locked in zip(records, existing["files"]):
            stat = os.stat(record["path"])
            if (
                locked.get("sample_id") != record["sample_id"]
                or int(locked.get("bytes", -1)) != int(stat.st_size)
                or int(locked.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
            ):
                raise RuntimeError(
                    "dataset file changed after snapshot: %s; regenerate the "
                    "snapshot and dependent feature caches explicitly"
                    % record["sample_id"]
                )
        return existing

    files = []
    for record in records:
        path = record["path"]
        stat = os.stat(path)
        files.append(
            {
                "sample_id": record["sample_id"],
                "label": int(record["label"]),
                "wnid": record["wnid"],
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(path),
            }
        )
    snapshot = {
        "schema_version": _SCHEMA_VERSION,
        "split": dataset.split,
        "count": len(files),
        "identity_sha256": identity_hash,
        "files": files,
    }
    snapshot["content_sha256"] = stable_json_sha256(files)
    atomic_write_json(output_path, snapshot)
    return snapshot


class FeatureCacheWriter:
    """Persist one frozen feature matrix with batch-level resume safety."""

    def __init__(
        self,
        output_dir,
        provenance,
        sample_ids,
        labels,
        overwrite=False,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.work_dir = self.output_dir + ".inprogress"
        self.provenance = dict(provenance)
        self.sample_ids = [str(value) for value in sample_ids]
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if len(self.sample_ids) != self.labels.size:
            raise ValueError("sample ID and label counts differ")
        if not self.sample_ids:
            raise ValueError("feature cache cannot be empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique")

        self.provenance_hash = stable_json_sha256(self.provenance)
        self.sample_ids_hash = stable_json_sha256(self.sample_ids)
        self.labels_hash = hashlib.sha256(self.labels.tobytes()).hexdigest()
        self.progress = None
        self._features = None

        if overwrite:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            shutil.rmtree(self.work_dir, ignore_errors=True)
        if os.path.isdir(self.output_dir):
            manifest = self._read_json(
                os.path.join(self.output_dir, "manifest.json")
            )
            self._validate_identity(manifest)
            self.progress = {
                "completed": int(manifest["count"]),
                "feature_dim": int(manifest["feature_dim"]),
            }
            return

        if os.path.isdir(self.work_dir):
            self.progress = self._read_json(
                os.path.join(self.work_dir, "progress.json")
            )
            self._validate_identity(self.progress)
            feature_dim = self.progress.get("feature_dim")
            if feature_dim is not None:
                self._features = np.lib.format.open_memmap(
                    os.path.join(self.work_dir, "features.npy"),
                    mode="r+",
                    dtype=np.float32,
                    shape=(len(self.sample_ids), int(feature_dim)),
                )
            return

        os.makedirs(self.work_dir)
        np.save(
            os.path.join(self.work_dir, "labels.npy"),
            self.labels,
            allow_pickle=False,
        )
        atomic_write_json(
            os.path.join(self.work_dir, "sample_ids.json"),
            self.sample_ids,
        )
        self.progress = {
            "schema_version": _SCHEMA_VERSION,
            "provenance": self.provenance,
            "provenance_sha256": self.provenance_hash,
            "sample_ids_sha256": self.sample_ids_hash,
            "labels_sha256": self.labels_hash,
            "count": len(self.sample_ids),
            "completed": 0,
            "feature_dim": None,
            "extraction_seconds": 0.0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_reserved_bytes": 0,
            "feature_metadata": {},
        }
        self._save_progress()

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _validate_identity(self, value):
        checks = (
            ("provenance_sha256", self.provenance_hash),
            ("sample_ids_sha256", self.sample_ids_hash),
            ("labels_sha256", self.labels_hash),
            ("count", len(self.sample_ids)),
        )
        mismatches = [
            key for key, expected in checks if value.get(key) != expected
        ]
        if mismatches:
            raise RuntimeError(
                "feature cache identity mismatch for %s: %s; use overwrite "
                "only after confirming the new dataset/model configuration"
                % (self.output_dir, ", ".join(mismatches))
            )

    def _save_progress(self):
        atomic_write_json(
            os.path.join(self.work_dir, "progress.json"),
            self.progress,
        )

    def _close_features(self):
        if self._features is None:
            return
        self._features.flush()
        memory_map = getattr(self._features, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        self._features = None

    @property
    def is_complete(self):
        return os.path.isdir(self.output_dir)

    @property
    def completed(self):
        return int(self.progress["completed"])

    @property
    def count(self):
        return len(self.sample_ids)

    def write_batch(
        self,
        features,
        start,
        elapsed_seconds,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
        feature_metadata=None,
    ):
        if self.is_complete:
            raise RuntimeError("cannot append to a completed feature cache")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("features must have shape (N, D)")
        if not np.isfinite(values).all():
            raise ValueError("features contain NaN or Inf")
        start = int(start)
        stop = start + values.shape[0]
        if start != self.completed:
            raise ValueError(
                "cache writes must be contiguous: expected %d, got %d"
                % (self.completed, start)
            )
        if stop > self.count:
            raise ValueError("feature batch exceeds expected sample count")

        feature_dim = int(values.shape[1])
        if self._features is None:
            recorded_dim = self.progress.get("feature_dim")
            if recorded_dim is not None and int(recorded_dim) != feature_dim:
                raise ValueError("feature dimension changed while resuming")
            self._features = np.lib.format.open_memmap(
                os.path.join(self.work_dir, "features.npy"),
                mode="w+",
                dtype=np.float32,
                shape=(self.count, feature_dim),
            )
            self.progress["feature_dim"] = feature_dim
            self.progress["feature_metadata"] = dict(feature_metadata or {})
        elif int(self.progress["feature_dim"]) != feature_dim:
            raise ValueError("feature dimension changed between batches")

        self._features[start:stop] = values
        self._features.flush()
        self.progress["completed"] = stop
        self.progress["extraction_seconds"] += float(elapsed_seconds)
        self.progress["peak_cuda_allocated_bytes"] = max(
            int(self.progress["peak_cuda_allocated_bytes"]),
            int(peak_allocated_bytes),
        )
        self.progress["peak_cuda_reserved_bytes"] = max(
            int(self.progress["peak_cuda_reserved_bytes"]),
            int(peak_reserved_bytes),
        )
        self._save_progress()

    def finalize(self, extra_manifest=None):
        if self.is_complete:
            return self._read_json(
                os.path.join(self.output_dir, "manifest.json")
            )
        if self.completed != self.count:
            raise RuntimeError(
                "cannot finalize incomplete cache: %d/%d"
                % (self.completed, self.count)
            )
        if self.progress.get("feature_dim") is None:
            raise RuntimeError("feature cache has no feature dimension")
        self._close_features()

        features_path = os.path.join(self.work_dir, "features.npy")
        labels_path = os.path.join(self.work_dir, "labels.npy")
        sample_ids_path = os.path.join(self.work_dir, "sample_ids.json")
        seconds = float(self.progress["extraction_seconds"])
        manifest = {
            key: value
            for key, value in self.progress.items()
            if key != "completed"
        }
        manifest.update(
            {
                "completed": True,
                "feature_dim": int(self.progress["feature_dim"]),
                "features_sha256": file_sha256(features_path),
                "features_bytes": int(os.path.getsize(features_path)),
                "labels_file_sha256": file_sha256(labels_path),
                "sample_ids_file_sha256": file_sha256(sample_ids_path),
                "images_per_second": (
                    float(self.count) / seconds if seconds > 0 else None
                ),
                "gpu_hours": seconds / 3600.0,
            }
        )
        if extra_manifest:
            manifest.update(dict(extra_manifest))
        atomic_write_json(
            os.path.join(self.work_dir, "manifest.json"), manifest
        )
        os.unlink(os.path.join(self.work_dir, "progress.json"))
        os.replace(self.work_dir, self.output_dir)
        return manifest


def load_feature_cache(path, verify=True, mmap_mode="r"):
    """Load a completed cache and optionally verify all persisted hashes."""
    path = os.path.abspath(path)
    manifest_path = os.path.join(path, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("completed"):
        raise RuntimeError("feature cache manifest is not complete")
    features_path = os.path.join(path, "features.npy")
    labels_path = os.path.join(path, "labels.npy")
    sample_ids_path = os.path.join(path, "sample_ids.json")
    if verify:
        checks = (
            (features_path, manifest["features_sha256"]),
            (labels_path, manifest["labels_file_sha256"]),
            (sample_ids_path, manifest["sample_ids_file_sha256"]),
        )
        for file_path, expected in checks:
            observed = file_sha256(file_path)
            if observed != expected:
                raise RuntimeError(
                    "feature cache checksum mismatch: %s" % file_path
                )
    features = np.load(
        features_path, mmap_mode=mmap_mode, allow_pickle=False
    )
    labels = np.load(labels_path, mmap_mode=mmap_mode, allow_pickle=False)
    with open(sample_ids_path, "r", encoding="utf-8") as handle:
        sample_ids = json.load(handle)
    if (
        features.shape != (int(manifest["count"]), int(manifest["feature_dim"]))
        or labels.shape != (int(manifest["count"]),)
        or len(sample_ids) != int(manifest["count"])
    ):
        raise RuntimeError("feature cache arrays do not match manifest shapes")
    return {
        "features": features,
        "labels": labels,
        "sample_ids": sample_ids,
        "manifest": manifest,
    }
