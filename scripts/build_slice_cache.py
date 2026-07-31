"""Build a compact slice cache for OCT pretraining.

Why this exists
---------------
``OCTSliceDataset`` samples ``num_slices`` B-scans from each volume, but a
FairVision ``.npz`` stores the whole ``(200, 200, 200)`` stack in one
compressed member.  Reading a single 40 KB slice therefore decodes the full
8 MB volume -- roughly 200x read amplification.

That is invisible on a warm page cache but fatal on spinning media: measured on
a 7200 rpm SATA disk the training loop stalled at 7.6 img/s with the GPU idle
at 30 W and the disk pinned at its 51 MB/s sequential ceiling, i.e. ~68 days
for the run instead of ~2.9.

This script writes exactly the sampled slices into one flat ``uint8`` memmap
per split, so a sample costs one contiguous 40 KB read.  The bytes handed to
the transform are unchanged, so the cache cannot alter training results; a
regression test asserts bit-equality against the ``.npz`` path.

Usage
-----
    python scripts/build_slice_cache.py \
        --data-dir D:\\jepa_phase0\\fairvision-glaucoma\\data \
        --splits Training Validation --num-slices 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.oct_slices import SLICE_CACHE_ARRAY, SLICE_CACHE_MANIFEST

NATIVE = 200


def build_split(data_dir, split, cache_root, num_slices, report_every):
    src = os.path.join(data_dir, split)
    files = sorted(
        os.path.join(src, name)
        for name in os.listdir(src)
        if name.endswith(".npz")
    )
    if not files:
        raise RuntimeError("No .npz files under %s" % src)

    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)
    array_path = os.path.join(out_dir, SLICE_CACHE_ARRAY)
    manifest_path = os.path.join(out_dir, SLICE_CACHE_MANIFEST)
    progress_path = os.path.join(out_dir, "progress.json")

    slice_indices = np.linspace(0, 199, num=num_slices, dtype=np.int64)
    shape = (len(files), num_slices, NATIVE, NATIVE)
    total_bytes = int(np.prod(shape))
    print(
        "%s: %d volumes x %d slices -> %.1f GB at %s"
        % (split, len(files), num_slices, total_bytes / 1e9, array_path),
        flush=True,
    )

    # Resume support: a partial build records how many volumes are final.
    done = 0
    if os.path.isfile(progress_path) and os.path.isfile(array_path):
        with open(progress_path) as handle:
            state = json.load(handle)
        if (
            state.get("volumes") == [os.path.basename(f) for f in files]
            and int(state.get("num_slices", -1)) == num_slices
            and os.path.getsize(array_path) == total_bytes
        ):
            done = int(state.get("done", 0))
            print("  resuming at volume %d" % done, flush=True)

    mode = "r+" if (done and os.path.isfile(array_path)) else "w+"
    cache = np.memmap(array_path, dtype=np.uint8, mode=mode, shape=shape)

    started = time.time()
    for i in range(done, len(files)):
        with np.load(files[i], allow_pickle=True) as data:
            volume = data["oct_bscans"]
            cache[i] = volume[slice_indices]
        if (i + 1) % report_every == 0 or i + 1 == len(files):
            cache.flush()
            with open(progress_path, "w") as handle:
                json.dump(
                    {
                        "done": i + 1,
                        "num_slices": num_slices,
                        "volumes": [os.path.basename(f) for f in files],
                    },
                    handle,
                )
            elapsed = time.time() - started
            rate = (i + 1 - done) / max(elapsed, 1e-6)
            remaining = (len(files) - i - 1) / max(rate, 1e-6)
            print(
                "  %d/%d  %.1f vol/s  ETA %.1f min"
                % (i + 1, len(files), rate, remaining / 60),
                flush=True,
            )
    cache.flush()
    del cache

    with open(manifest_path, "w") as handle:
        json.dump(
            {
                "split": split,
                "source_dir": src,
                "volumes": [os.path.basename(f) for f in files],
                "num_slices": num_slices,
                "slice_indices": slice_indices.tolist(),
                "height": NATIVE,
                "width": NATIVE,
                "dtype": "uint8",
            },
            handle,
            indent=1,
        )
    if os.path.isfile(progress_path):
        os.remove(progress_path)
    print("  %s complete" % split, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["Training", "Validation"])
    ap.add_argument("--num-slices", type=int, default=100)
    ap.add_argument("--report-every", type=int, default=100)
    args = ap.parse_args()

    for split in args.splits:
        build_split(
            args.data_dir,
            split,
            args.cache_dir,
            args.num_slices,
            args.report_every,
        )


if __name__ == "__main__":
    main()
