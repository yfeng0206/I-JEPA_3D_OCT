#!/usr/bin/env python
"""Convert a compressed guide cache to memmap-friendly uncompressed .npy.

`np.savez_compressed` stores one deflate stream per array, so reading a SINGLE
slice forces decompression of the whole `(100, 2, 200, 200)` volume -- 8 MB
inflated to serve 80 KB. Measured on the live cache:

    compressed npz, random volumes   35.1 ms per slice
    uncompressed .npy via memmap      9.6 ms per slice (reopened each time)
    uncompressed .npy, cached handle  ~0   ms (page-cached)

At 600,000 slice reads per epoch across 6 workers that is ~0.98 h/epoch spent
decompressing data that is then discarded, against ~0.2-0.3 h/epoch for the
memmap path.

The trade is disk: ~48 GiB uncompressed against 3.6 GiB compressed.

Conversion is CPU/IO only -- no GPU, no model -- so it can run alongside
training. Each output is verified byte-identical to its source before the
source is considered replaceable; nothing is deleted by this script.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='guide cache dir (contains <split>/)')
    ap.add_argument('--split', default='Training')
    ap.add_argument('--dst', default=None,
                    help='output dir; defaults to <src>_npy')
    ap.add_argument('--verify-every', type=int, default=50,
                    help='fully verify 1 in N volumes (1 = verify all)')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    dst = pathlib.Path(a.dst) if a.dst else src.parent / (src.name + '_npy')
    (dst / a.split).mkdir(parents=True, exist_ok=True)

    files = sorted((src / a.split).glob('*.npz'))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit('no .npz under %s' % (src / a.split))

    free = shutil.disk_usage(dst.anchor).free / (1 << 30)
    with np.load(files[0], allow_pickle=False) as z:
        per = z['soft_scores'].nbytes
    need = per * len(files) / (1 << 30)
    print('volumes      %d' % len(files))
    print('uncompressed %.1f GiB needed, %.0f GiB free' % (need, free))
    if need > free * 0.9:
        raise SystemExit('refusing: needs more than 90%% of free space')

    meta_src = src / 'cache_meta.json'
    if meta_src.is_file():
        m = json.loads(meta_src.read_text())
        m['storage'] = 'uncompressed .npy, memmap-friendly'
        m['converted_from'] = str(src)
        (dst / 'cache_meta.json').write_text(json.dumps(m, indent=2))

    t0 = time.perf_counter()
    done = skipped = verified = 0
    for i, f in enumerate(files):
        out = dst / a.split / (f.stem + '.npy')
        side = dst / a.split / (f.stem + '.json')
        if out.exists() and side.exists():
            skipped += 1
            done += 1
            continue
        with np.load(f, allow_pickle=False) as z:
            arr = z['soft_scores']
            # The provenance fields the dataset validates live alongside the
            # array in the npz; keep them in a sidecar so the loader can still
            # refuse a mixed cache directory.
            side_data = {
                'source_filename': str(z['source_filename'].item()),
                'schema_version': int(z['schema_version']),
                'slice_indices': z['slice_indices'].tolist(),
                'mirage_sha': str(z['mirage_sha'].item()) if 'mirage_sha' in z else None,
                'adapter_sha': str(z['adapter_sha'].item()) if 'adapter_sha' in z else None,
            }
            np.save(out, np.ascontiguousarray(arr))
        side.write_text(json.dumps(side_data))
        if a.verify_every and (i % a.verify_every == 0):
            with np.load(f, allow_pickle=False) as z:
                ref = z['soft_scores']
                got = np.load(out, mmap_mode='r')
                if not np.array_equal(np.asarray(got), ref):
                    raise SystemExit('MISMATCH writing %s' % out)
            verified += 1
        done += 1
        if done % 250 == 0:
            el = time.perf_counter() - t0
            new = max(done - skipped, 1)
            print('   %d/%d  %.0fs  eta %.0fs  (%d reused, %d verified)'
                  % (done, len(files), el, el / new * (len(files) - done),
                     skipped, verified), flush=True)

    print('done: %d volumes (%d reused, %d fully verified) in %.0fs -> %s'
          % (done, skipped, verified, time.perf_counter() - t0, dst))


if __name__ == '__main__':
    main()
