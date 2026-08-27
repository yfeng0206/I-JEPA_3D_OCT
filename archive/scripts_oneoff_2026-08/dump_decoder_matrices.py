"""Write the final decoder stages out as plain-text matrices you can just open.

Covers only the last three stages of the pipeline:

    1. final_layer 1x1 conv   (1, 4, 128, 128)      <- the real prediction grid
    2. bilinear x8            (1, 4, 1024, 1024)    <- interpolation only
    3. argmax(dim=1)          (1024, 1024)          <- class ids

Reads the tensors already captured by decoder_stage_dump.py, so this is pure
CPU work and needs no checkpoint and no GPU.

Every matrix is written twice where it is small enough to be useful:
  *.csv  comma separated, opens directly in Excel (<=1024 cols, well inside
         Excel's 16384 column limit)
  *.txt  fixed-width aligned, readable in Notepad

Plus, for each stage, a ``trace_column`` file: a vertical walk down one column
showing all four class logits side by side and which one wins.  That file is
the clearest answer to "how does it actually classify a pixel" -- four numbers
in, one class id out, one row per pixel.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from fairvision_raw_native import V1_NAMES, MERGED_NAMES


def write_matrix(path: pathlib.Path, mat: np.ndarray, fmt: str, sep: str) -> int:
    """Write a 2-D array as text.  Returns bytes written."""
    if mat.dtype.kind in 'iu':
        rows = (sep.join(str(int(v)) for v in row) for row in mat)
    else:
        rows = (sep.join(fmt % v for v in row) for row in mat)
    text = '\n'.join(rows) + '\n'
    path.write_text(text)
    return len(text)


def trace_column(path: pathlib.Path, logits: np.ndarray, names, col: int,
                 row0: int, row1: int, tag: str) -> None:
    """One column, every class logit, and the winner -- the classifier, unrolled."""
    lines = [
        'Vertical trace down column %d of the %s grid' % (col, tag),
        'Each row is one pixel.  The 4 numbers are the raw class logits.',
        'argmax picks the LARGEST of the four -- that is the entire classifier.',
        '',
        '%6s | %11s %11s %11s %11s | %s' % (
            'row', names[0][:11], names[1][:11], names[2][:11], names[3][:11], 'WINNER'),
        '-' * 78,
    ]
    for r in range(row0, row1):
        v = logits[:, r, col]
        w = int(np.argmax(v))
        lines.append('%6d | %11.4f %11.4f %11.4f %11.4f | %d %s'
                     % (r, v[0], v[1], v[2], v[3], w, names[w]))
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', required=True, help='npz from decoder_stage_dump.py')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--full-1024', action='store_true',
                    help='also write the four full 1024x1024 logit CSVs (~7 MB each)')
    a = ap.parse_args()

    z = np.load(a.stages, allow_pickle=True)
    arm = str(z['arm'])
    title = str(z['title'])
    names = V1_NAMES if arm == 'v1' else MERGED_NAMES

    lg128 = z['logits_128'][0]        # (4, 128, 128)
    lg1024 = z['logits_1024'][0]      # (4, 1024, 1024)
    hard = z['hard_1024']             # (1024, 1024)
    hard128 = lg128.argmax(0).astype(np.uint8)

    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # rows where anything is predicted, so the excerpt lands on the retina
    rows = np.where((hard > 0).any(axis=1))[0]
    mid = int(rows.mean()) if rows.size else 512
    mid128 = mid // 8

    # ---------- stage 1: final_layer, the true prediction resolution ----------
    d1 = out / '1_final_layer_1x1conv__4x128x128'
    d1.mkdir(exist_ok=True)
    for c in range(4):
        base = 'class%d_%s_128x128' % (c, names[c].replace('/', '-'))
        n1 = write_matrix(d1 / (base + '.csv'), lg128[c], '%.4f', ',')
        n2 = write_matrix(d1 / (base + '.txt'), lg128[c], '%9.4f', ' ')
        written += [(d1.name + '/' + base + '.csv', n1), (d1.name + '/' + base + '.txt', n2)]
    n = write_matrix(d1 / 'argmax_128x128.csv', hard128, '%d', ',')
    written.append((d1.name + '/argmax_128x128.csv', n))
    n = write_matrix(d1 / 'argmax_128x128.txt', hard128, '%d', ' ')
    written.append((d1.name + '/argmax_128x128.txt', n))
    trace_column(d1 / ('trace_column%d.txt' % (lg128.shape[2] // 2)), lg128, names,
                 lg128.shape[2] // 2, max(0, mid128 - 24), min(128, mid128 + 24),
                 '128x128 final_layer')

    # ---------- stage 2: after bilinear x8 ----------
    d2 = out / '2_bilinear_x8__4x1024x1024'
    d2.mkdir(exist_ok=True)
    if a.full_1024:
        for c in range(4):
            base = 'class%d_%s_1024x1024.csv' % (c, names[c].replace('/', '-'))
            n = write_matrix(d2 / base, lg1024[c], '%.3f', ',')
            written.append((d2.name + '/' + base, n))
    # a 64x64 window is small enough to read and still shows the transition
    r0, r1 = max(0, mid - 32), min(1024, mid + 32)
    c0, c1 = 480, 544
    for c in range(4):
        base = 'WINDOW_r%d-%d_c%d-%d_class%d_%s' % (r0, r1, c0, c1, c,
                                                    names[c].replace('/', '-'))
        n = write_matrix(d2 / (base + '.txt'), lg1024[c, r0:r1, c0:c1], '%9.3f', ' ')
        written.append((d2.name + '/' + base + '.txt', n))
    trace_column(d2 / 'trace_column512.txt', lg1024, names, 512,
                 max(0, mid - 96), min(1024, mid + 96), '1024x1024 upsampled')

    # ---------- stage 3: argmax ----------
    d3 = out / '3_argmax__1024x1024'
    d3.mkdir(exist_ok=True)
    n = write_matrix(d3 / 'argmax_1024x1024.csv', hard, '%d', ',')
    written.append((d3.name + '/argmax_1024x1024.csv', n))
    n = write_matrix(d3 / ('WINDOW_r%d-%d_c%d-%d.txt' % (r0, r1, c0, c1)),
                     hard[r0:r1, c0:c1], '%d', ' ')
    written.append((d3.name + '/WINDOW.txt', n))

    frac = {names[c]: float((hard == c).mean()) for c in range(4)}
    readme = [
        'MATRIX DUMP  -  %s  (arm: %s)' % (title, arm),
        '=' * 74,
        '',
        'class ids: ' + ',  '.join('%d = %s' % (i, names[i]) for i in range(4)),
        '',
        'STAGE 1  1_final_layer_1x1conv__4x128x128/',
        '   The 1x1 conv output, shape (4, 128, 128).  This is the TRUE',
        '   prediction resolution - one value per class per 8x8 pixel block.',
        '   Raw logits, not probabilities: they are unbounded and can be',
        '   negative.  Larger = more confident.  One file per class.',
        '',
        'STAGE 2  2_bilinear_x8__4x1024x1024/',
        '   The same 4 maps stretched 8x to (4, 1024, 1024).  Pure bilinear',
        '   interpolation - NO new information is added here.  Full CSVs are',
        '   ~7 MB each so they are written only with --full-1024; the WINDOW',
        '   files show a %dx%d patch across the retina instead.' % (r1 - r0, c1 - c0),
        '',
        'STAGE 3  3_argmax__1024x1024/',
        '   argmax over the 4 classes -> one class id per pixel, (1024, 1024).',
        '   argmax_1024x1024.csv is the final hard segmentation as integers.',
        '',
        'trace_column*.txt in stages 1 and 2 are the most useful files: they',
        'walk down a single column and show all four logits plus the winner,',
        'so you can watch Elsewhere -> retina -> Choroid handovers happen.',
        '',
        'class pixel fractions in the final argmax:',
    ]
    readme += ['   %-16s %.4f' % (k, v) for k, v in frac.items()]
    readme += ['', 'files written:']
    readme += ['   %-64s %8.1f KB' % (f, n / 1024.0) for f, n in written]
    (out / 'README.txt').write_text('\n'.join(readme) + '\n')

    print('wrote %s' % out)
    for f, n in written:
        print('   %-64s %8.1f KB' % (f, n / 1024.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
