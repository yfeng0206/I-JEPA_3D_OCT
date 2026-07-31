"""Live status of the MIRAGE run, compared against the oracle reference.

Reads the training log and prints each completed epoch beside the matching row
from ``docs/experiments/pretraining/oracle_100ep.md`` so drift is obvious at a
glance.  Safe to run while training is in progress.

    python scripts/run_status.py
    python scripts/run_status.py --watch      # refresh every 60 s
"""

import argparse
import os
import re
import sys
import time

LOG = r'D:\jepa_phase0\runs\patch_mirage_envelope\train.log'

# From docs/experiments/pretraining/oracle_100ep.md -- the arm this run must be
# compared against.  epoch -> (train, val, cos_sim, rep_div)
ORACLE = {
    26: (0.1186, 0.1202, 0.875, 0.282),
    30: (0.1197, 0.1242, 0.838, 0.298),
    35: (0.1232, 0.1310, 0.850, 0.247),
    50: (0.1316, 0.1400, 0.807, 0.268),
    60: (0.1388, 0.1489, 0.766, 0.266),
    75: (0.1404, 0.1507, 0.823, 0.262),
    88: (0.1335, 0.1454, 0.863, 0.226),
    95: (0.1306, 0.1449, 0.801, 0.222),
    100: (0.1303, 0.1432, 0.844, 0.210),
}

EPOCH_RE = re.compile(
    r'Epoch (\d+)/\d+\s+\((\d+)s\)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)')
DIAG_RE = re.compile(
    r'\[DIAG\] Epoch (\d+): cos_sim=([\d.-]+).*?l2_dist=([\d.]+)\s+rep_diversity=([\d.]+)')
ITER_RE = re.compile(r'Epoch (\d+)/\d+ \| Iter (\d+)/(\d+)\] loss=([\d.]+)')
RAMP_RE = re.compile(r'Curriculum\] ep=(\d+).*?r_t=([\d.]+)')


def parse(path):
    epochs, diags, ramps, last_iter = {}, {}, {}, None
    if not os.path.isfile(path):
        return epochs, diags, ramps, last_iter
    with open(path, 'r', errors='replace') as handle:
        for line in handle:
            m = EPOCH_RE.search(line)
            if m:
                epochs[int(m.group(1))] = (
                    int(m.group(2)), float(m.group(3)), float(m.group(4)))
                continue
            m = DIAG_RE.search(line)
            if m:
                diags[int(m.group(1))] = (
                    float(m.group(2)), float(m.group(3)), float(m.group(4)))
                continue
            m = RAMP_RE.search(line)
            if m:
                ramps[int(m.group(1))] = float(m.group(2))
                continue
            m = ITER_RE.search(line)
            if m:
                last_iter = (int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), float(m.group(4)))
    return epochs, diags, ramps, last_iter


def show(path):
    epochs, diags, ramps, last_iter = parse(path)
    print('MIRAGE thr 0.25 run   vs   oracle_100ep.md')
    print('log: %s' % path)
    print()
    if not epochs:
        print('No completed epochs yet.')
    else:
        print('%-6s %-5s %-8s %-8s %-8s %-8s   %-17s %s'
              % ('epoch', 'r_t', 'train', 'val', 'cos_sim', 'rep_div',
                 'oracle train/val', 'note'))
        print('-' * 96)
        for ep in sorted(epochs):
            secs, train, val = epochs[ep]
            cos, l2, rep = diags.get(ep, (float('nan'),) * 3)
            # The log prints epoch+1 while the curriculum uses the loop index.
            rt = ramps.get(ep - 1)
            ref = ORACLE.get(ep)
            ref_s = '%.4f / %.4f' % (ref[0], ref[1]) if ref else '-'
            note = ''
            if ref:
                note = 'train %+.4f  val %+.4f' % (train - ref[0], val - ref[1])
            flag = ''
            if rep == rep and rep > 0.60:
                flag = '  <-- rep_diversity HIGH, watch for collapse'
            elif rep == rep and rep < 0.10:
                flag = '  <-- rep_diversity LOW'
            print('%-6d %-5s %-8.4f %-8.4f %-8.4f %-8.4f   %-17s %s%s'
                  % (ep, ('%.1f' % rt) if rt is not None else '?', train, val,
                     cos, rep, ref_s, note, flag))
        print()
        avg = sum(v[0] for v in epochs.values()) / len(epochs)
        done = max(epochs)
        print('mean epoch time %.0f s (%.2f h)   -> %d epochs left ~ %.1f days'
              % (avg, avg / 3600, 100 - done, (100 - done) * avg / 86400))
    if last_iter:
        ep, itr, tot, loss = last_iter
        print()
        print('in progress: epoch %d, iter %d/%d (%.0f%%), running loss %.4f'
              % (ep, itr, tot, 100.0 * itr / tot, loss))
    print()
    print('healthy bands (oracle): rep_diversity 0.17-0.33 (1.0 = collapsed), '
          'cos_sim 0.69-0.88')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=60)
    args = ap.parse_args()
    while True:
        if args.watch:
            os.system('cls' if os.name == 'nt' else 'clear')
        show(args.log)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
