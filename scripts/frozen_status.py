"""Live status of the frozen MeanPool probe, against the published references.

    python scripts/frozen_status.py            # one shot
    python scripts/frozen_status.py --watch    # refresh every 30 s

Reference Test AUCs are the published frozen MeanPool numbers from
docs/experiments/frozen/oracle_meanpool_sweep.md, measured on the same
3,000-volume FairVision Test split with the same probe protocol.
"""

import argparse
import os
import re
import time

LOG = r'D:\jepa_phase0\runs\frozen_meanpool_mirage_ep100\eval.log'

REFERENCE = [
    ('random ep100', 0.8746),
    ('oracle ep100', 0.8855),
]

ENC_RE = re.compile(r'(Training|Validation|Test): (\d+)/(\d+) volumes \((\d+)s\)')
EPOCH_RE = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+\(\s*([\d.]+)s\)\s+\|\s+Train: ([\d.]+) \(AUC ([\d.]+)\)'
    r'\s+\|\s+Val: ([\d.]+)\s+\|\s+AUC: ([\d.]+)')
TEST_RE = re.compile(r'Test\s+AUC[:\s=]+([\d.]+)', re.I)
SENS_RE = re.compile(r'[Ss]ensitivity[:\s=]+([\d.]+)')
SPEC_RE = re.compile(r'[Ss]pecificity[:\s=]+([\d.]+)')
STOP_RE = re.compile(r'Early stopping at epoch (\d+)')


def show(path):
    if not os.path.isfile(path):
        print('no log at %s' % path)
        return
    with open(path, 'r', errors='replace') as fh:
        text = fh.read()

    print('Frozen MeanPool probe - MIRAGE ep100')
    print('log: %s' % path)
    print()

    epochs = EPOCH_RE.findall(text)
    if not epochs:
        enc = ENC_RE.findall(text)
        if enc:
            split, done, total, secs = enc[-1]
            done, total, secs = int(done), int(total), int(secs)
            before = {'Training': 0, 'Validation': 6000, 'Test': 7000}[split]
            overall = before + done
            rate = done / max(secs, 1)
            remain = 10000 - overall
            print('encoding features with the frozen encoder')
            print('  %s %d/%d   overall %d/10,000   %.1f vol/s'
                  % (split, done, total, overall, rate))
            print('  ~%.0f min left, then 50 probe epochs on cached features'
                  % (remain / rate / 60))
        else:
            print('starting up')
        return

    print('%-6s %-9s %-9s %-9s %s' % ('epoch', 'train_auc', 'val_loss', 'val_auc', ''))
    best = 0.0
    for ep, tot, secs, tl, ta, vl, va in epochs:
        va_f = float(va)
        mark = ''
        if va_f > best:
            best = va_f
            mark = ' *best'
        print('%-6s %-9s %-9s %-9s%s' % (ep, ta, vl, va, mark))
    print()
    print('best val AUC so far: %.4f  (epoch %s of %s)'
          % (best, [e[0] for e in epochs if float(e[6]) == best][0], epochs[0][1]))

    stop = STOP_RE.search(text)
    if stop:
        print('early stopped at epoch %s' % stop.group(1))

    test = TEST_RE.search(text)
    if test:
        print()
        print('=' * 46)
        print('%-16s %s' % ('MIRAGE ep100', test.group(1)))
        for name, auc in REFERENCE:
            print('%-16s %.4f' % (name, auc))
        mine = float(test.group(1))
        print('-' * 46)
        for name, auc in REFERENCE:
            print('vs %-13s %+.4f' % (name, mine - auc))
        sens, spec = SENS_RE.search(text), SPEC_RE.search(text)
        if sens and spec:
            print('sensitivity %s   specificity %s'
                  % (sens.group(1), spec.group(1)))
        print('=' * 46)
    else:
        print()
        print('reference Test AUC: random %.4f, oracle %.4f'
              % (REFERENCE[0][1], REFERENCE[1][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=30)
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
