"""Compact status view for the merged-segmentation training run.

Reads MIRAGE's own per-epoch log.txt (which is flushed every epoch, unlike the
buffered stdout redirect) and prints progress, the best epoch so far, and an
ETA derived from observed wall-clock between epochs.

    python scripts/seg_run_status.py [--watch]
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time

RUN = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\mergedv2\MergedV2'
                   r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore')

# The GOALS-only baseline, for orientation only. NOT directly comparable:
# different val set (15 GOALS vs 15 GOALS + 22 Duke) and different taxonomy
# (4-class vs 3-class + ignore). The comparable number is the GOALS-test score
# from scripts/score_goals_merged.py.
BASELINE_BEST_VAL_MIOU = 90.64
BASELINE_BEST_EPOCH = 162

# GOALS presentations in the baseline run (55 images x 200 epochs).  With the
# source-balanced sampler at GOALS=0.25 the merged run shows 312 GOALS images
# per epoch, so epoch 35 is the exposure-matched control point.
EXPOSURE_MATCHED_EPOCH = 35


def render(total_epochs: int = 200) -> bool:
    log = RUN / 'log.txt'
    if not log.exists():
        print('no log yet at', log)
        return False
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        print('log.txt is empty')
        return False

    mtime = datetime.datetime.fromtimestamp(log.stat().st_mtime)
    done = len(rows)
    best = max(rows, key=lambda r: r.get('val/mean_iou', -1))

    print('=' * 74)
    print('merged segmentation run   epochs %d/%d   last write %s'
          % (done, total_epochs, mtime.strftime('%H:%M:%S')))
    print('=' * 74)
    print('  %5s %10s %10s %10s %9s' %
          ('epoch', 'train loss', 'val loss', 'val mIoU', 'val acc'))
    show = rows[-8:] if done > 8 else rows
    for r in show:
        star = ' *' if r is best else ''
        print('  %5d %10.4f %10.4f %10.2f %9.2f%s'
              % (r.get('epoch', -1), r.get('train/[Epoch] loss', float('nan')),
                 r.get('val/loss', float('nan')), r.get('val/mean_iou', float('nan')),
                 r.get('val/mean_accuracy', float('nan')), star))

    print('\n  best so far : epoch %d, val mIoU %.2f'
          % (best.get('epoch', -1), best.get('val/mean_iou', float('nan'))))
    print('  baseline    : epoch %d, val mIoU %.2f  (NOT comparable -- different'
          % (BASELINE_BEST_EPOCH, BASELINE_BEST_VAL_MIOU))
    print('                val set and taxonomy; use the GOALS-test scorer)')

    if done >= EXPOSURE_MATCHED_EPOCH:
        em = [r for r in rows if r.get('epoch') == EXPOSURE_MATCHED_EPOCH - 1]
        if em:
            print('  exposure-matched control (epoch %d): val mIoU %.2f'
                  % (EXPOSURE_MATCHED_EPOCH, em[0].get('val/mean_iou', float('nan'))))

    ckpts = sorted(RUN.glob('checkpoint*.pth'))
    if ckpts:
        print('\n  checkpoints:')
        for c in ckpts:
            print('    %-34s %7.1f MB  %s'
                  % (c.name, c.stat().st_size / 1e6,
                     datetime.datetime.fromtimestamp(
                         c.stat().st_mtime).strftime('%H:%M:%S')))

    if done >= 2:
        span = (log.stat().st_mtime -
                (RUN / 'log.txt').stat().st_ctime)
        per = span / max(done, 1)
        if per > 0:
            left = (total_epochs - done) * per
            eta = datetime.datetime.now() + datetime.timedelta(seconds=left)
            print('\n  ~%.1f min/epoch   remaining %d epochs   ETA %s'
                  % (per / 60, total_epochs - done, eta.strftime('%H:%M')))
    return done >= total_epochs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--interval', type=int, default=300)
    args = ap.parse_args()
    while True:
        finished = render(args.epochs)
        if not args.watch or finished:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
