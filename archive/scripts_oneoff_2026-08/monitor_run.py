#!/usr/bin/env python3
"""Health monitor for the anatomy-guided pretraining run.

Reads the log and stdout, reports status, and FLAGS anomalies without ever
stopping the run.  Designed to be called periodically and to say nothing
alarming unless something is genuinely wrong.

Checks:
  loss          non-finite, or a sustained rise beyond noise
  masking       fallback rate, on-region, context tokens, target size
  ramp          whether guidance is actually engaged for the current epoch
  progress      iterations/epoch pace, ETA
  checkpoints   that save_every is producing files
  hardware      VRAM headroom, and whether the process is still alive
"""
from __future__ import annotations

import csv
import json
import pathlib
import os
import re
import subprocess
import sys
import time

RUN = pathlib.Path(os.environ.get(
    'JEPA_MONITOR_RUN', r'D:\jepa_phase0\runs\anatomy_v2_ep25'))
LOG = RUN / 'jepa_patch_mirage-log.csv'
OUT = pathlib.Path(os.environ.get(
    'JEPA_MONITOR_LOG', r'D:\jepa_phase0\runs\anatomy_v2_ep25_stdout.log'))
STATE = RUN / 'monitor_state.json'

TOTAL_EPOCHS = int(os.environ.get('JEPA_MONITOR_TOTAL', '100'))
START_EPOCH = int(os.environ.get('JEPA_MONITOR_START', '25'))
# Fallback only. The real pace is measured from the iteration log below; a
# hardcoded constant combined with a missing stdout file is exactly how this
# monitor once reported a 160-hour ETA for a run that costs ~1.33 h/epoch.
SEC_PER_EPOCH = 4800.0

MIRAGE_RE = re.compile(
    r'patches/block=([\d.]+)\s+unique_targets=([\d.]+)\s+context=([\d.]+)\s+'
    r'on_region=([\d.]+)\s+background=([\d.]+)\s+fallbacks=(\d+)')
ITER_RE = re.compile(r'Epoch (\d+)/(\d+) \| Iter (\d+)/(\d+)\] loss=([\d.eE+-]+)')
EPOCH_RE = re.compile(r'Epoch (\d+)/\d+\s+\((\d+)s\)\s+train_loss=([\d.]+)(?:\s+val_loss=([\d.]+))?')


def tail(path, n=400):
    if not path.exists():
        return []
    with open(path, 'r', errors='ignore') as f:
        return f.readlines()[-n:]


EPOCH_LINE = re.compile(
    r'Epoch (\d+)/\d+\s+\((\d+)s\)\s+train_loss=([\d.]+)(?:\s+val_loss=([\d.]+))?')

# Reference runs for cross-run comparison. Validation ALWAYS uses the plain
# 42-cell rectangle MaskCollator regardless of training mode, which is what
# makes val loss comparable ACROSS runs (and, for the same reason, not
# comparable against train loss within a curriculum run).
REFERENCE_RUNS = {
    'envelope': pathlib.Path(
        r'C:\Users\Gary\Desktop\jepa\results\pretraining'
        r'\pretrain_mirage_envelope\combined_stdout.log'),
}
# Oracle and random kept no per-epoch log, only plots; these are the published
# checkpoints from docs/experiments/pretraining/{oracle,random}_100ep.md.
REFERENCE_POINTS = {
    'oracle': {26: 0.1202, 100: 0.1430},
    'random': {25: 0.1197, 50: 0.1423, 75: 0.1469, 100: 0.1419},
}


def _epochs_from_log(path):
    """{epoch: (train_loss, val_loss)} from a run's stdout log."""
    try:
        txt = pathlib.Path(path).read_text(errors='ignore')
    except Exception:
        return {}
    out = {}
    for m in EPOCH_LINE.finditer(txt):
        out[int(m.group(1))] = (
            float(m.group(3)),
            float(m.group(4)) if m.group(4) else None)
    return out


def _pace_from_csv(iters_per_epoch, sample=4000):
    """Seconds/epoch from the timing columns the trainer logs every iteration.

    Returns None when the log is unusable, so the caller can say so rather
    than present a fabricated ETA.
    """
    try:
        import csv as _csv
        with open(LOG, 'r', errors='ignore') as f:
            rows = list(_csv.DictReader(f))[-sample:]
        if not rows:
            return None
        cols = [c for c in rows[0] if c and 'time_ms' in c]
        if not cols:
            return None
        tot = 0.0
        for r in rows:
            tot += sum(float(r[c] or 0) for c in cols)
        per_iter_ms = tot / len(rows)
        if per_iter_ms <= 0:
            return None
        return per_iter_ms * iters_per_epoch / 1000.0
    except Exception:
        return None


def gpu():
    try:
        o = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20).stdout.strip()
        u, t, g = [int(x) for x in o.split(',')]
        return u, t, g
    except Exception:
        return None, None, None


def alive():
    try:
        o = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                           capture_output=True, text=True, timeout=20).stdout
        return o.count('python.exe')
    except Exception:
        return -1


def main():
    flags, notes = [], []
    lines = tail(OUT, 600)

    it = [ITER_RE.search(l) for l in lines]
    it = [m for m in it if m]
    ep_done = [EPOCH_RE.search(l) for l in lines]
    ep_done = [m for m in ep_done if m]

    if not it:
        print('NO ITERATION LINES FOUND — run may not have started')
        return

    last = it[-1]
    epoch, tot_ep, itr, tot_it = (int(last.group(1)), int(last.group(2)),
                                  int(last.group(3)), int(last.group(4)))
    loss = float(last.group(5))

    # ---- loss health ----
    # The CSV is appended across runs, so restrict to rows from THIS run
    # (epoch >= START_EPOCH + 1). Otherwise the aborted run's epoch-1 losses
    # get compared against the resumed run's and look like a regression.
    losses = []
    if LOG.exists():
        with open(LOG) as f:
            losses = [float(r['loss']) for r in csv.DictReader(f)
                      if int(r['epoch']) > START_EPOCH]
    if losses:
        import math
        if any(not math.isfinite(x) for x in losses):
            flags.append('NON-FINITE LOSS in log')
        n = len(losses)
        if n > 4000:
            a = sum(losses[-4000:-2000]) / 2000
            b = sum(losses[-2000:]) / 2000
            if b > a * 2.5:
                flags.append('loss rising: %.5f -> %.5f over last 4000 iters '
                             'of this run' % (a, b))
        notes.append('loss: %d iters this run, latest mean %.5f'
                     % (n, sum(losses[-200:]) / min(200, n)))

    # ---- masking health ----
    mir = [MIRAGE_RE.search(l) for l in lines]
    mir = [m for m in mir if m]
    if mir:
        k = mir[-12:]
        fb = sum(int(m.group(6)) for m in k) / len(k)
        onr = sum(float(m.group(4)) for m in k) / len(k)
        ctx = sum(float(m.group(3)) for m in k) / len(k)
        uni = sum(float(m.group(2)) for m in k) / len(k)
        blk = sum(float(m.group(1)) for m in k) / len(k)
        notes.append('mask: targets %.1f cells, union %.1f, context %.1f, '
                     'on-region %.3f, fallback %.0f/64' % (blk, uni, ctx, onr, fb))
        # train_patch.py calls set_epoch(epoch) with the 0-INDEXED epoch but
        # logs epoch+1, so displayed "Epoch 30" is internally epoch 29 and
        # r_t = (29-25)/5 = 0.80, not 1.00. Getting this wrong makes the
        # ramp's own designed fallback rate look like a guide failure.
        internal_epoch = epoch - 1
        r_t = 0.0 if internal_epoch <= 25 else min(1.0, (internal_epoch - 25) / 5.0)
        expected_fb = 64 * (1.0 - r_t)
        if epoch >= 32 and fb > 16:
            flags.append('fallback %.0f/64 at epoch %d — guide failing too often'
                         % (fb, epoch))
        elif fb > expected_fb + 12:
            flags.append('fallback %.0f/64 exceeds the %.0f/64 the ramp alone '
                         'explains at r_t=%.2f' % (fb, expected_fb, r_t))
        if epoch >= 32 and onr < 0.75:
            flags.append('on-region %.3f at epoch %d — expected >0.9 under full '
                         'guidance' % (onr, epoch))
        if ctx < 60:
            flags.append('context collapsed to %.1f tokens' % ctx)
        notes.append('ramp: displayed epoch %d = internal %d -> r_t=%.2f  '
                     '(%s; ramp alone explains %.0f/64 fallback)'
                     % (epoch, internal_epoch, r_t,
                        'anatomy active' if r_t > 0 else 'random bootstrap',
                        expected_fb))

    # ---- progress / ETA ----
    done_ep = epoch - 1 - START_EPOCH
    frac = done_ep + itr / tot_it
    remain = (TOTAL_EPOCHS - START_EPOCH) - frac
    secs = [int(m.group(2)) for m in ep_done]
    if secs:
        per = sum(secs) / len(secs)
        pace_src = 'measured epochs'
    else:
        # Derive pace from the iteration log rather than a constant: the CSV
        # always exists while the stdout file may not, and a stale constant
        # produces an ETA that is confidently wrong.
        per = _pace_from_csv(tot_it)
        pace_src = 'iteration log' if per else 'fallback constant'
        per = per or SEC_PER_EPOCH
    eta_h = remain * per / 3600.0
    notes.append('progress: epoch %d/%d iter %d/%d  (%.1f%% of the %d-epoch run)'
                 % (epoch, tot_ep, itr, tot_it,
                    100 * frac / (TOTAL_EPOCHS - START_EPOCH),
                    TOTAL_EPOCHS - START_EPOCH))
    notes.append('pace: %.0f s/epoch from %s, ETA %.1f h (%.1f days)'
                 % (per, pace_src, eta_h, eta_h / 24))

    for m in ep_done[-3:]:
        notes.append('  completed epoch %s in %ss  train %s%s'
                     % (m.group(1), m.group(2), m.group(3),
                        '  val %s' % m.group(4) if m.group(4) else ''))

    # ---- validation trend ----
    # The val split is UNGUIDED (random rectangles), so train and val measure
    # DIFFERENT tasks and a rising val loss is not overfitting in the usual
    # sense: an encoder specialising toward anatomy targets is expected to get
    # worse at predicting random background. Tracked because it cannot be
    # distinguished from genuine degradation without downstream AUC.
    vals = [(int(m.group(1)), float(m.group(4)))
            for m in ep_done if m.group(4)]
    if len(vals) >= 3:
        rises = sum(1 for a, b in zip(vals, vals[1:]) if b[1] > a[1])
        notes.append('val trend: %s  (%d/%d consecutive rises)'
                     % (' -> '.join('%.4f' % v for _, v in vals[-5:]),
                        rises, len(vals) - 1))
        if rises == len(vals) - 1 and len(vals) >= 5:
            growth = vals[-1][1] / vals[0][1]
            if growth > 1.5:
                flags.append('val loss up %.0f%% over %d epochs while train '
                             'falls — specialisation or degradation, only '
                             'downstream AUC can tell'
                             % (100 * (growth - 1), len(vals) - 1))

    # ---- checkpoints ----
    cks = sorted(RUN.glob('*.pth.tar'))
    # ---- cross-run comparison ----
    # Val loss is the only loss comparable across runs, because validation
    # always uses the same 42-cell rectangle collator. A curriculum run's TRAIN
    # loss is not comparable to a baseline's: it measures a different task.
    mine = _epochs_from_log(OUT)
    if mine:
        ref = _epochs_from_log(REFERENCE_RUNS['envelope'])
        rows = []
        for ep in sorted(mine):
            t, v = mine[ep]
            rt, rv = ref.get(ep, (None, None))
            delta = ('%+.4f' % (v - rv)) if (v is not None and rv is not None) else '   -'
            rows.append('    ep%-4d ours %s/%s   envelope %s/%s   val %s'
                        % (ep,
                           '%.4f' % t,
                           '%.4f' % v if v is not None else '  -   ',
                           '%.4f' % rt if rt is not None else '  -   ',
                           '%.4f' % rv if rv is not None else '  -   ',
                           delta))
        notes.append('val vs envelope (same uniform val task; lower is better):')
        notes.extend(rows[-6:])
        for nm, pts in REFERENCE_POINTS.items():
            near = [e for e in pts if e in mine]
            if near:
                e = near[-1]
                notes.append('    ep%-4d ours %.4f   %s %.4f'
                             % (e, mine[e][1] or float('nan'), nm, pts[e]))

    notes.append('checkpoints: %d saved%s'
                 % (len(cks), '  latest ' + cks[-1].name if cks else ''))
    if epoch >= 31 and not cks:
        flags.append('epoch %d and still no checkpoint written' % epoch)

    # ---- hardware ----
    u, t, g = gpu()
    if u is not None:
        notes.append('gpu: %d/%d MB total (%.0f%%), util %d%%'
                     % (u, t, 100 * u / t, g))
        # nvidia-smi reports the whole device, which includes the DataLoader
        # workers' CUDA contexts. The figure that matters is what the training
        # process itself reports, which is parsed from the log below.
        own = None
        for l in reversed(lines):
            mm = re.search(r'gpu=(\d+)MB', l)
            if mm:
                own = int(mm.group(1))
                break
        if own is not None:
            notes.append('gpu: %d MB used by the training process itself' % own)
            if own > 0.92 * t:
                flags.append('training process at %d/%d MB — close to OOM'
                             % (own, t))
    np_ = alive()
    if np_ == 0:
        flags.append('NO PYTHON PROCESS — the run has died')

    print('=' * 66)
    print('anatomy pretraining monitor   %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 66)
    for s in notes:
        print('  ' + s)
    print()
    if flags:
        print('  !! %d FLAG(S)' % len(flags))
        for f in flags:
            print('     - ' + f)
    else:
        print('  no anomalies')

    STATE.write_text(json.dumps(
        {'time': time.time(), 'epoch': epoch, 'iter': itr, 'loss': loss,
         'eta_hours': eta_h, 'flags': flags, 'notes': notes}, indent=2))


if __name__ == '__main__':
    main()
