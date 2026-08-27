#!/usr/bin/env python3
"""Compare the three matched pretraining arms and render the figure.

  armA random_default   rectangles, pred_mask_scale 0.15-0.2  (shipped I-JEPA)
  armB random_matched   rectangles, scale lowered to match anatomy's area
  armC anatomy          connected anatomy-shaped targets

armB is the arm that isolates SHAPE.  Without it, anatomy differs from random
in both WHERE it masks and HOW MUCH, so a difference could not be attributed
to targeting.

Loss is NOT comparable across arms as a quality measure -- each arm predicts a
different set of tokens, and an easier target set gives a lower loss.  It is
reported to show that each arm trained stably.  The claims that can be made
from these runs are about the MASK (coverage, targeting, context) and about
representation health (diversity, no collapse).
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / 'results/masking/arms'
RUNS = pathlib.Path(r'D:\jepa_phase0\runs')

ARMS = [
    ('random_default', RUNS / 'patch_mirage_anatomy' / 'jepa_patch_mirage-log.csv',
     RUNS / 'armA_random_default_stdout.log', '#d62728'),
    ('random_matched', RUNS / 'arm_random_matched' / 'jepa_patch_mirage-log.csv',
     RUNS / 'armB_random_matched_stdout.log', '#ff7f0e'),
    ('anatomy', RUNS / 'arm_anatomy' / 'jepa_patch_mirage-log.csv',
     RUNS / 'armC_anatomy_stdout.log', '#2ca02c'),
]

MIRAGE_RE = re.compile(
    r'patches/block=([\d.]+)\s+unique_targets=([\d.]+)\s+context=([\d.]+)\s+'
    r'on_region=([\d.]+)\s+background=([\d.]+)\s+fallbacks=(\d+)')
DIAG_RE = re.compile(
    r'cos_sim=([\d.]+).*?l2_dist=([\d.]+)\s+rep_diversity=([\d.]+)')
EPOCH_RE = re.compile(r'train_loss=([\d.]+)\s+val_loss=([\d.]+)')


def read_log(csv_path):
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r['loss']) for r in rows])


def read_stdout(p):
    if not p.exists():
        return {}
    txt = p.read_text(errors='ignore')
    m = MIRAGE_RE.findall(txt)
    d = DIAG_RE.search(txt)
    e = EPOCH_RE.search(txt)
    out = {}
    if m:
        arr = np.array(m[len(m) // 4:], float)      # skip warmup
        out.update(patches_per_block=arr[:, 0].mean(),
                   unique_targets=arr[:, 1].mean(),
                   context=arr[:, 2].mean(),
                   on_region=arr[:, 3].mean(),
                   fallbacks=arr[:, 5].mean())
    if d:
        out.update(cos_sim=float(d.group(1)), l2=float(d.group(2)),
                   rep_diversity=float(d.group(3)))
    if e:
        out.update(train_loss=float(e.group(1)), val_loss=float(e.group(2)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, curves = {}, {}
    for tag, csvp, outp, _ in ARMS:
        L = read_log(csvp)
        s = read_stdout(outp)
        if L is None:
            print('  %-16s MISSING (%s)' % (tag, csvp))
            continue
        curves[tag] = L
        s['iters'] = int(len(L))
        res[tag] = s

    hdr = ('%-16s %7s %9s %9s %9s %10s %9s %8s' %
           ('arm', 'iters', 'hidden', 'context', 'on-region', 'fallback',
            'rep-div', 'val'))
    print(hdr); print('-' * len(hdr))
    for tag in [a[0] for a in ARMS]:
        if tag not in res:
            continue
        r = res[tag]
        print('%-16s %7d %9.1f %9.1f %9.3f %9.1f%% %9.4f %8s'
              % (tag, r['iters'], r.get('unique_targets', float('nan')),
                 r.get('context', float('nan')), r.get('on_region', float('nan')),
                 100 * r.get('fallbacks', 0) / 64.0,
                 r.get('rep_diversity', float('nan')),
                 ('%.4f' % r['val_loss']) if 'val_loss' in r else '-'))

    if 'anatomy' in res and 'random_default' in res:
        a, b = res['anatomy'], res['random_default']
        print()
        print('anatomy vs shipped random baseline')
        print('  targets on region   %.3f vs %.3f   -> %.2fx'
              % (a['on_region'], b['on_region'], a['on_region'] / b['on_region']))
        print('  context tokens      %.1f vs %.1f   -> %+.1f%%'
              % (a['context'], b['context'],
                 100 * (a['context'] / b['context'] - 1)))
        print('  cells hidden        %.1f vs %.1f   -> %.2fx'
              % (a['unique_targets'], b['unique_targets'],
                 a['unique_targets'] / b['unique_targets']))
        print('  rep diversity       %.4f vs %.4f   (1.0 = collapsed)'
              % (a.get('rep_diversity', float('nan')),
                 b.get('rep_diversity', float('nan'))))

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for tag, _, _, col in ARMS:
        if tag not in curves:
            continue
        L = curves[tag]
        w = 100
        sm = np.convolve(L, np.ones(w) / w, 'valid')
        ax[0].plot(sm, label=tag, color=col, lw=1.8)
    ax[0].set_yscale('log'); ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('loss (smoothed)')
    ax[0].set_title('training loss\n(NOT comparable as quality: different targets)',
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    keys = ['unique_targets', 'context']
    labels = ['cells hidden', 'context tokens']
    x = np.arange(len(keys))
    w = 0.26
    for i, (tag, _, _, col) in enumerate(ARMS):
        if tag not in res:
            continue
        v = [res[tag].get(k, 0) for k in keys]
        ax[1].bar(x + (i - 1) * w, v, w, label=tag, color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_title('mask budget', fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='y')

    tags = [t for t, _, _, _ in ARMS if t in res]
    vals = [res[t].get('on_region', 0) for t in tags]
    cols = [c for t, _, _, c in ARMS if t in res]
    ax[2].bar(tags, vals, color=cols)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel('fraction of targeted patches on retina')
    ax[2].set_title('targeting precision\n(on_region, threshold 0.25)', fontsize=10)
    ax[2].grid(alpha=.3, axis='y')
    for i, v in enumerate(vals):
        ax[2].text(i, v + .02, '%.3f' % v, ha='center', fontsize=9)

    fig.suptitle('Matched pretraining arms, one full FairVision epoch each '
                 '(600,000 slices, 9,375 iterations)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / 'arms.png', dpi=115)
    (OUT / 'arms.json').write_text(json.dumps(res, indent=2))
    print()
    print('wrote', OUT / 'arms.png')


if __name__ == '__main__':
    main()
