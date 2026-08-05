"""Pre-flight checks for the MergedV2 segmentation training set.

Answers, with measurements rather than assumptions:

L-MERGE   per-source class histograms; is Choroid supervised only by GOALS,
          and is the sub-BM region of Duke/AROI genuinely *ignored* (so the
          absent choroid gradient is neutral) rather than labelled Elsewhere
          (which would actively suppress choroid)?
L-TRUTH   InnerRetina fraction per source, and column ordering sanity.
L-DIM     stored aspect ratios per source.
L-CODE    the loss reduces to MIRAGE's when nothing is ignored, and a real
          batch produces a finite loss and finite gradients.

Run with the repo venv:
    D:\\jepa_phase0\\.venv\\Scripts\\python.exe scripts/seg_v2_preflight.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IGNORE_VALUE = 1
VALUE_TO_NAME = {0: 'Elsewhere', 128: 'InnerRetina', 255: 'Choroid', 1: 'ignore'}


def source_of(name: str) -> str:
    """Filenames are written by the builder as ``SOURCE__original.png``."""
    return name.split('__')[0] if '__' in name else 'UNKNOWN'


def scan_split(root: pathlib.Path, split: str, limit_per_source: int | None):
    semseg = root / split / 'semseg'
    bscan = root / split / 'bscan'
    per_source_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    per_source_images = collections.Counter()
    per_source_aspect: dict[str, list] = collections.defaultdict(list)
    # Of the pixels strictly BELOW the lowest InnerRetina pixel in each column,
    # what are they labelled?  This is the region where choroid lives.
    sub_retina: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    seen = collections.Counter()

    for mask_path in sorted(semseg.glob('*.png')):
        src = source_of(mask_path.name)
        if limit_per_source is not None and seen[src] >= limit_per_source:
            continue
        seen[src] += 1
        m = np.array(Image.open(mask_path))
        per_source_images[src] += 1
        per_source_counts[src].update(
            dict(zip(*[a.tolist() for a in np.unique(m, return_counts=True)])))

        img_path = bscan / mask_path.name
        if img_path.exists():
            with Image.open(img_path) as im:
                per_source_aspect[src].append(im.size)  # (W, H)

        inner = m == 128
        cols = np.where(inner.any(axis=0))[0]
        if cols.size:
            last = inner.shape[0] - 1 - np.argmax(inner[::-1, :], axis=0)
            for c in cols:
                below = m[last[c] + 1:, c]
                if below.size:
                    vals, cnts = np.unique(below, return_counts=True)
                    sub_retina[src].update(dict(zip(vals.tolist(), cnts.tolist())))
    return per_source_counts, per_source_images, per_source_aspect, sub_retina


def true_source_totals(root: pathlib.Path, split: str) -> collections.Counter:
    """True on-disk image count per source.

    scan_split() stops at --limit-per-source, so the image counts it returns
    understate the real imbalance (60/60/55 looks balanced when the true split
    is 1105/88/55).  Evidence-volume checks MUST use these numbers instead.
    Counts filenames only -- no image is decoded, so this is cheap.
    """
    totals = collections.Counter()
    for mask_path in (root / split / 'semseg').glob('*.png'):
        totals[source_of(mask_path.name)] += 1
    return totals


def pct(counter: collections.Counter) -> dict:
    total = sum(counter.values())
    if not total:
        return {}
    return {VALUE_TO_NAME.get(k, str(k)): counter[k] / total
            for k in sorted(counter)}


def fmt(d: dict) -> str:
    return '  '.join('%s=%.4f' % (k, v) for k, v in d.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\jepa_phase0\mirage-datasets\MergedV2')
    ap.add_argument('--limit-per-source', type=int, default=60,
                    help='cap images scanned per source for speed')
    ap.add_argument('--skip-torch', action='store_true')
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    ok = True

    print('=' * 78)
    print('MergedV2 pre-flight :', root)
    print('=' * 78)

    info = json.loads((root / 'INFO.json').read_text())
    ignore_idx = None
    mapping = {}
    for k, v in info.items():
        mapping[v['value']] = int(k)
        if 'background' in v['label'].lower() or 'bg' in v['label'].lower():
            ignore_idx = int(k)
    print('INFO.json mapping value->index :', mapping)
    print('auto-derived ignore_index      :', ignore_idx,
          '(MIRAGE matches on the substring "background")')
    if ignore_idx is None:
        print('  FAIL: MIRAGE would train with ignore_index=None')
        ok = False
    print('num_classes                    :', len(mapping))

    for split in ('train', 'val', 'test'):
        counts, images, aspect, sub = scan_split(
            root, split, args.limit_per_source)
        print()
        print('-' * 78)
        print('SPLIT %s   (scanning <=%d imgs/source)' % (split.upper(), args.limit_per_source))
        print('-' * 78)
        for src in ('GOALS', 'Duke_DME', 'AROI'):
            if not images[src]:
                continue
            print('%-9s n=%-4d %s' % (src, images[src], fmt(pct(counts[src]))))
            sizes = set(aspect[src])
            ar = sorted({round(h / w, 4) for (w, h) in sizes})
            print('%-9s stored sizes=%s  aspect(H/W)=%s'
                  % ('', sorted(sizes)[:3], ar[:3]))
            print('%-9s below-retina label mix: %s'
                  % ('', fmt(pct(sub[src]))))

        if split == 'train':
            chor = {s: counts[s].get(255, 0) for s in ('GOALS', 'Duke_DME', 'AROI')}
            print()
            print('CHOROID supervision by source (pixels):', chor)
            non_goals_chor = chor['Duke_DME'] + chor['AROI']
            if non_goals_chor:
                print('  note: Choroid also supervised outside GOALS')
            # The key question: in Duke/AROI, is the sub-retina region IGNORED
            # (neutral) or Elsewhere (actively suppresses choroid)?
            for s in ('Duke_DME', 'AROI'):
                mix = pct(sub[s])
                ig = mix.get('ignore', 0.0)
                els = mix.get('Elsewhere', 0.0)
                verdict = ('no active suppression (ignore, not Elsewhere)'
                           if ig >= 0.5 else
                           'ADVERSE  - Elsewhere actively suppresses choroid')
                print('  %-9s below-retina ignore=%.3f elsewhere=%.3f  -> %s'
                      % (s, ig, els, verdict))
                if ig < 0.5:
                    ok = False

            # L-MERGE evidence volume.
            #
            # The check above only asks "is anything pushing AGAINST this
            # class?".  V2 passed it -- Duke/AROI ignore the sub-retina region,
            # so gradient there is exactly 0.0 -- and still lost FairVision
            # choroid coverage (0.995 -> 0.25).  Absence of suppression is NOT
            # sufficient; the class also needs enough POSITIVE evidence.
            # Measured cause: only 55 of 1248 train images (4.4%) carried any
            # choroid label at all.
            totals = true_source_totals(root, split)
            n_all = sum(totals.values())
            print()
            print('EVIDENCE VOLUME  (true on-disk image counts: %s)'
                  % dict(totals))
            for val, name in ((0, 'Elsewhere'), (128, 'InnerRetina'),
                              (255, 'Choroid')):
                supervising = [s for s in totals if counts[s].get(val, 0) > 0]
                covered = sum(totals[s] for s in supervising)
                frac = covered / n_all if n_all else 0.0
                if frac < 0.10:
                    tag, ok = 'FAIL  - starved, expect recall collapse', False
                elif frac < 0.33:
                    tag = 'WARN  - thin, recall may degrade off-domain'
                else:
                    tag = 'ok'
                print('  %-12s supervised by %-28s %5.1f%% of imgs  -> %s'
                      % (name, ','.join(sorted(supervising)) or '(none)',
                         100 * frac, tag))
            print('  note: a source-balanced sampler changes EFFECTIVE '
                  'exposure; if GOALS is capped at 0.25 share, a GOALS-only '
                  'class is seen at most 25% of steps regardless of the above.')

    if args.skip_torch:
        print('\n(skipping torch checks)')
        return 0 if ok else 1

    print()
    print('-' * 78)
    print('L-CODE  loss behaviour on real merged batches')
    print('-' * 78)
    import torch
    from src.losses.ignore_cegdice import IgnoreAwareCEGDice

    num_classes = len(mapping)
    crit = IgnoreAwareCEGDice(num_classes=num_classes, ignore_index=ignore_idx)

    value_to_index = {v: k for v, k in mapping.items()}
    rng = np.random.default_rng(0)
    names = sorted(p.name for p in (root / 'train' / 'semseg').glob('*.png'))
    by_src = collections.defaultdict(list)
    for n in names:
        by_src[source_of(n)].append(n)

    def load_batch(picks):
        tgt = []
        for n in picks:
            m = np.array(Image.open(root / 'train' / 'semseg' / n))
            idx = np.zeros_like(m, dtype=np.int64)
            for val, i in value_to_index.items():
                idx[m == val] = i
            im = Image.fromarray(idx.astype(np.uint8)).resize(
                (256, 256), Image.NEAREST)
            tgt.append(np.array(im).astype(np.int64))
        return torch.from_numpy(np.stack(tgt))

    scenarios = {
        'all-AROI (no GOALS, no choroid)': [
            by_src['AROI'][i] for i in rng.integers(0, len(by_src['AROI']), 4)],
        'all-GOALS (choroid present)': [
            by_src['GOALS'][i] for i in rng.integers(0, len(by_src['GOALS']), 4)],
        'mixed': [by_src['GOALS'][0], by_src['Duke_DME'][0],
                  by_src['AROI'][0], by_src['AROI'][1]],
    }
    for label, picks in scenarios.items():
        target = load_batch(picks)
        logits = torch.randn(target.shape[0], num_classes, 256, 256,
                             requires_grad=True)
        loss = crit(logits, target)
        loss.backward()
        g = logits.grad
        present = sorted(set(target.unique().tolist()))
        # does the ignore class receive gradient anywhere it should not?
        ig_frac = float((target == ignore_idx).float().mean())
        print('%-32s loss=%.4f finite=%s grad_finite=%s ignore_frac=%.3f classes=%s'
              % (label, float(loss), bool(torch.isfinite(loss)),
                 bool(torch.isfinite(g).all()), ig_frac, present))
        if not (torch.isfinite(loss) and torch.isfinite(g).all()):
            ok = False
        # gradient must be exactly zero on ignored pixels
        if ig_frac > 0:
            gz = g.permute(0, 2, 3, 1)[target == ignore_idx]
            mx = float(gz.abs().max())
            print('%-32s max|grad| on ignored pixels = %.3e %s'
                  % ('', mx, 'OK' if mx == 0.0 else 'FAIL - leakage'))
            if mx != 0.0:
                ok = False

    print()
    print('=' * 78)
    print('PRE-FLIGHT', 'PASS' if ok else 'FAIL')
    print('=' * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
