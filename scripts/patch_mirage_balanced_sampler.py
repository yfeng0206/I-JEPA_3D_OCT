"""Patch MIRAGE's run_seg_tuning.py with an optional source-balanced sampler.

Motivation, measured rather than assumed:

The merged training set is 1,248 images -- GOALS 55, Duke_DME 88, AROI 1,105.
GOALS is the ONLY source that supervises Choroid (verified by
scripts/seg_v2_preflight.py: Choroid pixels are 2,407,885 in GOALS and 0 in
both Duke and AROI, because the sub-BM region of those two datasets includes
sclera and is therefore mapped to ignore).

With plain RandomSampler and batch size 4, P(no GOALS in a batch) is
(1193/1248)^4 = 0.83, so 83% of optimiser steps would carry no Choroid
gradient at all.  Worse, a fixed epoch budget gives GOALS only `epochs`
presentations, whereas the GOALS-only baseline gave it 200 -- so a shortened
merged run would starve Choroid relative to the very baseline we must beat.

A WeightedRandomSampler fixes both at once: it raises the per-batch chance of
Choroid supervision AND restores GOALS' total exposure, without changing the
epoch length (num_samples = len(dataset)).

Idempotent: re-running this script is a no-op if the patch is already present.
"""
from __future__ import annotations

import ast
import pathlib
import sys

TARGET = pathlib.Path(r'D:\jepa_phase0\mirage-goals\MIRAGE\run_seg_tuning.py')

ARG_ANCHOR = """    parser.add_argument(
        '--ignore_index',"""

ARG_PATCH = """    parser.add_argument(
        '--balance_sources', action='store_true', default=False,
        help='Sample the training set with per-source weights instead of '
             'uniformly. Filenames must be named SOURCE__original.ext. '
             'Required for merged multi-dataset training, where the rarest '
             'source may be the only one supervising a class.'
    )
    parser.add_argument(
        '--source_share', action='append', default=None, metavar='NAME=FRAC',
        help='Explicit sampling share for one source, e.g. GOALS=0.25. '
             'Repeatable. Sources left unspecified split the remaining mass '
             'in proportion to sqrt(count), which keeps a small source from '
             'being repeated an extreme number of times per epoch.'
    )
    parser.add_argument(
        '--ignore_index',"""

SAMPLER_ANCHOR = "    sampler_train = torch.utils.data.RandomSampler(dataset_train)"

SAMPLER_PATCH = '''    if args.balance_sources:
        sampler_train = build_source_balanced_sampler(dataset_train, args)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)'''

HELPER_ANCHOR = "def main(args):"

HELPER = '''def build_source_balanced_sampler(dataset, args):
    """Weighted sampler over sources inferred from ``SOURCE__original.ext``.

    Epoch length is preserved (``num_samples == len(dataset)``) so the LR
    schedule and ``num_training_steps_per_epoch`` are unaffected.
    """
    import collections
    import math

    task = args.in_domains[0]
    paths = [p for p, _ in dataset.samples[task]]
    sources = [os.path.basename(p).split('__')[0] if '__' in os.path.basename(p)
               else 'UNKNOWN' for p in paths]
    counts = collections.Counter(sources)

    explicit = {}
    for item in (args.source_share or []):
        name, _, frac = item.partition('=')
        if not _:
            raise ValueError('--source_share expects NAME=FRACTION, got %r' % item)
        explicit[name] = float(frac)
    unknown = set(explicit) - set(counts)
    if unknown:
        raise ValueError('--source_share names not present in the data: %s '
                         '(available: %s)' % (sorted(unknown), sorted(counts)))
    assigned = sum(explicit.values())
    if assigned > 1.0 + 1e-9:
        raise ValueError('--source_share fractions sum to %.4f > 1' % assigned)

    remaining_sources = [s for s in counts if s not in explicit]
    shares = dict(explicit)
    if remaining_sources:
        norm = sum(math.sqrt(counts[s]) for s in remaining_sources)
        for s in remaining_sources:
            shares[s] = (1.0 - assigned) * math.sqrt(counts[s]) / norm
    elif abs(assigned - 1.0) > 1e-6:
        raise ValueError('every source was given an explicit share but they '
                         'sum to %.4f, not 1' % assigned)

    # Per-SAMPLE weight = per-source share / number of samples in that source,
    # so every image inside a source stays equally likely.
    weights = [shares[s] / counts[s] for s in sources]

    n = len(paths)
    print('Source-balanced sampling enabled:')
    print('  %-12s %7s %8s %10s %12s' %
          ('source', 'images', 'share', 'draws/ep', 'repeats/img'))
    for s in sorted(counts):
        draws = shares[s] * n
        print('  %-12s %7d %8.4f %10.1f %12.2f'
              % (s, counts[s], shares[s], draws, draws / counts[s]))
    bs = args.batch_size
    for s in sorted(counts):
        p_absent = (1.0 - shares[s]) ** bs
        print('  P(batch of %d contains no %s) = %.3f' % (bs, s, p_absent))

    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=n, replacement=True)


def main(args):'''


def apply(text: str, anchor: str, patch: str, label: str, marker: str) -> str:
    if marker in text:
        print('  %-10s already present, skipping' % label)
        return text
    if anchor not in text:
        raise SystemExit('FAILED: anchor for %s not found' % label)
    if text.count(anchor) != 1:
        raise SystemExit('FAILED: anchor for %s is not unique (%d)'
                         % (label, text.count(anchor)))
    print('  %-10s patched' % label)
    return text.replace(anchor, patch, 1)


def main() -> int:
    text = TARGET.read_text(encoding='utf-8')

    text = apply(text, ARG_ANCHOR, ARG_PATCH, 'args', "'--balance_sources'")
    text = apply(text, HELPER_ANCHOR, HELPER, 'helper',
                 'def build_source_balanced_sampler')
    text = apply(text, SAMPLER_ANCHOR, SAMPLER_PATCH, 'sampler',
                 'if args.balance_sources:')
    TARGET.write_text(text, encoding='utf-8')

    ast.parse(TARGET.read_text(encoding='utf-8'))
    print('syntax OK')

    src = TARGET.read_text(encoding='utf-8')
    for needle in ('--balance_sources', '--source_share',
                   'build_source_balanced_sampler', 'WeightedRandomSampler'):
        print('  %-34s %s' % (needle, 'present' if needle in src else 'MISSING'))
    if 'import os' not in src:
        print('  WARNING: os is not imported in the target module')
    return 0


if __name__ == '__main__':
    sys.exit(main())
