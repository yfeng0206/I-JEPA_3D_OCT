"""Synthetic cr_seed_v1 result fixtures for autopilot/make_cr_results.py (never real results).

Builds sealed probe directories whose per-head-seed test AUCs equal chosen values exactly
(rank construction around a shared latent score, so paired intervals are realistic), unseals
them and runs the real ``cr_stats.py final`` on them. The final JSON therefore has the
producer's exact schema. An MD JSON is written in the schema of
``cr_md_regression.py --split test`` (values derived from the AUCs; illustrative only).

Scenario patterns (deltas versus RANDOM of the same seed for seeds A, B, C), with RANDOM's
realizations 0.8650 (original), 0.8668, 0.8641, 0.8655 (range 0.0027):
  A  consistent direction        all positive, mean above the range
  B  within run-to-run variation  positive mean, but a negative seed or mean <= range
  C  reversed                     mean <= 0
Matched control (RANDOM-CB at seed A): M+, M0, M-, M_partial, M_na (no run), M_desc (late run).
"""
import contextlib
import io
import json
import os
import sys
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'autopilot')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src import eval_downstream as evaluation  # noqa: E402
import cr_stats  # noqa: E402

PRIMARY = 'patchmean_slicemean'
VARIANTS = (PRIMARY, 'patchmean_slicemax', 'patchmean_slicemeanmax', 'patchmax_slicemean')
HEADS = (42, 43, 44, 45, 46)
SEED_INDEX = {1234: 0, 5678: 1, 9012: 2}
RANDOM = {None: 0.8650, 1234: 0.8668, 5678: 0.8641, 9012: 0.8655}
PATTERNS = {
    'A': {'centroid': (0.0074, 0.0077, 0.0069), 'envelope': (0.0053, 0.0064, 0.0049)},
    'B': {'centroid': (0.0034, 0.0011, 0.0020), 'envelope': (-0.0013, 0.0079, 0.0040)},
    'C': {'centroid': (-0.0018, 0.0014, -0.0006), 'envelope': (-0.0046, -0.0031, -0.0040)},
    # V6 counterexamples: B with the LARGER mean (one non-positive continuation), and A with a
    # small seed-A gap (so an M0 control can have a negative budget shift).
    'Bbig': {'centroid': (-0.0005, 0.0203, 0.0100), 'envelope': (-0.0005, 0.0203, 0.0100)},
    'Asmall': {'centroid': (0.0010, 0.0140, 0.0075), 'envelope': (0.0012, 0.0110, 0.0060)},
}
ORIGINAL_DELTA = {'centroid': 0.0095, 'envelope': 0.0112}
POOL_SHIFT = {PRIMARY: 0.0, 'patchmean_slicemax': -0.0021, 'patchmean_slicemeanmax': 0.0004,
              'patchmax_slicemean': -0.0048}
# Envelope loses part of its gain with patch maxima, so the pooling label can change.
POOL_ARM = {('patchmean_slicemax', 'envelope'): -0.0045}
LATE = '2026-10-22T09:00:00-07:00'
HEAD_SD = 0.0008


def exact_probs(labels, target, latent, rng):
    """float32 probabilities whose ROC AUC is round(target*P*N)/(P*N) exactly."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_neg = neg.size
    neg_order = np.argsort(latent[neg], kind='mergesort')
    sorted_neg = latent[neg][neg_order]
    below = np.searchsorted(sorted_neg, latent[pos])
    need = int(round(target * pos.size * n_neg)) - int(below.sum())
    while need:
        step = 1 if need > 0 else -1
        room = np.flatnonzero((n_neg - below) > 0 if step > 0 else below > 0)
        chosen = rng.choice(room, size=min(abs(need), room.size), replace=False)
        below[chosen] += step
        need -= step * chosen.size
    scores = np.empty(labels.size, dtype=np.float64)
    ranks = np.empty(n_neg, dtype=np.float64)
    ranks[neg_order] = 2.0 * np.arange(n_neg)
    scores[neg] = ranks
    scores[pos] = 2.0 * below - 1.0
    return ((scores + 1.5) / (2.0 * n_neg + 2.0)).astype(np.float32)


def _manifest(split, n):
    prefix = split[0].lower()
    return {'split': split, 'dataset_root': 'synthetic', 'num_slices': 100,
            'subject_ids': ['%s%05d' % (prefix, i) for i in range(n)],
            'ordered_files': [{'name': '%s%05d.npz' % (prefix, i)} for i in range(n)],
            'dataset_identity_kind': 'synthetic'}


class Cohort(object):
    def __init__(self, n_test, seed=20261022):
        rng = np.random.default_rng(seed)
        self.labels = rng.permutation(np.arange(n_test) < int(round(0.49 * n_test))).astype(int)
        self.common = 1.61 * self.labels + rng.normal(0, 1, n_test)
        self.manifest = _manifest('Test', n_test)


def make_probe(root, cohort, name, arm, seed, role, aucs, stop_time=None, head_seeds=HEADS):
    out = os.path.join(str(root), name)
    os.makedirs(out)
    rng = np.random.default_rng(zlib.crc32(name.encode('utf-8')))
    ckpt = '%064x' % (zlib.crc32(name.encode('utf-8')) * 7919)
    identity = {'arm': arm, 'train_seed': seed, 'role': role,
                'run_uuid': 'synthetic-' + name, 'checkpoint_sha256': ckpt[-64:], 'epoch': 50}
    now = cr_stats._now().isoformat(timespec='seconds')
    if role == 'new':
        stop = os.path.join(str(root), 'train_' + name, 'stop_epoch_050.json')
        os.makedirs(os.path.dirname(stop))
        with open(stop, 'w') as stream:
            json.dump({'schema': 'jepa_stop_epoch_v1', 'epoch': 50,
                       'timestamp_utc': stop_time or now,
                       'checkpoints': [{'role': 'periodic', 'epoch': 50,
                                        'sha256': identity['checkpoint_sha256']}]}, stream)
        identity.update({'run_provenance': os.path.realpath(stop),
                         'run_provenance_sha256': evaluation.file_sha256(stop)})
    run_latent = cohort.common + 0.22 * rng.normal(0, 1, cohort.labels.size)
    summaries, entries = {}, []
    for variant, auc in aucs.items():
        offsets = rng.normal(0, 1, len(head_seeds))
        offsets = HEAD_SD * (offsets - offsets.mean()) / (offsets.std(ddof=1) or 1.0)
        variant_latent = run_latent + 0.1 * rng.normal(0, 1, cohort.labels.size)
        per_seed = []
        for head_seed, offset in zip(head_seeds, offsets):
            seed_dir = os.path.join(out, variant, 'seed%d' % head_seed)
            os.makedirs(seed_dir)
            latent = variant_latent + 0.05 * rng.normal(0, 1, cohort.labels.size)
            probs = exact_probs(cohort.labels, auc + offset, latent, rng)
            path = os.path.join(seed_dir, 'test_predictions_sealed_seed%d.npz' % head_seed)
            evaluation.save_predictions(path, cohort.labels.astype(np.float32), probs,
                                        dict(cohort.manifest, sealed=True))
            sidecar = os.path.splitext(path)[0] + '.manifest.json'
            rec = {'variant': variant, 'head_seed': head_seed, 'best_epoch': 12,
                   'best_val_auc': float(auc + 0.0040 + offset + 0.0003 * rng.normal()),
                   'test_predictions': os.path.relpath(path, out),
                   'test_predictions_sha256': evaluation.file_sha256(path),
                   'test_auc': None, 'test_sealed': True}
            per_seed.append(rec)
            entries.append({'variant': variant, 'head_seed': head_seed,
                            'path': rec['test_predictions'], 'sha256': rec['test_predictions_sha256'],
                            'sidecar_path': os.path.relpath(sidecar, out),
                            'sidecar_sha256': evaluation.file_sha256(sidecar),
                            'head_checkpoint_sha256': 'synthetic'})
        vals = [r['best_val_auc'] for r in per_seed]
        summaries[variant] = {'per_seed': per_seed, 'n_head_seeds': len(per_seed),
                              'val_auc_mean': float(np.mean(vals)),
                              'val_auc_sd': float(np.std(vals, ddof=1)),
                              'test_auc_mean': None, 'test_auc_sd': None}
    evaluation.write_sealed_manifest(out, entries, cohort.labels, cohort.manifest, identity)
    config = {'mode': 'patch', 'data': {'num_slices': 100, 'use_amp': False},
              'model': {'encoder_checkpoint': 'synthetic-' + name, 'probe_type': 'mean_pool'},
              'training': {'epochs': 50, 'lr': 4e-4}, 'probe': {'head_seeds': list(head_seeds)}}
    results = {'schema': 'cr_probe_v1', 'status': 'complete', 'sealed': True,
               'head_seeds': list(head_seeds), 'identity': identity, 'variants': summaries,
               'best_val_auc': summaries[PRIMARY]['val_auc_mean'], 'test_auc': None,
               'config': config, 'started': now, 'finished': now}
    with open(os.path.join(out, 'results.json'), 'w') as stream:
        json.dump(results, stream)
    return out


def primary_aucs(centroid, envelope, seeds):
    values = {('random', None): RANDOM[None],
              ('centroid', None): RANDOM[None] + ORIGINAL_DELTA['centroid'],
              ('envelope', None): RANDOM[None] + ORIGINAL_DELTA['envelope']}
    for seed in seeds:
        i = SEED_INDEX[seed]
        values[('random', seed)] = RANDOM[seed]
        values[('centroid', seed)] = RANDOM[seed] + PATTERNS[centroid]['centroid'][i]
        values[('envelope', seed)] = RANDOM[seed] + PATTERNS[envelope]['envelope'][i]
    return values


def cb_auc(matched, values, spread, seed=1234, placement=None):
    c1, r1 = values[('centroid', seed)], values[('random', seed)]
    if placement is not None:  # explicit P = AUC(C1) - AUC(CB)
        return c1 - placement
    gap = c1 - r1
    if matched in ('M+', 'M_desc'):
        return c1 - max(0.75 * gap, spread + 0.0015)
    if matched == 'M0':
        return c1 - 0.3 * spread
    if matched == 'M-':
        return c1 + spread + 0.0015
    if matched == 'M_partial':
        if gap / 2 <= spread + 0.0003:
            raise ValueError('M_partial needs a same-seed gap above twice the RANDOM range')
        return c1 - (spread + gap / 2) / 2
    raise ValueError(matched)


def md_payload(final, labels):
    """MD JSON in the cr_md_regression --split test schema; values derived from the AUCs."""
    block = final['analysis'][PRIMARY]
    runs, excluded = {}, {}
    n_glaucoma = int(labels.sum())
    base = {'mae': 4.671, 'rmse': 5.884, 'spearman': None, 'n': n_glaucoma}
    for name, entry in final['runs'].items():
        ident = entry['identity']
        if ident['role'] == 'anchor' and ident['arm'] == 'random':
            excluded[name] = 'cache without verified per-case identity; no MD regression'
            continue
        auc = block['per_run_test_auc'][name]
        mae = 3.55 - 40.0 * (auc - 0.865)
        ridge = {'mae': mae, 'rmse': 1.31 * mae, 'spearman': 0.52 + 8.0 * (auc - 0.865),
                 'n': n_glaucoma}
        subset = {'subset': 'glaucoma', 'n_train': 2940, 'alpha': 100.0,
                  'alpha_rule': 'min validation RMSE; ties -> larger alpha',
                  'test': {'ridge': ridge, 'baseline_train_mean': dict(base),
                           'diagnostic_class_mean': dict(base, mae=4.671)}}
        runs[name] = {'identity': ident, 'eligibility': entry['eligibility'],
                      'subsets': {'glaucoma': subset, 'all': dict(subset, subset='all')}}
    contrasts = []
    by_slot = {(e['identity']['arm'], e['identity']['train_seed']): n for n, e in runs.items()
               if e['identity']['role'] == 'new' and e['eligibility']['eligible']}
    for seed in sorted(set(s for _, s in by_slot)):
        r = by_slot.get(('random', seed))
        for arm in ('centroid', 'envelope', 'random_cb'):
            a = by_slot.get((arm, seed))
            if r is None or a is None:
                continue
            for subset in ('glaucoma', 'all'):
                metrics = {}
                for key, half in (('mae', 0.12), ('rmse', 0.16), ('spearman', 0.03)):
                    pa = runs[a]['subsets']['glaucoma']['test']['ridge'][key]
                    pb = runs[r]['subsets']['glaucoma']['test']['ridge'][key]
                    metrics[key] = {'a': pa, 'b': pb, 'delta': pa - pb,
                                    'ci95': [pa - pb - half, pa - pb + half]}
                contrasts.append({'kind': '%s-random' % arm, 'train_seed': seed, 'subset': subset,
                                  'primary': subset == 'glaucoma' and arm != 'random_cb',
                                  'a': a, 'b': r, 'metrics': metrics})
    return {'generated': final['generated'], 'split': 'Test', 'before_freeze': True,
            'metadata': 'synthetic', 'n_boot': final['n_boot'], 'boot_seed': 20261022,
            'features': 'patch-token mean per B-scan, mean over B-scans (primary pooling)',
            'primary': 'glaucoma subset; contrasts within seed index vs RANDOM',
            'note': 'synthetic fixture', 'runs': runs, 'excluded_runs': excluded,
            'contrasts': contrasts}


def cb_audit_payload():
    """A RANDOM-CB run audit in the cr_cb_budget_audit_v1 schema (synthetic values)."""
    shadow = {'unique_targets_U': 102.6, 'loss_slots_L': 160.3, 'duplicate_slots_D': 57.7,
              'context_tokens': 80.8}
    return {'schema': 'cr_cb_budget_audit_v1', 'source': 'synthetic fixture', 'images': 64000,
            'mismatches': {'U': 0, 'L': 0, 'D': 0, 'C': 0, 'H': 0},
            'rows': {'random': {'unique_targets_U': 113.1, 'loss_slots_L': 160.2,
                                'duplicate_slots_D': 47.1, 'context_tokens': 70.4,
                                'target_purity_pct': 35.9},
                     'random_cb': dict(shadow, target_purity_pct=36.7),
                     'centroid': dict(shadow, target_purity_pct=45.9)}}


def build_design(root, centroid='A', envelope='A', matched='M+', seeds=(1234, 5678),
                 drop=(), late=(), n_test=3000, n_boot=1000, with_md=True, quiet=True,
                 cb_seed=1234, cb_placement=None, cb_late=False):
    """Probe dirs + unseal receipts + real cr_stats final (+ MD JSON) under ``root``."""
    root = str(root)
    probes = os.path.join(root, 'probes')
    os.makedirs(probes)
    cohort = Cohort(n_test)
    values = primary_aucs(centroid, envelope, seeds)
    complete = [s for s in seeds if not any((a, s) in drop or (a, s) in late
                                            for a in ('random', 'centroid', 'envelope'))]
    realizations = [RANDOM[None]] + [RANDOM[s] for s in complete]
    spread = max(realizations) - min(realizations)
    if matched not in ('M_na',) and cb_seed in seeds:
        values[('random_cb', cb_seed)] = cb_auc(matched, values, spread, cb_seed, cb_placement)
    sink = io.StringIO()
    runs = []
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        for (arm, seed), auc in sorted(values.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
            if (arm, seed) in drop:
                continue
            role = 'anchor' if seed is None else 'new'
            name = ('orig_%s' % arm) if seed is None else '%s_s%d' % (arm, seed)
            aucs = {}
            for variant in VARIANTS:
                if variant == 'patchmax_slicemean' and (arm, seed) == ('random', None):
                    continue  # legacy RANDOM anchor: no weights, no patch-max cache
                aucs[variant] = auc + POOL_SHIFT[variant] + POOL_ARM.get((variant, arm), 0.0)
            stop = LATE if ((arm, seed) in late or (arm == 'random_cb' and
                                                     (matched == 'M_desc' or cb_late))) else None
            runs.append(make_probe(probes, cohort, name, arm, seed, role, aucs, stop_time=stop))
        for run_dir in runs:
            cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
        final_path = os.path.join(root, 'final.json')
        final = cr_stats.cmd_final(runs, n_boot=n_boot, out=final_path, allow_before_freeze=True)
        inventory = os.path.join(root, 'inventory.json')
        cr_stats.cmd_inventory(runs, inventory)
    md_path = None
    if with_md:
        md_path = os.path.join(root, 'md_test.json')
        with open(md_path, 'w') as stream:
            json.dump(md_payload(final, cohort.labels), stream, indent=1)
    cb_path = os.path.join(root, 'cb_audit.json')
    with open(cb_path, 'w') as stream:
        json.dump(cb_audit_payload(), stream, indent=1)
    return {'root': root, 'runs': runs, 'final': final_path, 'md': md_path,
            'inventory': inventory, 'cb_audit': cb_path, 'values': values, 'spread': spread,
            'expected': {'centroid': centroid, 'envelope': envelope, 'matched': matched}}
