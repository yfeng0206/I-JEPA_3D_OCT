"""Synthetic cr_seed_v1 campaign for rehearsing autopilot/cr_postfreeze.ps1 (never real results).

Writes under one root the layout the post-freeze pipeline reads after the freeze:

  campaign.json   cr_campaign_v1 subset (runs, anchors, run_root, state_dir, freeze)
  state.json      cr_campaign_state_v1 subset (train/probe status and attempts, result dirs with
                  the runner's file hashes, stop files, trainer logs)
  runs/cr_probe_<ID>_<arm>_s<seed>_ep050_<sha12>_a0     SEALED probe dirs (not unsealed): exact
                  head-seed test AUCs (cr_results_fixtures.exact_probs) and v2 MD feature caches
  runs/cr_probe_anchor_<name>_ep050_<sha12>_a0          anchors (RANDOM: legacy cache, no MD)
  runs/cr_seed_v1_<arm>_s<seed>/stop_epoch_050.json     trainer stop files (late: after the freeze)
  runs/cr_seed_v1_random_cb_s1234/train_a0.log          RANDOM-CB [CB] lines (format_cb_stats)
  metadata.csv    visual-field MD metadata of the synthetic cohort

Timestamps are fixed synthetic times (fx.ON_TIME_TIMES / fx.LATE_TIMES), never the wall clock,
so generation after the freeze gives the same eligibility (V8-5). The root holds
CR_SYNTHETIC_FIXTURE.json and every campaign.json / state.json / results.json carries
"synthetic": true, which the rehearsal preflight requires (V8-1).

Run states: on time (default), ``late`` (stop file after the freeze; probe complete),
``in_progress`` (probe running at the freeze: partial dir without results.json) and ``absent``
(training failed; no probe). AUC patterns and the matched-control construction are those of
tests/cr_results_fixtures.py.
"""
import hashlib
import json
import os
import sys
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'autopilot'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cr_results_fixtures as fx  # noqa: E402
import cr_stats  # noqa: E402
from src import eval_downstream as evaluation  # noqa: E402

RUNS = (('R1', 'random', 1234), ('C1', 'centroid', 1234), ('E1', 'envelope', 1234),
        ('R2', 'random', 5678), ('C2', 'centroid', 5678), ('CB', 'random_cb', 1234),
        ('E2', 'envelope', 5678), ('R3', 'random', 9012), ('C3', 'centroid', 9012),
        ('E3', 'envelope', 9012))
ANCHORS = (('anchor_centroid_ep50', 'centroid'), ('anchor_envelope_ep50', 'envelope'),
           ('anchor_random_ep50_legacy_cache', 'random'))
MD_DIM, MD_SLICES = 16, 4
FIXTURE_MARKER = 'CR_SYNTHETIC_FIXTURE.json'


def _sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=1)
    return evaluation.file_sha256(path)


class MDCohort(object):
    """Training/Validation/Test cases; Test equals the sealed-prediction cohort (same labels)."""

    def __init__(self, cohort, n_train, n_val, seed=7):
        rng = np.random.default_rng(seed)
        self.splits = {}
        for split, n, prefix in (('Training', n_train, 'tr'), ('Validation', n_val, 'va')):
            glaucoma = rng.permutation(np.arange(n) < int(round(0.49 * n)))
            self.splits[split] = {'names': ['%s%05d.npz' % (prefix, i) for i in range(n)],
                                  'glaucoma': glaucoma}
        test = cohort.manifest
        self.splits['Test'] = {'names': [f['name'] for f in test['ordered_files']],
                               'glaucoma': cohort.labels.astype(bool)}
        for split in self.splits.values():
            g = split['glaucoma']
            split['md'] = np.where(g, rng.normal(-7, 4, g.size), rng.normal(0, 1.5, g.size)).round(2)
        self.weights = rng.normal(0, 1, MD_DIM)

    def write_metadata(self, path):
        use = {'Training': 'training', 'Validation': 'validation', 'Test': 'test'}
        lines = ['filename,age,md,glaucoma,use']
        for split, c in self.splits.items():
            for name, g, md in zip(c['names'], c['glaucoma'], c['md']):
                lines.append('%s,60,%s,%s,%s' % (name, md, 'yes' if g else 'no', use[split]))
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')


def _md_caches(out, md, ckpt, noise, legacy, rng):
    import torch
    os.makedirs(os.path.join(out, 'feature_cache'), exist_ok=True)
    prov = {}
    for split, c in md.splits.items():
        z = (c['md'] - c['md'].mean()) / 4.0
        vol = np.outer(z, md.weights) + noise * rng.normal(0, 1, (z.size, MD_DIM))
        feats = vol[:, None, :] + 0.05 * rng.normal(0, 1, (z.size, MD_SLICES, MD_DIM))
        manifest = {'split': split, 'ordered_files': [{'name': n} for n in c['names']],
                    'subject_ids': [os.path.splitext(n)[0] for n in c['names']],
                    'dataset_root': 'synthetic', 'encoder_source': {'checkpoint_sha256': ckpt}}
        path = os.path.join(out, 'feature_cache', '%s_v2_%s.pt' % (split, ckpt[:12]))
        payload = {'features': torch.tensor(feats, dtype=torch.float32),
                   'labels': torch.tensor(c['glaucoma'].astype(np.int64))}
        if not legacy:
            payload['source_manifest'] = manifest
        torch.save(payload, path)
        rec = {'state': 'legacy_verified' if legacy else 'cold', 'path': path,
               'identity_sha256': evaluation._identity_digest(manifest),
               'file_sha256': evaluation.file_sha256(path)}
        if legacy:
            rec['evidence'] = {'row_identity': 'label sequence only (synthetic legacy cache)'}
        prov[split] = {'primary': rec}
    return prov


def make_probe(runs_root, cohort, md, name, arm, seed, role, aucs, ckpt, stop=None,
               legacy=False, late=False):
    """Sealed probe dir in the eval_downstream layout (cr_results_fixtures.make_probe + MD)."""
    out = os.path.join(runs_root, name)
    os.makedirs(out)
    rng = np.random.default_rng(zlib.crc32(name.encode('utf-8')))
    identity = {'arm': arm, 'train_seed': seed, 'role': role, 'checkpoint_sha256': ckpt,
                'epoch': 50, 'run_uuid': 'synthetic-' + (name.split('_ep050_')[0])}
    if stop is not None:
        identity.update({'run_provenance': os.path.realpath(stop),
                         'run_provenance_sha256': evaluation.file_sha256(stop),
                         'run_provenance_checkpoint_role': 'periodic'})
    run_latent = cohort.common + 0.22 * rng.normal(0, 1, cohort.labels.size)
    summaries, entries = {}, []
    for variant, auc in aucs.items():
        offsets = rng.normal(0, 1, len(fx.HEADS))
        offsets = fx.HEAD_SD * (offsets - offsets.mean()) / (offsets.std(ddof=1) or 1.0)
        variant_latent = run_latent + 0.1 * rng.normal(0, 1, cohort.labels.size)
        per_seed = []
        for head_seed, offset in zip(fx.HEADS, offsets):
            seed_dir = os.path.join(out, variant, 'seed%d' % head_seed)
            os.makedirs(seed_dir)
            latent = variant_latent + 0.05 * rng.normal(0, 1, cohort.labels.size)
            probs = fx.exact_probs(cohort.labels, auc + offset, latent, rng)
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
    times = fx.LATE_TIMES if late else fx.ON_TIME_TIMES
    fx.set_seal_created(out, times['seal'])  # fixed synthetic time, not the wall clock (V8-5)
    noise = float(np.clip(1.0 + 40.0 * (0.87 - aucs[fx.PRIMARY]), 0.3, 3.0))
    prov = _md_caches(out, md, ckpt, noise, legacy, rng)
    config = {'mode': 'patch', 'data': {'num_slices': 100, 'use_amp': False},
              'model': {'encoder_checkpoint': 'synthetic-' + name, 'probe_type': 'mean_pool'},
              'training': {'epochs': 50, 'lr': 4e-4}, 'probe': {'head_seeds': list(fx.HEADS)}}
    results = {'schema': 'cr_probe_v1', 'status': 'complete', 'sealed': True, 'synthetic': True,
               'head_seeds': list(fx.HEADS), 'identity': identity, 'variants': summaries,
               'best_val_auc': summaries[fx.PRIMARY]['val_auc_mean'], 'test_auc': None,
               'config': config, 'cache_provenance': prov, 'started': times['started'],
               'finished': times['finished']}
    _write_json(os.path.join(out, 'results.json'), results)
    return out


def _cb_log(path, n=40):
    from src.train_patch import format_cb_stats
    base = dict(matching='strict', bypass_ramp0=0, target_len=40, context=81, loss_slots=160,
                unique_targets=102.5, duplicates=57.5, hidden_shadow=112.8, hidden_matched=112.8,
                proposals_mean=4000.0, proposals_max=20000, exact_fallbacks=0, shadow_ms=80.0,
                match_ms=40.0)
    rng = np.random.default_rng(3)
    lines = ['synthetic RANDOM-CB trainer log (rehearsal fixture)']
    for i in range(n):
        ramp = min(1.0, (i + 1) / 10.0)
        u = 102.5 + rng.normal(0, 1.5)
        lines.append('epoch 26 batch %d loss 0.1190' % i)
        lines.append(format_cb_stats(dict(base, r_t=ramp, unique_targets=u, duplicates=160 - u,
                                          context=int(80 + rng.integers(-2, 3)),
                                          bypass_ramp0=int(ramp == 0))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')


def build_campaign(root, centroid='A', envelope='A', matched='M+', seeds=(1234, 5678, 9012),
                   late=(), in_progress=(), absent=(), n_test=3000, n_train=6000, n_val=1000,
                   cb_placement=None):
    """Sealed synthetic campaign under ``root``; returns the paths the pipeline needs."""
    root = os.path.abspath(str(root))
    runs_root = os.path.join(root, 'runs')
    os.makedirs(runs_root)
    late, in_progress, absent = set(late), set(in_progress), set(absent)
    if matched == 'M_na':
        absent.add(('random_cb', 1234))
    cohort = fx.Cohort(n_test)
    md = MDCohort(cohort, n_train, n_val)
    md.write_metadata(os.path.join(root, 'metadata.csv'))
    values = fx.primary_aucs(centroid, envelope, seeds)
    skipped = late | in_progress | absent
    complete = [s for s in seeds if not any((a, s) in skipped for a in cr_stats.PRIMARY_ARMS)]
    realizations = [fx.RANDOM[None]] + [fx.RANDOM[s] for s in complete]
    spread = max(realizations) - min(realizations)
    if ('random_cb', 1234) not in absent and 1234 in seeds:
        values[('random_cb', 1234)] = fx.cb_auc(matched if matched != 'M_na' else 'M+', values,
                                                spread, 1234, cb_placement)
    state = {'schema': 'cr_campaign_state_v1', 'campaign': 'cr_seed_v1', 'synthetic': True,
             'status': 'RUNNING' if in_progress else 'DONE',
             'created_at': '2026-10-08T05:36:51-07:00', 'runs': {}, 'events': []}
    anchors_state = {}

    def aucs_for(arm, seed):
        out = {}
        for variant in fx.VARIANTS:
            if variant == 'patchmax_slicemean' and (arm, seed) == ('random', None):
                continue  # legacy RANDOM anchor: no weights, no patch-max cache
            out[variant] = values[(arm, seed)] + fx.POOL_SHIFT[variant] + \
                fx.POOL_ARM.get((variant, arm), 0.0)
        return out

    def result_record(out):
        return {'out_dir': out, 'identity': None, 'files': {
            'aggregate': {'path': os.path.join(out, 'results.json'),
                          'sha256': evaluation.file_sha256(os.path.join(out, 'results.json'))},
            'sealed_manifest': {'path': os.path.join(out, 'sealed_manifest.json'),
                                'sha256': evaluation.file_sha256(
                                    os.path.join(out, 'sealed_manifest.json'))}}}

    for name, arm in ANCHORS:
        ckpt = _sha_text('synthetic-anchor-' + name)
        out = make_probe(runs_root, cohort, md, 'cr_probe_%s_ep050_%s_a0' % (name, ckpt[:12]),
                         arm, None, 'anchor', aucs_for(arm, None), ckpt,
                         legacy=(arm == 'random'))
        rec = result_record(out)
        rec['identity'] = ckpt
        anchors_state[name] = {'status': 'DONE', 'attempts': [
            {'k': 0, 'status': 'DONE', 'out_dir': out}], 'result': rec}
    state['runs']['ANCHORS'] = {'train': {'status': 'PENDING', 'attempts': []},
                                'probe': {'status': 'DONE', 'attempts': [],
                                          'items': anchors_state},
                                'spec': {'id': 'ANCHORS', 'arm': None, 'seed': None}}
    cb_logs = []
    for run_id, arm, seed in RUNS:
        if seed not in seeds:
            continue  # the campaign never reached this run
        run_dir = os.path.join(runs_root, 'cr_seed_v1_%s_s%d' % (arm, seed))
        os.makedirs(run_dir)
        log = os.path.join(run_dir, 'train_a0.log')
        if arm == 'random_cb':
            _cb_log(log)
            cb_logs.append(log)
        else:
            with open(log, 'w', encoding='utf-8') as stream:
                stream.write('synthetic trainer log (rehearsal fixture)\n')
        entry = {'spec': {'id': run_id, 'arm': arm, 'seed': seed},
                 'train': {'status': 'DONE', 'run_dir': run_dir,
                           'attempts': [{'k': 0, 'status': 'EXITED', 'log': log}]},
                 'probe': {'status': 'PENDING', 'attempts': []}}
        state['runs'][run_id] = entry
        if (arm, seed) in absent:
            entry['train']['status'] = 'FAILED'
            continue
        ckpt = _sha_text('synthetic-%s-%d' % (arm, seed))
        stop = os.path.join(run_dir, 'stop_epoch_050.json')
        stamp = (fx.LATE_TIMES if (arm, seed) in late else fx.ON_TIME_TIMES)['stop']
        entry['train']['stop_file'] = {'path': stop, 'sha256': _write_json(stop, {
            'schema': 'jepa_stop_epoch_v1', 'epoch': 50, 'timestamp_utc': stamp,
            'checkpoints': [{'role': 'periodic', 'epoch': 50, 'sha256': ckpt}]})}
        name = 'cr_probe_%s_%s_s%d_ep050_%s_a0' % (run_id, arm, seed, ckpt[:12])
        if (arm, seed) in in_progress:
            partial = os.path.join(runs_root, name)
            os.makedirs(os.path.join(partial, 'feature_cache'))
            with open(os.path.join(partial, 'probe.log'), 'w', encoding='utf-8') as stream:
                stream.write('synthetic probe in progress\n')
            entry['probe'] = {'status': 'RUNNING', 'attempts': [
                {'k': 0, 'status': 'RUNNING', 'out_dir': partial}]}
            state['current'] = {'step': '%s:probe' % run_id, 'run_dir': run_dir}
            continue
        out = make_probe(runs_root, cohort, md, name, arm, seed, 'new', aucs_for(arm, seed),
                         ckpt, stop=stop, late=(arm, seed) in late)
        rec = result_record(out)
        rec['identity'] = ckpt
        entry['probe'] = {'status': 'DONE', 'attempts': [{'k': 0, 'status': 'DONE',
                                                           'out_dir': out}], 'result': rec}
    campaign = {'schema': 'cr_campaign_v1', 'campaign': 'cr_seed_v1', 'synthetic': True,
                'description': 'SYNTHETIC rehearsal campaign (tests/cr_postfreeze_fixtures.py)',
                'run_root': runs_root, 'state_dir': root, 'freeze': cr_stats.FREEZE,
                'runs': [{'id': i, 'arm': a, 'seed': s, 'enabled': True} for i, a, s in RUNS],
                'anchors': {'id': 'ANCHORS', 'enabled': True,
                            'items': [{'name': n, 'arm': a, 'enabled': True} for n, a in ANCHORS]}}
    _write_json(os.path.join(root, 'campaign.json'), campaign)
    state['updated_at'] = '2026-10-22T08:00:00-07:00'
    _write_json(os.path.join(root, 'state.json'), state)
    _write_json(os.path.join(root, FIXTURE_MARKER), {
        'synthetic': True, 'generator': 'tests/cr_postfreeze_fixtures.py',
        'note': 'SYNTHETIC rehearsal campaign; never real results'})
    return {'root': root, 'runs_root': runs_root, 'campaign': os.path.join(root, 'campaign.json'),
            'state': os.path.join(root, 'state.json'),
            'metadata': os.path.join(root, 'metadata.csv'), 'cb_logs': cb_logs,
            'values': {'%s/%s' % k: v for k, v in values.items()}, 'spread': spread,
            'complete_seeds': complete}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='build a synthetic sealed cr_seed_v1 campaign')
    ap.add_argument('root')
    ap.add_argument('--centroid', default='A')
    ap.add_argument('--envelope', default='A')
    ap.add_argument('--matched', default='M+')
    ap.add_argument('--seeds', default='1234,5678,9012')
    ap.add_argument('--late', nargs='*', default=[], help='arm:seed')
    ap.add_argument('--in-progress', nargs='*', default=[], help='arm:seed')
    ap.add_argument('--absent', nargs='*', default=[], help='arm:seed')
    ap.add_argument('--n-test', type=int, default=3000)
    ap.add_argument('--n-train', type=int, default=6000)
    ap.add_argument('--n-val', type=int, default=1000)
    a = ap.parse_args()

    def pairs(items):
        return [(x.split(':')[0], int(x.split(':')[1])) for x in items]

    if os.path.exists(a.root) and os.listdir(a.root):
        sys.exit('refusing to write into a non-empty directory: %s' % a.root)
    if os.path.realpath(a.root).lower().startswith(os.path.realpath(r'D:\jepa_phase0').lower()):
        sys.exit('synthetic fixtures never go under D:\\jepa_phase0')
    built = build_campaign(a.root, a.centroid, a.envelope, a.matched,
                           tuple(int(s) for s in a.seeds.split(',')), pairs(a.late),
                           pairs(a.in_progress), pairs(a.absent), a.n_test, a.n_train, a.n_val)
    print(json.dumps({k: v for k, v in built.items() if k != 'values'}, indent=1))
