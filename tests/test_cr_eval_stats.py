"""cr_seed_v1 statistics (E11/E18/E14): inventory identities, validation-only gate,
unseal/final analysis on synthetic sealed probes, p-value formatting. CPU only."""
import datetime
import json
import os
import shutil
import sys
import zlib

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import eval_downstream as evaluation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'autopilot'))
import cr_stats  # noqa: E402

PRIMARY = evaluation.PRIMARY_VARIANT
N_VAL, N_TEST = 120, 400
Y_VAL = np.arange(N_VAL) % 2
Y_TEST = np.arange(N_TEST) % 2
COMMON = np.random.default_rng(7).normal(0, 1, N_VAL + N_TEST)


def _manifest(split, n):
    prefix = split[0].lower()
    return {'split': split, 'dataset_root': 'synthetic', 'num_slices': 4,
            'subject_ids': ['%s%04d' % (prefix, i) for i in range(n)],
            'ordered_files': [{'name': '%s%04d.npz' % (prefix, i)} for i in range(n)],
            'dataset_identity_kind': 'synthetic'}


def _probs(labels, signal, common, rng):
    z = signal * labels + common + 0.25 * rng.normal(0, 1, labels.size)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def make_run(root, name, arm, seed, role, val_signal=1.0, test_signal=1.0, uuid=None,
             ckpt=None, epoch=50, variants=(PRIMARY,), head_seeds=(42, 43), stop_time=None,
             finished=None):
    out = os.path.join(str(root), name)
    os.makedirs(out)
    rng = np.random.default_rng(zlib.crc32(name.encode('utf-8')))
    identity = {'arm': arm, 'train_seed': seed, 'role': role,
                'run_uuid': uuid if uuid is not None else 'uuid-' + name,
                'checkpoint_sha256': ckpt or ('%064x' % zlib.crc32(name.encode('utf-8'))),
                'epoch': epoch}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    if role == 'new':
        # The trainer's epoch-50 stop file, passed to the probe as --run-provenance.
        stop = os.path.join(str(root), 'train_' + name, 'stop_epoch_050.json')
        os.makedirs(os.path.dirname(stop))
        with open(stop, 'w') as stream:
            json.dump({'schema': 'jepa_stop_epoch_v1', 'epoch': epoch,
                       'timestamp_utc': stop_time or now,
                       'checkpoints': [{'role': 'periodic', 'epoch': epoch,
                                        'sha256': identity['checkpoint_sha256']}]}, stream)
        identity.update({'run_provenance': os.path.realpath(stop),
                         'run_provenance_sha256': evaluation.file_sha256(stop)})
    val_manifest, test_manifest = _manifest('Validation', N_VAL), _manifest('Test', N_TEST)
    summaries, entries = {}, []
    for variant in variants:
        per_seed = []
        for head_seed in head_seeds:
            seed_dir = os.path.join(out, variant, 'seed%d' % head_seed)
            os.makedirs(seed_dir)
            pv = _probs(Y_VAL, val_signal, COMMON[:N_VAL], rng)
            pt = _probs(Y_TEST, test_signal, COMMON[N_VAL:], rng)
            val_path = os.path.join(seed_dir, 'val_predictions.npz')
            evaluation.save_predictions(val_path, Y_VAL.astype(np.float32), pv, val_manifest)
            test_path = os.path.join(seed_dir, 'test_predictions_sealed_seed%d.npz' % head_seed)
            evaluation.save_predictions(test_path, Y_TEST.astype(np.float32), pt,
                                        dict(test_manifest, sealed=True))
            sidecar = os.path.splitext(test_path)[0] + '.manifest.json'
            rec = {'variant': variant, 'head_seed': head_seed, 'best_epoch': 10,
                   'best_val_auc': float(roc_auc_score(Y_VAL, pv)),
                   'val_predictions': os.path.relpath(val_path, out),
                   'test_predictions': os.path.relpath(test_path, out),
                   'test_predictions_sha256': evaluation.file_sha256(test_path),
                   'test_auc': None, 'test_sealed': True}
            per_seed.append(rec)
            entries.append({'variant': variant, 'head_seed': head_seed,
                            'path': rec['test_predictions'],
                            'sha256': rec['test_predictions_sha256'],
                            'sidecar_path': os.path.relpath(sidecar, out),
                            'sidecar_sha256': evaluation.file_sha256(sidecar),
                            'head_checkpoint_sha256': 'h'})
        vals = [r['best_val_auc'] for r in per_seed]
        summaries[variant] = {'per_seed': per_seed, 'val_auc_mean': float(np.mean(vals)),
                              'val_auc_sd': float(np.std(vals, ddof=1)), 'test_auc_mean': None}
    evaluation.write_sealed_manifest(out, entries, Y_TEST, test_manifest, identity)
    config = {'mode': 'patch', 'data': {'num_slices': 100, 'use_amp': False},
              'model': {'encoder_checkpoint': 'ckpt-' + name, 'probe_type': 'mean_pool'},
              'training': {'epochs': 50}, 'probe': {'head_seeds': list(head_seeds)}}
    results = {'schema': 'cr_probe_v1', 'status': 'complete', 'sealed': True,
               'head_seeds': list(head_seeds), 'identity': identity, 'variants': summaries,
               'best_val_auc': summaries[PRIMARY]['val_auc_mean'], 'test_auc': None,
               'config': config, 'finished': finished or now}
    with open(os.path.join(out, 'results.json'), 'w') as stream:
        json.dump(results, stream)
    return out


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def test_inventory_table_and_refuses_duplicates(tmp_path, capsys):
    a = make_run(tmp_path, 'r1', 'random', 1234, 'new', uuid='u1', ckpt='c1',
                 variants=(PRIMARY, 'patchmean_slicemax'))
    b = make_run(tmp_path, 'c1', 'centroid', 1234, 'new', uuid='u2', ckpt='c2')
    out = str(tmp_path / 'inv.json')
    inv = cr_stats.cmd_inventory([a, b], out)
    assert len(inv['rows']) == 2 * 2 + 2 and inv['n_runs'] == 2
    assert {r['variant'] for r in inv['runs']} == {PRIMARY, 'patchmean_slicemax'}
    assert json.load(open(out))['n_runs'] == 2
    assert 'u1' in capsys.readouterr().out
    keys = {(r['run_uuid'], r['checkpoint_sha256'], r['arm'], r['train_seed'], r['epoch'],
             r['variant'], r['head_seed']) for r in inv['rows']}
    assert len(keys) == len(inv['rows'])

    dup = make_run(tmp_path, 'r1_copy', 'random', 1234, 'new', uuid='u1', ckpt='c1')
    with pytest.raises(cr_stats.IntegrityError, match='duplicate identity'):
        cr_stats.cmd_inventory([a, dup])
    relabel = make_run(tmp_path, 'relabel', 'centroid', 1234, 'new', uuid='u3', ckpt='c1')
    with pytest.raises(cr_stats.IntegrityError, match='conflicting labels'):
        cr_stats.cmd_inventory([a, relabel])
    second = make_run(tmp_path, 'r1_other', 'random', 1234, 'new', uuid='u4', ckpt='c4')
    with pytest.raises(cr_stats.IntegrityError, match='several checkpoints'):
        cr_stats.cmd_inventory([a, second])
    anonymous = make_run(tmp_path, 'anon', 'envelope', 1234, 'new', uuid='', ckpt='c5')
    with pytest.raises(cr_stats.IntegrityError, match='missing identity fields'):
        cr_stats.cmd_inventory([anonymous])
    anchor = make_run(tmp_path, 'anchor_no_uuid', 'random', None, 'anchor', uuid='', ckpt='c9' * 8)
    row = cr_stats.cmd_inventory([anchor])['rows'][0]
    assert row['run_uuid'] == 'anchor-random-ep50-' + ('c9' * 6) and row['run_uuid_derived']
    assert cr_stats.main(['inventory', '--runs', a, dup]) == 2


# ---------------------------------------------------------------------------
# gate (validation only)
# ---------------------------------------------------------------------------

def _drop_test_files(run_dir):
    for root, _, files in os.walk(run_dir):
        for name in files:
            if name.startswith('test_predictions') or name.startswith('sealed_manifest'):
                os.remove(os.path.join(root, name))


def test_gate_is_validation_only_and_flags(tmp_path):
    anchor = make_run(tmp_path, 'anchor_r', 'random', None, 'anchor', val_signal=1.0)
    close = make_run(tmp_path, 'r_close', 'random', 1234, 'new', val_signal=1.0)
    far = make_run(tmp_path, 'r_far', 'random', 5678, 'new', val_signal=2.5)
    other_arm = make_run(tmp_path, 'c_new', 'centroid', 1234, 'new')
    for run_dir in (anchor, close, far):
        _drop_test_files(run_dir)  # the gate must not need any test prediction
    g_far = cr_stats.gate_pair(far, anchor)
    g_close = cr_stats.gate_pair(close, anchor)
    assert g_far['validation_only'] and g_far['primary_flag'] is True
    expected = np.mean([r['best_val_auc'] for r in json.load(open(
        os.path.join(far, 'results.json')))['variants'][PRIMARY]['per_seed']]) - np.mean(
        [r['best_val_auc'] for r in json.load(open(os.path.join(
            anchor, 'results.json')))['variants'][PRIMARY]['per_seed']])
    assert g_far['primary_delta'] == pytest.approx(expected, abs=1e-12)
    assert abs(g_close['primary_delta']) < abs(g_far['primary_delta'])
    assert g_close['primary_flag'] == (abs(g_close['primary_delta']) > 0.01)
    with pytest.raises(cr_stats.IntegrityError, match='same arm'):
        cr_stats.gate_pair(other_arm, anchor)
    assert cr_stats.main(['gate', '--run', far, '--anchor', anchor, '--fail-on-flag',
                          '--out', str(tmp_path / 'gate.json')]) == 3
    assert json.load(open(str(tmp_path / 'gate.json')))['gates'][0]['primary_flag'] is True
    pairs, unpaired = cr_stats._gate_pairs_from_inventory(
        {'rows': cr_stats.inventory_rows([anchor, close, far, other_arm])})
    assert sorted(pairs) == sorted([(os.path.realpath(close), os.path.realpath(anchor)),
                                    (os.path.realpath(far), os.path.realpath(anchor))])
    assert unpaired == [os.path.realpath(other_arm)]


# ---------------------------------------------------------------------------
# bootstrap machinery and outcome labels
# ---------------------------------------------------------------------------

def test_weighted_auc_matches_explicit_resample():
    rng = np.random.default_rng(3)
    labels = np.r_[np.ones(40), np.zeros(55)].astype(int)
    scores = np.round(rng.normal(0, 1, labels.size) + 0.7 * labels, 1)  # many ties
    auc = cr_stats.WeightedAUC(scores, labels)
    assert auc(np.ones((1, labels.size)))[0] == pytest.approx(roc_auc_score(labels, scores),
                                                              abs=1e-12)
    weights = cr_stats.bootstrap_weights(labels, 25, np.random.default_rng(5))
    assert np.all(weights[:, labels == 1].sum(axis=1) == 40)
    assert np.all(weights[:, labels == 0].sum(axis=1) == 55)
    got = auc(weights)
    for row, value in zip(weights, got):
        idx = np.repeat(np.arange(labels.size), row.astype(int))
        assert value == pytest.approx(roc_auc_score(labels[idx], scores[idx]), abs=1e-12)


def test_outcome_labels_follow_preregistration():
    random_values = [0.864, 0.866, 0.868]  # range 0.004
    assert cr_stats.outcome_label([0.010, 0.006], random_values) == 'consistent_direction'
    assert cr_stats.outcome_label([0.010, -0.001], random_values) == \
        'within_run_to_run_variation'
    assert cr_stats.outcome_label([0.003, 0.004], random_values) == \
        'within_run_to_run_variation'
    assert cr_stats.outcome_label([0.002, -0.004], random_values) == 'reversed'
    assert cr_stats.outcome_label([0.0, 0.0], random_values) == 'reversed'
    assert cr_stats.outcome_label([], random_values) == 'not_available'
    # V2 P1-1: incomplete designs never get a directional label.
    assert cr_stats.outcome_label([0.5], random_values) == 'incomplete'
    assert cr_stats.outcome_label([0.01, 0.01], [0.86, 0.861]) == 'incomplete'
    assert cr_stats.outcome_label([0.01, 0.01], [0.86]) == 'incomplete'
    assert cr_stats.outcome_label([-0.01], random_values) == 'incomplete'
    # Amendment 1: three complete seed indices, four RANDOM realizations.
    four = [0.864, 0.865, 0.867, 0.868]  # range 0.004
    assert cr_stats.outcome_label([0.010, 0.006, 0.008], four) == 'consistent_direction'
    assert cr_stats.outcome_label([0.010, 0.006, -0.001], four) == 'within_run_to_run_variation'
    assert cr_stats.outcome_label([0.003, 0.004, 0.005], four) == 'within_run_to_run_variation'
    assert cr_stats.outcome_label([-0.010, 0.002, 0.001], four) == 'reversed'
    assert cr_stats.outcome_label([0.010, 0.006, 0.008], random_values) == 'incomplete'
    assert 9012 in cr_stats.REGISTERED_SEEDS


def test_freeze_guard():
    with pytest.raises(cr_stats.IntegrityError, match='freeze'):
        cr_stats.check_freeze(now=datetime.datetime.fromisoformat('2026-10-22T07:59:00-07:00'))
    assert cr_stats.check_freeze(
        now=datetime.datetime.fromisoformat('2026-10-22T08:00:00-07:00')) is False
    assert cr_stats.check_freeze(True, now=datetime.datetime.fromisoformat(
        '2026-10-09T00:00:00-07:00')) is True


# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------

HEADS = (42, 43)


@pytest.fixture(scope='module')
def design(tmp_path_factory):
    """Complete registered design (anchors, R/C/E x 2 seeds, CB s1234), unsealed."""
    root = tmp_path_factory.mktemp('design')
    variants = (PRIMARY, 'patchmean_slicemax')
    runs = {
        'orig_random': make_run(root, 'orig_random', 'random', None, 'anchor', test_signal=1.00,
                                variants=variants),
        'orig_centroid': make_run(root, 'orig_centroid', 'centroid', None, 'anchor',
                                  test_signal=1.6, variants=variants),
        'orig_envelope': make_run(root, 'orig_envelope', 'envelope', None, 'anchor',
                                  test_signal=0.9, variants=(PRIMARY,)),
    }
    for seed in (1234, 5678):
        for prefix, arm, signal in (('r', 'random', 1.0), ('c', 'centroid', 1.6),
                                    ('e', 'envelope', 0.6)):
            name = '%s_%d' % (prefix, seed)
            runs[name] = make_run(root, name, arm, seed, 'new', test_signal=signal,
                                  variants=variants)
    runs['cb_1234'] = make_run(root, 'cb_1234', 'random_cb', 1234, 'new', test_signal=1.3,
                               variants=variants)
    with pytest.raises(cr_stats.IntegrityError, match='freeze'):
        cr_stats.cmd_final(list(runs.values()), n_boot=50, head_seeds=HEADS)
    with pytest.raises(cr_stats.IntegrityError, match='not been unsealed'):
        cr_stats.cmd_final(list(runs.values()), n_boot=50, allow_before_freeze=True,
                           head_seeds=HEADS)
    unsealed = {name: cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
                for name, run_dir in runs.items()}
    return root, runs, unsealed


def final(run_dirs, **kwargs):
    kwargs.setdefault('n_boot', 200)
    return cr_stats.cmd_final(list(run_dirs), allow_before_freeze=True, head_seeds=HEADS,
                              **kwargs)


def test_final_primary_analysis_end_to_end(design, tmp_path):
    root, runs, unsealed = design
    out = str(tmp_path / 'final.json')
    res = final(runs.values(), n_boot=300, out=out)
    assert res['seed_level_inference'].startswith('none')
    primary = res['analysis'][PRIMARY]
    assert len(primary['primary_contrasts']) == 4
    assert {c['kind'] for c in primary['matched_control']} == {'centroid-random_cb',
                                                               'random_cb-random'}
    for c in primary['primary_contrasts'] + primary['matched_control']:
        assert c['ci95'][0] <= c['delta'] <= c['ci95'][1]
    for name, payload in unsealed.items():
        assert primary['per_run_test_auc'][name] == pytest.approx(
            payload['variants'][PRIMARY]['test_auc_mean'], abs=1e-12)
    outcomes = primary['outcomes']
    assert outcomes['centroid']['label'] == 'consistent_direction'
    assert outcomes['envelope']['label'] == 'reversed'
    assert outcomes['centroid']['design_complete'] is True
    assert len(outcomes['centroid']['random_realizations']) == 3
    sens = outcomes['centroid']['sensitivity_with_original']
    assert sens['n'] == 3 and sens['descriptive_only'] is True
    # orig_envelope has no secondary variant: that contrast set simply lacks the original.
    secondary = res['analysis']['patchmean_slicemax']['outcomes']['envelope']
    assert secondary['sensitivity_with_original']['original_delta'] is None
    assert json.load(open(out))['n_boot'] == 300
    # Same seed -> identical intervals (reproducible bootstrap).
    again = final(runs.values(), n_boot=300)
    assert again['analysis'][PRIMARY]['primary_contrasts'] == primary['primary_contrasts']


def test_v2_runs_finishing_after_the_freeze_are_not_primary(design, tmp_path):
    root, runs, _ = design
    late = '2026-10-22T08:00:01-07:00'
    variants = (PRIMARY, 'patchmean_slicemax')
    for case, kwargs in (('stop_file_late', {'stop_time': late}),
                         ('probe_late', {'finished': late})):
        extra = []
        for prefix, arm, signal in (('r', 'random', 1.0), ('c', 'centroid', 1.6),
                                    ('e', 'envelope', 0.6)):
            name = '%s_9012_%s' % (prefix, case)
            extra.append(make_run(tmp_path / case, name, arm, 9012, 'new', test_signal=signal,
                                  variants=variants, **(kwargs if arm == 'envelope' else {})))
            cr_stats.cmd_unseal(extra[-1], allow_before_freeze=True)
        res = final(list(runs.values()) + extra, n_boot=50)
        block = res['analysis'][PRIMARY]
        assert block['complete_seed_indices'] == [1234, 5678], case
        assert block['incomplete_seed_indices'] == [9012], case
        reasons = {(c['kind'], c['reason']) for c in block['descriptive_incomplete_contrasts']}
        assert reasons == {('centroid-random', 'seed index incomplete'),
                           ('envelope-random', 'run not finished before the freeze')}, case
        late_run = res['runs'][os.path.basename(extra[-1])]['eligibility']
        assert late_run['eligible'] is False and any('after the freeze' in r
                                                      for r in late_run['reasons'])
        assert len(block['outcomes']['centroid']['random_realizations']) == 3
    # A new run without a valid stop file is not eligible either.
    orphan = make_run(tmp_path / 'orphan', 'e_9012_nostop', 'envelope', 9012, 'new',
                      variants=variants)
    os.remove(json.load(open(os.path.join(orphan, 'results.json')))['identity']['run_provenance'])
    elig = cr_stats.completion_eligibility(
        orphan, cr_stats.load_run(orphan), cr_stats.read_seal_manifest(orphan)[0])
    assert elig['eligible'] is False and 'no epoch-50 stop file' in elig['reasons']


def test_v2_incomplete_designs_are_labelled_incomplete(design):
    root, runs, _ = design
    cases = {'missing_E2': ['e_5678'], 'missing_anchor': ['orig_random'],
             'missing_R2': ['r_5678']}
    for case, drop in cases.items():
        res = final([d for n, d in runs.items() if n not in drop], n_boot=50)
        block = res['analysis'][PRIMARY]
        outcomes = block['outcomes']
        for arm in ('centroid', 'envelope'):
            assert outcomes[arm]['label'] == 'incomplete', (case, arm)
            assert outcomes[arm]['design_complete'] is False
        if case == 'missing_E2':
            # Amendment 1: index 5678 lacks ENVELOPE, so C2-R2 is descriptive only.
            assert block['complete_seed_indices'] == [1234]
            assert block['incomplete_seed_indices'] == [5678]
            assert [(c['kind'], c['train_seed']) for c in
                    block['descriptive_incomplete_contrasts']] == [('centroid-random', 5678)]
            assert outcomes['centroid']['n_new_seeds'] == 1


def test_amendment1_third_seed_index(design, tmp_path):
    root, runs, _ = design
    variants = (PRIMARY, 'patchmean_slicemax')
    extra = {}
    for prefix, arm, signal in (('r', 'random', 1.0), ('c', 'centroid', 1.6),
                                ('e', 'envelope', 0.6)):
        name = '%s_9012' % prefix
        extra[name] = make_run(tmp_path, name, arm, 9012, 'new', test_signal=signal,
                               variants=variants)
        cr_stats.cmd_unseal(extra[name], allow_before_freeze=True)
    # Three complete indices: three deltas, four RANDOM realizations.
    res = final(list(runs.values()) + list(extra.values()), n_boot=100)
    block = res['analysis'][PRIMARY]
    assert block['complete_seed_indices'] == [1234, 5678, 9012]
    assert len(block['primary_contrasts']) == 6 and block['descriptive_incomplete_contrasts'] == []
    outcomes = block['outcomes']
    assert outcomes['centroid']['label'] == 'consistent_direction'
    assert outcomes['envelope']['label'] == 'reversed'
    assert len(outcomes['centroid']['random_realizations']) == 4
    assert outcomes['centroid']['design_complete'] is True
    # Index 9012 without ENVELOPE: descriptive only, R3 excluded from the label's range.
    res = final(list(runs.values()) + [extra['r_9012'], extra['c_9012']], n_boot=100)
    block = res['analysis'][PRIMARY]
    assert block['complete_seed_indices'] == [1234, 5678]
    assert block['incomplete_seed_indices'] == [9012]
    assert [(c['kind'], c['train_seed'], c['descriptive_only']) for c in
            block['descriptive_incomplete_contrasts']] == [('centroid-random', 9012, True)]
    outcomes = block['outcomes']
    assert len(outcomes['centroid']['random_realizations']) == 3
    assert outcomes['centroid']['label'] == 'consistent_direction'
    assert outcomes['centroid']['random_range_all_completed_runs'] >= \
        outcomes['centroid']['random_range']


def _clone(src, dst):
    shutil.copytree(src, dst)
    return str(dst)


def test_v2_results_identity_must_match_the_seal(design, tmp_path):
    root, runs, _ = design
    for field, value in (('arm', 'centroid'), ('checkpoint_sha256', 'f' * 64),
                         ('run_uuid', 'relabelled')):
        # Drop the run whose slot the relabelled clone would take, so only the seal check fires.
        drop = {'r_1234', 'c_1234'} if field == 'arm' else {'r_1234'}
        others = [d for n, d in runs.items() if n not in drop]
        clone = _clone(runs['r_1234'], tmp_path / field)
        path = os.path.join(clone, 'results.json')
        res = json.load(open(path))
        res['identity'][field] = value
        json.dump(res, open(path, 'w'))
        with pytest.raises(cr_stats.IntegrityError, match='contradicts the seal'):
            final(others + [clone], n_boot=20)
    clone = _clone(runs['r_1234'], tmp_path / 'coverage')
    path = os.path.join(clone, 'results.json')
    res = json.load(open(path))
    res['variants'][PRIMARY]['per_seed'][0]['test_predictions_sha256'] = '0' * 64
    json.dump(res, open(path, 'w'))
    with pytest.raises(cr_stats.IntegrityError, match='differ from the sealed files'):
        final(others + [clone], n_boot=20)


def test_v2_fixed_endpoint_and_registered_design(design, tmp_path):
    root, runs, _ = design
    ep75 = make_run(tmp_path, 'zzz_random_1234_ep75', 'random', 1234, 'new', epoch=75,
                    variants=(PRIMARY, 'patchmean_slicemax'))
    cr_stats.cmd_unseal(ep75, allow_before_freeze=True)
    with pytest.raises(cr_stats.IntegrityError, match='registered endpoint'):
        final(list(runs.values()) + [ep75], n_boot=20)
    with pytest.raises(cr_stats.IntegrityError, match='epoch'):
        cr_stats.gate_pair(ep75, runs['orig_random'])
    odd = make_run(tmp_path, 'r_9999', 'random', 9999, 'new',
                   variants=(PRIMARY, 'patchmean_slicemax'))
    cr_stats.cmd_unseal(odd, allow_before_freeze=True)
    with pytest.raises(cr_stats.IntegrityError, match='not registered'):
        final(list(runs.values()) + [odd], n_boot=20)
    with pytest.raises(cr_stats.IntegrityError, match='head seeds'):
        cr_stats.cmd_final(list(runs.values()), n_boot=20, allow_before_freeze=True)
    recipe = _clone(runs['c_5678'], tmp_path / 'recipe')
    path = os.path.join(recipe, 'results.json')
    res = json.load(open(path))
    res['config']['training']['epochs'] = 30
    json.dump(res, open(path, 'w'))
    others = [d for n, d in runs.items() if n != 'c_5678']
    with pytest.raises(cr_stats.IntegrityError, match='recipe'):
        final(others + [recipe], n_boot=20)


def test_unseal_is_once_only(design):
    root, runs, unsealed = design
    run_dir = runs['r_1234']
    before = open(os.path.join(run_dir, 'unsealed_results.json')).read()
    again = cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    assert again == unsealed['r_1234']
    assert open(os.path.join(run_dir, 'unsealed_results.json')).read() == before
    assert 'reverified_at' in open(os.path.join(run_dir, 'unseal_log.jsonl')).read()


def test_final_refuses_mismatched_test_cases(tmp_path):
    a = make_run(tmp_path, 'ra', 'random', 1234, 'new')
    b = make_run(tmp_path, 'ca', 'centroid', 1234, 'new')
    for run_dir in (a, b):
        cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    manifest_path = os.path.join(b, 'sealed_manifest.json')
    manifest = json.load(open(manifest_path))
    manifest['test_label_identity']['subject_ids_sha256'] = '0' * 64
    with open(manifest_path, 'w') as stream:
        json.dump(manifest, stream)
    with open(manifest_path + '.sha256', 'w') as stream:
        stream.write(evaluation.file_sha256(manifest_path) + '  sealed_manifest.json\n')
    with pytest.raises(cr_stats.IntegrityError, match='not bound to the current seal'):
        final([a, b], n_boot=20)
    receipt = os.path.join(b, 'unsealed_results.json')
    payload = json.load(open(receipt))
    payload['sealed_manifest_sha256'] = evaluation.file_sha256(manifest_path)
    json.dump(payload, open(receipt, 'w'))
    with pytest.raises(cr_stats.IntegrityError, match='case order'):
        final([a, b], n_boot=20)


# ---------------------------------------------------------------------------
# E14: p-value formatting in the fp32 table generator
# ---------------------------------------------------------------------------

def test_fp32_table_p_value_formatting():
    import p3b_integrate_fp32 as p3b
    assert p3b.format_p(8.02747e-5) == r'$<$0.001'
    assert p3b.format_p(0.0009996) == r'$<$0.001'
    assert p3b.format_p(0.001) == '0.001'
    assert p3b.format_p(0.128) == '0.128'
    assert p3b.format_p(float('nan')) == '---'
    row = {'arm': 'random', 'epoch': 100, 'auc_fp16': 0.874581, 'auc_fp32': 0.874485,
           'delta_fp32_minus_fp16': -0.000096, 'delong_p': 8.02747e-5}
    table = p3b.render_fp32_table([row, dict(row, arm='oracle', delong_p=0.767)])
    assert r'& $<$0.001 \\' in table and '0.000 \\\\' not in table
    assert r'\ArmBest{} & 100' in table and '& 0.767 \\\\' in table
    committed = open(os.path.join(REPO, 'paper', 'genai4health2026', 'auto', 'table_fp32.tex'),
                     encoding='utf-8').read()
    assert '& 0.000 \\\\' not in committed and r'& $<$0.001 \\' in committed
