"""cr_seed_v1 secondary MD severity regression (autopilot/cr_md_regression.py): seal,
metadata join, ridge/alpha selection, baselines and paired bootstrap. CPU, synthetic only."""
import datetime
import json
import os
import shutil
import sys
import zlib

import numpy as np
import pytest
import torch

from src import eval_downstream as evaluation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'autopilot'))
import cr_md_regression as md_reg  # noqa: E402
import cr_stats  # noqa: E402

N = {'Training': 160, 'Validation': 60, 'Test': 80}
USE = {'Training': 'training', 'Validation': 'validation', 'Test': 'test'}
D, S = 16, 4


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp('md')
    rng = np.random.default_rng(11)
    data_dir = root / 'data'
    cases, start = {}, 1
    for split, n in N.items():
        (data_dir / split).mkdir(parents=True)
        names = ['data_%05d.npz' % (start + i) for i in range(n)]
        start += n
        glaucoma = np.arange(n) % 2 == 1
        md = np.where(glaucoma, rng.normal(-7, 4, n), rng.normal(0, 1.5, n)).round(2)
        for name in names:
            (data_dir / split / name).write_bytes(b'')  # listing only (legacy-state runs)
        cases[split] = {'names': names, 'glaucoma': glaucoma, 'md': md}
    lines = ['filename,age,md,glaucoma,use']
    for split, c in cases.items():
        for name, g, m in zip(c['names'], c['glaucoma'], c['md']):
            lines.append('%s,50,%s,%s,%s' % (name, m, 'yes' if g else 'no', USE[split]))
    meta = root / 'meta.csv'
    meta.write_text('\n'.join(lines) + '\n')
    garbled = root / 'meta_test_garbled.csv'
    garbled.write_text('\n'.join(l if not l.endswith(',test') else l.replace(',50,', ',50,x')
                                 for l in lines) + '\n')
    weights = rng.normal(0, 1, D)
    return {'root': root, 'data_dir': data_dir, 'cases': cases, 'meta': str(meta),
            'garbled': str(garbled), 'weights': weights}


def make_md_run(cohort, name, arm, seed, role, noise, legacy=False, root=None):
    out = (root or cohort['root'] / 'runs') / name
    (out / 'feature_cache').mkdir(parents=True)
    rng = np.random.default_rng(zlib.crc32(name.encode('utf-8')))
    ckpt = '%064x' % zlib.crc32(name.encode('utf-8'))
    identity = {'arm': arm, 'train_seed': seed, 'role': role, 'run_uuid': 'u-' + name,
                'checkpoint_sha256': ckpt, 'epoch': 50}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    if role == 'new':
        stop = out.parent / ('train_' + name) / 'stop_epoch_050.json'
        stop.parent.mkdir(parents=True)
        stop.write_text(json.dumps({'schema': 'jepa_stop_epoch_v1', 'epoch': 50,
                                    'timestamp_utc': now, 'checkpoints': [
                                        {'role': 'periodic', 'sha256': ckpt, 'epoch': 50}]}))
        identity.update({'run_provenance': os.path.realpath(str(stop)),
                         'run_provenance_sha256': evaluation.file_sha256(str(stop))})
    prov = {}
    for split, c in cohort['cases'].items():
        z = (c['md'] - c['md'].mean()) / 4.0
        vol = np.outer(z, cohort['weights']) + noise * rng.normal(0, 1, (z.size, D))
        feats = vol[:, None, :] + 0.05 * rng.normal(0, 1, (z.size, S, D))
        manifest = {'split': split, 'ordered_files': [{'name': n} for n in c['names']],
                    'subject_ids': c['names'], 'dataset_root': str(cohort['data_dir']),
                    'encoder_source': {'checkpoint_sha256': ckpt}}
        path = out / 'feature_cache' / ('%s_v2_x.pt' % split)
        payload = {'features': torch.tensor(feats, dtype=torch.float32),
                   'labels': torch.tensor(c['glaucoma'].astype(np.int64))}
        if not legacy:
            payload['source_manifest'] = manifest
        torch.save(payload, path)
        rec = {'state': 'legacy_verified' if legacy else 'cold', 'path': str(path),
               'identity_sha256': evaluation._identity_digest(manifest),
               'file_sha256': evaluation.file_sha256(str(path))}
        if legacy:
            rec['evidence'] = {'row_identity': 'label sequence only (test)'}
        prov[split] = {'primary': rec}
    test = cohort['cases']['Test']
    pred_path = out / 'p' / 'seed42' / 'test_predictions_sealed_seed42.npz'
    pred_path.parent.mkdir(parents=True)
    test_manifest = {'split': 'Test', 'dataset_root': str(cohort['data_dir']),
                     'subject_ids': test['names'],
                     'ordered_files': [{'name': n} for n in test['names']]}
    evaluation.save_predictions(str(pred_path), test['glaucoma'].astype(np.float32),
                                np.full(len(test['names']), 0.5, dtype=np.float32),
                                test_manifest)
    rel = os.path.relpath(str(pred_path), str(out))
    sidecar = os.path.splitext(str(pred_path))[0] + '.manifest.json'
    entry = {'variant': evaluation.PRIMARY_VARIANT, 'head_seed': 42, 'path': rel,
             'sha256': evaluation.file_sha256(str(pred_path)),
             'sidecar_path': os.path.relpath(sidecar, str(out)),
             'sidecar_sha256': evaluation.file_sha256(sidecar), 'head_checkpoint_sha256': 'h'}
    evaluation.write_sealed_manifest(str(out), [entry], test['glaucoma'], test_manifest, identity)
    results = {'schema': 'cr_probe_v1', 'status': 'complete', 'sealed': True,
               'identity': identity, 'cache_provenance': prov, 'finished': now,
               'config': {'data': {'data_dir': str(cohort['data_dir'])}},
               'variants': {evaluation.PRIMARY_VARIANT: {'per_seed': [
                   {'head_seed': 42, 'best_val_auc': 0.8, 'best_epoch': 1,
                    'test_predictions': rel, 'test_predictions_sha256': entry['sha256']}]}}}
    (out / 'results.json').write_text(json.dumps(results))
    return str(out)


@pytest.fixture(scope='module')
def md_runs(cohort):
    return {
        'orig_random': make_md_run(cohort, 'orig_random', 'random', None, 'anchor', 1.0,
                                   legacy=True),
        'r_1234': make_md_run(cohort, 'r_1234', 'random', 1234, 'new', 1.0),
        'c_1234': make_md_run(cohort, 'c_1234', 'centroid', 1234, 'new', 0.3),
        'e_1234': make_md_run(cohort, 'e_1234', 'envelope', 1234, 'new', 3.0),
    }


def test_validation_mode_never_reads_test(cohort, md_runs, tmp_path):
    # Test caches removed and Test metadata rows garbled: validation mode must not care.
    moved = []
    for run_dir in md_runs.values():
        path = os.path.join(run_dir, 'feature_cache', 'Test_v2_x.pt')
        os.replace(path, path + '.hidden')
        moved.append(path)
    try:
        res = md_reg.cmd_md(list(md_runs.values()), 'validation', metadata=cohort['garbled'],
                            n_boot=200, out=str(tmp_path / 'md_val.json'))
    finally:
        for path in moved:
            os.replace(path + '.hidden', path)
    assert res['split'] == 'Validation' and res['before_freeze'] is None
    c = res['runs']['c_1234']['subsets']
    for subset in ('glaucoma', 'all'):
        assert 'test' not in c[subset]
        val = c[subset]['validation']
        assert val['ridge']['mae'] < val['baseline_train_mean']['mae']
        assert c[subset]['alpha'] in md_reg.ALPHAS
    assert c['glaucoma']['n_train'] == N['Training'] // 2
    # V2 P1-B3: the label-only legacy anchor is excluded from per-case MD, not caveated.
    assert 'orig_random' in res['excluded_runs'] and 'orig_random' not in res['runs']
    kinds = {(x['kind'], x['subset']) for x in res['contrasts']}
    assert kinds == {('centroid-random', 'glaucoma'), ('centroid-random', 'all'),
                     ('envelope-random', 'glaucoma'), ('envelope-random', 'all')}
    good = [x for x in res['contrasts'] if x['kind'] == 'centroid-random'
            and x['subset'] == 'glaucoma'][0]['metrics']
    assert good['mae']['delta'] < 0 and good['mae']['ci95'][0] <= good['mae']['delta'] \
        <= good['mae']['ci95'][1]
    assert json.load(open(str(tmp_path / 'md_val.json')))['split'] == 'Validation'
    with pytest.raises(ValueError):  # the garbled Test rows really are unparseable
        md_reg.load_metadata(cohort['garbled'], cohort['cases']['Test']['names'][:1], {'test'})
    with pytest.raises(cr_stats.IntegrityError, match='use=test'):  # refused before parsing MD
        md_reg.load_metadata(cohort['garbled'], cohort['cases']['Test']['names'][:1],
                             {'training', 'validation'})


def test_test_mode_uses_the_unseal_pathway(cohort, md_runs, tmp_path):
    runs = list(md_runs.values())
    with pytest.raises(cr_stats.IntegrityError, match='freeze'):
        md_reg.cmd_md(runs, 'test', metadata=cohort['meta'], n_boot=50)
    with pytest.raises(cr_stats.IntegrityError, match='unsealed'):
        md_reg.cmd_md(runs, 'test', metadata=cohort['meta'], n_boot=50, allow_before_freeze=True)
    for run_dir in runs:
        cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    res = md_reg.cmd_md(runs, 'test', metadata=cohort['meta'], n_boot=300,
                        allow_before_freeze=True)
    assert res['split'] == 'Test' and res['before_freeze'] is True
    glaucoma = res['runs']['c_1234']['subsets']['glaucoma']
    assert glaucoma['test']['ridge']['n'] == N['Test'] // 2
    assert glaucoma['test']['ridge']['spearman'] > 0.5
    contrast = [x for x in res['contrasts'] if x['kind'] == 'envelope-random'
                and x['subset'] == 'glaucoma'][0]
    assert contrast['primary'] and contrast['metrics']['mae']['delta'] > 0  # noisier encoder
    # A tampered seal blocks test mode exactly as it blocks test AUC.
    path = os.path.join(md_runs['r_1234'], 'sealed_manifest.json')
    original = open(path).read()
    try:
        open(path, 'w').write(original.replace('"metrics_computed": false',
                                               '"metrics_computed": true'))
        with pytest.raises(cr_stats.IntegrityError, match='hash mismatch'):
            md_reg.cmd_md(runs, 'test', metadata=cohort['meta'], n_boot=20,
                          allow_before_freeze=True)
    finally:
        open(path, 'w').write(original)


def test_ridge_path_matches_closed_form():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 6)) * [1, 2, 3, 4, 5, 6], rng.normal(size=50)
    ridge = md_reg.RidgePath(x, y)
    xs = (x - x.mean(0)) / x.std(0)
    for alpha in (0.1, 10.0):
        w = np.linalg.solve(xs.T @ xs + alpha * np.eye(6), xs.T @ (y - y.mean()))
        np.testing.assert_allclose(ridge.predict(x, alpha), xs @ w + y.mean(), atol=1e-10)


def test_metadata_join_and_paired_bootstrap_identity(cohort, md_runs, tmp_path):
    res = cr_stats.load_run(md_runs['r_1234'])
    ident = cr_stats._identity(res)
    vol, labels, names = md_reg.load_split(res, 'Validation', ident)
    meta = md_reg.load_metadata(cohort['meta'], names, {'validation'})
    labels = labels.copy()
    labels[0] = 1 - labels[0]
    with pytest.raises(cr_stats.IntegrityError, match='disagree'):
        md_reg._attach_md(vol, labels, names, meta, 'Validation', 'x')
    original_labels = md_reg.load_split(res, 'Validation', ident)[1]
    with pytest.raises(cr_stats.IntegrityError, match='use=training'):
        md_reg._attach_md(vol, original_labels, names, meta, 'Training', 'x')
    y = np.linspace(-10, 2, 40)
    case = {'y': y, 'pred': y + np.sin(y), 'names': list(range(40)),
            'strata': np.arange(40) % 2 == 0}
    same = md_reg.paired_bootstrap(case, dict(case), 100, 1, stratified=True)
    for metric in ('mae', 'rmse', 'spearman'):
        assert same[metric]['delta'] == 0 and same[metric]['ci95'] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# V2 re-check (section 6) regressions
# ---------------------------------------------------------------------------

def _clone_run(src, dst):
    shutil.copytree(src, str(dst))
    return str(dst)


def _rewrite_results(run, edit):
    path = os.path.join(run, 'results.json')
    res = json.load(open(path))
    edit(res)
    json.dump(res, open(path, 'w'))


def test_v2_wrong_split_mapping_never_reads_test_md(cohort, md_runs, tmp_path, monkeypatch):
    calls = []
    original = md_reg.load_metadata
    monkeypatch.setattr(md_reg, 'load_metadata',
                        lambda path, names, allowed: calls.append(list(names)) or
                        original(path, names, allowed))
    # (a) The Test cache record placed under the Validation key.
    run = _clone_run(md_runs['c_1234'], tmp_path / 'swap')
    _rewrite_results(run, lambda r: r['cache_provenance'].__setitem__(
        'Validation', r['cache_provenance']['Test']))
    with pytest.raises(cr_stats.IntegrityError, match='not a Validation cache'):
        md_reg.cmd_md([run], 'validation', metadata=cohort['meta'], n_boot=10)
    # (b) Test cache bytes renamed as a Validation cache: refused on the manifest split.
    run = _clone_run(md_runs['c_1234'], tmp_path / 'renamed')
    src = json.load(open(os.path.join(run, 'results.json')))['cache_provenance']['Test']['primary']
    dst = os.path.join(run, 'feature_cache', 'Validation_v2_y.pt')
    shutil.copy2(src['path'], dst)
    _rewrite_results(run, lambda r: r['cache_provenance'].__setitem__(
        'Validation', {'primary': dict(src, path=dst)}))
    with pytest.raises(cr_stats.IntegrityError, match="holds split 'Test'"):
        md_reg.cmd_md([run], 'validation', metadata=cohort['meta'], n_boot=10)
    test_names = set(cohort['cases']['Test']['names'])
    assert not any(test_names & set(names) for names in calls)  # zero Test MD reads


def test_v2_receipt_checked_before_any_sealed_array(cohort, tmp_path, monkeypatch):
    run = make_md_run(cohort, 'r_1234_fresh', 'random', 1234, 'new', 1.0, root=tmp_path)
    opened = []
    monkeypatch.setattr(cr_stats, 'load_sealed', lambda run_dir: opened.append(run_dir))
    with pytest.raises(cr_stats.IntegrityError, match='not been unsealed'):
        md_reg.cmd_md([run], 'test', metadata=cohort['meta'], n_boot=10,
                      allow_before_freeze=True)
    assert opened == []


def test_v2_md_identity_is_bound_to_the_seal(cohort, md_runs, tmp_path):
    for field, value in (('arm', 'centroid'), ('run_uuid', 'relabelled'),
                         ('checkpoint_sha256', 'f' * 64)):
        run = _clone_run(md_runs['r_1234'], tmp_path / field)
        _rewrite_results(run, lambda r: r['identity'].__setitem__(field, value))
        with pytest.raises(cr_stats.IntegrityError, match='contradicts the seal'):
            md_reg.cmd_md([run], 'validation', metadata=cohort['meta'], n_boot=10)


def test_v2_md_refuses_changed_cache_bytes_and_skips_legacy(cohort, md_runs, tmp_path,
                                                           monkeypatch):
    run = _clone_run(md_runs['c_1234'], tmp_path / 'bytes')
    src = json.load(open(os.path.join(run, 'results.json')))['cache_provenance']['Training']
    data = torch.load(src['primary']['path'], weights_only=False)
    order = torch.arange(len(data['labels']))
    for value in (0, 1):  # reverse rows within each diagnosis class (labels unchanged)
        idx = torch.nonzero(data['labels'] == value).flatten()
        order[idx] = idx.flip(0)
    data['features'] = data['features'][order]
    dst = os.path.join(run, 'feature_cache', 'Training_v2_x.pt')
    torch.save(data, dst)
    _rewrite_results(run, lambda r: r['cache_provenance']['Training']['primary'].__setitem__(
        'path', dst))
    with pytest.raises(cr_stats.IntegrityError, match='differ from the recorded hash'):
        md_reg.cmd_md([run], 'validation', metadata=cohort['meta'], n_boot=10)
    loaded = []
    original = md_reg.load_split
    monkeypatch.setattr(md_reg, 'load_split', lambda res, split, ident:
                        loaded.append(res['_run_dir']) or original(res, split, ident))
    res = md_reg.cmd_md([md_runs['orig_random'], md_runs['r_1234']], 'validation',
                        metadata=cohort['meta'], n_boot=10)
    assert os.path.realpath(md_runs['orig_random']) not in loaded
    assert 'orig_random' in res['excluded_runs'] and 'r_1234' in res['runs']
