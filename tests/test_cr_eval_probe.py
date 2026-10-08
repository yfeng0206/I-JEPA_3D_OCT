"""cr_seed_v1 evaluator: cache-independent head seeds (E10), sealed test (E18),
patch-max pooling variants (E16) and verified legacy caches. CPU only, tiny data."""
import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch

from src import eval_downstream as evaluation
from src.helper import _VIT_CONFIGS
from src.datasets.oct_volumes import OCTVolumeDataset
from src.models.vision_transformer import VisionTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'autopilot'))
import cr_stats  # noqa: E402

SPLITS = {'Training': 32, 'Validation': 16, 'Test': 16}
EMBED = _VIT_CONFIGS['vit_tiny']['embed_dim']


class Tiny(object):
    pass


def _sha(path):
    return evaluation.file_sha256(str(path))


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp('cr_eval')
    rng = np.random.default_rng(0)
    data = root / 'data'
    for split, n in SPLITS.items():
        (data / split).mkdir(parents=True)
        for i in range(n):
            label = i % 2
            vol = rng.integers(0, 200, size=(200, 8, 8)).astype(np.int64)
            vol[:, :4, :] += 40 * label
            np.savez(data / split / ('case%03d.npz' % i),
                     oct_bscans=np.clip(vol, 0, 255).astype(np.uint8), glaucoma=np.array(label))
    torch.manual_seed(0)
    cfg = _VIT_CONFIGS['vit_tiny']
    encoder = VisionTransformer(img_size=32, patch_size=16, embed_dim=cfg['embed_dim'],
                                depth=cfg['depth'], num_heads=cfg['num_heads'])
    ckpt = root / 'ckpt' / 'jepa_tiny-ep50.pth.tar'
    ckpt.parent.mkdir()
    torch.save({'target_encoder': encoder.state_dict(), 'epoch': 50}, ckpt)
    t = Tiny()
    t.root, t.data, t.ckpt, t.encoder = root, data, ckpt, encoder
    return t


def make_config(tiny, out, **probe):
    config = {
        'mode': 'patch',
        'data': {'data_dir': str(tiny.data), 'num_slices': 4, 'slice_size': 32,
                 'batch_size': 8, 'num_workers': 0, 'encode_chunk_size': 2,
                 'use_amp': False, 'legacy_cache_policy': 'reject'},
        'model': {'encoder_checkpoint': str(tiny.ckpt), 'encoder_name': 'vit_tiny',
                  'patch_size': 16, 'crop_size': 32, 'freeze_encoder': True,
                  'probe_type': 'mean_pool', 'head_type': 'linear'},
        'training': {'lr_probe': 4e-3, 'lr_head': 4e-3, 'weight_decay': 0.05, 'epochs': 4,
                     'patience': 3, 'warmup_epochs': 1, 'seed': 42},
        'probe': {'head_seeds': [42, 43], 'seal_test': False, 'extra_pooling': None},
        'logging': {'output_dir': str(out)},
    }
    config['probe'].update(probe)
    return config


def run(config):
    evaluation.set_amp(False)
    return evaluation.run_patch_downstream(copy.deepcopy(config), torch.device('cpu'))


def copy_caches(src, dst, splits=('Training', 'Validation', 'Test'), kinds=('primary',)):
    os.makedirs(os.path.join(str(dst), 'feature_cache'), exist_ok=True)
    prov = json.load(open(os.path.join(str(src), 'cache_provenance.json')))
    for split in splits:
        for kind in kinds:
            path = prov[split][kind]['path']
            shutil.copy2(path, os.path.join(str(dst), 'feature_cache', os.path.basename(path)))


def artifacts(out, variant, seed):
    seed_dir = os.path.join(str(out), variant, 'seed%d' % seed)
    res = json.load(open(os.path.join(seed_dir, 'results.json')))
    head = torch.load(os.path.join(seed_dir, 'best_model.pt'), map_location='cpu')
    with np.load(os.path.join(seed_dir, 'val_predictions.npz')) as z:
        val = z['probs'].copy()
    with np.load(os.path.join(str(out), res['test_predictions'])) as z:
        test = z['probs'].copy()
    return res, head, val, test


def assert_same_fit(out_a, out_b, variant, seed):
    ra, ha, va, ta = artifacts(out_a, variant, seed)
    rb, hb, vb, tb = artifacts(out_b, variant, seed)
    for key in ('init_head_sha256', 'train_order_sha256', 'best_epoch', 'best_val_auc',
                'epochs_run', 'head_params'):
        assert ra[key] == rb[key], key
    assert ha['epoch'] == hb['epoch']
    for name in ha['head']:
        assert torch.equal(ha['head'][name], hb['head'][name]), name
    assert np.array_equal(va, vb) and np.array_equal(ta, tb)


@pytest.fixture(scope='module')
def cold_run(tiny):
    out = tiny.root / 'runs' / 'cold'
    return out, run(make_config(tiny, out))


@pytest.fixture(scope='module')
def single_run(tiny):
    out = tiny.root / 'runs' / 'single'
    return out, run(make_config(tiny, out, head_seeds=[42]))


@pytest.fixture(scope='module')
def sealed_run(tiny, cold_run):
    out = tiny.root / 'runs' / 'sealed'
    copy_caches(cold_run[0], out)
    return out, run(make_config(tiny, out, seal_test=True))


# ---------------------------------------------------------------------------
# E10: cache-independent head seeds
# ---------------------------------------------------------------------------

def test_cold_and_warm_caches_give_identical_heads(tiny, cold_run, tmp_path):
    cold, res_cold = cold_run
    warm, partial = tmp_path / 'warm', tmp_path / 'partial'
    copy_caches(cold, warm)
    res_warm = run(make_config(tiny, warm))
    copy_caches(cold, partial, splits=('Training',))
    res_partial = run(make_config(tiny, partial))
    states = lambda res: {s: res['cache_provenance'][s]['primary']['state']  # noqa: E731
                          for s in SPLITS}
    assert states(res_cold) == {s: 'cold' for s in SPLITS}
    assert states(res_warm) == {s: 'warm' for s in SPLITS}
    assert states(res_partial) == {'Training': 'warm', 'Validation': 'cold', 'Test': 'cold'}
    for split in SPLITS:
        rec = res_cold['cache_provenance'][split]['primary']
        assert rec['identity_sha256'] in os.path.basename(rec['path'])
        assert rec['file_sha256'] == _sha(rec['path'])
        assert rec['identity_sha256'] == res_warm['cache_provenance'][split]['primary'][
            'identity_sha256']
    for seed in (42, 43):
        assert_same_fit(cold, warm, evaluation.PRIMARY_VARIANT, seed)
        assert_same_fit(cold, partial, evaluation.PRIMARY_VARIANT, seed)
    assert res_cold['best_val_auc'] == res_warm['best_val_auc'] == res_partial['best_val_auc']
    assert os.path.exists(os.path.join(str(cold), 'cache_provenance.json'))


def test_head_seeds_differ_and_are_aggregated(cold_run):
    out, res = cold_run
    r42, h42, v42, _ = artifacts(out, evaluation.PRIMARY_VARIANT, 42)
    r43, h43, v43, _ = artifacts(out, evaluation.PRIMARY_VARIANT, 43)
    assert r42['init_head_sha256'] != r43['init_head_sha256']
    assert r42['train_order_sha256'] != r43['train_order_sha256']
    assert not np.array_equal(v42, v43)
    summary = res['variants'][evaluation.PRIMARY_VARIANT]
    vals = [r42['best_val_auc'], r43['best_val_auc']]
    assert res['head_seeds'] == [42, 43] and summary['n_head_seeds'] == 2
    assert summary['val_auc_mean'] == pytest.approx(np.mean(vals), abs=1e-15)
    assert summary['val_auc_sd'] == pytest.approx(np.std(vals, ddof=1), abs=1e-15)
    assert res['best_val_auc'] == summary['val_auc_mean']
    assert res['status'] == 'complete' and res['schema'] == 'cr_probe_v1'
    assert res['identity']['checkpoint_sha256'] == _sha(
        res['config']['model']['encoder_checkpoint'])
    assert res['identity']['epoch'] == 50


def test_seed_reset_is_what_makes_the_head_cache_independent():
    # The pre-fix coupling: consuming DataLoader iterators (cold extraction)
    # before head construction changed the head; reseeding removes it.
    torch.manual_seed(42)
    for _ in range(3):
        iter(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(2))))
    coupled = evaluation.LinearHead(8).state_dict()
    evaluation.seed_everything(42)
    fixed_a = evaluation.LinearHead(8).state_dict()
    torch.manual_seed(42)
    evaluation.seed_everything(42)
    fixed_b = evaluation.LinearHead(8).state_dict()
    assert not torch.equal(coupled['linear.weight'], fixed_a['linear.weight'])
    assert all(torch.equal(fixed_a[k], fixed_b[k]) for k in fixed_a)


def test_completed_output_dir_is_refused(tiny, cold_run):
    with pytest.raises(RuntimeError, match='completed probe'):
        run(make_config(tiny, cold_run[0]))


# ---------------------------------------------------------------------------
# E18: sealed test predictions
# ---------------------------------------------------------------------------

def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            for item in _walk(value):
                yield item
    elif isinstance(node, list):
        for value in node:
            for item in _walk(value):
                yield item


def test_sealed_mode_writes_no_test_metrics_and_hashes_verify(tiny, cold_run, tmp_path,
                                                              capsys):
    out = tmp_path / 'sealed_capture'
    copy_caches(cold_run[0], out)
    capsys.readouterr()
    res = run(make_config(tiny, out, seal_test=True))
    printed = capsys.readouterr().out
    assert 'TEST AUC' not in printed and 'Sensitivity' not in printed
    assert 'test_auc     =' not in printed and 'SEALED' in printed
    assert 'Val AUC' in printed
    metric_keys = {'test_auc', 'test_loss', 'sensitivity', 'specificity', 'test_auc_mean',
                   'test_auc_sd'}
    with open(os.path.join(str(out), 'results.json')) as stream:
        on_disk = json.load(stream)
    for key, value in list(_walk(on_disk)) + list(_walk(res)):
        if key in metric_keys:
            assert value is None, key
    # Campaign runner contract (i2): stable keys, and no numeric value under a test-AUC-like key.
    leak = re.compile(r'test.*auc|auc.*test', re.I)
    assert [(k, v) for k, v in _walk(on_disk) if leak.search(str(k))
            and isinstance(v, (int, float)) and not isinstance(v, bool)] == []
    assert on_disk['schema'] == 'cr_probe_v1' and on_disk['status'] == 'complete'
    assert on_disk['sealed'] is True and isinstance(on_disk['best_val_auc'], float)
    assert on_disk['identity']['checkpoint_sha256'] == _sha(tiny.ckpt)
    for seed in (42, 43):
        seed_dir = os.path.join(str(out), evaluation.PRIMARY_VARIANT, 'seed%d' % seed)
        assert not os.path.exists(os.path.join(seed_dir, 'test_predictions.npz'))
        assert not os.path.exists(os.path.join(seed_dir, 'diagnostic_plots.png'))
        assert os.path.exists(os.path.join(seed_dir, 'test_predictions_sealed_seed%d.npz' % seed))
        assert os.path.exists(os.path.join(seed_dir, 'val_predictions.npz'))
    with open(os.path.join(str(out), 'sealed_manifest.json')) as stream:
        manifest = json.load(stream)
    assert manifest['metrics_computed'] is False and len(manifest['files']) == 2
    for entry in manifest['files']:
        assert _sha(os.path.join(str(out), entry['path'])) == entry['sha256']
    sidecar = open(os.path.join(str(out), 'sealed_manifest.json.sha256')).read().split()[0]
    assert sidecar == _sha(os.path.join(str(out), 'sealed_manifest.json'))
    labels = np.array([i % 2 for i in range(SPLITS['Test'])], dtype=np.int8)
    assert manifest['test_label_identity']['labels_int8_sha256'] == \
        hashlib.sha256(labels.tobytes()).hexdigest()
    cr_stats.load_sealed(str(out))
    with pytest.raises(RuntimeError, match='sealed_manifest|sealed test'):
        run(make_config(tiny, out, seal_test=True))


def test_unseal_recomputes_auc_equal_to_unsealed_path(cold_run, sealed_run):
    cold, res_cold = cold_run
    sealed, res_sealed = sealed_run
    with pytest.raises(cr_stats.IntegrityError, match='freeze'):
        cr_stats.cmd_unseal(str(sealed))
    unsealed = cr_stats.cmd_unseal(str(sealed), allow_before_freeze=True)
    assert unsealed['unsealed_before_freeze'] is True
    per_seed = {r['head_seed']: r for r in
                unsealed['variants'][evaluation.PRIMARY_VARIANT]['per_seed']}
    for rec in res_cold['variants'][evaluation.PRIMARY_VARIANT]['per_seed']:
        assert per_seed[rec['head_seed']]['test_auc'] == rec['test_auc']
        assert per_seed[rec['head_seed']]['sensitivity'] == rec['sensitivity']
        assert per_seed[rec['head_seed']]['specificity'] == rec['specificity']
    assert unsealed['variants'][evaluation.PRIMARY_VARIANT]['test_auc_mean'] == \
        pytest.approx(res_cold['test_auc'], abs=1e-15)
    for seed in (42, 43):
        assert_same_fit(cold, sealed, evaluation.PRIMARY_VARIANT, seed)
    assert os.path.exists(os.path.join(str(sealed), 'unseal_log.jsonl'))


def test_tampered_sealed_files_are_rejected(sealed_run, tmp_path):
    for target in ('npz', 'manifest'):
        out = tmp_path / target
        shutil.copytree(str(sealed_run[0]), str(out))
        if target == 'npz':
            path = os.path.join(str(out), evaluation.PRIMARY_VARIANT, 'seed42',
                                'test_predictions_sealed_seed42.npz')
            with np.load(path) as z:
                arrays = {k: z[k].copy() for k in z.files}
            arrays['probs'][0] = 1.0 - arrays['probs'][0]
            np.savez(path, **arrays)
        else:
            path = os.path.join(str(out), 'sealed_manifest.json')
            text = open(path).read().replace('"metrics_computed": false',
                                             '"metrics_computed": true')
            open(path, 'w').write(text)
        with pytest.raises(cr_stats.IntegrityError, match='hash mismatch'):
            cr_stats.cmd_unseal(str(out), allow_before_freeze=True)


# ---------------------------------------------------------------------------
# E16: patch-max in the same encoder pass
# ---------------------------------------------------------------------------

def test_extra_pooling_off_is_bit_identical(tiny, single_run, tmp_path):
    off, res_off = single_run
    on = tmp_path / 'on'
    res_on = run(make_config(tiny, on, head_seeds=[42], extra_pooling='patch_max'))
    for split in SPLITS:
        a = res_off['cache_provenance'][split]['primary']
        b = res_on['cache_provenance'][split]['primary']
        assert a['state'] == b['state'] == 'cold'
        assert os.path.basename(a['path']) == os.path.basename(b['path'])
        da, db = torch.load(a['path'], weights_only=False), torch.load(b['path'], weights_only=False)
        assert torch.equal(da['features'], db['features'])
        assert da['source_manifest'] == db['source_manifest']
        assert 'patch_max' not in res_off['cache_provenance'][split]
        assert res_on['cache_provenance'][split]['patch_max']['state'] == 'cold'
    assert_same_fit(off, on, evaluation.PRIMARY_VARIANT, 42)
    primary_on = res_on['variants'][evaluation.PRIMARY_VARIANT]
    assert primary_on['val_auc_mean'] == res_off['best_val_auc']
    assert primary_on['test_auc_mean'] == res_off['test_auc']
    assert set(res_on['variants']) == set(evaluation.POOLING_VARIANTS)
    assert res_on['skipped_variants'] == {}
    assert res_on['variants']['patchmean_slicemeanmax']['head_params'] == \
        2 * (2 * EMBED) + 2 * EMBED + 1
    for variant in ('patchmax_slicemean', 'patchmean_slicemax'):
        assert res_on['variants'][variant]['head_params'] == 2 * EMBED + EMBED + 1
    # Historical single-seed layout is mirrored only when nothing new is requested.
    assert os.path.exists(os.path.join(str(off), 'test_predictions.npz'))
    assert not os.path.exists(os.path.join(str(on), 'test_predictions.npz'))


def test_patch_max_cache_matches_encoder_output(tiny, tmp_path):
    out = tmp_path / 'max'
    res = run(make_config(tiny, out, head_seeds=[42], extra_pooling='patch_max'))
    rec = res['cache_provenance']['Training']['patch_max']
    data = torch.load(rec['path'], weights_only=False)
    assert tuple(data['features'].shape) == (SPLITS['Training'], 4, EMBED)
    assert data['source_manifest']['feature_reduction'] == 'patch_token_max'
    mean_cache = torch.load(res['cache_provenance']['Training']['primary']['path'],
                            weights_only=False)
    dataset = OCTVolumeDataset(os.path.join(str(tiny.data), 'Training'), num_slices=4,
                               slice_size=32, return_label=True)
    tiny.encoder.eval()
    with torch.no_grad():
        for idx in (0, 7):
            tokens = tiny.encoder(evaluation.imagenet_normalize(dataset[idx][0]))
            torch.testing.assert_close(data['features'][idx], tokens.amax(dim=1),
                                       rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(mean_cache['features'][idx], tokens.mean(dim=1),
                                       rtol=1e-5, atol=1e-5)


def test_warm_primary_with_cold_patch_max_keeps_primary(tiny, single_run, tmp_path):
    off, res_off = single_run
    mix = tmp_path / 'mix'
    copy_caches(off, mix)
    res = run(make_config(tiny, mix, head_seeds=[42], extra_pooling='patch_max'))
    for split in SPLITS:
        prov = res['cache_provenance'][split]
        assert prov['primary']['state'] == 'warm' and prov['patch_max']['state'] == 'cold'
        assert prov['primary']['fresh_recompute_max_abs_diff'] == 0.0
    assert_same_fit(off, mix, evaluation.PRIMARY_VARIANT, 42)


# ---------------------------------------------------------------------------
# Legacy caches for original anchors
# ---------------------------------------------------------------------------

def make_legacy_dir(tiny, single_run, root, named_ckpt, guard_sha=None, name='meanpool_tiny_ep50_fp32'):
    source, res = single_run
    run_dir = root / 'runs' / ('frozen_' + name)
    cache_dir = run_dir / 'feature_cache'
    cache_dir.mkdir(parents=True)
    with open(str(tiny.ckpt), 'rb') as stream:
        key = hashlib.sha256(stream.read(1 << 20)).hexdigest()[:12]
    log = ['GPU: test', 'Loading encoder from %s ...' % named_ckpt,
           '  Loaded target_encoder weights (epoch 50)', '',
           '--- Pre-computing features with frozen encoder ---',
           '  precision: fp32 (AMP disabled)']
    for split in SPLITS:
        data = torch.load(res['cache_provenance'][split]['primary']['path'], weights_only=False)
        path = cache_dir / ('%s_s4_r32_fp32_%s.pt' % (split, key))
        torch.save({'features': data['features'], 'labels': data['labels']}, path)
        log.append('  Cached to %s (1.0 MB)' % path)
    for name_ in ('best_model.pt', 'val_predictions.npz', 'test_predictions.npz', 'train_log.csv'):
        shutil.copy2(os.path.join(str(source), name_), str(run_dir / name_))
    legacy_results = {'best_val_auc': res['best_val_auc'], 'test_auc': res['test_auc'],
                      'best_epoch': res['best_epoch'], 'config': copy.deepcopy(res['config'])}
    legacy_results['config']['model']['encoder_checkpoint'] = str(named_ckpt)
    legacy_results['config']['logging'] = {'output_dir': str(run_dir)}
    (run_dir / 'results.json').write_text(json.dumps(legacy_results))
    (run_dir / 'eval.log').write_text('\n'.join(log) + '\n')
    guard_dir = root / 'autopilot_out' / 'probe_guards'
    guard_dir.mkdir(parents=True, exist_ok=True)
    sha = guard_sha or _sha(tiny.ckpt)
    # The guard records completion after every cache was written (as in the real run).
    finished = (datetime.datetime.now().astimezone()
                + datetime.timedelta(seconds=2)).strftime('%Y-%m-%dT%H:%M:%S%z')
    (guard_dir / ('guard_%s.json' % name)).write_text(json.dumps({
        'name': name, 'encoder_checkpoint': str(named_ckpt), 'sha256_before': sha,
        'sha256_after': sha, 'encoder_unchanged': True, 'freeze_encoder': True,
        'use_amp': False, 'return_code': 0, 'output_dir': str(run_dir), 'valid': True,
        'finished': finished}))
    return run_dir


def pin_legacy(legacy, date='2026-10-08'):
    """Trust-on-first-use pins of the legacy caches and evidence files as they are now."""
    paths = list((legacy / 'feature_cache').iterdir())
    paths += [legacy / name for name in evaluation.LEGACY_EVIDENCE_FILES]
    paths.append(evaluation.default_guard_json(str(legacy)))
    return {'schema': evaluation.LEGACY_PIN_SCHEMA, 'pinned': date,
            'files': {str(p): _sha(p) for p in paths if os.path.exists(str(p))}}


def legacy_config(tiny, out, legacy_dir, **probe):
    config = make_config(tiny, out, **dict({'head_seeds': [42]}, **probe))
    config['data']['legacy_cache_policy'] = 'allow_verified'
    # Tiny fits move within an epoch, so the logged train AUC/loss drift more than at full scale.
    config['legacy_cache'] = {'run_dir': str(legacy_dir),
                              'expected_checkpoint_sha256': _sha(tiny.ckpt),
                              'expected_cache_sha256': pin_legacy(legacy_dir),
                              'train_replay_auc_atol': 0.1, 'train_replay_loss_atol': 0.2}
    return config


def test_verified_legacy_cache_with_checkpoint_on_disk(tiny, single_run, tmp_path):
    legacy = make_legacy_dir(tiny, single_run, tmp_path, tiny.ckpt)
    out = tmp_path / 'anchor'
    res = run(legacy_config(tiny, out, legacy))
    for split in SPLITS:
        prov = res['cache_provenance'][split]['primary']
        assert prov['state'] == 'legacy_verified'
        checks = {c['name']: c for c in prov['evidence']['checks']}
        assert all(c['ok'] for c in checks.values())
        assert checks['filename_key_matches_checkpoint_prefix']['required'] is True
        assert prov['evidence']['checkpoint_sha256'] == _sha(tiny.ckpt)
        assert prov['file_sha256'] == _sha(prov['path'])
    replay = {c['name']: c for c in res['cache_provenance']['Validation']['primary'][
        'evidence']['checks']}['legacy_head_replay']
    assert replay['required'] and replay['detail']['max_abs_prob_diff'] <= 1e-6
    assert_same_fit(single_run[0], out, evaluation.PRIMARY_VARIANT, 42)


def test_verified_legacy_cache_without_weights_skips_patch_max(tiny, single_run, tmp_path):
    missing = tmp_path / 'gone' / 'jepa_patch-ep050.pth.tar'
    legacy = make_legacy_dir(tiny, single_run, tmp_path, missing)
    out = tmp_path / 'random_anchor'
    config = legacy_config(tiny, out, legacy, extra_pooling='patch_max', seal_test=True)
    config['model']['encoder_checkpoint'] = str(missing)
    config['identity'] = {'arm': 'random', 'role': 'anchor', 'run_uuid': 'orig-random-ep50'}
    res = run(config)
    assert res['identity']['checkpoint_sha256'] == _sha(tiny.ckpt)
    assert 'guard sidecar' in res['identity']['checkpoint_sha256_source']
    assert res['identity']['epoch'] == 50
    assert set(res['skipped_variants']) == {'patchmax_slicemean'}
    assert set(res['variants']) == set(evaluation.POOLING_VARIANTS) - {'patchmax_slicemean'}
    for split in SPLITS:
        prov = res['cache_provenance'][split]
        assert prov['primary']['state'] == 'legacy_verified'
        assert prov['patch_max']['state'] == 'unavailable'
        key_check = {c['name']: c for c in prov['primary']['evidence']['checks']}[
            'filename_key_matches_checkpoint_prefix']
        assert key_check['required'] is False and 'unverifiable' in key_check['detail']['status']
    cr_stats.load_sealed(str(out))


def test_legacy_cache_with_wrong_order_is_rejected(tiny, single_run, tmp_path):
    legacy = make_legacy_dir(tiny, single_run, tmp_path, tiny.ckpt)
    path = [p for p in (legacy / 'feature_cache').iterdir() if p.name.startswith('Validation')][0]
    data = torch.load(path, weights_only=False)
    order = torch.arange(SPLITS['Validation'])
    order[[0, 1]] = order[[1, 0]]  # labels 0 and 1: a real order change
    torch.save({'features': data['features'][order], 'labels': data['labels'][order]}, path)
    with pytest.raises(ValueError, match='label_order_matches_split'):
        run(legacy_config(tiny, tmp_path / 'bad_order', legacy))


def _training_cache(legacy):
    return [p for p in (legacy / 'feature_cache').iterdir() if p.name.startswith('Training')][0]


def _set_mtime(legacy, path, offset_from_run_end):
    """Place a cache mtime relative to the legacy run's recorded completion time."""
    with open(evaluation.default_guard_json(str(legacy))) as stream:
        finished = datetime.datetime.strptime(json.load(stream)['finished'],
                                              '%Y-%m-%dT%H:%M:%S%z').timestamp()
    os.utime(str(path), (finished + offset_from_run_end, finished + offset_from_run_end))


def test_v2_modified_training_caches_are_rejected(tiny, single_run, tmp_path):
    # V2 P1-2 cases: same-label row swap and all-zero features in the Training cache.
    for case in ('swap_same_label', 'all_zero', 'all_zero_backdated'):
        legacy = make_legacy_dir(tiny, single_run, tmp_path / case, tiny.ckpt)
        path = _training_cache(legacy)
        data = torch.load(path, weights_only=False)
        if case == 'swap_same_label':
            order = torch.arange(SPLITS['Training'])
            order[[0, 2]] = order[[2, 0]]  # cases 0 and 2 share label 0
            assert data['labels'][0] == data['labels'][2]
            data = {'features': data['features'][order], 'labels': data['labels'][order]}
        else:
            data = {'features': torch.zeros_like(data['features']), 'labels': data['labels']}
        torch.save(data, path)
        _set_mtime(legacy, path, -0.5 if case == 'all_zero_backdated' else 3600)
        expected = ('legacy_training_head_replay' if case == 'all_zero_backdated'
                    else 'cache_not_modified_after_run')
        with pytest.raises(ValueError, match=expected):
            run(legacy_config(tiny, tmp_path / ('out_' + case), legacy))


def test_v2_label_preserving_training_permutation_limit_is_recorded(tiny, single_run, tmp_path):
    # A same-label permutation that predates the run's completion cannot be detected; it
    # leaves the training (feature, label) multiset unchanged, and the evidence says so.
    legacy = make_legacy_dir(tiny, single_run, tmp_path, tiny.ckpt)
    path = _training_cache(legacy)
    data = torch.load(path, weights_only=False)
    order = torch.arange(SPLITS['Training'])
    order[[0, 2]] = order[[2, 0]]
    torch.save({'features': data['features'][order], 'labels': data['labels'][order]}, path)
    _set_mtime(legacy, path, -0.5)
    res = run(legacy_config(tiny, tmp_path / 'out', legacy))
    prov = res['cache_provenance']
    assert 'label sequence only' in prov['Training']['primary']['evidence']['row_identity']
    for split in ('Validation', 'Test'):
        assert 'verified by old-head replay' in prov[split]['primary']['evidence']['row_identity']


def test_v2_missing_replay_evidence_is_rejected(tiny, single_run, tmp_path):
    for missing in ('val_predictions.npz', 'train_log.csv'):
        legacy = make_legacy_dir(tiny, single_run, tmp_path / missing.split('.')[0], tiny.ckpt)
        os.remove(str(legacy / missing))
        expected = 'legacy_head_replay' if missing.startswith('val') else \
            'legacy_training_head_replay'
        # Missing evidence files now fail the pinned-evidence check first; replay is the
        # second line of defence.
        with pytest.raises(ValueError, match='evidence_files_match_pinned_sha256|' + expected):
            run(legacy_config(tiny, tmp_path / ('out_' + missing.split('.')[0]), legacy))


def test_v2_sealed_run_refuses_stale_output_artifacts(tiny, single_run, tmp_path):
    # Old-format completed output (no status key) and an interrupted unsealed output.
    old = tmp_path / 'old_format'
    old.mkdir()
    (old / 'results.json').write_text(json.dumps({'test_auc': 0.9, 'best_val_auc': 0.8}))
    shutil.copy2(os.path.join(str(single_run[0]), 'test_predictions.npz'),
                 str(old / 'test_predictions.npz'))
    (old / 'eval.log').write_text('Best epoch: 3  |  Val AUC: 0.8  |  TEST AUC: 0.9\n')
    interrupted = tmp_path / 'interrupted'
    copy_caches(single_run[0], interrupted)
    shutil.copytree(os.path.join(str(single_run[0]), evaluation.PRIMARY_VARIANT),
                    str(interrupted / evaluation.PRIMARY_VARIANT))
    for out in (old, interrupted):
        before = sorted(os.listdir(str(out)))
        with pytest.raises(RuntimeError, match='fresh output_dir'):
            run(make_config(tiny, out, head_seeds=[42], seal_test=True))
        assert sorted(os.listdir(str(out))) == before  # nothing deleted or added
    cache_only = tmp_path / 'cache_only'
    copy_caches(single_run[0], cache_only)
    res = run(make_config(tiny, cache_only, head_seeds=[42], seal_test=True))
    assert res['sealed'] and res['cache_provenance']['Training']['primary']['state'] == 'warm'


def test_interrupted_cache_write_is_not_a_warm_cache(tiny, single_run, tmp_path):
    out = tmp_path / 'partial_write'
    copy_caches(single_run[0], out)
    prov = single_run[1]['cache_provenance']['Validation']['primary']
    final_name = os.path.join(str(out), 'feature_cache', os.path.basename(prov['path']))
    os.replace(final_name, final_name + '.partial')  # what an interrupted save leaves behind
    res = run(make_config(tiny, out, head_seeds=[42]))
    assert res['cache_provenance']['Validation']['primary']['state'] == 'cold'
    assert os.path.exists(final_name)


def test_tofu_pins_refuse_changes_after_pinning(tiny, single_run, tmp_path):
    legacy = make_legacy_dir(tiny, single_run, tmp_path, tiny.ckpt)
    config = legacy_config(tiny, tmp_path / 'pinned_ok', legacy)
    res = run(config)
    tofu = 'tofu_pinned_2026-10-08; historical integrity supported by timestamps and head ' \
           'replay, not cryptographic'
    assert res['legacy_cache_integrity'] == tofu
    for split in SPLITS:
        evidence = res['cache_provenance'][split]['primary']['evidence']
        assert evidence['integrity'] == tofu
        assert {c['name']: c['ok'] for c in evidence['checks']}['cache_matches_pinned_sha256']
    # Cache modified after pinning (mtime kept before the run's completion): refused.
    path = _training_cache(legacy)
    data = torch.load(path, weights_only=False)
    torch.save({'features': data['features'] * 1.0001, 'labels': data['labels']}, path)
    _set_mtime(legacy, path, -0.5)
    with pytest.raises(ValueError, match='cache_matches_pinned_sha256'):
        run(dict(config, logging={'output_dir': str(tmp_path / 'pinned_cache_changed')}))
    # Evidence file modified after pinning: refused before any cache is read.
    legacy2 = make_legacy_dir(tiny, single_run, tmp_path / 'b', tiny.ckpt)
    config2 = legacy_config(tiny, tmp_path / 'b_out', legacy2)
    with open(str(legacy2 / 'eval.log'), 'a') as stream:
        stream.write('edited\n')
    with pytest.raises(ValueError, match='evidence_files_match_pinned_sha256'):
        run(config2)
    # allow_verified without pins: refused.
    del config2['legacy_cache']['expected_cache_sha256']
    with pytest.raises(ValueError, match='evidence_files_match_pinned_sha256'):
        run(dict(config2, logging={'output_dir': str(tmp_path / 'unpinned')}))


def test_committed_random_ep50_pins():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, 'configs', 'cr_seed_v1', 'legacy_random_ep50_cache_hashes.json')
    pins = evaluation.load_legacy_pins(path)
    assert pins['integrity'] == ('tofu_pinned_2026-10-08; historical integrity supported by '
                                 'timestamps and head replay, not cryptographic')
    names = sorted(os.path.basename(p) for p in pins['files'])
    assert sum(n.endswith('_s100_r256_fp32_967c4b4373ce.pt') for n in names) == 3
    for name in evaluation.LEGACY_EVIDENCE_FILES + ('guard_meanpool_random_ep50_fp32.json',):
        assert name in names
    for p, sha in pins['files'].items():  # small evidence files only; caches are ~3 GB
        if os.path.exists(p) and 'feature_cache' not in p:
            assert _sha(p) == sha, p


def test_legacy_cache_with_wrong_checkpoint_identity_is_rejected(tiny, single_run, tmp_path):
    legacy = make_legacy_dir(tiny, single_run, tmp_path, tiny.ckpt)
    config = legacy_config(tiny, tmp_path / 'bad_sha', legacy)
    config['legacy_cache']['expected_checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checkpoint_sha256_consistent'):
        run(config)
    other = make_legacy_dir(tiny, single_run, tmp_path / 'b', tiny.ckpt, guard_sha='1' * 64)
    with pytest.raises(ValueError, match='checkpoint_sha256_consistent'):
        run(legacy_config(tiny, tmp_path / 'bad_guard', other))


def test_default_policy_rejects_legacy_cache_and_missing_weights(tiny, single_run, tmp_path):
    out = tmp_path / 'reject'
    (out / 'feature_cache').mkdir(parents=True)
    with open(str(tiny.ckpt), 'rb') as stream:
        key = hashlib.sha256(stream.read(1 << 20)).hexdigest()[:12]
    src = single_run[1]['cache_provenance']['Training']['primary']['path']
    shutil.copy2(src, str(out / 'feature_cache' / ('Training_s4_r32_fp32_%s.pt' % key)))
    with pytest.raises(ValueError, match='Legacy cache lacks verified source identity'):
        run(make_config(tiny, out, head_seeds=[42]))
    config = make_config(tiny, tmp_path / 'no_weights', head_seeds=[42])
    config['model']['encoder_checkpoint'] = str(tmp_path / 'absent.pth.tar')
    with pytest.raises(FileNotFoundError):
        run(config)


# ---------------------------------------------------------------------------
# CLI and protocol config
# ---------------------------------------------------------------------------

def test_cli_overrides_and_cr_config():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, 'configs', 'downstream_cr_seed_v1.yaml')
    import yaml
    config = yaml.safe_load(open(path))
    canonical = yaml.safe_load(open(os.path.join(repo, 'configs',
                                                 'downstream_frozen_meanpool_canonical.yaml')))
    for section in ('data', 'training'):
        assert config[section] == canonical[section]
    model = dict(config['model'], encoder_checkpoint=canonical['model']['encoder_checkpoint'])
    assert model == canonical['model']
    assert config['data']['use_amp'] is False
    assert config['probe'] == {'head_seeds': [42, 43, 44, 45, 46], 'seal_test': True,
                               'extra_pooling': 'patch_max', 'hash_cache_files': True}
    assert config['model']['encoder_checkpoint'] is None and config['logging']['output_dir'] is None
    args = evaluation.build_arg_parser().parse_args([
        '--config', path, '--encoder-checkpoint', 'C:/x.pth.tar', '--output-dir', 'C:/out',
        '--head-seeds', '42,43', '--no-seal-test', '--extra-pooling', 'none', '--arm', 'centroid',
        '--train-seed', '1234', '--run-uuid', 'u1', '--role', 'new',
        '--legacy-cache-policy', 'allow_verified', '--legacy-cache-run-dir', 'C:/old',
        '--expected-checkpoint-sha256', 'ab' * 32])
    merged = evaluation.apply_cli_overrides(config, args)
    assert merged['model']['encoder_checkpoint'] == 'C:/x.pth.tar'
    assert merged['logging']['output_dir'] == 'C:/out'
    assert merged['probe']['head_seeds'] == [42, 43]
    assert merged['probe']['seal_test'] is False and merged['probe']['extra_pooling'] is None
    assert merged['identity'] == {'arm': 'centroid', 'train_seed': 1234, 'run_uuid': 'u1',
                                  'role': 'new'}
    assert merged['legacy_cache'] == {'run_dir': 'C:/old', 'expected_checkpoint_sha256': 'ab' * 32}
    assert merged['data']['legacy_cache_policy'] == 'allow_verified'
    assert config['probe']['seal_test'] is True  # original untouched
    untouched = evaluation.apply_cli_overrides(
        config, evaluation.build_arg_parser().parse_args(['--config', path]))
    assert untouched['probe'] == config['probe'] and 'identity' not in untouched
    with pytest.raises(SystemExit):
        evaluation.build_arg_parser().parse_args(['--config', path, '--arm', 'oracle'])


def test_run_provenance_is_recorded_and_contradictions_refused(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, 'configs', 'downstream_cr_seed_v1.yaml')
    import yaml
    config = yaml.safe_load(open(path))
    manifest = tmp_path / 'run_manifest.json'
    manifest.write_text(json.dumps({'schema': 'jepa_run_manifest_v1', 'run_uuid': 'trainer-u',
                                    'seed': 1234}))
    parser = evaluation.build_arg_parser()
    base = ['--config', path, '--run-provenance', str(manifest), '--arm', 'random']
    ident = evaluation.apply_cli_overrides(config, parser.parse_args(
        base + ['--run-uuid', 'campaign-u', '--train-seed', '1234']))['identity']
    assert ident['run_uuid'] == 'campaign-u' and ident['run_provenance_run_uuid'] == 'trainer-u'
    assert ident['train_seed'] == 1234 and ident['run_provenance_sha256'] == _sha(manifest)
    only = evaluation.apply_cli_overrides(config, parser.parse_args(base))['identity']
    assert only['run_uuid'] == 'trainer-u' and only['train_seed'] == 1234
    with pytest.raises(ValueError, match='contradicts run provenance'):
        evaluation.apply_cli_overrides(config, parser.parse_args(base + ['--train-seed', '5678']))


def test_stop_file_provenance_pins_checkpoint_and_arm(tiny, tmp_path):
    parser = evaluation.build_arg_parser()
    stop = tmp_path / 'stop_epoch_050.json'

    def write_stop(sha):
        stop.write_text(json.dumps({
            'schema': 'jepa_stop_epoch_v1', 'run_uuid': 'a' * 32, 'seed': 1234, 'epoch': 50,
            'write_tag': 'jepa_patch_cr_seed_v1_random_cb_s1234',
            'checkpoints': [{'role': 'last', 'sha256': sha, 'epoch': 50},
                            {'role': 'periodic', 'sha256': sha, 'epoch': 50}]}))

    write_stop(_sha(tiny.ckpt))
    args = ['--config', 'unused.yaml', '--run-provenance', str(stop), '--role', 'new',
            '--run-uuid', 'campaign-1']
    config = evaluation.apply_cli_overrides(
        make_config(tiny, tmp_path / 'ok', head_seeds=[42]), parser.parse_args(args))
    assert config['identity']['arm'] == 'random_cb' and config['identity']['train_seed'] == 1234
    res = run(config)
    assert res['identity']['run_provenance_checkpoint_role'] == 'periodic'
    assert res['identity']['run_provenance_run_uuid'] == 'a' * 32
    with pytest.raises(ValueError, match='contradicts run provenance'):
        evaluation.apply_cli_overrides(make_config(tiny, tmp_path / 'x'),
                                       parser.parse_args(args + ['--arm', 'random']))
    write_stop('0' * 64)
    config = evaluation.apply_cli_overrides(
        make_config(tiny, tmp_path / 'bad', head_seeds=[42]), parser.parse_args(args))
    with pytest.raises(ValueError, match='not listed in the run provenance'):
        run(config)
