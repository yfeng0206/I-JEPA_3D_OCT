"""Camera-ready trainer run control (E1, E6, E8, E19, G0) on the tiny CPU loop.

The fixture mirrors tests/test_training_contracts.py: a synthetic dataset and a
one-block ViT drive the real ``src.train_patch.main`` end to end.
"""

import csv
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src import helper
from src import train_patch as training
from src.helper import file_sha256
from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


class TinyDataset:
    def __init__(self, data_dir, **kwargs):
        self.length = 7 if data_dir.endswith('Training') else 3
        self.file_paths = ['synthetic_%d.npz' % index for index in range(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.rand(3, 16, 16) + np.random.rand() + random.random()


def fake_source(source_hash):
    def collect(code_root, subdir='src'):
        return {'code_root': code_root, 'subdir': subdir, 'listing': 'stub',
                'source_hash': source_hash, 'files': {'src/stub.py': source_hash}}
    return collect


def install_tiny(set_attr, stub_source=None):
    set_attr(torch.cuda, 'is_available', lambda: False)
    set_attr(training, 'init_distributed', lambda: (1, 0))
    set_attr(training, 'OCTSliceDataset', TinyDataset)
    set_attr(training, 'init_patch_model', lambda **kwargs: (
        VisionTransformer(img_size=16, patch_size=4, embed_dim=16, depth=1, num_heads=2),
        VisionTransformerPredictor(16, 16, 16, depth=1, num_heads=2)))
    set_attr(training, 'upload_to_blob', lambda *args, **kwargs: None)
    set_attr(training, '_nvidia_smi_query', lambda: {'error': 'stubbed in tests'})
    if stub_source is not None:
        set_attr(training, 'collect_source_identity', fake_source(stub_source))


def make_config(tmp_path, folder, epochs=4, save_every=5, patience=10, seed=19):
    root = tmp_path / 'dataset'
    (root / 'Training').mkdir(parents=True, exist_ok=True)
    (root / 'Validation').mkdir(exist_ok=True)
    return {
        'data': {'data_dir': str(root), 'crop_size': 16, 'batch_size': 1,
                 'num_workers': 0, 'val_num_workers': 0, 'num_slices': 1},
        'mask': {'patch_size': 4, 'enc_mask_scale': [0.8, 1.0],
                 'pred_mask_scale': [0.1, 0.2], 'aspect_ratio': [0.8, 1.2],
                 'num_enc_masks': 1, 'num_pred_masks': 2, 'min_keep': 1, 'allow_overlap': False},
        'meta': {'model_name': 'vit_tiny', 'pred_depth': 1, 'pred_emb_dim': 16,
                 'seed': seed, 'use_bfloat16': True},
        'optimization': {'epochs': epochs, 'accum_steps': 4, 'weight_decay': 0.04,
                         'final_weight_decay': 0.4, 'start_lr': 0.001, 'lr': 0.002,
                         'final_lr': 0.0001, 'warmup': 0, 'ema': [0.9, 0.99],
                         'patience': patience, 'save_every': save_every},
        'logging': {'folder': str(folder), 'write_tag': 'test'},
    }


def write_config(tmp_path, config, name='config.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


def run(tmp_path, config, stop=None, allow=False):
    path = write_config(tmp_path, config)
    return training.main(SimpleNamespace(config=path, stop_after_epoch=stop,
                                         allow_source_change=allow))


def fork_from(config, checkpoint, folder, start_epoch):
    config = json.loads(json.dumps(config))
    config['meta'].update({'load_checkpoint': True, 'read_checkpoint': str(checkpoint),
                           'resume_policy': 'fork', 'fork_start_epoch': start_epoch})
    config['logging']['folder'] = str(folder)
    return config


def exact_from(config, checkpoint):
    config = json.loads(json.dumps(config))
    config['meta'].update({'load_checkpoint': True, 'read_checkpoint': str(checkpoint)})
    return config


@pytest.fixture(autouse=True)
def isolated_hash_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('JEPA_SHA256_CACHE', str(tmp_path / 'sha256_cache.json'))


# ---------------------------------------------------------------------------
# E1: stop after a verified epoch
# ---------------------------------------------------------------------------

def test_stop_file_written_only_after_verified_saves_and_exit_zero(tmp_path, monkeypatch, capsys):
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / 'run'
    events = []
    real_verify, real_write = training.verify_checkpoint_file, training.atomic_write_json

    def verify(path, *args, **kwargs):
        events.append(('verify', os.path.basename(path)))
        return real_verify(path, *args, **kwargs)

    def write(path, payload):
        if os.path.basename(path).startswith('stop_epoch_'):
            events.append(('stop_file', os.path.basename(path)))
        return real_write(path, payload)

    monkeypatch.setattr(training, 'verify_checkpoint_file', verify)
    monkeypatch.setattr(training, 'atomic_write_json', write)
    config_path = write_config(tmp_path, make_config(tmp_path, folder, epochs=4, save_every=5))
    assert training.cli(['--config', config_path, '--stop-after-epoch', '2']) == 0

    assert events[-1] == ('stop_file', 'stop_epoch_002.json')
    verified = {name for kind, name in events[:-1] if kind == 'verify'}
    assert {'test-last.pth.tar', 'test-ep2.pth.tar'} <= verified
    stop = json.loads((folder / 'stop_epoch_002.json').read_text())
    assert stop['schema'] == training.STOP_FILE_SCHEMA
    assert stop['epoch'] == 2 and stop['status'] == 'verified'
    roles = {record['role']: record for record in stop['checkpoints']}
    assert {'last', 'periodic'} <= set(roles)
    for record in stop['checkpoints']:
        assert record['sha256'] == file_sha256(record['path'])
        assert record['bytes'] == os.path.getsize(record['path'])
        assert record['epoch'] == 2
    # save_every=5 does not divide 2: the stop epoch still gets a milestone.
    assert roles['periodic']['file'] == 'test-ep2.pth.tar'
    last = torch.load(folder / 'test-last.pth.tar', weights_only=False)
    assert last['epoch'] == 2
    assert stop['successful_updates'] == last['training_state']['successful_updates']
    assert stop['run_contract_sha256'] == last['training_state']['topology']['run_contract_sha256']
    assert stop['source_hash'] == last['training_state']['source_identity']['source_hash']
    for key in ('config_hash', 'seed', 'git_commit', 'timestamp_utc', 'source_hash'):
        assert key in stop
    assert stop['seed'] == 19
    with open(folder / 'test-log.csv') as stream:
        assert max(int(row['epoch']) for row in csv.DictReader(stream)) == 2
    assert sorted(os.listdir(folder)) == sorted([
        'run_manifest.json', 'run_manifests', 'stop_epoch_002.json', 'test-ep2.pth.tar',
        'test-last.pth.tar', 'test-log.csv'] + (['test-best.pth.tar']
                                                 if (folder / 'test-best.pth.tar').exists() else []))
    out = capsys.readouterr().out
    assert '[CKPT] epoch=2 saved:' in out and 'STOP: epoch 2 checkpoints verified' in out


def test_save_failure_raises_and_leaves_no_stop_file(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / 'run'
    real_save = training.save_checkpoint

    def failing_save(path, *args, **kwargs):
        if str(path).endswith('test-ep2.pth.tar'):
            raise OSError('disk full (simulated)')
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(training, 'save_checkpoint', failing_save)
    with pytest.raises(OSError, match='simulated'):
        run(tmp_path, make_config(tmp_path, folder), stop=2)
    assert not (folder / 'stop_epoch_002.json').exists()
    assert not (folder / 'test-ep2.pth.tar').exists()


def test_corrupt_checkpoint_fails_verification_without_stop_file(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / 'run'
    real_save = training.save_checkpoint

    def truncating_save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if str(path).endswith('test-last.pth.tar') and args[5] == 2:
            size = os.path.getsize(path)
            with open(path, 'r+b') as stream:
                stream.truncate(size // 2)

    monkeypatch.setattr(training, 'save_checkpoint', truncating_save)
    with pytest.raises(helper.CheckpointVerificationError):
        run(tmp_path, make_config(tmp_path, folder), stop=2)
    assert not (folder / 'stop_epoch_002.json').exists()


DRIVER = r'''
import sys
sys.path[:0] = [{repo!r}, {tests!r}]
import test_cr_trainer_run_control as fixture
from src import train_patch as training
fixture.install_tiny(setattr)
if {fail!r}:
    real_save = training.save_checkpoint
    def failing_save(path, *args, **kwargs):
        if str(path).endswith('test-ep2.pth.tar'):
            raise OSError('simulated save failure')
        return real_save(path, *args, **kwargs)
    training.save_checkpoint = failing_save
sys.exit(training.cli(['--config', {config!r}, '--stop-after-epoch', '2']))
'''


@pytest.mark.parametrize('fail', [False, True])
def test_process_exit_code(tmp_path, fail):
    folder = tmp_path / 'run'
    config_path = write_config(tmp_path, make_config(tmp_path, folder))
    driver = tmp_path / 'driver.py'
    driver.write_text(DRIVER.format(repo=REPO_ROOT, tests=TESTS_DIR, fail=fail,
                                    config=config_path))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JEPA_SHA256_CACHE=str(tmp_path / 'c.json'))
    result = subprocess.run([sys.executable, str(driver)], cwd=str(tmp_path), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    stop = folder / 'stop_epoch_002.json'
    if fail:
        assert result.returncode != 0
        assert b'simulated save failure' in result.stderr
        assert not stop.exists()
    else:
        assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')[-2000:]
        assert json.loads(stop.read_text())['epoch'] == 2


def test_stop_not_reached_returns_exit_3(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / 'run'
    # patience=0 early-stops after epoch 1, before the requested epoch 2.
    assert run(tmp_path, make_config(tmp_path, folder, patience=0), stop=2) == \
        training.EXIT_STOP_NOT_REACHED == 3
    assert not (folder / 'stop_epoch_002.json').exists()


def test_nonfinite_epoch_loss_aborts_with_exit_4_before_saving(tmp_path, monkeypatch, capsys):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    real_loss = training.jepa_forward_loss
    calls = [0]

    def poisoned(*args, **kwargs):
        calls[0] += 1
        loss, predictions, targets = real_loss(*args, **kwargs)
        # 7 training microbatches per epoch: poison one batch of epoch 2.
        return (loss * float('nan') if calls[0] == 10 else loss), predictions, targets

    monkeypatch.setattr(training, 'jepa_forward_loss', poisoned)
    assert run(tmp_path, make_config(tmp_path, folder, epochs=3), stop=3) == \
        training.EXIT_NONFINITE_LOSS == 4
    assert torch.load(folder / 'test-last.pth.tar', weights_only=False)['epoch'] == 1
    assert not (folder / 'test-ep2.pth.tar').exists()
    assert not any(name.startswith('stop_epoch_') for name in os.listdir(folder))
    assert 'non-finite loss at epoch 2' in capsys.readouterr().out


def test_stop_epoch_is_validated(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    for bad in (0, 4):
        with pytest.raises(ValueError, match='stop-after-epoch'):
            run(tmp_path, config, stop=bad)
    assert run(tmp_path, config, stop=1) == 0
    last = folder / 'test-last.pth.tar'
    with pytest.raises(ValueError, match='already exists'):
        run(tmp_path, exact_from(config, last), stop=1)
    assert run(tmp_path, exact_from(config, last), stop=2) == 0
    # Exact resume from epoch 2 toward stop 1 is impossible.
    os.remove(folder / 'stop_epoch_001.json')
    with pytest.raises(ValueError, match='not after the start epoch'):
        run(tmp_path, exact_from(config, last), stop=1)
    # A fork that would start at the stop epoch has nothing to train or verify.
    with pytest.raises(ValueError, match='not after the start epoch'):
        run(tmp_path, fork_from(config, last, tmp_path / 'fork', 2), stop=2)


def test_crash_between_save_and_stop_file_recovers_by_rerun(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    real_finalize = training.finalize_stop_epoch

    def crash(*args, **kwargs):
        raise MemoryError('process died after saving (simulated)')

    monkeypatch.setattr(training, 'finalize_stop_epoch', crash)
    with pytest.raises(MemoryError):
        run(tmp_path, config, stop=2)
    assert not (folder / 'stop_epoch_002.json').exists()
    monkeypatch.setattr(training, 'finalize_stop_epoch', real_finalize)
    last = folder / 'test-last.pth.tar'
    roles = torch.load(last, weights_only=False)['training_state']['epoch_save_roles']
    assert 'last' in roles and 'periodic' in roles
    files = {'best': 'test-best.pth.tar', 'last': 'test-last.pth.tar',
             'periodic': 'test-ep2.pth.tar'}
    digests = {role: file_sha256(str(folder / files[role])) for role in roles}
    with open(folder / 'test-log.csv') as stream:
        rows = stream.read()
    assert run(tmp_path, exact_from(config, last), stop=2) == 0
    stop = json.loads((folder / 'stop_epoch_002.json').read_text())
    assert {record['role']: record['sha256'] for record in stop['checkpoints']} == digests
    with open(folder / 'test-log.csv') as stream:
        assert stream.read() == rows
    # Without the epoch-N milestone the recovery fails closed.
    os.remove(folder / 'stop_epoch_002.json')
    os.remove(folder / 'test-ep2.pth.tar')
    with pytest.raises(helper.CheckpointVerificationError, match='periodic'):
        run(tmp_path, exact_from(config, last), stop=2)
    assert not (folder / 'stop_epoch_002.json').exists()


def crash_after_epoch_saves(tmp_path, monkeypatch, config, stop):
    """Run to ``stop`` but die after every checkpoint write, before the receipt."""
    real_finalize = training.finalize_stop_epoch

    def crash(*args, **kwargs):
        raise MemoryError('process died after saving (simulated)')

    monkeypatch.setattr(training, 'finalize_stop_epoch', crash)
    with pytest.raises(MemoryError):
        run(tmp_path, config, stop=stop)
    monkeypatch.setattr(training, 'finalize_stop_epoch', real_finalize)


def test_recovery_requires_the_complete_save_set_including_best(tmp_path, monkeypatch):
    # P1.1: a crash-recovery receipt must certify every file written for epoch N.
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    crash_after_epoch_saves(tmp_path, monkeypatch, config, stop=1)
    last, best = folder / 'test-last.pth.tar', folder / 'test-best.pth.tar'
    assert torch.load(last, weights_only=False)['training_state']['epoch_save_roles'] == [
        'best', 'last', 'periodic']
    good_best = best.read_bytes()
    best.write_bytes(b'x' * 42)
    with pytest.raises(helper.CheckpointVerificationError, match='best'):
        run(tmp_path, exact_from(config, last), stop=1)
    os.remove(best)
    with pytest.raises(helper.CheckpointVerificationError, match="missing"):
        run(tmp_path, exact_from(config, last), stop=1)
    assert not (folder / 'stop_epoch_001.json').exists()
    best.write_bytes(good_best)
    assert run(tmp_path, exact_from(config, last), stop=1) == 0
    stop = json.loads((folder / 'stop_epoch_001.json').read_text())
    assert sorted(record['role'] for record in stop['checkpoints']) == ['best', 'last', 'periodic']


def test_recovery_of_legacy_checkpoint_still_checks_a_same_epoch_best(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    crash_after_epoch_saves(tmp_path, monkeypatch, config, stop=1)
    last, best = folder / 'test-last.pth.tar', folder / 'test-best.pth.tar'
    legacy = torch.load(last, weights_only=False)
    del legacy['training_state']['epoch_save_roles']
    torch.save(legacy, last)
    good_best = best.read_bytes()
    best.write_bytes(b'x' * 42)
    with pytest.raises(helper.CheckpointVerificationError, match='best'):
        run(tmp_path, exact_from(config, last), stop=1)
    best.write_bytes(good_best)
    assert run(tmp_path, exact_from(config, last), stop=1) == 0
    stop = json.loads((folder / 'stop_epoch_001.json').read_text())
    assert sorted(record['role'] for record in stop['checkpoints']) == ['best', 'last', 'periodic']


def _mutate_run_uuid(checkpoint):
    checkpoint['training_state']['run_uuid'] = 'foreign-run'


def _mutate_topology(checkpoint):
    checkpoint['training_state']['topology'] = dict(
        checkpoint['training_state']['topology'], run_contract_sha256='wrong-contract')


def _mutate_source(checkpoint):
    checkpoint['training_state']['source_identity'] = dict(
        checkpoint['training_state']['source_identity'], source_hash='f' * 64)


def _mutate_teacher(checkpoint):
    key = sorted(checkpoint['target_encoder'])[0]
    checkpoint['target_encoder'][key] = checkpoint['target_encoder'][key] + 1.0


def _mutate_empty_predictor(checkpoint):
    checkpoint['predictor'] = {}


@pytest.mark.parametrize('mutate, message', [
    (_mutate_run_uuid, 'run_uuid'), (_mutate_topology, 'topology'),
    (_mutate_source, 'source_identity'), (_mutate_teacher, 'model weights'),
    (_mutate_empty_predictor, 'empty')])
def test_recovery_rejects_artifacts_from_another_run(tmp_path, monkeypatch, mutate, message):
    # P1.2: recovered files must belong to the resumed run, not merely share epoch/updates.
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    crash_after_epoch_saves(tmp_path, monkeypatch, config, stop=1)
    periodic = folder / 'test-ep1.pth.tar'
    foreign = torch.load(periodic, weights_only=False)
    mutate(foreign)
    torch.save(foreign, periodic)
    with pytest.raises(helper.CheckpointVerificationError, match=message):
        run(tmp_path, exact_from(config, folder / 'test-last.pth.tar'), stop=1)
    assert not (folder / 'stop_epoch_001.json').exists()


def test_verify_only_receipt_keeps_the_producing_source(tmp_path, monkeypatch):
    # P1.3: verification under new code must not relabel who produced the files.
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    crash_after_epoch_saves(tmp_path, monkeypatch, config, stop=1)
    last = folder / 'test-last.pth.tar'
    produced = torch.load(last, weights_only=False)['training_state']
    monkeypatch.setattr(training, 'collect_source_identity', fake_source('c' * 64))
    with pytest.raises(ValueError, match='source hash changed'):
        run(tmp_path, exact_from(config, last), stop=1)
    assert run(tmp_path, exact_from(config, last), stop=1, allow=True) == 0
    stop = json.loads((folder / 'stop_epoch_001.json').read_text())
    assert stop['source_hash'] == 'a' * 64
    assert stop['verifier_source']['source_hash'] == 'c' * 64
    assert stop['recovered_without_training'] is True
    assert stop['run_uuid'] == produced['run_uuid']
    for record in stop['checkpoints']:
        state = torch.load(record['path'], weights_only=False)['training_state']
        assert state['source_identity']['source_hash'] == stop['source_hash']
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    assert [item['source_hash'] for item in manifest['source_history']] == ['a' * 64]


def test_stop_flag_is_outside_the_run_contract(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    assert run(tmp_path, config, stop=1) == 0
    first = torch.load(folder / 'test-last.pth.tar', weights_only=False)
    # Exact resume with a different (or no) stop flag continues the same run.
    assert run(tmp_path, exact_from(config, folder / 'test-last.pth.tar')) == 0
    final = torch.load(folder / 'test-last.pth.tar', weights_only=False)
    assert final['epoch'] == 3
    assert (first['training_state']['topology']
            == final['training_state']['topology'])


def test_run_uuid_persists_across_exact_resume_and_is_new_for_a_fork(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    assert run(tmp_path, config, stop=1) == 0
    first = json.loads((folder / 'stop_epoch_001.json').read_text())
    run_uuid = first['run_uuid']
    assert len(run_uuid) == 32 and first['mask_policy'] == 'uniform_multiblock'
    last = folder / 'test-last.pth.tar'
    assert torch.load(last, weights_only=False)['training_state']['run_uuid'] == run_uuid
    assert run(tmp_path, exact_from(config, last), stop=2) == 0
    assert json.loads((folder / 'stop_epoch_002.json').read_text())['run_uuid'] == run_uuid
    assert json.loads((folder / 'run_manifest.json').read_text())['run_uuid'] == run_uuid
    assert run(tmp_path, fork_from(config, last, tmp_path / 'fork', 2), stop=3) == 0
    forked = json.loads((tmp_path / 'fork' / 'stop_epoch_003.json').read_text())
    assert forked['run_uuid'] != run_uuid


def test_save_checkpoint_is_atomic(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = str(tmp_path / 'x-ep5.pth.tar')
    helper.save_checkpoint(path, model, model, model, optimizer, None, 5, 0.1, 1, 1, 0.1)
    before = file_sha256(path)
    real_save = torch.save

    def partial_then_crash(obj, stream, *args, **kwargs):
        stream.write(b'partial bytes')
        raise OSError('crash mid-write (simulated)')

    monkeypatch.setattr(helper.torch, 'save', partial_then_crash)
    with pytest.raises(OSError, match='simulated'):
        helper.save_checkpoint(path, model, model, model, optimizer, None, 6, 0.1, 1, 1, 0.1)
    monkeypatch.setattr(helper.torch, 'save', real_save)
    assert file_sha256(path) == before
    assert os.listdir(tmp_path) == ['x-ep5.pth.tar']


def test_verify_checkpoint_rejects_wrong_epoch_and_missing_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = str(tmp_path / 'x.pth.tar')
    state = {'successful_updates': 3, 'lr_scheduler': {}, 'wd_scheduler': {},
             'topology': {}, 'rank_states': []}
    helper.save_checkpoint(path, model, model, model, optimizer, None, 5, 0.1, 1, 1, 0.1,
                           training_state=state)
    record = helper.verify_checkpoint_file(path, 5, expected_successful_updates=3)
    assert record['sha256'] == file_sha256(path) and record['bytes'] == os.path.getsize(path)
    with pytest.raises(helper.CheckpointVerificationError, match='epoch'):
        helper.verify_checkpoint_file(path, 6)
    with pytest.raises(helper.CheckpointVerificationError, match='successful updates'):
        helper.verify_checkpoint_file(path, 5, expected_successful_updates=4)
    helper.save_checkpoint(path, model, model, model, optimizer, None, 5, 0.1, 1, 1, 0.1)
    with pytest.raises(helper.CheckpointVerificationError, match='training_state'):
        helper.verify_checkpoint_file(path, 5)


# ---------------------------------------------------------------------------
# E8: provenance manifest, source hash, cached read-checkpoint hash
# ---------------------------------------------------------------------------

def test_run_manifest_records_required_provenance(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / 'run'
    assert run(tmp_path, make_config(tmp_path, folder), stop=1) == 0
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    assert manifest['schema'] == training.RUN_MANIFEST_SCHEMA
    for key in ('timestamp_utc', 'invocation', 'config', 'config_hash', 'run_contract_sha256',
                'code', 'source_hash', 'git_commit', 'git_dirty', 'seed', 'sampler_seed',
                'resume_policy', 'read_checkpoint', 'resume_state_at_start', 'train_sampler',
                'val_sampler', 'amp', 'loader', 'batch', 'data', 'runtime'):
        assert key in manifest, key
    code = manifest['code']
    assert code['code_root'] == os.path.realpath(REPO_ROOT)
    assert code['listing'] == 'git_ls_files'
    assert code['files']['src/train_patch.py'] == file_sha256(
        os.path.join(REPO_ROOT, 'src', 'train_patch.py'))
    assert code['source_hash'] == manifest['source_hash']
    assert len(code['git_commit']) == 40 and isinstance(code['git_dirty'], bool)
    assert manifest['config_hash'] == training.config_sha256(manifest['config'])
    assert manifest['config']['meta']['seed'] == manifest['seed'] == 19
    assert manifest['invocation']['stop_after_epoch'] == 1
    assert manifest['train_sampler']['seed'] == 19
    assert manifest['train_sampler']['seed_source'] == 'meta.seed'
    assert manifest['val_sampler']['shuffle'] is False
    assert manifest['amp'] == {'use_bfloat16': True, 'amp_target': False, 'grad_scaler': False}
    assert manifest['batch']['accum_steps'] == 4 and manifest['batch']['world_size'] == 1
    assert manifest['loader']['num_workers'] == 0
    runtime = manifest['runtime']
    for key in ('hostname', 'torch', 'cuda_runtime', 'cudnn_version', 'cudnn_benchmark',
                'cudnn_deterministic', 'gpu_name', 'nvidia_smi'):
        assert key in runtime, key
    assert manifest['read_checkpoint'] is None
    assert len(os.listdir(folder / 'run_manifests')) == 1
    last = torch.load(folder / 'test-last.pth.tar', weights_only=False)
    assert last['training_state']['source_identity']['source_hash'] == manifest['source_hash']
    assert last['training_state']['sampler_seed'] == 19


def test_data_identity_hashes_manifests_not_data(tmp_path):
    cache = tmp_path / 'cache'
    for split in ('Training', 'Validation'):
        (cache / split).mkdir(parents=True)
        (cache / split / 'slice_cache.json').write_text('{"split": "%s"}' % split)
        (cache / split / 'slice_cache.u8').write_bytes(b'\0' * 10)
    guides = tmp_path / 'root' / 'mirage_guides'
    (guides / 'Training').mkdir(parents=True)
    (guides / 'Training' / 'data_00001.npz').write_bytes(b'abc')
    (tmp_path / 'root' / 'manifests').mkdir()
    (tmp_path / 'root' / 'manifests' / 'mirage-guides-complete.json').write_text('{}')
    info = training.data_identity({'data_dir': str(tmp_path), 'slice_cache_dir': str(cache)},
                                  str(guides), TinyDataset('x/Training'), None)
    record = info['slice_cache']['Training']
    assert record['manifest_sha256'] == file_sha256(str(cache / 'Training' / 'slice_cache.json'))
    assert record['array_bytes'] == 10
    assert info['guides']['files'] == 1 and info['guides']['total_bytes'] == 3
    assert len(info['guides']['manifest_sha256']) == 1
    before = info['guides']['listing_sha256']
    (guides / 'Training' / 'data_00002.npz').write_bytes(b'd')
    assert training.data_identity({'data_dir': str(tmp_path)}, str(guides), None, None)[
        'guides']['listing_sha256'] != before


def test_nvidia_smi_failure_is_tolerated(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(training.subprocess, 'run', missing)
    assert 'nvidia-smi' in training._nvidia_smi_query()['error']


def test_file_hash_cache_keys_on_size_and_mtime(tmp_path):
    path = tmp_path / 'big.bin'
    path.write_bytes(b'x' * 100)
    cache = str(tmp_path / 'cache.json')
    first = helper.file_sha256_cached(str(path), cache)
    assert first == (file_sha256(str(path)), 'computed')
    assert helper.file_sha256_cached(str(path), cache) == (first[0], 'cache')
    path.write_bytes(b'y' * 101)
    assert helper.file_sha256_cached(str(path), cache) == (file_sha256(str(path)), 'computed')


def test_exact_resume_refuses_changed_source_hash(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3)
    assert run(tmp_path, config, stop=1) == 0
    last = folder / 'test-last.pth.tar'
    before = file_sha256(str(last))
    monkeypatch.setattr(training, 'collect_source_identity', fake_source('b' * 64))
    with pytest.raises(ValueError, match='source hash changed.*allow-source-change'):
        run(tmp_path, exact_from(config, last), stop=2)
    assert file_sha256(str(last)) == before
    assert not (folder / 'stop_epoch_002.json').exists()
    with pytest.warns(UserWarning, match='CHANGED source'):
        assert run(tmp_path, exact_from(config, last), stop=2, allow=True) == 0
    state = torch.load(last, weights_only=False)['training_state']
    assert state['source_identity']['source_hash'] == 'b' * 64
    assert [(item['source_hash'][0], item['from_epoch']) for item in state['source_history']] \
        == [('a', 0), ('b', 1)]
    stop = json.loads((folder / 'stop_epoch_002.json').read_text())
    assert stop['source_hash'] == 'b' * 64


def test_load_checkpoint_source_check_and_legacy_warning(tmp_path):
    model = torch.nn.Linear(2, 2)
    teacher = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = str(tmp_path / 'x.pth.tar')
    state = {'topology': {'t': 1}, 'rank_states': [],
             'source_identity': {'source_hash': 'a' * 64}}
    helper.save_checkpoint(path, model, model, teacher, optimizer, None, 3, 0.1, 1, 1, 0.1,
                           training_state=state)
    kwargs = dict(training_state={}, topology={'t': 1})
    with pytest.raises(ValueError, match='source hash changed'):
        helper.load_checkpoint('cpu', path, model, model, teacher, optimizer, None,
                               source_hash='b' * 64, **kwargs)
    with pytest.warns(UserWarning):
        helper.load_checkpoint('cpu', path, model, model, teacher, optimizer, None,
                               source_hash='a' * 64, **kwargs)
    del state['source_identity']
    helper.save_checkpoint(path, model, model, teacher, optimizer, None, 3, 0.1, 1, 1, 0.1,
                           training_state=state)
    with pytest.warns(UserWarning, match='predates source-hash tracking'):
        helper.load_checkpoint('cpu', path, model, model, teacher, optimizer, None,
                               source_hash='b' * 64, **kwargs)


# ---------------------------------------------------------------------------
# E19 fork guard and G0 start-state logging
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('stale', ['old-last.pth.tar', 'old-ep5.pth.tar.tmp'])
def test_fork_refuses_folder_with_checkpoints(tmp_path, monkeypatch, stale):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    source_folder = tmp_path / 'source'
    config = make_config(tmp_path, source_folder, epochs=3)
    assert run(tmp_path, config, stop=1) == 0
    target = tmp_path / 'fork'
    target.mkdir()
    (target / stale).write_bytes(b'stale')
    with pytest.raises(ValueError, match="fork.*refuses to start"):
        run(tmp_path, fork_from(config, source_folder / 'test-ep1.pth.tar', target, 1), stop=2)
    assert sorted(os.listdir(target)) == [stale]


def test_fork_logs_start_state_and_records_checkpoint_hash(tmp_path, monkeypatch, capsys):
    from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    source_folder = tmp_path / 'source'
    config = make_config(tmp_path, source_folder, epochs=3)
    assert run(tmp_path, config, stop=1) == 0
    source = source_folder / 'test-ep1.pth.tar'
    source_state = torch.load(source, weights_only=False)
    adam_step = int(max(float(s['step']) for s in source_state['opt']['state'].values()))
    capsys.readouterr()

    assert run(tmp_path, fork_from(config, source, tmp_path / 'fork1', 1), stop=2) == 0
    line = [row for row in capsys.readouterr().out.splitlines() if '[RESUME-STATE]' in row]
    assert len(line) == 1
    fields = dict(item.split('=', 1) for item in line[0].split()[1:])
    ipe, total = 2, 3 * 2
    probe = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.001, weight_decay=0.04)
    lr_schedule = WarmupCosineSchedule(probe, 0, 0.001, 0.002, 0.0001, total)
    wd_schedule = CosineWDSchedule(probe, 0.04, 0.4, total)
    for _ in range(ipe):
        lr_schedule.step()
        wd_schedule.step()
    expected_ema = list(training.momentum_schedule(0.9, 0.99, total))[ipe]
    assert fields['policy'] == 'fork' and fields['loaded_epoch'] == '1'
    assert fields['start_epoch'] == '1' and int(fields['adam_step']) == adam_step
    assert fields['grad_scaler_scale'] == 'None'
    assert float(fields['first_update_lr']) == pytest.approx(probe.param_groups[0]['lr'], rel=1e-9)
    assert float(fields['first_update_wd']) == pytest.approx(
        probe.param_groups[0]['weight_decay'], rel=1e-9)
    assert float(fields['first_update_ema']) == pytest.approx(expected_ema, rel=1e-9)
    assert fields['iterations_per_epoch'] == '2' and fields['accum_steps'] == '4'

    manifest = json.loads((tmp_path / 'fork1' / 'run_manifest.json').read_text())
    assert manifest['read_checkpoint']['sha256'] == file_sha256(str(source))
    assert manifest['read_checkpoint']['sha256_source'] == 'computed'
    assert manifest['resume_state_at_start']['adam_step_max'] == adam_step
    assert manifest['lineage']['source_checkpoint_sha256'] == file_sha256(str(source))
    assert run(tmp_path, fork_from(config, source, tmp_path / 'fork2', 1), stop=2) == 0
    manifest = json.loads((tmp_path / 'fork2' / 'run_manifest.json').read_text())
    assert manifest['read_checkpoint']['sha256_source'] == 'cache'


def test_grad_scaler_scale_reported_exactly():
    scaler = torch.amp.GradScaler('cpu', init_scale=2.0 ** 20)
    assert training.grad_scaler_scale(scaler) == 1048576
    assert training.grad_scaler_scale(None) is None


# ---------------------------------------------------------------------------
# E6 inside the trainer
# ---------------------------------------------------------------------------

def test_trainer_sampler_uses_seed_and_legacy_exact_resume_keeps_zero(tmp_path, monkeypatch):
    install_tiny(monkeypatch.setattr, stub_source='a' * 64)
    samplers = []
    real_make = training.make_train_sampler

    def capture(*args, **kwargs):
        samplers.append(real_make(*args, **kwargs))
        return samplers[-1]

    monkeypatch.setattr(training, 'make_train_sampler', capture)
    folder = tmp_path / 'run'
    config = make_config(tmp_path, folder, epochs=3, seed=1234)
    assert run(tmp_path, config, stop=1) == 0
    assert samplers[-1].seed == 1234 and samplers[-1].shuffle
    legacy = torch.load(folder / 'test-last.pth.tar', weights_only=False)
    del legacy['training_state']['sampler_seed']
    legacy_path = tmp_path / 'legacy.pth.tar'
    torch.save(legacy, legacy_path)
    assert run(tmp_path, exact_from(config, legacy_path), stop=2) == 0
    assert samplers[-1].seed == 0
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    assert manifest['train_sampler']['seed_source'] == 'legacy_checkpoint_default_0'
    assert torch.load(folder / 'test-last.pth.tar',
                      weights_only=False)['training_state']['sampler_seed'] == 0
