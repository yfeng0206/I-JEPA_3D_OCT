"""Executable contracts for token ordering, optimizer steps and epoch resume."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.helper import load_checkpoint, save_checkpoint
from src.masks.utils import apply_masks
from src.models.vision_transformer import (
    VisionTransformer, VisionTransformerPredictor, get_2d_sincos_pos_embed,
)
from src.utils.tensors import repeat_interleave_batch
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule


def test_cover_logs_do_not_substitute_complement_for_delivered_context():
    from src.train_patch import format_cover_stats
    legacy = {'cover_hidden_frac': 0.8, 'cover_visible_cells': 20, 'cover_floor_ok': 1.0}
    assert 'stats_scope=policy_target_complement' in format_cover_stats(legacy)
    assert 'delivered_context_floor=not_reported' in format_cover_stats(legacy)
    from scripts.campaign_supervisor import COVER_RE
    assert COVER_RE.search(format_cover_stats(legacy)).groups() == ('0.800', '20.0', '1.000')
    measured = dict(legacy, delivered_context_floor_satisfied=1,
                    delivered_context_floor_unsatisfied=1, delivered_context_interventions=1)
    message = format_cover_stats(measured)
    assert 'delivered_context_floor_satisfied=1' in message
    assert 'delivered_context_floor_unsatisfied=1' in message
    assert 'delivered_context_interventions=1' in message
    assert 'not_reported' not in message


def test_joint_context_loss_rows_preserve_image_context_target_order():
    from scripts.training_layer_diagnostic import joint_context_loss_summary
    slots = torch.arange(1, 17).float().reshape(8, 2) / 100
    context = [torch.tensor([[3], [3]]), torch.tensor([[3], [3]])]
    targets = [torch.tensor([[0, 1], [0, 1]]), torch.tensor([[1, 2], [1, 2]])]
    tissue = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    duplicates = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    duplicates[1, :, :, 0] = True
    rows = joint_context_loss_summary(slots, context, targets, tissue, duplicates,
                                     torch.tensor([True, False]))
    assert [row['loss_slots'] for row in rows] == [8, 8]
    assert [row['partitions']['tissue']['slots'] for row in rows] == [2, 6]
    assert rows[0]['partitions']['background_repeat']['loss_sum'] == pytest.approx(0.22)
    assert rows[1]['partitions']['tissue_repeat']['loss_sum'] == pytest.approx(0.26)
    assert rows[0]['scalar_loss'] == pytest.approx(0.075)
    assert rows[1]['scalar_loss'] == pytest.approx(0.095)
    assert rows[0]['context_tissue_cells_per_group'] == [0, 0]
    assert rows[1]['guide_valid'] is False


def test_private_guided_export_adapter_preserves_fixed_tensors(tmp_path):
    from scripts.training_layer_diagnostic import load_diagnostic_cases
    names = ['cover_legacy', 'cover_v2', 'cover_v2_guard']
    images = torch.zeros(2, 3, 16, 16)
    context, targets = [torch.tensor([[3], [3]])], [torch.tensor([[0, 1], [1, 2]])]
    payload = {
        'private_real_data': True,
        'batches': {'bs2': {
            'images': images, 'tissue_labels': torch.ones(2, 2, 2, dtype=torch.bool),
            'guide_valid': torch.tensor([True, True]), 'ordinals': ['private-a', 'private-b'],
            'policies': {name: {'masks_enc': context, 'masks_pred': targets,
                                'target_sources': [['guided'], ['random_legal']]} for name in names},
        }},
    }
    path = tmp_path / 'private.pt'
    torch.save(payload, path)
    cases = load_diagnostic_cases(path, {'fixture_policies': names})
    assert [case['label'] for case in cases] == names
    for case in cases:
        torch.testing.assert_close(case['images'], images, rtol=0, atol=0)
        torch.testing.assert_close(case['masks_pred'][0], targets[0], rtol=0, atol=0)
        assert case['tissue'].shape == (2, 4)
        assert 'ordinals' not in case


def test_predictor_image_context_target_sentinel_order():
    batch, nenc, npred = 3, 2, 3
    context = [torch.tensor([[0, 1], [1, 2], [2, 3]]) + c * 4 for c in range(nenc)]
    targets = [torch.tensor([[8, 9], [9, 10], [10, 11]]) + p for p in range(npred)]
    tokens = torch.arange(batch)[:, None, None] * 1000 + torch.arange(16)[None, :, None]
    tokens = tokens.float().expand(-1, -1, 8)
    selected = apply_masks(tokens, targets)
    repeated = repeat_interleave_batch(selected, batch, nenc)
    expected = torch.stack([
        tokens[b, targets[p][b]] for p in range(npred)
        for c in range(nenc) for b in range(batch)
    ])
    torch.testing.assert_close(repeated, expected)

    predictor = VisionTransformerPredictor(16, 8, 8, depth=1, num_heads=2)
    predictor.predictor_embed = nn.Identity()
    predictor.predictor_norm = nn.Identity()
    predictor.predictor_proj = nn.Identity()
    with torch.no_grad():
        predictor.mask_token.zero_()
        predictor.predictor_pos_embed.copy_(torch.arange(16)[None, :, None].expand(1, 16, 8))
    recorded = []
    handle = predictor.predictor_blocks[0].register_forward_pre_hook(
        lambda module, args: recorded.append(args[0].detach().clone()))
    predictor(apply_masks(tokens, context), context, targets)
    handle.remove()
    for p in range(npred):
        for c in range(nenc):
            for b in range(batch):
                row = (p * nenc + c) * batch + b
                torch.testing.assert_close(recorded[0][row, :2],
                                           tokens[b, context[c][b]] + context[c][b, :, None])
                torch.testing.assert_close(recorded[0][row, 2:, 0], targets[p][b].float())
    # A deliberately wrong repeat order must be caught by the sentinel.
    assert not torch.equal(selected.repeat(nenc, 1, 1), expected)


def test_masked_online_does_not_observe_hidden_pixels():
    torch.manual_seed(4)
    encoder = VisionTransformer(img_size=16, patch_size=4, embed_dim=16, depth=2, num_heads=2)
    image = torch.randn(2, 3, 16, 16, requires_grad=True)
    masks = [torch.tensor([[0, 1], [0, 1]])]
    changed = image.detach().clone()
    changed[:, :, 8:, :] += 17
    online = encoder(image, masks)
    torch.testing.assert_close(online, encoder(changed, masks), rtol=0, atol=0)
    assert not torch.allclose(encoder(image), encoder(changed))
    online.square().sum().backward()
    assert image.grad[:, :, 8:, :].count_nonzero() == 0
    assert image.grad[:, :, :4, :8].abs().sum() > 0


def test_positional_axes_and_teacher_normalization():
    pos = get_2d_sincos_pos_embed(16, (3, 4)).reshape(3, 4, 16)
    np.testing.assert_array_equal(pos[0, 0, :8], pos[0, 3, :8])
    np.testing.assert_array_equal(pos[0, 0, 8:], pos[2, 0, 8:])
    assert not np.array_equal(pos[0, 0, :8], pos[2, 0, :8])
    from src.train_patch import jepa_forward_loss
    encoder = VisionTransformer(img_size=16, patch_size=4, embed_dim=16, depth=2, num_heads=2)
    predictor = VisionTransformerPredictor(16, 16, 16, depth=2, num_heads=2)
    teacher = copy.deepcopy(encoder).requires_grad_(False)
    images = torch.randn(2, 3, 16, 16)
    context = [torch.tensor([[0, 1], [1, 2]]), torch.tensor([[2, 3], [3, 4]])]
    targets = [torch.tensor([[6, 7], [7, 8]]), torch.tensor([[8, 9], [9, 10]])]
    loss, prediction, target = jepa_forward_loss(encoder, predictor, teacher, images, context, targets)
    assert prediction.shape == target.shape == (8, 2, 16)
    torch.testing.assert_close(target.mean(-1), torch.zeros(8, 2), atol=2e-7, rtol=0)
    torch.testing.assert_close(target.var(-1, unbiased=False), torch.ones(8, 2), atol=3e-5, rtol=0)
    loss.backward()
    for model in (encoder, predictor):
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in teacher.parameters())
    torch.testing.assert_close(loss, torch.nn.functional.smooth_l1_loss(prediction, target))


def test_partial_accumulation_matches_actual_batch():
    from src.train_patch import accumulation_window_size
    batches = [torch.tensor([float(i + 1)]) for i in range(7)]
    actual = nn.Linear(1, 1, bias=False)
    reference = copy.deepcopy(actual)
    opt = torch.optim.SGD(actual.parameters(), lr=0.01)
    refopt = torch.optim.SGD(reference.parameters(), lr=0.01)
    for i, batch in enumerate(batches):
        if i % 4 == 0:
            opt.zero_grad()
        (actual(batch).square().mean() / accumulation_window_size(i, 7, 4)).backward()
        if (i + 1) % 4 == 0 or i == 6:
            opt.step()
    for start in (0, 4):
        refopt.zero_grad()
        reference(torch.stack(batches[start:start + 4])).square().mean().backward()
        refopt.step()
    torch.testing.assert_close(actual.weight, reference.weight)
    assert accumulation_window_size(4, 7, 4) == 3


def test_historical_closure_reproduces_partial_window_underweight():
    import ast
    import subprocess
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    source = subprocess.check_output(
        ['git', 'show', 'de145d7:src/train_patch.py'], text=True, encoding='utf-8')
    closure = next(node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.FunctionDef) and node.name == '_forward_backward')

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.3))

        def forward(self, images, masks=None):
            out = images * self.weight
            return apply_masks(out, masks) if masks is not None else out

    class Predictor(nn.Module):
        def forward(self, x, context, targets):
            return x[:, :1]

    online = Encoder()
    teacher = Encoder().requires_grad_(False)
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
    image = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    namespace = dict(torch=torch, F=F, autocast=autocast, apply_masks=apply_masks,
                     repeat_interleave_batch=repeat_interleave_batch,
                     itr=0, accum_steps=4, optimizer=optimizer, h_full=None,
                     amp_target=False, target_encoder=teacher, imgs=image,
                     masks_pred=[torch.tensor([[1]])], masks_enc=[torch.tensor([[0]])],
                     B=1, scaler=None, encoder=online, predictor=Predictor(),
                     train_loader=[0, 1, 2], use_curriculum=False)
    exec(compile(ast.Module(body=[closure], type_ignores=[]), '<de145d7 closure>', 'exec'), namespace)
    for iteration in range(3):
        namespace['itr'] = iteration
        namespace['_forward_backward']()
    historical = online.weight.detach().clone()
    reference = Encoder()
    opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    from src.train_patch import jepa_forward_loss
    loss, _, _ = jepa_forward_loss(reference, Predictor(), teacher, image,
                                    namespace['masks_enc'], namespace['masks_pred'])
    loss.backward()
    opt.step()
    # Real baseline closure assigns 3/4 of the correct gradient to this window.
    torch.testing.assert_close(0.3 - historical, (0.3 - reference.weight) * 0.75)
    assert not torch.allclose(historical, reference.weight)


class SkipScaler:
    def __init__(self, skip):
        self.skip = skip
        self.scale_value = 8.0

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self):
        if self.skip:
            self.scale_value /= 2


@pytest.mark.parametrize("skip", [False, True])
def test_skip_step_gates_schedule_and_ema(skip):
    from src.helper import optimizer_step, update_ema
    encoder = nn.Linear(2, 2)
    target = copy.deepcopy(encoder).requires_grad_(False)
    opt = torch.optim.SGD(encoder.parameters(), lr=0.1, weight_decay=0.1)
    lr = WarmupCosineSchedule(opt, 2, 0.1, 0.2, 0.01, 10)
    wd = CosineWDSchedule(opt, 0.1, 0.2, 10)
    before = copy.deepcopy(target.state_dict())
    encoder(torch.ones(1, 2)).sum().backward()
    success = optimizer_step(opt, SkipScaler(skip))
    if success:
        lr.step()
        wd.step()
        update_ema(encoder, target, 0.9)
    assert success == (not skip)
    assert lr._step == wd._step == int(not skip)
    for key, value in target.state_dict().items():
        expected = before[key] if skip else before[key] * 0.9 + encoder.state_dict()[key] * 0.1
        torch.testing.assert_close(value, expected)


def test_real_gradscaler_overflow_reports_skipped_update():
    from src.helper import optimizer_step
    if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'GradScaler'):
        pytest.skip('CPU GradScaler is unavailable in older supported PyTorch versions')
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=128)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    model.weight.grad.fill_(float('inf'))
    before = copy.deepcopy(model.state_dict())
    assert optimizer_step(optimizer, scaler) is False
    assert scaler.get_scale() == 64
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)


def test_historical_schedule_block_advances_without_optimizer_update():
    import ast
    import subprocess
    source = subprocess.check_output(
        ['git', 'show', 'de145d7:src/train_patch.py'], text=True, encoding='utf-8')
    block = next(node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                 and node.test.id == 'is_step'
                 and any(isinstance(child, ast.Assign)
                         and isinstance(child.targets[0], ast.Name)
                         and child.targets[0].id == 'lr_val' for child in node.body))
    online, teacher = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(1)
        teacher.weight.zero_()
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1, weight_decay=0.1)
    lr = WarmupCosineSchedule(optimizer, 1, 0.1, 0.2, 0.01, 10)
    wd = CosineWDSchedule(optimizer, 0.1, 0.2, 10)
    namespace = dict(torch=torch, is_step=True, encoder=online, target_encoder=teacher,
                     lr_scheduler=lr, wd_scheduler=wd, mom_schedule=iter([0.9]))
    # No optimizer step has taken place (the AMP-overflow case).
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<de145d7 schedule>', 'exec'), namespace)
    assert lr._step == wd._step == 1
    torch.testing.assert_close(teacher.weight, torch.tensor([[0.1]]))


def test_checkpoint_recovers_rng_schedules_and_selection_state(tmp_path):
    from src.helper import capture_rng_state
    from src.masks.curriculum import CurriculumMaskGenerator
    encoder, predictor = nn.Linear(2, 2), nn.Linear(2, 2)
    target = copy.deepcopy(encoder)
    opt = torch.optim.SGD(encoder.parameters(), lr=0.1, weight_decay=0.1)
    lr = WarmupCosineSchedule(opt, 2, 0.1, 0.2, 0.01, 10)
    wd = CosineWDSchedule(opt, 0.1, 0.2, 10)
    lr.step()
    wd.step()
    mask = CurriculumMaskGenerator(curriculum_cfg={'mode': 'loss_guided'})
    mask.set_epoch(27, 100)
    state = {'successful_updates': 1, 'best_val_loss': 0.4, 'epochs_no_improve': 3,
             'lr_scheduler': lr.state_dict(), 'wd_scheduler': wd.state_dict(),
             'lr': 0.1, 'wd': 0.1, 'ema': 0.99,
             'rank_states': [{'rng': capture_rng_state(), 'curriculum': mask.state_dict()}],
             'topology': {'world_size': 1, 'num_workers': 0}}
    path = str(tmp_path / 'checkpoint.pt')
    save_checkpoint(path, encoder, predictor, target, opt, None, 28, 0.5, 2, 1, 0.1,
                    mask_gen=mask, training_state=state)
    assert torch.load(path, weights_only=True)['epoch'] == 28
    expected = (random.random(), np.random.rand(), torch.rand(3))
    mask.set_epoch(0, 100)
    restored = {}
    loaded = load_checkpoint('cpu', path, encoder, predictor, target, opt, None,
                             mask_gen=mask, training_state=restored)
    got = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == got[:2]
    torch.testing.assert_close(expected[2], got[2])
    assert loaded[-1] == 28 and restored['epochs_no_improve'] == 3
    assert restored['best_val_loss'] == 0.4 and mask._epoch == 27
    lr.load_state_dict(restored['lr_scheduler'])
    wd.load_state_dict(restored['wd_scheduler'])
    assert lr._step == wd._step == 1
    del state['rank_states']
    # Legacy checkpoints remain loadable but must not claim exact continuation.
    save_checkpoint(path, encoder, predictor, target, opt, None, 28, 0.5, 2, 1, 0.1)
    with pytest.warns(UserWarning, match="not exact"):
        load_checkpoint('cpu', path, encoder, predictor, target, opt, None, training_state={})


def test_resume_reproduces_next_optimizer_update(tmp_path):
    from src.helper import capture_rng_state, optimizer_step
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    encoder, predictor = nn.Linear(2, 2), nn.Linear(2, 2)
    teacher = copy.deepcopy(encoder)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=0.01)

    def step():
        optimizer.zero_grad()
        x = torch.randn(3, 2) * random.random() + np.random.rand()
        predictor(encoder(x)).square().mean().backward()
        optimizer_step(optimizer)

    step()
    path = str(tmp_path / 'next-step.pt')
    save_checkpoint(path, encoder, predictor, teacher, optimizer, None, 1, 0.5, 3, 1, 0.01,
                    training_state={'rank_states': [{'rng': capture_rng_state()}]})
    step()
    expected = {key: value.clone() for key, value in encoder.state_dict().items()}
    load_checkpoint('cpu', path, encoder, predictor, teacher, optimizer, None, training_state={})
    step()
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_new_checkpoint_exact_rejects_mismatch_but_explicit_fork_retains_weights_optimizer(tmp_path):
    from src.helper import capture_rng_state

    class CurriculumState:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return {'epoch': self.epoch}

        def load_state_dict(self, state):
            self.epoch = state['epoch']

    encoder, predictor = nn.Linear(2, 2), nn.Linear(2, 2)
    teacher = copy.deepcopy(encoder)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=0.01)
    predictor(encoder(torch.ones(1, 2))).sum().backward()
    optimizer.step()
    scaler = torch.amp.GradScaler('cpu', init_scale=128)
    source = {
        'topology': {'policy': 'old', 'epochs': 100},
        'successful_updates': 50, 'best_val_loss': 0.1, 'epochs_no_improve': 4,
        'rank_states': [{'rng': capture_rng_state()}],
    }
    path = str(tmp_path / 'new-format.pt')
    save_checkpoint(path, encoder, predictor, teacher, optimizer, scaler, 9, 0.2, 2, 1, 0.01,
                    mask_gen=CurriculumState(9), training_state=source)
    expected = [copy.deepcopy(model.state_dict()) for model in (encoder, predictor, teacher)]
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    changed = {'policy': 'new', 'epochs': 200}
    with pytest.raises(ValueError, match="resume_policy.*fork"):
        load_checkpoint('cpu', path, encoder, predictor, teacher, optimizer, scaler,
                        training_state={}, topology=changed)
    fresh_curriculum = CurriculumState(0)
    fork_state = {}
    with torch.no_grad():
        for model in (encoder, predictor, teacher):
            for param in model.parameters():
                param.zero_()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=0.7)
    rng_before = torch.get_rng_state()
    scaler = torch.amp.GradScaler('cpu', init_scale=16)
    result = load_checkpoint('cpu', path, encoder, predictor, teacher, optimizer, scaler,
                             mask_gen=fresh_curriculum, training_state=fork_state,
                             topology=changed, resume_policy='fork')
    assert result[-1] == 9
    for model, wanted in zip((encoder, predictor, teacher), expected):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, wanted[key], rtol=0, atol=0)
    assert all(value['step'].item() == 1 for value in optimizer.state.values())
    for state, wanted in zip(optimizer.state.values(), expected_optimizer['state'].values()):
        torch.testing.assert_close(state['exp_avg'], wanted['exp_avg'], rtol=0, atol=0)
        torch.testing.assert_close(state['exp_avg_sq'], wanted['exp_avg_sq'], rtol=0, atol=0)
    assert scaler.get_scale() == 128
    assert fresh_curriculum.epoch == 0
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert 'best_val_loss' not in fork_state and 'successful_updates' not in fork_state
    assert fork_state['lineage']['source_epoch'] == 9
    assert len(fork_state['lineage']['source_checkpoint_sha256']) == 64


@pytest.mark.parametrize('workers', [0, 2])
def test_epoch_boundary_loader_replay_nonpersistent_workers(workers):
    from src.helper import capture_rng_state, restore_rng_state
    from torch.utils.data import DataLoader
    from src.masks.multiblock import MaskCollator
    # Nonpersistent workers are reconstructed from the saved parent RNG.
    images = torch.arange(4 * 3 * 256 * 256).reshape(4, 3, 256, 256).float()
    loader = DataLoader(list(images), batch_size=2, num_workers=workers,
                        shuffle=True, collate_fn=MaskCollator(), persistent_workers=False)
    state = capture_rng_state()
    first = list(loader)
    restore_rng_state(state)
    again = list(loader)
    for (a, ca, pa), (b, cb, pb) in zip(first, again):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for left, right in zip(ca + pa, cb + pb):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('skip_first_update', [False, True])
def test_production_loop_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, skip_first_update):
    from types import SimpleNamespace
    import yaml
    from src import train_patch as training
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(training, 'init_distributed', lambda: (1, 0))

    class TinyDataset:
        def __init__(self, data_dir, **kwargs):
            self.length = 7 if data_dir.endswith('Training') else 3
            self.file_paths = ['synthetic'] * self.length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return torch.rand(3, 16, 16) + np.random.rand() + random.random()

    monkeypatch.setattr(training, 'OCTSliceDataset', TinyDataset)
    monkeypatch.setattr(training, 'init_patch_model', lambda **kwargs: (
        VisionTransformer(img_size=16, patch_size=4, embed_dim=16, depth=1, num_heads=2),
        VisionTransformerPredictor(16, 16, 16, depth=1, num_heads=2)))
    monkeypatch.setattr(training, 'upload_to_blob', lambda *args, **kwargs: None)
    original_step = training.optimizer_step
    attempts = [0]

    def step_with_optional_skip(optimizer, scaler):
        attempts[0] += 1
        if skip_first_update and attempts[0] == 1:
            return False
        return original_step(optimizer, scaler)

    monkeypatch.setattr(training, 'optimizer_step', step_with_optional_skip)
    root = tmp_path / 'dataset'
    (root / 'Training').mkdir(parents=True)
    (root / 'Validation').mkdir()
    config = {
        'data': {'data_dir': str(root), 'crop_size': 16, 'batch_size': 1,
                 'num_workers': 0, 'val_num_workers': 0, 'num_slices': 1},
        'mask': {'patch_size': 4, 'enc_mask_scale': [0.8, 1.0],
                 'pred_mask_scale': [0.1, 0.2], 'aspect_ratio': [0.8, 1.2],
                 'num_enc_masks': 1, 'num_pred_masks': 2, 'min_keep': 1, 'allow_overlap': False},
        'meta': {'model_name': 'vit_tiny', 'pred_depth': 1, 'pred_emb_dim': 16,
                 'seed': 19, 'use_bfloat16': True},
        'optimization': {'epochs': 2, 'accum_steps': 4, 'weight_decay': 0.04,
                         'final_weight_decay': 0.4, 'start_lr': 0.001, 'lr': 0.002,
                         'final_lr': 0.0001, 'warmup': 0, 'ema': [0.9, 0.99],
                         'patience': 10, 'save_every': 1},
        'logging': {'folder': str(tmp_path / 'full'), 'write_tag': 'test'},
    }
    config_path = tmp_path / 'config.yaml'

    def run():
        config_path.write_text(yaml.safe_dump(config))
        training.main(SimpleNamespace(config=str(config_path)))

    run()
    full = torch.load(tmp_path / 'full' / 'test-last.pth.tar', weights_only=True)
    assert full['training_state']['successful_updates'] == 4 - int(skip_first_update)
    assert full['training_state']['lr_scheduler']['_step'] == 4 - int(skip_first_update)
    attempts[0] = 0
    config['logging']['folder'] = str(tmp_path / 'resumed')

    def interrupt(path, *args, **kwargs):
        if str(path).endswith('test-ep1.pth.tar'):
            raise InterruptedError('test epoch boundary')

    monkeypatch.setattr(training, 'upload_to_blob', interrupt)
    with pytest.raises(InterruptedError, match="epoch boundary"):
        run()
    boundary = torch.load(tmp_path / 'resumed' / 'test-last.pth.tar', weights_only=True)
    monkeypatch.setattr(training, 'upload_to_blob', lambda *args, **kwargs: None)
    config['meta'].update({'load_checkpoint': True,
                           'read_checkpoint': str(tmp_path / 'resumed' / 'test-last.pth.tar')})
    run()
    resumed = torch.load(tmp_path / 'resumed' / 'test-last.pth.tar', weights_only=True)
    for model in ('encoder', 'predictor', 'target_encoder'):
        for key in full[model]:
            torch.testing.assert_close(full[model][key], resumed[model][key], rtol=0, atol=0)
    for key in ('successful_updates', 'best_val_loss', 'epochs_no_improve',
                'lr_scheduler', 'wd_scheduler', 'lr', 'wd', 'ema'):
        assert full['training_state'][key] == resumed['training_state'][key]
    import csv
    with open(tmp_path / 'full' / 'test-log.csv') as stream:
        full_rows = list(csv.DictReader(stream))
    with open(tmp_path / 'resumed' / 'test-log.csv') as stream:
        resumed_rows = list(csv.DictReader(stream))
    for expected, actual in zip(full_rows, resumed_rows):
        for key in ('epoch', 'iteration', 'loss', 'lr', 'wd', 'ema'):
            assert expected[key] == actual[key]
    boundary['training_state']['epochs_no_improve'] = config['optimization']['patience']
    exhausted_path = tmp_path / 'exhausted.pt'
    torch.save(boundary, exhausted_path)
    config['meta']['read_checkpoint'] = str(exhausted_path)
    before_attempts = attempts[0]
    run()
    assert attempts[0] == before_attempts
    # A declared new-format fork retains full learned state but deliberately
    # starts fresh schedules/selection/curriculum under its new run contract.
    config['logging']['folder'] = str(tmp_path / 'fork')
    config['optimization'].update({'epochs': 1, 'warmup': 0})
    config['mask']['pred_mask_scale'] = [0.2, 0.3]
    config['mask']['curriculum'] = {
        'enabled': True, 'mode': 'intensity_foreground', 'T_warm': 0,
        'T_total': 1, 'r_max': 1.0}
    config['meta'].update({'resume_policy': 'exact', 'read_checkpoint': str(exhausted_path)})
    with pytest.raises(ValueError, match="resume_policy.*fork"):
        run()
    config['meta'].update({'resume_policy': 'fork', 'fork_start_epoch': 0})
    run()
    forked = torch.load(tmp_path / 'fork' / 'test-last.pth.tar', weights_only=True)
    assert forked['epoch'] == 1
    assert forked['training_state']['successful_updates'] == 2
    assert forked['training_state']['epochs_no_improve'] == 0
    assert forked['training_state']['lineage']['source_epoch'] == 1
    assert forked['training_state']['lineage']['fork_start_epoch'] == 0
    assert forked['training_state']['rank_states'][0]['curriculum']['epoch'] == 0
    with open(tmp_path / 'fork' / 'test-log.csv') as stream:
        fork_rows = list(csv.DictReader(stream))
    assert float(fork_rows[0]['lr']) == config['optimization']['start_lr']
