"""Bounded production JEPA layer/loss/EMA replay; never a pretraining runner."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.datasets.oct_slices import OCTSliceDataset
from src.eval_downstream import file_sha256, imagenet_normalize
from src.helper import init_patch_model, optimizer_step, update_ema
from src.masks.curriculum import CurriculumMaskGenerator
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from src.train_patch import jepa_forward_loss
from src.utils.tensors import repeat_interleave_batch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_diagnostic_cases(path, config):
    if config.get('fixtures_sha256') and file_sha256(path) != config['fixtures_sha256']:
        raise ValueError("Frozen mask fixture digest mismatch")
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if 'cases' in payload:
        return payload['cases']
    if not payload.get('private_real_data') or 'batches' not in payload:
        raise ValueError("Unknown frozen mask fixture schema")
    batch = payload['batches']['bs2']
    cases = []
    for policy in config['fixture_policies']:
        masks = batch['policies'][policy]
        cases.append({
            'label': policy, 'images': batch['images'],
            'masks_enc': masks['masks_enc'], 'masks_pred': masks['masks_pred'],
            'tissue': batch['tissue_labels'].flatten(1),
            'guide_valid': batch['guide_valid'], 'target_sources': masks['target_sources'],
        })
    return cases


def make_cases(config, private):
    dataset = OCTSliceDataset(config['data_dir'], num_slices=config['num_slices'],
                              slice_size=config['model']['crop_size'],
                              slice_cache=config.get('slice_cache'))
    indices = config['slice_indices']
    images = imagenet_normalize(torch.stack([dataset[index] for index in indices]))
    write_json(private / 'input_manifest.json', {
        'split': 'Training', 'preprocessing': 'fixed full-resize PIL bilinear/ImageNet; no random crop',
        'samples': [{'file': dataset.file_paths[index // dataset.num_slices],
                     'depth': int(dataset.slice_indices[index % dataset.num_slices])}
                    for index in indices], 'tensor_sha256': tensor_sha256(images)})
    cases = []
    for batch in config['batches']:
        for policy in config['policies']:
            random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            selected = images[:batch]
            kwargs = dict(nenc=config.get('nenc', 1), npred=config.get('npred', 4))
            if policy == 'uniform':
                _, context, targets = MaskCollator(**kwargs)(list(selected))
            elif policy == 'intensity_foreground':
                generator = CurriculumMaskGenerator(
                    **kwargs, curriculum_cfg={'mode': policy, 'T_warm': 0,
                                               'T_total': 100, 'r_max': 1.0})
                generator.set_epoch(100, 100)
                context, targets = generator.generate(batch, imgs_cpu=selected)
            else:
                raise ValueError("Unsupported built-in diagnostic policy")
            cases.append({'label': '%s_B%d' % (policy, batch), 'images': selected,
                          'masks_enc': context, 'masks_pred': targets})
    torch.save({'cases': cases}, private / 'fixed_cases.pt')
    return cases


def audit_frozen_fixture_cpu(path, output, tissue_threshold):
    """Reuse a synthetic handoff verbatim; no sampler calls or optimizer updates."""
    from scripts.delivered_mask_audit import validate_delivered
    fixture = torch.load(path, map_location='cpu', weights_only=True)
    if fixture['metadata'].get('source') != 'synthetic_coordinate_codes':
        raise ValueError("Public CPU fixture report accepts synthetic coordinate fixtures only")
    images = imagenet_normalize(fixture['images'])
    context, targets = fixture['masks_enc'], fixture['masks_pred']
    batch, nenc, npred = len(images), len(context), len(targets)
    validate_delivered(context, targets, batch_size=batch, nenc=nenc, npred=npred)
    if len({mask.shape[1] for mask in context}) != 1:
        raise ValueError("Unequal context group budgets cannot be concatenated")
    torch.manual_seed(20260904)
    encoder = VisionTransformer(img_size=256, patch_size=16, embed_dim=16,
                                depth=2, num_heads=2)
    predictor = VisionTransformerPredictor(256, 16, 16, depth=2, num_heads=2)
    teacher = copy.deepcopy(encoder).requires_grad_(False)
    loss, prediction, target = jepa_forward_loss(
        encoder, predictor, teacher, images, context, targets)
    slots = F.smooth_l1_loss(prediction, target, reduction='none').mean(-1)
    tissue_grid = (fixture['guides'][:, 0].flatten(1) >= tissue_threshold).float()
    tissue = apply_masks(tissue_grid.unsqueeze(-1), targets)
    tissue = repeat_interleave_batch(tissue, batch, nenc).squeeze(-1).bool()
    duplicate = torch.zeros(npred, batch, targets[0].shape[1], dtype=torch.bool)
    for b in range(batch):
        seen = set()
        for p, mask in enumerate(targets):
            for k, token in enumerate(mask[b].tolist()):
                duplicate[p, b, k] = token in seen
                seen.add(token)
    duplicate = duplicate[:, None].expand(-1, nenc, -1, -1).reshape_as(slots)
    per_target = slots.reshape(npred, nenc, batch, -1).mean((1, 2, 3))
    partitions = {}
    for name, selected in [('tissue', tissue), ('background', ~tissue),
                           ('first_occurrence', ~duplicate), ('duplicate', duplicate)]:
        partitions[name] = {'slots': int(selected.sum()),
                            'loss_sum': slots[selected].sum().item()}
    for left, right in [('tissue', 'background'), ('first_occurrence', 'duplicate')]:
        reconstructed = (partitions[left]['loss_sum'] + partitions[right]['loss_sum']) / slots.numel()
        assert abs(reconstructed - loss.item()) < 1e-6
    torch.testing.assert_close(per_target.mean(), loss, atol=1e-6, rtol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for model in (encoder, predictor) for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in teacher.parameters())
    result = {
        'status': 'passed', 'scope': 'synthetic CPU selected-token partition; not production-size GPU',
        'fixture_sha256': file_sha256(path), 'fixture_metadata': fixture['metadata'],
        'images_preprocessing': 'supplied [0,1] coordinates -> fixed ImageNet normalization',
        'tissue_definition': {'guide_channel': 0, 'threshold': tissue_threshold},
        'model': {'encoder_dim': 16, 'encoder_depth': 2, 'predictor_dim': 16, 'predictor_depth': 2},
        'batch': batch, 'nenc': nenc, 'npred': npred, 'loss': loss.item(),
        'slots': slots.numel(), 'per_target_loss': per_target.tolist(), 'partitions': partitions,
        'finite_online_predictor_gradients': True, 'teacher_gradients': False,
        'optimizer_updates': 0, 'masks_redrawn': False,
        'code_sha256': {name: file_sha256(name) for name in
                        ('scripts\\training_layer_diagnostic.py', 'scripts\\delivered_mask_audit.py',
                         'src\\train_patch.py', 'src\\models\\vision_transformer.py')},
    }
    write_json(output, result)
    return result


def joint_context_loss_summary(slots, context, targets, tissue_grid, duplicate,
                               guide_valid=None):
    """Joint delivered context and duplicate-weighted loss; rows are ordinals."""
    batch, nenc, npred = tissue_grid.shape[0], len(context), len(targets)
    tissue_grid = tissue_grid.to(slots.device).bool()
    tissue = apply_masks(tissue_grid.float().unsqueeze(-1), targets)
    tissue = repeat_interleave_batch(tissue, batch, nenc).squeeze(-1).bool()
    losses = slots.reshape(npred, nenc, batch, -1)
    tissue = tissue.reshape_as(losses)
    duplicate = duplicate.reshape_as(losses)
    rows = []
    for b in range(batch):
        values, is_tissue, is_duplicate = losses[:, :, b], tissue[:, :, b], duplicate[:, :, b]
        count = values.numel()
        partitions = {}
        for name, selected in (
                ('tissue', is_tissue), ('background', ~is_tissue),
                ('tissue_first', is_tissue & ~is_duplicate),
                ('tissue_repeat', is_tissue & is_duplicate),
                ('background_first', ~is_tissue & ~is_duplicate),
                ('background_repeat', ~is_tissue & is_duplicate)):
            n = int(selected.sum())
            total = values[selected].double().sum().item()
            partitions[name] = {
                'slots': n, 'loss_sum': total,
                'mean_loss_on_selected_slots': total / n if n else None,
                'contribution_to_image_scalar_loss': total / count,
            }
        scalar = values.double().mean().item()
        reconstructed = (partitions['tissue']['loss_sum'] + partitions['background']['loss_sum']) / count
        assert abs(reconstructed - scalar) < 1e-7
        tissue_cells = int(tissue_grid[b].sum())
        context_tissue = [int(tissue_grid[b, mask[b]].sum()) for mask in context]
        target_union = torch.cat([mask[b] for mask in targets]).unique()
        rows.append({
            'row_ordinal': b,
            'guide_valid': bool(guide_valid[b]) if guide_valid is not None else None,
            'tissue_cells_in_full_grid': tissue_cells,
            'context_cells_per_group': [mask.shape[1] for mask in context],
            'context_tissue_cells_per_group': context_tissue,
            'context_tissue_fraction_of_full_tissue_per_group':
                [n / tissue_cells if tissue_cells else None for n in context_tissue],
            'unique_target_cells': target_union.numel(),
            'unique_target_tissue_cells': int(tissue_grid[b, target_union].sum()),
            'loss_slots': count, 'scalar_loss': scalar, 'partitions': partitions,
            'per_target_mean_loss': values.double().mean((1, 2)).tolist(),
        })
    assert abs(sum(row['scalar_loss'] for row in rows) / batch - slots.double().mean().item()) < 1e-7
    return rows


def block_hooks(encoder, predictor, teacher, records, gradient_scale=1.0):
    handles = []
    for prefix, blocks in [('online', encoder.blocks), ('teacher', teacher.blocks),
                           ('predictor', predictor.predictor_blocks)]:
        for index, block in enumerate(blocks):
            name = '%s.%02d' % (prefix, index)

            def record(module, args, output, name=name):
                value = output.detach().float()
                assert torch.isfinite(value).all(), 'Nonfinite activation: ' + name
                item = records.setdefault(name, {'forward_calls': 0, 'backward_calls': 0})
                item.update({'forward_calls': item['forward_calls'] + 1,
                             'shape': list(value.shape),
                             'std': value.std(unbiased=False).item(),
                             'rms': value.square().mean().sqrt().item()})
                if output.requires_grad:
                    def backward(gradient):
                        assert torch.isfinite(gradient).all(), 'Nonfinite block gradient: ' + name
                        item['backward_calls'] += 1
                        item['gradient_rms'] = (
                            gradient.detach().float() / gradient_scale).square().mean().sqrt().item()
                    output.register_hook(backward)
            handles.append(block.register_forward_hook(record))
    return handles


def run_case(case, precision, config, states, models):
    encoder, predictor, teacher = models
    encoder.load_state_dict(states['encoder'])
    predictor.load_state_dict(states['predictor'])
    teacher.load_state_dict(states['target_encoder'])
    encoder.train()
    predictor.train()
    teacher.eval().requires_grad_(False)
    images = case['images'].to('cuda')
    context = [value.to('cuda') for value in case['masks_enc']]
    targets = [value.to('cuda') for value in case['masks_pred']]
    assert 1 <= len(images) <= 2 and len(context) >= 1 and len(targets) >= 1
    for c in context:
        for t in targets:
            assert not (c.unsqueeze(-1) == t.unsqueeze(-2)).any(), 'Context/target overlap'
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), **config['optimizer'])
    scaler = torch.cuda.amp.GradScaler(enabled=precision == 'amp', init_scale=128.0)
    records = {}
    handles = block_hooks(encoder, predictor, teacher, records, scaler.get_scale())
    started = time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        loss, prediction, target = jepa_forward_loss(
            encoder, predictor, teacher, images, context, targets,
            use_amp=precision == 'amp', amp_target=False)
        assert torch.isfinite(loss), 'Nonfinite loss'
        slots = F.smooth_l1_loss(prediction.float(), target, reduction='none').mean(-1)
        shaped = slots.reshape(len(targets), len(context), len(images), -1)
        per_target = shaped.mean((1, 2, 3))
        torch.testing.assert_close(loss.float(), per_target.mean(), atol=1e-6, rtol=1e-6)
        duplicate = torch.zeros(len(targets), len(images), targets[0].shape[1],
                                dtype=torch.bool, device='cuda')
        for b in range(len(images)):
            seen = set()
            for p, mask in enumerate(targets):
                for k, token in enumerate(mask[b].tolist()):
                    duplicate[p, b, k] = token in seen
                    seen.add(token)
        duplicate = duplicate[:, None].expand(-1, len(context), -1, -1).reshape_as(slots)
        duplicate_sum = slots[duplicate].sum().item()
        first_occurrence_sum = slots[~duplicate].sum().item()
        assert abs((duplicate_sum + first_occurrence_sum) / slots.numel() - loss.item()) < 1e-6
        if 'tissue' in case:
            tissue = apply_masks(case['tissue'].float().to('cuda').unsqueeze(-1), targets)
            tissue = repeat_interleave_batch(tissue, len(images), len(context)).squeeze(-1).bool()
            tissue_sum = slots[tissue].sum().item()
            background_sum = slots[~tissue].sum().item()
            assert abs((tissue_sum + background_sum) / slots.numel() - loss.item()) < 1e-6
            partition = {'tissue_slots': int(tissue.sum()), 'background_slots': int((~tissue).sum()),
                         'tissue_loss_sum': tissue_sum, 'background_loss_sum': background_sum}
            joint = joint_context_loss_summary(
                slots, context, targets, case['tissue'], duplicate, case.get('guide_valid'))
        else:
            partition = {'not_run': 'No segmentation tissue fixture; intensity prior is not tissue truth.'}
            joint = None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {}
        for prefix, model in [('online', encoder), ('predictor', predictor)]:
            norm_sq = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None and torch.isfinite(param.grad).all(), prefix + '.' + name
                    norm_sq += param.grad.detach().float().square().sum().item()
            gradients[prefix + '_norm'] = norm_sq ** 0.5
            assert gradients[prefix + '_norm'] > 0
        assert all(param.grad is None for param in teacher.parameters())
        representative = torch.cat([
            encoder.patch_embed.proj.weight.grad.detach().float().flatten().cpu(),
            predictor.predictor_proj.weight.grad.detach().float().flatten().cpu()])
        assert optimizer_step(optimizer, scaler), 'Unexpected overflow on bounded diagnostic'
        update_norms = {}
        for prefix, model, key in [('online', encoder, 'encoder'),
                                   ('predictor', predictor, 'predictor')]:
            update_norms[prefix] = sum(
                (value.detach().cpu() - states[key][name]).float().square().sum().item()
                for name, value in model.named_parameters()) ** 0.5
            assert update_norms[prefix] > 0
        update_ema(encoder, teacher, config['ema'])
        ema_error = 0.0
        for name, online in encoder.named_parameters():
            before = states['target_encoder'][name].to('cuda')
            expected = before.mul(config['ema']).add((1 - config['ema']) * online.detach())
            error = (dict(teacher.named_parameters())[name] - expected).abs().max().item()
            ema_error = max(ema_error, error)
        assert ema_error == 0.0, 'EMA differs from production formula'

        # Hold preprocessing fixed; perturb only patches absent from every
        # context. A full-image teacher is intentionally allowed to respond.
        visible = torch.zeros(len(images), encoder.num_patches, dtype=torch.bool, device='cuda')
        for mask in context:
            visible.scatter_(1, mask, True)
        side = int(encoder.num_patches ** 0.5)
        hidden = (~visible).reshape(len(images), 1, side, side)
        pixels = hidden.repeat_interleave(16, 2).repeat_interleave(16, 3)
        changed = images + pixels * 0.5
        with torch.no_grad():
            a = encoder(images, context)
            b = encoder(changed, context)
            online_error = (a - b).abs().max().item()
            teacher_effect = (teacher(images) - teacher(changed)).abs().max().item()
        assert online_error == 0 and teacher_effect > 0
        assert all(item['forward_calls'] > 0 and item['std'] > 0 for item in records.values())
        assert all(item['backward_calls'] == (0 if name.startswith('teacher') else 1)
                   for name, item in records.items())
        duplicate_slots = []
        for b in range(len(images)):
            ids = torch.cat([mask[b] for mask in targets])
            duplicate_slots.append(len(ids) - len(ids.unique()))
        result = {
            'case': case['label'], 'precision': precision, 'loss': loss.item(),
            'batch': len(images), 'nenc': len(context), 'npred': len(targets),
            'target_slots': slots.numel(), 'per_target_loss': per_target.tolist(),
            'duplicate_target_slots_per_image': duplicate_slots,
            'duplicate_loss_partition': {'repeat_loss_sum': duplicate_sum,
                                        'first_occurrence_loss_sum': first_occurrence_sum,
                                        'repeat_slots': int(duplicate.sum()),
                                        'total_slots': slots.numel()},
            'tissue_partition': partition, 'joint_context_loss': joint, 'gradient': gradients,
            'parameter_update_norm': update_norms, 'teacher_has_gradient': False,
            'ema_max_absolute_error': ema_error, 'masked_online_max_difference': online_error,
            'full_teacher_hidden_pixel_effect': teacher_effect,
            'blocks': records, 'elapsed_seconds': time.perf_counter() - started,
            'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'images_sha256': tensor_sha256(case['images']),
            'target_sources_by_image': case.get('target_sources'),
        }
        return result, representative
    finally:
        for handle in handles:
            handle.remove()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--fixtures', help='Optional trusted local .pt with cases list')
    parser.add_argument('--cpu-fixture', help='Validate a frozen synthetic mask handoff without GPU')
    parser.add_argument('--output', help='JSON report path for --cpu-fixture')
    parser.add_argument('--tissue-threshold', type=float, default=0.25)
    args = parser.parse_args()
    if args.cpu_fixture:
        if not args.output:
            parser.error('--cpu-fixture requires --output')
        result = audit_frozen_fixture_cpu(args.cpu_fixture, args.output, args.tissue_threshold)
        print('CPU fixture passed: %d selected slots, loss %.8f, no mask redraws or optimizer updates'
              % (result['slots'], result['loss']))
        return
    if not args.config:
        parser.error('--config is required for leased GPU diagnostics')
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if config.get('fixtures_required') and not args.fixtures:
        parser.error('This diagnostic requires the mask owner frozen --fixtures export')
    output, private = Path(config['output_dir']), Path(config['private_dir'])
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    lease_path = Path(config['lease'])
    lease = json.loads(lease_path.read_text(encoding='utf-8'))
    if lease.get('owner') != 'repair-training' or lease.get('state') != 'reserved':
        raise RuntimeError("Coordinator-visible reserved GPU lease required before launch")
    started = time.perf_counter()
    lease.update({'state': 'running', 'pid': os.getpid(), 'started_at': time.time()})
    write_json(lease_path, lease)
    results = []
    models = None
    try:
        digest = file_sha256(config['checkpoint'])
        if digest != config['checkpoint_sha256']:
            raise RuntimeError("Ancestor checkpoint digest mismatch")
        source_files = [Path(__file__), Path('src\\train_patch.py'), Path('src\\helper.py'),
                        Path('src\\models\\vision_transformer.py'), Path('src\\masks\\curriculum.py'),
                        Path('src\\masks\\multiblock.py')]
        manifest = {
            'pid': os.getpid(), 'config': config, 'config_sha256': file_sha256(args.config),
            'checkpoint_sha256': digest,
            'code_sha256': {str(path): file_sha256(path) for path in source_files},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu_preflight': subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                text=True), 'purpose': 'bounded production layer/loss/EMA diagnostic; no training campaign',
            'bounds': {'max_updates': config['max_updates'], 'max_batches': config['max_batches'],
                       'max_images_per_batch': 2, 'max_forward_calls_per_batch': 7,
                       'max_backward_calls_per_batch': 1},
            'stop_conditions': ['first nonfinite', 'contract failure', 'resource conflict', 'configured cap']}
        active_compute = manifest['gpu_preflight'].splitlines()
        conflicts = [line for line in active_compute if 'python' in line.lower()
                     and not line.strip().startswith(str(os.getpid()) + ',')]
        if conflicts:
            raise RuntimeError("Another Python GPU process is active; refusing diagnostic launch")
        manifest['amp_scaler_initial_scale'] = 128.0
        write_json(output / 'manifest.json', manifest)
        cases = (load_diagnostic_cases(args.fixtures, config)
                 if args.fixtures else make_cases(config, private))
        count = len(cases) * len(config['precisions'])
        if count > min(config['max_updates'], config['max_batches']):
            raise RuntimeError("Fixture list exceeds declared cap")
        if args.fixtures:
            manifest['fixtures_sha256'] = file_sha256(args.fixtures)
            write_json(output / 'manifest.json', manifest)
        if config.get('matched_guided_fixture'):
            if len(cases) != 3 or config['precisions'] != ['fp32']:
                raise ValueError("Guided follow-up is exactly three fp32 independent cases")
            for case in cases:
                if len(case['images']) != 2 or 'tissue' not in case or 'guide_valid' not in case:
                    raise ValueError("Guided fixture requires two images, tissue and validity")
                if not torch.equal(case['images'], cases[0]['images']):
                    raise ValueError("Guided policies do not share identical preprocessed images")
                if not torch.equal(case['tissue'], cases[0]['tissue']):
                    raise ValueError("Guided policies do not share identical tissue definition")
                if not torch.equal(case['guide_valid'], cases[0]['guide_valid']):
                    raise ValueError("Guided policies do not share identical guide validity")
                if [list(x.shape) for x in case['masks_pred']] != [
                        list(x.shape) for x in cases[0]['masks_pred']]:
                    raise ValueError("Guided target budgets are not matched")
            if [case['label'] for case in cases] != config['fixture_policies']:
                raise ValueError("Guided fixture policy order differs from declared selection")
            if not all(torch.equal(a, b) for a, b in
                       zip(cases[1]['masks_pred'], cases[2]['masks_pred'])):
                raise ValueError("Prefix-only and context-guard target tensors must be identical")
            if [list(x.shape) for x in cases[1]['masks_enc']] != [
                    list(x.shape) for x in cases[2]['masks_enc']]:
                raise ValueError("Context guard must retain prefix-only context token budget")
        states = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        states = {key: states[key] for key in ('encoder', 'predictor', 'target_encoder')}
        encoder, predictor = init_patch_model(torch.device('cuda'), **config['model'])
        teacher = copy.deepcopy(encoder).requires_grad_(False)
        models = encoder, predictor, teacher
        for case in cases:
            fp32_gradient = None
            fp32_loss = None
            for precision in config['precisions']:
                result, gradient = run_case(case, precision, config, states, models)
                torch.save(gradient, private / ('%s_%s_gradient.pt' % (case['label'], precision)))
                if precision == 'fp32':
                    fp32_gradient, fp32_loss = gradient, result['loss']
                elif fp32_gradient is not None:
                    result['fp32_comparison'] = {
                        'loss_absolute_difference': abs(result['loss'] - fp32_loss),
                        'representative_gradient_cosine': F.cosine_similarity(
                            gradient.double(), fp32_gradient.double(), dim=0).item(),
                        'representative_gradient_max_absolute_difference':
                            (gradient - fp32_gradient).abs().max().item()}
                results.append(result)
                write_json(output / 'results.json', results)
                print('%s %s loss=%.8f EMA error=%.1g blocks=%d' %
                      (result['case'], precision, result['loss'],
                       result['ema_max_absolute_error'], len(result['blocks'])), flush=True)
        write_json(output / 'verdict.json', {
            'status': 'passed', 'batches': len(results), 'optimizer_updates': len(results),
            'elapsed_seconds': time.perf_counter() - started,
            'overfit_test': 'omitted: finite active paths, actual parameter updates and correct EMA were decisive',
            'downstream_probe': 'omitted: no discrepancy requiring new probe; no AUC inference'})
        lease['state'] = 'completed'
    except BaseException as exc:
        lease['state'] = 'failed'
        write_json(output / 'verdict.json', {'status': 'failed', 'error': str(exc),
                                            'completed_batches': len(results)})
        raise
    finally:
        if models is not None:
            for model in models:
                model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        lease.update({'finished_at': time.time(), 'active_pid': None,
                      'completed_batches': len(results), 'hooks_released': True,
                      'sustained_pretraining': False})
        write_json(lease_path, lease)


if __name__ == '__main__':
    main()
