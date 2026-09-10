"""
Model initialization, optimizer setup, and checkpoint management for I-JEPA.

Provides factory functions for both patch-level and slice-level I-JEPA
encoders/predictors, as well as the frozen ConvNeXt feature extractor
used in the slice-level approach.  Delegates to the model definitions in
``src.models.vision_transformer`` and ``src.models.feature_extractor``.

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import copy
import hashlib
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

from src.models.vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
    SliceEncoder,
    SlicePredictor,
    VIT_EMBED_DIMS,
    vit_base,
    vit_predictor,
    slice_encoder,
    slice_predictor,
)
try:
    from src.models.feature_extractor import FrozenFeatureExtractor
except ImportError:
    FrozenFeatureExtractor = None  # Slice-level approach (archived)
from src.utils.tensors import trunc_normal_
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule


# ---------------------------------------------------------------------------
# ViT model configs (mirrors VIT_EMBED_DIMS for convenience)
# ---------------------------------------------------------------------------

_VIT_CONFIGS = {
    'vit_tiny':  dict(embed_dim=192,  depth=12, num_heads=3),
    'vit_small': dict(embed_dim=384,  depth=12, num_heads=6),
    'vit_base':  dict(embed_dim=768,  depth=12, num_heads=12),
    'vit_large': dict(embed_dim=1024, depth=24, num_heads=16),
    'vit_huge':  dict(embed_dim=1280, depth=32, num_heads=16),
}


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def init_patch_model(device, patch_size=16, crop_size=256, model_name='vit_base',
                     pred_depth=6, pred_emb_dim=384):
    """Initialize encoder + predictor for patch-level I-JEPA.

    Args:
        device: Target torch device.
        patch_size: Patch size for the ViT.
        crop_size: Input image spatial resolution.
        model_name: One of 'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge'.
        pred_depth: Number of transformer blocks in the predictor.
        pred_emb_dim: Hidden dimension of the predictor.

    Returns:
        (encoder, predictor) tuple on *device*.
    """
    cfg = _VIT_CONFIGS[model_name]

    encoder = VisionTransformer(
        img_size=crop_size,
        patch_size=patch_size,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
    )

    num_patches = encoder.patch_embed.num_patches

    predictor = VisionTransformerPredictor(
        num_patches=num_patches,
        embed_dim=cfg['embed_dim'],
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=cfg['num_heads'],
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    return encoder, predictor


def init_slice_model(device, num_slices=32, embed_dim=768, enc_depth=6,
                     pred_depth=6, pred_emb_dim=384, num_heads=12):
    """Initialize encoder + predictor for slice-level I-JEPA.

    Args:
        device: Target torch device.
        num_slices: Number of slice tokens.
        embed_dim: Embedding dimension.
        enc_depth: Number of transformer blocks in the encoder.
        pred_depth: Number of transformer blocks in the predictor.
        pred_emb_dim: Hidden dimension of the predictor.
        num_heads: Number of attention heads.

    Returns:
        (encoder, predictor) tuple on *device*.
    """
    encoder = SliceEncoder(
        num_slices=num_slices,
        embed_dim=embed_dim,
        depth=enc_depth,
        num_heads=num_heads,
    )

    predictor = SlicePredictor(
        num_slices=num_slices,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads,
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    return encoder, predictor


def init_feature_extractor(device, checkpoint_path=None, freeze=True):
    """Initialize a ConvNeXt feature extractor for the slice-level approach.

    Args:
        device: Target torch device.
        checkpoint_path: Optional path to SLIViT pretrained ConvNeXt weights.
        freeze: If True, all params are frozen. If False, params are trainable
                (use a low LR like 1e-6).

    Returns:
        FrozenFeatureExtractor on *device*.
    """
    fe = FrozenFeatureExtractor(checkpoint_path=checkpoint_path, freeze=freeze)
    fe = fe.to(device)
    return fe


def init_opt(encoder, predictor, wd, final_wd, start_lr, ref_lr, final_lr,
             iterations_per_epoch, warmup, num_epochs, ipe_scale=1.0,
             use_bfloat16=False, feature_extractor=None, fe_lr=None):
    """Initialize AdamW optimizer with warmup cosine LR and cosine WD schedules.

    Creates parameter groups for encoder, predictor, and optionally a
    feature extractor (with its own learning rate).

    Args:
        encoder: The encoder model.
        predictor: The predictor model.
        wd: Reference weight decay.
        final_wd: Final weight decay at end of schedule.
        start_lr: Learning rate at iteration 0.
        ref_lr: Peak learning rate.
        final_lr: Minimum learning rate.
        iterations_per_epoch: Number of training iterations per epoch.
        warmup: Number of warmup epochs.
        num_epochs: Total number of training epochs.
        ipe_scale: Scale factor for iterations per epoch.
        use_bfloat16: Whether to use bfloat16 mixed precision.
        feature_extractor: Optional unfrozen feature extractor to include
            in the optimizer with a separate learning rate.
        fe_lr: Learning rate for the feature extractor (e.g., 1e-6).

    Returns:
        (optimizer, scaler, lr_scheduler, wd_scheduler)
    """
    # Separate parameters that should and should not get weight decay
    enc_wd_params, enc_no_wd_params = _split_wd_params(encoder)
    pred_wd_params, pred_no_wd_params = _split_wd_params(predictor)

    param_groups = [
        {'params': enc_wd_params, 'weight_decay': wd},
        {'params': pred_wd_params, 'weight_decay': wd},
        {'params': enc_no_wd_params, 'weight_decay': 0.0},
        {'params': pred_no_wd_params, 'weight_decay': 0.0},
    ]

    # Add feature extractor params with its own LR
    if feature_extractor is not None and fe_lr is not None:
        fe_wd_params, fe_no_wd_params = _split_wd_params(feature_extractor)
        if fe_wd_params:
            param_groups.append({'params': fe_wd_params, 'weight_decay': wd, 'lr': fe_lr})
        if fe_no_wd_params:
            param_groups.append({'params': fe_no_wd_params, 'weight_decay': 0.0, 'lr': fe_lr})

    optimizer = torch.optim.AdamW(param_groups, lr=start_lr)

    ipe = int(iterations_per_epoch * ipe_scale)
    T_max = int(num_epochs * ipe)
    warmup_steps = int(warmup * ipe)

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=T_max,
    )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=T_max,
    )

    # GradScaler for mixed precision; disabled if using bfloat16 or CPU
    if use_bfloat16:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler()

    return optimizer, scaler, lr_scheduler, wd_scheduler


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def optimizer_step(optimizer, scaler=None):
    """Return whether an optimizer update occurred (including AMP overflow)."""
    if scaler is None:
        optimizer.step()
        return True
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= old_scale


@torch.no_grad()
def update_ema(encoder, target_encoder, momentum):
    encoder = encoder.module if hasattr(encoder, 'module') else encoder
    for online, target in zip(encoder.parameters(), target_encoder.parameters()):
        target.mul_(momentum).add_((1.0 - momentum) * online.detach())


def capture_rng_state():
    """Capture the current rank's main-process RNGs, not DataLoader workers."""
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def load_checkpoint(device, r_path, encoder, predictor, target_encoder, opt, scaler,
                    mask_gen=None, training_state=None, rank=0, topology=None,
                    resume_policy='exact'):
    """Load a training checkpoint.

    Args:
        device: Device to map tensors to.
        r_path: Path to the checkpoint file.
        encoder: Encoder model (state_dict will be loaded in-place).
        predictor: Predictor model (state_dict will be loaded in-place).
        target_encoder: Target (EMA) encoder model.
        opt: Optimizer.
        scaler: GradScaler (may be None).
        mask_gen: Optional curriculum mask generator with
            ``state_dict``/``load_state_dict`` methods.  When set, the
            ``curriculum`` block in the checkpoint is restored into it.
            Missing or ``None`` is fine — fresh resume from R1 baseline
            won't carry curriculum state.
        training_state: Optional output dict for successful-step schedules,
            model-selection counters and rank-local epoch-boundary RNGs.
            Passing a dict opts into restoration and legacy-state warnings.
        rank: Current rank, selecting its saved RNG/curriculum state.
        topology: Optional exact run/worker contract to check before restoration.
        resume_policy: ``exact`` (default) restores continuation state and rejects
            changed contracts. ``fork`` retains all three models and optimizer/
            scaler, but does not restore RNG/curriculum/scheduler/selection state.
            The caller deliberately reconstructs those from the new run config.

    Global RNG restoration does not restore persistent workers or an in-flight
    prefetched iterator. The trainer saves only completed-epoch boundaries with
    nonpersistent workers. Changed topology/config is a new run, not exact resume.

    Returns:
        (encoder, predictor, target_encoder, opt, scaler, start_epoch)
    """
    if resume_policy not in ('exact', 'fork'):
        raise ValueError("resume_policy must be 'exact' or 'fork'")
    # These are trusted local training checkpoints, including Python/NumPy RNG.
    checkpoint = torch.load(r_path, map_location=device, weights_only=False)
    resume = checkpoint.get('training_state')
    if (resume_policy == 'exact' and resume is not None and topology is not None
            and resume.get('topology') != topology):
        raise ValueError("Resume worker/rank/batch/config contract differs from checkpoint; "
                         "set meta.resume_policy='fork' for an intentional new run. "
                         "Exact continuation refuses this mismatch.")

    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    opt.load_state_dict(checkpoint['opt'])

    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    if resume_policy == 'fork':
        digest = hashlib.sha256()
        with open(r_path, 'rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if training_state is not None:
            training_state.clear()
            training_state['lineage'] = {
                'resume_policy': 'fork', 'source_checkpoint': os.path.realpath(r_path),
                'source_checkpoint_sha256': digest.hexdigest(),
                'source_epoch': checkpoint.get('epoch', 0),
                'source_topology': resume.get('topology') if resume is not None else None,
                'retained': ['encoder', 'predictor', 'target_encoder', 'optimizer_state',
                             'scaler_when_present'],
                'reset': ['rng_from_config_seed', 'curriculum', 'schedules_from_new_config',
                          'best_val_loss', 'epochs_no_improve'],
            }
        print("[Checkpoint] Explicit FORK from %s (source epoch %d): retained "
              "encoder/predictor/teacher, optimizer and available scaler; "
              "RNG/curriculum/schedules/best/patience NOT restored."
              % (r_path, checkpoint.get('epoch', 0)))
        return encoder, predictor, target_encoder, opt, scaler, checkpoint.get('epoch', 0)

    if training_state is not None:
        training_state.clear()
        if resume is None:
            warnings.warn("Legacy checkpoint lacks RNG/scheduler/best/patience state; "
                          "resume is not exact. Schedules are reconstructed from epoch.")
        else:
            training_state.update(resume)
            rank_states = resume.get('rank_states', [])
            if rank >= len(rank_states) or 'rng' not in rank_states[rank]:
                warnings.warn("Checkpoint lacks this rank's RNG state; resume is not exact.")
            else:
                restore_rng_state(rank_states[rank]['rng'])
                if mask_gen is not None and rank_states[rank].get('curriculum') is not None:
                    checkpoint['curriculum'] = rank_states[rank]['curriculum']

    if mask_gen is not None:
        if checkpoint.get('curriculum') is not None:
            # The checkpoint has curriculum state — restoring is REQUIRED;
            # a failure here is fatal (silently cold-starting would secretly
            # change the experiment).  Only the legitimate "no curriculum
            # key" case (R1 checkpoint) is allowed to no-op.
            try:
                mask_gen.load_state_dict(checkpoint['curriculum'])
                print("[Checkpoint] Restored curriculum state from %s" % r_path)
            except (KeyError, RuntimeError) as e:
                raise RuntimeError(
                    "Failed to restore curriculum state from %s: %s — "
                    "refusing to silently cold-start, which would change "
                    "the experiment.  Either fix the checkpoint or pass "
                    "mask_gen=None to deliberately discard the state."
                    % (r_path, e)
                )
        else:
            # No curriculum key — typical when resuming from R1.  This is
            # expected and benign; just log it.
            warnings.warn("No curriculum state in %s; starting curriculum from scratch. "
                          "This is a new curriculum branch, not exact resume." % r_path)

    start_epoch = checkpoint.get('epoch', 0)
    print("[Checkpoint] Loaded from %s  (epoch %d)" % (r_path, start_epoch))
    return encoder, predictor, target_encoder, opt, scaler, start_epoch


def save_checkpoint(path, encoder, predictor, target_encoder, optimizer,
                    scaler, epoch, loss, batch_size, world_size, lr,
                    mask_gen=None, training_state=None):
    """Save a training checkpoint.

    Args:
        path: File path to write.
        encoder: Encoder model (or DDP-wrapped).
        predictor: Predictor model (or DDP-wrapped).
        target_encoder: Target (EMA) encoder.
        optimizer: Optimizer.
        scaler: GradScaler (may be None).
        epoch: Current epoch number.
        loss: Last training loss value.
        batch_size: Per-GPU batch size.
        world_size: Number of distributed processes.
        lr: Current learning rate.
        mask_gen: Optional curriculum mask generator — if provided and it
            exposes a ``state_dict``, the dict is stored under ``curriculum``
            so an AML preempt + resume restores loss-map / cluster state.
        training_state: Optional complete epoch-boundary state assembled by the
            trainer on all ranks. Omitting it preserves the legacy file schema
            fields but cannot provide exact RNG/scheduler/selection continuation.
    """
    enc_state = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
    pred_state = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
    te_state = target_encoder.module.state_dict() if hasattr(target_encoder, 'module') else target_encoder.state_dict()

    state = {
        'encoder': enc_state,
        'predictor': pred_state,
        'target_encoder': te_state,
        'opt': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'loss': loss,
        'batch_size': batch_size,
        'world_size': world_size,
        'lr': lr,
        'training_state': training_state,
        'curriculum': (
            mask_gen.state_dict()
            if (mask_gen is not None and hasattr(mask_gen, 'state_dict'))
            else None
        ),
    }

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_wd_params(model):
    """Split model parameters into those that should receive weight decay and those that should not.

    Bias parameters and LayerNorm parameters are excluded from weight decay.

    Returns:
        (wd_params, no_wd_params): Two lists of parameter tensors.
    """
    wd_params = []
    no_wd_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or 'bias' in name or 'norm' in name.lower():
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return wd_params, no_wd_params
