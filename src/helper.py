"""
Model initialization, optimizer setup, and checkpoint management for I-JEPA.

Provides factory functions for both patch-level and slice-level I-JEPA
encoders/predictors, as well as the frozen ConvNeXt feature extractor
used in the slice-level approach.  Delegates to the model definitions in
``src.models.vision_transformer`` and ``src.models.feature_extractor``.

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import copy
import gc
import hashlib
import json
import os
import random
import subprocess
import time
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
                    resume_policy='exact', source_hash=None, allow_source_change=False,
                    checkpoint_sha256=None):
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
        source_hash: Combined SHA-256 of the executing ``src/`` tree. On exact
            resume of a checkpoint that recorded one, a different value is
            refused unless ``allow_source_change`` is set.
        allow_source_change: Permit exact resume across a source-hash change.
        checkpoint_sha256: Precomputed digest of ``r_path`` for fork lineage
            (computed here when omitted).

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
    if resume_policy == 'exact' and resume is not None and source_hash is not None:
        stored = (resume.get('source_identity') or {}).get('source_hash')
        if stored is None:
            warnings.warn("Checkpoint %s predates source-hash tracking; the code that "
                          "produced it cannot be compared with the current source." % r_path)
        elif stored != source_hash:
            if not allow_source_change:
                raise ValueError(
                    "Exact resume refused: src/ source hash changed since %s was written "
                    "(checkpoint %s, current %s). Run the checkpoint's code, or pass "
                    "--allow-source-change to continue deliberately under new code."
                    % (r_path, stored, source_hash))
            warnings.warn("Exact resume under CHANGED source (checkpoint %s, current %s); "
                          "allowed by --allow-source-change." % (stored, source_hash))

    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    opt.load_state_dict(checkpoint['opt'])

    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    if resume_policy == 'fork':
        if checkpoint_sha256 is None:
            checkpoint_sha256 = file_sha256(r_path)
        if training_state is not None:
            training_state.clear()
            training_state['lineage'] = {
                'resume_policy': 'fork', 'source_checkpoint': os.path.realpath(r_path),
                'source_checkpoint_sha256': checkpoint_sha256,
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

    The file is written to ``path + '.tmp'``, fsynced and moved into place, so a
    crash mid-write never leaves a truncated file under the final name.
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
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Provenance helpers: hashing, atomic writes, checkpoint verification
# ---------------------------------------------------------------------------

REQUIRED_CHECKPOINT_KEYS = ('encoder', 'predictor', 'target_encoder', 'opt', 'scaler',
                            'epoch', 'training_state')
REQUIRED_TRAINING_STATE_KEYS = ('successful_updates', 'lr_scheduler', 'wd_scheduler',
                                'topology', 'rank_states')


class CheckpointVerificationError(RuntimeError):
    """A saved checkpoint failed its post-save re-read."""


def replace_with_retry(src, dst, attempts=20, delay=0.5):
    """os.replace, retried briefly: on Windows a reader (e.g. a virus scanner)
    holding ``dst`` open makes the rename fail transiently."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def file_sha256(path, block=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return str(value)


def atomic_write_json(path, payload):
    """Write JSON to a unique temp file, fsync it, then move it into place."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = '%s.tmp.%d' % (path, os.getpid())
    try:
        with open(tmp_path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def default_sha256_cache_path():
    return (os.environ.get('JEPA_SHA256_CACHE')
            or os.path.join(os.path.expanduser('~'), '.cache', 'jepa', 'sha256_cache.json'))


def file_sha256_cached(path, cache_path=None):
    """SHA-256 of ``path``, reused from a cache keyed by realpath+size+mtime_ns.

    Returns ``(sha256, source)`` with source ``'cache'`` or ``'computed'``.
    Cache read/write failures only cost a re-hash.
    """
    cache_path = cache_path or default_sha256_cache_path()
    key = os.path.realpath(path)
    before = os.stat(path)
    try:
        with open(cache_path, 'r', encoding='utf-8') as stream:
            cache = json.load(stream)
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(key)
    if (isinstance(entry, dict) and entry.get('bytes') == before.st_size
            and entry.get('mtime_ns') == before.st_mtime_ns and entry.get('sha256')):
        return entry['sha256'], 'cache'
    sha = file_sha256(path)
    after = os.stat(path)
    if (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns):
        cache[key] = {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': sha}
        try:
            atomic_write_json(cache_path, cache)
        except OSError:
            pass
    return sha, 'computed'


def model_state_digest(encoder_state, predictor_state, target_state):
    """SHA-256 over the three model state dicts (names, dtypes, shapes, bytes)."""
    digest = hashlib.sha256()
    for name, state in (('encoder', encoder_state), ('predictor', predictor_state),
                        ('target_encoder', target_state)):
        for key in sorted(state):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(('%s.%s|%s|%s\n' % (name, key, tensor.dtype, tuple(tensor.shape)))
                          .encode('utf-8'))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def models_state_digest(encoder, predictor, target_encoder):
    unwrap = lambda model: model.module if hasattr(model, 'module') else model
    return model_state_digest(unwrap(encoder).state_dict(), unwrap(predictor).state_dict(),
                              unwrap(target_encoder).state_dict())


def verify_checkpoint_file(path, expected_epoch, expected_successful_updates=None,
                           expected_identity=None, expected_model_digest=None):
    """Hash ``path``, re-read it on CPU and check it is a complete epoch checkpoint.

    ``expected_identity`` maps training_state keys (e.g. run_uuid, topology,
    source_identity) to required values; ``expected_model_digest`` must equal the
    file's ``model_state_digest``. Raises CheckpointVerificationError on any
    mismatch; returns ``{'path', 'file', 'bytes', 'sha256', 'epoch', 'successful_updates'}``.
    """
    try:
        size = os.path.getsize(path)
        sha = file_sha256(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:
        raise CheckpointVerificationError("Cannot re-read checkpoint %s: %r" % (path, error))
    try:
        if not isinstance(checkpoint, dict):
            raise CheckpointVerificationError("Checkpoint %s is not a dict" % path)
        missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
        if missing:
            raise CheckpointVerificationError("Checkpoint %s lacks keys %s" % (path, missing))
        empty = [key for key in ('encoder', 'predictor', 'target_encoder')
                 if not checkpoint[key]]
        if empty or not checkpoint['opt'].get('param_groups'):
            raise CheckpointVerificationError("Checkpoint %s has empty model/optimizer state %s"
                                              % (path, empty))
        if int(checkpoint['epoch']) != int(expected_epoch):
            raise CheckpointVerificationError(
                "Checkpoint %s holds epoch %s, expected %d"
                % (path, checkpoint['epoch'], expected_epoch))
        state = checkpoint['training_state']
        if not isinstance(state, dict):
            raise CheckpointVerificationError("Checkpoint %s lacks training_state" % path)
        missing = [key for key in REQUIRED_TRAINING_STATE_KEYS if key not in state]
        if missing:
            raise CheckpointVerificationError(
                "Checkpoint %s training_state lacks %s" % (path, missing))
        updates = int(state['successful_updates'])
        if expected_successful_updates is not None and updates != int(expected_successful_updates):
            raise CheckpointVerificationError(
                "Checkpoint %s holds %d successful updates, expected %d"
                % (path, updates, expected_successful_updates))
        for key, value in (expected_identity or {}).items():
            if state.get(key) != value:
                raise CheckpointVerificationError(
                    "Checkpoint %s training_state.%s differs from the run being certified "
                    "(%r != %r)" % (path, key, state.get(key), value))
        if expected_model_digest is not None:
            digest = model_state_digest(checkpoint['encoder'], checkpoint['predictor'],
                                        checkpoint['target_encoder'])
            if digest != expected_model_digest:
                raise CheckpointVerificationError(
                    "Checkpoint %s model weights differ from the run's epoch-%d state"
                    % (path, expected_epoch))
    finally:
        del checkpoint
        gc.collect()
    if os.path.getsize(path) != size:
        raise CheckpointVerificationError("Checkpoint %s changed while being verified" % path)
    return {'path': os.path.abspath(path), 'file': os.path.basename(path), 'bytes': size,
            'sha256': sha, 'epoch': int(expected_epoch), 'successful_updates': updates}


def read_checkpoint_epoch(path):
    """Saved epoch of ``path``; CheckpointVerificationError if it cannot be read."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return int(checkpoint['epoch'])
    except Exception as error:
        raise CheckpointVerificationError("Cannot re-read checkpoint %s: %r" % (path, error))
    finally:
        checkpoint = None
        gc.collect()


def _git(code_root, *args):
    result = subprocess.run(['git', '--no-optional-locks', '-C', code_root] + list(args),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace').strip())
    return result.stdout.decode('utf-8', 'replace')


def collect_source_identity(code_root, subdir='src'):
    """Per-file and combined SHA-256 of the tracked ``*.py`` under ``code_root/subdir``.

    Files are listed with ``git ls-files`` from ``code_root`` (which may be a git
    worktree) and hashed as they are on disk, so uncommitted edits change the
    hash. Without git, every ``*.py`` found on disk is hashed instead and the
    listing method is recorded.
    """
    listing = 'git_ls_files'
    try:
        names = [name for name in _git(code_root, 'ls-files', '-z', '--', subdir).split('\0')
                 if name.endswith('.py')]
        if not names:
            raise RuntimeError('no tracked files')
    except Exception:
        listing = 'filesystem_walk'
        names = []
        for root, dirs, files in os.walk(os.path.join(code_root, subdir)):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            names.extend(os.path.relpath(os.path.join(root, f), code_root).replace(os.sep, '/')
                         for f in files if f.endswith('.py'))
    files = {}
    for name in sorted(set(names)):
        path = os.path.join(code_root, *name.split('/'))
        files[name] = file_sha256(path) if os.path.isfile(path) else 'MISSING'
    combined = hashlib.sha256(''.join('%s\0%s\n' % item for item in sorted(files.items()))
                              .encode('utf-8')).hexdigest()
    return {'code_root': os.path.realpath(code_root), 'subdir': subdir, 'listing': listing,
            'source_hash': combined, 'files': files}


def collect_git_identity(code_root, subdir='src'):
    """Commit, branch and dirty state of the checkout that holds ``code_root``."""
    try:
        commit = _git(code_root, 'rev-parse', 'HEAD').strip()
        branch = _git(code_root, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
        dirty = [line for line in _git(code_root, 'status', '--porcelain',
                                       '--untracked-files=no').splitlines() if line.strip()]
        untracked = [name for name in _git(code_root, 'ls-files', '--others',
                                           '--exclude-standard', '--', subdir).splitlines()
                     if name.endswith('.py')]
        return {'git_commit': commit, 'git_branch': branch, 'git_dirty': bool(dirty),
                'git_dirty_paths': dirty[:200], 'untracked_src_py': untracked[:200],
                'error': None}
    except Exception as error:
        return {'git_commit': None, 'git_branch': None, 'git_dirty': None,
                'git_dirty_paths': [], 'untracked_src_py': [], 'error': repr(error)}


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
