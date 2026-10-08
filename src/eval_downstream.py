"""
Downstream glaucoma classification using pretrained I-JEPA encoder.

Supports both patch-level and slice-level pretrained models:
  - Patch-level: each slice is encoded by the frozen ViT, mean-pooled to one
    token per slice, then the configured probe aggregates across slices.
    The canonical frozen protocol uses parameter-free MeanPool + LinearHead.
    Attentive probes remain available for historical-config compatibility.
  - Slice-level: slices are encoded by frozen ConvNeXt + frozen slice encoder,
    then mean-pooled and classified by a trainable MLP head.

Usage:
    # Canonical frozen MeanPool + LinearHead, fp32, verified source-aware cache
    python eval_downstream.py --config configs/downstream_frozen_meanpool_canonical.yaml

    # Slice-level pretrained -> MLP only
    python eval_downstream.py --config configs/downstream_slice.yaml

    # Camera-ready protocol (cr_seed_v1): 5 head seeds fitted after the
    # feature caches exist, sealed test predictions, patch-max collected in
    # the same encoder pass. See configs/downstream_cr_seed_v1.yaml.
    python eval_downstream.py --config configs/downstream_cr_seed_v1.yaml \
        --encoder-checkpoint <ckpt> --output-dir <new dir>

Head RNG is reset from each head seed after every feature cache is created or
loaded, so cold and warm caches give the same head initialization and the
same training order. Outputs per pooling variant and head seed live in
``<output_dir>/<variant>/seed<k>/``; ``<output_dir>/results.json`` is the
aggregate and is written last.

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast

# Downstream precision switch.
#
# `autocast()` was called unconditionally at six sites, so setting
# `data.use_amp: false` in a config silently had no effect outside
# precompute_features and every evaluation ran in fp16. That matters when the
# stated requirement is fp32 for numerical accuracy. This module-level flag is
# set once from the config and honoured everywhere via `amp_ctx()`.
_USE_AMP = True


def set_amp(enabled):
    """Set the global downstream precision mode. Call once from the config."""
    global _USE_AMP
    _USE_AMP = bool(enabled)


def amp_ctx():
    """autocast when AMP is enabled, otherwise a no-op context (true fp32)."""
    return autocast() if _USE_AMP else nullcontext()
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import yaml

# ImageNet normalization (must match pretraining transforms in src/transforms.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(x):
    """Normalize a batch of [0,1] images to ImageNet mean/std.

    Args:
        x: (B, 3, H, W) tensor in [0, 1] range.
    Returns:
        (B, 3, H, W) tensor normalized to ImageNet distribution.
    """
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src.models.vision_transformer import (
    VisionTransformer, SliceEncoder, Block, VIT_EMBED_DIMS,
)
from src.models.attentive_pool_minimal import CrossAttnPool, MeanPool

_PROBE_TYPES = ('attentive', 'cross_attn_pool', 'mean_pool')


def _build_probe(probe_type, num_slices, embed_dim, model_cfg, device):
    """Instantiate the slice-aggregation probe. Fails fast on unknown types."""
    if probe_type not in _PROBE_TYPES:
        raise ValueError(
            "Unknown probe_type=%r. Valid values: %s"
            % (probe_type, ', '.join(_PROBE_TYPES))
        )
    if probe_type == 'mean_pool':
        probe = MeanPool(num_slices=num_slices, embed_dim=embed_dim).to(device)
        desc = 'mean_pool (0 params)'
    elif probe_type == 'cross_attn_pool':
        head_dim = model_cfg.get('probe_head_dim', 64)
        probe = CrossAttnPool(
            num_slices=num_slices, embed_dim=embed_dim, head_dim=head_dim,
        ).to(device)
        desc = 'cross_attn_pool (head_dim=%d)' % head_dim
    else:  # 'attentive'
        depth = model_cfg.get('probe_depth', 2)
        probe = AttentiveProbe(
            num_slices=num_slices,
            embed_dim=embed_dim,
            num_heads=model_cfg.get('probe_num_heads', 12),
            depth=depth,
        ).to(device)
        desc = 'attentive (depth=%d)' % depth
    return probe, desc
try:
    from src.models.feature_extractor import FrozenFeatureExtractor
except ImportError:
    FrozenFeatureExtractor = None  # Slice-level approach (archived)
from src.datasets.oct_volumes import OCTVolumeDataset
from src.helper import _VIT_CONFIGS, optimizer_step
from src.utils.distributed import init_distributed


# ---------------------------------------------------------------------------
# Attentive probe for patch-level downstream (I-JEPA paper design)
# ---------------------------------------------------------------------------

class AttentiveProbe(nn.Module):
    """Slice-level attention probe for 3D OCT volume aggregation.

    Adapted from the I-JEPA attentive probe (Assran et al., 2023).  The
    paper uses a single block because patch tokens already carry global
    context from 12 encoder layers.  Our slice tokens are independently
    encoded, so we default to ``depth=2`` to give the model a chance to
    learn inter-slice relationships (configurable for ablation).

    Input:  (B, num_slices, embed_dim) -- one token per slice.
    Output: (B, embed_dim) -- volume representation from CLS token.

    Parameters (depth=2, dim=768):
        cls_token:  1 x 768          =       768
        pos_embed:  101 x 768        =    77,568
        2 x Block (SA + MLP):      ~14,175,744
        final norm:                      1,536
        Total:                     ~14,255,616
    """

    def __init__(self, num_slices=100, embed_dim=768, num_heads=12, depth=2):
        super(AttentiveProbe, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, S+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token -> (B, D)


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """Linear classification head (I-JEPA paper protocol)."""

    def __init__(self, in_dim, out_dim=1):
        super(LinearHead, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=1, dropout=0.1):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SliceMaxPool(nn.Module):
    """Across-B-scan max of per-B-scan vectors: (B, S, D) -> (B, D). No parameters."""

    def forward(self, x):
        return x.amax(dim=1)


class SliceMeanMaxPool(nn.Module):
    """Across-B-scan mean concatenated with max: (B, S, D) -> (B, 2D). No parameters."""

    def forward(self, x):
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)


# Pooling variants fitted by the frozen probe. The first entry is the primary
# protocol (patch-token mean per B-scan, mean across B-scans); the others are
# fitted only with ``probe.extra_pooling: patch_max`` and reuse the same head
# recipe. ``features`` names the per-B-scan reduction stored in the cache.
PRIMARY_VARIANT = 'patchmean_slicemean'
POOLING_VARIANTS = {
    'patchmean_slicemean': {'features': 'patch_mean', 'slice_pool': 'mean', 'width': 1},
    'patchmax_slicemean': {'features': 'patch_max', 'slice_pool': 'mean', 'width': 1},
    'patchmean_slicemax': {'features': 'patch_mean', 'slice_pool': 'max', 'width': 1},
    'patchmean_slicemeanmax': {'features': 'patch_mean', 'slice_pool': 'meanmax', 'width': 2},
}
EXTRA_POOLING_CHOICES = ('none', 'patch_max')


def normalize_extra_pooling(value):
    if value in (None, False, '', 'none', 'None'):
        return None
    if value != 'patch_max':
        raise ValueError("Unknown extra_pooling=%r (valid: none, patch_max)" % (value,))
    return value


# ---------------------------------------------------------------------------
# LR schedule with warmup
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_finetune_param_groups(encoder, probe, head, train_cfg):
    """Build AdamW param_groups for fine-tuning.

    If ``layer_decay`` in train_cfg is strictly in (0, 1), applies MAE-style
    Layer-wise LR Decay (LLRD) to the encoder:
      - base LR = ``lr_probe`` (also used for probe / encoder.norm)
      - encoder.patch_embed + pos_embed: base * decay^(num_layers)
      - encoder.blocks[i]:               base * decay^(num_layers - (i+1))
      - encoder.norm + probe:            base
      - head:                            ``lr_head`` (usually = base)
    For ViT-B/16 with 12 blocks and decay=0.65, the deepest layer gets
    ~0.65^13 ≈ 5.69e-3 of base; the top block gets 0.65 of base.

    If ``layer_decay`` is missing or >= 1.0, falls back to the older flat
    3-group setup (encoder, probe, head) so previous configs still work.

    Returns ``(param_groups, mode)`` where mode is 'llrd' or 'flat'.
    The first group is always the "deepest" / smallest-LR encoder group
    and the last two are probe / head, so downstream logging can read
    ``group[0]`` as the encoder floor and ``group[-2]`` / ``group[-1]``
    as probe / head LR without branching on mode.
    """
    lr_probe = train_cfg.get('lr_probe', 1e-4)
    lr_head = train_cfg.get('lr_head', lr_probe)
    layer_decay = train_cfg.get('layer_decay', 1.0)
    if layer_decay is None:
        layer_decay = 1.0

    if 0.0 < layer_decay < 1.0:
        num_blocks = len(encoder.blocks)
        num_layers = num_blocks + 1  # embed is layer 0, head is layer num_blocks+1
        base_lr = lr_probe
        groups = []
        embed_lr = base_lr * (layer_decay ** num_layers)
        groups.append({
            'params': list(encoder.patch_embed.parameters()) + [encoder.pos_embed],
            'lr': embed_lr,
            'name': 'embed',
        })
        for i, block in enumerate(encoder.blocks):
            lr_i = base_lr * (layer_decay ** (num_layers - (i + 1)))
            groups.append({'params': list(block.parameters()), 'lr': lr_i,
                           'name': f'block_{i}'})
        groups.append({'params': list(encoder.norm.parameters()), 'lr': base_lr,
                       'name': 'encoder_norm'})
        groups.append({'params': list(probe.parameters()), 'lr': base_lr,
                       'name': 'probe'})
        groups.append({'params': list(head.parameters()), 'lr': lr_head,
                       'name': 'head'})
        # Drop empty groups so AdamW doesn't reject them (e.g. MeanPool probe
        # has 0 trainable parameters).
        groups = [g for g in groups if len(g['params']) > 0]
        return groups, 'llrd'

    # Flat fallback
    lr_encoder = train_cfg.get('lr_encoder', 5e-6)
    groups = [
        {'params': list(encoder.parameters()), 'lr': lr_encoder, 'name': 'encoder'},
        {'params': list(probe.parameters()), 'lr': lr_probe, 'name': 'probe'},
        {'params': list(head.parameters()), 'lr': lr_head, 'name': 'head'},
    ]
    groups = [g for g in groups if len(g['params']) > 0]
    return groups, 'flat'


# ---------------------------------------------------------------------------
# Evaluation (works on cached feature tensors)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(probe, head, loader, criterion, device, return_predictions=False):
    """Run evaluation on cached features.

    Returns:
        (loss, auc) or (loss, auc, labels, probs) if return_predictions=True.
    """
    probe.eval()
    head.eval()

    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with amp_ctx():
            pooled = probe(features)             # (B, D)
            logits = head(pooled).squeeze(-1)    # (B,)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    if return_predictions:
        return avg_loss, auc, all_labels, all_probs
    return avg_loss, auc


# ---------------------------------------------------------------------------
# Feature pre-computation (one-time cost, cached to disk)
# ---------------------------------------------------------------------------

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     default=str).encode('utf-8')).hexdigest()


def dataset_order_manifest(dataset, split, encoder_source=None):
    """Local provenance; file stems identify source cases, not verified patients."""
    files = []
    for path in dataset.file_paths:
        stat = os.stat(path)
        files.append({'name': os.path.basename(path), 'bytes': stat.st_size,
                      'mtime_ns': stat.st_mtime_ns})
    ids = [hashlib.sha256(os.path.splitext(item['name'])[0].encode('utf-8')).hexdigest()
           for item in files]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source-case identifiers in %s" % split)
    return {
        'schema_version': 2,
        'dataset_root': os.path.realpath(dataset.data_dir),
        'split': split, 'ordered_files': files, 'subject_ids': ids,
        'subject_id_scheme': 'sha256(source_file_stem); case identity, patient linkage unverified',
        'dataset_identity_kind': 'ordered_paths_sizes_mtimes_not_volume_content_hashes',
        'encoder_source': encoder_source,
        'num_slices': dataset.num_slices, 'slice_size': dataset.slice_size,
        'slice_indices': dataset.slice_indices.tolist(),
        'preprocessing': 'PIL-bilinear-RGB/ImageNet-normalize/patch-token-mean-v1',
        'dataset_code_sha256': file_sha256(sys.modules[OCTVolumeDataset.__module__].__file__),
        'model_code_sha256': file_sha256(sys.modules[VisionTransformer.__module__].__file__),
    }


def _encoder_source(encoder, source_identity):
    if source_identity is not None:
        return source_identity
    digest = hashlib.sha256()
    for name, tensor in encoder.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {'state_dict_sha256': digest.hexdigest(), 'class': type(encoder).__name__}


def _validate_feature_cache(data, manifest, allow_unverified=False):
    expected_n = len(manifest['subject_ids'])
    features, labels = data['features'], data['labels']
    if (features.ndim != 3 or features.shape[:2] != (expected_n, manifest['num_slices'])
            or labels.shape != (expected_n,)):
        raise ValueError("Feature cache shape/order length mismatch")
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError("Feature cache contains nonfinite values")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Feature cache has invalid binary labels")
    if data.get('source_manifest') != manifest and not allow_unverified:
        raise ValueError("Feature cache source identity mismatch or missing provenance")
    return features, labels


def save_predictions(path, labels, probs, manifest):
    if not len(labels) == len(probs) == len(manifest['subject_ids']):
        raise ValueError("Prediction/subject-order length mismatch")
    np.savez(path, labels=labels, probs=probs,
             subject_ids=np.asarray(manifest['subject_ids']),
             row_index=np.arange(len(labels)),
             subject_order_verified=np.asarray('legacy_cache_unverified' not in manifest),
             source_manifest_sha256=np.asarray(_identity_digest(manifest)))
    with open(os.path.splitext(path)[0] + '.manifest.json', 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)


_LEGACY_CACHE_POLICIES = ('reject', 'allow_unverified', 'allow_verified')


def patch_max_manifest(manifest):
    """Identity of the per-B-scan patch-token max cache (same encoder pass as the mean)."""
    out = dict(manifest)
    out['preprocessing'] = 'PIL-bilinear-RGB/ImageNet-normalize/patch-token-max-v1'
    out['feature_reduction'] = 'patch_token_max'
    return out


def _cache_record(state, path, manifest):
    return {'state': state, 'path': os.path.realpath(path) if path else None,
            'identity_sha256': _identity_digest(manifest)}


def _atomic_torch_save(obj, path):
    """Write to a sibling temp file and rename, so a partial file never looks like a cache."""
    partial = path + '.partial'
    torch.save(obj, partial)
    os.replace(partial, path)


def load_feature_cache(path, manifest):
    """Load a v2 cache and check it against the expected source manifest."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return _validate_feature_cache(data, manifest)


def _encode_split(encoder, dataset, split, device, chunk_size, use_amp, num_workers,
                  want_max=False):
    """One encoder pass. Returns (patch-mean features, labels, patch-max features or None)."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_features = []
    all_labels = []
    max_features = None

    encoder.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (volume, label) in enumerate(loader):
            volume = volume.to(device)       # (1, S, 3, H, W)
            flat = volume.squeeze(0)          # (S, 3, H, W)

            parts = []
            max_parts = []
            for j in range(0, flat.size(0), chunk_size):
                chunk = flat[j:j + chunk_size]
                chunk = imagenet_normalize(chunk)  # match pretraining distribution
                with autocast(enabled=use_amp):
                    out = encoder(chunk)      # (chunk, patches, D)
                parts.append(out.mean(dim=1).cpu())  # (chunk, D)
                if want_max:
                    max_parts.append(out.amax(dim=1).cpu())

            all_features.append(torch.cat(parts, dim=0))  # (S, D)
            all_labels.append(label.squeeze())
            if want_max:
                volume_max = torch.cat(max_parts, dim=0)
                if max_features is None:
                    max_features = torch.empty((len(dataset),) + tuple(volume_max.shape),
                                               dtype=volume_max.dtype)
                max_features[i] = volume_max

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print('    %s: %d/%d volumes (%.0fs)'
                      % (split, i + 1, len(dataset), elapsed))

    features = torch.stack(all_features)     # (N, S, D)
    labels = torch.stack(all_labels).long()  # (N,)
    elapsed = time.time() - t0
    print('  %s: %d volumes encoded in %.0fs (%.1f vol/s)'
          % (split, len(dataset), elapsed, len(dataset) / max(elapsed, 1)))
    return features, labels, max_features


def precompute_features(encoder, data_dir, split, num_slices, slice_size,
                        device, chunk_size=50, cache_dir=None, use_amp=True,
                        cache_key='', source_identity=None, return_manifest=False,
                        legacy_cache_policy='reject', num_workers=4,
                        extra_pooling=None, legacy_spec=None, provenance=None):
    """Encode all volumes in a split with the frozen ViT and cache to disk.

    ``extra_pooling='patch_max'`` also stores the per-B-scan patch-token max
    from the same encoder pass in a separate cache file; the patch-mean result
    and its cache identity are unchanged. ``provenance`` (a dict) receives the
    cache state of the split: cold, warm, legacy_verified or legacy_unverified.
    ``encoder`` may be None only when a usable cache exists.

    Returns:
        features: (N, num_slices, embed_dim), encoder-output dtype (AMP or fp32)
        labels:   (N,) long
    """
    if legacy_cache_policy not in _LEGACY_CACHE_POLICIES:
        raise ValueError("Unknown legacy_cache_policy")
    extra_pooling = normalize_extra_pooling(extra_pooling)
    if extra_pooling and not cache_dir:
        raise ValueError("extra_pooling requires cache_dir (patch-max features are cached)")
    if legacy_cache_policy == 'allow_verified' and not legacy_spec:
        raise ValueError("legacy_cache_policy=allow_verified requires legacy cache evidence")
    prov = provenance if provenance is not None else {}
    dataset = OCTVolumeDataset(
        os.path.join(data_dir, split), num_slices=num_slices,
        slice_size=slice_size, return_label=True)
    manifest = dataset_order_manifest(dataset, split, _encoder_source(encoder, source_identity))
    manifest.update({'use_amp': bool(use_amp), 'chunk_size': chunk_size,
                     'torch_version': torch.__version__})
    max_manifest = patch_max_manifest(manifest) if extra_pooling else None
    features = labels = None
    cache_path = max_path = None
    if cache_dir:
        suffix = 'amp' if use_amp else 'fp32'
        parts = [split, 's%d' % num_slices, 'r%d' % slice_size, suffix]
        if cache_key:
            parts.append(cache_key)
        legacy_path = os.path.join(cache_dir, '%s.pt' % '_'.join(parts))
        cache_path = os.path.join(cache_dir, '%s_v2_%s.pt' %
                                  (split, _identity_digest(manifest)))
        if extra_pooling:
            max_path = os.path.join(cache_dir, '%s_patchmax_v2_%s.pt' %
                                    (split, _identity_digest(max_manifest)))
        legacy_found = None
        if legacy_cache_policy == 'allow_verified' and not os.path.exists(cache_path):
            legacy_found = locate_legacy_cache(legacy_spec, split, num_slices, slice_size,
                                               legacy_path)
        if os.path.exists(cache_path):
            print('  Loading cached %s features from %s' % (split, cache_path))
            features, labels = load_feature_cache(cache_path, manifest)
            prov['primary'] = _cache_record('warm', cache_path, manifest)
        elif legacy_found:
            print('  Verifying legacy %s cache %s' % (split, legacy_found))
            data = torch.load(legacy_found, map_location='cpu', weights_only=False)
            evidence = verify_legacy_cache(legacy_found, data, manifest, dataset, legacy_spec)
            features, labels = data['features'], data['labels']
            del data
            manifest['legacy_cache_verified'] = {
                'path': os.path.realpath(legacy_found),
                'file_sha256': evidence['cache_file_sha256'],
                'row_identity': evidence['row_identity'],
                'integrity': evidence['integrity'],
                'evidence_sha256': _identity_digest(evidence)}
            prov['primary'] = dict(_cache_record('legacy_verified', legacy_found, manifest),
                                   evidence=evidence)
            print('  Legacy %s cache verified (%d checks)' % (split, len(evidence['checks'])))
        elif os.path.exists(legacy_path):
            if legacy_cache_policy != 'allow_unverified':
                raise ValueError("Legacy cache lacks verified source identity: %s. "
                                 "Use a fresh output directory or explicitly allow_unverified."
                                 % legacy_path)
            warnings.warn("Using legacy cache with unverified source/order: %s" % legacy_path)
            data = torch.load(legacy_path, map_location='cpu', weights_only=False)
            features, labels = _validate_feature_cache(data, manifest, allow_unverified=True)
            manifest['legacy_cache_unverified'] = os.path.realpath(legacy_path)
            prov['primary'] = _cache_record('legacy_unverified', legacy_path, manifest)

    need_max = bool(extra_pooling) and not os.path.exists(max_path)
    if extra_pooling and not need_max:
        prov['patch_max'] = dict(_cache_record('warm', max_path, max_manifest),
                                 _manifest=max_manifest)
    if features is not None and not need_max:
        result = (features, labels)
        return result + (manifest,) if return_manifest else result
    if encoder is None:
        if features is None:
            raise ValueError("No usable %s feature cache and no encoder weights to "
                             "extract features" % split)
        prov['patch_max'] = {'state': 'unavailable', 'path': None,
                             'reason': 'encoder weights not available; only the '
                                       'patch-mean cache exists'}
        result = (features, labels)
        return result + (manifest,) if return_manifest else result

    fresh, fresh_labels, fresh_max = _encode_split(
        encoder, dataset, split, device, chunk_size, use_amp, num_workers, want_max=need_max)
    if features is None:
        features, labels = fresh, fresh_labels
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            _atomic_torch_save({'features': features, 'labels': labels,
                                'source_manifest': manifest}, cache_path)
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print('  Cached to %s (%.1f MB)' % (cache_path, size_mb))
        prov['primary'] = _cache_record('cold', cache_path, manifest)
    else:
        # The existing patch-mean cache stays authoritative; the re-encoded
        # means are only an extraction-reproducibility diagnostic.
        if not torch.equal(fresh_labels, labels.long()):
            raise ValueError("Fresh %s labels differ from the cached labels" % split)
        prov['primary']['fresh_recompute_max_abs_diff'] = float(
            (fresh.float() - features.float()).abs().max())
        del fresh
    if need_max:
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_torch_save({'features': fresh_max, 'labels': labels,
                            'source_manifest': max_manifest}, max_path)
        print('  Cached patch-max to %s (%.1f MB)'
              % (max_path, os.path.getsize(max_path) / (1024 * 1024)))
        prov['patch_max'] = dict(_cache_record('cold', max_path, max_manifest),
                                 _manifest=max_manifest)
        del fresh_max

    result = (features, labels)
    return result + (manifest,) if return_manifest else result


# ---------------------------------------------------------------------------
# Legacy (pre-v2) cache verification
# ---------------------------------------------------------------------------

def _norm_path(path):
    return os.path.normcase(os.path.realpath(path)) if path else None


def default_guard_json(run_dir):
    """run_guarded_probe.py wrote guards to <root>/autopilot_out/probe_guards."""
    name = os.path.basename(os.path.normpath(run_dir))
    if name.startswith('frozen_'):
        name = name[len('frozen_'):]
    root = os.path.dirname(os.path.dirname(os.path.realpath(run_dir)))
    return os.path.join(root, 'autopilot_out', 'probe_guards', 'guard_%s.json' % name)


def locate_legacy_cache(spec, split, num_slices, slice_size, legacy_path=None):
    """Find the single legacy ``<split>_s<S>_r<R>_fp32_<key>.pt`` cache, or None."""
    if legacy_path and os.path.exists(legacy_path):
        return legacy_path
    cache_dir = spec.get('cache_dir') or os.path.join(spec['run_dir'], 'feature_cache')
    if not os.path.isdir(cache_dir):
        return None
    pattern = re.compile(r'^%s_s%d_r%d_fp32_([0-9a-f]+)\.pt$' % (re.escape(split), num_slices,
                                                                 slice_size))
    hits = sorted(name for name in os.listdir(cache_dir) if pattern.match(name))
    if len(hits) > 1:
        raise ValueError("Ambiguous legacy caches for %s in %s: %s" % (split, cache_dir, hits))
    return os.path.join(cache_dir, hits[0]) if hits else None


LEGACY_PIN_SCHEMA = 'cr_legacy_cache_hashes_v1'
# Files of the original probe run that serve as identity evidence; all must be pinned.
LEGACY_EVIDENCE_FILES = ('results.json', 'eval.log', 'best_model.pt', 'val_predictions.npz',
                         'test_predictions.npz', 'train_log.csv')


def load_legacy_pins(source):
    """Pinned SHA-256 of legacy cache/evidence files (trust on first use).

    ``source`` is a JSON file (schema cr_legacy_cache_hashes_v1) or a dict with
    ``files`` = {absolute path: sha256}. No historical hash of these files exists,
    so pins taken now only protect against changes after pinning.
    """
    if isinstance(source, str):
        with open(source, 'r', encoding='utf-8') as stream:
            pins = json.load(stream)
        pins.setdefault('pin_file', os.path.realpath(source))
        pins.setdefault('pin_file_sha256', file_sha256(source))
    else:
        pins = dict(source)
    if pins.get('schema') != LEGACY_PIN_SCHEMA or not isinstance(pins.get('files'), dict):
        raise ValueError("Legacy cache pins must use schema %s with a files map" % LEGACY_PIN_SCHEMA)
    pins['files'] = {_norm_path(k): str(v).lower() for k, v in pins['files'].items()}
    pins.setdefault('integrity', 'tofu_pinned_%s; historical integrity supported by timestamps '
                                 'and head replay, not cryptographic' % pins.get('pinned', '?'))
    return pins


def _pin_check(spec, path):
    """(ok, detail) comparing a file's SHA-256 with its pinned value."""
    pins = spec.get('pins') or {}
    expected = (pins.get('files') or {}).get(_norm_path(path))
    actual = file_sha256(path) if os.path.exists(path) else None
    return (expected is not None and actual == expected,
            {'path': os.path.realpath(path), 'pinned_sha256': expected, 'sha256': actual})


def collect_legacy_run_evidence(spec):
    """Run-level identity evidence for a legacy probe directory.

    Required: the run's results.json protocol matches, its eval.log names the
    checkpoint and fp32 extraction, and the guard sidecar records the
    checkpoint SHA-256 unchanged across the run and consistent with every
    other available source (expected value, local file, HF manifest). Every
    evidence file and the guard must match its pinned (TOFU) SHA-256.
    """
    checks = []

    def check(name, ok, detail, required=True):
        checks.append({'name': name, 'ok': bool(ok), 'required': required, 'detail': detail})

    run_dir = spec['run_dir']
    pinned = [_pin_check(spec, os.path.join(run_dir, name)) for name in LEGACY_EVIDENCE_FILES]
    pinned.append(_pin_check(spec, spec.get('guard_json') or default_guard_json(run_dir)))
    check('evidence_files_match_pinned_sha256', bool(spec.get('pins')) and all(ok for ok, _ in pinned),
          {'files': [detail for _, detail in pinned],
           'pin_file': (spec.get('pins') or {}).get('pin_file')})
    ckpt = spec['checkpoint_path']
    model_cfg = spec.get('model_config') or {}
    results_path = os.path.join(run_dir, 'results.json')
    try:
        with open(results_path, 'r', encoding='utf-8') as stream:
            cfg = json.load(stream).get('config', {})
        data, model = cfg.get('data', {}), cfg.get('model', {})
        mismatches = []
        if _norm_path(data.get('data_dir')) != _norm_path(spec['data_dir']):
            mismatches.append('data_dir')
        for key in ('num_slices', 'slice_size'):
            if data.get(key) != spec[key]:
                mismatches.append(key)
        if data.get('use_amp') is not False:
            mismatches.append('use_amp (must be explicitly false)')
        if _norm_path(model.get('encoder_checkpoint')) != _norm_path(ckpt):
            mismatches.append('encoder_checkpoint')
        for key in ('encoder_name', 'patch_size', 'crop_size'):
            if key in model_cfg and model.get(key) != model_cfg[key]:
                mismatches.append(key)
        if model.get('freeze_encoder') is not True:
            mismatches.append('freeze_encoder')
        check('run_config_matches', not mismatches,
              {'results_json': results_path, 'mismatches': mismatches,
               'legacy_encode_chunk_size': data.get('encode_chunk_size')})
    except (OSError, ValueError, AttributeError) as exc:
        check('run_config_matches', False, {'results_json': results_path, 'error': str(exc)})

    log_path = os.path.join(run_dir, 'eval.log')
    log_text = ''
    epoch = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as stream:
            log_text = stream.read()
        loads = [_norm_path(m.strip()) for m in
                 re.findall(r'Loading encoder from (.+?) \.\.\.', log_text)]
        epochs = re.findall(r'Loaded target_encoder weights \(epoch (-?\d+)\)', log_text)
        epoch = int(epochs[-1]) if epochs else None
        fp32 = 'precision: fp32 (AMP disabled)' in log_text
        ok = loads == [_norm_path(ckpt)] and fp32 and epoch is not None
        if spec.get('expected_epoch') is not None:
            ok = ok and epoch == int(spec['expected_epoch'])
        check('run_log_names_checkpoint_fp32', ok,
              {'eval_log': log_path, 'encoder_loads': len(loads), 'fp32_logged': fp32,
               'epoch_logged': epoch})
    except OSError as exc:
        check('run_log_names_checkpoint_fp32', False, {'eval_log': log_path, 'error': str(exc)})

    guard_path = spec.get('guard_json') or default_guard_json(run_dir)
    guard_sha = guard_finished = None
    try:
        with open(guard_path, 'r', encoding='utf-8') as stream:
            guard = json.load(stream)
        guard_sha = guard.get('sha256_before')
        guard_finished = guard.get('finished')
        ok = (_norm_path(guard.get('encoder_checkpoint')) == _norm_path(ckpt)
              and bool(guard_sha) and guard_sha == guard.get('sha256_after')
              and guard.get('encoder_unchanged') is True
              and guard.get('use_amp') is False and guard.get('return_code') == 0
              and _norm_path(guard.get('output_dir')) == _norm_path(run_dir))
        check('guard_sidecar_checkpoint_sha256', ok,
              {'guard_json': guard_path, 'sha256': guard_sha,
               'finished': guard.get('finished')})
    except (OSError, ValueError, AttributeError) as exc:
        check('guard_sidecar_checkpoint_sha256', False, {'guard_json': guard_path,
                                                         'error': str(exc)})

    sources = {'guard_sidecar': guard_sha}
    if spec.get('expected_checkpoint_sha256'):
        sources['expected_cli'] = spec['expected_checkpoint_sha256'].lower()
    if spec.get('local_checkpoint_sha256'):
        sources['local_file'] = spec['local_checkpoint_sha256']
    hf_manifest = spec.get('hf_manifest')
    if hf_manifest and os.path.exists(hf_manifest):
        with open(hf_manifest, 'r', encoding='utf-8') as stream:
            entries = json.load(stream)
        rel = os.path.relpath(_norm_path(ckpt), _norm_path(os.path.dirname(hf_manifest)))
        for key, entry in entries.items():
            if os.path.normcase(os.path.normpath(key)) == rel:
                sources['hf_manifest'] = entry.get('sha256')
    values = set(v for v in sources.values() if v)
    check('checkpoint_sha256_consistent', guard_sha is not None and values == {guard_sha},
          {'sources': sources, 'sources_beyond_sidecar': sorted(k for k in sources
                                                                 if k != 'guard_sidecar')})
    return {'checks': checks, 'checkpoint_sha256': guard_sha, 'checkpoint_epoch': epoch,
            'run_dir': os.path.realpath(run_dir), 'guard_json': guard_path,
            'guard_finished': guard_finished, 'log_text': log_text}


def _replay_legacy_head(run_dir, split, features, labels, atol):
    """Old best head applied to the legacy cache must reproduce its saved predictions."""
    pred_name = {'Validation': 'val_predictions.npz', 'Test': 'test_predictions.npz'}.get(split)
    head_path = os.path.join(run_dir, 'best_model.pt')
    if pred_name is None or not os.path.exists(head_path) \
            or not os.path.exists(os.path.join(run_dir, pred_name)):
        return None
    state = torch.load(head_path, map_location='cpu', weights_only=False)
    head = LinearHead(in_dim=features.shape[-1])
    head.load_state_dict(state['head'])
    head.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, features.shape[0], 256):
            pooled = features[i:i + 256].float().mean(dim=1)
            probs.append(torch.sigmoid(head(pooled).squeeze(-1)))
    probs = torch.cat(probs).numpy().astype(np.float64)
    with np.load(os.path.join(run_dir, pred_name)) as saved:
        saved_labels = saved['labels'].astype(np.int64)
        saved_probs = saved['probs'].astype(np.float64)
    same_labels = np.array_equal(saved_labels, labels.numpy().astype(np.int64))
    diff = float(np.abs(saved_probs - probs).max()) if saved_probs.shape == probs.shape else None
    return {'ok': bool(same_labels and diff is not None and diff <= atol),
            'detail': {'predictions': pred_name, 'labels_equal': same_labels,
                       'max_abs_prob_diff': diff, 'atol': atol}}


def _parse_timestamp(text):
    for parse in (lambda s: datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z'),
                  datetime.datetime.fromisoformat):
        try:
            return parse(str(text))
        except (TypeError, ValueError):
            continue
    return None


def _replay_legacy_training(run_dir, features, labels, auc_atol, loss_atol):
    """Old best head on the legacy Training cache must reproduce the logged train AUC/loss
    of its best epoch (binds the Training bytes to the run; there are no saved train
    predictions). Returns None when the run lacks best_model.pt / results.json / train_log.csv."""
    head_path = os.path.join(run_dir, 'best_model.pt')
    log_path = os.path.join(run_dir, 'train_log.csv')
    results_path = os.path.join(run_dir, 'results.json')
    if not all(os.path.exists(p) for p in (head_path, log_path, results_path)):
        return None
    with open(results_path, 'r', encoding='utf-8') as stream:
        best_epoch = json.load(stream).get('best_epoch')
    logged = None
    with open(log_path, 'r', encoding='utf-8') as stream:
        header = stream.readline().strip().split(',')
        for line in stream:
            row = dict(zip(header, line.strip().split(',')))
            if row.get('epoch') and int(row['epoch']) == best_epoch:
                logged = row
    if logged is None:
        return {'ok': False, 'detail': {'error': 'best epoch %r not in train_log.csv' % best_epoch}}
    state = torch.load(head_path, map_location='cpu', weights_only=False)
    head = LinearHead(in_dim=features.shape[-1])
    head.load_state_dict(state['head'])
    head.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, features.shape[0], 256):
            logits.append(head(features[i:i + 256].float().mean(dim=1)).squeeze(-1))
    logits = torch.cat(logits)
    y = labels.float()
    loss = float(F.binary_cross_entropy_with_logits(logits, y))
    auc = float(roc_auc_score(y.numpy(), torch.sigmoid(logits).numpy()))
    d_auc = abs(auc - float(logged['train_auc']))
    d_loss = abs(loss - float(logged['train_loss']))
    return {'ok': bool(d_auc <= auc_atol and d_loss <= loss_atol),
            'detail': {'best_epoch': best_epoch, 'replay_train_auc': auc,
                       'logged_train_auc': float(logged['train_auc']), 'replay_train_loss': loss,
                       'logged_train_loss': float(logged['train_loss']),
                       'auc_atol': auc_atol, 'loss_atol': loss_atol}}


def verify_legacy_cache(cache_path, data, manifest, dataset, spec):
    """Accept a legacy (N, S, D) fp32 cache only with complete identity evidence.

    Legacy caches store no file names, so order is verified by the label
    sequence against the current split listing plus the requirement that no
    split file changed after the cache was written. Raises ValueError listing
    every failed required check.
    """
    split = manifest['split']
    run_evidence = spec.get('_run_evidence') or collect_legacy_run_evidence(spec)
    checks = [dict(c) for c in run_evidence['checks']]

    def check(name, ok, detail, required=True):
        checks.append({'name': name, 'ok': bool(ok), 'required': required, 'detail': detail})

    n, num_slices = len(manifest['subject_ids']), manifest['num_slices']
    embed_dim = spec.get('embed_dim', 768)
    features, labels = data.get('features'), data.get('labels')
    shape_ok = False
    try:
        _validate_feature_cache(data, manifest, allow_unverified=True)
        shape_ok = (features.dtype == torch.float32
                    and tuple(features.shape) == (n, num_slices, embed_dim))
        check('shape_dtype', shape_ok, {'shape': list(features.shape),
                                        'dtype': str(features.dtype),
                                        'expected': [n, num_slices, embed_dim]})
    except (ValueError, AttributeError, KeyError) as exc:
        check('shape_dtype', False, str(exc))

    split_labels = np.array([int(np.load(path, allow_pickle=True)['glaucoma'].item())
                             for path in dataset.file_paths], dtype=np.int64)
    if shape_ok:
        cache_labels = labels.numpy().astype(np.int64)
        check('label_order_matches_split', np.array_equal(cache_labels, split_labels),
              {'n': n, 'n_mismatch': int((cache_labels != split_labels).sum())})
    else:
        check('label_order_matches_split', False, 'shape check failed')

    cache_mtime = os.stat(cache_path).st_mtime_ns
    newest = max(item['mtime_ns'] for item in manifest['ordered_files'])
    check('split_files_older_than_cache', newest < cache_mtime,
          {'newest_split_file_mtime_ns': newest, 'cache_mtime_ns': cache_mtime,
           'n_files': len(manifest['ordered_files'])})
    # The original run wrote the cache before its guard recorded completion; a cache
    # modified afterwards (edited, re-saved, replaced) no longer belongs to that run.
    finished = _parse_timestamp(run_evidence.get('guard_finished'))
    check('cache_not_modified_after_run',
          finished is not None and cache_mtime / 1e9 <= finished.timestamp(),
          {'cache_mtime_ns': cache_mtime, 'guard_finished': run_evidence.get('guard_finished')})

    cached = [_norm_path(m.strip()) for m in
              re.findall(r'Cached to (.+?) \(', run_evidence.get('log_text', ''))]
    check('run_log_wrote_this_cache', _norm_path(cache_path) in cached,
          {'cache_path': os.path.realpath(cache_path)})

    match = re.search(r'_fp32_([0-9a-f]+)\.pt$', os.path.basename(cache_path))
    key = match.group(1) if match else None
    local_key = spec.get('local_checkpoint_prefix_key')
    if local_key:
        check('filename_key_matches_checkpoint_prefix',
              key is not None and local_key.startswith(key),
              {'filename_key': key, 'local_prefix_key': local_key})
    else:
        check('filename_key_matches_checkpoint_prefix', True,
              {'filename_key': key, 'status': 'unverifiable: checkpoint not on disk; key is '
                                               'sha256(first 1 MiB of checkpoint)[:12]'},
              required=False)

    # Validation/Test: the old best head must reproduce the saved per-case predictions
    # (required: missing files fail). Training has no saved predictions: the old head must
    # reproduce the logged train AUC/loss of its best epoch instead.
    if split == 'Training':
        replay = _replay_legacy_training(
            spec['run_dir'], features, labels, spec.get('train_replay_auc_atol', 0.01),
            spec.get('train_replay_loss_atol', 0.02)) if shape_ok else None
        name = 'legacy_training_head_replay'
        row_identity = ('label sequence only: label-preserving row permutations are not '
                        'detectable and leave the training (feature, label) multiset unchanged; '
                        'do not use these rows for per-case analyses')
    else:
        replay = _replay_legacy_head(spec['run_dir'], split, features, labels,
                                     spec.get('head_replay_atol', 1e-4)) if shape_ok else None
        name = 'legacy_head_replay'
        row_identity = 'per-case order verified by old-head replay of saved predictions'
    if replay is None:
        check(name, False, {'error': 'missing old head/predictions/log for replay'})
    else:
        check(name, replay['ok'], replay['detail'])

    pin_ok, pin_detail = _pin_check(spec, cache_path)
    check('cache_matches_pinned_sha256', pin_ok, pin_detail)

    failed = [c['name'] for c in checks if c['required'] and not c['ok']]
    if failed:
        raise ValueError("Legacy cache %s failed verification: %s"
                         % (cache_path, ', '.join(failed)))
    return {'cache_path': os.path.realpath(cache_path), 'cache_file_sha256': pin_detail['sha256'],
            'cache_mtime_ns': cache_mtime, 'checkpoint_path': spec['checkpoint_path'],
            'checkpoint_sha256': run_evidence['checkpoint_sha256'],
            'checkpoint_epoch': run_evidence['checkpoint_epoch'],
            'run_dir': run_evidence['run_dir'], 'guard_json': run_evidence['guard_json'],
            'order_evidence': 'label sequence equals the current split listing and every '
                              'split file predates the cache; the legacy cache stores no names',
            'row_identity': row_identity,
            'integrity': spec['pins']['integrity'],
            'pin_file': spec['pins'].get('pin_file'),
            'pin_file_sha256': spec['pins'].get('pin_file_sha256'),
            'checks': checks}


# ---------------------------------------------------------------------------
# Diagnostic plots (generated at end of training)
# ---------------------------------------------------------------------------

def _save_diagnostic_plots(output_dir, test_labels, test_probs, test_auc,
                           val_labels, val_probs):
    """Generate ROC curve, confusion matrix, and prediction histogram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, confusion_matrix
    except ImportError:
        print('  Skipping plots (matplotlib not available)')
        return

    if test_labels is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curve
    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='Test AUC = %.3f' % (test_auc or 0))
    if val_labels is not None:
        fpr_v, tpr_v, _ = roc_curve(val_labels, val_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        ax.plot(fpr_v, tpr_v, 'g--', linewidth=1.5, label='Val AUC = %.3f' % val_auc)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # 2. Confusion matrix at threshold=0.5
    ax = axes[1]
    preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(test_labels, preds)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (threshold=0.5)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Glaucoma', 'Glaucoma'])
    ax.set_yticklabels(['Non-Glaucoma', 'Glaucoma'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)

    # 3. Prediction histogram
    ax = axes[2]
    ax.hist(test_probs[test_labels == 0], bins=30, alpha=0.6, color='blue',
            label='Non-Glaucoma (n=%d)' % (test_labels == 0).sum(), density=True)
    ax.hist(test_probs[test_labels == 1], bins=30, alpha=0.6, color='red',
            label='Glaucoma (n=%d)' % (test_labels == 1).sum(), density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')
    ax.set_xlabel('P(Glaucoma)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()

    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'diagnostic_plots.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved diagnostic_plots.png')


# ---------------------------------------------------------------------------
# Patch-level downstream
# ---------------------------------------------------------------------------

def seed_everything(seed):
    """Seed python, numpy, torch CPU and CUDA from one head seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


class _RecordingRandomSampler(torch.utils.data.RandomSampler):
    """RandomSampler that hashes the training order it yields (audit only)."""

    def __init__(self, data_source, generator):
        super(_RecordingRandomSampler, self).__init__(data_source, generator=generator)
        self.order_digest = hashlib.sha256()

    def __iter__(self):
        for index in super(_RecordingRandomSampler, self).__iter__():
            self.order_digest.update(int(index).to_bytes(8, 'little'))
            yield index


@torch.no_grad()
def predict(probe, head, loader, device):
    """Labels and probabilities only: no loss or metric is computed (sealed test)."""
    probe.eval()
    head.eval()
    all_labels = []
    all_probs = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with amp_ctx():
            pooled = probe(features)
            logits = head(pooled).squeeze(-1)
        all_labels.append(labels.cpu())
        all_probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()


def resolve_head_seeds(config):
    """``probe.head_seeds`` (list or comma string); defaults to [training.seed]."""
    seeds = (config.get('probe') or {}).get('head_seeds')
    if seeds is None:
        seeds = [config.get('training', {}).get('seed', 42)]
    if isinstance(seeds, str):
        seeds = [item for item in seeds.split(',') if item.strip()]
    if isinstance(seeds, int):
        seeds = [seeds]
    seeds = [int(item) for item in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe.head_seeds must be a non-empty list of distinct integers")
    return seeds


def _refuse_completed_output(output_dir, seal_test=False):
    if os.path.exists(os.path.join(output_dir, 'sealed_manifest.json')):
        raise RuntimeError("%s already holds sealed test predictions; use a new output_dir"
                           % output_dir)
    results_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as stream:
                status = json.load(stream).get('status')
        except (OSError, ValueError, AttributeError):
            status = None
        if status == 'complete':
            raise RuntimeError("%s already holds a completed probe; use a new output_dir"
                               % output_dir)
    if seal_test and os.path.isdir(output_dir):
        # A sealed run must not sit next to older (possibly unsealed) results, heads,
        # predictions, plots or logs. Only a feature cache may be reused; nothing is deleted.
        stale = sorted(set(os.listdir(output_dir)) - {'feature_cache', 'cache_provenance.json'})
        if stale:
            raise RuntimeError("Sealed probe needs a fresh output_dir (only feature_cache/ may "
                               "be reused); %s already holds %s" % (output_dir, stale[:10]))


def _code_identity():
    info = {'eval_downstream_sha256': file_sha256(os.path.abspath(__file__)),
            'git_head': None, 'git_dirty_tracked_files': None}
    try:
        info['git_head'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_project_root, stderr=subprocess.DEVNULL,
            timeout=30).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=_project_root,
            stderr=subprocess.DEVNULL, timeout=30).decode().splitlines()
        info['git_dirty_tracked_files'] = len(dirty)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _numeric_backend(device):
    cuda = torch.cuda.is_available()
    device_name = 'cpu'
    if torch.device(device).type == 'cuda':
        try:
            device_name = torch.cuda.get_device_name(torch.device(device))
        except (AssertionError, RuntimeError):
            device_name = 'unavailable'
    return {'torch_version': torch.__version__, 'cuda_available': cuda,
            'device': str(device), 'device_name': device_name,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def _build_pool(variant, model_cfg, num_slices, embed_dim, device):
    if variant == PRIMARY_VARIANT:
        return _build_probe(model_cfg.get('probe_type', 'attentive'), num_slices, embed_dim,
                            model_cfg, device)
    slice_pool = POOLING_VARIANTS[variant]['slice_pool']
    if slice_pool == 'mean':
        return MeanPool(num_slices=num_slices, embed_dim=embed_dim).to(device), 'mean_pool (0 params)'
    if slice_pool == 'max':
        return SliceMaxPool().to(device), 'slice_max_pool (0 params)'
    return SliceMeanMaxPool().to(device), 'slice_mean_max_pool (0 params)'


def fit_probe_head(variant, head_seed, data, config, device, output_dir, seal_test=False,
                   make_plots=True):
    """Fit one head for one pooling variant and head seed.

    All RNGs are reset from ``head_seed`` here, after every feature cache was
    created or loaded, so head initialization and training order do not depend
    on cache state. ``data`` maps split -> (features, labels, manifest).
    """
    model_cfg, train_cfg, data_cfg = config['model'], config['training'], config['data']
    seed_dir = os.path.join(output_dir, variant, 'seed%d' % head_seed)
    os.makedirs(seed_dir, exist_ok=True)
    best_path = os.path.join(seed_dir, 'best_model.pt')
    if os.path.exists(best_path):
        os.remove(best_path)  # never select a stale head from an interrupted attempt
    train_feats, train_labels, _ = data['Training']
    val_feats, val_labels, val_manifest = data['Validation']
    test_feats, test_labels, test_manifest = data['Test']
    num_slices, embed_dim = train_feats.shape[1], train_feats.shape[2]
    in_dim = embed_dim * POOLING_VARIANTS[variant]['width']
    batch_size = data_cfg.get('batch_size', 16)
    tag = '[%s seed %d]' % (variant, head_seed)
    print('\n--- Model %s ---' % tag)

    seed_everything(head_seed)
    probe, probe_desc = _build_pool(variant, model_cfg, num_slices, embed_dim, device)
    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=in_dim, dropout=train_cfg.get('dropout', 0.1)).to(device)
    else:
        head = LinearHead(in_dim=in_dim).to(device)
    init_head_sha256 = state_digest(head)
    train_generator = _seeded_generator(head_seed)
    train_dataset = TensorDataset(train_feats, train_labels.float())
    train_sampler = _RecordingRandomSampler(train_dataset, train_generator)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True,
        pin_memory=True, generator=train_generator)
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True,
        generator=_seeded_generator(head_seed))
    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True,
        generator=_seeded_generator(head_seed))

    probe_params = sum(p.numel() for p in probe.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    print('  Probe (%s): %s params (trainable)' % (probe_desc, format(probe_params, ',')))
    print('  Head (%s, in_dim=%d):     %s params (trainable)'
          % (head_type, in_dim, format(head_params, ',')))

    # Skip empty param groups (e.g. MeanPool probe has zero parameters) so
    # AdamW doesn't complain about a group with no tensors to optimize.
    param_groups = [
        {'params': list(probe.parameters()), 'lr': train_cfg.get('lr_probe', 1e-4)},
        {'params': list(head.parameters()), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], len(train_loader),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=_USE_AMP)

    csv_path = os.path.join(seed_dir, 'train_log.csv')
    csv_file = open(csv_path, 'w')
    csv_file.write('epoch,train_loss,train_auc,val_loss,val_auc,lr_probe,lr_head,elapsed_s\n')
    csv_file.flush()

    print('--- Training %s ---' % tag)
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        probe.train()
        head.train()
        total_loss = 0.0
        n_samples = 0
        train_labels_epoch = []
        train_probs_epoch = []

        t0 = time.time()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_ctx():
                pooled = probe(features)         # (B, D)
                logits = head(pooled).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if optimizer_step(optimizer, scaler):
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            with torch.no_grad():
                train_labels_epoch.append(labels.cpu())
                train_probs_epoch.append(torch.sigmoid(logits).cpu())

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_samples, 1)
        train_labels_np = torch.cat(train_labels_epoch).numpy()
        train_probs_np = torch.cat(train_probs_epoch).numpy()
        train_auc = roc_auc_score(train_labels_np, train_probs_np) if len(np.unique(train_labels_np)) >= 2 else 0.5

        val_loss, val_auc = evaluate(probe, head, val_loader, criterion, device)
        # Head is always the last param group. lr_probe only exists when the
        # probe has trainable params (mean_pool has none → its empty group
        # was filtered, so only the head group remains — report probe LR 0).
        lr_head = optimizer.param_groups[-1]['lr']
        lr_probe = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else 0.0

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print('Epoch %2d/%d (%4.1fs) | Train: %.4f (AUC %.3f) | Val: %.4f | AUC: %.4f | LR: %.2e/%.2e%s'
              % (epoch, epochs, elapsed, train_loss, train_auc, val_loss, val_auc,
                 lr_probe, lr_head, marker))

        csv_file.write('%d,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.1f\n'
                       % (epoch, train_loss, train_auc, val_loss, val_auc,
                          lr_probe, lr_head, elapsed))
        csv_file.flush()

        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'probe': probe.state_dict(),
                'head': head.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch %d (patience=%d)' % (epoch, patience))
                break

    csv_file.close()

    if not os.path.exists(best_path):
        raise RuntimeError('%s no epoch improved validation AUC above 0' % tag)
    best_ckpt = torch.load(best_path, map_location=device)
    probe.load_state_dict(best_ckpt['probe'])
    head.load_state_dict(best_ckpt['head'])
    best_epoch = best_ckpt['epoch']
    head_sha = file_sha256(best_path)

    val_loss_f, val_auc_f, val_labels_f, val_probs_f = evaluate(
        probe, head, val_loader, criterion, device, return_predictions=True)
    prediction_source = {
        'head_checkpoint_sha256': head_sha, 'pooling_variant': variant,
        'head_seed': head_seed, 'model_config': model_cfg, 'training_config': train_cfg,
        'probe_type': model_cfg.get('probe_type'), 'head_type': head_type,
    }
    val_path = os.path.join(seed_dir, 'val_predictions.npz')
    save_predictions(val_path, val_labels_f, val_probs_f,
                     dict(val_manifest, prediction_source=prediction_source))

    result = {
        'variant': variant, 'head_seed': head_seed, 'best_epoch': best_epoch,
        'epochs_run': epochs_run, 'best_val_auc': best_auc,
        'best_val_loss': best_ckpt['val_loss'], 'final_val_auc': val_auc_f,
        'probe_params': probe_params, 'head_params': head_params, 'head_in_dim': in_dim,
        'init_head_sha256': init_head_sha256,
        'train_order_sha256': train_sampler.order_digest.hexdigest(),
        'head_checkpoint_sha256': head_sha,
        'val_predictions': os.path.relpath(val_path, output_dir),
        'test_sealed': bool(seal_test),
    }
    if seal_test:
        test_labels_np, test_probs_np = predict(probe, head, test_loader, device)
        name = 'test_predictions_sealed_seed%d.npz' % head_seed
        test_path = os.path.join(seed_dir, name)
        save_predictions(test_path, test_labels_np, test_probs_np,
                         dict(test_manifest, prediction_source=prediction_source, sealed=True))
        sidecar = os.path.splitext(test_path)[0] + '.manifest.json'
        result.update({'test_auc': None, 'test_loss': None, 'sensitivity': None,
                       'specificity': None,
                       'test_predictions': os.path.relpath(test_path, output_dir),
                       'test_predictions_sha256': file_sha256(test_path),
                       'test_predictions_sidecar': os.path.relpath(sidecar, output_dir),
                       'test_predictions_sidecar_sha256': file_sha256(sidecar)})
        print('%s best epoch %d | Val AUC %.4f | test predictions SEALED (no test metrics)'
              % (tag, best_epoch, best_auc))
    else:
        test_loss, test_auc, test_labels_np, test_probs_np = evaluate(
            probe, head, test_loader, criterion, device, return_predictions=True)
        test_preds = (test_probs_np >= 0.5).astype(int)
        tp = ((test_preds == 1) & (test_labels_np == 1)).sum()
        tn = ((test_preds == 0) & (test_labels_np == 0)).sum()
        fp = ((test_preds == 1) & (test_labels_np == 0)).sum()
        fn = ((test_preds == 0) & (test_labels_np == 1)).sum()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        test_path = os.path.join(seed_dir, 'test_predictions.npz')
        save_predictions(test_path, test_labels_np, test_probs_np,
                         dict(test_manifest, prediction_source=prediction_source))
        result.update({'test_auc': test_auc, 'test_loss': test_loss,
                       'sensitivity': float(sensitivity), 'specificity': float(specificity),
                       'test_predictions': os.path.relpath(test_path, output_dir),
                       'test_predictions_sha256': file_sha256(test_path)})
        print('%s Best epoch: %d  |  Val AUC: %.4f  |  TEST AUC: %.4f'
              % (tag, best_epoch, best_auc, test_auc))
        print('  Sensitivity: %.4f  |  Specificity: %.4f  (threshold=0.5)'
              % (sensitivity, specificity))
        if make_plots:
            _save_diagnostic_plots(seed_dir, test_labels_np, test_probs_np, test_auc,
                                   val_labels_f, val_probs_f)
    with open(os.path.join(seed_dir, 'results.json'), 'w') as stream:
        json.dump(result, stream, indent=2)
    return result


def _summarize_variant(per_seed, sealed):
    vals = np.array([r['best_val_auc'] for r in per_seed], dtype=np.float64)
    summary = {
        'per_seed': per_seed, 'n_head_seeds': len(per_seed),
        'val_auc_mean': float(vals.mean()),
        'val_auc_sd': float(vals.std(ddof=1)) if len(vals) > 1 else None,
        'val_auc_min': float(vals.min()), 'val_auc_max': float(vals.max()),
        'probe_params': per_seed[0]['probe_params'], 'head_params': per_seed[0]['head_params'],
        'test_auc_mean': None, 'test_auc_sd': None,
    }
    if not sealed:
        tests = np.array([r['test_auc'] for r in per_seed], dtype=np.float64)
        summary['test_auc_mean'] = float(tests.mean())
        summary['test_auc_sd'] = float(tests.std(ddof=1)) if len(tests) > 1 else None
    return summary


def write_sealed_manifest(output_dir, entries, test_labels, test_manifest, identity):
    """Hash every sealed test-prediction file; no test metric is computed here."""
    labels = np.asarray(test_labels).astype(np.int8)
    names = [item['name'] for item in test_manifest['ordered_files']]
    manifest = {
        'schema': 'cr_sealed_v1',
        'created': _now_iso(),
        'run_dir': os.path.realpath(output_dir),
        'identity': identity,
        'metrics_computed': False,
        'test_label_identity': {
            'split': test_manifest['split'], 'dataset_root': test_manifest['dataset_root'],
            'n': int(labels.size),
            'labels_int8_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
            'ordered_file_names_sha256': _sha256_text('\n'.join(names)),
            'subject_ids_sha256': _sha256_text('\n'.join(test_manifest['subject_ids'])),
            'dataset_identity_kind': test_manifest.get('dataset_identity_kind'),
        },
        'files': entries,
        'unseal': 'python autopilot/cr_stats.py unseal --run <run dir>  (after the results freeze)',
    }
    path = os.path.join(output_dir, 'sealed_manifest.json')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    with open(path + '.sha256', 'w', encoding='utf-8') as stream:
        stream.write('%s  sealed_manifest.json\n' % file_sha256(path))
    return path


def _public_provenance(cache_prov):
    """Drop in-memory manifests before serializing cache provenance."""
    return {split: {kind: {k: v for k, v in rec.items() if not k.startswith('_')}
                    for kind, rec in kinds.items()}
            for split, kinds in cache_prov.items()}


def build_legacy_spec(config, ckpt_path, embed_dim, local_sha=None, local_prefix_key=None):
    legacy_cfg = config.get('legacy_cache') or {}
    run_dir = legacy_cfg.get('run_dir')
    if not run_dir:
        raise ValueError("legacy_cache_policy=allow_verified requires legacy_cache.run_dir "
                         "(--legacy-cache-run-dir)")
    data_cfg = config['data']
    pins_source = legacy_cfg.get('expected_cache_sha256')
    return {
        'run_dir': run_dir,
        'pins': load_legacy_pins(pins_source) if pins_source else None,
        'cache_dir': legacy_cfg.get('cache_dir'),
        'guard_json': legacy_cfg.get('guard_json'),
        'hf_manifest': legacy_cfg.get('hf_manifest'),
        'expected_checkpoint_sha256': legacy_cfg.get('expected_checkpoint_sha256'),
        'expected_epoch': legacy_cfg.get('expected_epoch'),
        'head_replay_atol': legacy_cfg.get('head_replay_atol', 1e-4),
        'train_replay_auc_atol': legacy_cfg.get('train_replay_auc_atol', 0.01),
        'train_replay_loss_atol': legacy_cfg.get('train_replay_loss_atol', 0.02),
        'checkpoint_path': ckpt_path,
        'local_checkpoint_sha256': local_sha,
        'local_checkpoint_prefix_key': local_prefix_key,
        'model_config': config['model'],
        'data_dir': data_cfg['data_dir'],
        'num_slices': data_cfg['num_slices'],
        'slice_size': data_cfg.get('slice_size', 256),
        'embed_dim': embed_dim,
    }


def run_patch_downstream(config, device):
    """Downstream glaucoma classification with I-JEPA pretrained encoder.

    Protocol:
      1. Pre-compute: encode all volumes with frozen ViT, cache to disk
         (optionally the per-B-scan patch-token max from the same pass)
      2. For each pooling variant and head seed: reset RNGs from the head
         seed, train the configured probe + head on cached features
      3. Early stop on val AUC (``training.patience``)
      4. Best-validation head -> test predictions; sealed (hashed, no
         metrics) when ``probe.seal_test`` is true
    """
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']
    probe_cfg = config.get('probe') or {}
    head_seeds = resolve_head_seeds(config)
    seal_test = bool(probe_cfg.get('seal_test', False))
    extra_pooling = normalize_extra_pooling(probe_cfg.get('extra_pooling'))
    if extra_pooling and model_cfg.get('probe_type', 'attentive') != 'mean_pool':
        raise ValueError("extra_pooling requires model.probe_type: mean_pool")

    output_dir = log_cfg.get('output_dir')
    ckpt_path = model_cfg.get('encoder_checkpoint')
    if not output_dir:
        raise ValueError("logging.output_dir must be set (--output-dir)")
    if not ckpt_path:
        raise ValueError("model.encoder_checkpoint must be set (--encoder-checkpoint)")
    _refuse_completed_output(output_dir, seal_test)
    os.makedirs(output_dir, exist_ok=True)
    started = _now_iso()

    print('=' * 70)
    print('Downstream Classification — Frozen Encoder Probe')
    print('=' * 70)
    print('  head seeds: %s | sealed test: %s | extra pooling: %s'
          % (','.join(str(s) for s in head_seeds), seal_test, extra_pooling or 'none'))

    vit_cfg = _VIT_CONFIGS[model_cfg['encoder_name']]
    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']
    slice_size = data_cfg.get('slice_size', 256)
    chunk_size = data_cfg.get('encode_chunk_size', 50)
    use_amp = data_cfg.get('use_amp', True)
    legacy_policy = data_cfg.get('legacy_cache_policy', 'reject')

    # ---- Load pretrained encoder -------------------------------------------
    encoder = None
    ckpt_epoch = None
    local_sha = enc_key = None
    if os.path.exists(ckpt_path):
        encoder = VisionTransformer(
            img_size=model_cfg['crop_size'],
            patch_size=model_cfg['patch_size'],
            embed_dim=vit_cfg['embed_dim'],
            depth=vit_cfg['depth'],
            num_heads=vit_cfg['num_heads'],
        ).to(device)
        print('Loading encoder from %s ...' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['target_encoder'])
        ckpt_epoch = ckpt.get('epoch', -1)
        print('  Loaded target_encoder weights (epoch %d)' % ckpt_epoch)
        del ckpt
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        local_sha = file_sha256(ckpt_path)
        # Recognize the historical short-prefix filename only for explicit legacy
        # rejection/opt-in. The v2 cache uses the complete source manifest.
        with open(ckpt_path, 'rb') as stream:
            enc_key = hashlib.sha256(stream.read(1 << 20)).hexdigest()[:12]
    elif legacy_policy != 'allow_verified':
        raise FileNotFoundError("Encoder checkpoint not found: %s (only "
                                "legacy_cache_policy=allow_verified may run from verified "
                                "legacy caches)" % ckpt_path)
    else:
        print('Encoder checkpoint not on disk: %s -> verified legacy caches only' % ckpt_path)

    legacy_spec = None
    sha_source = 'checkpoint file'
    ckpt_sha = local_sha
    if legacy_policy == 'allow_verified':
        legacy_spec = build_legacy_spec(config, ckpt_path, embed_dim, local_sha, enc_key)
        run_evidence = collect_legacy_run_evidence(legacy_spec)
        failed = [c['name'] for c in run_evidence['checks'] if c['required'] and not c['ok']]
        if failed:
            raise ValueError("Legacy run evidence failed for %s: %s"
                             % (legacy_spec['run_dir'], ', '.join(failed)))
        legacy_spec['_run_evidence'] = run_evidence
        if encoder is None:
            ckpt_sha = run_evidence['checkpoint_sha256']
            ckpt_epoch = run_evidence['checkpoint_epoch']
            sha_source = 'legacy probe guard sidecar (checkpoint not on disk)'

    # A training-run stop file lists the checkpoints it wrote; the probed one must be among them.
    provenance_role = None
    listed = (config.get('identity') or {}).get('run_provenance_checkpoints')
    if listed:
        matches = sorted((c for c in listed if c.get('sha256') == ckpt_sha),
                         key=lambda c: c.get('role') != 'periodic')
        if not matches:
            raise ValueError("Checkpoint SHA-256 %s is not listed in the run provenance %s"
                             % (ckpt_sha, config['identity'].get('run_provenance')))
        provenance_role = matches[0].get('role')

    # ---- Pre-compute features (one-time) -----------------------------------
    print('\n--- Pre-computing features with frozen encoder ---')
    # Apply it globally, not just to feature precompute: probe training, test
    # evaluation and the finetune paths all used autocast unconditionally.
    set_amp(use_amp)
    print('  precision: %s' % ('AMP fp16' if use_amp else 'fp32 (AMP disabled)'))
    cache_dir = os.path.join(output_dir, 'feature_cache')
    # Identify the cache by the encoder that produced it, so a config that
    # swaps checkpoints but keeps output_dir cannot reuse stale features.
    source = {'checkpoint_path': os.path.realpath(ckpt_path),
              'checkpoint_sha256': ckpt_sha,
              'checkpoint_component': 'target_encoder',
              'model_config': model_cfg}
    cache_kwargs = dict(
        use_amp=use_amp, cache_key=enc_key or '', source_identity=source, return_manifest=True,
        legacy_cache_policy=legacy_policy,
        num_workers=data_cfg.get('num_workers', 4),
        extra_pooling=extra_pooling, legacy_spec=legacy_spec)

    splits = ('Training', 'Validation', 'Test')
    data = {}
    cache_prov = {}
    for split in splits:
        cache_prov[split] = {}
        data[split] = precompute_features(
            encoder, data_cfg['data_dir'], split,
            num_slices, slice_size, device, chunk_size, cache_dir,
            provenance=cache_prov[split], **cache_kwargs)

    # Free encoder from GPU after feature extraction
    if encoder is not None:
        encoder.cpu()
        del encoder
    torch.cuda.empty_cache()

    if probe_cfg.get('hash_cache_files', True):
        for split in splits:
            for rec in cache_prov[split].values():
                if rec.get('path') and 'file_sha256' not in rec:
                    rec['file_sha256'] = (rec.get('evidence') or {}).get('cache_file_sha256') \
                        or file_sha256(rec['path'])
    for split in splits:
        states = ', '.join('%s=%s' % (kind, rec.get('state'))
                           for kind, rec in cache_prov[split].items())
        print('  cache %s: %s' % (split, states))
    with open(os.path.join(output_dir, 'cache_provenance.json'), 'w', encoding='utf-8') as stream:
        json.dump(_public_provenance(cache_prov), stream, indent=2)

    train_labels = data['Training'][1]
    n_pos = int(train_labels.sum().item())
    n_neg = len(train_labels) - n_pos
    print('  Train: %d volumes (%d pos, %d neg, %.1f%% prevalence)'
          % (len(train_labels), n_pos, n_neg, 100.0 * n_pos / len(train_labels)))
    print('  Val:   %d volumes' % len(data['Validation'][1]))
    print('  Test:  %d volumes' % len(data['Test'][1]))

    identity = dict(config.get('identity') or {})
    identity.update({'checkpoint_path': os.path.realpath(ckpt_path),
                     'checkpoint_sha256': ckpt_sha, 'checkpoint_sha256_source': sha_source,
                     'epoch': ckpt_epoch})
    if provenance_role is not None:
        identity['run_provenance_checkpoint_role'] = provenance_role

    # ---- Heads: pooling variant x head seed --------------------------------
    order = [PRIMARY_VARIANT]
    if extra_pooling:
        order += ['patchmean_slicemax', 'patchmean_slicemeanmax', 'patchmax_slicemean']
    variants = {}
    skipped = {}
    sealed_entries = []
    for position, variant in enumerate(order):
        variant_data = data
        if POOLING_VARIANTS[variant]['features'] == 'patch_max':
            missing = [s for s in splits
                       if cache_prov[s].get('patch_max', {}).get('state') not in ('cold', 'warm')]
            if missing:
                skipped[variant] = ('patch-max features unavailable for %s: %s'
                                    % (', '.join(missing),
                                       cache_prov[missing[0]].get('patch_max', {}).get('reason')))
                print('  Skipping %s: %s' % (variant, skipped[variant]))
                continue
            if all(POOLING_VARIANTS[v]['features'] == 'patch_max' for v in order[position:]):
                # Patch-mean features are no longer needed: keep peak RAM at one feature set.
                data = {split: (None,) + tuple(data[split][1:]) for split in splits}
            variant_data = {}
            for split in splits:
                rec = cache_prov[split]['patch_max']
                max_feats, max_labels = load_feature_cache(rec['path'], rec['_manifest'])
                if not torch.equal(max_labels.long(), data[split][1].long()):
                    raise ValueError("Patch-max cache labels differ from patch-mean (%s)" % split)
                variant_data[split] = (max_feats, data[split][1], rec['_manifest'])
        per_seed = []
        for head_seed in head_seeds:
            result = fit_probe_head(variant, head_seed, variant_data, config, device, output_dir,
                                    seal_test=seal_test, make_plots=(variant == PRIMARY_VARIANT))
            per_seed.append(result)
            if seal_test:
                sealed_entries.append({
                    'variant': variant, 'head_seed': head_seed,
                    'path': result['test_predictions'],
                    'sha256': result['test_predictions_sha256'],
                    'sidecar_path': result['test_predictions_sidecar'],
                    'sidecar_sha256': result['test_predictions_sidecar_sha256'],
                    'head_checkpoint_sha256': result['head_checkpoint_sha256']})
        variants[variant] = _summarize_variant(per_seed, seal_test)
        del variant_data

    sealed_path = None
    if seal_test:
        sealed_path = write_sealed_manifest(output_dir, sealed_entries, data['Test'][1],
                                            data['Test'][2], identity)

    primary = variants[PRIMARY_VARIANT]
    single = len(head_seeds) == 1
    first = primary['per_seed'][0]
    # Single-seed, unsealed, primary-only runs keep the historical top-level layout.
    if single and not seal_test and not extra_pooling:
        seed_dir = os.path.join(output_dir, PRIMARY_VARIANT, 'seed%d' % head_seeds[0])
        for name in ('best_model.pt', 'train_log.csv', 'val_predictions.npz',
                     'val_predictions.manifest.json', 'test_predictions.npz',
                     'test_predictions.manifest.json', 'diagnostic_plots.png'):
            if os.path.exists(os.path.join(seed_dir, name)):
                shutil.copy2(os.path.join(seed_dir, name), os.path.join(output_dir, name))

    results = {
        'schema': 'cr_probe_v1',
        'status': 'complete',
        'mode': 'patch',
        'head_type': model_cfg.get('head_type', 'linear'),
        'num_slices': num_slices,
        'probe_depth': model_cfg.get('probe_depth', 2),
        'head_seeds': head_seeds,
        'sealed': seal_test,
        'extra_pooling': extra_pooling or 'none',
        'primary_variant': PRIMARY_VARIANT,
        'best_epoch': first['best_epoch'] if single else [r['best_epoch'] for r in primary['per_seed']],
        'best_val_auc': primary['val_auc_mean'],
        'best_val_auc_sd': primary['val_auc_sd'],
        'test_auc': None if seal_test else primary['test_auc_mean'],
        'test_loss': first['test_loss'] if single else None,
        'sensitivity': first['sensitivity'] if single else None,
        'specificity': first['specificity'] if single else None,
        'probe_params': primary['probe_params'],
        'head_params': primary['head_params'],
        'variants': variants,
        'skipped_variants': skipped,
        'identity': identity,
        'legacy_cache_integrity': (lambda found: found[0] if len(found) == 1 else (found or None))(
            sorted(set(rec['evidence']['integrity'] for kinds in cache_prov.values()
                       for rec in kinds.values() if rec.get('state') == 'legacy_verified'))),
        'cache_provenance': _public_provenance(cache_prov),
        'sealed_manifest': os.path.basename(sealed_path) if sealed_path else None,
        'numeric_backend': _numeric_backend(device),
        'code': _code_identity(),
        'started': started,
        'finished': _now_iso(),
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to %s' % output_dir)
    print('Validation AUC by pooling variant (head seeds %s):'
          % ','.join(str(s) for s in head_seeds))
    for variant, summary in variants.items():
        sd = summary['val_auc_sd']
        print('  %-24s mean %.4f  sd %s  [%s]'
              % (variant, summary['val_auc_mean'], '%.4f' % sd if sd is not None else '-',
                 ', '.join('%.4f' % r['best_val_auc'] for r in summary['per_seed'])))
    for variant, reason in skipped.items():
        print('  %-24s skipped: %s' % (variant, reason))
    print('  best_val_auc = %.4f' % primary['val_auc_mean'])
    if seal_test:
        print('  test         = SEALED (%s)' % sealed_path)
    else:
        print('  test_auc     = %.4f' % primary['test_auc_mean'])
    print('  encoder: %s' % ckpt_path)

    return results


# ---------------------------------------------------------------------------
# Slice-level downstream
# ---------------------------------------------------------------------------

def evaluate_slice(encode_fn, head, loader, criterion, device):
    """Evaluate slice-level downstream (non-cached path)."""
    head.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            features = encode_fn(volumes)
            logits = head(features).squeeze(-1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    return avg_loss, auc


def run_slice_downstream(config, device):
    """Downstream evaluation using a slice-level I-JEPA pretrained encoder."""
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Downstream Classification (slice-level pretrained)')
    print('=' * 70)

    # ---- Frozen feature extractor ------------------------------------------
    fe_checkpoint = model_cfg.get('fe_checkpoint', None)
    feature_extractor = FrozenFeatureExtractor(checkpoint_path=fe_checkpoint).to(device)

    # ---- Load pretrained slice encoder -------------------------------------
    slice_encoder = SliceEncoder(
        num_slices=data_cfg['num_slices'],
        embed_dim=model_cfg['enc_dim'],
        depth=model_cfg['enc_depth'],
        num_heads=model_cfg['enc_heads'],
    ).to(device)

    ckpt_path = model_cfg['slice_encoder_checkpoint']
    print('Loading slice encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    slice_encoder.load_state_dict(ckpt['target_encoder'])
    print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    if model_cfg.get('freeze_encoder', True):
        for p in slice_encoder.parameters():
            p.requires_grad = False
        slice_encoder.eval()

    embed_dim = model_cfg['enc_dim']

    # ---- MLP head ----------------------------------------------------------
    head = MLPHead(in_dim=embed_dim).to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print('  Head params: %s' % format(head_params, ','))

    # ---- Encode function ---------------------------------------------------
    @torch.no_grad()
    def encode_fn(volumes):
        """Encode volume: frozen ConvNeXt -> frozen slice encoder -> mean pool."""
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        slice_features = feature_extractor(flat)  # (B*S, 768)
        slice_features = slice_features.reshape(B, S, -1)  # (B, S, 768)
        encoded = slice_encoder(slice_features)  # (B, S, D)
        pooled = encoded.mean(dim=1)  # (B, D)
        return pooled

    # ---- Datasets ----------------------------------------------------------
    num_slices = data_cfg['num_slices']
    slice_size = data_cfg.get('slice_size', 256)

    train_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Training'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    val_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Validation'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'],
                              shuffle=True, num_workers=data_cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'],
                            shuffle=False, num_workers=data_cfg['num_workers'],
                            pin_memory=True)

    print('  Train: %d volumes' % len(train_dataset))
    print('  Val:   %d volumes' % len(val_dataset))

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=3, total_epochs=train_cfg['epochs'],
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=_USE_AMP)

    # ---- Training loop -----------------------------------------------------
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)

    for epoch in range(1, train_cfg['epochs'] + 1):
        head.train()
        total_loss = 0.0
        n_samples = 0

        t0 = time.time()
        for volumes, labels in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with amp_ctx():
                features = encode_fn(volumes)
                logits = head(features).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if optimizer_step(optimizer, scaler):
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_samples, 1)
        val_loss, val_auc = evaluate_slice(encode_fn, head, val_loader, criterion, device)

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print('Epoch %d/%d (%4.0fs) | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f%s'
              % (epoch, train_cfg['epochs'], elapsed, train_loss, val_loss, val_auc, marker))

        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'head': head.state_dict(),
                'val_auc': val_auc,
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch %d' % epoch)
                break

    # ---- Test evaluation ---------------------------------------------------
    test_dir = os.path.join(data_cfg['data_dir'], 'Test')
    test_auc = None
    test_loss = None
    if os.path.isdir(test_dir):
        best_ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
        head.load_state_dict(best_ckpt['head'])

        test_dataset = OCTVolumeDataset(test_dir, num_slices=num_slices,
                                        slice_size=slice_size, return_label=True)
        test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'],
                                 shuffle=False, num_workers=data_cfg['num_workers'],
                                 pin_memory=True)
        print('  Test: %d volumes' % len(test_dataset))
        test_loss, test_auc = evaluate_slice(encode_fn, head, test_loader, criterion, device)
        print('TEST Loss: %.4f | TEST AUC: %.4f' % (test_loss, test_auc))
    else:
        print('No Test directory found, skipping test evaluation.')

    # ---- Save results ------------------------------------------------------
    results = {
        'mode': 'slice',
        'best_val_auc': best_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to %s' % output_dir)

    return results


# ---------------------------------------------------------------------------
# Combined model for DDP fine-tuning
# ---------------------------------------------------------------------------

class DownstreamModel(nn.Module):
    """End-to-end model: ViT encoder + AttentiveProbe + head.

    Wraps the full pipeline so DDP can sync gradients correctly.
    Encodes slices in chunks to fit memory while preserving gradients.
    """

    def __init__(self, encoder, probe, head, chunk_size=25):
        super(DownstreamModel, self).__init__()
        self.encoder = encoder
        self.probe = probe
        self.head = head
        self.chunk_size = chunk_size

    def forward(self, volumes):
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        flat = imagenet_normalize(flat)  # match pretraining distribution
        parts = []
        for i in range(0, flat.size(0), self.chunk_size):
            chunk = flat[i:i + self.chunk_size]
            out = self.encoder(chunk)          # (chunk, patches, D)
            parts.append(out.mean(dim=1))      # (chunk, D)
        features = torch.cat(parts, dim=0)     # (B*S, D)
        features = features.reshape(B, S, -1)  # (B, S, D)
        pooled = self.probe(features)          # (B, D)
        return self.head(pooled).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Patch-level fine-tuning (encoder unfrozen, DDP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_finetune(model, loader, criterion, device, return_predictions=False):
    """Evaluate fine-tune model on a data loader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with amp_ctx():
            logits = model(volumes)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Gather across ranks for full AUC
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_labels = [None] * dist.get_world_size()
        gathered_probs = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_probs, all_probs)
        all_labels = np.concatenate(gathered_labels)
        all_probs = np.concatenate(gathered_probs)

        # Gather loss across ranks
        loss_tensor = torch.tensor([total_loss, float(n_samples)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        n_samples = int(loss_tensor[1].item())

    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    if return_predictions:
        return avg_loss, auc, all_labels, all_probs
    return avg_loss, auc


def run_patch_finetune(config, device, rank=0, world_size=1):
    """Fine-tune encoder + probe + head end-to-end with DDP.

    Protocol:
      - Encoder: very low LR (5e-6), unfrozen
      - Probe + head: normal LR
      - batch_size=1 per GPU, gradient accumulation, DDP
      - Early stop on val AUC, patience=5
    """
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    is_main = (rank == 0)

    if is_main:
        print('=' * 70)
        print('Downstream Fine-tuning — Encoder + Probe + Head (DDP)')
        print('  World size: %d' % world_size)
        print('=' * 70)

    # ---- Build model -------------------------------------------------------
    vit_cfg = _VIT_CONFIGS[model_cfg['encoder_name']]
    encoder = VisionTransformer(
        img_size=model_cfg['crop_size'],
        patch_size=model_cfg['patch_size'],
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    )

    ckpt_path = model_cfg['encoder_checkpoint']
    if is_main:
        print('Loading encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(ckpt['target_encoder'])
    if is_main:
        print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']

    probe_type = model_cfg.get('probe_type', 'attentive')
    # _build_probe fails fast on unknown probe_type; safe here.
    probe, probe_desc = _build_probe(
        probe_type, num_slices, embed_dim, model_cfg, device='cpu',
    )

    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1))
    else:
        head = LinearHead(in_dim=embed_dim)

    chunk_size = data_cfg.get('encode_chunk_size', 25)
    model = DownstreamModel(encoder, probe, head, chunk_size).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw = model.module if hasattr(model, 'module') else model

    enc_params = sum(p.numel() for p in raw.encoder.parameters())
    probe_params = sum(p.numel() for p in raw.probe.parameters())
    head_params = sum(p.numel() for p in raw.head.parameters())
    if is_main:
        print('  Encoder:  %s params (trainable, lr=%.1e)'
              % (format(enc_params, ','), train_cfg.get('lr_encoder', 5e-6)))
        print('  Probe (%s): %s params (trainable, lr=%.1e)'
              % (probe_desc, format(probe_params, ','), train_cfg.get('lr_probe', 1e-4)))
        print('  Head:     %s params (trainable, lr=%.1e)'
              % (format(head_params, ','), train_cfg.get('lr_head', 1e-3)))

    # ---- Datasets ----------------------------------------------------------
    slice_size = data_cfg.get('slice_size', 256)
    batch_size = data_cfg.get('batch_size', 1)
    accum_steps = train_cfg.get('accum_steps', 4)

    train_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Training'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    val_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Validation'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=data_cfg.get('num_workers', 2), pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=data_cfg.get('num_workers', 2), pin_memory=True)

    eff_batch = batch_size * world_size * accum_steps
    if is_main:
        print('  Train: %d volumes  (bs=%d × %d GPUs × %d accum = %d eff)'
              % (len(train_dataset), batch_size, world_size, accum_steps, eff_batch))
        print('  Val:   %d volumes' % len(val_dataset))

    # ---- Optimizer ---------------------------------------------------------
    # LLRD if train_cfg['layer_decay'] in (0, 1); flat otherwise. By convention
    # groups[0] = deepest encoder layer (embed in LLRD; whole encoder in flat),
    # groups[-2] = probe, groups[-1] = head. Logging reads these three indices.
    param_groups, pg_mode = build_finetune_param_groups(
        raw.encoder, raw.probe, raw.head, train_cfg,
    )
    if is_main:
        if pg_mode == 'llrd':
            lr_top_block = param_groups[-4]['lr']  # encoder.blocks[-1]
            print('  Optimizer: AdamW + LLRD (decay=%.2f, %d groups)'
                  % (train_cfg.get('layer_decay', 1.0), len(param_groups)))
            print('    LR range: embed=%.2e .. top_block=%.2e .. head=%.2e'
                  % (param_groups[0]['lr'], lr_top_block, param_groups[-1]['lr']))
        else:
            print('  Optimizer: AdamW + flat (encoder=%.2e, probe=%.2e, head=%.2e)'
                  % (param_groups[0]['lr'], param_groups[-2]['lr'], param_groups[-1]['lr']))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], max(steps_per_epoch, 1),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=_USE_AMP)

    # ---- CSV logger --------------------------------------------------------
    csv_file = None
    if is_main:
        csv_path = os.path.join(output_dir, 'train_log.csv')
        csv_file = open(csv_path, 'w')
        csv_file.write('epoch,train_loss,val_loss,val_auc,lr_enc,lr_probe,lr_head,elapsed_s\n')
        csv_file.flush()

    # ---- Training loop -----------------------------------------------------
    if is_main:
        print('\n--- Training ---')
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_samples = 0
        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        for step, (volumes, labels) in enumerate(train_loader):
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with amp_ctx():
                logits = model(volumes)
                window_size = min(accum_steps, len(train_loader) - (step // accum_steps) * accum_steps)
                loss = criterion(logits, labels) / window_size

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if optimizer_step(optimizer, scaler):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * window_size * labels.size(0)
            n_samples += labels.size(0)

        elapsed = time.time() - t0
        # Aggregate train_loss across ranks so the logged curve matches the
        # global training loss, not rank 0's shard only.
        if dist.is_initialized() and world_size > 1:
            stats = torch.tensor(
                [total_loss, float(n_samples)], device=device, dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = (stats[0] / stats[1]).item()
        else:
            train_loss = total_loss / max(n_samples, 1)
        val_loss, val_auc = evaluate_finetune(model, val_loader, criterion, device)
        # Read LRs by named tag rather than index so empty groups (e.g. MeanPool
        # probe with 0 trainable params, filtered out) don't shift indices
        # and cause the wrong LR to be reported.
        _grp_lr = {g.get('name', f'g{i}'): g['lr']
                   for i, g in enumerate(optimizer.param_groups)}
        lr_enc = _grp_lr.get('embed', _grp_lr.get('encoder', 0.0))
        lr_probe = _grp_lr.get('probe', 0.0)
        lr_head = _grp_lr.get('head', 0.0)

        should_stop = False
        # Track best_model across all epochs (warmup included). Val AUC on
        # a supervised task is a real metric, not an EMA-target artifact —
        # if the model happens to peak during LR ramp-up, that's a
        # legitimate checkpoint to keep. We still gate early-stop triggers
        # on past_warmup so a noisy warmup dip can't prematurely end the
        # run.
        past_warmup = epoch > train_cfg.get('warmup_epochs', 3)
        if is_main:
            improved = val_auc > best_auc
            marker = ' *' if improved else ''
            print('Epoch %2d/%d (%5.0fs) | Train: %.4f | Val: %.4f | AUC: %.4f | LR: %.1e/%.1e/%.1e%s'
                  % (epoch, epochs, elapsed, train_loss, val_loss, val_auc,
                     lr_enc, lr_probe, lr_head, marker))

            if csv_file:
                csv_file.write('%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.1f\n'
                               % (epoch, train_loss, val_loss, val_auc,
                                  lr_enc, lr_probe, lr_head, elapsed))
                csv_file.flush()

            if improved:
                best_auc = val_auc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder': raw.encoder.state_dict(),
                    'probe': raw.probe.state_dict(),
                    'head': raw.head.state_dict(),
                    'val_auc': val_auc,
                }, os.path.join(output_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                # Only allow early-stop after warmup so a single noisy warmup
                # epoch doesn't kill a run whose real training hasn't started.
                if past_warmup and patience_counter >= patience:
                    print('Early stopping at epoch %d (patience=%d)' % (epoch, patience))
                    should_stop = True

        # Broadcast early stop decision — ALL ranks must reach this
        if world_size > 1:
            stop_tensor = torch.tensor([should_stop], device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
        elif should_stop:
            break

    if csv_file:
        csv_file.close()

    # ---- Tear down DDP before test eval (prevents NCCL timeout) ------------
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    # ---- Test evaluation (rank 0 only, no DDP) -----------------------------
    if is_main:
        print('\n--- Test Evaluation ---')
        best_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device)
            raw.encoder.load_state_dict(best_ckpt['encoder'])
            raw.probe.load_state_dict(best_ckpt['probe'])
            raw.head.load_state_dict(best_ckpt['head'])
            best_epoch = best_ckpt['epoch']
        else:
            best_epoch = 0

        test_dataset = OCTVolumeDataset(
            os.path.join(data_cfg['data_dir'], 'Test'),
            num_slices=num_slices, slice_size=slice_size, return_label=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        test_model = raw.to(device)
        test_loss, test_auc, test_labels, test_probs = evaluate_finetune(
            test_model, test_loader, criterion, device, return_predictions=True)
        print('Best epoch: %d  |  Val AUC: %.4f  |  TEST AUC: %.4f'
              % (best_epoch, best_auc, test_auc))

        # Sensitivity / specificity at threshold=0.5
        sensitivity = specificity = None
        if test_labels is not None:
            test_preds = (test_probs >= 0.5).astype(int)
            tp = ((test_preds == 1) & (test_labels == 1)).sum()
            tn = ((test_preds == 0) & (test_labels == 0)).sum()
            fp = ((test_preds == 1) & (test_labels == 0)).sum()
            fn = ((test_preds == 0) & (test_labels == 1)).sum()
            sensitivity = float(tp / max(tp + fn, 1))
            specificity = float(tn / max(tn + fp, 1))
            print('  Sensitivity: %.4f  |  Specificity: %.4f  (threshold=0.5)'
                  % (sensitivity, specificity))

        # Save predictions
        test_manifest = dataset_order_manifest(
            test_dataset, 'Test',
            {'checkpoint_path': os.path.realpath(best_path),
             'checkpoint_sha256': file_sha256(best_path) if os.path.isfile(best_path) else None,
             'selected_checkpoint_exists': os.path.isfile(best_path),
             'model_config': model_cfg, 'training_config': train_cfg})
        test_manifest['use_amp'] = _USE_AMP
        save_predictions(os.path.join(output_dir, 'test_predictions.npz'),
                         test_labels, test_probs, test_manifest)
        print('  Saved test_predictions.npz (%d samples)' % len(test_labels))

        # Diagnostic plots
        _save_diagnostic_plots(output_dir, test_labels, test_probs, test_auc,
                               None, None)

        results = {
            'mode': 'patch_finetune',
            'head_type': head_type,
            'num_slices': num_slices,
            'probe_depth': model_cfg.get('probe_depth', 2),
            'best_epoch': best_epoch,
            'best_val_auc': best_auc,
            'test_auc': test_auc,
            'test_loss': test_loss,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'lr_encoder': train_cfg.get('lr_encoder', 5e-6),
            'accum_steps': accum_steps,
            'effective_batch': eff_batch,
            'config': config,
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print('\nResults saved to %s' % output_dir)
        print('  best_val_auc = %.4f' % best_auc)
        print('  test_auc     = %.4f' % test_auc)
        print('  encoder: %s' % config.get('model', {}).get('encoder_checkpoint', 'unknown'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ARMS = ('random', 'centroid', 'envelope', 'random_cb')
_ROLES = ('new', 'anchor')


def _identity_from_provenance(path):
    """Read run identity (UUID, arm, seed) from a training-run provenance JSON."""
    with open(path, 'r', encoding='utf-8') as stream:
        prov = json.load(stream)
    found = {'run_provenance': os.path.realpath(path),
             'run_provenance_sha256': file_sha256(path)}
    for key, aliases in (('run_uuid', ('run_uuid', 'run_id', 'uuid')),
                         ('arm', ('arm',)),
                         ('train_seed', ('train_seed', 'seed'))):
        for alias in aliases:
            if prov.get(alias) is not None:
                found[key] = prov[alias]
                found['run_provenance_' + key] = prov[alias]
                break
    # Trainer stop files / run manifests carry no arm; the campaign's write tag does.
    tag = re.search(r'cr_seed_v1_(random_cb|random|centroid|envelope)_s(\d+)$',
                    str(prov.get('write_tag') or ''))
    if tag and 'arm' not in found:
        found['arm'] = found['run_provenance_arm'] = tag.group(1)
    if prov.get('schema'):
        found['run_provenance_schema'] = prov['schema']
    if isinstance(prov.get('checkpoints'), list):
        found['run_provenance_checkpoints'] = [
            {'role': c.get('role'), 'sha256': c.get('sha256'), 'epoch': c.get('epoch'),
             'file': c.get('file')} for c in prov['checkpoints'] if isinstance(c, dict)]
    return found


def apply_cli_overrides(config, args):
    """Return a copy of ``config`` with command-line overrides applied."""
    config = copy.deepcopy(config)
    for section in ('data', 'model', 'training', 'logging'):
        config.setdefault(section, {})
    if getattr(args, 'encoder_checkpoint', None):
        config['model']['encoder_checkpoint'] = args.encoder_checkpoint
    if getattr(args, 'output_dir', None):
        config['logging']['output_dir'] = args.output_dir
    probe = dict(config.get('probe') or {})
    if getattr(args, 'head_seeds', None):
        probe['head_seeds'] = [int(s) for s in args.head_seeds.split(',') if s.strip()]
    if getattr(args, 'seal_test', None) is not None:
        probe['seal_test'] = bool(args.seal_test)
    if getattr(args, 'extra_pooling', None) is not None:
        probe['extra_pooling'] = normalize_extra_pooling(args.extra_pooling)
    if probe:
        config['probe'] = probe
    if getattr(args, 'legacy_cache_policy', None):
        config['data']['legacy_cache_policy'] = args.legacy_cache_policy
    legacy = dict(config.get('legacy_cache') or {})
    for attr, key in (('legacy_cache_run_dir', 'run_dir'), ('legacy_guard_json', 'guard_json'),
                      ('legacy_hf_manifest', 'hf_manifest'),
                      ('legacy_cache_hashes', 'expected_cache_sha256'),
                      ('expected_checkpoint_sha256', 'expected_checkpoint_sha256')):
        if getattr(args, attr, None):
            legacy[key] = getattr(args, attr)
    if legacy:
        config['legacy_cache'] = legacy
    identity = dict(config.get('identity') or {})
    if getattr(args, 'run_provenance', None):
        identity.update(_identity_from_provenance(args.run_provenance))
    for attr in ('arm', 'train_seed', 'run_uuid', 'role'):
        if getattr(args, attr, None) is not None:
            identity[attr] = getattr(args, attr)
    if identity:
        if identity.get('arm') is not None and identity['arm'] not in _ARMS:
            raise ValueError("identity.arm must be one of %s" % (_ARMS,))
        if identity.get('role') is not None and identity['role'] not in _ROLES:
            raise ValueError("identity.role must be one of %s" % (_ROLES,))
        if identity.get('train_seed') is not None:
            identity['train_seed'] = int(identity['train_seed'])
        # A campaign label that contradicts the training run's own provenance is refused.
        for key in ('train_seed', 'arm'):
            recorded = identity.get('run_provenance_' + key)
            if recorded is not None and str(recorded) != str(identity.get(key)):
                raise ValueError("identity.%s=%r contradicts run provenance %r"
                                 % (key, identity.get(key), recorded))
        config['identity'] = identity
    return config


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Downstream glaucoma classification')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--encoder-checkpoint', help='Override model.encoder_checkpoint')
    parser.add_argument('--output-dir', help='Override logging.output_dir')
    parser.add_argument('--head-seeds', help='Comma-separated head seeds, e.g. 42,43,44,45,46')
    parser.add_argument('--seal-test', dest='seal_test', action='store_true',
                        help='Write hashed test predictions without computing test metrics')
    parser.add_argument('--no-seal-test', dest='seal_test', action='store_false')
    parser.add_argument('--extra-pooling', choices=EXTRA_POOLING_CHOICES,
                        help='patch_max: also cache per-B-scan patch-token max and fit '
                             'the pooling variants')
    parser.add_argument('--legacy-cache-policy', choices=_LEGACY_CACHE_POLICIES)
    parser.add_argument('--legacy-cache-run-dir',
                        help='Original probe directory holding legacy fp32 caches')
    parser.add_argument('--legacy-guard-json', help='Guard sidecar with the checkpoint SHA-256')
    parser.add_argument('--legacy-hf-manifest', help='checkpoints_hf MANIFEST.json cross-check')
    parser.add_argument('--legacy-cache-hashes',
                        help='JSON of pinned SHA-256 for the legacy caches and evidence files '
                             '(required by allow_verified; trust on first use)')
    parser.add_argument('--expected-checkpoint-sha256',
                        help='Required SHA-256 of the named checkpoint (legacy caches)')
    parser.add_argument('--arm', choices=_ARMS)
    parser.add_argument('--train-seed', type=int)
    parser.add_argument('--run-uuid')
    parser.add_argument('--role', choices=_ROLES)
    parser.add_argument('--run-provenance', help='Training-run provenance JSON (run_uuid/arm/seed)')
    parser.set_defaults(seal_test=None)
    return parser


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = apply_cli_overrides(config, args)

    seed = config.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    freeze_encoder = config.get('model', {}).get('freeze_encoder', True)

    # Set precision ONCE here, before any mode dispatches. Setting it inside
    # run_patch_downstream only covered the frozen-probe path, so fine-tune and
    # slice configs specifying `use_amp: false` silently still ran fp16.
    set_amp(config.get('data', {}).get('use_amp', True))

    if not freeze_encoder:
        # DDP mode for fine-tuning
        world_size, rank = init_distributed()
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        if rank == 0:
            print('GPU: %s' % torch.cuda.get_device_name(0))
        run_patch_finetune(config, device, rank, world_size)
    else:
        # Single GPU for frozen probe
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('GPU: %s' % torch.cuda.get_device_name(0))
        mode = config.get('mode', 'patch')
        if mode == 'patch':
            run_patch_downstream(config, device)
        elif mode == 'slice':
            run_slice_downstream(config, device)
        else:
            raise ValueError("Unknown mode: %s" % mode)


if __name__ == '__main__':
    # Line-buffer stdout so per-epoch prints appear in real time under `tee`
    # (default block buffering hides progress until ~4KB accumulates).
    sys.stdout.reconfigure(line_buffering=True)
    args = build_arg_parser().parse_args()
    main(args)
