"""
Downstream glaucoma classification using pretrained I-JEPA encoder.

Supports both patch-level and slice-level pretrained models:
  - Patch-level: each slice is encoded by the frozen ViT, mean-pooled to one
    token per slice, then a trainable attentive probe (single transformer
    block with learnable [CLS] token) aggregates across slices, followed by
    a linear classifier.  Follows the I-JEPA evaluation protocol (Assran
    et al., 2023).
  - Slice-level: slices are encoded by frozen ConvNeXt + frozen slice encoder,
    then mean-pooled and classified by a trainable MLP head.

Usage:
    # Patch-level pretrained -> AttentiveProbe + Linear
    python eval_downstream.py --config configs/downstream_patch.yaml

    # Slice-level pretrained -> MLP only
    python eval_downstream.py --config configs/downstream_slice.yaml

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
from src.models.anatomical_moe_pool import AnatomicalMoEPool

_PROBE_TYPES = ('attentive', 'cross_attn_pool', 'mean_pool')
_POOL_TYPES = ('mean', 'anatomical_moe')
_MOE_SCOPES = ('per_slice', 'volume')
_AXIAL_POS_TYPES = ('none', 'learned', 'sincos')


def _build_aggregator(model_cfg, embed_dim, num_slices, device):
    """Build the patch aggregator and report the resulting probe shape.

    Args:
        num_slices: dataset's per-volume slice count, needed to compute the
            total token count the probe will see.

    Returns a tuple:
        aggregator: nn.Module or None (None for mean-pool path)
        probe_num_tokens: int — sequence length the probe will see
        moe_scope: 'per_slice' | 'volume' | 'none'  (selects DownstreamModel branch)
        axial_pos_embed_type: 'none' | 'learned' | 'sincos' (volume scope only)
        desc: human-readable summary

    Pool-type → behavior:
      - 'mean' (default): aggregator=None, probe sees (B, num_slices, D).
      - 'anatomical_moe' + moe_scope='per_slice': aggregator runs once per
        slice; probe sees (B, num_slices*E*S, D).
      - 'anatomical_moe' + moe_scope='volume': aggregator runs ONCE on the
        concatenated (B, num_slices*P, D) patch sequence; probe sees
        (B, E*S, D). For OCT, pair this with axial_pos_embed_type='learned'
        so prototypes can specialize by axial slice position.
    """
    pool_type = model_cfg.get('pool_type', 'mean')
    if pool_type not in _POOL_TYPES:
        raise ValueError(
            "Unknown pool_type=%r. Valid values: %s"
            % (pool_type, ', '.join(_POOL_TYPES)))
    if pool_type == 'mean':
        return None, num_slices, 'none', 'none', 'mean'

    cfg = model_cfg.get('anatomical_moe', {}) or {}
    moe_scope = cfg.get('moe_scope', 'per_slice')
    axial_type = cfg.get('axial_pos_embed', 'none')
    skip_wq = cfg.get('skip_wq', False)
    if moe_scope not in _MOE_SCOPES:
        raise ValueError("Unknown moe_scope=%r. Valid: %s"
                         % (moe_scope, ', '.join(_MOE_SCOPES)))
    if axial_type not in _AXIAL_POS_TYPES:
        raise ValueError("Unknown axial_pos_embed=%r. Valid: %s"
                         % (axial_type, ', '.join(_AXIAL_POS_TYPES)))

    aggregator = AnatomicalMoEPool(
        embed_dim=embed_dim,
        num_experts=cfg.get('num_experts', 8),
        num_slots=cfg.get('num_slots', 4),
        num_heads=cfg.get('num_heads', 8),
        slot_dim=cfg.get('slot_dim', 256),
        lora_rank=cfg.get('lora_rank', 16),
        share_phi=cfg.get('share_phi', True),
        skip_wq=skip_wq,
        dropout=cfg.get('dropout', 0.0),
    ).to(device)

    n_proto = aggregator.num_experts * aggregator.num_slots  # E*S
    if moe_scope == 'per_slice':
        probe_num_tokens = num_slices * n_proto
    else:  # 'volume'
        probe_num_tokens = n_proto

    desc = ('anatomical_moe (E=%d S=%d H=%d, scope=%s, skip_wq=%s, axial=%s '
            '-> probe sees %d tokens)'
            % (aggregator.num_experts, aggregator.num_slots,
               aggregator.num_heads, moe_scope, skip_wq, axial_type,
               probe_num_tokens))
    return aggregator, probe_num_tokens, moe_scope, axial_type, desc


def _build_probe(probe_type, num_tokens, embed_dim, model_cfg, device,
                 use_pos_embed=True):
    """Instantiate the slice-aggregation probe. Fails fast on unknown types.

    Args:
        num_tokens: sequence length the probe will see (caller computes this
            from the aggregator: num_slices for mean, num_slices*E*S for
            per-slice MoE, E*S for volume MoE).
        use_pos_embed: True for mean-pool baseline; False for MoE paths.
            (For volume-scope MoE, axial position info is injected BEFORE
            the aggregator via DownstreamModel.axial_pos_embed, so the probe
            should not add another position embedding on top.)
    """
    if probe_type not in _PROBE_TYPES:
        raise ValueError(
            "Unknown probe_type=%r. Valid values: %s"
            % (probe_type, ', '.join(_PROBE_TYPES))
        )
    if probe_type == 'mean_pool':
        probe = MeanPool(num_slices=num_tokens, embed_dim=embed_dim).to(device)
        desc = 'mean_pool (0 params, ablation floor)'
    elif probe_type == 'cross_attn_pool':
        head_dim = model_cfg.get('probe_head_dim', 64)
        probe = CrossAttnPool(
            num_slices=num_tokens, embed_dim=embed_dim, head_dim=head_dim,
            use_pos_embed=use_pos_embed,
        ).to(device)
        desc = ('cross_attn_pool (head_dim=%d, n_tokens=%d, pos_embed=%s)'
                % (head_dim, num_tokens, use_pos_embed))
    else:  # 'attentive'
        depth = model_cfg.get('probe_depth', 2)
        probe = AttentiveProbe(
            num_slices=num_tokens,
            embed_dim=embed_dim,
            num_heads=model_cfg.get('probe_num_heads', 12),
            depth=depth,
            use_pos_embed=use_pos_embed,
        ).to(device)
        desc = ('attentive (depth=%d, n_tokens=%d, pos_embed=%s)'
                % (depth, num_tokens, use_pos_embed))
    return probe, desc
try:
    from src.models.feature_extractor import FrozenFeatureExtractor
except ImportError:
    FrozenFeatureExtractor = None  # Slice-level approach (archived)
from src.datasets.oct_volumes import OCTVolumeDataset
from src.helper import _VIT_CONFIGS
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

    def __init__(self, num_slices=100, embed_dim=768, num_heads=12, depth=2,
                 use_pos_embed=True):
        super(AttentiveProbe, self).__init__()
        self.use_pos_embed = use_pos_embed
        # NB: param creation and init order preserved to match pre-flag layout.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, embed_dim))
        else:
            self.pos_embed = None
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """x: (B, N, D) -> (B, D). N is num_slices in mean-pool path, or
        num_slices*E*S in AnatomicalMoEPool path. When use_pos_embed=False,
        we do not apply additive position embedding (encoder pos info is
        already baked into the patch features that prototypes summarize).
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, N+1, D)
        if self.use_pos_embed:
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


def build_finetune_param_groups(encoder, probe, head, train_cfg, aggregator=None,
                                axial_pos_embed=None):
    """Build AdamW param_groups for fine-tuning.

    If ``layer_decay`` in train_cfg is strictly in (0, 1), applies MAE-style
    Layer-wise LR Decay (LLRD) to the encoder:
      - base LR = ``lr_probe`` (also used for probe / encoder.norm)
      - encoder.patch_embed + pos_embed: base * decay^(num_layers)
      - encoder.blocks[i]:               base * decay^(num_layers - (i+1))
      - encoder.norm + probe + aggregator: base
      - head:                            ``lr_head`` (usually = base)
    For ViT-B/16 with 12 blocks and decay=0.65, the deepest layer gets
    ~0.65^13 ≈ 5.69e-3 of base; the top block gets 0.65 of base.

    If ``layer_decay`` is missing or >= 1.0, falls back to the older flat
    setup (encoder, [aggregator], probe, head) so previous configs still work.

    Args:
        aggregator: optional within-slice aggregation module (e.g.
            AnatomicalMoEPool). If provided, its parameters are added at the
            base/probe LR. Pass None to retain the prior 3-group behavior.

    Returns ``(param_groups, mode)`` where mode is 'llrd' or 'flat'.
    The deepest encoder group is always groups[0]; the head is always
    groups[-1]. The probe and (if present) aggregator sit just before head.
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
        if aggregator is not None:
            groups.append({'params': list(aggregator.parameters()), 'lr': base_lr,
                           'name': 'aggregator'})
        if axial_pos_embed is not None:
            groups.append({'params': [axial_pos_embed], 'lr': base_lr,
                           'name': 'axial_pos_embed'})
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
    ]
    if aggregator is not None:
        groups.append({'params': list(aggregator.parameters()), 'lr': lr_probe,
                       'name': 'aggregator'})
    if axial_pos_embed is not None:
        groups.append({'params': [axial_pos_embed], 'lr': lr_probe,
                       'name': 'axial_pos_embed'})
    groups.append({'params': list(probe.parameters()), 'lr': lr_probe, 'name': 'probe'})
    groups.append({'params': list(head.parameters()), 'lr': lr_head, 'name': 'head'})
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

        with autocast():
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

def precompute_features(encoder, data_dir, split, num_slices, slice_size,
                        device, chunk_size=50, cache_dir=None,
                        keep_patches=False):
    """Encode all volumes in a split with the frozen ViT and cache to disk.

    Args:
        keep_patches: if False (default, mean-pool path), per-slice features
            are mean-pooled across patches → (N, num_slices, embed_dim).
            If True (anatomical-MoE path), the raw patch features are kept
            → (N, num_slices, num_patches, embed_dim). The downstream
            aggregator (AnatomicalMoEPool) is then applied at probe time.

    Returns:
        features: (N, num_slices, embed_dim) if keep_patches=False
                  (N, num_slices, num_patches, embed_dim) if keep_patches=True
        labels:   (N,) long
    """
    cache_path = None
    if cache_dir:
        # Preserve original cache filename for the mean-pool path so existing
        # caches remain valid. Only the new keep_patches path uses a suffix.
        if keep_patches:
            cache_path = os.path.join(
                cache_dir, '%s_s%d_patches.pt' % (split, num_slices))
        else:
            cache_path = os.path.join(cache_dir, '%s_s%d.pt' % (split, num_slices))
        if os.path.exists(cache_path):
            print('  Loading cached %s features from %s' % (split, cache_path))
            data = torch.load(cache_path, map_location='cpu')
            return data['features'], data['labels']

    split_dir = os.path.join(data_dir, split)
    dataset = OCTVolumeDataset(
        split_dir, num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features = []
    all_labels = []

    encoder.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (volume, label) in enumerate(loader):
            volume = volume.to(device)       # (1, S, 3, H, W)
            flat = volume.squeeze(0)          # (S, 3, H, W)

            parts = []
            for j in range(0, flat.size(0), chunk_size):
                chunk = flat[j:j + chunk_size]
                chunk = imagenet_normalize(chunk)  # match pretraining distribution
                with autocast():
                    out = encoder(chunk)      # (chunk, patches, D)
                if keep_patches:
                    parts.append(out.cpu())              # (chunk, P, D)
                else:
                    parts.append(out.mean(dim=1).cpu())  # (chunk, D)

            all_features.append(torch.cat(parts, dim=0))  # (S, [P,] D)
            all_labels.append(label.squeeze())

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print('    %s: %d/%d volumes (%.0fs)'
                      % (split, i + 1, len(dataset), elapsed))

    features = torch.stack(all_features)     # (N, S, [P,] D)
    labels = torch.stack(all_labels).long()  # (N,)
    elapsed = time.time() - t0
    print('  %s: %d volumes encoded in %.0fs (%.1f vol/s)'
          % (split, len(dataset), elapsed, len(dataset) / max(elapsed, 1)))

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'features': features, 'labels': labels}, cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print('  Cached to %s (%.1f MB)' % (cache_path, size_mb))

    return features, labels


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

def run_patch_downstream(config, device):
    """Downstream glaucoma classification with I-JEPA pretrained encoder.

    Protocol:
      1. Pre-compute: encode all volumes with frozen ViT, cache to disk
      2. Train: AttentiveProbe (2 blocks) + LinearHead on cached features
      3. Early stop on val AUC, patience=5
      4. Evaluate best model on test set, report test AUC
    """
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Downstream Classification — I-JEPA Attentive Probe')
    print('=' * 70)

    # ---- Load pretrained encoder -------------------------------------------
    vit_cfg = _VIT_CONFIGS[model_cfg['encoder_name']]
    encoder = VisionTransformer(
        img_size=model_cfg['crop_size'],
        patch_size=model_cfg['patch_size'],
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    ).to(device)

    ckpt_path = model_cfg['encoder_checkpoint']
    print('Loading encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['target_encoder'])
    print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']

    # ---- Hard guard: AnatomicalMoEPool is not yet wired for frozen probe ---
    # The cached-feature frozen-probe path stores mean-pooled (N, S, D) features.
    # AnatomicalMoEPool needs raw (N, S, P, D) patches AND requires its
    # parameters to be in the optimizer with gradients during probe training.
    # Wiring those changes is non-trivial (cache size 150-800 GB at fp32, plus
    # optimizer plumbing). Until done, fail fast rather than silently run the
    # mean-pool path while the config says anatomical_moe.
    if model_cfg.get('pool_type', 'mean') != 'mean':
        raise NotImplementedError(
            "pool_type=%r is not supported in run_patch_downstream "
            "(frozen-probe path) yet. Use run_patch_finetune (fine-tune) "
            "or run with pool_type='mean'. The frozen-probe MoE path "
            "requires raw-patch caching and aggregator-in-optimizer wiring "
            "that is tracked as a follow-up TODO."
            % model_cfg.get('pool_type'))

    # ---- Pre-compute features (one-time) -----------------------------------
    print('\n--- Pre-computing features with frozen encoder ---')
    slice_size = data_cfg.get('slice_size', 256)
    chunk_size = data_cfg.get('encode_chunk_size', 50)
    cache_dir = os.path.join(output_dir, 'feature_cache')

    train_feats, train_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Training',
        num_slices, slice_size, device, chunk_size, cache_dir)
    val_feats, val_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Validation',
        num_slices, slice_size, device, chunk_size, cache_dir)
    test_feats, test_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Test',
        num_slices, slice_size, device, chunk_size, cache_dir)

    # Free encoder from GPU after feature extraction
    encoder.cpu()
    torch.cuda.empty_cache()

    n_pos = int(train_labels.sum().item())
    n_neg = len(train_labels) - n_pos
    print('  Train: %d volumes (%d pos, %d neg, %.1f%% prevalence)'
          % (len(train_labels), n_pos, n_neg, 100.0 * n_pos / len(train_labels)))
    print('  Val:   %d volumes' % len(val_labels))
    print('  Test:  %d volumes' % len(test_labels))

    # ---- Data loaders on cached features -----------------------------------
    batch_size = data_cfg.get('batch_size', 16)

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels.float()),
        batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True)

    # ---- Slice-aggregation probe + classification head ---------------------
    # probe_type selects the slice-pooling architecture. See _build_probe.
    print('\n--- Model ---')
    probe_type = model_cfg.get('probe_type', 'attentive')
    probe, probe_desc = _build_probe(probe_type, num_slices, embed_dim, model_cfg, device)

    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1)).to(device)
    else:
        head = LinearHead(in_dim=embed_dim).to(device)

    probe_params = sum(p.numel() for p in probe.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    print('  Frozen encoder:  %s params' % format(enc_params, ','))
    print('  Probe (%s): %s params (trainable)' % (probe_desc, format(probe_params, ',')))
    print('  Head (%s):     %s params (trainable)' % (head_type, format(head_params, ',')))
    print('  Total trainable: %s' % format(probe_params + head_params, ','))

    # ---- Optimizer ------------------------------------------------------------
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
    scaler = GradScaler()

    # ---- CSV logger --------------------------------------------------------
    csv_path = os.path.join(output_dir, 'train_log.csv')
    csv_file = open(csv_path, 'w')
    csv_file.write('epoch,train_loss,train_auc,val_loss,val_auc,lr_probe,lr_head,elapsed_s\n')
    csv_file.flush()

    # ---- Training loop -----------------------------------------------------
    print('\n--- Training ---')
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']

    for epoch in range(1, epochs + 1):
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

            with autocast():
                pooled = probe(features)         # (B, D)
                logits = head(pooled).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch %d (patience=%d)' % (epoch, patience))
                break

    csv_file.close()

    # ---- Test evaluation (with best model) ---------------------------------
    print('\n--- Test Evaluation ---')
    best_path = os.path.join(output_dir, 'best_model.pt')
    test_loss = test_auc = best_epoch = None
    test_labels = test_probs = None
    val_labels_final = val_probs_final = None

    if not os.path.exists(best_path):
        print('ERROR: best_model.pt not found — no epoch improved over AUC=0')
        best_epoch = 0
    else:
        best_ckpt = torch.load(best_path, map_location=device)
        probe.load_state_dict(best_ckpt['probe'])
        head.load_state_dict(best_ckpt['head'])
        best_epoch = best_ckpt['epoch']

        # Val predictions (for ROC curve)
        val_loss_f, val_auc_f, val_labels_final, val_probs_final = evaluate(
            probe, head, val_loader, criterion, device, return_predictions=True)

        # Test predictions
        test_loss, test_auc, test_labels, test_probs = evaluate(
            probe, head, test_loader, criterion, device, return_predictions=True)

        print('Best epoch: %d  |  Val AUC: %.4f  |  TEST AUC: %.4f'
              % (best_epoch, best_auc, test_auc))

        # Sensitivity / specificity at threshold=0.5
        if test_labels is not None:
            test_preds = (test_probs >= 0.5).astype(int)
            tp = ((test_preds == 1) & (test_labels == 1)).sum()
            tn = ((test_preds == 0) & (test_labels == 0)).sum()
            fp = ((test_preds == 1) & (test_labels == 0)).sum()
            fn = ((test_preds == 0) & (test_labels == 1)).sum()
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            print('  Sensitivity: %.4f  |  Specificity: %.4f  (threshold=0.5)' % (sensitivity, specificity))

    # ---- Save predictions --------------------------------------------------
    if test_labels is not None:
        np.savez(os.path.join(output_dir, 'test_predictions.npz'),
                 labels=test_labels, probs=test_probs)
        print('  Saved test_predictions.npz (%d samples)' % len(test_labels))
    if val_labels_final is not None:
        np.savez(os.path.join(output_dir, 'val_predictions.npz'),
                 labels=val_labels_final, probs=val_probs_final)
        print('  Saved val_predictions.npz (%d samples)' % len(val_labels_final))

    # ---- Generate diagnostic plots -----------------------------------------
    _save_diagnostic_plots(output_dir, test_labels, test_probs, test_auc,
                           val_labels_final, val_probs_final)

    # ---- Save results ------------------------------------------------------
    results = {
        'mode': 'patch',
        'head_type': head_type,
        'num_slices': num_slices,
        'probe_depth': model_cfg.get('probe_depth', 2),
        'best_epoch': best_epoch,
        'best_val_auc': best_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'sensitivity': float(sensitivity) if test_labels is not None else None,
        'specificity': float(specificity) if test_labels is not None else None,
        'probe_params': probe_params,
        'head_params': head_params,
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to %s' % output_dir)
    print('  best_val_auc = %.4f' % best_auc)
    print('  test_auc     = %.4f' % (test_auc if test_auc else 0))
    print('  encoder: %s' % config.get('model', {}).get('encoder_checkpoint', 'unknown'))

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
    ckpt = torch.load(ckpt_path, map_location=device)
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
    scaler = GradScaler()

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

            with autocast():
                features = encode_fn(volumes)
                logits = head(features).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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

def _build_axial_pos_embed(num_slices, embed_dim, kind):
    """Construct an axial (slice-axis) position embedding for volume-scope MoE.

    Returns either a learnable nn.Parameter of shape (1, num_slices, 1, embed_dim)
    or a non-trainable buffer of the same shape (1D sincos), or None.
    """
    if kind == 'none':
        return None, False
    if kind == 'learned':
        p = nn.Parameter(torch.zeros(1, num_slices, 1, embed_dim))
        nn.init.trunc_normal_(p, std=0.02)
        return p, True
    if kind == 'sincos':
        # Standard 1D sincos: dim/2 sin + dim/2 cos
        pos = torch.arange(num_slices, dtype=torch.float32).unsqueeze(1)  # (S, 1)
        i = torch.arange(embed_dim // 2, dtype=torch.float32).unsqueeze(0)  # (1, D/2)
        omega = 1.0 / (10000 ** (2 * i / embed_dim))
        ang = pos * omega                             # (S, D/2)
        sincos = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (S, D)
        sincos = sincos.view(1, num_slices, 1, embed_dim)
        return sincos, False  # buffer, not trainable
    raise ValueError("Unknown axial_pos_embed kind=%r" % kind)


class DownstreamModel(nn.Module):
    """End-to-end model: ViT encoder + (aggregator) + probe + head.

    Three pooling paths controlled by (aggregator, moe_scope):

    1) mean-pool (aggregator=None):
         out.mean(dim=1) per slice → features (B, S, D) → probe.
         Original behavior, preserved bit-for-bit.

    2) per-slice MoE (aggregator set, moe_scope='per_slice'):
         For each chunk of slices:
           encoder(chunk) → (chunk, P, D)
           aggregator(out) → (chunk, E*S, D)         ← MoE called per slice
         Stack across slices → (B, S*E*S, D) → probe.

    3) volume MoE (aggregator set, moe_scope='volume'):
         encoder(chunk) → (chunk, P, D)
         Stack across slices → (B, S, P, D)
         features += axial_pos_embed[s]              ← optional axial pos
         flatten → (B, S*P, D)
         aggregator(features) → (B, E*S, D)          ← single MoE call per volume
         → probe.

       Volume scope is the cleanest MAMMOTH analog: one aggregator call per
       "slide-equivalent" (= one volume). axial_pos_embed_type='learned' is
       recommended for OCT so prototypes can specialize by axial slice
       position (peripapillary vs macular vs peripheral).
    """

    def __init__(self, encoder, probe, head, chunk_size=25, aggregator=None,
                 moe_scope='none', num_slices=64, embed_dim=768,
                 axial_pos_embed_type='none'):
        super(DownstreamModel, self).__init__()
        # Hard guard against silent misconfiguration: if an aggregator is
        # active, the probe sees a token sequence with no inherent axial
        # ordering (per_slice path) or whose axial info is encoded via
        # axial_pos_embed BEFORE the aggregator (volume path). In either
        # case the probe should not add its own pos_embed on top.
        if aggregator is not None and getattr(probe, 'use_pos_embed', False):
            raise ValueError(
                'DownstreamModel: aggregator is set but probe has '
                'use_pos_embed=True. Build the probe with '
                'use_pos_embed=False when using an aggregator.')
        if aggregator is not None and moe_scope not in _MOE_SCOPES:
            raise ValueError(
                'DownstreamModel: aggregator set but moe_scope=%r is not in %s.'
                % (moe_scope, _MOE_SCOPES))
        if aggregator is None and moe_scope not in ('none', 'per_slice'):
            # 'per_slice' is the historical default that was implicit pre-MoE.
            raise ValueError(
                'DownstreamModel: moe_scope=%r requires aggregator to be set.'
                % moe_scope)

        self.encoder = encoder
        self.probe = probe
        self.head = head
        self.chunk_size = chunk_size
        self.aggregator = aggregator
        self.moe_scope = moe_scope
        self.num_slices = num_slices
        self.embed_dim = embed_dim

        # Axial position embedding (volume scope only).
        if aggregator is not None and moe_scope == 'volume':
            ape, trainable = _build_axial_pos_embed(
                num_slices, embed_dim, axial_pos_embed_type)
        else:
            ape, trainable = None, False
        if ape is None:
            self.axial_pos_embed = None
        elif trainable:
            self.axial_pos_embed = ape  # nn.Parameter, picked up by .parameters()
        else:
            # Non-trainable sincos: register as buffer so it moves with .to(device)
            self.register_buffer('axial_pos_embed', ape)

    def _encode_per_chunk(self, flat):
        """Run encoder over flattened slice batch in chunks. Returns list of
        (chunk, P, D) tensors with grads attached for FT mode."""
        parts = []
        for i in range(0, flat.size(0), self.chunk_size):
            chunk = flat[i:i + self.chunk_size]
            out = self.encoder(chunk)              # (chunk, P, D)
            parts.append(out)
        return parts

    def forward(self, volumes):
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        flat = imagenet_normalize(flat)  # match pretraining distribution

        # ---- Mean-pool path (aggregator=None) -----------------------------
        if self.aggregator is None:
            parts = []
            for i in range(0, flat.size(0), self.chunk_size):
                chunk = flat[i:i + self.chunk_size]
                out = self.encoder(chunk)          # (chunk, P, D)
                parts.append(out.mean(dim=1))      # (chunk, D)
            features = torch.cat(parts, dim=0)     # (B*S, D)
            features = features.reshape(B, S, -1)  # (B, S, D)

        # ---- Per-slice MoE path -------------------------------------------
        elif self.moe_scope == 'per_slice':
            parts = []
            for i in range(0, flat.size(0), self.chunk_size):
                chunk = flat[i:i + self.chunk_size]
                out = self.encoder(chunk)          # (chunk, P, D)
                parts.append(self.aggregator(out)) # (chunk, E*S, D)
            features = torch.cat(parts, dim=0)     # (B*S, E*S, D)
            features = features.reshape(B, S * features.size(1), -1)

        # ---- Volume MoE path ----------------------------------------------
        else:  # self.moe_scope == 'volume'
            patch_parts = self._encode_per_chunk(flat)         # list of (chunk, P, D)
            features = torch.cat(patch_parts, dim=0)           # (B*S, P, D)
            P = features.size(1)
            features = features.reshape(B, S, P, -1)           # (B, S, P, D)
            if self.axial_pos_embed is not None:
                features = features + self.axial_pos_embed     # broadcasts over B and P
            features = features.reshape(B, S * P, -1)          # (B, S*P, D)
            features = self.aggregator(features)               # (B, E*S, D)

        pooled = self.probe(features)              # (B, D)
        return self.head(pooled).squeeze(-1)       # (B,)


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
        with autocast():
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
    ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['target_encoder'])
    if is_main:
        print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']

    probe_type = model_cfg.get('probe_type', 'attentive')
    # ---- Within-slice aggregator (opt-in via model_cfg.pool_type) ----------
    # Default pool_type='mean' returns aggregator=None and reproduces the
    # original mean-pool flow exactly.
    aggregator, probe_num_tokens, moe_scope, axial_type, agg_desc = (
        _build_aggregator(model_cfg, embed_dim, num_slices, device='cpu')
    )
    if is_main:
        print('  Aggregator: %s' % agg_desc)

    # _build_probe fails fast on unknown probe_type; safe here.
    probe, probe_desc = _build_probe(
        probe_type, probe_num_tokens, embed_dim, model_cfg, device='cpu',
        use_pos_embed=(aggregator is None),
    )

    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1))
    else:
        head = LinearHead(in_dim=embed_dim)

    chunk_size = data_cfg.get('encode_chunk_size', 25)
    model = DownstreamModel(
        encoder, probe, head, chunk_size,
        aggregator=aggregator,
        moe_scope=moe_scope,
        num_slices=num_slices,
        embed_dim=embed_dim,
        axial_pos_embed_type=axial_type,
    ).to(device)

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
    # Pass the learnable axial pos embed too if it exists (volume MoE +
    # axial_pos_embed_type='learned' creates an nn.Parameter on the model).
    learnable_axial = (
        raw.axial_pos_embed
        if isinstance(getattr(raw, 'axial_pos_embed', None), nn.Parameter)
        else None
    )
    param_groups, pg_mode = build_finetune_param_groups(
        raw.encoder, raw.probe, raw.head, train_cfg,
        aggregator=raw.aggregator,
        axial_pos_embed=learnable_axial,
    )
    if is_main:
        if pg_mode == 'llrd':
            # Look up the highest-index 'block_X' group by name. Index-based
            # access ([-4]) breaks once aggregator / axial_pos_embed groups
            # are inserted between encoder_norm and probe.
            block_groups = [g for g in param_groups
                            if g.get('name', '').startswith('block_')]
            lr_top_block = block_groups[-1]['lr'] if block_groups else float('nan')
            head_group = next((g for g in param_groups
                               if g.get('name') == 'head'), param_groups[-1])
            print('  Optimizer: AdamW + LLRD (decay=%.2f, %d groups)'
                  % (train_cfg.get('layer_decay', 1.0), len(param_groups)))
            print('    LR range: embed=%.2e .. top_block=%.2e .. head=%.2e'
                  % (param_groups[0]['lr'], lr_top_block, head_group['lr']))
        else:
            enc_g = next((g for g in param_groups
                          if g.get('name') == 'encoder'), param_groups[0])
            probe_g = next((g for g in param_groups
                            if g.get('name') == 'probe'), None)
            head_g = next((g for g in param_groups
                           if g.get('name') == 'head'), param_groups[-1])
            print('  Optimizer: AdamW + flat (encoder=%.2e, probe=%s, head=%.2e)'
                  % (enc_g['lr'],
                     ('%.2e' % probe_g['lr']) if probe_g else 'n/a',
                     head_g['lr']))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    steps_per_epoch = len(train_loader) // accum_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], max(steps_per_epoch, 1),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

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

            with autocast():
                logits = model(volumes)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * accum_steps * labels.size(0)
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
                ckpt = {
                    'epoch': epoch,
                    'encoder': raw.encoder.state_dict(),
                    'probe': raw.probe.state_dict(),
                    'head': raw.head.state_dict(),
                    'val_auc': val_auc,
                }
                if raw.aggregator is not None:
                    ckpt['aggregator'] = raw.aggregator.state_dict()
                # Persist learnable axial pos embed (volume MoE only).
                if isinstance(getattr(raw, 'axial_pos_embed', None), nn.Parameter):
                    ckpt['axial_pos_embed'] = raw.axial_pos_embed.detach().cpu()
                torch.save(ckpt, os.path.join(output_dir, 'best_model.pt'))
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
            if raw.aggregator is not None:
                if 'aggregator' not in best_ckpt:
                    raise RuntimeError(
                        "DownstreamModel has an aggregator but checkpoint at "
                        "%s has no 'aggregator' key. The checkpoint was likely "
                        "saved before aggregator support was added; cannot "
                        "safely resume MoE training from it." % best_path)
                raw.aggregator.load_state_dict(best_ckpt['aggregator'])
            # Restore learnable axial pos embed if model has it.
            if isinstance(getattr(raw, 'axial_pos_embed', None), nn.Parameter):
                if 'axial_pos_embed' not in best_ckpt:
                    raise RuntimeError(
                        "DownstreamModel has a learnable axial_pos_embed but "
                        "checkpoint at %s has no 'axial_pos_embed' key. Cannot "
                        "safely resume." % best_path)
                with torch.no_grad():
                    raw.axial_pos_embed.copy_(best_ckpt['axial_pos_embed'].to(
                        raw.axial_pos_embed.device))
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
        np.savez(os.path.join(output_dir, 'test_predictions.npz'),
                 labels=test_labels, probs=test_probs)
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

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    freeze_encoder = config.get('model', {}).get('freeze_encoder', True)

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
    parser = argparse.ArgumentParser(description='Downstream glaucoma classification')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args)
