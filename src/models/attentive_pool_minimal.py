"""Minimal cross-attention pool for 3D OCT slice aggregation.

A lightweight alternative to :class:`eval_downstream.AttentiveProbe` for
the frozen-probe setting. Motivated by:

1. I-JEPA paper uses a single transformer block for the attentive probe,
   with self-attention across all tokens plus FFN (~7M params for ViT-B).
2. V-JEPA uses cross-attention with a learnable query plus FFN, similarly
   sized (~6.5M for ViT-B). No positional embeddings in its probe because
   the encoder bakes positional info into patch tokens.
3. "Attention, Please! Revisiting Attentive Probing" (ICLR 2026) documents
   that standard attentive probes are "over-parameterized" for evaluation
   and proposes ~200K-param alternatives.

Our setting requires explicit **slice-axis positional embeddings** because
patches are mean-pooled within each slice before the probe, so slice order
is NOT carried in the feature values (unlike V-JEPA's patch tokens whose
spatial positions are encoded by the ViT's own pos_embed).

Architecture (ViT-B defaults: embed_dim=768, num_slices=100, head_dim=64):

    Input:  (B, 100, 768)              -- per-slice mean-pooled features
       |
       +-- pos_embed: (1, 100, 768)    -- learnable axial position
       |
       v
    Projections (single head, head_dim=64, no FFN):
       Q (from learnable query): Linear(768, 64)
       K (from slices):           Linear(768, 64)
       V (from slices):           Linear(768, 64)
       attention(Q, K, V):        single-head softmax, no mask
       Output:                    Linear(64, 768)
       LayerNorm(768)
       |
       v
    Output: (B, 768) pooled volume representation

Parameter count (~277K):
    query token         : 768
    pos_embed           : 76,800   (0.28 of total)
    q_proj (768 -> 64)  : 49,216
    k_proj (768 -> 64)  : 49,216
    v_proj (768 -> 64)  : 49,216
    o_proj (64 -> 768)  : 49,920
    final LayerNorm     : 1,536
    -----------------------------
    Total               : ~277K   (~26x smaller than AttentiveProbe d=1)

This sits between "mean-pool + linear" (800 params, no slice weighting)
and AttentiveProbe d=1 (~7M, full self-attention + FFN). It preserves
attention-based slice weighting and axial position information while
stripping slice-to-slice interaction and the per-slice FFN transform.
"""

import torch
import torch.nn as nn


class CrossAttnPool(nn.Module):
    """Minimal cross-attention pool: single-head, no FFN, optional axial pos_embed.

    Args:
        num_slices: number of slice tokens per volume (N in (B, N, D))
        embed_dim:  feature dim of each token (D)
        head_dim:   dimensionality of the compressed attention space
        use_pos_embed: if True (default, original behavior), an axial pos_embed
            of shape (1, num_slices, D) is added before attention. If False
            (e.g. AnatomicalMoEPool path where prototype tokens have no
            intrinsic axial ordering), the layer is permutation-equivariant.
    """

    def __init__(self, num_slices=100, embed_dim=768, head_dim=64,
                 use_pos_embed=True):
        super(CrossAttnPool, self).__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_pos_embed = use_pos_embed
        # NB: parameter creation and initialization order is preserved from
        # the original (pre-use_pos_embed-flag) layout so that same-seed runs
        # produce identical Linear weights. See audit P1 ("RNG init order").
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, embed_dim))
        else:
            self.pos_embed = None
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.o_proj = nn.Linear(head_dim, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.query, std=0.02)
        if use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """x: (B, N, embed_dim) -> (B, embed_dim).

        N is num_slices in the standard mean-pool path, or num_slices*E*S
        in the AnatomicalMoEPool path. When use_pos_embed=False, the layer
        is permutation-equivariant over N.
        """
        B = x.size(0)
        if self.use_pos_embed:
            x = x + self.pos_embed
        q = self.q_proj(self.query.expand(B, -1, -1))   # (B, 1, head_dim)
        k = self.k_proj(x)                               # (B, S, head_dim)
        v = self.v_proj(x)                               # (B, S, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, 1, S)
        attn = attn.softmax(dim=-1)
        pooled = attn @ v                                # (B, 1, head_dim)
        pooled = self.o_proj(pooled).squeeze(1)          # (B, embed_dim)
        return self.norm(pooled)


class MeanPool(nn.Module):
    """Unweighted mean across the slice dimension. Zero parameters.

    Ablation floor for the attentive-probe family. Discards slice
    ordering entirely — two volumes with the same set of slice features
    in different axial orders pool to the same vector. Meant only to
    quantify how much the position-aware probes (AttentiveProbe,
    CrossAttnPool) are actually contributing. Not recommended for the
    OCT-volume task itself: disease signals localize to specific axial
    positions (glaucoma → optic cup region, AMD → macular middle,
    etc.), so an order-blind pool throws away information.

    Constructor args are kept for API parity with the other probes but
    are not used internally.
    """

    def __init__(self, num_slices=100, embed_dim=768):
        super(MeanPool, self).__init__()
        self.num_slices = num_slices
        self.embed_dim = embed_dim

    def forward(self, x):
        """x: (B, num_slices, embed_dim) -> (B, embed_dim)."""
        return x.mean(dim=1)
