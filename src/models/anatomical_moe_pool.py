"""Anatomical Mixture-of-Experts pool for OCT patch aggregation.

Replaces ``patches.mean(dim=1)`` with soft slot-prototype pooling that
produces ``E*S`` anatomical prototype tokens. The module itself is
shape-agnostic over the token axis ``N`` — it works equally well at
two scales, controlled at the call site (in ``DownstreamModel``):

  per_slice scope: aggregator called PER SLICE on (B*S, P=256, D)
                   → output (B*S, E*S, D); stacked to (B, S*E*S, D) for probe.
                   Closer to the initial proposal; preserves slice identity.

  volume scope:    aggregator called ONCE on (B, S*P=16384, D) per volume
                   → output (B, E*S, D); probe sees just E*S tokens.
                   RECOMMENDED for OCT — true MAMMOTH analog (one MoE call
                   per "slide-equivalent"), and an order of magnitude
                   smaller probe sequence. Pair with axial_pos_embed
                   (added by DownstreamModel before the MoE call) so
                   prototypes can specialize by axial slice position.

Hypothesis: OCT volumes have known layered anatomical structure (vitreous,
RNFL, IPL, INL, OPL, ONL, RPE, choroid, sclera × axial slice position).
A soft MoE that learns to route patch tokens to slot prototypes should
specialize those slots toward distinct retinal-layer × axial-position
concepts, giving better downstream signal AND interpretable anatomical
decomposition.

Architecturally based on MAMMOTH (Shao et al., ICLR 2026); see also
``docs/design/anatomical_moe_integration.md`` for design rationale and
audit history.

Key design choices:
  - Soft routing (Puigcerver 2024) — every patch contributes to every slot
    via softmax weights, scaled by 1/sqrt(head_dim) (attention-style).
    No top-K argmax. Stable training at small data.
  - Multi-head partitioning of the routing space — H independent subspaces
    let different feature dimensions drive routing independently.
  - Low-rank factorized experts with shared Phi matrix — keeps the param
    count manageable while supporting many experts.
  - skip_wq=True: route directly in encoder feature space (slot prototypes
    interpretable via routing weights as encoder-space vectors).
  - Slot prototypes are learnable; can be warm-started from k-means cluster
    centers of pre-extracted features (PaMoE-style — see
    ``init_slots_from_kmeans``).

Pipeline integration: replaces the ``out.mean(dim=1)`` call site in
``DownstreamModel.forward``. The ``moe_scope`` ctor argument selects
per_slice or volume.
"""
from typing import Optional

import torch
import torch.nn as nn


class AnatomicalMoEPool(nn.Module):
    """Within-slice multi-head soft MoE pooling.

    Args:
        embed_dim:   ViT feature dim (768 for ViT-B, 384 for ViT-S).
        num_experts: E — number of expert heads in the MoE.
        num_slots:   S — slots per expert. Each slot is a learnable prototype.
        num_heads:   H — independent routing subspaces.
        slot_dim:    Hidden dim for routing (after initial projection).
                     Must be divisible by num_heads.
        lora_rank:   Bottleneck rank Q for the factorized expert layers.
        share_phi:   If True, the low-rank factor Phi is shared across
                     experts within a head (recommended; smaller param
                     budget). If False, per-expert Phi (more capacity).
        out_dim:     Output dim per prototype. Defaults to embed_dim for
                     drop-in compatibility with existing probes.
        dropout:     Dropout on slot prototypes during training.

    Shapes:
        Input:  (B, N, embed_dim)              N = 256 patches per slice
        Output: (B, E*S, out_dim)              E*S = 32 prototypes by default

    Default param count (E=8, S=4, H=8, slot_dim=256, lora=16, embed=768):
        wq:                 768 * 256 + 256        = 196,864
        slots:              8 * 8 * 4 * 32         =  8,192
        Phi (shared):       8 * 16 * 32            =  4,096
        expert_W:           8 * 8 * 16 * 96        = 98,304
        biases + norms:                            ~ 8,500
        ----------------------------------------    --------
        Total:                                     ~ 316K params
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_experts: int = 8,
        num_slots: int = 4,
        num_heads: int = 8,
        slot_dim: int = 256,
        lora_rank: int = 16,
        share_phi: bool = True,
        skip_wq: bool = False,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super(AnatomicalMoEPool, self).__init__()
        # When skip_wq is set, routing happens directly in encoder feature
        # space; slot_dim is forced to embed_dim. This trades the (cheap)
        # projection compute for keeping prototypes interpretable in the
        # same space as raw patch features.
        if skip_wq:
            slot_dim = embed_dim
        if slot_dim % num_heads != 0:
            raise ValueError(
                'slot_dim (%d) must be divisible by num_heads (%d)'
                % (slot_dim, num_heads))
        out_dim = out_dim if out_dim is not None else embed_dim
        if out_dim % num_heads != 0:
            raise ValueError(
                'out_dim (%d) must be divisible by num_heads (%d)'
                % (out_dim, num_heads))

        self.embed_dim   = embed_dim
        self.num_experts = num_experts
        self.num_slots   = num_slots
        self.num_heads   = num_heads
        self.slot_dim    = slot_dim
        self.head_dim    = slot_dim // num_heads
        self.out_dim     = out_dim
        self.out_per_head = out_dim // num_heads
        self.share_phi   = share_phi
        self.skip_wq     = skip_wq
        # 1/sqrt(head_dim) attention-style scaling — keeps routing logits
        # bounded so softmax(over N tokens) doesn't collapse to ~hard top-k.
        # MAMMOTH gets away without this because its head_dim_input=16 keeps
        # logit std small (~4). With skip_wq=True we have head_dim=96 and
        # the unscaled logit std is ~10, which collapses softmax to near-hard
        # selection (verified: ~2.8 effective tokens routed out of 16K).
        self.routing_scale = self.head_dim ** -0.5

        # Initial projection: encoder feature dim -> slot_dim.
        # Identity when skip_wq=True (slot_dim==embed_dim then).
        self.wq = nn.Identity() if skip_wq else nn.Linear(embed_dim, slot_dim, bias=True)
        self.norm_q = nn.LayerNorm(slot_dim)

        # Slot prototypes: (E, H, S, head_dim).
        # Trainable, randomly initialized. Can be warm-started externally.
        self.slot_embeds = nn.Parameter(
            torch.empty(num_experts, num_heads, num_slots, self.head_dim))
        nn.init.trunc_normal_(self.slot_embeds, std=0.02)
        self.norm_slots = nn.LayerNorm(self.head_dim)
        self.slot_dropout = nn.Dropout(dropout)

        # Low-rank expert factor Phi: maps head_dim -> lora_rank
        # share_phi=True:  (H, head_dim, lora_rank)            shared across experts
        # share_phi=False: (E, H, head_dim, lora_rank)         per-expert
        if share_phi:
            self.phi = nn.Parameter(
                torch.empty(num_heads, self.head_dim, lora_rank))
        else:
            self.phi = nn.Parameter(
                torch.empty(num_experts, num_heads, self.head_dim, lora_rank))
        nn.init.trunc_normal_(self.phi, std=0.02)
        if share_phi:
            self.phi_bias = nn.Parameter(torch.zeros(num_heads, lora_rank))
        else:
            self.phi_bias = nn.Parameter(
                torch.zeros(num_experts, num_heads, lora_rank))

        # Per-expert output projection: (E, H, lora_rank, out_per_head)
        self.expert_w = nn.Parameter(
            torch.empty(num_experts, num_heads, lora_rank, self.out_per_head))
        nn.init.trunc_normal_(self.expert_w, std=0.02)
        self.expert_b = nn.Parameter(
            torch.zeros(num_experts, num_heads, self.out_per_head))

        self.act = nn.ReLU()
        self.norm_out = nn.LayerNorm(out_dim)

    @torch.no_grad()
    def init_slots_from_kmeans(self, cluster_centers: torch.Tensor) -> None:
        """Warm-start slot prototypes from external k-means cluster centers.

        Args:
            cluster_centers: (E*S, slot_dim) tensor — K-means centers
                **already in slot_dim space** (i.e. computed over features
                that have been projected through ``self.wq`` and normalized
                through ``self.norm_q``).

                Caller is responsible for the projection because at module
                construction time ``self.wq`` is randomly initialized; running
                the projection internally would silently destroy the
                semantically-meaningful cluster structure. Recommended usage:

                    with torch.no_grad():
                        feats = encoder(sample_input)        # (B, N, embed_dim)
                        proj  = pool.norm_q(pool.wq(feats))  # (B, N, slot_dim)
                    centers = run_kmeans(proj, K=E*S)         # (E*S, slot_dim)
                    pool.init_slots_from_kmeans(centers)

                Note that this means warm-starting is most useful AFTER ``wq``
                has trained for some steps, OR if ``wq`` is initialized to
                identity / a known-good projection.
        """
        target_n = self.num_experts * self.num_slots
        if cluster_centers.size(0) != target_n:
            raise ValueError(
                'Expected %d cluster centers, got %d' % (target_n, cluster_centers.size(0)))
        if cluster_centers.size(1) != self.slot_dim:
            raise ValueError(
                'Expected cluster centers in slot_dim=%d (post-wq), got dim=%d. '
                'See docstring: caller must project through self.wq first.'
                % (self.slot_dim, cluster_centers.size(1)))
        # Reshape (E*S, slot_dim) -> (E, S, H, head_dim) -> (E, H, S, head_dim)
        proj = cluster_centers.view(
            self.num_experts, self.num_slots, self.num_heads, self.head_dim)
        proj = proj.permute(0, 2, 1, 3).contiguous()
        self.slot_embeds.copy_(proj)

    def routing_logits(self, x_heads: torch.Tensor) -> torch.Tensor:
        """Compute routing logits per (token, expert, head, slot).

        Args:
            x_heads: (B, N, H, head_dim)

        Returns:
            logits: (B, N, E, H, S), scaled by 1/sqrt(head_dim).
        """
        slots = self.slot_dropout(self.norm_slots(self.slot_embeds))
        # Per-head similarity, scaled to keep softmax-over-tokens reasonably soft.
        return torch.einsum('bnhd,ehsd->bnehs', x_heads, slots) * self.routing_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-route N input tokens to (E*S) prototype output tokens.

        Args:
            x: (B, N, embed_dim) — typically (B*num_slices, 256, 768)
               from a ViT encoder before mean-pooling.

        Returns:
            (B, E*S, out_dim) — anatomical prototype tokens per slice.
        """
        B, N, _ = x.shape
        E, S, H = self.num_experts, self.num_slots, self.num_heads

        # 1. Project to routing space and split into heads
        q = self.norm_q(self.wq(x))                              # (B, N, slot_dim)
        x_heads = q.view(B, N, H, self.head_dim)                 # (B, N, H, d)

        # 2. Routing logits and dispatch weights (softmax over patches)
        logits = self.routing_logits(x_heads)                    # (B, N, E, H, S)
        dispatch = logits.softmax(dim=1)                          # softmax over N

        # 3. Pool patches into per-(expert, head, slot) prototypes
        slots = torch.einsum('bnhd,bnehs->behsd', x_heads, dispatch)
        # slots: (B, E, H, S, head_dim)

        # 4. Low-rank expert: slots -> Phi*slots -> expert_W * (Phi*slots)
        if self.share_phi:
            # phi: (H, d, r) — shared across experts within a head
            r = torch.einsum('behsd,hdr->behsr', slots, self.phi)
            r = r + self.phi_bias.view(1, 1, H, 1, -1)            # (B, E, H, S, r)
        else:
            # phi: (E, H, d, r) — per-expert
            r = torch.einsum('behsd,ehdr->behsr', slots, self.phi)
            r = r + self.phi_bias.view(1, E, H, 1, -1)
        r = self.act(r)

        # 5. Per-expert output projection: (E, H, r, out_per_head)
        z = torch.einsum('behsr,ehrp->behsp', r, self.expert_w)
        z = z + self.expert_b.view(1, E, H, 1, -1)               # (B, E, H, S, out_per_head)

        # 6. Concat heads into out_dim per (E, S) prototype
        # (B, E, H, S, p) -> (B, E, S, H, p) -> (B, E*S, H*p) = (B, E*S, out_dim)
        z = z.permute(0, 1, 3, 2, 4).contiguous()
        z = z.view(B, E * S, self.out_dim)

        return self.norm_out(z)
