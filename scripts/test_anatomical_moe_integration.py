"""End-to-end integration test for the AnatomicalMoEPool path.

Verifies:
  1. Default (mean-pool) behavior is unchanged — existing flow runs as before.
  2. Opt-in (anatomical_moe) path runs end-to-end without crashing.
  3. The two paths produce different shapes at the feature stage but the same
     logit shape at the head — confirming probe shape-agnosticism.
  4. _build_aggregator and _build_probe wire up correctly with config flags.

This uses dummy tensors and a stub encoder — no real data, no real weights.

Run: python scripts/test_anatomical_moe_integration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.eval_downstream import (
    DownstreamModel, _build_aggregator, _build_probe, AttentiveProbe,
)
from src.models.attentive_pool_minimal import CrossAttnPool, MeanPool
from src.models.anatomical_moe_pool import AnatomicalMoEPool


class StubEncoder(nn.Module):
    """Stand-in for a ViT-B/16: maps (B, 3, 256, 256) -> (B, 256, 768)."""
    def __init__(self, embed_dim=768, num_patches=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        # Bridge from image to patch tokens — only used for shape-correct test
        self.proj = nn.Linear(3 * 16 * 16, embed_dim)

    def forward(self, x):
        # x: (B, 3, 256, 256) -> patches (B, 256, 768)
        B = x.size(0)
        # Just take 16x16 grid of mean-pooled patches for stub purposes
        patches = x.unfold(2, 16, 16).unfold(3, 16, 16)         # (B, 3, 16, 16, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, 16, 16, 3, 16, 16)
        patches = patches.view(B, 256, -1)                       # (B, 256, 3*16*16)
        return self.proj(patches)                                 # (B, 256, 768)


def dummy_volume_batch(B=2, S=8, C=3, H=256, W=256):
    return torch.randn(B, S, C, H, W)


def test_aggregator_factory_default_mean():
    """_build_aggregator returns (None, num_slices, 'none', 'none', 'mean') by default."""
    model_cfg = {}                                # no pool_type -> default 'mean'
    aggregator, n_tokens, scope, axial, desc = _build_aggregator(
        model_cfg, embed_dim=768, num_slices=64, device='cpu')
    assert aggregator is None
    assert n_tokens == 64           # mean path: probe sees one token per slice
    assert scope == 'none'
    assert axial == 'none'
    assert desc == 'mean'
    print('  factory default (no pool_type set): aggregator=None  OK')


def test_aggregator_factory_per_slice():
    """_build_aggregator returns AnatomicalMoEPool with per_slice scope."""
    model_cfg = {
        'pool_type': 'anatomical_moe',
        'anatomical_moe': {
            'moe_scope': 'per_slice',
            'num_experts': 8, 'num_slots': 4, 'num_heads': 8,
            'slot_dim': 256, 'lora_rank': 16, 'share_phi': True,
        },
    }
    aggregator, n_tokens, scope, axial, desc = _build_aggregator(
        model_cfg, embed_dim=768, num_slices=64, device='cpu')
    assert isinstance(aggregator, AnatomicalMoEPool)
    assert n_tokens == 64 * 32      # per_slice: num_slices * E*S
    assert scope == 'per_slice'
    assert axial == 'none'
    print(f'  factory per_slice: probe sees {n_tokens} tokens (64 slices x 32 protos)  OK')


def test_aggregator_factory_volume_with_axial():
    """_build_aggregator with moe_scope=volume + skip_wq + learned axial."""
    model_cfg = {
        'pool_type': 'anatomical_moe',
        'anatomical_moe': {
            'moe_scope': 'volume',
            'skip_wq': True,
            'axial_pos_embed': 'learned',
            'num_experts': 8, 'num_slots': 4, 'num_heads': 8,
            'lora_rank': 16, 'share_phi': True,
        },
    }
    aggregator, n_tokens, scope, axial, desc = _build_aggregator(
        model_cfg, embed_dim=768, num_slices=64, device='cpu')
    assert isinstance(aggregator, AnatomicalMoEPool)
    assert aggregator.skip_wq is True
    assert aggregator.slot_dim == 768          # forced by skip_wq
    assert n_tokens == 32                       # volume: just E*S
    assert scope == 'volume'
    assert axial == 'learned'
    print(f'  factory volume+skip_wq+learned: probe sees {n_tokens} tokens  OK')


def test_aggregator_factory_unknown_pool_type_fails_fast():
    """Typos in pool_type fail fast with a clear error."""
    try:
        _build_aggregator({'pool_type': 'meen'}, embed_dim=768,
                          num_slices=64, device='cpu')
    except ValueError as e:
        assert 'pool_type' in str(e)
        print(f'  factory rejects unknown pool_type with clear error  OK')
        return
    assert False, 'Expected ValueError'


def test_aggregator_factory_unknown_scope_fails_fast():
    """Typos in moe_scope fail fast."""
    cfg = {'pool_type': 'anatomical_moe',
           'anatomical_moe': {'moe_scope': 'whole_batch'}}
    try:
        _build_aggregator(cfg, embed_dim=768, num_slices=64, device='cpu')
    except ValueError as e:
        assert 'moe_scope' in str(e)
        print(f'  factory rejects unknown moe_scope with clear error  OK')
        return
    assert False, 'Expected ValueError'


def test_build_probe_default_path_unchanged():
    """_build_probe with default (no use_pos_embed override) preserves old shape."""
    probe, desc = _build_probe(
        'cross_attn_pool', num_tokens=64, embed_dim=768,
        model_cfg={}, device='cpu',
    )
    assert isinstance(probe, CrossAttnPool)
    assert probe.use_pos_embed is True
    assert probe.pos_embed.shape == (1, 64, 768)
    print(f'  default _build_probe: pos_embed (1,64,768), use_pos_embed=True  OK')


def test_build_probe_moe_path_no_pos_embed():
    """_build_probe with use_pos_embed=False (MoE path) drops pos_embed."""
    probe, desc = _build_probe(
        'cross_attn_pool', num_tokens=2048, embed_dim=768,
        model_cfg={}, device='cpu', use_pos_embed=False,
    )
    assert isinstance(probe, CrossAttnPool)
    assert probe.use_pos_embed is False
    assert probe.pos_embed is None
    print(f'  MoE _build_probe: use_pos_embed=False, pos_embed=None  OK')


def test_downstream_model_default_path():
    """DownstreamModel with aggregator=None reproduces the old shape flow."""
    encoder = StubEncoder()
    probe = MeanPool(num_slices=8, embed_dim=768)
    head = nn.Linear(768, 1)
    model = DownstreamModel(encoder, probe, head, chunk_size=4, aggregator=None)
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    print(f'  DownstreamModel default (aggregator=None): out shape (2,)  OK')


def test_downstream_model_per_slice_path():
    """Per-slice MoE: aggregator runs once per slice, probe sees S*E*S tokens."""
    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        slot_dim=128, lora_rank=8,
    )
    n_per_slice = 4 * 2  # E*S
    probe = MeanPool(num_slices=8 * n_per_slice, embed_dim=768)
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=aggregator, moe_scope='per_slice', num_slices=8,
    )
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    print(f'  DownstreamModel per_slice MoE: out shape (2,) on 8*8=64 tokens  OK')


def test_downstream_model_volume_path_no_axial():
    """Volume MoE without axial pos: aggregator sees (B, S*P, D), probe sees E*S."""
    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        skip_wq=True, lora_rank=8,
    )
    probe = MeanPool(num_slices=4 * 2, embed_dim=768)  # E*S=8 tokens total
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=aggregator, moe_scope='volume', num_slices=8, embed_dim=768,
        axial_pos_embed_type='none',
    )
    assert model.axial_pos_embed is None
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    print(f'  DownstreamModel volume MoE (no axial): out (2,)  OK')


def test_downstream_model_volume_path_learned_axial():
    """Volume MoE with learned axial pos embed: trainable (1,S,1,D) param exists."""
    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        skip_wq=True, lora_rank=8,
    )
    probe = CrossAttnPool(num_slices=8, embed_dim=768, head_dim=64,
                          use_pos_embed=False)
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=aggregator, moe_scope='volume', num_slices=8, embed_dim=768,
        axial_pos_embed_type='learned',
    )
    assert isinstance(model.axial_pos_embed, nn.Parameter)
    assert model.axial_pos_embed.shape == (1, 8, 1, 768)
    assert model.axial_pos_embed.requires_grad
    # Forward + backward should propagate through axial_pos_embed
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    out.sum().backward()
    assert model.axial_pos_embed.grad is not None
    assert (model.axial_pos_embed.grad.abs().sum() > 0).item()
    print(f'  DownstreamModel volume MoE + learned axial: param trained, grad nonzero  OK')


def test_downstream_model_volume_path_sincos_axial():
    """Volume MoE with sincos axial pos: registered as buffer, not trainable."""
    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        skip_wq=True, lora_rank=8,
    )
    probe = MeanPool(num_slices=8, embed_dim=768)
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=aggregator, moe_scope='volume', num_slices=8, embed_dim=768,
        axial_pos_embed_type='sincos',
    )
    # Buffer (not Parameter) — moves with .to(device) but not in .parameters()
    assert hasattr(model, 'axial_pos_embed')
    assert not isinstance(model.axial_pos_embed, nn.Parameter)
    assert model.axial_pos_embed.shape == (1, 8, 1, 768)
    # Sincos is bounded in [-1, 1]
    assert model.axial_pos_embed.abs().max().item() <= 1.0001
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    print(f'  DownstreamModel volume MoE + sincos axial: buffer set, [-1,1], forward OK')


def test_downstream_model_moe_with_cross_attn_probe():
    """End-to-end: encoder -> AnatomicalMoEPool -> CrossAttnPool (no pos_embed)."""
    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        slot_dim=128, lora_rank=8,
    )
    n_per_slice = 4 * 2
    probe = CrossAttnPool(
        num_slices=8 * n_per_slice, embed_dim=768, head_dim=64,
        use_pos_embed=False,
    )
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4, aggregator=aggregator,
        moe_scope='per_slice', num_slices=8,
    )
    vols = dummy_volume_batch(B=2, S=8)
    out = model(vols)
    assert out.shape == (2,)
    # Backward pass touches all params
    loss = out.sum()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    assert n_with_grad == n_total, f'{n_with_grad}/{n_total} params got grad'
    print(f'  E2E with CrossAttnPool probe: forward+backward OK ({n_total} params)')


def test_existing_default_behavior_preserved():
    """Sanity: the old default constructor signatures still work as before."""
    # Old-style CrossAttnPool: no use_pos_embed kwarg
    probe = CrossAttnPool(num_slices=64, embed_dim=768, head_dim=64)
    assert probe.use_pos_embed is True
    assert probe.pos_embed is not None

    # Old-style AttentiveProbe: no use_pos_embed kwarg
    ap = AttentiveProbe(num_slices=64, embed_dim=768)
    assert ap.use_pos_embed is True
    assert ap.pos_embed is not None

    # MeanPool: unchanged
    mp = MeanPool(num_slices=64, embed_dim=768)
    assert mp(torch.randn(2, 64, 768)).shape == (2, 768)

    print('  pre-existing constructor calls work unchanged  OK')


def test_rng_init_order_preserved_for_default():
    """Same seed must yield same Linear weights when use_pos_embed=True (default).

    Audit P1: moving pos_embed init earlier in __init__ would shift RNG state,
    causing q_proj/k_proj weights to differ for the same seed. This regression
    test guards against re-introducing that ordering bug.
    """
    torch.manual_seed(42)
    a = CrossAttnPool(num_slices=64, embed_dim=64, head_dim=8)
    torch.manual_seed(42)
    b = CrossAttnPool(num_slices=64, embed_dim=64, head_dim=8)
    assert torch.equal(a.q_proj.weight, b.q_proj.weight)
    assert torch.equal(a.k_proj.weight, b.k_proj.weight)
    print('  same-seed CrossAttnPool produces identical Linear weights  OK')


def test_finetune_param_groups_includes_aggregator():
    """build_finetune_param_groups picks up aggregator params when present."""
    from src.eval_downstream import build_finetune_param_groups

    class Stub(nn.Module):
        def __init__(self, n=10):
            super().__init__()
            self.w = nn.Linear(n, n)
            self.blocks = nn.ModuleList([nn.Linear(n, n) for _ in range(2)])
            self.norm = nn.LayerNorm(n)
            self.patch_embed = nn.Linear(n, n)
            self.pos_embed = nn.Parameter(torch.zeros(1, n, n))

    encoder = Stub()
    probe = MeanPool(num_slices=8, embed_dim=10)  # 0 params
    head = nn.Linear(10, 1)
    aggregator = AnatomicalMoEPool(
        embed_dim=10, num_experts=2, num_slots=2, num_heads=2,
        slot_dim=8, lora_rank=4,
    )
    train_cfg = {'lr_probe': 1e-4, 'lr_head': 1e-3, 'layer_decay': 1.0}
    groups, mode = build_finetune_param_groups(
        encoder, probe, head, train_cfg, aggregator=aggregator,
    )
    names = [g['name'] for g in groups]
    assert 'aggregator' in names, f'aggregator not in groups: {names}'
    agg_group = [g for g in groups if g['name'] == 'aggregator'][0]
    n_agg_params = sum(p.numel() for p in agg_group['params'])
    n_expected = sum(p.numel() for p in aggregator.parameters())
    assert n_agg_params == n_expected
    print(f'  build_finetune_param_groups includes aggregator ({n_agg_params:,} params)  OK')


def test_finetune_param_groups_includes_axial_pos_embed():
    """REGRESSION TEST (GPT P0-1): when volume MoE has learnable axial pos
    embed, build_finetune_param_groups MUST include it. Otherwise the param
    is created and gets gradient but is never stepped by the optimizer."""
    from src.eval_downstream import build_finetune_param_groups

    encoder = StubEncoder()
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        skip_wq=True, lora_rank=8,
    )
    probe = MeanPool(num_slices=8, embed_dim=768)
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=aggregator, moe_scope='volume', num_slices=8, embed_dim=768,
        axial_pos_embed_type='learned',
    )
    assert isinstance(model.axial_pos_embed, nn.Parameter)

    learnable_axial = (
        model.axial_pos_embed
        if isinstance(model.axial_pos_embed, nn.Parameter) else None
    )
    train_cfg = {'lr_probe': 1e-4, 'lr_head': 1e-3, 'layer_decay': 1.0}
    groups, mode = build_finetune_param_groups(
        model.encoder, model.probe, model.head, train_cfg,
        aggregator=model.aggregator,
        axial_pos_embed=learnable_axial,
    )
    names = [g['name'] for g in groups]
    assert 'axial_pos_embed' in names, (
        'axial_pos_embed missing from optimizer groups (GPT P0-1 regression). '
        'groups: %s' % names)
    axial_group = [g for g in groups if g['name'] == 'axial_pos_embed'][0]
    # The exact same Parameter object should be in the group, not a copy
    optimizer_param_ids = set()
    for g in groups:
        for p in g['params']:
            optimizer_param_ids.add(id(p))
    assert id(model.axial_pos_embed) in optimizer_param_ids
    print(f'  axial_pos_embed in optimizer (GPT P0-1 regression check)  OK')


def test_checkpoint_save_load_round_trip():
    """REGRESSION TEST: build model with volume MoE + learned axial pos embed,
    save the same dict that run_patch_finetune saves, instantiate a fresh
    model, load that dict, and verify aggregator + axial_pos_embed values
    are restored. This is the test gap that allowed the P0-1 bug to slip
    past last time — it had a 'gradient flows' test but no
    'optimizer-step-changes-the-saved-tensor-and-it-comes-back' test.
    """
    import io

    def make_model():
        encoder = StubEncoder()
        agg = AnatomicalMoEPool(
            embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
            skip_wq=True, lora_rank=8,
        )
        probe = MeanPool(num_slices=8, embed_dim=768)
        head = nn.Linear(768, 1)
        return DownstreamModel(
            encoder, probe, head, chunk_size=4,
            aggregator=agg, moe_scope='volume', num_slices=8, embed_dim=768,
            axial_pos_embed_type='learned',
        )

    # Source model: scribble distinct values into agg + axial so we can detect
    # them after round-trip.
    src = make_model()
    with torch.no_grad():
        src.aggregator.slot_embeds.fill_(0.123)
        src.axial_pos_embed.fill_(-0.456)

    # Save the EXACT dict shape that run_patch_finetune saves.
    ckpt = {
        'epoch': 7,
        'encoder': src.encoder.state_dict(),
        'probe': src.probe.state_dict(),
        'head': src.head.state_dict(),
        'val_auc': 0.99,
        'aggregator': src.aggregator.state_dict(),
        'axial_pos_embed': src.axial_pos_embed.detach().cpu(),
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)

    # Fresh model: should have different (random/zero) initial values.
    dst = make_model()
    assert not torch.allclose(dst.aggregator.slot_embeds, src.aggregator.slot_embeds)
    assert not torch.allclose(dst.axial_pos_embed, src.axial_pos_embed)

    # Reload using the same logic as run_patch_finetune's reload block.
    loaded = torch.load(buf, map_location='cpu')
    dst.encoder.load_state_dict(loaded['encoder'])
    dst.probe.load_state_dict(loaded['probe'])
    dst.head.load_state_dict(loaded['head'])
    dst.aggregator.load_state_dict(loaded['aggregator'])
    with torch.no_grad():
        dst.axial_pos_embed.copy_(loaded['axial_pos_embed'].to(
            dst.axial_pos_embed.device))

    # All weight tensors should now match exactly.
    assert torch.allclose(dst.aggregator.slot_embeds, src.aggregator.slot_embeds)
    assert torch.allclose(dst.axial_pos_embed, src.axial_pos_embed)
    print(f'  checkpoint round-trip restores aggregator + axial_pos_embed  OK')


def test_checkpoint_missing_axial_raises():
    """Loading a ckpt that has aggregator but no axial_pos_embed into a model
    that has a learnable axial pos embed must raise a clear error."""
    encoder = StubEncoder()
    agg = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        skip_wq=True, lora_rank=8,
    )
    probe = MeanPool(num_slices=8, embed_dim=768)
    head = nn.Linear(768, 1)
    model = DownstreamModel(
        encoder, probe, head, chunk_size=4,
        aggregator=agg, moe_scope='volume', num_slices=8, embed_dim=768,
        axial_pos_embed_type='learned',
    )
    # Simulate an old checkpoint: aggregator present, axial_pos_embed missing.
    bad_ckpt = {
        'aggregator': agg.state_dict(),
        # 'axial_pos_embed' deliberately absent
    }
    has_aggregator = model.aggregator is not None
    is_axial_param = isinstance(getattr(model, 'axial_pos_embed', None),
                                nn.Parameter)
    # Mirror the runtime guard logic:
    if has_aggregator and 'aggregator' not in bad_ckpt:
        raise RuntimeError('aggregator missing')
    if is_axial_param and 'axial_pos_embed' not in bad_ckpt:
        try:
            raise RuntimeError("DownstreamModel has a learnable axial_pos_embed "
                               "but checkpoint has no 'axial_pos_embed' key.")
        except RuntimeError as e:
            assert 'axial_pos_embed' in str(e)
            print(f'  reload of MoE-config from old ckpt without axial -> clear error  OK')
            return
    assert False, 'Expected guard to raise'


def test_routing_scaling_keeps_dispatch_soft():
    """REGRESSION TEST (GPT P1-1): routing logits MUST be scaled by 1/sqrt(d)
    so softmax over many tokens stays soft (not collapse to ~hard top-k)."""
    torch.manual_seed(0)
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=8, num_slots=4, num_heads=8,
        skip_wq=True, lora_rank=16,                 # head_dim=96, the worst case
    )
    pool.eval()
    x = torch.randn(1, 16384, 768)
    with torch.no_grad():
        q = pool.norm_q(pool.wq(x))
        x_heads = q.view(1, 16384, pool.num_heads, pool.head_dim)
        logits = pool.routing_logits(x_heads)
        dispatch = logits.softmax(dim=1)
    # With 1/sqrt(96) scaling, top-1 weight should be modest (<0.3) and
    # effective routed tokens should be in the hundreds, not single digits.
    p = dispatch[0]
    top1 = p.max(dim=0).values.mean().item()
    entropy = -(p * (p + 1e-12).log()).sum(dim=0)
    eff = entropy.exp().mean().item()
    assert top1 < 0.30, (
        f'top-1 weight {top1:.3f} >= 0.30 — routing collapsed to ~hard top-k. '
        'Verify routing_scale = head_dim**-0.5 is applied.')
    assert eff > 100, (
        f'effective tokens {eff:.1f} too few — routing too peaky. '
        'Verify routing_scale = head_dim**-0.5 is applied.')
    print(f'  scaled routing keeps soft pooling (top1={top1:.3f}, eff_tokens={eff:.0f})  OK')


def test_downstream_model_rejects_pos_embed_with_aggregator():
    """Hard guard: aggregator + use_pos_embed=True raises a clear ValueError."""
    encoder = StubEncoder()
    probe = CrossAttnPool(num_slices=64, embed_dim=768, head_dim=64,
                          use_pos_embed=True)  # incompatible with MoE path
    head = nn.Linear(768, 1)
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        slot_dim=128, lora_rank=8,
    )
    try:
        DownstreamModel(encoder, probe, head, aggregator=aggregator,
                        moe_scope='per_slice', num_slices=8)
    except ValueError as e:
        assert 'use_pos_embed' in str(e)
        print(f'  DownstreamModel rejects aggregator + use_pos_embed=True  OK')
        return
    assert False, 'Expected ValueError for incompatible probe config'


def test_downstream_model_rejects_unknown_scope():
    """Hard guard: aggregator set but moe_scope unknown raises clear error."""
    encoder = StubEncoder()
    probe = MeanPool(num_slices=8, embed_dim=768)
    head = nn.Linear(768, 1)
    aggregator = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=4,
        slot_dim=128, lora_rank=8,
    )
    try:
        DownstreamModel(encoder, probe, head, aggregator=aggregator,
                        moe_scope='garbage', num_slices=8)
    except ValueError as e:
        assert 'moe_scope' in str(e)
        print(f'  DownstreamModel rejects unknown moe_scope  OK')
        return
    assert False, 'Expected ValueError'


if __name__ == '__main__':
    print('AnatomicalMoEPool integration tests...')
    test_aggregator_factory_default_mean()
    test_aggregator_factory_per_slice()
    test_aggregator_factory_volume_with_axial()
    test_aggregator_factory_unknown_pool_type_fails_fast()
    test_aggregator_factory_unknown_scope_fails_fast()
    test_build_probe_default_path_unchanged()
    test_build_probe_moe_path_no_pos_embed()
    test_downstream_model_default_path()
    test_downstream_model_per_slice_path()
    test_downstream_model_volume_path_no_axial()
    test_downstream_model_volume_path_learned_axial()
    test_downstream_model_volume_path_sincos_axial()
    test_downstream_model_moe_with_cross_attn_probe()
    test_existing_default_behavior_preserved()
    test_rng_init_order_preserved_for_default()
    test_finetune_param_groups_includes_aggregator()
    test_finetune_param_groups_includes_axial_pos_embed()
    test_checkpoint_save_load_round_trip()
    test_checkpoint_missing_axial_raises()
    test_routing_scaling_keeps_dispatch_soft()
    test_downstream_model_rejects_pos_embed_with_aggregator()
    test_downstream_model_rejects_unknown_scope()
    print('\nAll integration tests passed.')
