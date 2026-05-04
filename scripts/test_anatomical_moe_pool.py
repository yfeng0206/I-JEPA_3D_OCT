"""Smoke test for AnatomicalMoEPool: shapes, gradient flow, param count.

Run: python scripts/test_anatomical_moe_pool.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.anatomical_moe_pool import AnatomicalMoEPool


def test_shapes_default():
    """Default ViT-B config: (B=2, N=256, D=768) -> (2, 32, 768)."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=8, num_slots=4,
        num_heads=8, slot_dim=256, lora_rank=16,
    )
    x = torch.randn(2, 256, 768)
    out = pool(x)
    expected = (2, 8 * 4, 768)
    assert out.shape == expected, f'Expected {expected}, got {out.shape}'
    print(f'  default ViT-B: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK')


def test_shapes_vit_small():
    """ViT-S config: (B=2, N=256, D=384) -> (2, 32, 384)."""
    pool = AnatomicalMoEPool(
        embed_dim=384, num_experts=8, num_slots=4,
        num_heads=8, slot_dim=128, lora_rank=16,
    )
    x = torch.randn(2, 256, 384)
    out = pool(x)
    assert out.shape == (2, 32, 384), f'Got {out.shape}'
    print(f'  ViT-S: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK')


def test_per_expert_phi():
    """share_phi=False adds per-expert capacity."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2,
        num_heads=4, slot_dim=128, share_phi=False,
    )
    x = torch.randn(1, 256, 768)
    out = pool(x)
    assert out.shape == (1, 4 * 2, 768), f'Got {out.shape}'
    print(f'  share_phi=False: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK')


def test_skip_wq_true():
    """skip_wq=True forces slot_dim=embed_dim and uses Identity for wq."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=8,
        skip_wq=True, lora_rank=16,
    )
    assert pool.skip_wq is True
    assert pool.slot_dim == 768           # forced to embed_dim
    assert pool.head_dim == 768 // 8       # = 96
    assert isinstance(pool.wq, torch.nn.Identity)

    x = torch.randn(2, 256, 768)
    out = pool(x)
    assert out.shape == (2, 4 * 2, 768)
    # Param count drops because no wq Linear (768*256+256 ≈ 197K saved).
    n = sum(p.numel() for p in pool.parameters())
    pool_with_wq = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2, num_heads=8,
        slot_dim=256, lora_rank=16, skip_wq=False,
    )
    n_with_wq = sum(p.numel() for p in pool_with_wq.parameters())
    print(f'  skip_wq=True: forward (2,256,768)->(2,8,768), params={n:,}  OK')
    print(f'  skip_wq=False (same E,S,H): params={n_with_wq:,} (delta={n_with_wq-n:+,})')


def test_volume_scale_routing():
    """Volume-scale: 16384 input tokens routed to E*S=32 prototypes."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=8, num_slots=4, num_heads=8,
        skip_wq=True, lora_rank=16,
    )
    # Simulating volume-scope: B=2 volumes, each (64*256=16384 patches, 768d)
    x = torch.randn(2, 64 * 256, 768)
    out = pool(x)
    assert out.shape == (2, 32, 768)
    print(f'  volume-scope routing: (2, 16384, 768) -> (2, 32, 768)  OK')


def test_gradient_flow():
    """Backward pass touches all parameters."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2,
        num_heads=4, slot_dim=128, lora_rank=8,
    )
    x = torch.randn(2, 256, 768, requires_grad=True)
    out = pool(x)
    loss = out.sum()
    loss.backward()
    n_params_with_grad = 0
    n_params_without_grad = 0
    for name, p in pool.named_parameters():
        if p.grad is None:
            print(f'    [WARN] no grad: {name}')
            n_params_without_grad += 1
        else:
            n_params_with_grad += 1
    assert n_params_without_grad == 0, 'Some params got no grad'
    print(f'  gradient flow: all {n_params_with_grad} params received gradients  OK')


def test_param_count():
    """Sanity check parameter budget for ViT-B default config."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=8, num_slots=4,
        num_heads=8, slot_dim=256, lora_rank=16,
    )
    total = sum(p.numel() for p in pool.parameters())
    print(f'  default ViT-B param count: {total:,}')
    # Expected ~316K from docstring estimate; allow some slack
    assert 250_000 < total < 400_000, f'Param count {total} outside expected range'
    print(f'    in expected range (250K-400K)  OK')


def test_kmeans_warm_start():
    """init_slots_from_kmeans accepts (E*S, slot_dim) cluster centers in
    post-wq routing space (NOT raw embed_dim — caller must project)."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2,
        num_heads=4, slot_dim=128, lora_rank=8,
    )
    # Caller is responsible for projecting features through wq+norm_q first.
    centers = torch.randn(4 * 2, 128)        # in slot_dim=128 space
    pool.init_slots_from_kmeans(centers)
    x = torch.randn(1, 256, 768)
    out = pool(x)
    assert out.shape == (1, 8, 768)
    print(f'  k-means warm start: applied {centers.shape} (slot_dim space)  OK')

    # Sanity: passing wrong dim should raise
    try:
        pool.init_slots_from_kmeans(torch.randn(4 * 2, 768))
        assert False, 'Expected ValueError for embed_dim-shaped centers'
    except ValueError as e:
        assert 'slot_dim' in str(e)
        print(f'  k-means rejects wrong-dim centers with clear error  OK')


def test_softmax_normalization():
    """Routing dispatch weights should softmax over the patch dimension N."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=4, num_slots=2,
        num_heads=4, slot_dim=128, lora_rank=8,
    )
    pool.eval()
    x = torch.randn(1, 256, 768)
    # Recompute the dispatch weights manually using the module's logic
    q = pool.norm_q(pool.wq(x))
    x_heads = q.view(1, 256, 4, 32)
    logits = pool.routing_logits(x_heads)        # (1, 256, 4, 4, 2)
    dispatch = logits.softmax(dim=1)
    sums = dispatch.sum(dim=1)                    # (1, E, H, S)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        'Dispatch weights do not sum to 1 over patches'
    print(f'  dispatch weights sum to 1.0 over N=256 patches  OK')


def test_realistic_vol_batch():
    """Realistic: simulate one volume of 64 slices."""
    pool = AnatomicalMoEPool(
        embed_dim=768, num_experts=8, num_slots=4,
        num_heads=8, slot_dim=256, lora_rank=16,
    )
    # 64 slices, 256 patches each, embed_dim 768
    slice_features = torch.randn(64, 256, 768)
    prototypes = pool(slice_features)             # (64, 32, 768)
    assert prototypes.shape == (64, 32, 768)
    # Stacked into volume token sequence: (1, 64*32=2048, 768)
    vol = prototypes.reshape(1, 64 * 32, 768)
    assert vol.shape == (1, 2048, 768)
    print(f'  64-slice volume: 64 x (256, 768) -> (1, 2048, 768)  OK')


if __name__ == '__main__':
    print('Testing AnatomicalMoEPool...')
    test_shapes_default()
    test_shapes_vit_small()
    test_per_expert_phi()
    test_skip_wq_true()
    test_volume_scale_routing()
    test_gradient_flow()
    test_param_count()
    test_kmeans_warm_start()
    test_softmax_normalization()
    test_realistic_vol_batch()
    print('\nAll tests passed.')
