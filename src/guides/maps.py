"""Model-neutral map extraction and label-free map diagnostics."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import GuideOutput


def minmax_normalize(score_map, eps=1e-8):
    """Normalize each (H, W) map independently to [0, 1]."""
    if score_map.dim() != 3:
        raise ValueError("score_map must have shape (B, H, W)")
    flat = score_map.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1)
    return (score_map - minimum) / (maximum - minimum + eps)


def rank_normalize(score_map):
    """Return per-image average mid-ranks in (0, 1)."""
    if score_map.dim() != 3:
        raise ValueError("score_map must have shape (B, H, W)")
    flat = score_map.flatten(1)
    sorted_values, order = torch.sort(flat, dim=1)
    ranks = torch.empty_like(flat, dtype=torch.float32)
    for batch_index in range(flat.size(0)):
        _, counts = torch.unique_consecutive(
            sorted_values[batch_index], return_counts=True
        )
        start = 0
        for count in counts.tolist():
            stop = start + int(count)
            average_rank = 0.5 * float(start + stop - 1)
            ranks[
                batch_index,
                order[batch_index, start:stop],
            ] = average_rank
            start = stop
    ranks = (ranks + 0.5) / float(flat.size(1))
    return ranks.view_as(score_map)


def global_patch_cosine(output):
    """Cosine similarity between same-space global and patch features."""
    if output.global_token is None:
        raise ValueError("guide output has no same-space global token")
    patches = F.normalize(output.patch_tokens.float(), dim=-1)
    global_token = F.normalize(output.global_token.float(), dim=-1)
    scores = torch.einsum("bnd,bd->bn", patches, global_token)
    return scores.view(output.batch_size, *output.grid_size)


def token_pca_map(output):
    """First patch-token principal component with deterministic sign."""
    tokens = output.patch_tokens.float()
    centered = tokens - tokens.mean(dim=1, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    direction = right[:, 0]
    pivot_index = direction.abs().argmax(dim=1, keepdim=True)
    pivot = direction.gather(1, pivot_index)
    direction = direction * torch.where(
        pivot < 0,
        -torch.ones_like(pivot),
        torch.ones_like(pivot),
    )
    scores = torch.einsum("bnd,bd->bn", centered, direction)
    return scores.view(output.batch_size, *output.grid_size)


def resize_map(score_map, size):
    """Resize maps while retaining a (B, H, W) tensor contract."""
    if score_map.dim() != 3:
        raise ValueError("score_map must have shape (B, H, W)")
    return F.interpolate(
        score_map.unsqueeze(1).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def top_fraction_mask(score_map, fraction):
    """Select exactly round(fraction*N) highest-scoring cells per image."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    flat = score_map.flatten(1)
    count = max(1, int(round(fraction * flat.size(1))))
    indices = torch.topk(flat, k=count, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.view_as(score_map)


def map_entropy(score_map, temperature=0.15, eps=1e-8):
    """Entropy of a min-max-normalized spatial softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = minmax_normalize(score_map).flatten(1) / float(temperature)
    probabilities = torch.softmax(flat, dim=1)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)
    normalizer = torch.log(
        torch.tensor(float(flat.size(1)), device=flat.device)
    )
    return entropy / normalizer


def effective_support(score_map, temperature=0.15, eps=1e-8):
    """Effective spatial support exp(H(p))/N."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = minmax_normalize(score_map).flatten(1) / float(temperature)
    probabilities = torch.softmax(flat, dim=1)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)
    return torch.exp(entropy) / float(flat.size(1))


def normalized_total_variation(score_map, eps=1e-8):
    """Mean adjacent absolute difference divided by map dynamic range."""
    score_map = score_map.float()
    horizontal = (score_map[:, :, 1:] - score_map[:, :, :-1]).abs().mean(
        dim=(1, 2)
    )
    vertical = (score_map[:, 1:, :] - score_map[:, :-1, :]).abs().mean(
        dim=(1, 2)
    )
    dynamic_range = (
        score_map.flatten(1).max(dim=1).values
        - score_map.flatten(1).min(dim=1).values
    )
    return 0.5 * (horizontal + vertical) / (dynamic_range + eps)


def center_distance(score_map, eps=1e-8):
    """Distance of non-negative score mass from image center, normalized."""
    weights = minmax_normalize(score_map).float() + eps
    batch, height, width = weights.shape
    ys = torch.linspace(0.0, 1.0, height, device=weights.device)
    xs = torch.linspace(0.0, 1.0, width, device=weights.device)
    mass = weights.sum(dim=(1, 2))
    cy = (weights * ys.view(1, height, 1)).sum(dim=(1, 2)) / mass
    cx = (weights * xs.view(1, 1, width)).sum(dim=(1, 2)) / mass
    return torch.sqrt((cy - 0.5).pow(2) + (cx - 0.5).pow(2))


def border_occupancy(score_map, fraction=0.25):
    """Fraction of top-score cells occupying the outer one-cell border."""
    mask = top_fraction_mask(score_map, fraction)
    border = torch.zeros_like(mask)
    border[:, 0, :] = True
    border[:, -1, :] = True
    border[:, :, 0] = True
    border[:, :, -1] = True
    selected = mask.sum(dim=(1, 2)).clamp(min=1)
    return (mask & border).sum(dim=(1, 2)).float() / selected.float()


def summarize_map(score_map, temperature=0.15):
    """Return scalar label-free diagnostics for every map in a batch."""
    normalized = minmax_normalize(score_map)
    return {
        "minimum": normalized.flatten(1).min(dim=1).values,
        "maximum": normalized.flatten(1).max(dim=1).values,
        "mean": normalized.mean(dim=(1, 2)),
        "std": normalized.std(dim=(1, 2), unbiased=False),
        "entropy": map_entropy(score_map, temperature=temperature),
        "effective_support": effective_support(
            score_map, temperature=temperature
        ),
        "total_variation": normalized_total_variation(score_map),
        "center_distance": center_distance(score_map),
        "border_occupancy_25": border_occupancy(score_map, fraction=0.25),
    }


def extract_maps(output):
    """Extract every map supported by a guide output."""
    maps = {}
    if output.native_map is not None:
        maps["native"] = output.native_map.float()
    if output.global_token is not None:
        maps["global_cosine"] = global_patch_cosine(output)
    maps["token_pca"] = token_pca_map(output)
    return maps
