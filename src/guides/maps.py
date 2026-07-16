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


def token_pca_rgb(output, foreground_mask=None, whiten=True):
    """Project patch tokens to three PCA channels for visualization.

    This follows the DINOv3 notebook's use of three whitened principal
    components. A supplied foreground mask controls which patches fit PCA;
    without one, all patches are used and the result must be labeled as an
    adapted diagnostic rather than the official foreground-PCA recipe.

    Returns:
        CPU float tensor with shape ``(B, H, W, 3)``.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError(
            "token_pca_rgb requires scikit-learn in the Phase-0 environment"
        ) from exc

    tokens = output.patch_tokens.detach().float().cpu()
    batch, count, _ = tokens.shape
    height, width = output.grid_size
    if foreground_mask is not None:
        if tuple(foreground_mask.shape) != (batch, height, width):
            raise ValueError(
                "foreground_mask must have shape (B, H, W)"
            )
        foreground_mask = foreground_mask.detach().bool().cpu().flatten(1)

    images = []
    for index in range(batch):
        values = tokens[index].numpy()
        fit_values = values
        if foreground_mask is not None:
            selected = foreground_mask[index].numpy()
            if int(selected.sum()) >= 3:
                fit_values = values[selected]
        components = min(3, fit_values.shape[0], fit_values.shape[1])
        if components <= 0:
            raise ValueError("PCA requires at least one patch feature")
        transformed = PCA(
            n_components=components,
            whiten=bool(whiten),
            svd_solver="full",
        ).fit(fit_values).transform(values)
        if components < 3:
            padding = torch.zeros(count, 3 - components)
            transformed = torch.cat(
                [torch.from_numpy(transformed).float(), padding], dim=1
            )
        else:
            transformed = torch.from_numpy(transformed).float()
        images.append(transformed.view(height, width, 3))
    return torch.stack(images, dim=0)


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


def _normalized_point(point):
    x, y = point.point_2d
    if point.coordinate_space == "normalized_1000":
        return float(x) / 1000.0, float(y) / 1000.0
    if point.coordinate_space == "pixels" and point.image_size is not None:
        width, height = point.image_size
        return float(x) / float(width), float(y) / float(height)
    raise ValueError("grounding point has no usable coordinate space")


def _normalized_box(box):
    x1, y1, x2, y2 = box.bbox_2d
    if box.coordinate_space == "normalized_1000":
        return tuple(float(value) / 1000.0 for value in (x1, y1, x2, y2))
    if box.coordinate_space == "pixels" and box.image_size is not None:
        width, height = box.image_size
        return (
            float(x1) / float(width),
            float(y1) / float(height),
            float(x2) / float(width),
            float(y2) / float(height),
        )
    raise ValueError("grounding box has no usable coordinate space")


def grounding_score_map(
    grounding_regions,
    grounding_points,
    grid_size=(16, 16),
    point_sigma_fraction=0.08,
):
    """Rasterize only model-native boxes/points onto a shared score grid."""
    height, width = int(grid_size[0]), int(grid_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("grid_size must contain positive integers")
    sigma = float(point_sigma_fraction)
    if sigma <= 0:
        raise ValueError("point_sigma_fraction must be positive")
    if grounding_regions is None and grounding_points is None:
        raise ValueError("at least one grounding collection is required")
    batch = len(
        grounding_regions
        if grounding_regions is not None
        else grounding_points
    )
    if grounding_regions is None:
        grounding_regions = [[] for _ in range(batch)]
    if grounding_points is None:
        grounding_points = [[] for _ in range(batch)]
    if len(grounding_regions) != batch or len(grounding_points) != batch:
        raise ValueError("grounding batch lengths differ")

    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    center_y, center_x = torch.meshgrid(ys, xs, indexing="ij")
    cell_y1 = torch.arange(height, dtype=torch.float32) / height
    cell_y2 = torch.arange(1, height + 1, dtype=torch.float32) / height
    cell_x1 = torch.arange(width, dtype=torch.float32) / width
    cell_x2 = torch.arange(1, width + 1, dtype=torch.float32) / width
    cell_area = 1.0 / float(height * width)

    maps = []
    for boxes, points in zip(grounding_regions, grounding_points):
        score = torch.zeros(height, width, dtype=torch.float32)
        for box in boxes:
            x1, y1, x2, y2 = _normalized_box(box)
            overlap_y = (
                torch.minimum(cell_y2, torch.tensor(y2))
                - torch.maximum(cell_y1, torch.tensor(y1))
            ).clamp(min=0)
            overlap_x = (
                torch.minimum(cell_x2, torch.tensor(x2))
                - torch.maximum(cell_x1, torch.tensor(x1))
            ).clamp(min=0)
            coverage = overlap_y[:, None] * overlap_x[None, :] / cell_area
            score = torch.maximum(score, coverage.clamp(max=1.0))
        for point in points:
            point_x, point_y = _normalized_point(point)
            squared_distance = (
                (center_x - point_x).pow(2)
                + (center_y - point_y).pow(2)
            )
            gaussian = torch.exp(
                -0.5 * squared_distance / (sigma * sigma)
            )
            score = torch.maximum(score, gaussian)
        maps.append(score)
    return torch.stack(maps, dim=0)


def illustrative_target_rectangles(
    score_map,
    block_size=(4, 4),
    count=4,
):
    """Select non-overlapping top-average rectangles for visualization only."""
    if score_map.dim() != 3:
        raise ValueError("score_map must have shape (B, H, W)")
    block_height, block_width = int(block_size[0]), int(block_size[1])
    if block_height <= 0 or block_width <= 0:
        raise ValueError("block_size must contain positive integers")
    if (
        block_height > score_map.size(1)
        or block_width > score_map.size(2)
    ):
        raise ValueError("block_size exceeds score map")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")

    averages = F.avg_pool2d(
        score_map.unsqueeze(1).float(),
        kernel_size=(block_height, block_width),
        stride=1,
    ).squeeze(1)
    results = []
    for batch_index in range(score_map.size(0)):
        candidates = averages[batch_index].clone()
        occupied = torch.zeros(
            score_map.size(1), score_map.size(2), dtype=torch.bool
        )
        rectangles = []
        for _ in range(count):
            valid = torch.ones_like(candidates, dtype=torch.bool)
            for row in range(candidates.size(0)):
                for column in range(candidates.size(1)):
                    if occupied[
                        row:row + block_height,
                        column:column + block_width,
                    ].any():
                        valid[row, column] = False
            if not valid.any():
                break
            ranked = candidates.masked_fill(~valid, float("-inf"))
            flat_index = int(ranked.flatten().argmax().item())
            row = flat_index // ranked.size(1)
            column = flat_index % ranked.size(1)
            value = float(ranked[row, column].item())
            rectangles.append(
                (row, column, block_height, block_width, value)
            )
            occupied[
                row:row + block_height,
                column:column + block_width,
            ] = True
        results.append(tuple(rectangles))
    return tuple(results)


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
    if output.metadata.get("spatial_token_grid", True):
        if output.global_token is not None:
            maps["global_cosine"] = global_patch_cosine(output)
        maps["token_pca"] = token_pca_map(output)
    return maps
