"""Thin wrapper around the official TokenCut normalized-cut implementation.

TokenCut was released for DINO key features. Passing another model's patch
features is an adaptation and must be reported as ``TokenCut-style NCut``.
"""

from dataclasses import dataclass
import importlib.util
import os

import numpy as np
import torch


@dataclass
class TokenCutResult:
    masks: torch.Tensor
    eigenvectors: torch.Tensor
    boxes: tuple
    seeds: tuple
    metadata: dict


def load_official_ncut(tokencut_root=None):
    """Load ``ncut`` from a pinned external TokenCut checkout."""
    root = tokencut_root or os.environ.get("TOKENCUT_ROOT")
    if not root:
        raise ValueError(
            "TokenCut root is required; set TOKENCUT_ROOT to the official "
            "checkout pinned in the evidence manifest"
        )
    module_path = os.path.join(root, "object_discovery.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            "official TokenCut object_discovery.py not found: %s"
            % module_path
        )
    spec = importlib.util.spec_from_file_location(
        "_phase0_official_tokencut", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load TokenCut module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ncut


def tokencut_partition(
    patch_tokens,
    grid_size,
    image_size,
    tokencut_root=None,
    tau=0.2,
    eps=1e-5,
    ncut_fn=None,
):
    """Run official TokenCut NCut over a batch of adapted patch features.

    Args:
        patch_tokens: ``(B, N, D)`` final normalized patch features.
        grid_size: ``(H, W)`` matching ``N``.
        image_size: ``(height, width)`` in pixels.
        tokencut_root: Pinned official checkout root.
        tau: Official binary affinity threshold.
        eps: Official absent-edge weight.
        ncut_fn: Test-only injected function with the official signature.
    """
    if patch_tokens.dim() != 3:
        raise ValueError("patch_tokens must have shape (B, N, D)")
    height, width = (int(grid_size[0]), int(grid_size[1]))
    if patch_tokens.size(1) != height * width:
        raise ValueError("patch token count does not match grid size")
    image_height, image_width = (int(image_size[0]), int(image_size[1]))
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_size must contain positive integers")
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    official_ncut = ncut_fn or load_official_ncut(tokencut_root)
    scales = (
        float(image_height) / float(height),
        float(image_width) / float(width),
    )
    masks = []
    eigenvectors = []
    boxes = []
    seeds = []
    for index in range(patch_tokens.size(0)):
        patches = patch_tokens[index].detach().float().cpu()
        # The official function ignores the prefix value but slices it away.
        prefix = torch.zeros(1, patches.size(1), dtype=patches.dtype)
        official_input = torch.cat([prefix, patches], dim=0).unsqueeze(0)
        box, _, mask, seed, _, eigenvector = official_ncut(
            official_input,
            (height, width),
            scales,
            (3, image_height, image_width),
            tau=float(tau),
            eps=float(eps),
        )
        masks.append(torch.from_numpy(np.asarray(mask)).float())
        eigenvectors.append(
            torch.from_numpy(np.asarray(eigenvector)).float()
        )
        boxes.append(tuple(float(value) for value in box))
        seeds.append(int(seed))

    return TokenCutResult(
        masks=torch.stack(masks, dim=0),
        eigenvectors=torch.stack(eigenvectors, dim=0),
        boxes=tuple(boxes),
        seeds=tuple(seeds),
        metadata={
            "algorithm": "TokenCut-style NCut",
            "source_repository": "YangtaoWANG95/TokenCut",
            "source_commit": "fed52cd5b60891baefd8ec7110dafa73be816ee1",
            "tau": float(tau),
            "eps": float(eps),
            "feature_source": "adapted_patch_tokens",
        },
    )
