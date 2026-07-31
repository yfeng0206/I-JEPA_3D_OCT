"""Paper-style and unsupervised PCA visualizations for DINO patch features."""

from itertools import permutations, product

import numpy as np
from PIL import Image
from scipy import signal
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_aspect_to_patch_grid(image, image_height=768, patch_size=16):
    """Match the official tutorial's aspect-preserving, patch-aligned resize."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    image_height = int(image_height)
    patch_size = int(patch_size)
    if image_height <= 0 or patch_size <= 0:
        raise ValueError("image_height and patch_size must be positive")
    height_patches = image_height // patch_size
    if height_patches <= 0:
        raise ValueError("image_height must cover at least one patch")
    width, height = image.size
    width_patches = max(
        1, int((width * image_height) / (height * patch_size))
    )
    target_size = (
        height_patches * patch_size,
        width_patches * patch_size,
    )
    tensor = TF.to_tensor(TF.resize(image, target_size))
    return tensor, (height_patches, width_patches)


def normalize_imagenet(tensor):
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError("tensor must have shape (3, H, W)")
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def quantize_foreground_mask(mask, grid_size, patch_size=16):
    """Average an alpha/greyscale mask over the exact DINO patch grid."""
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL image")
    height, width = int(grid_size[0]), int(grid_size[1])
    resized = TF.to_tensor(
        TF.resize(mask.convert("L"), (height * patch_size, width * patch_size))
    )
    pooled = torch.nn.functional.avg_pool2d(
        resized.unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return pooled.squeeze(0).squeeze(0).flatten()


def classifier_state(classifier):
    return {
        "coef": np.asarray(classifier.coef_, dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
    }


def predict_foreground_probability(features, state):
    values = np.asarray(features, dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    intercept = np.asarray(state["intercept"], dtype=np.float64)
    classes = np.asarray(state["classes"], dtype=np.int64)
    if values.ndim != 2 or coef.ndim != 2:
        raise ValueError("features and classifier coefficients must be matrices")
    if values.shape[1] != coef.shape[1]:
        raise ValueError("classifier feature dimension mismatch")
    if classes.tolist() != [0, 1] or coef.shape[0] != 1:
        raise ValueError("expected a binary sklearn logistic classifier")
    logits = values @ coef[0] + intercept[0]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def median_foreground_mask(probability, grid_size, threshold=0.5):
    height, width = int(grid_size[0]), int(grid_size[1])
    values = np.asarray(probability, dtype=np.float32)
    if values.shape != (height * width,):
        raise ValueError("foreground probabilities do not match the grid")
    filtered = signal.medfilt2d(
        values.reshape(height, width), kernel_size=3
    )
    return filtered, filtered > float(threshold)


def paper_style_pca(features, grid_size, foreground_mask, whiten=True):
    """Released-notebook PCA: foreground fit, sigmoid colors, black background."""
    values = np.asarray(features, dtype=np.float32)
    height, width = int(grid_size[0]), int(grid_size[1])
    mask = np.asarray(foreground_mask, dtype=bool)
    if values.ndim != 2 or values.shape[0] != height * width:
        raise ValueError("features do not match grid_size")
    if mask.shape != (height, width):
        raise ValueError("foreground_mask does not match grid_size")
    selected = mask.reshape(-1)
    if int(selected.sum()) < 3:
        raise ValueError("paper-style PCA requires at least three foreground patches")
    pca = PCA(n_components=3, whiten=bool(whiten))
    pca.fit(values[selected])
    projected = pca.transform(values).reshape(height, width, 3)
    rgb = 1.0 / (1.0 + np.exp(-2.0 * projected))
    rgb *= mask[..., None]
    return {
        "pca": pca,
        "projected": projected.astype(np.float32),
        "rgb": rgb.astype(np.float32),
    }


def pca_orientation_variants(projected, foreground_mask):
    """Return the 48 sign/channel variants inspected in DINOv3 Section 6.1.1."""
    values = np.asarray(projected, dtype=np.float32)
    mask = np.asarray(foreground_mask, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("projected must have shape (H, W, 3)")
    if mask.shape != values.shape[:2]:
        raise ValueError("foreground_mask shape mismatch")
    variants = []
    for order in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            oriented = values[..., order] * np.asarray(signs, dtype=np.float32)
            rgb = 1.0 / (1.0 + np.exp(-2.0 * oriented))
            rgb *= mask[..., None]
            variants.append(
                {
                    "order": tuple(int(item) for item in order),
                    "signs": tuple(int(item) for item in signs),
                    "rgb": rgb.astype(np.float32),
                }
            )
    if len(variants) != 48:
        raise AssertionError("expected 48 PCA orientation variants")
    return tuple(variants)


def _channel_minmax(values, fit_mask=None, eps=1e-8):
    result = np.zeros_like(values, dtype=np.float32)
    if fit_mask is None:
        fit_mask = np.ones(values.shape[:2], dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != values.shape[:2] or not fit_mask.any():
        raise ValueError("fit_mask must select at least one spatial value")
    for channel in range(values.shape[-1]):
        plane = values[..., channel]
        minimum = float(plane[fit_mask].min())
        maximum = float(plane[fit_mask].max())
        result[..., channel] = (plane - minimum) / (
            maximum - minimum + eps
        )
    return np.clip(result, 0.0, 1.0)


def dinov2_two_stage_pca(features, grid_size, threshold=0.5):
    """Two-stage unsupervised PCA with both unresolved PC1 polarities."""
    values = np.asarray(features, dtype=np.float32)
    height, width = int(grid_size[0]), int(grid_size[1])
    if values.ndim != 2 or values.shape[0] != height * width:
        raise ValueError("features do not match grid_size")
    first = PCA(n_components=1).fit_transform(values).reshape(height, width)
    first_min = float(first.min())
    first_max = float(first.max())
    normalized = (first - first_min) / (first_max - first_min + 1e-8)
    outputs = []
    for name, mask in (
        ("high_pc1", normalized > float(threshold)),
        ("low_pc1", normalized <= float(threshold)),
    ):
        selected = mask.reshape(-1)
        if int(selected.sum()) < 3:
            outputs.append(
                {"polarity": name, "mask": mask, "rgb": None}
            )
            continue
        projected = PCA(n_components=3).fit(values[selected]).transform(values)
        rgb = _channel_minmax(
            projected.reshape(height, width, 3), fit_mask=mask
        )
        rgb *= mask[..., None]
        outputs.append(
            {
                "polarity": name,
                "mask": mask,
                "rgb": rgb.astype(np.float32),
            }
        )
    return {
        "pc1": first.astype(np.float32),
        "pc1_normalized": normalized.astype(np.float32),
        "polarities": tuple(outputs),
    }
