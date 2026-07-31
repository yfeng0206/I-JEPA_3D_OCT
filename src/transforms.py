"""Minimal data augmentation for I-JEPA OCT pretraining.

I-JEPA relies on very light augmentation -- typically just a random
resized crop and normalisation.  This module exposes a single factory
function ``make_transforms`` whose defaults match the original I-JEPA
paper.

Compatible with PyTorch 1.13.1 / Python 3.8.
"""

from typing import List, Optional, Tuple

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ImageNet statistics (used as a reasonable default for transfer learning).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(
    crop_size=256,                          # type: int
    crop_scale=(0.3, 1.0),                  # type: Tuple[float, float]
    gaussian_blur=False,                    # type: bool
    horizontal_flip=False,                  # type: bool
    color_distortion=False,                 # type: bool
    color_jitter=0.0,                       # type: float
    normalize_mean=IMAGENET_MEAN,           # type: Tuple[float, ...]
    normalize_std=IMAGENET_STD,             # type: Tuple[float, ...]
):
    """Build a ``torchvision.transforms.Compose`` pipeline.

    The pipeline expects a PIL Image as input and returns a
    ``(3, crop_size, crop_size)`` float tensor normalised with the
    given channel statistics.

    Args:
        crop_size: Output spatial size after random-resized crop.
        crop_scale: (min, max) area fraction for ``RandomResizedCrop``.
        gaussian_blur: Apply Gaussian blur (kernel 23, sigma [0.1, 2.0]).
        horizontal_flip: Apply random horizontal flip (p=0.5).
        color_distortion: Enable color jitter augmentation.
        color_jitter: Strength of color jitter (brightness, contrast,
            saturation all set to this value; hue set to 0).
        normalize_mean: Per-channel mean for ``Normalize``.
        normalize_std: Per-channel std for ``Normalize``.

    Returns:
        ``torchvision.transforms.Compose`` instance.
    """
    ops = []  # type: List[object]

    # -- Spatial ----------------------------------------------------------
    ops.append(
        T.RandomResizedCrop(
            crop_size,
            scale=crop_scale,
            interpolation=T.InterpolationMode.BICUBIC
            if hasattr(T.InterpolationMode, "BICUBIC")
            else 3,  # PIL.Image.BICUBIC for older torchvision
        )
    )

    if horizontal_flip:
        ops.append(T.RandomHorizontalFlip(p=0.5))

    # -- Color ------------------------------------------------------------
    if color_distortion and color_jitter > 0.0:
        ops.append(
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=0.0,
            )
        )

    # -- Blur -------------------------------------------------------------
    if gaussian_blur:
        # GaussianBlur was added in torchvision 0.8.
        ops.append(T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))

    # -- Tensor conversion & normalisation --------------------------------
    ops.append(T.ToTensor())  # PIL -> (C, H, W) float in [0, 1]
    ops.append(T.Normalize(mean=normalize_mean, std=normalize_std))

    return T.Compose(ops)


class PairedRandomResizedCrop(object):
    """Apply one ``RandomResizedCrop`` draw to an OCT slice *and* its guide.

    A precomputed semantic guide (e.g. the MIRAGE retinal union) cannot be
    cropped independently of the image it describes: the two must share the
    exact same crop rectangle or the guide points at the wrong anatomy.  This
    transform draws the crop parameters once and applies them to both.

    The image path is deliberately identical to :func:`make_transforms`
    (bicubic resized crop -> ``ToTensor`` -> ``Normalize``) so the OCT tensor is
    numerically unchanged for a given RNG state.  The guide uses nearest
    interpolation, which keeps hard labels hard, and is never normalised.
    """

    def __init__(
        self,
        crop_size=256,                      # type: int
        crop_scale=(0.3, 1.0),              # type: Tuple[float, float]
        ratio=(3.0 / 4.0, 4.0 / 3.0),       # type: Tuple[float, float]
        normalize_mean=IMAGENET_MEAN,       # type: Tuple[float, ...]
        normalize_std=IMAGENET_STD,         # type: Tuple[float, ...]
    ):
        self.crop_size = crop_size
        self.crop_scale = tuple(crop_scale)
        self.ratio = tuple(ratio)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self._interpolation = (
            T.InterpolationMode.BICUBIC
            if hasattr(T.InterpolationMode, "BICUBIC")
            else 3
        )
        self._guide_interpolation = (
            T.InterpolationMode.NEAREST
            if hasattr(T.InterpolationMode, "NEAREST")
            else 0
        )

    def __call__(self, image, guide=None):
        """Return ``(image_tensor, guide_image)`` sharing one crop rectangle.

        Args:
            image: PIL image of the OCT slice.
            guide: optional single-channel PIL image of the aligned guide, at
                the same resolution as ``image``.
        """
        if guide is not None and guide.size != image.size:
            raise ValueError(
                "Guide size %s must match image size %s before the paired crop"
                % (guide.size, image.size)
            )
        top, left, height, width = T.RandomResizedCrop.get_params(
            image, list(self.crop_scale), list(self.ratio)
        )
        cropped = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [self.crop_size, self.crop_size],
            self._interpolation,
        )
        tensor = TF.normalize(
            TF.to_tensor(cropped),
            mean=list(self.normalize_mean),
            std=list(self.normalize_std),
        )
        if guide is None:
            return tensor, None
        cropped_guide = TF.resized_crop(
            guide,
            top,
            left,
            height,
            width,
            [self.crop_size, self.crop_size],
            self._guide_interpolation,
        )
        return tensor, cropped_guide


def make_paired_transforms(
    crop_size=256,                          # type: int
    crop_scale=(0.3, 1.0),                  # type: Tuple[float, float]
    gaussian_blur=False,                    # type: bool
    horizontal_flip=False,                  # type: bool
    color_distortion=False,                 # type: bool
    color_jitter=0.0,                       # type: float
    normalize_mean=IMAGENET_MEAN,           # type: Tuple[float, ...]
    normalize_std=IMAGENET_STD,             # type: Tuple[float, ...]
):
    # type: (...) -> PairedRandomResizedCrop
    """Guide-aware counterpart of :func:`make_transforms`.

    Only the augmentations used by the OCT pretraining configs (a random
    resized crop plus normalisation) are supported.  Anything that would move
    or recolour the image without a matching guide operation is rejected
    loudly rather than silently desynchronising the pair.
    """
    unsupported = []  # type: List[str]
    if gaussian_blur:
        unsupported.append("gaussian_blur")
    if horizontal_flip:
        unsupported.append("horizontal_flip")
    if color_distortion and color_jitter > 0.0:
        unsupported.append("color_distortion")
    if unsupported:
        raise ValueError(
            "make_paired_transforms does not support %s; the guide would "
            "desynchronise from the image." % ", ".join(unsupported)
        )

    return PairedRandomResizedCrop(
        crop_size=crop_size,
        crop_scale=crop_scale,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
