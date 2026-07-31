"""Independent tissue-truth references for scoring masking policies.

Scoring a masking policy against the segmentation that placed it is circular,
so the reference must come from the image itself.  The first attempt
thresholded the **patch mean** with Otsu, which under-counts tissue twice over:

1.  A patch straddling the retinal boundary is half bright tissue and half
    vitreous, so its mean sits below the threshold and the whole patch is
    scored as background.  Every edge patch of the retina is lost.
2.  Otsu on a strongly bimodal histogram lands between "bright RNFL/RPE" and
    "dark vitreous", so genuinely dim tissue -- choroid, and anything under a
    vessel shadow -- falls on the background side.

Both are fixed by segmenting at the **pixel** level and only then measuring how
much of each patch is covered.

Definitions provided, cheapest first:

``patchmean_otsu``
    The original.  Kept so the bias can be quantified rather than argued about.
``pixel_otsu``
    Otsu over pixels, then patch occupancy.  Fixes (1) but not (2).
``noise_band``
    Threshold calibrated against the vitreous noise floor, then a per-column
    retinal band.  Fixes (1) and (2), still MIRAGE-free.
``noise_band_union_mirage``
    ``noise_band`` unioned with the raw MIRAGE union (no dilation, no envelope
    repair).  Partially circular and labelled as such, but it is the only
    reference that recovers tissue too dim for any intensity rule.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "otsu_threshold",
    "patch_coverage",
    "truth_patchmean_otsu",
    "truth_pixel_otsu",
    "truth_noise_band",
    "truth_union_mirage",
    "TRUTH_MODES",
    "DEFAULT_K",
    "build_truth",
]

# Sigma multiplier above the vitreous noise floor.  Calibrated on 40 volumes by
# sweeping k against recall of the MIRAGE envelope -- a trained segmenter, so
# its confident region is genuinely retina:
#
#     k     envelope recall     image marked
#     0.5        0.988              0.528
#     1.0        0.984              0.436
#     1.5        0.973              0.379
#     2.0        0.956              0.339
#     3.0        0.905              0.284
#
# Recall holds above 0.97 to k=1.5 while the marked area keeps falling, then
# degrades sharply.  1.5 is that elbow: it keeps 97% of what the segmenter
# calls retina for the smallest committed area.
DEFAULT_K = 1.5


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Classic Otsu threshold over a 1-D sample."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * centres)
    total_mean = cum_mean[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_bg = cum_mean / weight_bg
        mu_fg = (total_mean - cum_mean) / weight_fg
        between = weight_bg * weight_fg * (mu_bg - mu_fg) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centres[int(np.argmax(between))])


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable mean filter; suppresses OCT speckle before thresholding."""
    if radius <= 0:
        return image.astype(np.float32)
    size = 2 * radius + 1
    padded = np.pad(image.astype(np.float32), radius, mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    rows = np.empty_like(padded)
    rows[:size] = cumulative[:size]
    rows[size:] = cumulative[size:] - cumulative[:-size]
    rows = rows[size - 1:] / size
    cumulative = np.cumsum(rows, axis=1)
    cols = np.empty_like(rows)
    cols[:, :size] = cumulative[:, :size]
    cols[:, size:] = cumulative[:, size:] - cumulative[:, :-size]
    return cols[:, size - 1:] / size


def _column_band(mask: np.ndarray, bridge: int, min_run: int) -> np.ndarray:
    """Keep the longest vertical run per column after bridging short gaps.

    The retina is one horizontal band, so within a column the tissue is a
    single vertical run.  Speckle punches holes in it, which ``bridge``
    closes.  Taking the longest run afterwards -- rather than spanning first to
    last -- stops a bright artefact near the image floor from dragging the band
    down through the choroidal dark space.
    """
    height, width = mask.shape
    out = np.zeros_like(mask)
    for column in range(width):
        col = mask[:, column]
        if not col.any():
            continue
        rows = np.flatnonzero(col)
        splits = np.flatnonzero(np.diff(rows) > bridge + 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [rows.size - 1]))
        lengths = rows[ends] - rows[starts] + 1
        best = int(np.argmax(lengths))
        if lengths[best] < min_run:
            continue
        out[rows[starts[best]]:rows[ends[best]] + 1, column] = True
    return out


def patch_coverage(pixel_mask: np.ndarray, patch: int = 16) -> np.ndarray:
    """Fraction of each patch covered by a pixel-level mask."""
    grid = pixel_mask.shape[0] // patch
    return (
        pixel_mask.astype(np.float32)
        .reshape(grid, patch, grid, patch)
        .mean(axis=(1, 3))
    )


def truth_patchmean_otsu(image: np.ndarray, patch: int = 16, **_) -> np.ndarray:
    """Original reference: Otsu over patch means.  Under-counts tissue."""
    grid = image.shape[0] // patch
    means = (
        image.astype(np.float32)
        .reshape(grid, patch, grid, patch)
        .mean(axis=(1, 3))
    )
    return means >= otsu_threshold(means, bins=64)


def truth_pixel_otsu(
    image: np.ndarray, patch: int = 16, coverage: float = 0.5, **_
) -> np.ndarray:
    """Otsu at pixel level, then patch occupancy.  Recovers edge patches."""
    pixels = _box_blur(image, 1) >= otsu_threshold(image)
    return patch_coverage(pixels, patch) >= coverage


def _noise_floor(image: np.ndarray, bins: int = 64) -> tuple[float, float]:
    """Locate the vitreous noise peak and estimate its spread.

    The background is the most common intensity in an OCT B-scan, so a coarse
    histogram mode gives its centre.  Its width cannot be measured from the
    right flank, which tissue contaminates, nor from a low percentile, which is
    a *truncated* sample whose standard deviation understates the true spread
    by roughly half and drags the threshold down to the median.

    The left flank is uncontaminated -- tissue is brighter -- and for a
    Gaussian the second moment of either half equals the full variance,
    ``E[(X-mu)^2 | X < mu] = sigma^2``, so the spread is recovered exactly from
    the darker half alone.  Half-width at half-maximum was tried first and is
    unusable here: over a narrow intensity range the histogram is spiky enough
    that the walk terminates on the bin next to the peak, yielding sigma ~ 0.06
    against a true 2.02.
    """
    hist, edges = np.histogram(image, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    centre = float(centres[int(np.argmax(hist))])

    darker = image[image <= centre]
    if darker.size < 32:
        return centre, float(image.std())
    sigma = float(np.sqrt(np.mean((darker - centre) ** 2)))
    return centre, max(sigma, float(edges[1] - edges[0]))


def tissue_pixels_noise_band(
    image: np.ndarray,
    k: float = DEFAULT_K,
    blur: int = 1,
    bridge: int = 8,
    min_run: int = 6,
    **_,
) -> np.ndarray:
    """Pixel-level tissue via a vitreous-noise-calibrated threshold.

    Crops taken near a volume edge carry a strip of exact-zero padding.  That
    strip is a single spike in the histogram and can outvote the true vitreous
    peak -- on one measured slice it held 6,417 pixels against the vitreous
    bin's 5,680, pulling the mode from ~25 down to 1.3 and passing 90% of the
    image as tissue.  Real vitreous is noisy and effectively never lands on
    exact zero, so padding is excluded from both the estimate and the result.
    """
    valid = image > 0
    if valid.mean() < 0.25:          # degenerate crop: trust the whole frame
        valid = np.ones_like(valid)
    smooth = _box_blur(image, blur)
    centre, sigma = _noise_floor(smooth[valid])
    tissue = (smooth >= centre + k * sigma) & valid
    return _column_band(tissue, bridge=bridge, min_run=min_run)


def truth_noise_band(
    image: np.ndarray, patch: int = 16, coverage: float = 0.5, **kwargs
) -> np.ndarray:
    """MIRAGE-free reference that keeps dim tissue.  Recommended."""
    pixels = tissue_pixels_noise_band(
        image,
        k=kwargs.get("k", DEFAULT_K),
        blur=kwargs.get("blur", 1),
        bridge=kwargs.get("bridge", 8),
        min_run=kwargs.get("min_run", 6),
    )
    return patch_coverage(pixels, patch) >= coverage


def truth_union_mirage(
    image: np.ndarray,
    patch: int = 16,
    coverage: float = 0.5,
    mirage_raw: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """``noise_band`` unioned with the raw MIRAGE union (no dilation).

    Partially circular: any patch MIRAGE claims counts as tissue by
    construction, which flatters MIRAGE-guided samplers.  Reported alongside
    the MIRAGE-free references, never on its own.
    """
    pixels = tissue_pixels_noise_band(
        image,
        k=kwargs.get("k", DEFAULT_K),
        blur=kwargs.get("blur", 1),
        bridge=kwargs.get("bridge", 8),
        min_run=kwargs.get("min_run", 6),
    )
    if mirage_raw is not None:
        pixels = pixels | mirage_raw.astype(bool)
    return patch_coverage(pixels, patch) >= coverage


TRUTH_MODES = {
    "patchmean_otsu": truth_patchmean_otsu,
    "pixel_otsu": truth_pixel_otsu,
    "noise_band": truth_noise_band,
    "union_mirage": truth_union_mirage,
}


def build_truth(mode: str, image: np.ndarray, **kwargs) -> np.ndarray:
    """Dispatch to a named truth definition."""
    try:
        return TRUTH_MODES[mode](image, **kwargs)
    except KeyError:
        raise ValueError(
            f"unknown truth mode {mode!r}; expected one of {sorted(TRUTH_MODES)}"
        ) from None
