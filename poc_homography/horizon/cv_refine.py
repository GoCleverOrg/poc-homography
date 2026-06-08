"""Classical-CV horizon refinement and a coarse JPEG-size sky proxy.

The ground is texture-rich (vegetation, structures, terrain) while the sky is
near-uniform. The horizon therefore shows up as a *collapse* in per-row
edge energy when scanning from the textured ground (lower rows) up into the
smooth sky (upper rows). :func:`refine_horizon_cv` locates that transition.

These are PoC-grade heuristics tuned on the ``icozee-camptz-04`` calibration
frames (8-bit Sobel magnitudes). They *refine/validate* the geometric
prediction; the geometric path remains the workhorse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from poc_homography.horizon.models import FramePlacement, HorizonEstimate

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --- Tunable detector thresholds (raw 8-bit Sobel-magnitude units) ----------
DEFAULT_IN_FRAME_CONTRAST = 0.65
"""Minimum ``(hi-lo)/hi`` row-energy contrast ratio to trust an in-frame split.

In-frame horizons on the calibration set score ``0.79..0.98`` (a low-energy sky
band above a high-energy ground band); uniform frames score ``<0.56``.
"""

DEFAULT_LOW_TEXTURE_ENERGY = 40.0
"""Below this 85th-percentile row energy a uniform frame is treated as all-sky.

All-sky frames score ``~13``; all-ground frames score ``>115``.
"""


def _row_energy(gray: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-row mean Sobel gradient magnitude (edge energy)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude: NDArray[np.float64] = np.sqrt(gx * gx + gy * gy)
    return magnitude.mean(axis=1)


def _smooth(profile: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Moving-average smoothing of a 1-D profile."""
    window = max(3, window)
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode="same")


def _to_gray(image: np.ndarray) -> NDArray[np.float64]:
    """Coerce a BGR or grayscale array to a float64 grayscale array."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64)


def refine_horizon_cv(
    image: np.ndarray,
    *,
    in_frame_contrast: float = DEFAULT_IN_FRAME_CONTRAST,
    low_texture_energy: float = DEFAULT_LOW_TEXTURE_ENERGY,
) -> HorizonEstimate:
    """Estimate the horizon row from an image via edge-energy collapse.

    Args:
        image: BGR or grayscale frame as a numpy array.
        in_frame_contrast: Minimum row-energy contrast ratio to report a
            confident in-frame horizon.
        low_texture_energy: 85th-percentile energy below which a low-contrast
            frame is classified all-sky (rather than all-ground).

    Returns:
        A :class:`HorizonEstimate` with ``method="cv"``. The split row that
        best separates a low-energy band (above) from a high-energy band
        (below) is reported when the contrast is strong enough; otherwise the
        frame is classified all-sky or all-ground from its texture level.
    """
    gray = _to_gray(image)
    height = gray.shape[0]
    energy = _smooth(_row_energy(gray), height // 100)

    lo = float(np.percentile(energy, 15))
    hi = float(np.percentile(energy, 85))
    contrast_ratio = (hi - lo) / hi if hi > 0.0 else 0.0

    if contrast_ratio >= in_frame_contrast:
        row = _best_split_row(energy)
        return HorizonEstimate(
            placement=FramePlacement.IN_FRAME,
            image_height=height,
            row=float(row),
            ground_fraction=(height - row) / height,
            method="cv",
            confidence=contrast_ratio,
        )

    # Uniform frame: no sky/ground boundary. Classify by texture level.
    if hi < low_texture_energy:
        return HorizonEstimate(
            placement=FramePlacement.BELOW_FRAME,
            image_height=height,
            row=None,
            ground_fraction=0.0,
            method="cv",
            confidence=1.0 - contrast_ratio,
        )
    return HorizonEstimate(
        placement=FramePlacement.ABOVE_FRAME,
        image_height=height,
        row=None,
        ground_fraction=1.0,
        method="cv",
        confidence=1.0 - contrast_ratio,
    )


def _best_split_row(energy: NDArray[np.float64]) -> int:
    """Row maximising ``mean(below) - mean(above)`` of the energy profile.

    A 5%-of-height margin is excluded at each edge so the optimum is a genuine
    interior transition, not a degenerate boundary pick.
    """
    n = len(energy)
    margin = max(1, n // 20)
    cumulative = np.cumsum(energy)
    total = float(cumulative[-1])
    best_row = margin
    best_delta = -np.inf
    for row in range(margin, n - margin):
        mean_above = cumulative[row - 1] / row
        mean_below = (total - cumulative[row - 1]) / (n - row)
        delta = mean_below - mean_above
        if delta > best_delta:
            best_delta = delta
            best_row = row
    return best_row


def estimate_sky_fraction_from_jpeg_size(
    num_bytes: int,
    image_width: int,
    image_height: int,
    *,
    full_ground_bytes_per_pixel: float = 0.150,
    full_sky_bytes_per_pixel: float = 0.035,
) -> float:
    """Coarsely estimate the sky fraction from a JPEG's compressed size.

    Sky is low-texture and highly compressible, so JPEG file size collapses as
    sky fills the frame (observed ``590→433→278→139 KB`` on the calibration
    sweep). This maps bytes-per-pixel linearly between two anchor densities to a
    sky fraction in ``[0, 1]``.

    This is a *very coarse* proxy — a cheap pre-filter, never a substitute for
    the geometric prediction or the CV detector.

    Args:
        num_bytes: Encoded JPEG size in bytes.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        full_ground_bytes_per_pixel: Density anchor for an all-ground frame.
        full_sky_bytes_per_pixel: Density anchor for an all-sky frame.

    Returns:
        Estimated sky fraction in ``[0.0, 1.0]`` (higher = more sky).
    """
    bytes_per_pixel = num_bytes / float(image_width * image_height)
    span = full_ground_bytes_per_pixel - full_sky_bytes_per_pixel
    if span <= 0.0:
        raise ValueError("full_ground_bytes_per_pixel must exceed full_sky_bytes_per_pixel")
    sky_fraction = (full_ground_bytes_per_pixel - bytes_per_pixel) / span
    return float(min(1.0, max(0.0, sky_fraction)))
