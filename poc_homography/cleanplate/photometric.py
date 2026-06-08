"""
Basic per-visit photometric normalization for clean-plate reconstruction.

Frames captured across visits differ in exposure (gain, shutter) and ambient
light. Before fusing them, this module brings each frame toward a common
brightness reference so that median/mode reduction is not biased by exposure.

Two simple, pure strategies are provided:
    - Gain-based linear scaling toward a reference gain.
    - Per-frame median-gray normalization toward a target gray level.

All functions are pure (no global state); they return new arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poc_homography.types import Unitless

DEFAULT_TARGET_GRAY = 128.0


def normalize_by_gain(
    image: np.ndarray,
    gain: Unitless,
    reference_gain: Unitless,
) -> np.ndarray:
    """
    Linearly scale an image to compensate for sensor gain differences.

    Scales pixel values by ``reference_gain / gain`` so frames captured at
    different gains land on a common radiometric scale.

    Args:
        image: Source frame, ``(H, W, 3)`` uint8.
        gain: The frame's capture gain (> 0).
        reference_gain: Target reference gain (> 0).

    Returns:
        A new ``(H, W, 3)`` uint8 image scaled toward the reference gain.
    """
    if gain <= 0 or reference_gain <= 0:
        raise ValueError("gain and reference_gain must be positive")
    scale = float(reference_gain) / float(gain)
    scaled = np.asarray(image, dtype=np.float32) * scale
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def normalize_median_gray(
    image: np.ndarray,
    target_gray: float = DEFAULT_TARGET_GRAY,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Scale an image so its (masked) median gray equals ``target_gray``.

    A robust, exposure-agnostic normalization: computes the median luminance
    over valid pixels and rescales the frame to a common target level.

    Args:
        image: Source frame, ``(H, W, 3)`` uint8.
        target_gray: Desired median gray level in ``[0, 255]``.
        mask: Optional ``(H, W)`` mask; only True / non-zero pixels contribute
            to the median estimate (e.g. floor-only). If None, all pixels used.

    Returns:
        A new ``(H, W, 3)`` uint8 image normalized toward ``target_gray``.
    """
    img = np.asarray(image, dtype=np.float32)
    gray = img.mean(axis=2)
    if mask is not None:
        sel = np.asarray(mask) > 0
        values = gray[sel] if sel.any() else gray.reshape(-1)
    else:
        values = gray.reshape(-1)
    median = float(np.median(values))
    if median <= 1e-6:
        return np.asarray(image, dtype=np.uint8).copy()
    scale = target_gray / median
    scaled = img * scale
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
