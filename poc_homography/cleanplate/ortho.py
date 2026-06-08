"""
Ortho-rectify a single mask-aware frame into a common ground raster.

Given a frame image, its ground homography ``H`` (world meters -> image pixels)
and a target :class:`GroundRaster`, this module warps the frame (and an optional
floor mask) into raster space using the composed transform::

    H_image_to_raster = raster.world_to_raster_matrix() @ inv(H)

Near-singular homographies are detected and skipped gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from poc_homography.cleanplate.raster import GroundRaster

# Homographies with absolute determinant below this are treated as singular.
MIN_HOMOGRAPHY_DET = 1e-12


@dataclass(frozen=True)
class OrthoResult:
    """
    Result of ortho-rectifying one frame into the raster.

    Attributes:
        color: Warped color raster, ``(H, W, 3)`` uint8 (zeros where invalid).
        valid: Boolean validity raster, ``(H, W)``; True only where the warped
            pixel came from inside the source image AND was floor-labeled.
    """

    color: np.ndarray
    valid: np.ndarray


def ortho_rectify_frame(
    image: np.ndarray,
    ground_homography: np.ndarray,
    raster: GroundRaster,
    floor_mask: np.ndarray | None = None,
) -> OrthoResult | None:
    """
    Warp one frame into the common ground raster.

    Args:
        image: Source frame, ``(H, W, 3)`` uint8.
        ground_homography: ``(3, 3)`` matrix mapping world ground meters
            ``[x, y, 1]`` to image pixels ``[u, v, 1]``.
        raster: Target :class:`GroundRaster` defining the output grid.
        floor_mask: Optional ``(H, W)`` mask where True / 255 marks empty floor
            and False / 0 marks transient occluders. If None, all in-image
            pixels are considered floor.

    Returns:
        An :class:`OrthoResult`, or ``None`` if the homography is near-singular
        (frame skipped).
    """
    homography = np.asarray(ground_homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError(f"ground_homography must be 3x3, got {homography.shape}")

    if abs(float(np.linalg.det(homography))) < MIN_HOMOGRAPHY_DET:
        return None

    try:
        h_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None

    h_image_to_raster = raster.world_to_raster_matrix() @ h_inv

    out_h, out_w = raster.shape

    color = cv2.warpPerspective(
        image,
        h_image_to_raster,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if floor_mask is not None:
        # The warped mask (nearest, border 0) ALREADY encodes "inside the image
        # footprint AND floor": pixels mapping outside the source image get the
        # border value 0, and non-floor pixels are 0. So validity is the warped
        # mask alone — no separate all-ones footprint warp is needed.
        mask_u8 = (np.asarray(floor_mask) > 0).astype(np.uint8)
        warped_mask = cv2.warpPerspective(
            mask_u8,
            h_image_to_raster,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        valid = warped_mask > 0
    else:
        # No mask: warp an all-ones source plane to find raster cells that map
        # back inside the original image bounds (the in-image footprint).
        src_ones = np.ones(image.shape[:2], dtype=np.uint8)
        coverage = cv2.warpPerspective(
            src_ones,
            h_image_to_raster,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        valid = coverage > 0

    return OrthoResult(color=color, valid=valid)
