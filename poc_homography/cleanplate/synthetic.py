"""
Synthetic multi-visit generator for clean-plate tests and prototyping.

:func:`make_synthetic_visits` builds a deterministic empty-floor background on a
:class:`GroundRaster`, then renders several "visits" of that floor seen through
mild perspective ground homographies. Each visit composites transient "mover"
rectangles on top of the floor; the per-frame floor mask marks mover pixels as
NOT-floor (0) and floor elsewhere (255).

Two test-critical properties are guaranteed by construction:

    1. Every *recoverable* raster cell is seen empty (floor) in at least one
       visit, so median-over-floor recovers the clean background there.
    2. At least one raster cell is covered by a mover in EVERY visit (never seen
       empty), exercising the inpainting fallback in reconstruction.

Determinism: randomness is confined to a local ``np.random.default_rng(seed)``;
the global numpy RNG is never touched.
"""

from __future__ import annotations

import cv2
import numpy as np

from poc_homography.cleanplate.raster import GroundRaster
from poc_homography.cleanplate.reconstruct import CleanPlateFrame
from poc_homography.types import Meters, Unitless

# Cell that is always covered by a mover across all visits (row, col).
ALWAYS_COVERED_CELL = (8, 8)


def make_clean_background(raster: GroundRaster, *, seed: int = 0) -> np.ndarray:
    """
    Build a deterministic empty-floor background raster.

    The texture is a smooth low-frequency color gradient plus a faint
    deterministic checkerboard, giving spatially varying content without RNG
    that would break determinism across runs.

    Args:
        raster: The :class:`GroundRaster` defining the output grid.
        seed: Seed for the local RNG used for faint per-cell dither.

    Returns:
        A ``(H, W, 3)`` uint8 clean-floor background.
    """
    height, width = raster.shape
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 80.0 + 120.0 * (xx / max(width - 1, 1))
    g = 80.0 + 120.0 * (yy / max(height - 1, 1))
    b = 120.0 + 60.0 * np.sin(xx / 6.0) * np.cos(yy / 6.0)

    checker = (((xx.astype(int) // 4) + (yy.astype(int) // 4)) % 2).astype(np.float32)
    r = r + 15.0 * checker
    g = g + 10.0 * checker
    b = b + 15.0 * checker

    dither = rng.integers(-3, 4, size=(height, width, 1)).astype(np.float32)
    bg = np.stack([r, g, b], axis=2) + dither
    return np.clip(np.rint(bg), 0, 255).astype(np.uint8)


def _make_pose_homography(
    raster: GroundRaster,
    image_size: tuple[int, int],
    skew: float,
) -> np.ndarray:
    """
    Build a mild perspective ground homography (world meters -> image pixels).

    The homography is constructed from a base scale/translation (world extent ->
    image) perturbed by a small projective ``skew`` so that successive poses see
    the floor from slightly different vantage points.

    Args:
        raster: The :class:`GroundRaster` defining the world extent.
        image_size: ``(img_h, img_w)`` of the rendered frame.
        skew: Small projective coefficient (e.g. in ``[-0.02, 0.02]``).

    Returns:
        A ``(3, 3)`` world->image homography.
    """
    img_h, img_w = image_size
    # World corners (meters), CCW from bottom-left.
    world = np.array(
        [
            [raster.x_min, raster.y_min],
            [raster.x_max, raster.y_min],
            [raster.x_max, raster.y_max],
            [raster.x_min, raster.y_max],
        ],
        dtype=np.float32,
    )
    margin_x = img_w * 0.08
    margin_y = img_h * 0.08
    # Base image corners with a perspective tilt applied to the top edge.
    dst = np.array(
        [
            [margin_x, img_h - margin_y],
            [img_w - margin_x, img_h - margin_y],
            [img_w - margin_x - skew * img_w, margin_y + skew * img_h],
            [margin_x + skew * img_w, margin_y + skew * img_h],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(world, dst)
    return np.asarray(homography, dtype=np.float64)


def make_synthetic_visits(
    *,
    n_visits: int = 4,
    raster: GroundRaster | None = None,
    image_size: tuple[int, int] = (96, 96),
    seed: int = 0,
) -> tuple[np.ndarray, list[CleanPlateFrame]]:
    """
    Generate synthetic multi-visit frames over a known clean background.

    Args:
        n_visits: Number of visits/poses to render (>= 2 recommended).
        raster: Target :class:`GroundRaster`; a small default is used if None.
        image_size: ``(img_h, img_w)`` of each rendered frame.
        seed: Seed for all local RNG (deterministic; global RNG untouched).

    Returns:
        Tuple ``(clean_background, frames)`` where ``clean_background`` is the
        ground-truth ``(H, W, 3)`` uint8 raster for assertions and ``frames`` is
        a list of :class:`CleanPlateFrame`.
    """
    if raster is None:
        raster = GroundRaster(
            x_min=Meters(0.0),
            x_max=Meters(6.0),
            y_min=Meters(0.0),
            y_max=Meters(6.0),
            pixels_per_meter=Unitless(16.0),
        )
    height, width = raster.shape
    img_h, img_w = image_size

    background = make_clean_background(raster, seed=seed)
    world_to_raster = raster.world_to_raster_matrix()
    raster_to_world = np.linalg.inv(world_to_raster)

    rng = np.random.default_rng(seed + 1)
    skews = np.linspace(-0.015, 0.015, n_visits)

    # Mover footprints (in raster cells) per visit, so that union over visits
    # leaves every recoverable cell empty in >= 1 visit, while ALWAYS_COVERED
    # stays occluded in all visits.
    mover_size = max(4, height // 8)
    ar, ac = ALWAYS_COVERED_CELL

    frames: list[CleanPlateFrame] = []
    for v in range(n_visits):
        # Raster-space floor mask (255 = floor, 0 = mover) for this visit.
        floor_raster = np.full((height, width), 255, dtype=np.uint8)

        # Roving mover: sweeps across the grid between visits.
        rx = int((v / max(n_visits - 1, 1)) * (width - mover_size))
        ry = int(((n_visits - 1 - v) / max(n_visits - 1, 1)) * (height - mover_size))
        floor_raster[ry : ry + mover_size, rx : rx + mover_size] = 0

        # A second random mover for variety, avoiding ALWAYS_COVERED.
        jx = int(rng.integers(0, max(width - mover_size, 1)))
        jy = int(rng.integers(0, max(height - mover_size, 1)))
        if not (jy <= ar < jy + mover_size and jx <= ac < jx + mover_size):
            floor_raster[jy : jy + mover_size, jx : jx + mover_size] = 0

        # The always-covered mover: a small block centered on ALWAYS_COVERED.
        half = 2
        floor_raster[max(ar - half, 0) : ar + half + 1, max(ac - half, 0) : ac + half + 1] = 0

        homography = _make_pose_homography(raster, image_size, float(skews[v]))
        image_from_raster = homography @ raster_to_world

        # Render clean floor into image space.
        image = cv2.warpPerspective(
            background,
            image_from_raster,
            (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Warp the floor mask too (255 = floor) into image space.
        mask_image = cv2.warpPerspective(
            floor_raster,
            image_from_raster,
            (img_w, img_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )

        # Composite a solid mover color where the floor mask says NOT-floor,
        # but only inside the image footprint of the floor patch.
        footprint = cv2.warpPerspective(
            np.full((height, width), 255, dtype=np.uint8),
            image_from_raster,
            (img_w, img_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        mover_pixels = (footprint > 0) & (mask_image == 0)
        mover_color = np.array([200, 30, 30], dtype=np.uint8)
        image[mover_pixels] = mover_color

        floor_mask = np.where(mover_pixels, 0, 255).astype(np.uint8)
        floor_mask[footprint == 0] = 0

        frames.append(
            CleanPlateFrame(
                image=image,
                floor_mask=floor_mask,
                ground_homography=homography,
                gain=Unitless(1.0),
                time_bucket=f"visit_{v}",
            )
        )

    return background, frames
