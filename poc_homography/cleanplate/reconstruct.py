"""
Orchestrator for offline clean-plate (empty-floor) reconstruction.

This module fuses an iterable of mask-aware :class:`CleanPlateFrame` objects into
a single empty-floor orthophoto on a common :class:`GroundRaster`. The pipeline
per frame is:

    1. optional photometric normalization (toward a common exposure reference),
    2. ortho-rectification into raster space (mask-aware),
    3. accumulation of floor-only color samples per raster cell.

After all frames are processed, the accumulator is reduced (median/mode) and any
cell that was NEVER observed empty (coverage == 0) is filled by inpainting so the
output has no holes. Helpers are provided to persist the orthophoto and coverage
to disk (PNG / TIFF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import tifffile

from poc_homography.cleanplate.accumulate import (
    DEFAULT_MAX_SAMPLES_PER_CELL,
    CellAccumulator,
)
from poc_homography.cleanplate.ortho import ortho_rectify_frame
from poc_homography.cleanplate.photometric import normalize_median_gray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from poc_homography.cleanplate.raster import GroundRaster
    from poc_homography.types import Unitless

DEFAULT_INPAINT_RADIUS = 3


@dataclass(frozen=True)
class CleanPlateFrame:
    """
    One mask-aware capture used as input to clean-plate reconstruction.

    Attributes:
        image: Source frame, ``(H, W, 3)`` uint8.
        floor_mask: Optional ``(H, W)`` mask where True / 255 marks empty floor
            and False / 0 marks transient occluders. If None, all in-image
            pixels are treated as floor.
        ground_homography: ``(3, 3)`` matrix mapping world ground meters
            ``[x, y, 1]`` to image pixels ``[u, v, 1]``.
        gain: Optional capture gain for photometric normalization.
        time_bucket: Optional coarse time-of-day label (metadata; informational).
    """

    image: np.ndarray
    floor_mask: np.ndarray | None
    ground_homography: np.ndarray
    gain: Unitless | None = None
    time_bucket: str | None = None


@dataclass(frozen=True)
class CleanPlateResult:
    """
    Result of a clean-plate reconstruction over a :class:`GroundRaster`.

    Attributes:
        orthophoto: Empty-floor orthophoto, ``(H, W, 3)`` uint8, hole-free.
        coverage: Per-cell count of floor observations, ``(H, W)`` int32.
        inpainted_mask: ``(H, W)`` bool; True where a cell had coverage == 0 and
            was filled by inpainting.
        raster: The :class:`GroundRaster` the result is defined on.
    """

    orthophoto: np.ndarray
    coverage: np.ndarray
    inpainted_mask: np.ndarray
    raster: GroundRaster = field(repr=False)


def reconstruct_clean_plate(
    frames: Iterable[CleanPlateFrame],
    raster: GroundRaster,
    *,
    method: str = "median",
    photometric: bool = True,
    target_gray: float = 128.0,
    max_samples_per_cell: int = DEFAULT_MAX_SAMPLES_PER_CELL,
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS,
) -> CleanPlateResult:
    """
    Fuse mask-aware frames into a single empty-floor orthophoto.

    Args:
        frames: Iterable of :class:`CleanPlateFrame` captures.
        raster: Common :class:`GroundRaster` to reconstruct onto.
        method: Per-cell reduction, ``"median"`` or ``"mode"``.
        photometric: If True, median-gray normalize each frame (floor-masked)
            toward ``target_gray`` before accumulation.
        target_gray: Target median gray level for photometric normalization.
        max_samples_per_cell: Reservoir cap per raster cell.
        inpaint_radius: Telea inpainting radius for never-empty cells.

    Returns:
        A :class:`CleanPlateResult` with a hole-free orthophoto, coverage and
        the inpainted-cell mask.
    """
    accumulator = CellAccumulator(raster.shape, max_samples_per_cell=max_samples_per_cell)

    for frame in frames:
        image = frame.image
        if photometric:
            image = normalize_median_gray(image, target_gray=target_gray, mask=frame.floor_mask)

        ortho = ortho_rectify_frame(
            image, frame.ground_homography, raster, floor_mask=frame.floor_mask
        )
        if ortho is None:
            continue
        accumulator.add_frame(ortho.color, ortho.valid)

    orthophoto, coverage = accumulator.reduce(method=method)

    never_empty = coverage == 0
    inpainted_mask = np.zeros(raster.shape, dtype=bool)
    if never_empty.any() and (~never_empty).any():
        orthophoto = cv2.inpaint(
            orthophoto,
            never_empty.astype(np.uint8),
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )
        inpainted_mask = never_empty.copy()

    return CleanPlateResult(
        orthophoto=orthophoto,
        coverage=coverage,
        inpainted_mask=inpainted_mask,
        raster=raster,
    )


def write_clean_plate(
    result: CleanPlateResult,
    orthophoto_path: str | Path,
    coverage_path: str | Path | None = None,
) -> None:
    """
    Write a clean-plate result's orthophoto and coverage to disk.

    The orthophoto is written by extension: ``.png`` / ``.jpg`` / ``.jpeg`` via
    OpenCV, or ``.tif`` / ``.tiff`` via tifffile. Coverage (if a path is given)
    is written as a single-channel TIFF (``.tif`` / ``.tiff``) preserving the
    int32 counts.

    Args:
        result: The :class:`CleanPlateResult` to persist.
        orthophoto_path: Output path for the RGB orthophoto (PNG/JPEG/TIFF).
        coverage_path: Optional output path for the coverage raster (TIFF).

    Raises:
        ValueError: If a path has an unsupported extension.
    """
    ortho_path = Path(orthophoto_path)
    suffix = ortho_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        tifffile.imwrite(str(ortho_path), result.orthophoto)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        # OpenCV expects BGR for color writes.
        bgr = cv2.cvtColor(result.orthophoto, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(ortho_path), bgr):
            raise ValueError(f"Failed to write orthophoto to {ortho_path}")
    else:
        raise ValueError(f"Unsupported orthophoto extension: {suffix!r}")

    if coverage_path is not None:
        cov_path = Path(coverage_path)
        if cov_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Coverage must be a TIFF (.tif/.tiff), got {cov_path.suffix!r}")
        tifffile.imwrite(str(cov_path), result.coverage.astype(np.int32))
