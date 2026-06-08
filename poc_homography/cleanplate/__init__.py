"""
Offline clean-plate (empty-floor) reconstruction pipeline.

This package fuses many mask-aware frames into one common ground-plane raster
and reduces per-cell to an empty-floor orthophoto. Public API:

    - :class:`GroundRaster`: common ground-plane raster grid.
    - :func:`ortho_rectify_frame` / :class:`OrthoResult`: per-frame warping.
    - :class:`CellAccumulator`: memory-bounded per-cell color accumulation.
    - photometric helpers: :func:`normalize_by_gain`, :func:`normalize_median_gray`.
    - :class:`CleanPlateFrame`, :class:`CleanPlateResult`,
      :func:`reconstruct_clean_plate`, :func:`write_clean_plate`: orchestration.
    - :func:`make_synthetic_visits` / :func:`make_clean_background`: synthetic data.
    - :class:`CleanPlateDataset`: survey-run loader producing clean-plate frames.
"""

from poc_homography.cleanplate.accumulate import CellAccumulator
from poc_homography.cleanplate.dataset import CleanPlateDataset
from poc_homography.cleanplate.ortho import OrthoResult, ortho_rectify_frame
from poc_homography.cleanplate.photometric import (
    normalize_by_gain,
    normalize_median_gray,
)
from poc_homography.cleanplate.raster import GroundRaster
from poc_homography.cleanplate.reconstruct import (
    CleanPlateFrame,
    CleanPlateResult,
    reconstruct_clean_plate,
    write_clean_plate,
)
from poc_homography.cleanplate.synthetic import (
    make_clean_background,
    make_synthetic_visits,
)

__all__ = [
    "GroundRaster",
    "OrthoResult",
    "ortho_rectify_frame",
    "CellAccumulator",
    "normalize_by_gain",
    "normalize_median_gray",
    "CleanPlateFrame",
    "CleanPlateResult",
    "reconstruct_clean_plate",
    "write_clean_plate",
    "make_clean_background",
    "make_synthetic_visits",
    "CleanPlateDataset",
]
