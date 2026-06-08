"""Shared fixtures for clean-plate reconstruction tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from poc_homography.cleanplate import (
    CleanPlateFrame,
    GroundRaster,
    make_synthetic_visits,
)
from poc_homography.types import Meters, Unitless

if TYPE_CHECKING:
    import numpy as np


@pytest.fixture
def small_raster() -> GroundRaster:
    """A small 96x96 ground raster (6m x 6m at 16 ppm) for fast tests."""
    return GroundRaster(
        x_min=Meters(0.0),
        x_max=Meters(6.0),
        y_min=Meters(0.0),
        y_max=Meters(6.0),
        pixels_per_meter=Unitless(16.0),
    )


@pytest.fixture
def synthetic_visits(
    small_raster: GroundRaster,
) -> tuple[np.ndarray, list[CleanPlateFrame]]:
    """Deterministic synthetic background + visit frames on ``small_raster``."""
    return make_synthetic_visits(n_visits=4, raster=small_raster, seed=7)
