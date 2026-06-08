"""
Common ground-plane raster grid abstraction for clean-plate reconstruction.

This module defines :class:`GroundRaster`, a fixed world-extent grid onto which
many mask-aware frames are ortho-rectified and fused. The raster maps a planar
world ground patch (in meters, X=East, Y=North) to an integer pixel grid.

Coordinate Systems:
    - World Frame: X=East, Y=North (meters), ground plane Z=0.
    - Raster Frame: origin top-left, ``col`` increases with world-X (East),
      ``row`` increases as world-Y (North) DECREASES (y-axis flip), mirroring
      the ``world_to_map_pixels`` convention in
      ``poc_homography.homography.intrinsic_extrinsic``.

The forward (world -> raster) affine is::

    [col]   [ ppm    0    -x_min*ppm        ] [x_world]
    [row] = [ 0    -ppm   y_max*ppm         ] [y_world]
    [1  ]   [ 0      0     1                ] [1      ]

where ``ppm`` is ``pixels_per_meter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poc_homography.types import Meters, Unitless


@dataclass(frozen=True)
class GroundRaster:
    """
    Fixed-extent ground-plane raster grid in world meters.

    The raster discretizes a rectangular world patch
    ``[x_min, x_max] x [y_min, y_max]`` (meters) at ``pixels_per_meter``
    resolution. World-Y increases upward (North) while raster rows increase
    downward, so the world->raster mapping flips the Y axis.

    Attributes:
        x_min: Minimum world X (East) in meters (left raster edge).
        x_max: Maximum world X (East) in meters (right raster edge).
        y_min: Minimum world Y (North) in meters (bottom raster edge).
        y_max: Maximum world Y (North) in meters (top raster edge).
        pixels_per_meter: Raster resolution (cells per meter, must be > 0).
    """

    x_min: Meters
    x_max: Meters
    y_min: Meters
    y_max: Meters
    pixels_per_meter: Unitless

    def __post_init__(self) -> None:
        """Validate extent ordering, positive resolution and integral cell counts.

        The extent times resolution MUST be integral on each axis: ``width`` and
        ``height`` are ``round(extent * ppm)``, while ``world_to_cell`` /
        ``world_to_raster_matrix`` map the far edge to the un-rounded product. A
        non-integral product therefore desyncs the grid by a sub-pixel fraction,
        so it is rejected here rather than silently producing a skewed raster.

        Raises:
            ValueError: If an extent is non-positive/inverted, ``ppm`` is
                non-positive, or an axis' ``extent * ppm`` is not integral.
        """
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must exceed x_min ({self.x_min})")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must exceed y_min ({self.y_min})")
        if self.pixels_per_meter <= 0:
            raise ValueError(f"pixels_per_meter must be positive, got {self.pixels_per_meter}")
        ppm = float(self.pixels_per_meter)
        self._require_integral_product("x", float(self.x_max - self.x_min), ppm)
        self._require_integral_product("y", float(self.y_max - self.y_min), ppm)

    @staticmethod
    def _require_integral_product(axis: str, extent: float, ppm: float) -> None:
        """Raise if ``extent * ppm`` is not integral within tolerance."""
        product = extent * ppm
        if abs(product - round(product)) >= 1e-6:
            raise ValueError(
                f"{axis}-extent ({extent}) * pixels_per_meter ({ppm}) = {product} "
                "must be integral (within 1e-6); choose an extent/ppm that divides "
                "evenly into whole cells."
            )

    @property
    def width(self) -> int:
        """Raster width in cells (columns), spanning world-X extent."""
        return round((self.x_max - self.x_min) * self.pixels_per_meter)

    @property
    def height(self) -> int:
        """Raster height in cells (rows), spanning world-Y extent."""
        return round((self.y_max - self.y_min) * self.pixels_per_meter)

    @property
    def shape(self) -> tuple[int, int]:
        """Raster shape as ``(height, width)`` in cells."""
        return self.height, self.width

    def world_to_raster_matrix(self) -> np.ndarray:
        """
        Build the 3x3 affine mapping world meters to raster (col, row).

        The matrix maps homogeneous world points ``[x, y, 1]`` to homogeneous
        raster points ``[col, row, 1]`` with a Y-axis flip (row increases as
        world-Y decreases).

        Returns:
            A ``(3, 3)`` float64 affine matrix.
        """
        ppm = float(self.pixels_per_meter)
        return np.array(
            [
                [ppm, 0.0, -float(self.x_min) * ppm],
                [0.0, -ppm, float(self.y_max) * ppm],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def world_to_cell(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert world meters to fractional raster ``(col, row)`` coordinates.

        Accepts scalars or numpy arrays (vectorized). The result is NOT rounded
        to integer cells, so callers can round/floor as needed.

        Args:
            x: World X (East) in meters (scalar or array).
            y: World Y (North) in meters (scalar or array).

        Returns:
            Tuple ``(col, row)`` of float arrays (or 0-d arrays for scalars).
        """
        ppm = float(self.pixels_per_meter)
        col = (np.asarray(x, dtype=np.float64) - float(self.x_min)) * ppm
        row = (float(self.y_max) - np.asarray(y, dtype=np.float64)) * ppm
        return col, row

    def cell_to_world(
        self, col: np.ndarray | float, row: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raster ``(col, row)`` coordinates to world meters.

        Inverse of :meth:`world_to_cell`. Cell centers are at integer indices.

        Args:
            col: Raster column (scalar or array).
            row: Raster row (scalar or array).

        Returns:
            Tuple ``(x, y)`` of world coordinate float arrays in meters.
        """
        ppm = float(self.pixels_per_meter)
        x = np.asarray(col, dtype=np.float64) / ppm + float(self.x_min)
        y = float(self.y_max) - np.asarray(row, dtype=np.float64) / ppm
        return x, y
