"""Unit tests for pure functions and DTOs in the webapp layer.

Targets:
- ``compute_tile_bounds`` and ``TileBounds`` from ``homography_precision/services.py``
- ``perpendicular_distance`` from ``homography_precision/services.py``
- ``PointAnnotationDTO`` and ``LineAnnotationDTO`` from ``homography_web/frame_utils.py``

Run with: python -m pytest tests/test_services_pure.py -v
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

# Add webapp to sys.path so Django-free pure imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp"))


# ---------------------------------------------------------------------------
# Imports under test — deferred so the sys.path insert above takes effect.
# We import only pure functions and DTOs (no Django dependency).
# ---------------------------------------------------------------------------

from homography_precision.services import TileBounds, compute_tile_bounds, perpendicular_distance
from homography_web.dtos import LineAnnotationDTO, PointAnnotationDTO

# ============================================================================
# compute_tile_bounds
# ============================================================================


class TestComputeTileBounds:
    """Tests for the ``compute_tile_bounds`` pure function."""

    # -- basic cases ---------------------------------------------------------

    def test_single_tile_at_zoom_zero(self):
        """At z=0 a single tile covers the entire image (clamped to image size)."""
        tb = compute_tile_bounds(width=1024, height=1024, x=0, y=0, z=0)

        assert tb.x0 == 0, "x0 should start at 0"
        assert tb.y0 == 0, "y0 should start at 0"
        assert tb.x1 == 1024, "x1 should be clamped to image width"
        assert tb.y1 == 1024, "y1 should be clamped to image height"
        assert tb.is_empty is False, "tile covering the whole image is not empty"

    def test_tile_at_max_zoom_origin(self):
        """At the maximum zoom level the first tile covers [0, 256) on each axis."""
        width, height = 1024, 1024
        max_level = math.ceil(math.log2(max(width, height)))  # 10

        tb = compute_tile_bounds(width=width, height=height, x=0, y=0, z=max_level)

        assert tb.x0 == 0
        assert tb.y0 == 0
        assert tb.x1 == 256, "at max zoom, level_scale=1 so tile spans 256 px"
        assert tb.y1 == 256
        assert tb.is_empty is False

    def test_tile_at_max_zoom_second_tile(self):
        """Second tile on the x-axis at max zoom covers [256, 512)."""
        width, height = 1024, 1024
        max_level = math.ceil(math.log2(max(width, height)))

        tb = compute_tile_bounds(width=width, height=height, x=1, y=0, z=max_level)

        assert tb.x0 == 256
        assert tb.y0 == 0
        assert tb.x1 == 512
        assert tb.y1 == 256
        assert tb.is_empty is False

    def test_tile_at_intermediate_zoom(self):
        """At an intermediate zoom level, tiles are scaled by 2^(max_level - z)."""
        width, height = 1024, 1024
        max_level = math.ceil(math.log2(max(width, height)))  # 10
        z = max_level - 1  # 9 => level_scale = 2

        tb = compute_tile_bounds(width=width, height=height, x=0, y=0, z=z)

        assert tb.x0 == 0
        assert tb.y0 == 0
        assert tb.x1 == 512, "level_scale=2, so tile spans 256*2 = 512 px"
        assert tb.y1 == 512
        assert tb.is_empty is False

    # -- clamping / partial tiles -------------------------------------------

    def test_tile_partially_outside_right_edge(self):
        """A tile whose raw x1 exceeds image width is clamped."""
        width, height = 1000, 1000
        max_level = math.ceil(math.log2(max(width, height)))  # 10

        # At max zoom, tile (3, 0) spans [768, 1024). Width=1000 => x1 clamped to 1000.
        tb = compute_tile_bounds(width=width, height=height, x=3, y=0, z=max_level)

        assert tb.x0 == 768
        assert tb.x1 == 1000, "x1 clamped to image width"
        assert tb.is_empty is False

    def test_tile_partially_outside_bottom_edge(self):
        """A tile whose raw y1 exceeds image height is clamped."""
        width, height = 1024, 900
        max_level = math.ceil(math.log2(max(width, height)))  # 10

        # Tile (0, 3) at max zoom: y0=768, y1=1024 -> clamped to 900.
        tb = compute_tile_bounds(width=width, height=height, x=0, y=3, z=max_level)

        assert tb.y0 == 768
        assert tb.y1 == 900, "y1 clamped to image height"
        assert tb.is_empty is False

    # -- completely outside --------------------------------------------------

    def test_tile_completely_outside_is_empty(self):
        """A tile whose coordinates are entirely beyond the image is empty."""
        width, height = 1024, 1024
        max_level = math.ceil(math.log2(max(width, height)))  # 10

        # Tile (4, 0) at max zoom: x0=1024, x1=1280 -> clamped to (1024, 1024).
        tb = compute_tile_bounds(width=width, height=height, x=4, y=0, z=max_level)

        assert tb.x0 == 1024
        assert tb.x1 == 1024, "both clamped to width"
        assert tb.is_empty is True, "zero-width tile is empty"

    def test_tile_far_outside_is_empty(self):
        """A tile at a very large x index is empty."""
        tb = compute_tile_bounds(width=512, height=512, x=100, y=100, z=9)

        assert tb.is_empty is True

    # -- edge-case tile sizes -----------------------------------------------

    def test_size_one(self):
        """With size=1, each tile covers 1*level_scale pixels."""
        width, height = 16, 16
        max_level = math.ceil(math.log2(16))  # 4

        tb = compute_tile_bounds(width=16, height=16, x=0, y=0, z=max_level, size=1)

        # level_scale = 2^(4-4) = 1, so tile covers [0, 1).
        assert tb.x0 == 0
        assert tb.x1 == 1
        assert tb.is_empty is False

    def test_size_zero_yields_empty(self):
        """With size=0, x0==x1 and y0==y1 so the tile is empty."""
        tb = compute_tile_bounds(width=1024, height=1024, x=0, y=0, z=0, size=0)

        assert tb.x0 == 0
        assert tb.x1 == 0
        assert tb.y0 == 0
        assert tb.y1 == 0
        assert tb.is_empty is True

    # -- non-square image ----------------------------------------------------

    def test_non_square_image(self):
        """max_level is derived from the larger dimension."""
        width, height = 2048, 512
        max_level = math.ceil(math.log2(max(width, height)))  # 11

        tb = compute_tile_bounds(width=width, height=height, x=0, y=0, z=max_level)

        assert tb.x1 == 256, "at max zoom, level_scale=1"
        assert tb.y1 == 256
        assert tb.is_empty is False

    # -- TileBounds is frozen -----------------------------------------------

    def test_tile_bounds_is_frozen(self):
        """TileBounds is a frozen dataclass — mutation raises FrozenInstanceError."""
        tb = compute_tile_bounds(width=1024, height=1024, x=0, y=0, z=0)

        with pytest.raises(FrozenInstanceError):
            tb.x0 = 999  # type: ignore[misc]


# ============================================================================
# perpendicular_distance
# ============================================================================


class TestPerpendicularDistance:
    """Tests for the ``perpendicular_distance`` pure function."""

    def test_point_on_the_line(self):
        """A point that lies exactly on the line has distance 0."""
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([5.0, 0.0])  # midpoint, on the line

        assert perpendicular_distance(p, a, b) == pytest.approx(0.0)

    def test_point_at_line_endpoint(self):
        """A point coinciding with a line endpoint has distance 0."""
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])

        assert perpendicular_distance(a, a, b) == pytest.approx(0.0)
        assert perpendicular_distance(b, a, b) == pytest.approx(0.0)

    def test_point_above_midpoint_horizontal_line(self):
        """Point directly above midpoint of a horizontal line."""
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([5.0, 7.0])

        assert perpendicular_distance(p, a, b) == pytest.approx(7.0)

    def test_point_below_line(self):
        """Point below a horizontal line has positive distance."""
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([5.0, -3.0])

        assert perpendicular_distance(p, a, b) == pytest.approx(3.0)

    def test_point_on_line_extension_beyond_b(self):
        """Point on the line extension beyond endpoint b.

        The function computes the perpendicular distance to the *infinite* line
        through a and b (not clamped to the segment), so the distance is still 0
        for a point on the line extension.
        """
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([20.0, 0.0])  # on the extension

        assert perpendicular_distance(p, a, b) == pytest.approx(0.0)

    def test_point_off_line_extension_beyond_b(self):
        """Point off the line but beyond endpoint b.

        Since the function projects onto the infinite line, the perpendicular
        distance is the offset from the line, not from the segment endpoint.
        """
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([20.0, 5.0])

        # The infinite line is y=0; perpendicular distance is |5| = 5.
        assert perpendicular_distance(p, a, b) == pytest.approx(5.0)

    def test_diagonal_line(self):
        """Point perpendicular to a 45-degree line."""
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 10.0])
        # Point (0, 10) — distance to line y=x through origin is 10/sqrt(2).
        p = np.array([0.0, 10.0])

        expected = 10.0 / math.sqrt(2.0)
        assert perpendicular_distance(p, a, b) == pytest.approx(expected)

    def test_degenerate_line_zero_length(self):
        """When a == b (zero-length line), distance is just ||p - a||."""
        a = np.array([3.0, 4.0])
        b = np.array([3.0, 4.0])
        p = np.array([0.0, 0.0])

        expected = math.sqrt(3.0**2 + 4.0**2)  # 5.0
        assert perpendicular_distance(p, a, b) == pytest.approx(expected)

    def test_degenerate_line_point_at_same_location(self):
        """When a == b == p, distance is 0."""
        a = np.array([5.0, 5.0])
        b = np.array([5.0, 5.0])
        p = np.array([5.0, 5.0])

        assert perpendicular_distance(p, a, b) == pytest.approx(0.0)

    def test_returns_float(self):
        """Return type is a plain float, not a numpy scalar."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = np.array([0.5, 1.0])

        result = perpendicular_distance(p, a, b)
        assert isinstance(result, float)


# ============================================================================
# PointAnnotationDTO
# ============================================================================


class TestPointAnnotationDTO:
    """Tests for the ``PointAnnotationDTO`` frozen dataclass."""

    def test_construction(self):
        """DTO can be constructed with required fields."""
        dto = PointAnnotationDTO(gcp_id="GCP_1", pixel_x=100.5, pixel_y=200.3)

        assert dto.gcp_id == "GCP_1"
        assert dto.pixel_x == 100.5
        assert dto.pixel_y == 200.3

    def test_immutability(self):
        """PointAnnotationDTO is frozen — mutation raises FrozenInstanceError."""
        dto = PointAnnotationDTO(gcp_id="GCP_1", pixel_x=100.0, pixel_y=200.0)

        with pytest.raises(FrozenInstanceError):
            dto.gcp_id = "GCP_2"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dto.pixel_x = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """to_dict() returns a plain dict with exact keys and values."""
        dto = PointAnnotationDTO(gcp_id="GCP_A", pixel_x=10.0, pixel_y=20.0)
        d = dto.to_dict()

        assert d == {"gcp_id": "GCP_A", "pixel_x": 10.0, "pixel_y": 20.0}

    def test_to_dict_keys(self):
        """to_dict() produces exactly three keys."""
        dto = PointAnnotationDTO(gcp_id="X", pixel_x=0.0, pixel_y=0.0)
        assert set(dto.to_dict().keys()) == {"gcp_id", "pixel_x", "pixel_y"}

    def test_round_trip(self):
        """Construct DTO, serialize, and verify the dict matches expectations."""
        expected = {"gcp_id": "PT_42", "pixel_x": 123.456, "pixel_y": 789.012}
        dto = PointAnnotationDTO(**expected)

        assert dto.to_dict() == expected

    def test_equality(self):
        """Two DTOs with identical fields are equal."""
        dto1 = PointAnnotationDTO(gcp_id="A", pixel_x=1.0, pixel_y=2.0)
        dto2 = PointAnnotationDTO(gcp_id="A", pixel_x=1.0, pixel_y=2.0)

        assert dto1 == dto2

    def test_inequality(self):
        """DTOs with different fields are not equal."""
        dto1 = PointAnnotationDTO(gcp_id="A", pixel_x=1.0, pixel_y=2.0)
        dto2 = PointAnnotationDTO(gcp_id="B", pixel_x=1.0, pixel_y=2.0)

        assert dto1 != dto2


# ============================================================================
# LineAnnotationDTO
# ============================================================================


class TestLineAnnotationDTO:
    """Tests for the ``LineAnnotationDTO`` frozen dataclass."""

    def test_construction_without_points(self):
        """DTO can be constructed without the optional ``points`` field."""
        dto = LineAnnotationDTO(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=100.0,
            end_pixel_y=200.0,
        )

        assert dto.line_id == "L1"
        assert dto.start_pixel_x == 0.0
        assert dto.end_pixel_x == 100.0
        assert dto.points is None

    def test_construction_with_points(self):
        """DTO can be constructed with an explicit ``points`` polyline."""
        pts = [[0.0, 0.0], [50.0, 100.0], [100.0, 200.0]]
        dto = LineAnnotationDTO(
            line_id="L2",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=100.0,
            end_pixel_y=200.0,
            points=pts,
        )

        assert dto.points == pts

    def test_immutability(self):
        """LineAnnotationDTO is frozen — mutation raises FrozenInstanceError."""
        dto = LineAnnotationDTO(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=10.0,
            end_pixel_y=10.0,
        )

        with pytest.raises(FrozenInstanceError):
            dto.line_id = "L99"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dto.start_pixel_x = 999.0  # type: ignore[misc]

    def test_to_dict_without_points(self):
        """to_dict() omits the ``points`` key when points is None."""
        dto = LineAnnotationDTO(
            line_id="L1",
            start_pixel_x=1.0,
            start_pixel_y=2.0,
            end_pixel_x=3.0,
            end_pixel_y=4.0,
        )
        d = dto.to_dict()

        assert d == {
            "line_id": "L1",
            "start_pixel_x": 1.0,
            "start_pixel_y": 2.0,
            "end_pixel_x": 3.0,
            "end_pixel_y": 4.0,
        }
        assert "points" not in d

    def test_to_dict_with_points(self):
        """to_dict() includes the ``points`` key when points is provided."""
        pts = [[0.0, 0.0], [5.0, 5.0]]
        dto = LineAnnotationDTO(
            line_id="L3",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=5.0,
            end_pixel_y=5.0,
            points=pts,
        )
        d = dto.to_dict()

        assert "points" in d
        assert d["points"] == pts

    def test_to_dict_keys_without_points(self):
        """to_dict() without points has exactly 5 keys."""
        dto = LineAnnotationDTO(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=0.0,
            end_pixel_y=0.0,
        )
        assert set(dto.to_dict().keys()) == {
            "line_id",
            "start_pixel_x",
            "start_pixel_y",
            "end_pixel_x",
            "end_pixel_y",
        }

    def test_to_dict_keys_with_points(self):
        """to_dict() with points has exactly 6 keys."""
        dto = LineAnnotationDTO(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=0.0,
            end_pixel_y=0.0,
            points=[[1.0, 2.0]],
        )
        assert set(dto.to_dict().keys()) == {
            "line_id",
            "start_pixel_x",
            "start_pixel_y",
            "end_pixel_x",
            "end_pixel_y",
            "points",
        }

    def test_round_trip_without_points(self):
        """Construct, serialize, and verify the dict matches expectations (no points)."""
        expected = {
            "line_id": "LINE_7",
            "start_pixel_x": 10.5,
            "start_pixel_y": 20.5,
            "end_pixel_x": 30.5,
            "end_pixel_y": 40.5,
        }
        dto = LineAnnotationDTO(**expected)

        assert dto.to_dict() == expected

    def test_round_trip_with_points(self):
        """Construct, serialize, and verify the dict matches expectations (with points)."""
        expected = {
            "line_id": "LINE_8",
            "start_pixel_x": 0.0,
            "start_pixel_y": 0.0,
            "end_pixel_x": 100.0,
            "end_pixel_y": 100.0,
            "points": [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]],
        }
        dto = LineAnnotationDTO(**expected)

        assert dto.to_dict() == expected

    def test_equality(self):
        """Two DTOs with identical fields are equal."""
        kwargs = dict(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=10.0,
            end_pixel_y=10.0,
        )
        assert LineAnnotationDTO(**kwargs) == LineAnnotationDTO(**kwargs)

    def test_inequality(self):
        """DTOs with different fields are not equal."""
        base = dict(
            line_id="L1",
            start_pixel_x=0.0,
            start_pixel_y=0.0,
            end_pixel_x=10.0,
            end_pixel_y=10.0,
        )
        dto1 = LineAnnotationDTO(**base)
        dto2 = LineAnnotationDTO(**{**base, "line_id": "L2"})

        assert dto1 != dto2
