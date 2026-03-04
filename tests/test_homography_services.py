"""Tests for homography_precision.services computation functions.

Tests the critical service-layer functions:
- compute_point_homography
- compute_line_homography
- compute_line_errors

Uses real OpenCV homography computation with synthetic but geometrically
valid test data. Filesystem access (GCP repo loading) is mocked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Django setup — same pattern as other test files
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")
sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))

import django

django.setup()

# Module under test
from homography_precision.services import (
    LineErrorResult,
    LineHomographyResult,
    PointHomographyResult,
    compute_line_errors,
    compute_line_homography,
    compute_point_homography,
    load_gcp_registry,
    perpendicular_distance,
)
from homography_web.dtos import LineAnnotationDTO, PointAnnotationDTO

from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points import GCPRegistry, MapPoint

# ---------------------------------------------------------------------------
# Fixtures — synthetic data with a known 2x-scale homography
# ---------------------------------------------------------------------------
#
# The "camera" coordinates are (x, y) and the "map" coordinates are (2x, 2y).
# This creates a simple, exact homography H that OpenCV can recover perfectly.
# With 6+ well-distributed points, RANSAC should find all as inliers.

_CAMERA_MAP_PAIRS: list[tuple[str, float, float, float, float]] = [
    # (gcp_id, camera_x, camera_y, map_x, map_y)
    ("gcp_01", 100.0, 100.0, 200.0, 200.0),
    ("gcp_02", 300.0, 100.0, 600.0, 200.0),
    ("gcp_03", 500.0, 100.0, 1000.0, 200.0),
    ("gcp_04", 100.0, 400.0, 200.0, 800.0),
    ("gcp_05", 300.0, 400.0, 600.0, 800.0),
    ("gcp_06", 500.0, 400.0, 1000.0, 800.0),
]


@pytest.fixture()
def gcp_registry() -> GCPRegistry:
    """GCPRegistry with 6 well-distributed points (map coords = 2x camera)."""
    points = {
        gcp_id: MapPoint(pixel_x=mx, pixel_y=my)
        for gcp_id, _cx, _cy, mx, my in _CAMERA_MAP_PAIRS
    }
    return GCPRegistry(map_id="test_map", points=points)


@pytest.fixture()
def point_annotations() -> list[PointAnnotationDTO]:
    """Point annotations matching the camera coords in _CAMERA_MAP_PAIRS."""
    return [
        PointAnnotationDTO(gcp_id=gcp_id, pixel_x=cx, pixel_y=cy)
        for gcp_id, cx, cy, _mx, _my in _CAMERA_MAP_PAIRS
    ]


# Lines: two horizontal lines across the image/map.
# Camera line endpoints -> map endpoints follow the same 2x scaling.
_LINE_DATA: list[tuple[str, float, float, float, float, float, float, float, float]] = [
    # (line_id, cam_sx, cam_sy, cam_ex, cam_ey, map_sx, map_sy, map_ex, map_ey)
    ("line_A", 100.0, 150.0, 500.0, 150.0, 200.0, 300.0, 1000.0, 300.0),
    ("line_B", 100.0, 350.0, 500.0, 350.0, 200.0, 700.0, 1000.0, 700.0),
    ("line_C", 200.0, 100.0, 200.0, 400.0, 400.0, 200.0, 400.0, 800.0),
]


@pytest.fixture()
def line_annotations_dicts() -> list[dict]:
    """Line annotations as plain dicts (the format compute_line_homography expects)."""
    return [
        {
            "line_id": lid,
            "start_pixel_x": csx,
            "start_pixel_y": csy,
            "end_pixel_x": cex,
            "end_pixel_y": cey,
        }
        for lid, csx, csy, cex, cey, _msx, _msy, _mex, _mey in _LINE_DATA
    ]


@pytest.fixture()
def line_annotations_dtos() -> list[LineAnnotationDTO]:
    """Line annotations as LineAnnotationDTO instances."""
    return [
        LineAnnotationDTO(
            line_id=lid,
            start_pixel_x=csx,
            start_pixel_y=csy,
            end_pixel_x=cex,
            end_pixel_y=cey,
        )
        for lid, csx, csy, cex, cey, _msx, _msy, _mex, _mey in _LINE_DATA
    ]


@pytest.fixture()
def line_registry() -> dict[str, dict[str, float]]:
    """Line registry mapping line_id -> map line definition."""
    return {
        lid: {
            "start_x": msx,
            "start_y": msy,
            "end_x": mex,
            "end_y": mey,
        }
        for lid, _csx, _csy, _cex, _cey, msx, msy, mex, mey in _LINE_DATA
    }


# ---------------------------------------------------------------------------
# Tests: compute_point_homography
# ---------------------------------------------------------------------------


class TestComputePointHomography:
    """Tests for compute_point_homography service function."""

    def test_returns_point_homography_result(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """The function must return a PointHomographyResult dataclass."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert isinstance(result, PointHomographyResult)

    def test_metrics_keys(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """Metrics dict must contain all expected keys."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        expected_keys = {
            "num_gcps",
            "num_inliers",
            "inlier_ratio",
            "mean_reproj_error",
            "max_reproj_error",
            "rmse",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_num_gcps_matches_annotations(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """num_gcps must equal the number of annotations passed in."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert result.metrics["num_gcps"] == len(point_annotations)

    def test_all_inliers_with_perfect_data(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """With a perfect 2x-scale mapping, all points should be inliers."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert result.metrics["num_inliers"] == len(point_annotations)
        assert result.metrics["inlier_ratio"] == 1.0

    def test_low_reprojection_error_with_perfect_data(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """With exact correspondences, reprojection error must be near zero."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert result.metrics["mean_reproj_error"] < 1.0
        assert result.metrics["max_reproj_error"] < 1.0
        assert result.metrics["rmse"] < 1.0

    def test_per_point_errors_structure(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """Each per_point_errors entry must have the expected keys."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert len(result.per_point_errors) == len(point_annotations)

        expected_keys = {
            "gcp_id",
            "error_px",
            "camera_dx",
            "camera_dy",
            "map_dx",
            "map_dy",
            "camera_original",
            "camera_reprojected",
            "map_original",
            "map_projected",
        }
        for entry in result.per_point_errors:
            assert set(entry.keys()) == expected_keys

    def test_per_point_gcp_ids_match_annotations(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """GCP IDs in per_point_errors must match the input annotations."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        result_ids = [e["gcp_id"] for e in result.per_point_errors]
        expected_ids = [a.gcp_id for a in point_annotations]
        assert result_ids == expected_ids

    def test_per_point_errors_near_zero_for_perfect_data(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """With perfect correspondences, per-point error_px should be near zero."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        for entry in result.per_point_errors:
            assert entry["error_px"] < 1.0, f"GCP {entry['gcp_id']} error too large"

    def test_overlays_structure(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """Overlays dict must have expected top-level and nested keys."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        assert "camera" in result.overlays
        assert "map" in result.overlays
        assert "annotations" in result.overlays["camera"]
        assert "reprojected_gcps" in result.overlays["camera"]
        assert "gcps" in result.overlays["map"]
        assert "projected_annotations" in result.overlays["map"]

    def test_overlays_lengths(
        self,
        point_annotations: list[PointAnnotationDTO],
        gcp_registry: GCPRegistry,
    ) -> None:
        """Each overlay list must have the same length as annotations."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ):
            result = compute_point_homography(point_annotations, "test_map")

        n = len(point_annotations)
        assert len(result.overlays["camera"]["annotations"]) == n
        assert len(result.overlays["camera"]["reprojected_gcps"]) == n
        assert len(result.overlays["map"]["gcps"]) == n
        assert len(result.overlays["map"]["projected_annotations"]) == n

    def test_too_few_annotations_raises_value_error(
        self,
        gcp_registry: GCPRegistry,
    ) -> None:
        """Fewer than 4 annotations must raise ValueError."""
        few = [
            PointAnnotationDTO(gcp_id="gcp_01", pixel_x=100.0, pixel_y=100.0),
            PointAnnotationDTO(gcp_id="gcp_02", pixel_x=300.0, pixel_y=100.0),
            PointAnnotationDTO(gcp_id="gcp_03", pixel_x=500.0, pixel_y=100.0),
        ]
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ), pytest.raises(ValueError, match="at least 4"):
            compute_point_homography(few, "test_map")

    def test_missing_gcp_raises_value_error(
        self,
        gcp_registry: GCPRegistry,
    ) -> None:
        """Annotations referencing a missing GCP must raise ValueError."""
        bad_annotations = [
            PointAnnotationDTO(gcp_id="gcp_01", pixel_x=100.0, pixel_y=100.0),
            PointAnnotationDTO(gcp_id="gcp_02", pixel_x=300.0, pixel_y=100.0),
            PointAnnotationDTO(gcp_id="gcp_03", pixel_x=500.0, pixel_y=100.0),
            PointAnnotationDTO(gcp_id="MISSING", pixel_x=200.0, pixel_y=200.0),
        ]
        with patch(
            "homography_precision.services.from_gcp_repo",
            return_value=gcp_registry,
        ), pytest.raises(ValueError, match="GCP not found"):
            compute_point_homography(bad_annotations, "test_map")


# ---------------------------------------------------------------------------
# Tests: compute_line_homography
# ---------------------------------------------------------------------------


class TestComputeLineHomography:
    """Tests for compute_line_homography service function."""

    def test_returns_line_homography_result(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """The function must return a LineHomographyResult dataclass."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        assert isinstance(result, LineHomographyResult)

    def test_metrics_keys(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """Metrics dict must contain all expected keys."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        expected_keys = {
            "num_lines",
            "num_inliers",
            "inlier_ratio",
            "mean_perp_error",
            "max_perp_error",
            "rmse",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_num_lines_matches_input(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """num_lines must equal the number of line annotations."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        assert result.metrics["num_lines"] == len(line_annotations_dicts)

    def test_homography_matrix_shape(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """homography_matrix must be a 3x3 list of lists."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        matrix = result.homography_matrix
        assert len(matrix) == 3
        for row in matrix:
            assert len(row) == 3

    def test_low_error_with_perfect_data(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """With perfectly corresponding lines, errors should be low."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        # With exact 2x scaling the perpendicular errors should be small
        assert result.metrics["mean_perp_error"] < 5.0
        assert result.metrics["rmse"] < 5.0

    def test_homography_matrix_is_serializable(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> None:
        """The homography_matrix must be plain Python floats (JSON-serializable)."""
        result = compute_line_homography(
            line_annotations_dicts, line_registry, "test_map"
        )
        for row in result.homography_matrix:
            for val in row:
                assert isinstance(val, float)

    def test_too_few_lines_raises_value_error(
        self,
        line_registry: dict,
    ) -> None:
        """Fewer than 2 lines must raise ValueError."""
        one_line = [
            {
                "line_id": "line_A",
                "start_pixel_x": 100.0,
                "start_pixel_y": 150.0,
                "end_pixel_x": 500.0,
                "end_pixel_y": 150.0,
            }
        ]
        with pytest.raises(ValueError, match="at least 2"):
            compute_line_homography(one_line, line_registry, "test_map")


# ---------------------------------------------------------------------------
# Tests: compute_line_errors
# ---------------------------------------------------------------------------


class TestComputeLineErrors:
    """Tests for compute_line_errors service function."""

    @pytest.fixture()
    def precomputed_homography(
        self,
        line_annotations_dicts: list[dict],
        line_registry: dict,
    ) -> MapPointHomography:
        """Compute a homography from lines for use in line-error tests."""
        h = MapPointHomography(map_id="test_map")
        h.compute_from_lines(
            line_annotations=line_annotations_dicts,
            line_registry=line_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.3,
        )
        return h

    def test_returns_line_error_result(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """Must return a LineErrorResult dataclass."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        assert isinstance(result, LineErrorResult)

    def test_metrics_keys(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """Metrics dict must have expected keys."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        expected_keys = {"num_lines", "mean_line_error", "max_line_error"}
        assert set(result.metrics.keys()) == expected_keys

    def test_num_lines_matches_input(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """num_lines in metrics must match the input count."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        assert result.metrics["num_lines"] == len(line_annotations_dtos)

    def test_per_line_errors_structure(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """Each per_line_errors entry must have the expected keys."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        assert len(result.per_line_errors) == len(line_annotations_dtos)

        expected_keys = {
            "line_id",
            "error_px",
            "start_error",
            "end_error",
            "map_start_error",
            "map_end_error",
        }
        for entry in result.per_line_errors:
            assert set(entry.keys()) == expected_keys

    def test_per_line_ids_match(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """Line IDs in per_line_errors must match the input order."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        result_ids = [e["line_id"] for e in result.per_line_errors]
        expected_ids = [a.line_id for a in line_annotations_dtos]
        assert result_ids == expected_ids

    def test_low_errors_with_matching_homography(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """When line annotations match the homography, errors should be low."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        assert result.metrics["mean_line_error"] < 5.0
        assert result.metrics["max_line_error"] < 10.0

    def test_line_overlays_structure(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """line_overlays must have expected top-level and nested keys."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        assert "camera" in result.line_overlays
        assert "map" in result.line_overlays
        assert "annotations" in result.line_overlays["camera"]
        assert "reprojected_lines" in result.line_overlays["camera"]
        assert "gcp_lines" in result.line_overlays["map"]
        assert "projected_lines" in result.line_overlays["map"]

    def test_line_overlays_lengths(
        self,
        line_annotations_dtos: list[LineAnnotationDTO],
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """Each overlay list must have one entry per line annotation."""
        result = compute_line_errors(
            line_annotations_dtos, line_registry, precomputed_homography
        )
        n = len(line_annotations_dtos)
        assert len(result.line_overlays["camera"]["annotations"]) == n
        assert len(result.line_overlays["camera"]["reprojected_lines"]) == n
        assert len(result.line_overlays["map"]["gcp_lines"]) == n
        assert len(result.line_overlays["map"]["projected_lines"]) == n

    def test_missing_line_in_registry_raises_key_error(
        self,
        line_registry: dict,
        precomputed_homography: MapPointHomography,
    ) -> None:
        """A line_id not in the registry must raise KeyError."""
        bad_dto = LineAnnotationDTO(
            line_id="NONEXISTENT",
            start_pixel_x=100.0,
            start_pixel_y=100.0,
            end_pixel_x=200.0,
            end_pixel_y=200.0,
        )
        with pytest.raises(KeyError, match="NONEXISTENT"):
            compute_line_errors([bad_dto], line_registry, precomputed_homography)


# ---------------------------------------------------------------------------
# Tests: perpendicular_distance helper
# ---------------------------------------------------------------------------


class TestPerpendicularDistance:
    """Tests for the perpendicular_distance geometry helper."""

    def test_point_on_line_returns_zero(self) -> None:
        """A point on the line must have zero distance."""
        p = np.array([5.0, 5.0])
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 10.0])
        assert perpendicular_distance(p, a, b) == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_distance_horizontal_line(self) -> None:
        """Distance from (5, 3) to horizontal line y=0 is 3."""
        p = np.array([5.0, 3.0])
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        assert perpendicular_distance(p, a, b) == pytest.approx(3.0, abs=1e-10)

    def test_perpendicular_distance_vertical_line(self) -> None:
        """Distance from (4, 5) to vertical line x=0 is 4."""
        p = np.array([4.0, 5.0])
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 10.0])
        assert perpendicular_distance(p, a, b) == pytest.approx(4.0, abs=1e-10)

    def test_zero_length_line_returns_point_distance(self) -> None:
        """When a == b, distance is simply the Euclidean distance from p to a."""
        p = np.array([3.0, 4.0])
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert perpendicular_distance(p, a, b) == pytest.approx(5.0, abs=1e-10)

    def test_point_beyond_segment_still_uses_infinite_line(self) -> None:
        """The function projects onto the infinite line, not the segment."""
        # Point is at (15, 0). Segment from (0,0) to (10,0).
        # Closest on infinite line is (15, 0) -> distance 0.
        p = np.array([15.0, 0.0])
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        assert perpendicular_distance(p, a, b) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# load_gcp_registry
# ============================================================================


class TestLoadGCPRegistry:
    """Tests for the ``load_gcp_registry`` service function."""

    def test_returns_serialised_points(self) -> None:
        """load_gcp_registry returns dict mapping point IDs to pixel coords."""
        registry = GCPRegistry(
            map_id="test_map",
            points={
                "PS1": MapPoint(pixel_x=100.0, pixel_y=200.0),
                "PS2": MapPoint(pixel_x=300.5, pixel_y=400.5),
            },
        )
        with patch("homography_precision.services.from_gcp_repo", return_value=registry):
            result = load_gcp_registry("test_map")

        assert result == {
            "PS1": {"pixel_x": 100.0, "pixel_y": 200.0},
            "PS2": {"pixel_x": 300.5, "pixel_y": 400.5},
        }

    def test_empty_registry(self) -> None:
        """An empty registry returns an empty dict."""
        registry = GCPRegistry(map_id="test_map", points={})
        with patch("homography_precision.services.from_gcp_repo", return_value=registry):
            result = load_gcp_registry("test_map")

        assert result == {}

    def test_propagates_key_error(self) -> None:
        """KeyError from from_gcp_repo propagates to caller."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            side_effect=KeyError("no such map"),
        ):
            with pytest.raises(KeyError, match="no such map"):
                load_gcp_registry("nonexistent_map")

    def test_propagates_os_error(self) -> None:
        """OSError from missing data dir propagates to caller."""
        with patch(
            "homography_precision.services.from_gcp_repo",
            side_effect=OSError("data dir not found"),
        ):
            with pytest.raises(OSError, match="data dir not found"):
                load_gcp_registry("test_map")


# ---------------------------------------------------------------------------
# View guard ordering — map_id=None must return 422 before loading data
# ---------------------------------------------------------------------------

from django.test import Client


class TestMapIdGuardOrdering:
    """Verify that views return 422 when tenant has no map configured.

    The fix ensures ``svc.require_map_id()`` is checked *before* any
    data-loading calls (``load_test_case_by_name``, ``load_line_test_case_by_name``).
    Without the guard, the view would reach a data-loading call that raises an
    unrelated error or returns a misleading 404.
    """

    TENANT_ID = "test_tenant"

    @pytest.fixture()
    def client(self):
        """Django test client with tenant_id and require_map_id mocked."""
        with (
            patch(
                "homography_precision.views.get_tenant_id",
                return_value=self.TENANT_ID,
            ),
            patch(
                "homography_precision.views.svc.require_map_id",
                return_value=None,
            ),
        ):
            yield Client()

    def test_compute_homography_returns_422_when_no_map(self, client) -> None:
        """POST /api/compute-homography/ → 422 when map_id is None."""
        import json

        resp = client.post(
            "/homography-precision/api/compute-homography/?tenant_id=" + self.TENANT_ID,
            data=json.dumps({"test_case_name": "anything"}),
            content_type="application/json",
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["success"] is False
        assert "No map configured" in data["error"]

    def test_compute_homography_from_lines_returns_422_when_no_map(self, client) -> None:
        """POST /api/compute-homography-from-lines/ → 422 when map_id is None."""
        import json

        resp = client.post(
            "/homography-precision/api/compute-homography-from-lines/?tenant_id=" + self.TENANT_ID,
            data=json.dumps({"test_case_name": "anything"}),
            content_type="application/json",
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["success"] is False
        assert "No map configured" in data["error"]

    def test_compute_line_errors_returns_422_when_no_map(self, client) -> None:
        """POST /api/compute-line-errors/ → 422 when map_id is None."""
        import json

        resp = client.post(
            "/homography-precision/api/compute-line-errors/?tenant_id=" + self.TENANT_ID,
            data=json.dumps({"test_case_name": "anything"}),
            content_type="application/json",
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["success"] is False
        assert "No map configured" in data["error"]
