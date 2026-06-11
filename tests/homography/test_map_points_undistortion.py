"""Unit tests for lens undistortion in MapPointHomography.

Self-contained tests using synthetic data -- no external test data required.
Validates that MapPointHomography correctly integrates lens undistortion
into both GCP-based and line-based homography computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points import GCPRegistry, MapPoint

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

CAMERA_PIXELS = [
    [200.0, 150.0],
    [1700.0, 150.0],
    [200.0, 900.0],
    [1700.0, 900.0],
    [960.0, 540.0],
    [500.0, 400.0],
    [1400.0, 700.0],
]

MAP_COORDS = [
    [100.0, 75.0],
    [850.0, 75.0],
    [100.0, 450.0],
    [850.0, 450.0],
    [480.0, 270.0],
    [250.0, 200.0],
    [700.0, 350.0],
]

DISTORTION_PARAMS: dict[str, float] = {
    "k1": -0.1,
    "k2": 0.01,
    "k3": 0.0,
    "p1": 0.001,
    "p2": -0.001,
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
}

ZERO_DISTORTION_PARAMS: dict[str, float] = {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gcps() -> list[dict[str, object]]:
    """Build the GCP annotation list expected by compute_from_gcps."""
    return [
        {"pixel_x": cam[0], "pixel_y": cam[1], "gcp_id": f"P{i + 1}"}
        for i, cam in enumerate(CAMERA_PIXELS)
    ]


def _make_registry() -> GCPRegistry:
    """Build a GCPRegistry with the synthetic map coords."""
    points = {
        f"P{i + 1}": MapPoint(pixel_x=mc[0], pixel_y=mc[1]) for i, mc in enumerate(MAP_COORDS)
    }
    return GCPRegistry(map_id="test", points=points)


def _make_line_annotations() -> list[dict[str, object]]:
    """Build synthetic line annotations (at least 2 lines, 4 correspondences).

    Each annotation has a start/end pair of camera pixels that correspond
    to lines defined in the line registry.
    """
    return [
        {
            "line_id": "L1",
            "start_pixel_x": 200.0,
            "start_pixel_y": 150.0,
            "end_pixel_x": 1700.0,
            "end_pixel_y": 150.0,
        },
        {
            "line_id": "L2",
            "start_pixel_x": 200.0,
            "start_pixel_y": 900.0,
            "end_pixel_x": 1700.0,
            "end_pixel_y": 900.0,
        },
        {
            "line_id": "L3",
            "start_pixel_x": 200.0,
            "start_pixel_y": 150.0,
            "end_pixel_x": 200.0,
            "end_pixel_y": 900.0,
        },
        {
            "line_id": "L4",
            "start_pixel_x": 1700.0,
            "start_pixel_y": 150.0,
            "end_pixel_x": 1700.0,
            "end_pixel_y": 900.0,
        },
    ]


def _make_line_registry() -> dict[str, dict[str, float]]:
    """Build a line registry matching the synthetic annotations."""
    return {
        "L1": {"start_x": 100.0, "start_y": 75.0, "end_x": 850.0, "end_y": 75.0},
        "L2": {"start_x": 100.0, "start_y": 450.0, "end_x": 850.0, "end_y": 450.0},
        "L3": {"start_x": 100.0, "start_y": 75.0, "end_x": 100.0, "end_y": 450.0},
        "L4": {"start_x": 850.0, "start_y": 75.0, "end_x": 850.0, "end_y": 450.0},
    }


# ===================================================================
# Test 1 & 2 -- undistortion applied / skipped
# ===================================================================


class TestUndistortionApplied:
    """Verify _undistort_camera_pixels transforms points when coefficients are
    provided, and returns the same object when they are absent."""

    def test_undistortion_transforms_points(self) -> None:
        """Non-zero distortion coefficients should move the points."""
        mph = MapPointHomography(map_id="test", **DISTORTION_PARAMS)

        raw = np.array(CAMERA_PIXELS, dtype=np.float32)
        undistorted = mph._undistort_camera_pixels(raw)

        # The undistorted points must differ from the raw distorted ones
        assert not np.allclose(raw, undistorted, atol=1e-4), (
            "Undistorted points should differ from raw distorted points"
        )

        # Sanity: the result should still be finite and roughly in the same region
        assert np.all(np.isfinite(undistorted))

    def test_undistortion_produces_valid_homography(self) -> None:
        """compute_from_gcps with distortion should succeed and yield valid metrics."""
        mph = MapPointHomography(map_id="test", **DISTORTION_PARAMS)
        result = mph.compute_from_gcps(_make_gcps(), _make_registry())

        assert result.homography_matrix.shape == (3, 3)
        assert result.num_inliers >= 4
        assert result.inlier_ratio >= 0.5
        assert np.isfinite(result.mean_reproj_error)

    def test_undistortion_skipped_when_no_coefficients(self) -> None:
        """Without distortion params the returned array must be the SAME object."""
        mph = MapPointHomography(map_id="test")

        raw = np.array(CAMERA_PIXELS, dtype=np.float32)
        result = mph._undistort_camera_pixels(raw)

        assert result is raw, (
            "Expected the exact same array object (identity) when no distortion is set"
        )


# ===================================================================
# Test 3 -- different homography matrix with / without distortion
# ===================================================================


class TestDifferentHomographyWithDistortion:
    """Same GCPs, different distortion settings must yield different H matrices."""

    def test_matrices_differ(self) -> None:
        gcps = _make_gcps()
        registry = _make_registry()

        # Case A -- no distortion
        mph_no = MapPointHomography(map_id="test")
        result_no = mph_no.compute_from_gcps(gcps, registry)

        # Case B -- with distortion
        mph_dist = MapPointHomography(map_id="test", **DISTORTION_PARAMS)
        result_dist = mph_dist.compute_from_gcps(gcps, registry)

        assert not np.allclose(result_no.homography_matrix, result_dist.homography_matrix), (
            "Homography matrices should differ when distortion is applied"
        )


# ===================================================================
# Test 4 & 5 -- constructor validation
# ===================================================================


class TestConstructorValidation:
    """Validate all-or-nothing parameter semantics and edge cases."""

    @pytest.mark.parametrize(
        "partial_kwargs",
        [
            {"k1": -0.1},
            {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
            {"k1": -0.1, "k2": 0.01, "k3": 0.0},
            {"p1": 0.001, "p2": -0.001},
            {
                "k1": -0.1,
                "k2": 0.01,
                "k3": 0.0,
                "p1": 0.001,
                "p2": -0.001,
                "fx": 1000.0,
                "fy": 1000.0,
                # missing cx and cy
            },
        ],
        ids=[
            "only-k1",
            "only-intrinsics",
            "only-radial",
            "only-tangential",
            "missing-cx-cy",
        ],
    )
    def test_partial_params_raise_value_error(self, partial_kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="All nine"):
            MapPointHomography(map_id="test", **partial_kwargs)

    def test_all_zero_counts_as_provided(self) -> None:
        """All-zero distortion still sets _has_distortion = True."""
        mph = MapPointHomography(map_id="test", **ZERO_DISTORTION_PARAMS)

        assert mph._has_distortion is True

    def test_all_zero_undistort_still_called(self) -> None:
        """Even with zero coefficients the undistort path is taken (not identity)."""
        mph = MapPointHomography(map_id="test", **ZERO_DISTORTION_PARAMS)

        raw = np.array(CAMERA_PIXELS, dtype=np.float32)
        result = mph._undistort_camera_pixels(raw)

        # The code path enters undistort_points; for zero k/p the output
        # should be (almost) identical to input but NOT the same object.
        assert result is not raw
        assert np.allclose(raw, result, atol=1e-4)

    def test_no_params_means_no_distortion(self) -> None:
        mph = MapPointHomography(map_id="test")
        assert mph._has_distortion is False


# ===================================================================
# Test 6 -- undistortion for compute_from_lines
# ===================================================================


class TestLineUndistortion:
    """Verify that compute_from_lines integrates undistortion."""

    def test_line_matrices_differ_with_distortion(self) -> None:
        """Same line data with / without distortion should yield different H."""
        annotations = _make_line_annotations()
        registry = _make_line_registry()

        # Without distortion
        mph_no = MapPointHomography(map_id="test")
        result_no = mph_no.compute_from_lines(annotations, registry)

        # With distortion
        mph_dist = MapPointHomography(map_id="test", **DISTORTION_PARAMS)
        result_dist = mph_dist.compute_from_lines(annotations, registry)

        assert not np.allclose(result_no.homography_matrix, result_dist.homography_matrix), (
            "Line homography should change when distortion is applied"
        )

    def test_line_computation_succeeds_with_distortion(self) -> None:
        """compute_from_lines with distortion should produce valid results."""
        mph = MapPointHomography(map_id="test", **DISTORTION_PARAMS)
        result = mph.compute_from_lines(_make_line_annotations(), _make_line_registry())

        assert result.homography_matrix.shape == (3, 3)
        assert result.num_lines == 4
        assert result.inlier_ratio >= 0.5
        assert np.isfinite(result.mean_perp_error)
