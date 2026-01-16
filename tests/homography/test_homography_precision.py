"""
Test homography pixel precision using MapPointHomography.

This test validates that a homography computed from 4 GCPs achieves
acceptable pixel precision when reprojecting points.

GCP Format:
    - pixel_x: Camera pixel x coordinate
    - pixel_y: Camera pixel y coordinate
    - map_point_id: ID referencing a MapPoint in the registry

Usage:
    pytest tests/homography/test_homography_precision.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.infrastructure.repositories import RepoYamlGroundControlPoint

if TYPE_CHECKING:
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint

TEST_DATA_DIR = Path(__file__).parent.parent.parent
GCPS_DIR = TEST_DATA_DIR / "data" / "gcps"
GCPS_FILE = Path(__file__).parent / "test_data" / "valte_gcps.yaml"
MAP_ID = "valte"


@pytest.fixture
def map_registry() -> dict[str, GroundControlPoint]:
    """Load map point registry from repository, keyed by simple name."""

    repo = RepoYamlGroundControlPoint(GCPS_DIR)
    gcps: dict[str, GroundControlPoint] = repo.get_by_map(MAP_ID)  # type: ignore[assignment]

    if not gcps:
        pytest.fail(f"No GCPs found for map '{MAP_ID}' in {GCPS_DIR}")

    # Convert to simple name keys for compatibility with consumer code
    return {gcp.name: gcp for gcp in gcps.values()}


def load_all_test_cases() -> list[dict[str, Any]]:
    """Load all test cases from YAML file."""
    if not GCPS_FILE.exists():
        return []

    try:
        with open(GCPS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("test_cases", []) if data else []
    except (yaml.YAMLError, OSError):
        return []


def get_test_case_names() -> list[str]:
    """Get list of all test case names from YAML file."""
    return [tc["name"] for tc in load_all_test_cases()]


def load_gcps_from_yaml(test_case_name: str | None = None) -> dict[str, Any]:
    """
    Load GCPs from YAML file.

    Args:
        test_case_name: Name of the test case to load. If None, loads the first one.

    Returns:
        Dictionary with 'image' and 'gcps' keys.
    """
    if not GCPS_FILE.exists():
        pytest.skip(f"GCPs file not found: {GCPS_FILE}")

    with open(GCPS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", []) if data else []
    if not test_cases:
        pytest.skip("No test cases found in GCPs file")

    if test_case_name:
        for tc in test_cases:
            if tc.get("name") == test_case_name:
                return tc
        pytest.skip(f"Test case '{test_case_name}' not found in GCPs file")

    return test_cases[0]


@pytest.fixture
def gcps_test_case() -> dict[str, Any]:
    """Load the first test case from GCPs YAML file."""
    return load_gcps_from_yaml()


@pytest.fixture
def gcps_4_points(gcps_test_case: dict[str, Any]) -> list[dict[str, Any]]:
    """
    4 GCPs for homography computation, loaded from YAML.

    Each GCP contains:
        - pixel_x: x coordinate in camera image (pixels)
        - pixel_y: y coordinate in camera image (pixels)
        - map_point_id: ID of corresponding MapPoint in registry
    """
    return gcps_test_case["gcps"]


@pytest.fixture
def test_image_path(gcps_test_case: dict[str, Any]) -> Path:
    """Path to the test image from the GCPs YAML file (relative to YAML location)."""
    return GCPS_FILE.parent / gcps_test_case["image"]


@pytest.fixture
def homography_provider(map_registry: dict[str, GroundControlPoint]) -> MapPointHomography:
    """Create a MapPointHomography instance."""
    return MapPointHomography(map_id=MAP_ID)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_pixel_precision(
    gcps: list[dict[str, Any]],
    homography: MapPointHomography,
    map_registry: dict[str, GroundControlPoint],
) -> dict[str, float]:
    """
    Measure pixel precision by reprojecting GCPs.

    Process:
        1. Get map coordinates for each GCP from registry
        2. Project map coordinates back to camera pixels using inverse homography
        3. Compare with original GCP pixel coordinates

    Args:
        gcps: List of GCPs with pixel_x, pixel_y, map_point_id
        homography: Computed MapPointHomography
        map_registry: Registry containing MapPoints

    Returns:
        Dictionary with mean_error, max_error, rmse (all in pixels)
    """
    errors = []

    for gcp in gcps:
        # Get the map coordinate from registry
        map_point = map_registry[gcp["map_point_id"]].map_point

        # Project map coordinate back to camera pixel
        map_coord = PixelPoint.create(map_point.pixel_point.x, map_point.pixel_point.y)
        projected_pixel = homography.map_to_camera(map_coord)

        # Compare with original GCP pixel coordinate
        original = np.array([gcp["pixel_x"], gcp["pixel_y"]])
        projected = np.array([projected_pixel.x, projected_pixel.y])

        error = float(np.linalg.norm(projected - original))
        errors.append(error)

    errors_array = np.array(errors)

    return {
        "mean_error": float(np.mean(errors_array)),
        "max_error": float(np.max(errors_array)),
        "rmse": float(np.sqrt(np.mean(errors_array**2))),
        "per_point_errors": errors,
    }


# =============================================================================
# Tests
# =============================================================================


class TestMapPointHomographyComputation:
    """Test homography computation using MapPointHomography."""

    def test_compute_homography_from_4_gcps(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: dict[str, GroundControlPoint],
        homography_provider: MapPointHomography,
    ):
        """Test computing homography from exactly 4 GCPs."""
        result = homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )

        assert result is not None
        assert result.homography_matrix.shape == (3, 3)
        assert result.inverse_matrix.shape == (3, 3)
        assert result.num_gcps >= 4  # May have more GCPs in test data
        assert homography_provider.is_valid()

    def test_homography_matrix_is_invertible(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: dict[str, GroundControlPoint],
        homography_provider: MapPointHomography,
    ):
        """Test that the homography matrix is invertible."""
        result = homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        H = result.homography_matrix
        H_inv = result.inverse_matrix

        # H * H_inv should be identity
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6)


class TestPixelPrecision:
    """Test pixel precision of homography reprojection."""

    def test_pixel_precision_with_4_gcps(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: dict[str, GroundControlPoint],
        homography_provider: MapPointHomography,
    ):
        """Test pixel precision when reprojecting GCPs."""
        # Compute homography
        result = homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
            ransac_threshold=50.0,
        )

        # Measure precision
        precision = compute_pixel_precision(
            gcps=gcps_4_points,
            homography=homography_provider,
            map_registry=map_registry,
        )

        print("\nPixel Precision Metrics:")
        print(f"  Mean error: {precision['mean_error']:.2f} pixels")
        print(f"  Max error:  {precision['max_error']:.2f} pixels")
        print(f"  RMSE:       {precision['rmse']:.2f} pixels")

        # Assert precision thresholds (allows for hand-labeled GCP variance)
        assert precision["mean_error"] < 15.0, (
            f"Mean error too high: {precision['mean_error']:.2f} pixels"
        )
        assert precision["max_error"] < 25.0, (
            f"Max error too high: {precision['max_error']:.2f} pixels"
        )


class TestRoundTrip:
    """Test round-trip projection: camera -> map -> camera."""

    def test_round_trip_camera_to_map_to_camera(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: dict[str, GroundControlPoint],
        homography_provider: MapPointHomography,
    ):
        """Test that round-trip projection preserves coordinates."""
        homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        for gcp in gcps_4_points:
            # Original camera pixel
            original_pixel = PixelPoint.create(gcp["pixel_x"], gcp["pixel_y"])

            # Camera -> Map
            map_point = homography_provider.camera_to_map(original_pixel)

            # Map -> Camera
            map_as_pixel = PixelPoint.create(map_point.pixel_point.x, map_point.pixel_point.y)
            recovered_pixel = homography_provider.map_to_camera(map_as_pixel)

            # Compare
            error = np.linalg.norm(
                np.array([recovered_pixel.x, recovered_pixel.y])
                - np.array([original_pixel.x, original_pixel.y])
            )

            # Round-trip should have low error (tolerance 0.1 pixel for numerical precision)
            assert error < 0.1, f"Round-trip error too high: {error:.4f} pixels"


class TestReprojectionMetrics:
    """Test the reprojection error metrics from MapPointHomography."""

    def test_computation_result_metrics(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: dict[str, GroundControlPoint],
        homography_provider: MapPointHomography,
    ):
        """Test that computation result contains valid metrics."""
        result = homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        # Check metrics are computed
        assert result.num_gcps >= 4  # May have more GCPs in test data
        assert result.num_inliers >= 4  # With 4+ points, all should be inliers
        assert result.inlier_ratio == 1.0  # All points should be inliers

        # Check error metrics
        assert result.mean_reproj_error >= 0.0
        assert result.max_reproj_error >= result.mean_reproj_error
        assert result.rmse >= 0.0

        print("\nMapPointHomography Metrics:")
        print(f"  Inliers: {result.num_inliers}/{result.num_gcps}")
        print(f"  Mean reproj error: {result.mean_reproj_error:.2f}")
        print(f"  Max reproj error:  {result.max_reproj_error:.2f}")
        print(f"  RMSE:              {result.rmse:.2f}")


# =============================================================================
# Parametrized Tests - Run against all test cases
# =============================================================================


class TestAllTestCases:
    """Run precision tests against all test cases defined in valte_gcps.yaml."""

    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_homography_computation(
        self,
        test_case_name: str,
        map_registry: dict[str, GroundControlPoint],
    ):
        """Test computing homography for each test case."""
        test_case = load_gcps_from_yaml(test_case_name)
        gcps = test_case["gcps"]

        assert len(gcps) >= 4, f"[{test_case_name}] Need at least 4 GCPs"

        homography = MapPointHomography(map_id=MAP_ID)
        result = homography.compute_from_gcps(
            gcps=gcps[:4],  # Use first 4 for computation
            map_registry=map_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )

        assert result is not None, f"Failed to compute homography for {test_case_name}"
        assert result.homography_matrix.shape == (3, 3)
        assert homography.is_valid()

    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_round_trip(
        self,
        test_case_name: str,
        map_registry: dict[str, GroundControlPoint],
    ):
        """Test round-trip projection for each test case."""
        test_case = load_gcps_from_yaml(test_case_name)
        gcps = test_case["gcps"]

        homography = MapPointHomography(map_id=MAP_ID)
        homography.compute_from_gcps(
            gcps=gcps[:4],  # Use first 4 for computation
            map_registry=map_registry,
        )

        # Round-trip test on training points (should be ~0 error)
        for gcp in gcps[:4]:
            original_pixel = PixelPoint.create(gcp["pixel_x"], gcp["pixel_y"])
            map_point = homography.camera_to_map(original_pixel)
            map_as_pixel = PixelPoint.create(map_point.pixel_point.x, map_point.pixel_point.y)
            recovered_pixel = homography.map_to_camera(map_as_pixel)

            error = np.linalg.norm(
                np.array([recovered_pixel.x, recovered_pixel.y])
                - np.array([original_pixel.x, original_pixel.y])
            )

            assert error < 0.01, f"[{test_case_name}] Round-trip error too high: {error:.4f} pixels"
