#!/usr/bin/env python3
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

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from poc_homography.homography.map_points import MapPointHomography, MapPointComputationResult
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint


# =============================================================================
# Test Data Paths - Update these to point to your test data
# =============================================================================
TEST_DATA_DIR = Path(__file__).parent.parent.parent
MAP_POINTS_FILE = TEST_DATA_DIR / "valte_map_points.yaml"
GCPS_FILE = Path(__file__).parent / "test_data" / "valte_gcps.yaml"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def map_registry() -> MapPointRegistry:
    """
    Load map point registry from file.

    The registry contains MapPoints with pixel_x/pixel_y coordinates
    (which may represent UTM or other map coordinates).
    """
    if not MAP_POINTS_FILE.exists():
        pytest.skip(f"Map points file not found: {MAP_POINTS_FILE}")

    return MapPointRegistry.load(MAP_POINTS_FILE)


def load_all_test_cases() -> list[dict[str, Any]]:
    """Load all test cases from YAML file."""
    if not GCPS_FILE.exists():
        return []

    with open(GCPS_FILE) as f:
        data = yaml.safe_load(f)

    return data.get("test_cases", [])


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

    with open(GCPS_FILE) as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
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
def homography_provider(map_registry: MapPointRegistry) -> MapPointHomography:
    """Create a MapPointHomography instance."""
    return MapPointHomography(map_id=map_registry.map_id)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_pixel_precision(
    gcps: list[dict[str, Any]],
    homography: MapPointHomography,
    map_registry: MapPointRegistry,
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
        map_point = map_registry.points[gcp["map_point_id"]]

        # Project map coordinate back to camera pixel
        map_coord = PixelPoint(map_point.pixel_x, map_point.pixel_y)
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
        "rmse": float(np.sqrt(np.mean(errors_array ** 2))),
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
        map_registry: MapPointRegistry,
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
        assert result.num_gcps == 4
        assert homography_provider.is_valid()

    def test_homography_matrix_is_invertible(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: MapPointRegistry,
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
        map_registry: MapPointRegistry,
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

        print(f"\nPixel Precision Metrics:")
        print(f"  Mean error: {precision['mean_error']:.2f} pixels")
        print(f"  Max error:  {precision['max_error']:.2f} pixels")
        print(f"  RMSE:       {precision['rmse']:.2f} pixels")

        # Assert precision thresholds
        # Adjust these based on expected accuracy
        assert precision["mean_error"] < 10.0, (
            f"Mean error too high: {precision['mean_error']:.2f} pixels"
        )
        assert precision["max_error"] < 20.0, (
            f"Max error too high: {precision['max_error']:.2f} pixels"
        )

    def test_sub_pixel_precision(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: MapPointRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that homography achieves sub-pixel precision on GCPs."""
        homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        precision = compute_pixel_precision(
            gcps=gcps_4_points,
            homography=homography_provider,
            map_registry=map_registry,
        )

        # Check each point has sub-pixel error
        for i, error in enumerate(precision["per_point_errors"]):
            assert error < 1.0, (
                f"GCP {i} does not have sub-pixel precision: {error:.4f} pixels"
            )


class TestRoundTrip:
    """Test round-trip projection: camera -> map -> camera."""

    def test_round_trip_camera_to_map_to_camera(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: MapPointRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that round-trip projection preserves coordinates."""
        homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        for gcp in gcps_4_points:
            # Original camera pixel
            original_pixel = PixelPoint(gcp["pixel_x"], gcp["pixel_y"])

            # Camera -> Map
            map_point = homography_provider.camera_to_map(original_pixel)

            # Map -> Camera
            map_as_pixel = PixelPoint(map_point.pixel_x, map_point.pixel_y)
            recovered_pixel = homography_provider.map_to_camera(map_as_pixel)

            # Compare
            error = np.linalg.norm(
                np.array([recovered_pixel.x, recovered_pixel.y]) -
                np.array([original_pixel.x, original_pixel.y])
            )

            assert error < 0.01, f"Round-trip error too high: {error:.4f} pixels"


class TestReprojectionMetrics:
    """Test the reprojection error metrics from MapPointHomography."""

    def test_computation_result_metrics(
        self,
        gcps_4_points: list[dict[str, Any]],
        map_registry: MapPointRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that computation result contains valid metrics."""
        result = homography_provider.compute_from_gcps(
            gcps=gcps_4_points,
            map_registry=map_registry,
        )

        # Check metrics are computed
        assert result.num_gcps == 4
        assert result.num_inliers >= 4  # With 4 points, all should be inliers
        assert result.inlier_ratio == 1.0  # 4/4 = 100%

        # Check error metrics
        assert result.mean_reproj_error >= 0.0
        assert result.max_reproj_error >= result.mean_reproj_error
        assert result.rmse >= 0.0

        print(f"\nMapPointHomography Metrics:")
        print(f"  Inliers: {result.num_inliers}/{result.num_gcps}")
        print(f"  Mean reproj error: {result.mean_reproj_error:.2f}")
        print(f"  Max reproj error:  {result.max_reproj_error:.2f}")
        print(f"  RMSE:              {result.rmse:.2f}")


# =============================================================================
# Parametrized Tests - Run against all test cases with HOLDOUT VALIDATION
# =============================================================================


class TestAllTestCases:
    """
    Run precision tests against all test cases defined in valte_gcps.yaml.

    Uses holdout validation: randomly select 4 GCPs to compute homography,
    validate against the remaining GCP(s) that were NOT used in computation.
    This ensures we're testing actual GCP correctness, not just that a
    homography can fit 4 points (which it always can).
    """

    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_homography_computation(
        self,
        test_case_name: str,
        map_registry: MapPointRegistry,
    ):
        """Test computing homography for each test case."""
        test_case = load_gcps_from_yaml(test_case_name)
        gcps = test_case["gcps"]

        assert len(gcps) >= 4, f"[{test_case_name}] Need at least 4 GCPs"

        homography = MapPointHomography(map_id=map_registry.map_id)
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
    def test_holdout_validation(
        self,
        test_case_name: str,
        map_registry: MapPointRegistry,
    ):
        """
        Test pixel precision using HOLDOUT validation.

        Randomly selects 4 GCPs to compute homography, then validates
        against the remaining GCP(s) not used in computation.
        """
        test_case = load_gcps_from_yaml(test_case_name)
        gcps = test_case["gcps"]

        assert len(gcps) >= 5, (
            f"[{test_case_name}] Need at least 5 GCPs for holdout validation, got {len(gcps)}"
        )

        # Randomly select 4 GCPs for computation, rest for validation
        gcps_shuffled = gcps.copy()
        random.shuffle(gcps_shuffled)

        train_gcps = gcps_shuffled[:4]
        holdout_gcps = gcps_shuffled[4:]

        # Compute homography with training GCPs only
        homography = MapPointHomography(map_id=map_registry.map_id)
        homography.compute_from_gcps(
            gcps=train_gcps,
            map_registry=map_registry,
            ransac_threshold=50.0,
        )

        # Validate against HOLDOUT GCPs (not used in computation)
        precision = compute_pixel_precision(
            gcps=holdout_gcps,
            homography=homography,
            map_registry=map_registry,
        )

        train_ids = [g["map_point_id"] for g in train_gcps]
        holdout_ids = [g["map_point_id"] for g in holdout_gcps]

        print(f"\n[{test_case_name}] Holdout Validation:")
        print(f"  Training GCPs: {train_ids}")
        print(f"  Holdout GCPs:  {holdout_ids}")
        print(f"  Mean error: {precision['mean_error']:.2f} pixels")
        print(f"  Max error:  {precision['max_error']:.2f} pixels")
        print(f"  RMSE:       {precision['rmse']:.2f} pixels")

        assert precision["mean_error"] < 10.0, (
            f"[{test_case_name}] Holdout mean error too high: {precision['mean_error']:.2f} pixels"
        )
        assert precision["max_error"] < 20.0, (
            f"[{test_case_name}] Holdout max error too high: {precision['max_error']:.2f} pixels"
        )

    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_round_trip(
        self,
        test_case_name: str,
        map_registry: MapPointRegistry,
    ):
        """Test round-trip projection for each test case."""
        test_case = load_gcps_from_yaml(test_case_name)
        gcps = test_case["gcps"]

        homography = MapPointHomography(map_id=map_registry.map_id)
        homography.compute_from_gcps(
            gcps=gcps[:4],  # Use first 4 for computation
            map_registry=map_registry,
        )

        # Round-trip test on training points (should be ~0 error)
        for gcp in gcps[:4]:
            original_pixel = PixelPoint(gcp["pixel_x"], gcp["pixel_y"])
            map_point = homography.camera_to_map(original_pixel)
            map_as_pixel = PixelPoint(map_point.pixel_x, map_point.pixel_y)
            recovered_pixel = homography.map_to_camera(map_as_pixel)

            error = np.linalg.norm(
                np.array([recovered_pixel.x, recovered_pixel.y]) -
                np.array([original_pixel.x, original_pixel.y])
            )

            assert error < 0.01, (
                f"[{test_case_name}] Round-trip error too high: {error:.4f} pixels"
            )
