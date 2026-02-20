#!/usr/bin/env python3
"""
Test homography pixel precision using MapPointHomography.

This test validates that a homography computed from annotation-GCP correspondences
achieves acceptable pixel precision when reprojecting points.

Terminology:
    - Annotation: A point marked on the camera image (camera pixels)
    - GCP (Ground Control Point): A reference point on the map (map pixels)

Annotation Format (from YAML):
    - pixel_x: Camera pixel x coordinate (annotation position)
    - pixel_y: Camera pixel y coordinate (annotation position)
    - gcp_id: ID referencing a GCP (MapPoint) in the registry

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

from poc_homography.camera_config import get_camera_by_name
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points import GCPRegistry
from poc_homography.pixel_point import PixelPoint

# =============================================================================
# Test Data Paths - Update these to point to your test data
# =============================================================================
TEST_DATA_DIR = Path(__file__).parent / "test_data"
MAP_POINTS_FILE = TEST_DATA_DIR / "Cartografia_valencia_gcps.yaml"
ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"

# Valte camera geotransform: 0.15 m/pixel in map space
# GeoTransform: [origin_easting, pixel_width, row_rotation, origin_northing, col_rotation, pixel_height]
VALTE_CONFIG = get_camera_by_name("Valte")
MAP_METERS_PER_PIXEL = (
    abs(VALTE_CONFIG["geotiff_params"]["geotransform"][1]) if VALTE_CONFIG else 0.15
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def map_registry() -> GCPRegistry:
    """
    Load map point registry from file.

    The registry contains MapPoints with pixel_x/pixel_y coordinates
    (which may represent UTM or other map coordinates).
    """
    if not MAP_POINTS_FILE.exists():
        pytest.skip(f"Map points file not found: {MAP_POINTS_FILE}")

    return GCPRegistry.load(MAP_POINTS_FILE)


def load_all_test_cases() -> list[dict[str, Any]]:
    """Load all test cases from YAML file."""
    if not ANNOTATIONS_FILE.exists():
        return []

    try:
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("test_cases", []) if data else []
    except (yaml.YAMLError, OSError):
        return []


def get_test_case_names() -> list[str]:
    """Get list of all test case names from YAML file."""
    return [tc["name"] for tc in load_all_test_cases()]


def load_annotations_from_yaml(test_case_name: str | None = None) -> dict[str, Any]:
    """
    Load annotations (camera pixels) and their corresponding GCP IDs from YAML file.

    Args:
        test_case_name: Name of the test case to load. If None, loads the first one.

    Returns:
        Dictionary with 'image' and 'annotations' keys.
    """
    if not ANNOTATIONS_FILE.exists():
        pytest.skip(f"Annotations file not found: {ANNOTATIONS_FILE}")

    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", []) if data else []
    if not test_cases:
        pytest.skip("No test cases found in annotations file")

    if test_case_name:
        for tc in test_cases:
            if tc.get("name") == test_case_name:
                return tc
        pytest.skip(f"Test case '{test_case_name}' not found in annotations file")

    return test_cases[0]


@pytest.fixture
def annotations_test_case() -> dict[str, Any]:
    """Load the first test case from annotations YAML file."""
    return load_annotations_from_yaml()


@pytest.fixture
def annotations_4_points(annotations_test_case: dict[str, Any]) -> list[dict[str, Any]]:
    """
    4 annotations for homography computation, loaded from YAML.

    Each annotation contains:
        - pixel_x: x coordinate in camera image (pixels)
        - pixel_y: y coordinate in camera image (pixels)
        - gcp_id: ID of corresponding GCP (MapPoint) in registry
    """
    return annotations_test_case["annotations"]


@pytest.fixture
def test_image_path(annotations_test_case: dict[str, Any]) -> Path:
    """Path to the test image from the annotations YAML file (relative to YAML location)."""
    return ANNOTATIONS_FILE.parent / annotations_test_case["image"]


@pytest.fixture
def homography_provider(map_registry: GCPRegistry) -> MapPointHomography:
    """Create a MapPointHomography instance."""
    return MapPointHomography(map_id=map_registry.map_id)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_pixel_precision(
    annotations: list[dict[str, Any]],
    homography: MapPointHomography,
    map_registry: GCPRegistry,
) -> dict[str, float]:
    """
    Measure pixel precision by reprojecting annotations.

    Process:
        1. Get GCP (map) coordinates for each annotation from registry
        2. Project GCP coordinates back to camera pixels using inverse homography
        3. Compare with original annotation pixel coordinates

    Args:
        annotations: List of annotations with pixel_x, pixel_y, gcp_id
        homography: Computed MapPointHomography
        map_registry: Registry containing GCPs (MapPoints)

    Returns:
        Dictionary with mean_error, max_error, rmse (all in pixels)
    """
    errors = []

    for annotation in annotations:
        # Get the GCP (map) coordinate from registry
        gcp = map_registry.points[annotation["gcp_id"]]

        # Project GCP coordinate back to camera pixel
        gcp_coord = PixelPoint(gcp.pixel_x, gcp.pixel_y)
        projected_pixel = homography.map_to_camera(gcp_coord)

        # Compare with original annotation pixel coordinate
        original = np.array([annotation["pixel_x"], annotation["pixel_y"]])
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


def compute_metric_precision(
    annotations: list[dict[str, Any]],
    homography: MapPointHomography,
    map_registry: GCPRegistry,
    meters_per_pixel: float = MAP_METERS_PER_PIXEL,
) -> dict[str, float]:
    """
    Measure precision in METERS by calculating error in map space.

    This is the CORRECT way to compute reprojection error for metric accuracy.
    Camera pixels have variable metric size due to perspective distortion,
    but map pixels have constant GSD (ground sample distance).

    Process:
        1. Project camera pixel coordinates to map space using homography
        2. Compare with known map coordinates from registry
        3. Calculate Euclidean distance in map pixels
        4. Convert to meters using geotransform GSD

    Args:
        annotations: List of annotations with pixel_x, pixel_y, gcp_id
        homography: Computed MapPointHomography
        map_registry: Registry containing MapPoints
        meters_per_pixel: Ground sample distance (m/pixel) from geotransform

    Returns:
        Dictionary with mean_error, max_error, rmse (all in meters),
        and per_point details including map pixel errors
    """
    errors_meters = []
    errors_map_pixels = []
    per_point_details = []

    for annotation in annotations:
        # Get camera pixel coordinates
        camera_pixel = PixelPoint(annotation["pixel_x"], annotation["pixel_y"])

        # Project camera pixel to map space using homography
        projected_map = homography.camera_to_map(camera_pixel)

        # Get the known map coordinate from registry
        known_map = map_registry.points[annotation["gcp_id"]]

        # Calculate error in MAP pixels (constant m/pixel)
        projected = np.array([projected_map.pixel_x, projected_map.pixel_y])
        actual = np.array([known_map.pixel_x, known_map.pixel_y])
        error_map_px = float(np.linalg.norm(projected - actual))

        # Convert to meters using GSD
        error_meters = error_map_px * meters_per_pixel

        errors_map_pixels.append(error_map_px)
        errors_meters.append(error_meters)
        per_point_details.append(
            {
                "gcp_id": annotation["gcp_id"],
                "error_map_px": error_map_px,
                "error_meters": error_meters,
                "projected_map": (projected_map.pixel_x, projected_map.pixel_y),
                "actual_map": (known_map.pixel_x, known_map.pixel_y),
            }
        )

    errors_array = np.array(errors_meters)

    return {
        "mean_error_m": float(np.mean(errors_array)),
        "max_error_m": float(np.max(errors_array)),
        "rmse_m": float(np.sqrt(np.mean(errors_array**2))),
        "mean_error_map_px": float(np.mean(errors_map_pixels)),
        "per_point_details": per_point_details,
    }


# =============================================================================
# Tests
# =============================================================================


class TestMapPointHomographyComputation:
    """Test homography computation using MapPointHomography."""

    def test_compute_homography_from_4_annotations(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test computing homography from exactly 4 annotation-GCP correspondences."""
        result = homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )

        assert result is not None
        assert result.homography_matrix.shape == (3, 3)
        assert result.inverse_matrix.shape == (3, 3)
        assert result.num_gcps >= 4  # May have more annotations in test data
        assert homography_provider.is_valid()

    def test_homography_matrix_is_invertible(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that the homography matrix is invertible."""
        result = homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
        )

        H = result.homography_matrix
        H_inv = result.inverse_matrix

        # H * H_inv should be identity
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6)


class TestPixelPrecision:
    """Test pixel precision of homography reprojection."""

    def test_pixel_precision_with_4_annotations(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test pixel precision when reprojecting annotations."""
        # Compute homography
        result = homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
            ransac_threshold=50.0,
        )

        # Measure precision
        precision = compute_pixel_precision(
            annotations=annotations_4_points,
            homography=homography_provider,
            map_registry=map_registry,
        )

        print("\nPixel Precision Metrics:")
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

    @pytest.mark.xfail(
        reason="Sub-pixel precision requires higher quality annotation data than currently available",
        strict=False,
    )
    def test_sub_pixel_precision(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that homography achieves sub-pixel precision on annotations."""
        homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
        )

        precision = compute_pixel_precision(
            annotations=annotations_4_points,
            homography=homography_provider,
            map_registry=map_registry,
        )

        # Check each point has sub-pixel error
        for i, error in enumerate(precision["per_point_errors"]):
            assert error < 1.0, (
                f"Annotation {i} does not have sub-pixel precision: {error:.4f} pixels"
            )


class TestMetricPrecision:
    """
    Test precision in METERS using correct map-space error calculation.

    Camera pixels have variable metric size due to perspective distortion.
    Error must be calculated in map space where GSD is constant.
    """

    def test_metric_precision_with_all_annotations(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test metric precision when reprojecting all annotations."""
        # Compute homography using ALL annotations (overdetermined system)
        result = homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
            ransac_threshold=50.0,
        )

        # Measure precision in METERS
        precision = compute_metric_precision(
            annotations=annotations_4_points,
            homography=homography_provider,
            map_registry=map_registry,
        )

        print("\nMetric Precision (in meters):")
        print(f"  Mean error: {precision['mean_error_m']:.2f} m")
        print(f"  Max error:  {precision['max_error_m']:.2f} m")
        print(f"  RMSE:       {precision['rmse_m']:.2f} m")
        print("\nPer-point errors:")
        for detail in sorted(precision["per_point_details"], key=lambda x: -x["error_meters"]):
            print(
                f"  {detail['gcp_id']}: {detail['error_meters']:.2f} m ({detail['error_map_px']:.1f} map px)"
            )

        # Assert metric thresholds (sub-meter accuracy is good for surveying)
        assert precision["mean_error_m"] < 2.0, (
            f"Mean error too high: {precision['mean_error_m']:.2f} m"
        )
        assert precision["max_error_m"] < 5.0, (
            f"Max error too high: {precision['max_error_m']:.2f} m"
        )

    @pytest.mark.xfail(
        reason="Test combines annotations from different camera poses which invalidates holdout validation"
    )
    def test_holdout_metric_precision(
        self,
        map_registry: GCPRegistry,
    ):
        """
        Test metric precision using N-1 (leave-one-out) validation.

        Collects ALL unique GCPs from ALL test cases, then for each GCP,
        computes homography from ALL OTHER GCPs and measures error on
        the held-out GCP.

        NOTE: This test is fundamentally flawed when test cases have different
        camera poses (pan/tilt angles). Homography is pose-specific, so combining
        annotations from different poses and doing holdout validation doesn't
        make mathematical sense.

        This provides much better coverage than single test case validation.
        """
        # Collect ALL unique annotations from ALL test cases
        all_test_cases = load_all_test_cases()
        unique_annotations = {}  # gcp_id -> annotation dict
        for test_case in all_test_cases:
            for annotation in test_case["annotations"]:
                # Keep the annotation if not seen, or if coordinates match
                gcp_id = annotation["gcp_id"]
                if gcp_id not in unique_annotations:
                    unique_annotations[gcp_id] = annotation

        annotations = list(unique_annotations.values())

        if len(annotations) < 5:
            pytest.skip("Need at least 5 unique annotations for holdout validation")

        print(
            f"\nUsing {len(annotations)} unique annotations from {len(all_test_cases)} test cases"
        )

        holdout_errors = []

        for i, holdout_annotation in enumerate(annotations):
            # Train on all annotations except holdout
            train_annotations = annotations[:i] + annotations[i + 1 :]

            homography = MapPointHomography(map_id=map_registry.map_id)
            homography.compute_from_gcps(
                gcps=train_annotations,  # annotations passed as gcps to the API
                map_registry=map_registry,
                ransac_threshold=50.0,
            )

            # Measure error on HOLDOUT annotation only
            precision = compute_metric_precision(
                annotations=[holdout_annotation],
                homography=homography,
                map_registry=map_registry,
            )

            holdout_errors.append(
                {
                    "gcp_id": holdout_annotation["gcp_id"],
                    "error_m": precision["mean_error_m"],
                }
            )

        # Sort by error for display
        holdout_errors.sort(key=lambda x: -x["error_m"])

        mean_error = np.mean([e["error_m"] for e in holdout_errors])
        max_error = max(e["error_m"] for e in holdout_errors)

        print("\nN-1 Holdout Validation (Metric):")
        print(f"  Mean holdout error: {mean_error:.2f} m")
        print(f"  Max holdout error:  {max_error:.2f} m")
        print("\nPer-point holdout errors:")
        for e in holdout_errors:
            print(f"  {e['gcp_id']}: {e['error_m']:.2f} m")

        # N-1 validation should achieve reasonable accuracy
        assert mean_error < 2.0, f"Mean holdout error too high: {mean_error:.2f} m"
        assert max_error < 5.0, f"Max holdout error too high: {max_error:.2f} m"


class TestRoundTrip:
    """Test round-trip projection: camera -> map -> camera."""

    def test_round_trip_camera_to_map_to_camera(
        self,
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that round-trip projection preserves coordinates."""
        homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
        )

        for annotation in annotations_4_points:
            # Original camera pixel (annotation)
            original_pixel = PixelPoint(annotation["pixel_x"], annotation["pixel_y"])

            # Camera -> Map (annotation -> GCP space)
            gcp_point = homography_provider.camera_to_map(original_pixel)

            # Map -> Camera (GCP space -> annotation)
            gcp_as_pixel = PixelPoint(gcp_point.pixel_x, gcp_point.pixel_y)
            recovered_pixel = homography_provider.map_to_camera(gcp_as_pixel)

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
        annotations_4_points: list[dict[str, Any]],
        map_registry: GCPRegistry,
        homography_provider: MapPointHomography,
    ):
        """Test that computation result contains valid metrics."""
        result = homography_provider.compute_from_gcps(
            gcps=annotations_4_points,  # annotations passed as gcps to the API
            map_registry=map_registry,
        )

        # Check metrics are computed
        assert result.num_gcps >= 4  # May have more annotations in test data
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
# Parametrized Tests - Run against all test cases with HOLDOUT VALIDATION
# =============================================================================


class TestAllTestCases:
    """
    Run precision tests against all test cases defined in valte_gcps.yaml.

    Uses holdout validation: randomly select 4 annotations to compute homography,
    validate against the remaining annotation(s) that were NOT used in computation.
    This ensures we're testing actual annotation-GCP correspondence correctness,
    not just that a homography can fit 4 points (which it always can).
    """

    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_homography_computation(
        self,
        test_case_name: str,
        map_registry: GCPRegistry,
    ):
        """Test computing homography for each test case."""
        test_case = load_annotations_from_yaml(test_case_name)
        annotations = test_case["annotations"]

        assert len(annotations) >= 4, f"[{test_case_name}] Need at least 4 annotations"

        homography = MapPointHomography(map_id=map_registry.map_id)
        result = homography.compute_from_gcps(
            gcps=annotations[:4],  # Use first 4 for computation
            map_registry=map_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )

        assert result is not None, f"Failed to compute homography for {test_case_name}"
        assert result.homography_matrix.shape == (3, 3)
        assert homography.is_valid()

    @pytest.mark.xfail(
        reason="Holdout validation requires higher quality annotation data than currently available",
        strict=False,
    )
    @pytest.mark.parametrize("test_case_name", get_test_case_names())
    def test_holdout_validation(
        self,
        test_case_name: str,
        map_registry: GCPRegistry,
    ):
        """
        Test pixel precision using HOLDOUT validation.

        Randomly selects 4 annotations to compute homography, then validates
        against the remaining annotation(s) not used in computation.
        """
        test_case = load_annotations_from_yaml(test_case_name)
        annotations = test_case["annotations"]

        assert len(annotations) >= 5, (
            f"[{test_case_name}] Need at least 5 annotations for holdout validation, got {len(annotations)}"
        )

        # Randomly select 4 annotations for computation, rest for validation
        # Use fixed seed for reproducibility (based on test case name)
        random.seed(hash(test_case_name) % (2**32))
        annotations_shuffled = annotations.copy()
        random.shuffle(annotations_shuffled)

        train_annotations = annotations_shuffled[:4]
        holdout_annotations = annotations_shuffled[4:]

        # Compute homography with training annotations only
        homography = MapPointHomography(map_id=map_registry.map_id)
        homography.compute_from_gcps(
            gcps=train_annotations,
            map_registry=map_registry,
            ransac_threshold=50.0,
        )

        # Validate against HOLDOUT annotations (not used in computation)
        precision = compute_pixel_precision(
            annotations=holdout_annotations,
            homography=homography,
            map_registry=map_registry,
        )

        train_ids = [a["gcp_id"] for a in train_annotations]
        holdout_ids = [a["gcp_id"] for a in holdout_annotations]

        print(f"\n[{test_case_name}] Holdout Validation:")
        print(f"  Training annotations (GCP IDs): {train_ids}")
        print(f"  Holdout annotations (GCP IDs):  {holdout_ids}")
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
        map_registry: GCPRegistry,
    ):
        """Test round-trip projection for each test case."""
        test_case = load_annotations_from_yaml(test_case_name)
        annotations = test_case["annotations"]

        homography = MapPointHomography(map_id=map_registry.map_id)
        homography.compute_from_gcps(
            gcps=annotations[:4],  # Use first 4 for computation
            map_registry=map_registry,
        )

        # Round-trip test on training points (should be ~0 error)
        for annotation in annotations[:4]:
            original_pixel = PixelPoint(annotation["pixel_x"], annotation["pixel_y"])
            gcp_point = homography.camera_to_map(original_pixel)
            gcp_as_pixel = PixelPoint(gcp_point.pixel_x, gcp_point.pixel_y)
            recovered_pixel = homography.map_to_camera(gcp_as_pixel)

            error = np.linalg.norm(
                np.array([recovered_pixel.x, recovered_pixel.y])
                - np.array([original_pixel.x, original_pixel.y])
            )

            assert error < 0.01, f"[{test_case_name}] Round-trip error too high: {error:.4f} pixels"
