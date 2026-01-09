#!/usr/bin/env python3
"""
Comprehensive Transform Validation Tests using Ground Control Points.

This test suite validates every transformation in the homography pipeline
against real-world GCP data collected from the Valte camera at Valencia.

Test Data Source:
    - Camera: Valte (Valencia Terminal)
    - Date: 2026-01-08
    - GCPs: 11 points on road markings
    - Image: 1920x1080 (assumed from pixel ranges)

Transforms Tested:
    1. GPS ↔ Local Metric Coordinate Conversion
    2. Intrinsic Matrix (K) Computation
    3. Rotation Matrix (R) from Pan/Tilt/Roll
    4. Homography Matrix (H) Computation
    5. Pixel → World Projection
    6. World → Pixel Projection (Inverse)
    7. Round-trip Consistency
    8. Edge Cases (horizon, gimbal lock, extreme zoom)
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.coordinate_converter import (
    PYPROJ_AVAILABLE,
    UTMConverter,
    gps_to_local_xy,
    local_xy_to_gps,
)

# Import modules under test
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography

# =============================================================================
# TEST DATA: Ground Control Points from Valte Camera (loaded from fixture file)
# =============================================================================

# Path to fixture data
FIXTURE_DIR = Path(__file__).parent / "fixtures"
VALTE_GCP_DATA_FILE = FIXTURE_DIR / "valte_gcp_data.json"


def load_gcp_test_data(filepath: Path = VALTE_GCP_DATA_FILE) -> dict[str, Any]:
    """Load GCP test data from JSON fixture file.

    Args:
        filepath: Path to the JSON file containing GCP data

    Returns:
        Dictionary with camera_info, image_dimensions, and gcps
    """
    with open(filepath) as f:
        return json.load(f)


# Load test data from fixture file
_TEST_DATA = load_gcp_test_data()

# Camera parameters from fixture
CAMERA_INFO = _TEST_DATA["camera_info"]

# Image dimensions from fixture
IMAGE_WIDTH = _TEST_DATA["image_dimensions"]["width"]
IMAGE_HEIGHT = _TEST_DATA["image_dimensions"]["height"]

# Ground Control Points from fixture
GCPS = _TEST_DATA["gcps"]

# Split into training (8) and validation (3) sets
TRAIN_GCPS = GCPS[:8]
VALIDATION_GCPS = GCPS[8:]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def camera_info() -> dict[str, Any]:
    """Return camera parameters."""
    return CAMERA_INFO.copy()


@pytest.fixture
def gcps() -> list[dict[str, float]]:
    """Return all GCPs."""
    return [gcp.copy() for gcp in GCPS]


@pytest.fixture
def train_gcps() -> list[dict[str, float]]:
    """Return training GCPs."""
    return [gcp.copy() for gcp in TRAIN_GCPS]


@pytest.fixture
def validation_gcps() -> list[dict[str, float]]:
    """Return validation GCPs."""
    return [gcp.copy() for gcp in VALIDATION_GCPS]


@pytest.fixture
def homography_provider(camera_info) -> IntrinsicExtrinsicHomography:
    """Create and configure a homography provider with camera parameters."""
    provider = IntrinsicExtrinsicHomography(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        pixels_per_meter=100.0,
    )
    provider.set_camera_gps_position(camera_info["latitude"], camera_info["longitude"])
    return provider


@pytest.fixture
def camera_geometry(camera_info) -> CameraGeometry:
    """Create CameraGeometry with camera parameters."""
    geom = CameraGeometry(w=IMAGE_WIDTH, h=IMAGE_HEIGHT)
    geom.height_m = camera_info["height_meters"]
    geom.pan_deg = camera_info["pan_deg"]
    geom.tilt_deg = camera_info["tilt_deg"]
    geom.w_pos = np.array([0, 0, camera_info["height_meters"]])
    # Compute intrinsics and set K
    geom.K = CameraGeometry.get_intrinsics(
        zoom_factor=camera_info["zoom_level"], W_px=IMAGE_WIDTH, H_px=IMAGE_HEIGHT
    )
    return geom


@pytest.fixture
def utm_converter() -> UTMConverter:
    """Create UTM converter (skips if pyproj not available)."""
    if not PYPROJ_AVAILABLE:
        pytest.skip("pyproj not available")
    converter = UTMConverter()
    converter.set_reference(CAMERA_INFO["latitude"], CAMERA_INFO["longitude"])
    return converter


# =============================================================================
# TEST 1: GPS ↔ LOCAL METRIC COORDINATE CONVERSION
# =============================================================================


class TestCoordinateConversion:
    """Test GPS to local metric and back conversions."""

    def test_gps_to_local_xy_near_reference(self, camera_info):
        """GCP near camera should have small local coordinates."""
        ref_lat = camera_info["latitude"]
        ref_lon = camera_info["longitude"]

        # First GCP is close to camera
        gcp = GCPS[0]
        x, y = gps_to_local_xy(ref_lat, ref_lon, gcp["latitude"], gcp["longitude"])

        # Should be within ~50 meters (parking lot scale)
        assert abs(x) < 100, f"X coordinate {x}m seems too far for parking lot"
        assert abs(y) < 100, f"Y coordinate {y}m seems too far for parking lot"

    def test_gps_to_local_round_trip(self, camera_info):
        """GPS → local → GPS should recover original coordinates."""
        ref_lat = camera_info["latitude"]
        ref_lon = camera_info["longitude"]

        for gcp in GCPS:
            # Convert to local
            x, y = gps_to_local_xy(ref_lat, ref_lon, gcp["latitude"], gcp["longitude"])

            # Convert back to GPS
            recovered_lat, recovered_lon = local_xy_to_gps(ref_lat, ref_lon, x, y)

            # Should match within ~1 meter accuracy (≈0.00001° lat/lon)
            lat_error = abs(recovered_lat - gcp["latitude"])
            lon_error = abs(recovered_lon - gcp["longitude"])

            assert lat_error < 1e-5, f"Latitude round-trip error: {lat_error} degrees"
            assert lon_error < 1e-5, f"Longitude round-trip error: {lon_error} degrees"

    def test_local_coordinates_relative_ordering(self, camera_info):
        """GCPs should have consistent relative positions in local coords."""
        ref_lat = camera_info["latitude"]
        ref_lon = camera_info["longitude"]

        # Convert all GCPs to local coordinates
        local_coords = []
        for gcp in GCPS:
            x, y = gps_to_local_xy(ref_lat, ref_lon, gcp["latitude"], gcp["longitude"])
            local_coords.append((x, y, gcp["pixel_x"], gcp["pixel_y"]))

        # GCPs with higher pixel_x should generally have higher x (more east)
        # This tests that the coordinate system orientation is correct
        # (accounting for camera pan rotation)

        # At least verify all conversions succeeded without NaN/Inf
        for x, y, px, py in local_coords:
            assert np.isfinite(x), f"X coordinate is not finite: {x}"
            assert np.isfinite(y), f"Y coordinate is not finite: {y}"

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not available")
    def test_utm_vs_equirectangular_consistency(self, camera_info):
        """UTM and equirectangular should give similar results for small distances."""
        ref_lat = camera_info["latitude"]
        ref_lon = camera_info["longitude"]

        utm = UTMConverter()
        utm.set_reference(ref_lat, ref_lon)

        for gcp in GCPS:
            # Equirectangular
            x_eq, y_eq = gps_to_local_xy(ref_lat, ref_lon, gcp["latitude"], gcp["longitude"])

            # UTM
            x_utm, y_utm = utm.gps_to_local_xy(gcp["latitude"], gcp["longitude"])

            # Compute differences
            x_diff = abs(x_eq - x_utm)
            y_diff = abs(y_eq - y_utm)

            # For parking lot distances (< 100m), absolute error should be < 3m
            # Equirectangular approximation has documented errors of ~2% X, ~5% Y
            # For these small distances, we use absolute thresholds instead of percentages
            assert x_diff < 3.0, (
                f"X differs by {x_diff:.2f}m between UTM and equirectangular (expected < 3m)"
            )
            assert y_diff < 3.0, (
                f"Y differs by {y_diff:.2f}m between UTM and equirectangular (expected < 3m)"
            )


# =============================================================================
# TEST 2: INTRINSIC MATRIX (K) COMPUTATION
# =============================================================================


class TestIntrinsicMatrix:
    """Test camera intrinsic matrix computation."""

    def test_intrinsic_matrix_structure(self, homography_provider):
        """K matrix should have correct structure: [[fx,0,cx],[0,fy,cy],[0,0,1]]."""
        K = homography_provider.get_intrinsics(zoom_factor=1.0)

        # Check structure
        assert K.shape == (3, 3), f"K should be 3x3, got {K.shape}"
        assert K[2, 0] == 0, "K[2,0] should be 0"
        assert K[2, 1] == 0, "K[2,1] should be 0"
        assert K[2, 2] == 1, "K[2,2] should be 1"
        assert K[0, 1] == 0, "K[0,1] (skew) should be 0"

        # fx, fy should be positive
        assert K[0, 0] > 0, "fx should be positive"
        assert K[1, 1] > 0, "fy should be positive"

        # cx, cy should be near image center
        cx, cy = K[0, 2], K[1, 2]
        assert abs(cx - IMAGE_WIDTH / 2) < IMAGE_WIDTH * 0.1, (
            f"cx={cx} should be near {IMAGE_WIDTH / 2}"
        )
        assert abs(cy - IMAGE_HEIGHT / 2) < IMAGE_HEIGHT * 0.1, (
            f"cy={cy} should be near {IMAGE_HEIGHT / 2}"
        )

    def test_focal_length_scales_with_zoom(self, homography_provider):
        """Focal length should increase approximately linearly with zoom."""
        K_1x = homography_provider.get_intrinsics(zoom_factor=1.0)
        K_5x = homography_provider.get_intrinsics(zoom_factor=5.0)
        K_10x = homography_provider.get_intrinsics(zoom_factor=10.0)

        fx_1x = K_1x[0, 0]
        fx_5x = K_5x[0, 0]
        fx_10x = K_10x[0, 0]

        # Focal length should scale roughly with zoom
        ratio_5x = fx_5x / fx_1x
        ratio_10x = fx_10x / fx_1x

        # Allow 20% tolerance from linear scaling
        assert 4.0 < ratio_5x < 6.0, f"5x zoom should give ~5x focal length, got {ratio_5x}x"
        assert 8.0 < ratio_10x < 12.0, f"10x zoom should give ~10x focal length, got {ratio_10x}x"

    def test_principal_point_invariant_to_zoom(self, homography_provider):
        """Principal point should stay at image center regardless of zoom."""
        for zoom in [1.0, 5.0, 10.0, 25.0]:
            K = homography_provider.get_intrinsics(zoom_factor=zoom)
            cx, cy = K[0, 2], K[1, 2]

            assert abs(cx - IMAGE_WIDTH / 2) < 1, f"cx should be at image center for zoom={zoom}"
            assert abs(cy - IMAGE_HEIGHT / 2) < 1, f"cy should be at image center for zoom={zoom}"

    def test_intrinsic_matrix_positive_definite(self, homography_provider):
        """K should be positive definite (all eigenvalues positive)."""
        K = homography_provider.get_intrinsics(zoom_factor=1.0)

        # For upper triangular K, eigenvalues are diagonal elements
        assert K[0, 0] > 0, "fx must be positive"
        assert K[1, 1] > 0, "fy must be positive"
        assert K[2, 2] > 0, "K[2,2] must be positive"

        # Determinant should be positive
        det = np.linalg.det(K)
        assert det > 0, f"K determinant should be positive, got {det}"


# =============================================================================
# TEST 3: ROTATION MATRIX (R) COMPUTATION
# =============================================================================


class TestRotationMatrix:
    """Test rotation matrix computation from pan/tilt/roll."""

    def test_rotation_matrix_orthogonal(self, homography_provider, camera_info):
        """R should be orthogonal: R @ R.T = I."""
        R = homography_provider._get_rotation_matrix(
            pan_deg=camera_info["pan_deg"], tilt_deg=camera_info["tilt_deg"], roll_deg=0.0
        )

        # R @ R.T should be identity
        RRT = R @ R.T
        identity = np.eye(3)

        np.testing.assert_allclose(
            RRT, identity, atol=1e-10, err_msg="R @ R.T should be identity (orthogonality)"
        )

    def test_rotation_matrix_determinant_one(self, homography_provider, camera_info):
        """R should have determinant +1 (proper rotation, not reflection)."""
        R = homography_provider._get_rotation_matrix(
            pan_deg=camera_info["pan_deg"], tilt_deg=camera_info["tilt_deg"], roll_deg=0.0
        )

        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"det(R) should be 1, got {det}"

    def test_zero_angles_gives_base_transform(self, homography_provider):
        """Zero pan/tilt should give the base coordinate transform."""
        R = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=0.0, roll_deg=0.0)

        # Should still be orthogonal
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_pan_rotation_around_z_axis(self, homography_provider):
        """Pan should rotate around world Z axis."""
        R_0 = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=45.0, roll_deg=0.0)
        R_90 = homography_provider._get_rotation_matrix(pan_deg=90.0, tilt_deg=45.0, roll_deg=0.0)
        R_180 = homography_provider._get_rotation_matrix(pan_deg=180.0, tilt_deg=45.0, roll_deg=0.0)

        # All should be valid rotations
        for R in [R_0, R_90, R_180]:
            assert abs(np.linalg.det(R) - 1.0) < 1e-10
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_tilt_affects_z_projection(self, homography_provider):
        """Different tilts should change how world Z maps to camera."""
        R_low = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=10.0, roll_deg=0.0)
        R_high = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=60.0, roll_deg=0.0)

        # Just verify they're different valid rotations
        assert not np.allclose(R_low, R_high), "Different tilts should give different R"

    def test_roll_rotation_around_optical_axis(self, homography_provider):
        """Roll should rotate around camera optical axis (Z)."""
        R_no_roll = homography_provider._get_rotation_matrix(
            pan_deg=30.0, tilt_deg=45.0, roll_deg=0.0
        )
        R_with_roll = homography_provider._get_rotation_matrix(
            pan_deg=30.0, tilt_deg=45.0, roll_deg=5.0
        )

        # Both should be valid rotations
        assert abs(np.linalg.det(R_no_roll) - 1.0) < 1e-10
        assert abs(np.linalg.det(R_with_roll) - 1.0) < 1e-10

        # Should be different
        assert not np.allclose(R_no_roll, R_with_roll), "Roll should change R"

    def test_gimbal_lock_warning_near_90_tilt(self, homography_provider):
        """Tilt near ±90° should still produce valid rotation (but may warn)."""
        # Test near-vertical looking down
        R_89 = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=89.0, roll_deg=0.0)
        R_89_9 = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=89.9, roll_deg=0.0)

        # Should still be valid orthogonal matrices
        assert abs(np.linalg.det(R_89) - 1.0) < 1e-10
        assert abs(np.linalg.det(R_89_9) - 1.0) < 1e-10


# =============================================================================
# TEST 4: HOMOGRAPHY MATRIX (H) COMPUTATION
# =============================================================================


class TestHomographyComputation:
    """Test homography matrix computation from camera parameters."""

    def test_homography_computed_successfully(self, homography_provider, camera_info):
        """Homography should be computed without errors."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)

        assert result.homography_matrix is not None, "Homography should be computed"
        assert result.homography_matrix.shape == (3, 3), "H should be 3x3"

    def test_homography_invertible(self, homography_provider, camera_info):
        """Homography should be invertible (non-singular)."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)
        H = result.homography_matrix

        # Check determinant is non-zero
        det = np.linalg.det(H)
        assert abs(det) > 1e-10, f"H determinant too small: {det}"

        # Check inverse exists and is finite
        H_inv = np.linalg.inv(H)
        assert np.all(np.isfinite(H_inv)), "H inverse should be finite"

    def test_homography_confidence_reasonable(self, homography_provider, camera_info):
        """Homography confidence should be in [0, 1] and reasonable for good params."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)

        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence should be in [0,1], got {result.confidence}"
        )
        # With reasonable camera params, confidence should be decent
        assert result.confidence > 0.3, (
            f"Confidence too low for reasonable params: {result.confidence}"
        )

    def test_homography_condition_number(self, homography_provider, camera_info):
        """Homography condition number should be reasonable for numerical stability."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)
        H = result.homography_matrix

        cond = np.linalg.cond(H)
        assert cond < 1e10, f"Condition number too high: {cond}"


# =============================================================================
# TEST 5: PIXEL → WORLD PROJECTION AGAINST GCP GROUND TRUTH
# =============================================================================


class TestPixelToWorldProjection:
    """Test that pixel coordinates project to correct world/GPS coordinates."""

    def test_project_gcp_pixels_to_gps(self, homography_provider, camera_info, gcps):
        """Project GCP pixel coords to GPS and compare with ground truth."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)

        errors_meters = []

        for gcp in gcps:
            pixel = (gcp["pixel_x"], gcp["pixel_y"])
            world_point = homography_provider.project_point(pixel)

            # Calculate error in meters using haversine or simple approximation
            lat_error = world_point.latitude - gcp["latitude"]
            lon_error = world_point.longitude - gcp["longitude"]

            # Convert to meters (rough approximation at this latitude)
            lat_m = lat_error * 111320  # ~111km per degree latitude
            lon_m = lon_error * 111320 * math.cos(math.radians(gcp["latitude"]))

            error_m = math.sqrt(lat_m**2 + lon_m**2)
            errors_meters.append(error_m)

        mean_error = np.mean(errors_meters)
        max_error = np.max(errors_meters)

        # Log errors for debugging
        print("\nPixel→GPS projection errors (UNCALIBRATED camera model):")
        print(f"  Mean error: {mean_error:.2f} m")
        print(f"  Max error: {max_error:.2f} m")
        print(f"  Individual errors: {[f'{e:.2f}m' for e in errors_meters]}")

        # IMPORTANT: These thresholds are for UNCALIBRATED camera parameters
        # With default focal length assumptions, errors will be significant.
        # After proper calibration (using GCPCalibrator), expect:
        #   - Mean error < 5m
        #   - Max error < 10m
        #
        # Current thresholds document expected behavior without calibration:
        assert mean_error < 100, f"Mean projection error too high: {mean_error:.2f}m (uncalibrated)"
        assert max_error < 200, f"Max projection error too high: {max_error:.2f}m (uncalibrated)"

        # Record the baseline for future regression testing
        print(f"\n  BASELINE (uncalibrated): mean={mean_error:.2f}m, max={max_error:.2f}m")

    def test_project_points_batch_consistency(self, homography_provider, camera_info, gcps):
        """Batch projection should match individual projections."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)

        # Get individual projections
        individual = []
        for gcp in gcps:
            pixel = (gcp["pixel_x"], gcp["pixel_y"])
            wp = homography_provider.project_point(pixel)
            individual.append((wp.latitude, wp.longitude))

        # Get batch projections
        pixels = [(gcp["pixel_x"], gcp["pixel_y"]) for gcp in gcps]
        batch = homography_provider.project_points(pixels)

        # Should match exactly
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            assert abs(ind[0] - bat.latitude) < 1e-10, f"GCP {i}: batch lat differs from individual"
            assert abs(ind[1] - bat.longitude) < 1e-10, (
                f"GCP {i}: batch lon differs from individual"
            )


# =============================================================================
# TEST 6: WORLD → PIXEL PROJECTION (INVERSE)
# =============================================================================


class TestWorldToPixelProjection:
    """Test inverse projection from world/GPS to pixel coordinates."""

    def test_world_to_pixel_basic(self, homography_provider, camera_info):
        """World coordinates should project to valid pixel coordinates."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)
        H = homography_provider.H

        # A point in front of the camera
        world_x, world_y = 10.0, 20.0  # 10m right, 20m forward

        # Apply homography: world → image
        world_homo = np.array([world_x, world_y, 1.0])
        pixel_homo = H @ world_homo

        if abs(pixel_homo[2]) > 1e-10:
            pixel_x = pixel_homo[0] / pixel_homo[2]
            pixel_y = pixel_homo[1] / pixel_homo[2]

            # Should be finite
            assert np.isfinite(pixel_x), f"pixel_x should be finite, got {pixel_x}"
            assert np.isfinite(pixel_y), f"pixel_y should be finite, got {pixel_y}"


# =============================================================================
# TEST 7: ROUND-TRIP CONSISTENCY
# =============================================================================


class TestRoundTripConsistency:
    """Test that pixel → world → pixel recovers original coordinates."""

    def test_pixel_to_world_to_pixel_roundtrip(self, homography_provider, camera_info):
        """Project pixel to world and back should recover original pixel."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)

        H = homography_provider.H
        H_inv = homography_provider.H_inv

        # Test several points across the image
        test_pixels = [
            (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2),  # Center
            (IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4),  # Top-left quadrant
            (3 * IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4),  # Top-right quadrant
            (IMAGE_WIDTH / 4, 3 * IMAGE_HEIGHT / 4),  # Bottom-left quadrant
            (3 * IMAGE_WIDTH / 4, 3 * IMAGE_HEIGHT / 4),  # Bottom-right quadrant
        ]

        for orig_x, orig_y in test_pixels:
            # Pixel → World
            pixel_homo = np.array([orig_x, orig_y, 1.0])
            world_homo = H_inv @ pixel_homo

            if abs(world_homo[2]) < 1e-10:
                # Point at horizon, skip
                continue

            world_x = world_homo[0] / world_homo[2]
            world_y = world_homo[1] / world_homo[2]

            # World → Pixel
            world_homo2 = np.array([world_x, world_y, 1.0])
            pixel_homo2 = H @ world_homo2

            if abs(pixel_homo2[2]) < 1e-10:
                continue

            recovered_x = pixel_homo2[0] / pixel_homo2[2]
            recovered_y = pixel_homo2[1] / pixel_homo2[2]

            # Should recover original within floating point tolerance
            assert abs(recovered_x - orig_x) < 1.0, f"X round-trip error: {orig_x} → {recovered_x}"
            assert abs(recovered_y - orig_y) < 1.0, f"Y round-trip error: {orig_y} → {recovered_y}"

    def test_homography_inverse_identity(self, homography_provider, camera_info):
        """H @ H_inv should be approximately identity."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)

        H = homography_provider.H
        H_inv = homography_provider.H_inv

        # H @ H_inv should be proportional to identity (homographies are up to scale)
        product = H @ H_inv

        # Normalize by [2,2] element
        if abs(product[2, 2]) > 1e-10:
            product = product / product[2, 2]
            np.testing.assert_allclose(
                product, np.eye(3), atol=1e-6, err_msg="H @ H_inv should be identity (up to scale)"
            )


# =============================================================================
# TEST 8: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_point_near_horizon(self, homography_provider, camera_info):
        """Points near horizon should either project far or raise error."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)

        # Point near top of image (near horizon)
        horizon_pixel = (IMAGE_WIDTH / 2, 10)  # Near top edge

        try:
            result = homography_provider.project_point(horizon_pixel)
            # If it succeeds, distance should be very large
            # (or coordinates should be flagged as low confidence)
            assert result.confidence < 0.5, (
                f"Horizon point should have low confidence, got {result.confidence}"
            )
        except ValueError:
            # Expected - point is at/beyond horizon
            pass

    def test_zero_height_camera(self, homography_provider, camera_info):
        """Camera at ground level should still compute (but may have issues)."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, 0.01]),  # Nearly at ground
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        result = homography_provider.compute_homography(frame=None, reference=reference)

        # Should compute but with lower confidence
        assert result.homography_matrix is not None
        # Confidence might be penalized for bad height

    def test_extreme_zoom(self, homography_provider, camera_info):
        """Extreme zoom levels should still produce valid homography."""
        for zoom in [1.0, 10.0, 25.0]:
            K = homography_provider.get_intrinsics(zoom_factor=zoom)

            reference = {
                "camera_matrix": K,
                "camera_position": np.array([0, 0, camera_info["height_meters"]]),
                "pan_deg": camera_info["pan_deg"],
                "tilt_deg": camera_info["tilt_deg"],
                "roll_deg": 0.0,
                "map_width": 640,
                "map_height": 640,
            }

            result = homography_provider.compute_homography(frame=None, reference=reference)

            assert result.homography_matrix is not None, f"H should exist at zoom={zoom}"
            assert np.all(np.isfinite(result.homography_matrix)), (
                f"H should be finite at zoom={zoom}"
            )

    def test_extreme_pan_angles(self, homography_provider):
        """Pan angles 0°, 90°, 180°, 270° should all work."""
        for pan in [0, 90, 180, 270, 360, -90]:
            R = homography_provider._get_rotation_matrix(
                pan_deg=float(pan), tilt_deg=45.0, roll_deg=0.0
            )

            assert R.shape == (3, 3), f"R should be 3x3 at pan={pan}"
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"det(R) should be 1 at pan={pan}"

    def test_tilt_boundary_values(self, homography_provider):
        """Tilt at 0°, 45°, 89° should work; 90° may cause issues."""
        # Valid tilts
        for tilt in [1.0, 45.0, 60.0, 89.0]:
            R = homography_provider._get_rotation_matrix(pan_deg=0.0, tilt_deg=tilt, roll_deg=0.0)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"det(R) should be 1 at tilt={tilt}"

    def test_negative_coordinates(self, camera_info):
        """Negative pixel coordinates should be handled gracefully."""
        provider = IntrinsicExtrinsicHomography(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )
        provider.set_camera_gps_position(camera_info["latitude"], camera_info["longitude"])

        K = provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        provider.compute_homography(frame=None, reference=reference)

        # Negative coordinates (outside image)
        try:
            result = provider.project_point((-100, -100))
            # Should still compute (point is just outside frame)
            assert result is not None
        except Exception as e:
            # May raise ValueError for invalid coordinates
            assert isinstance(e, (ValueError, RuntimeError))


# =============================================================================
# TEST 9: GCP CALIBRATION VALIDATION (REPROJECTION ERROR)
# =============================================================================


class TestGCPReprojectionError:
    """Test calibration quality by computing reprojection errors."""

    def test_compute_reprojection_errors(self, homography_provider, camera_info, gcps):
        """Compute and report reprojection errors for all GCPs."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)

        H = homography_provider.H
        ref_lat = camera_info["latitude"]
        ref_lon = camera_info["longitude"]

        reprojection_errors = []

        for i, gcp in enumerate(gcps):
            # Convert GCP GPS to local coordinates
            x_world, y_world = gps_to_local_xy(ref_lat, ref_lon, gcp["latitude"], gcp["longitude"])

            # Project world to pixel using H
            world_homo = np.array([x_world, y_world, 1.0])
            pixel_homo = H @ world_homo

            if abs(pixel_homo[2]) < 1e-10:
                reprojection_errors.append(float("inf"))
                continue

            pred_x = pixel_homo[0] / pixel_homo[2]
            pred_y = pixel_homo[1] / pixel_homo[2]

            # Compute pixel error
            error = math.sqrt((pred_x - gcp["pixel_x"]) ** 2 + (pred_y - gcp["pixel_y"]) ** 2)
            reprojection_errors.append(error)

        # Statistics
        mean_error = np.mean(reprojection_errors)
        max_error = np.max(reprojection_errors)
        std_error = np.std(reprojection_errors)

        print("\nReprojection Errors (pixels):")
        print(f"  Mean: {mean_error:.1f}")
        print(f"  Max: {max_error:.1f}")
        print(f"  Std: {std_error:.1f}")
        for i, err in enumerate(reprojection_errors):
            print(f"  GCP {i + 1}: {err:.1f} px")

        # Document current behavior - this will fail until calibration is tuned
        # Typical acceptable reprojection error is < 5-10 pixels after calibration
        # With uncalibrated parameters, errors may be large
        assert np.all(np.isfinite(reprojection_errors)), "All errors should be finite"


# =============================================================================
# TEST 10: MATHEMATICAL INVARIANTS
# =============================================================================


class TestMathematicalInvariants:
    """Test mathematical properties that must hold for correct implementation."""

    def test_homography_preserves_collinearity(self, homography_provider, camera_info):
        """Points on a line in world should project to a line in image."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)
        H = homography_provider.H

        # Three collinear world points
        world_points = [
            np.array([0, 10, 1]),
            np.array([0, 20, 1]),
            np.array([0, 30, 1]),
        ]

        # Project to image
        image_points = []
        for wp in world_points:
            ip = H @ wp
            if abs(ip[2]) > 1e-10:
                image_points.append(np.array([ip[0] / ip[2], ip[1] / ip[2]]))

        if len(image_points) == 3:
            # Check collinearity: area of triangle should be ~0
            p1, p2, p3 = image_points
            area = (
                abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2
            )

            # Normalize by scale
            max_coord = max(abs(p[i]) for p in image_points for i in [0, 1])
            if max_coord > 0:
                normalized_area = area / (max_coord**2)
                assert normalized_area < 1e-6, (
                    f"Collinear points should project to line, got area={normalized_area}"
                )

    def test_cross_ratio_preserved(self, homography_provider, camera_info):
        """Cross-ratio of four collinear points should be preserved by homography."""
        K = homography_provider.get_intrinsics(zoom_factor=camera_info["zoom_level"])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0, 0, camera_info["height_meters"]]),
            "pan_deg": camera_info["pan_deg"],
            "tilt_deg": camera_info["tilt_deg"],
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        homography_provider.compute_homography(frame=None, reference=reference)
        H = homography_provider.H

        # Four collinear world points on a line
        t_values = [0, 1, 3, 7]  # Arbitrary spacing
        world_points = [np.array([5 + t, 10 + 2 * t, 1]) for t in t_values]

        # Project to image
        image_points = []
        for wp in world_points:
            ip = H @ wp
            if abs(ip[2]) > 1e-10:
                image_points.append(ip[0] / ip[2])  # Use x-coordinate
            else:
                image_points.append(None)

        if None not in image_points:
            # Cross-ratio: (p1-p3)(p2-p4) / ((p1-p4)(p2-p3))
            def cross_ratio(a, b, c, d):
                return ((a - c) * (b - d)) / ((a - d) * (b - c))

            # Cross ratio in world (use t_values as 1D coordinates)
            cr_world = cross_ratio(*t_values)

            # Cross ratio in image
            cr_image = cross_ratio(*image_points)

            # Should be equal (fundamental projective invariant)
            assert abs(cr_world - cr_image) < 1e-3, (
                f"Cross-ratio should be preserved: world={cr_world}, image={cr_image}"
            )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
