#!/usr/bin/env python3
"""
Integration tests for MapPointHomography using real Valte test data.

These tests validate the complete workflow of:
1. Loading map points from registry
2. Computing homography from GCPs
3. Projecting coordinates bidirectionally
4. Validating quality metrics
"""

import json
from pathlib import Path

import numpy as np
import pytest

from poc_homography.homography_map_points import MapPointHomography
from poc_homography.map_points import MapPointRegistry


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent
MAP_POINTS_PATH = TEST_DATA_DIR / "map_points.json"
VALTE_GCP_PATH = TEST_DATA_DIR / "test_data_Valte_20260109_195052.json"


@pytest.fixture
def map_registry():
    """Load map point registry."""
    return MapPointRegistry.load(MAP_POINTS_PATH)


@pytest.fixture
def valte_gcps():
    """Load Valte GCP data."""
    with open(VALTE_GCP_PATH) as f:
        data = json.load(f)
    return data["gcps"]


@pytest.fixture
def homography(map_registry, valte_gcps):
    """Create and compute homography from test data."""
    h = MapPointHomography()
    h.compute_from_gcps(valte_gcps, map_registry)
    return h


class TestMapPointHomographyInitialization:
    """Test homography initialization and validation."""

    def test_initial_state_invalid(self):
        """Test that newly created homography is invalid."""
        h = MapPointHomography()
        assert not h.is_valid()
        assert h.get_result() is None

    def test_camera_to_map_before_compute_raises(self):
        """Test that projecting before computing raises error."""
        h = MapPointHomography()
        with pytest.raises(RuntimeError, match="No valid homography"):
            h.camera_to_map((960, 540))

    def test_map_to_camera_before_compute_raises(self):
        """Test that inverse projecting before computing raises error."""
        h = MapPointHomography()
        with pytest.raises(RuntimeError, match="No valid homography"):
            h.map_to_camera((251500.0, -360500.0))


class TestMapPointHomographyComputation:
    """Test homography computation from GCPs."""

    def test_compute_from_gcps_success(self, map_registry, valte_gcps):
        """Test successful homography computation."""
        h = MapPointHomography()
        result = h.compute_from_gcps(valte_gcps, map_registry)

        assert h.is_valid()
        assert result is not None
        assert result.homography_matrix.shape == (3, 3)
        assert result.inverse_matrix.shape == (3, 3)

    def test_compute_with_insufficient_gcps_raises(self, map_registry):
        """Test that too few GCPs raises error."""
        h = MapPointHomography()
        gcps = [
            {"pixel_x": 800, "pixel_y": 580, "map_point_id": "A7"},
            {"pixel_x": 1082, "pixel_y": 390, "map_point_id": "A6"},
        ]  # Only 2 GCPs, need at least 4

        with pytest.raises(ValueError, match="Need at least 4 GCPs"):
            h.compute_from_gcps(gcps, map_registry)

    def test_compute_with_missing_map_point_raises(self, map_registry):
        """Test that missing map point raises error."""
        h = MapPointHomography()
        gcps = [
            {"pixel_x": 800, "pixel_y": 580, "map_point_id": "A7"},
            {"pixel_x": 1082, "pixel_y": 390, "map_point_id": "A6"},
            {"pixel_x": 408, "pixel_y": 776, "map_point_id": "NONEXISTENT"},
            {"pixel_x": 568, "pixel_y": 846, "map_point_id": "X16"},
        ]

        with pytest.raises(ValueError, match="Map point not found"):
            h.compute_from_gcps(gcps, map_registry)

    def test_result_quality_metrics(self, homography):
        """Test that result contains valid quality metrics."""
        result = homography.get_result()

        assert result.num_gcps == 16  # Number of GCPs in test data
        assert result.num_inliers >= 4
        assert 0.0 <= result.inlier_ratio <= 1.0
        assert result.mean_reproj_error >= 0.0
        assert result.max_reproj_error >= 0.0
        assert result.rmse >= 0.0

        # Quality should be reasonable
        assert result.inlier_ratio >= 0.5, "Inlier ratio should be >= 50%"
        assert result.mean_reproj_error < 50.0, "Mean error should be < 50 meters"

    def test_matrices_are_invertible(self, homography):
        """Test that computed matrices are properly invertible."""
        H = homography.get_homography_matrix()
        H_inv = homography.get_inverse_matrix()

        # H * H_inv should be identity
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6)

        # H_inv * H should be identity
        identity2 = H_inv @ H
        assert np.allclose(identity2, np.eye(3), atol=1e-6)


class TestForwardProjection:
    """Test camera pixel to map coordinate projection."""

    def test_project_camera_center(self, homography):
        """Test projecting camera center point."""
        map_coord = homography.camera_to_map((960.0, 540.0))

        # Should return valid UTM coordinates
        assert isinstance(map_coord, tuple)
        assert len(map_coord) == 2
        assert 250000 < map_coord[0] < 253000, "Easting in expected range"
        assert -362000 < map_coord[1] < -359000, "Northing in expected range"

    def test_project_gcp_points(self, homography, map_registry, valte_gcps):
        """Test projecting actual GCP points."""
        for gcp in valte_gcps[:5]:  # Test first 5 GCPs
            camera_pixel = (gcp["pixel_x"], gcp["pixel_y"])
            map_coord = homography.camera_to_map(camera_pixel)

            # Get expected map coordinate
            map_point = map_registry.points[gcp["map_point_id"]]
            expected = (map_point.pixel_x, map_point.pixel_y)

            # Calculate error
            error = np.linalg.norm(np.array(map_coord) - np.array(expected))

            # Should be close (within 50 meters)
            assert error < 50.0, f"Error for {gcp['map_point_id']}: {error:.2f}m"

    def test_batch_projection(self, homography, valte_gcps):
        """Test batch projection of multiple camera pixels."""
        camera_pixels = [(gcp["pixel_x"], gcp["pixel_y"]) for gcp in valte_gcps]
        map_coords = homography.camera_to_map_batch(camera_pixels)

        assert len(map_coords) == len(camera_pixels)
        for coord in map_coords:
            assert isinstance(coord, tuple)
            assert len(coord) == 2
            assert 250000 < coord[0] < 253000
            assert -362000 < coord[1] < -359000


class TestInverseProjection:
    """Test map coordinate to camera pixel projection."""

    def test_project_map_point_to_camera(self, homography, map_registry):
        """Test projecting map point back to camera."""
        map_point = map_registry.points["A7"]
        map_coord = (map_point.pixel_x, map_point.pixel_y)

        camera_pixel = homography.map_to_camera(map_coord)

        # Should return valid pixel coordinates
        assert isinstance(camera_pixel, tuple)
        assert len(camera_pixel) == 2
        # Allow slight out-of-bounds for edge cases
        assert -100 < camera_pixel[0] < 2020, "X in reasonable range"
        assert -100 < camera_pixel[1] < 1180, "Y in reasonable range"

    def test_inverse_projection_accuracy(self, homography, map_registry, valte_gcps):
        """Test that inverse projection is accurate."""
        errors = []

        for gcp in valte_gcps[:10]:  # Test first 10
            # Get map coordinate
            map_point = map_registry.points[gcp["map_point_id"]]
            map_coord = (map_point.pixel_x, map_point.pixel_y)

            # Project to camera
            projected_pixel = homography.map_to_camera(map_coord)

            # Expected camera pixel
            expected_pixel = (gcp["pixel_x"], gcp["pixel_y"])

            # Calculate error in pixels
            error = np.linalg.norm(np.array(projected_pixel) - np.array(expected_pixel))
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Errors should be reasonable
        assert mean_error < 20.0, f"Mean error: {mean_error:.2f} pixels"
        assert max_error < 40.0, f"Max error: {max_error:.2f} pixels"

    def test_batch_inverse_projection(self, homography, map_registry, valte_gcps):
        """Test batch inverse projection."""
        # Get map coordinates for all GCPs
        map_coords = []
        for gcp in valte_gcps:
            map_point = map_registry.points[gcp["map_point_id"]]
            map_coords.append((map_point.pixel_x, map_point.pixel_y))

        # Project all at once
        camera_pixels = homography.map_to_camera_batch(map_coords)

        assert len(camera_pixels) == len(map_coords)
        for pixel in camera_pixels:
            assert isinstance(pixel, tuple)
            assert len(pixel) == 2


class TestRoundTripConsistency:
    """Test round-trip projection consistency."""

    def test_camera_to_map_to_camera(self, homography, valte_gcps):
        """Test camera -> map -> camera round-trip."""
        errors = []

        for gcp in valte_gcps:
            original = (gcp["pixel_x"], gcp["pixel_y"])

            # Forward then inverse
            map_coord = homography.camera_to_map(original)
            recovered = homography.map_to_camera(map_coord)

            # Calculate round-trip error
            error = np.linalg.norm(np.array(recovered) - np.array(original))
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip should have minimal error
        assert mean_error < 5.0, f"Mean round-trip error: {mean_error:.2f} pixels"
        assert max_error < 15.0, f"Max round-trip error: {max_error:.2f} pixels"

    def test_map_to_camera_to_map(self, homography, map_registry, valte_gcps):
        """Test map -> camera -> map round-trip."""
        errors = []

        for gcp in valte_gcps:
            map_point = map_registry.points[gcp["map_point_id"]]
            original = (map_point.pixel_x, map_point.pixel_y)

            # Inverse then forward
            camera_pixel = homography.map_to_camera(original)
            recovered = homography.camera_to_map(camera_pixel)

            # Calculate round-trip error in meters
            error = np.linalg.norm(np.array(recovered) - np.array(original))
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip errors in meters
        assert mean_error < 30.0, f"Mean round-trip error: {mean_error:.2f} meters"
        assert max_error < 80.0, f"Max round-trip error: {max_error:.2f} meters"


class TestMatrixRetrieval:
    """Test matrix getter methods."""

    def test_get_homography_matrix(self, homography):
        """Test retrieving forward homography matrix."""
        H = homography.get_homography_matrix()

        assert H.shape == (3, 3)
        assert np.linalg.det(H) != 0

        # Should be a copy, not reference
        H[0, 0] = 999.0
        H2 = homography.get_homography_matrix()
        assert H2[0, 0] != 999.0

    def test_get_inverse_matrix(self, homography):
        """Test retrieving inverse homography matrix."""
        H_inv = homography.get_inverse_matrix()

        assert H_inv.shape == (3, 3)
        assert np.linalg.det(H_inv) != 0

        # Should be a copy, not reference
        H_inv[0, 0] = 999.0
        H_inv2 = homography.get_inverse_matrix()
        assert H_inv2[0, 0] != 999.0

    def test_get_matrix_before_compute_raises(self):
        """Test that getting matrices before compute raises error."""
        h = MapPointHomography()

        with pytest.raises(RuntimeError, match="No valid homography"):
            h.get_homography_matrix()

        with pytest.raises(RuntimeError, match="No valid homography"):
            h.get_inverse_matrix()
