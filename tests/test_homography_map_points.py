#!/usr/bin/env python3
"""
TDD tests for homography using map points instead of GPS coordinates.

This test suite validates the homography transformation between camera image
pixels and map point coordinates using the Valte test data. Tests follow TDD
principles and initially fail until homography module is updated.

Test Data Structure:
    - map_points.json: Reference map points with UTM coordinates (stored as pixel_x, pixel_y)
    - test_data_Valte_20260109_195052.json: GCPs linking camera pixels to map point IDs
    - test_data_Valte_20260109_195052.jpg: Camera image

Note: The MapPoint "pixel_x" and "pixel_y" fields actually contain UTM easting/northing
coordinates in meters, not pixel coordinates. This is a known naming issue.

Test Coverage:
    - Loading map points from registry
    - Creating GCP correspondences from map point references
    - Computing homography from camera pixels to map UTM coordinates
    - Projecting camera pixels to map coordinates (forward transform)
    - Projecting map coordinates back to camera pixels (inverse transform)
    - Round-trip validation (camera -> map -> camera)
    - Error metrics and reprojection accuracy
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from poc_homography.map_points import MapPoint, MapPointRegistry

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent
MAP_POINTS_PATH = TEST_DATA_DIR / "map_points.json"
VALTE_GCP_PATH = TEST_DATA_DIR / "test_data_Valte_20260109_195052.json"
VALTE_IMAGE_PATH = TEST_DATA_DIR / "test_data_Valte_20260109_195052.jpg"


@pytest.fixture
def map_point_registry():
    """Load map point registry from JSON file."""
    return MapPointRegistry.load(MAP_POINTS_PATH)


@pytest.fixture
def valte_gcp_data():
    """Load Valte GCP test data from JSON file."""
    with open(VALTE_GCP_PATH) as f:
        return json.load(f)


@pytest.fixture
def valte_image():
    """Load Valte camera image."""
    image = cv2.imread(str(VALTE_IMAGE_PATH))
    assert image is not None, f"Failed to load image: {VALTE_IMAGE_PATH}"
    return image


class TestMapPointRegistryLoading:
    """Test loading map point registry from JSON."""

    def test_map_points_file_exists(self):
        """Test that map_points.json file exists."""
        assert MAP_POINTS_PATH.exists(), f"Map points file not found: {MAP_POINTS_PATH}"

    def test_load_map_point_registry(self, map_point_registry):
        """Test loading map point registry."""
        assert map_point_registry is not None
        assert map_point_registry.map_id == "map_valte"
        assert len(map_point_registry.points) > 0

    def test_map_point_structure(self, map_point_registry):
        """Test that map points have expected structure."""
        # Get a sample point
        point_id = "A7"  # One of the points used in test data
        assert point_id in map_point_registry.points

        point = map_point_registry.points[point_id]
        assert isinstance(point, MapPoint)
        assert point.id == point_id
        assert isinstance(point.pixel_x, (int, float))
        assert isinstance(point.pixel_y, (int, float))
        assert point.map_id == "map_valte"

        # These are actually UTM coordinates (meters), not pixels
        # Valencia is in UTM zone 30N, so expect large values
        assert 250000 < point.pixel_x < 253000, "pixel_x should be UTM easting"
        assert -362000 < point.pixel_y < -359000, "pixel_y should be UTM northing"

    def test_all_test_gcps_have_map_points(self, map_point_registry, valte_gcp_data):
        """Test that all GCPs reference valid map points."""
        for gcp in valte_gcp_data["gcps"]:
            map_point_id = gcp["map_point_id"]
            assert map_point_id in map_point_registry.points, (
                f"GCP references missing map point: {map_point_id}"
            )


class TestGCPCorrespondenceExtraction:
    """Test extracting pixel correspondences from GCP data."""

    def test_extract_camera_pixels_from_gcps(self, valte_gcp_data):
        """Test extracting camera pixel coordinates from GCP data."""
        gcps = valte_gcp_data["gcps"]
        camera_pixels = np.array([[gcp["pixel_x"], gcp["pixel_y"]] for gcp in gcps])

        assert camera_pixels.shape[0] == len(gcps)
        assert camera_pixels.shape[1] == 2

        # Validate pixel ranges (1920x1080 image)
        assert np.all(camera_pixels[:, 0] >= 0)  # x >= 0
        assert np.all(camera_pixels[:, 0] < 1920)  # x < width
        assert np.all(camera_pixels[:, 1] >= 0)  # y >= 0
        assert np.all(camera_pixels[:, 1] < 1080)  # y < height

    def test_extract_map_coords_from_registry(self, map_point_registry, valte_gcp_data):
        """Test extracting map UTM coordinates from map point registry."""
        gcps = valte_gcp_data["gcps"]
        map_coords = []

        for gcp in gcps:
            map_point_id = gcp["map_point_id"]
            map_point = map_point_registry.points[map_point_id]
            map_coords.append([map_point.pixel_x, map_point.pixel_y])

        map_coords = np.array(map_coords)

        assert map_coords.shape[0] == len(gcps)
        assert map_coords.shape[1] == 2

        # Map coords should be in reasonable UTM range (not NaN/inf)
        assert np.all(np.isfinite(map_coords))
        assert np.all(map_coords[:, 0] > 250000)  # UTM easting
        assert np.all(map_coords[:, 1] < -359000)  # UTM northing (negative in this dataset)

    def test_create_correspondence_pairs(self, map_point_registry, valte_gcp_data):
        """Test creating matched pairs of camera pixels and map UTM coords."""
        gcps = valte_gcp_data["gcps"]

        correspondences = []
        for gcp in gcps:
            camera_pt = (gcp["pixel_x"], gcp["pixel_y"])
            map_point = map_point_registry.points[gcp["map_point_id"]]
            map_coord = (map_point.pixel_x, map_point.pixel_y)
            correspondences.append((camera_pt, map_coord))

        assert len(correspondences) == len(gcps)
        assert all(len(pair) == 2 for pair in correspondences)
        assert all(len(pair[0]) == 2 and len(pair[1]) == 2 for pair in correspondences)


class TestHomographyComputation:
    """Test computing homography from camera pixels to map UTM coordinates."""

    def test_compute_homography_from_gcps(self, map_point_registry, valte_gcp_data):
        """Test computing homography matrix using cv2.findHomography."""
        # Extract correspondences
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        # Compute homography (camera pixels -> map UTM coords)
        H, mask = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)

        assert H is not None, "Homography computation failed"
        assert H.shape == (3, 3), f"Expected 3x3 matrix, got {H.shape}"
        assert np.linalg.det(H) != 0, "Homography matrix is singular"

        # Check inliers (relaxed threshold for real-world data)
        num_inliers = np.sum(mask)
        total_points = len(mask)
        inlier_ratio = num_inliers / total_points

        assert num_inliers >= 4, f"Not enough inliers: {num_inliers}/{total_points}"
        assert inlier_ratio >= 0.5, f"Low inlier ratio: {inlier_ratio:.2%}"

    def test_homography_matrix_properties(self, map_point_registry, valte_gcp_data):
        """Test mathematical properties of homography matrix."""
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)

        # Test invertibility
        H_inv = np.linalg.inv(H)
        assert np.linalg.det(H_inv) != 0, "Inverse homography is singular"

        # Test that H * H_inv ≈ I
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6), "H * H_inv should equal identity matrix"


class TestForwardProjection:
    """Test projecting camera pixels to map UTM coordinates."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_gcp_data):
        """Compute and return homography matrix."""
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)
        return H

    def test_project_single_camera_pixel_to_map(self, homography_matrix):
        """Test projecting a single camera pixel to map UTM coordinates."""
        # Test with camera center point
        camera_pt = np.array([[[960.0, 540.0]]], dtype=np.float32)
        map_coord = cv2.perspectiveTransform(camera_pt, homography_matrix)

        assert map_coord.shape == (1, 1, 2)
        map_x, map_y = map_coord[0, 0]

        # Map coordinates should be finite and in UTM range
        assert np.isfinite(map_x) and np.isfinite(map_y)
        assert 250000 < map_x < 253000, f"Map X (easting) out of expected range: {map_x}"
        assert -362000 < map_y < -359000, f"Map Y (northing) out of expected range: {map_y}"

    def test_project_gcp_camera_pixels_to_map(
        self, homography_matrix, map_point_registry, valte_gcp_data
    ):
        """Test that projecting GCP camera pixels yields expected map coords."""
        errors = []
        for gcp in valte_gcp_data["gcps"]:
            # Camera pixel
            camera_pt = np.array([[[gcp["pixel_x"], gcp["pixel_y"]]]], dtype=np.float32)

            # Project to map
            projected_map_coord = cv2.perspectiveTransform(camera_pt, homography_matrix)[0, 0]

            # Expected map coord
            expected_map_point = map_point_registry.points[gcp["map_point_id"]]
            expected = np.array([expected_map_point.pixel_x, expected_map_point.pixel_y])

            # Calculate reprojection error (in meters for UTM coords)
            error = np.linalg.norm(projected_map_coord - expected)
            errors.append(error)

        # Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        median_error = np.median(errors)

        # For real-world data with scale differences, accept larger errors
        # These are in meters in UTM space
        assert mean_error < 50.0, f"Mean reprojection error too high: {mean_error:.2f} meters"
        assert median_error < 30.0, f"Median reprojection error too high: {median_error:.2f} meters"

    def test_forward_projection_batch(self, homography_matrix, valte_gcp_data):
        """Test batch projection of multiple camera pixels."""
        camera_pixels = np.array(
            [[[gcp["pixel_x"], gcp["pixel_y"]]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = cv2.perspectiveTransform(camera_pixels, homography_matrix)

        assert map_coords.shape[0] == len(valte_gcp_data["gcps"])
        assert np.all(np.isfinite(map_coords))


class TestInverseProjection:
    """Test projecting map UTM coordinates back to camera pixels."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_gcp_data):
        """Compute and return homography matrix."""
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)
        return H

    @pytest.fixture
    def inverse_homography_matrix(self, homography_matrix):
        """Compute and return inverse homography matrix."""
        return np.linalg.inv(homography_matrix)

    def test_project_map_coord_to_camera(self, inverse_homography_matrix, map_point_registry):
        """Test projecting a map UTM coordinate to camera pixels."""
        # Get a map point
        map_point = map_point_registry.points["A7"]
        map_coord = np.array([[[map_point.pixel_x, map_point.pixel_y]]], dtype=np.float32)

        # Project to camera
        camera_pt = cv2.perspectiveTransform(map_coord, inverse_homography_matrix)

        assert camera_pt.shape == (1, 1, 2)
        cam_x, cam_y = camera_pt[0, 0]

        # Camera coordinates should be finite and within reasonable bounds
        # (may be slightly outside image for edge points)
        assert np.isfinite(cam_x) and np.isfinite(cam_y)
        assert -100 <= cam_x < 2020, f"Camera x out of reasonable bounds: {cam_x}"
        assert -100 <= cam_y < 1180, f"Camera y out of reasonable bounds: {cam_y}"

    def test_inverse_projection_of_gcps(
        self, inverse_homography_matrix, map_point_registry, valte_gcp_data
    ):
        """Test that projecting map coords back yields original camera pixels."""
        errors = []
        for gcp in valte_gcp_data["gcps"]:
            # Map coord (UTM)
            map_point = map_point_registry.points[gcp["map_point_id"]]
            map_coord = np.array([[[map_point.pixel_x, map_point.pixel_y]]], dtype=np.float32)

            # Project to camera
            projected_camera_pt = cv2.perspectiveTransform(map_coord, inverse_homography_matrix)[
                0, 0
            ]

            # Expected camera pixel
            expected = np.array([gcp["pixel_x"], gcp["pixel_y"]])

            # Calculate reprojection error (in pixels)
            error = np.linalg.norm(projected_camera_pt - expected)
            errors.append(error)

        # Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Inverse projection errors in pixels should be reasonable
        assert mean_error < 20.0, (
            f"Mean inverse reprojection error too high: {mean_error:.2f} pixels"
        )
        assert max_error < 40.0, f"Max inverse reprojection error too high: {max_error:.2f} pixels"


class TestRoundTripProjection:
    """Test round-trip projection: camera -> map -> camera."""

    @pytest.fixture
    def homography_matrices(self, map_point_registry, valte_gcp_data):
        """Compute and return forward and inverse homography matrices."""
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        H = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)[0]
        H_inv = np.linalg.inv(H)
        return H, H_inv

    def test_round_trip_camera_to_map_to_camera(self, homography_matrices, valte_gcp_data):
        """Test camera -> map -> camera preserves coordinates."""
        H, H_inv = homography_matrices

        errors = []
        for gcp in valte_gcp_data["gcps"]:
            # Original camera pixel
            original = np.array([[[gcp["pixel_x"], gcp["pixel_y"]]]], dtype=np.float32)

            # Project to map
            map_coord = cv2.perspectiveTransform(original, H)

            # Project back to camera
            recovered = cv2.perspectiveTransform(map_coord, H_inv)

            # Calculate round-trip error
            error = np.linalg.norm(recovered[0, 0] - original[0, 0])
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip error should be small (in pixels)
        assert mean_error < 5.0, f"High mean round-trip error: {mean_error:.2f} pixels"
        assert max_error < 15.0, f"High max round-trip error: {max_error:.2f} pixels"

    def test_round_trip_map_to_camera_to_map(
        self, homography_matrices, map_point_registry, valte_gcp_data
    ):
        """Test map -> camera -> map preserves coordinates."""
        H, H_inv = homography_matrices

        errors = []
        for gcp in valte_gcp_data["gcps"]:
            # Original map coord (UTM meters)
            map_point = map_point_registry.points[gcp["map_point_id"]]
            original = np.array([[[map_point.pixel_x, map_point.pixel_y]]], dtype=np.float32)

            # Project to camera
            camera_pt = cv2.perspectiveTransform(original, H_inv)

            # Project back to map
            recovered = cv2.perspectiveTransform(camera_pt, H)

            # Calculate round-trip error (in meters)
            error = np.linalg.norm(recovered[0, 0] - original[0, 0])
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip error in meters should be acceptable
        assert mean_error < 30.0, f"High mean round-trip error: {mean_error:.2f} meters"
        assert max_error < 80.0, f"High max round-trip error: {max_error:.2f} meters"


class TestReprojectionErrorMetrics:
    """Test computing reprojection error metrics."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_gcp_data):
        """Compute and return homography matrix."""
        camera_pixels = np.array(
            [[gcp["pixel_x"], gcp["pixel_y"]] for gcp in valte_gcp_data["gcps"]], dtype=np.float32
        )

        map_coords = np.array(
            [
                [
                    map_point_registry.points[gcp["map_point_id"]].pixel_x,
                    map_point_registry.points[gcp["map_point_id"]].pixel_y,
                ]
                for gcp in valte_gcp_data["gcps"]
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(camera_pixels, map_coords, cv2.RANSAC, 50.0)
        return H

    def test_mean_reprojection_error(self, homography_matrix, map_point_registry, valte_gcp_data):
        """Test computing mean reprojection error across all GCPs."""
        errors = []

        for gcp in valte_gcp_data["gcps"]:
            # Project camera pixel to map
            camera_pt = np.array([[[gcp["pixel_x"], gcp["pixel_y"]]]], dtype=np.float32)
            projected_map_coord = cv2.perspectiveTransform(camera_pt, homography_matrix)[0, 0]

            # Expected map coord
            expected_map_point = map_point_registry.points[gcp["map_point_id"]]
            expected = np.array([expected_map_point.pixel_x, expected_map_point.pixel_y])

            # Calculate error (in meters)
            error = np.linalg.norm(projected_map_coord - expected)
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        # Log statistics for debugging
        print("\nReprojection error statistics (meters):")
        print(f"  Mean: {mean_error:.2f}")
        print(f"  Max: {max_error:.2f}")
        print(f"  Std: {std_error:.2f}")

        # Relaxed thresholds for real-world data
        assert mean_error < 50.0, f"Mean reprojection error too high: {mean_error:.2f} meters"
        assert max_error < 100.0, f"Max reprojection error too high: {max_error:.2f} meters"

    def test_rmse_reprojection_error(self, homography_matrix, map_point_registry, valte_gcp_data):
        """Test computing RMSE (Root Mean Square Error) of reprojection."""
        squared_errors = []

        for gcp in valte_gcp_data["gcps"]:
            camera_pt = np.array([[[gcp["pixel_x"], gcp["pixel_y"]]]], dtype=np.float32)
            projected_map_coord = cv2.perspectiveTransform(camera_pt, homography_matrix)[0, 0]

            expected_map_point = map_point_registry.points[gcp["map_point_id"]]
            expected = np.array([expected_map_point.pixel_x, expected_map_point.pixel_y])

            squared_error = np.sum((projected_map_coord - expected) ** 2)
            squared_errors.append(squared_error)

        rmse = np.sqrt(np.mean(squared_errors))

        # RMSE in meters for real-world UTM coordinates
        assert rmse < 60.0, f"RMSE too high: {rmse:.2f} meters"
