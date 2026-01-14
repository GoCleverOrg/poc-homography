#!/usr/bin/env python3
"""
TDD tests for homography using map points instead of GPS coordinates.

This test suite validates the homography transformation between camera image
pixels and map point coordinates using the Valte test data. Tests follow TDD
principles and initially fail until homography module is updated.

Terminology:
    - Annotation: A point marked on the camera image (camera pixels)
    - GCP (Ground Control Point): A reference point on the map (map pixels/coordinates)

Test Data Structure:
    - valte_map_points.yaml: GCPs with UTM coordinates (stored as pixel_x, pixel_y)
    - test_data_Valte_20260109_195052.json: Capture data with annotations linking
      camera pixels to GCP IDs via the Annotation dataclass
    - test_data_Valte_20260109_195052.jpg: Camera image

Data Model:
    - Annotations are loaded from capture.annotations array in the JSON file
    - Each Annotation contains:
        - gcp_id: ID referencing a GCP (MapPoint) in the registry
        - pixel: PixelPoint with x/y coordinates in camera image
    - GCPs (MapPoint) "pixel_x" and "pixel_y" fields contain UTM easting/northing
      coordinates in meters (naming is historical)

Test Coverage:
    - Loading GCPs from registry
    - Creating annotation-GCP correspondences from test data
    - Computing homography from annotations (camera pixels) to GCPs (map coordinates)
    - Projecting annotations to GCP space (forward transform)
    - Projecting GCPs back to annotation space (inverse transform)
    - Round-trip validation (annotation -> GCP -> annotation)
    - Error metrics and reprojection accuracy
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from poc_homography.calibration.annotation import Annotation
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
MAP_POINTS_PATH = TEST_DATA_DIR / "Cartografia_valencia_gcps.yaml"
ANNOTATIONS_PATH = TEST_DATA_DIR / "valte_annotations.yaml"
VALTE_IMAGE_PATH = TEST_DATA_DIR / "valte_30.8_13.1_1_12-01-2026.png"


@pytest.fixture
def map_point_registry():
    """Load map point registry from YAML file."""
    return MapPointRegistry.load(MAP_POINTS_PATH)


@pytest.fixture
def valte_annotations():
    """Load Valte annotations (camera pixel points) from YAML file.

    Returns a list of Annotation objects parsed from the annotations array.
    Each annotation links a camera pixel location to a GCP ID.
    """
    with open(ANNOTATIONS_PATH) as f:
        data = yaml.safe_load(f)

    # Parse annotations from the first test case
    test_case = data["test_cases"][0]
    annotations = [
        Annotation(gcp_id=ann["gcp_id"], pixel=PixelPoint(ann["pixel_x"], ann["pixel_y"]))
        for ann in test_case["annotations"]
    ]
    return annotations


@pytest.fixture
def valte_image():
    """Load Valte camera image."""
    image = cv2.imread(str(VALTE_IMAGE_PATH))
    assert image is not None, f"Failed to load image: {VALTE_IMAGE_PATH}"
    return image


class TestMapPointRegistryLoading:
    """Test loading map point registry from YAML."""

    def test_map_points_file_exists(self):
        """Test that valte_map_points.yaml file exists."""
        assert MAP_POINTS_PATH.exists(), f"Map points file not found: {MAP_POINTS_PATH}"

    def test_load_map_point_registry(self, map_point_registry):
        """Test loading map point registry."""
        assert map_point_registry is not None
        assert map_point_registry.map_id == "Cartografia_valencia"
        assert len(map_point_registry.points) == 27  # PS1-17, AR1-6, EX1-4

    def test_map_point_structure(self, map_point_registry):
        """Test that map points have expected structure."""
        # Get a sample point
        point_id = "PS1"  # One of the points used in test data
        assert point_id in map_point_registry.points

        point = map_point_registry.points[point_id]
        assert isinstance(point, MapPoint)
        # Note: id and map_id are no longer fields of MapPoint - they're managed by the registry
        assert isinstance(point.pixel_x, (int, float))
        assert isinstance(point.pixel_y, (int, float))

        # These are map pixel coordinates
        assert 700 < point.pixel_x < 1300, "pixel_x should be in map pixel range"
        assert 200 < point.pixel_y < 800, "pixel_y should be in map pixel range"

    def test_all_annotations_reference_valid_gcps(self, map_point_registry, valte_annotations):
        """Test that all annotations reference valid GCPs in the registry."""
        for annotation in valte_annotations:
            gcp_id = annotation.gcp_id
            assert gcp_id in map_point_registry.points, (
                f"Annotation references missing GCP: {gcp_id}"
            )


class TestAnnotationGCPCorrespondenceExtraction:
    """Test extracting annotation-GCP pixel correspondences."""

    def test_extract_camera_pixels_from_annotations(self, valte_annotations):
        """Test extracting camera pixel coordinates from annotations."""
        camera_pixels = np.array([[ann.pixel.x, ann.pixel.y] for ann in valte_annotations])

        assert camera_pixels.shape[0] == len(valte_annotations)
        assert camera_pixels.shape[1] == 2

        # Validate pixel ranges (1920x1080 image)
        assert np.all(camera_pixels[:, 0] >= 0)  # x >= 0
        assert np.all(camera_pixels[:, 0] < 1920)  # x < width
        assert np.all(camera_pixels[:, 1] >= 0)  # y >= 0
        assert np.all(camera_pixels[:, 1] < 1080)  # y < height

    def test_extract_gcp_coords_from_registry(self, map_point_registry, valte_annotations):
        """Test extracting GCP (map) coordinates from registry."""
        gcp_coords = []

        for annotation in valte_annotations:
            gcp_id = annotation.gcp_id
            gcp = map_point_registry.points[gcp_id]
            gcp_coords.append([gcp.pixel_x, gcp.pixel_y])

        gcp_coords = np.array(gcp_coords)

        assert gcp_coords.shape[0] == len(valte_annotations)
        assert gcp_coords.shape[1] == 2

        # GCP coords should be in reasonable map pixel range (not NaN/inf)
        assert np.all(np.isfinite(gcp_coords))
        assert np.all(gcp_coords[:, 0] > 700)  # Map pixel x
        assert np.all(gcp_coords[:, 1] > 200)  # Map pixel y

    def test_create_correspondence_pairs(self, map_point_registry, valte_annotations):
        """Test creating matched pairs of annotation (camera) and GCP (map) coordinates."""
        correspondences = []
        for annotation in valte_annotations:
            annotation_pt = (annotation.pixel.x, annotation.pixel.y)
            gcp = map_point_registry.points[annotation.gcp_id]
            gcp_coord = (gcp.pixel_x, gcp.pixel_y)
            correspondences.append((annotation_pt, gcp_coord))

        assert len(correspondences) == len(valte_annotations)
        assert all(len(pair) == 2 for pair in correspondences)
        assert all(len(pair[0]) == 2 and len(pair[1]) == 2 for pair in correspondences)


class TestHomographyComputation:
    """Test computing homography from annotations (camera) to GCPs (map)."""

    def test_compute_homography_from_annotations(self, map_point_registry, valte_annotations):
        """Test computing homography matrix using cv2.findHomography."""
        # Extract correspondences: annotations (camera pixels) and GCPs (map coords)
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        # Compute homography (annotation pixels -> GCP coords)
        H, mask = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)

        assert H is not None, "Homography computation failed"
        assert H.shape == (3, 3), f"Expected 3x3 matrix, got {H.shape}"
        assert np.linalg.det(H) != 0, "Homography matrix is singular"

        # Check inliers (relaxed threshold for real-world data)
        num_inliers = np.sum(mask)
        total_points = len(mask)
        inlier_ratio = num_inliers / total_points

        assert num_inliers >= 4, f"Not enough inliers: {num_inliers}/{total_points}"
        assert inlier_ratio >= 0.5, f"Low inlier ratio: {inlier_ratio:.2%}"

    def test_homography_matrix_properties(self, map_point_registry, valte_annotations):
        """Test mathematical properties of homography matrix."""
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)

        # Test invertibility
        H_inv = np.linalg.inv(H)
        assert np.linalg.det(H_inv) != 0, "Inverse homography is singular"

        # Test that H * H_inv ≈ I
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6), "H * H_inv should equal identity matrix"


class TestForwardProjection:
    """Test projecting annotations (camera pixels) to GCP (map) coordinates."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_annotations):
        """Compute and return homography matrix (annotation -> GCP)."""
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)
        return H

    def test_project_single_annotation_to_gcp(self, homography_matrix):
        """Test projecting a single annotation (camera pixel) to GCP (map) coordinates."""
        # Test with camera center point
        camera_pt = np.array([[[960.0, 540.0]]], dtype=np.float32)
        map_coord = cv2.perspectiveTransform(camera_pt, homography_matrix)

        assert map_coord.shape == (1, 1, 2)
        map_x, map_y = map_coord[0, 0]

        # Map coordinates should be finite and in map pixel range
        assert np.isfinite(map_x) and np.isfinite(map_y)
        assert 700 < map_x < 1300, f"Map X out of expected range: {map_x}"
        assert 200 < map_y < 800, f"Map Y out of expected range: {map_y}"

    def test_project_annotations_to_gcps(
        self, homography_matrix, map_point_registry, valte_annotations
    ):
        """Test that projecting annotation pixels yields expected GCP coords."""
        errors = []
        for annotation in valte_annotations:
            # Annotation (camera pixel)
            annotation_pt = np.array([[[annotation.pixel.x, annotation.pixel.y]]], dtype=np.float32)

            # Project to GCP space
            projected_gcp_coord = cv2.perspectiveTransform(annotation_pt, homography_matrix)[0, 0]

            # Expected GCP coord
            expected_gcp = map_point_registry.points[annotation.gcp_id]
            expected = np.array([expected_gcp.pixel_x, expected_gcp.pixel_y])

            # Calculate reprojection error (in map pixels)
            error = np.linalg.norm(projected_gcp_coord - expected)
            errors.append(error)

        # Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        median_error = np.median(errors)

        # For real-world data with scale differences, accept larger errors
        # These are in map pixels
        assert mean_error < 20.0, f"Mean reprojection error too high: {mean_error:.2f} pixels"
        assert median_error < 15.0, f"Median reprojection error too high: {median_error:.2f} pixels"

    def test_forward_projection_batch(self, homography_matrix, valte_annotations):
        """Test batch projection of multiple camera pixels."""
        camera_pixels = np.array(
            [[[ann.pixel.x, ann.pixel.y]] for ann in valte_annotations], dtype=np.float32
        )

        map_coords = cv2.perspectiveTransform(camera_pixels, homography_matrix)

        assert map_coords.shape[0] == len(valte_annotations)
        assert np.all(np.isfinite(map_coords))


class TestInverseProjection:
    """Test projecting GCP (map) coordinates back to annotation (camera) pixels."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_annotations):
        """Compute and return homography matrix (annotation -> GCP)."""
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)
        return H

    @pytest.fixture
    def inverse_homography_matrix(self, homography_matrix):
        """Compute and return inverse homography matrix (GCP -> annotation)."""
        return np.linalg.inv(homography_matrix)

    def test_project_gcp_to_annotation(self, inverse_homography_matrix, map_point_registry):
        """Test projecting a GCP coordinate to annotation (camera) pixels."""
        # Get a GCP
        gcp = map_point_registry.points["PS1"]
        gcp_coord = np.array([[[gcp.pixel_x, gcp.pixel_y]]], dtype=np.float32)

        # Project to annotation (camera) space
        annotation_pt = cv2.perspectiveTransform(gcp_coord, inverse_homography_matrix)

        assert annotation_pt.shape == (1, 1, 2)
        ann_x, ann_y = annotation_pt[0, 0]

        # Annotation coordinates should be finite and within reasonable bounds
        # (may be slightly outside image for edge points)
        assert np.isfinite(ann_x) and np.isfinite(ann_y)
        assert -100 <= ann_x < 2020, f"Annotation x out of reasonable bounds: {ann_x}"
        assert -100 <= ann_y < 1180, f"Annotation y out of reasonable bounds: {ann_y}"

    def test_inverse_projection_of_gcps(
        self, inverse_homography_matrix, map_point_registry, valte_annotations
    ):
        """Test that projecting GCPs back yields original annotation pixels."""
        errors = []
        for annotation in valte_annotations:
            # GCP coord
            gcp = map_point_registry.points[annotation.gcp_id]
            gcp_coord = np.array([[[gcp.pixel_x, gcp.pixel_y]]], dtype=np.float32)

            # Project to annotation (camera) space
            projected_annotation_pt = cv2.perspectiveTransform(gcp_coord, inverse_homography_matrix)[
                0, 0
            ]

            # Expected annotation pixel
            expected = np.array([annotation.pixel.x, annotation.pixel.y])

            # Calculate reprojection error (in pixels)
            error = np.linalg.norm(projected_annotation_pt - expected)
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
    """Test round-trip projection: annotation -> GCP -> annotation."""

    @pytest.fixture
    def homography_matrices(self, map_point_registry, valte_annotations):
        """Compute and return forward and inverse homography matrices."""
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        H = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)[0]
        H_inv = np.linalg.inv(H)
        return H, H_inv

    def test_round_trip_annotation_to_gcp_to_annotation(self, homography_matrices, valte_annotations):
        """Test annotation -> GCP -> annotation preserves coordinates."""
        H, H_inv = homography_matrices

        errors = []
        for annotation in valte_annotations:
            # Original annotation (camera pixel)
            original = np.array([[[annotation.pixel.x, annotation.pixel.y]]], dtype=np.float32)

            # Project to GCP space
            gcp_coord = cv2.perspectiveTransform(original, H)

            # Project back to annotation space
            recovered = cv2.perspectiveTransform(gcp_coord, H_inv)

            # Calculate round-trip error
            error = np.linalg.norm(recovered[0, 0] - original[0, 0])
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip error should be small (in pixels)
        assert mean_error < 5.0, f"High mean round-trip error: {mean_error:.2f} pixels"
        assert max_error < 15.0, f"High max round-trip error: {max_error:.2f} pixels"

    def test_round_trip_gcp_to_annotation_to_gcp(
        self, homography_matrices, map_point_registry, valte_annotations
    ):
        """Test GCP -> annotation -> GCP preserves coordinates."""
        H, H_inv = homography_matrices

        errors = []
        for annotation in valte_annotations:
            # Original GCP coord (map pixels)
            gcp = map_point_registry.points[annotation.gcp_id]
            original = np.array([[[gcp.pixel_x, gcp.pixel_y]]], dtype=np.float32)

            # Project to annotation (camera) space
            annotation_pt = cv2.perspectiveTransform(original, H_inv)

            # Project back to GCP space
            recovered = cv2.perspectiveTransform(annotation_pt, H)

            # Calculate round-trip error (in map pixels)
            error = np.linalg.norm(recovered[0, 0] - original[0, 0])
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        # Round-trip error in map pixels should be acceptable
        assert mean_error < 5.0, f"High mean round-trip error: {mean_error:.2f} pixels"
        assert max_error < 15.0, f"High max round-trip error: {max_error:.2f} pixels"


class TestReprojectionErrorMetrics:
    """Test computing reprojection error metrics."""

    @pytest.fixture
    def homography_matrix(self, map_point_registry, valte_annotations):
        """Compute and return homography matrix (annotation -> GCP)."""
        annotation_pixels = np.array(
            [[ann.pixel.x, ann.pixel.y] for ann in valte_annotations], dtype=np.float32
        )

        gcp_coords = np.array(
            [
                [
                    map_point_registry.points[ann.gcp_id].pixel_x,
                    map_point_registry.points[ann.gcp_id].pixel_y,
                ]
                for ann in valte_annotations
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(annotation_pixels, gcp_coords, cv2.RANSAC, 50.0)
        return H

    def test_mean_reprojection_error(self, homography_matrix, map_point_registry, valte_annotations):
        """Test computing mean reprojection error across all annotation-GCP pairs."""
        errors = []

        for annotation in valte_annotations:
            # Project annotation (camera pixel) to GCP space
            annotation_pt = np.array([[[annotation.pixel.x, annotation.pixel.y]]], dtype=np.float32)
            projected_gcp_coord = cv2.perspectiveTransform(annotation_pt, homography_matrix)[0, 0]

            # Expected GCP coord
            expected_gcp = map_point_registry.points[annotation.gcp_id]
            expected = np.array([expected_gcp.pixel_x, expected_gcp.pixel_y])

            # Calculate error (in map pixels)
            error = np.linalg.norm(projected_gcp_coord - expected)
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        # Log statistics for debugging
        print("\nReprojection error statistics (map pixels):")
        print(f"  Mean: {mean_error:.2f}")
        print(f"  Max: {max_error:.2f}")
        print(f"  Std: {std_error:.2f}")

        # Thresholds for map pixel coordinates
        assert mean_error < 20.0, f"Mean reprojection error too high: {mean_error:.2f} pixels"
        assert max_error < 50.0, f"Max reprojection error too high: {max_error:.2f} pixels"

    def test_rmse_reprojection_error(self, homography_matrix, map_point_registry, valte_annotations):
        """Test computing RMSE (Root Mean Square Error) of reprojection."""
        squared_errors = []

        for annotation in valte_annotations:
            annotation_pt = np.array([[[annotation.pixel.x, annotation.pixel.y]]], dtype=np.float32)
            projected_gcp_coord = cv2.perspectiveTransform(annotation_pt, homography_matrix)[0, 0]

            expected_gcp = map_point_registry.points[annotation.gcp_id]
            expected = np.array([expected_gcp.pixel_x, expected_gcp.pixel_y])

            squared_error = np.sum((projected_gcp_coord - expected) ** 2)
            squared_errors.append(squared_error)

        rmse = np.sqrt(np.mean(squared_errors))

        # RMSE in map pixels
        assert rmse < 25.0, f"RMSE too high: {rmse:.2f} pixels"
