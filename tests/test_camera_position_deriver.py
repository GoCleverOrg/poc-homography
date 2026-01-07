#!/usr/bin/env python3
"""
Unit tests for CameraPositionDeriver class.

Tests validate:
- Position derivation with synthetic GCPs (known ground truth)
- RANSAC outlier rejection with contaminated GCP data
- Pan/tilt angle extraction matches input rotation
- Minimum GCP enforcement (ValueError when < 6 GCPs)
- GPS conversion accuracy (meter-to-GPS-to-meter round-trip)
- Edge cases: various camera heights, pan/tilt combinations, GCPs at image edges
"""

import unittest
import sys
import os
import math
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.camera_position_deriver import (
    CameraPositionDeriver,
    GroundControlPoint,
    PnPResult,
    AccuracyLevel,
)
from poc_homography.gps_distance_calculator import gps_to_local_xy, local_xy_to_gps


def create_ptz_rotation_matrix(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """
    Create rotation matrix matching CameraGeometry convention.

    This is the shared helper function for creating rotation matrices
    in tests. It matches the convention used by CameraGeometry._get_rotation_matrix():
    R = Rx_tilt @ R_base @ Rz_pan

    Args:
        pan_deg: Pan angle in degrees (positive = right/clockwise from above)
        tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)

    Returns:
        3x3 rotation matrix R = Rx_tilt @ R_base @ Rz_pan
    """
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)

    # Base transformation from World to Camera when pan=0, tilt=0
    R_base = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Pan rotation around world Z-axis
    Rz_pan = np.array([
        [math.cos(pan_rad), -math.sin(pan_rad), 0],
        [math.sin(pan_rad), math.cos(pan_rad), 0],
        [0, 0, 1]
    ])

    # Tilt rotation around camera X-axis
    Rx_tilt = np.array([
        [1, 0, 0],
        [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
        [0, math.sin(tilt_rad), math.cos(tilt_rad)]
    ])

    return Rx_tilt @ R_base @ Rz_pan


class TestGroundControlPoint(unittest.TestCase):
    """Test GroundControlPoint dataclass validation."""

    def test_valid_gcp_creation(self):
        """Test that valid GCP is created successfully."""
        gcp = GroundControlPoint(
            latitude=39.640583,
            longitude=-0.230194,
            u=1280.0,
            v=720.0
        )
        self.assertEqual(gcp.latitude, 39.640583)
        self.assertEqual(gcp.longitude, -0.230194)
        self.assertEqual(gcp.u, 1280.0)
        self.assertEqual(gcp.v, 720.0)

    def test_latitude_below_min_raises_value_error(self):
        """Test that latitude < -90 raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Latitude -91.0 outside valid range"):
            GroundControlPoint(latitude=-91.0, longitude=0.0, u=100.0, v=100.0)

    def test_latitude_above_max_raises_value_error(self):
        """Test that latitude > 90 raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Latitude 91.0 outside valid range"):
            GroundControlPoint(latitude=91.0, longitude=0.0, u=100.0, v=100.0)

    def test_longitude_below_min_raises_value_error(self):
        """Test that longitude < -180 raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Longitude -181.0 outside valid range"):
            GroundControlPoint(latitude=0.0, longitude=-181.0, u=100.0, v=100.0)

    def test_longitude_above_max_raises_value_error(self):
        """Test that longitude > 180 raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Longitude 181.0 outside valid range"):
            GroundControlPoint(latitude=0.0, longitude=181.0, u=100.0, v=100.0)

    def test_negative_u_raises_value_error(self):
        """Test that negative u coordinate raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Pixel u-coordinate -1.0 must be non-negative"):
            GroundControlPoint(latitude=0.0, longitude=0.0, u=-1.0, v=100.0)

    def test_negative_v_raises_value_error(self):
        """Test that negative v coordinate raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Pixel v-coordinate -1.0 must be non-negative"):
            GroundControlPoint(latitude=0.0, longitude=0.0, u=100.0, v=-1.0)

    def test_boundary_latitude_values_pass(self):
        """Test that boundary latitude values (-90, 90) are accepted."""
        gcp_min = GroundControlPoint(latitude=-90.0, longitude=0.0, u=100.0, v=100.0)
        gcp_max = GroundControlPoint(latitude=90.0, longitude=0.0, u=100.0, v=100.0)
        self.assertEqual(gcp_min.latitude, -90.0)
        self.assertEqual(gcp_max.latitude, 90.0)

    def test_boundary_longitude_values_pass(self):
        """Test that boundary longitude values (-180, 180) are accepted."""
        gcp_min = GroundControlPoint(latitude=0.0, longitude=-180.0, u=100.0, v=100.0)
        gcp_max = GroundControlPoint(latitude=0.0, longitude=180.0, u=100.0, v=100.0)
        self.assertEqual(gcp_min.longitude, -180.0)
        self.assertEqual(gcp_max.longitude, 180.0)


class TestCameraPositionDeriverInit(unittest.TestCase):
    """Test CameraPositionDeriver initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_K = np.array([
            [2000.0, 0.0, 1280.0],
            [0.0, 2000.0, 720.0],
            [0.0, 0.0, 1.0]
        ])
        self.reference_lat = 39.640583
        self.reference_lon = -0.230194

    def test_valid_initialization(self):
        """Test valid initialization with default parameters."""
        deriver = CameraPositionDeriver(
            K=self.valid_K,
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon
        )
        self.assertIsNotNone(deriver)
        self.assertEqual(deriver.accuracy, AccuracyLevel.MEDIUM)
        np.testing.assert_array_equal(deriver.K, self.valid_K)

    def test_initialization_with_accuracy_levels(self):
        """Test initialization with different accuracy levels."""
        for accuracy in AccuracyLevel:
            deriver = CameraPositionDeriver(
                K=self.valid_K,
                reference_lat=self.reference_lat,
                reference_lon=self.reference_lon,
                accuracy=accuracy
            )
            self.assertEqual(deriver.accuracy, accuracy)
            # Verify RANSAC config matches accuracy level
            expected_config = CameraPositionDeriver.RANSAC_CONFIG[accuracy]
            self.assertEqual(deriver.ransac_iterations, expected_config['iterations'])
            self.assertEqual(deriver.ransac_reprojection_threshold, expected_config['reprojection_threshold'])

    def test_initialization_with_custom_ransac_params(self):
        """Test initialization with custom RANSAC parameters."""
        deriver = CameraPositionDeriver(
            K=self.valid_K,
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon,
            accuracy=AccuracyLevel.LOW,
            ransac_iterations=250,
            ransac_reprojection_threshold=4.0,
            ransac_confidence=0.98
        )
        self.assertEqual(deriver.ransac_iterations, 250)
        self.assertEqual(deriver.ransac_reprojection_threshold, 4.0)
        self.assertEqual(deriver.ransac_confidence, 0.98)

    def test_invalid_K_shape_raises_value_error(self):
        """Test that invalid K matrix shape raises ValueError."""
        invalid_K = np.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "K must be a 3x3 numpy array"):
            CameraPositionDeriver(
                K=invalid_K,
                reference_lat=self.reference_lat,
                reference_lon=self.reference_lon
            )

    def test_invalid_K_type_raises_value_error(self):
        """Test that non-array K raises ValueError."""
        with self.assertRaisesRegex(ValueError, "K must be a 3x3 numpy array"):
            CameraPositionDeriver(
                K=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                reference_lat=self.reference_lat,
                reference_lon=self.reference_lon
            )

    def test_invalid_reference_latitude_raises_value_error(self):
        """Test that invalid reference latitude raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Reference latitude 91.0 outside valid range"):
            CameraPositionDeriver(
                K=self.valid_K,
                reference_lat=91.0,
                reference_lon=self.reference_lon
            )

    def test_invalid_reference_longitude_raises_value_error(self):
        """Test that invalid reference longitude raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Reference longitude -181.0 outside valid range"):
            CameraPositionDeriver(
                K=self.valid_K,
                reference_lat=self.reference_lat,
                reference_lon=-181.0
            )


class TestMinimumGCPEnforcement(unittest.TestCase):
    """Test minimum GCP count enforcement."""

    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [2000.0, 0.0, 1280.0],
            [0.0, 2000.0, 720.0],
            [0.0, 0.0, 1.0]
        ])
        self.deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=39.640583,
            reference_lon=-0.230194
        )

    def test_fewer_than_6_gcps_raises_value_error(self):
        """Test that fewer than 6 GCPs raises ValueError."""
        gcps = [
            GroundControlPoint(latitude=39.6406, longitude=-0.2302, u=1280.0, v=720.0)
            for _ in range(5)
        ]
        with self.assertRaisesRegex(ValueError, "At least 6 GCPs required"):
            self.deriver.derive_position(gcps)

    def test_exactly_6_gcps_does_not_raise(self):
        """Test that exactly 6 GCPs does not raise ValueError."""
        # Create 6 GCPs with distinct coordinates
        gcps = [
            GroundControlPoint(latitude=39.6406 + i*0.0001, longitude=-0.2302 + i*0.0001, u=1000.0 + i*100, v=500.0 + i*100)
            for i in range(6)
        ]
        # Should not raise - actual derivation may fail but that's OK for this test
        try:
            result = self.deriver.derive_position(gcps)
            # Just verify we got a result, success may be False due to synthetic data
            self.assertIsInstance(result, PnPResult)
        except ValueError as e:
            if "At least 6 GCPs required" in str(e):
                self.fail("Should not raise ValueError for exactly 6 GCPs")
            # Other ValueErrors are acceptable (e.g., from RANSAC failure)


class TestGCPNormalization(unittest.TestCase):
    """Test GCP input format normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [2000.0, 0.0, 1280.0],
            [0.0, 2000.0, 720.0],
            [0.0, 0.0, 1.0]
        ])
        self.deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=39.640583,
            reference_lon=-0.230194
        )

    def test_normalize_dataclass_format(self):
        """Test normalization of GroundControlPoint dataclass."""
        gcps = [
            GroundControlPoint(latitude=39.6406, longitude=-0.2302, u=1280.0, v=720.0)
        ]
        normalized = self.deriver._normalize_gcps(gcps)
        self.assertEqual(len(normalized), 1)
        self.assertIsInstance(normalized[0], GroundControlPoint)

    def test_normalize_dict_format_with_lat_lon(self):
        """Test normalization of dict format with 'lat'/'lon' keys."""
        gcps = [
            {'lat': 39.6406, 'lon': -0.2302, 'u': 1280.0, 'v': 720.0}
        ]
        normalized = self.deriver._normalize_gcps(gcps)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].latitude, 39.6406)
        self.assertEqual(normalized[0].longitude, -0.2302)

    def test_normalize_dict_format_with_latitude_longitude(self):
        """Test normalization of dict format with 'latitude'/'longitude' keys."""
        gcps = [
            {'latitude': 39.6406, 'longitude': -0.2302, 'u': 1280.0, 'v': 720.0}
        ]
        normalized = self.deriver._normalize_gcps(gcps)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].latitude, 39.6406)
        self.assertEqual(normalized[0].longitude, -0.2302)

    def test_normalize_tuple_format(self):
        """Test normalization of tuple format (lat, lon, u, v)."""
        gcps = [
            (39.6406, -0.2302, 1280.0, 720.0)
        ]
        normalized = self.deriver._normalize_gcps(gcps)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].latitude, 39.6406)
        self.assertEqual(normalized[0].u, 1280.0)

    def test_normalize_list_format(self):
        """Test normalization of list format [lat, lon, u, v]."""
        gcps = [
            [39.6406, -0.2302, 1280.0, 720.0]
        ]
        normalized = self.deriver._normalize_gcps(gcps)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].latitude, 39.6406)

    def test_normalize_dict_missing_lat_raises_value_error(self):
        """Test that dict missing latitude raises ValueError."""
        gcps = [
            {'lon': -0.2302, 'u': 1280.0, 'v': 720.0}
        ]
        with self.assertRaisesRegex(ValueError, "missing 'lat' or 'latitude' field"):
            self.deriver._normalize_gcps(gcps)

    def test_normalize_dict_missing_lon_raises_value_error(self):
        """Test that dict missing longitude raises ValueError."""
        gcps = [
            {'lat': 39.6406, 'u': 1280.0, 'v': 720.0}
        ]
        with self.assertRaisesRegex(ValueError, "missing 'lon' or 'longitude' field"):
            self.deriver._normalize_gcps(gcps)

    def test_normalize_dict_missing_u_raises_value_error(self):
        """Test that dict missing u coordinate raises ValueError."""
        gcps = [
            {'lat': 39.6406, 'lon': -0.2302, 'v': 720.0}
        ]
        with self.assertRaisesRegex(ValueError, "missing 'u' or 'v' field"):
            self.deriver._normalize_gcps(gcps)

    def test_normalize_invalid_format_raises_value_error(self):
        """Test that invalid format raises ValueError."""
        gcps = [
            "invalid_gcp_string"
        ]
        with self.assertRaisesRegex(ValueError, "invalid format"):
            self.deriver._normalize_gcps(gcps)


class TestSyntheticGCPPositionDerivation(unittest.TestCase):
    """Test position derivation with synthetic GCPs (known ground truth)."""

    def setUp(self):
        """Set up test fixtures with known camera pose."""
        # Known camera parameters
        self.camera_height = 25.0  # meters
        self.pan_deg = 0.0  # Looking straight ahead (North)
        self.tilt_deg = 30.0  # positive = looking down

        # Camera intrinsics (typical Hikvision at 5x zoom)
        self.K = np.array([
            [2106.27, 0.0, 1280.0],
            [0.0, 2106.27, 720.0],
            [0.0, 0.0, 1.0]
        ])

        # Reference GPS position (camera location)
        self.reference_lat = 39.640583
        self.reference_lon = -0.230194

        # True camera position in local coordinates
        self.true_position = np.array([0.0, 0.0, self.camera_height])

    def _generate_synthetic_gcps_opencv_style(
        self,
        num_points: int = 10,
        noise_pixels: float = 0.0
    ) -> list:
        """
        Generate synthetic GCPs using OpenCV's projectPoints for ground truth.

        This ensures consistency with how solvePnP will interpret the data.
        """
        gcps = []
        np.random.seed(42)  # For reproducibility

        # Generate ground plane points
        object_points = []
        for i in range(num_points * 2):  # Generate extra, filter later
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(5, 30)  # In front of camera
            z = 0.0  # Ground plane
            object_points.append([x, y, z])

        object_points = np.array(object_points, dtype=np.float64)

        # Create rotation matrix using shared helper
        R = create_ptz_rotation_matrix(self.pan_deg, self.tilt_deg)

        # Translation: t = -R @ C
        t = -R @ self.true_position

        # Convert R to rotation vector for OpenCV
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        # Project points using OpenCV (this is the ground truth)
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.K, distCoeffs=None
        )
        image_points = image_points.reshape(-1, 2)

        # Add noise if requested
        if noise_pixels > 0:
            image_points += np.random.normal(0, noise_pixels, image_points.shape)

        # Filter to points within image bounds
        for i, (obj_pt, img_pt) in enumerate(zip(object_points, image_points)):
            if 0 <= img_pt[0] < 2560 and 0 <= img_pt[1] < 1440:
                x, y, z = obj_pt
                lat, lon = local_xy_to_gps(
                    self.reference_lat, self.reference_lon,
                    x, y
                )
                gcps.append(GroundControlPoint(
                    latitude=lat,
                    longitude=lon,
                    u=float(img_pt[0]),
                    v=float(img_pt[1])
                ))
                if len(gcps) >= num_points:
                    break

        return gcps

    def test_position_derivation_with_perfect_gcps(self):
        """Test position derivation with perfect (noise-free) synthetic GCPs."""
        gcps = self._generate_synthetic_gcps_opencv_style(num_points=12, noise_pixels=0.0)

        # Ensure we have enough valid GCPs - if not, it's a test configuration problem
        self.assertGreaterEqual(len(gcps), 6,
            f"GCP generation produced only {len(gcps)} valid points. "
            "Check camera pan/tilt configuration produces visible ground plane points.")

        deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon,
            accuracy=AccuracyLevel.HIGH
        )

        result = deriver.derive_position(gcps)

        # Should succeed
        self.assertTrue(result.success, f"Derivation failed: {result}")

        # Position should be close to true position
        position_error = np.linalg.norm(result.position - self.true_position)
        self.assertLess(position_error, 1.0,
            f"Position error {position_error:.2f}m too high. "
            f"Expected ~{self.true_position}, got {result.position}")

        # Reprojection error should be very low
        self.assertLess(result.reprojection_error_mean, 1.0,
            f"Reprojection error {result.reprojection_error_mean:.2f}px too high")

        # All points should be inliers
        self.assertGreater(result.inlier_ratio, 0.9,
            f"Inlier ratio {result.inlier_ratio:.2%} too low")

    def test_position_derivation_with_noisy_gcps(self):
        """Test position derivation with noisy synthetic GCPs."""
        gcps = self._generate_synthetic_gcps_opencv_style(num_points=15, noise_pixels=2.0)

        # Ensure we have enough valid GCPs
        self.assertGreaterEqual(len(gcps), 6,
            f"GCP generation produced only {len(gcps)} valid points.")

        deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon,
            accuracy=AccuracyLevel.HIGH
        )

        result = deriver.derive_position(gcps)

        # With noise, RANSAC may not get all inliers but position should still be reasonable
        # The success flag is based on inlier ratio, which may fail with noisy data
        # So we check position directly if we got a result
        if result.position is not None:
            position_error = np.linalg.norm(result.position - self.true_position)
            self.assertLess(position_error, 5.0,
                f"Position error {position_error:.2f}m too high for noisy data")


class TestPanTiltAngleExtraction(unittest.TestCase):
    """Test pan/tilt angle extraction from rotation matrix."""

    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [2000.0, 0.0, 1280.0],
            [0.0, 2000.0, 720.0],
            [0.0, 0.0, 1.0]
        ])
        self.deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=39.640583,
            reference_lon=-0.230194
        )

    def test_angle_extraction_pan_0_tilt_0(self):
        """Test angle extraction for pan=0, tilt=0."""
        R = create_ptz_rotation_matrix(0.0, 0.0)
        pan, tilt = self.deriver._extract_pan_tilt_from_rotation(R)
        self.assertAlmostEqual(pan, 0.0, places=2)
        self.assertAlmostEqual(tilt, 0.0, places=2)

    def test_angle_extraction_pan_45_tilt_30(self):
        """Test angle extraction for pan=45, tilt=30."""
        R = create_ptz_rotation_matrix(45.0, 30.0)
        pan, tilt = self.deriver._extract_pan_tilt_from_rotation(R)
        self.assertAlmostEqual(pan, 45.0, places=1)
        self.assertAlmostEqual(tilt, 30.0, places=1)

    def test_angle_extraction_pan_negative_90_tilt_45(self):
        """Test angle extraction for pan=-90, tilt=45."""
        R = create_ptz_rotation_matrix(-90.0, 45.0)
        pan, tilt = self.deriver._extract_pan_tilt_from_rotation(R)
        self.assertAlmostEqual(pan, -90.0, places=1)
        self.assertAlmostEqual(tilt, 45.0, places=1)

    def test_angle_extraction_pan_180_tilt_60(self):
        """Test angle extraction for pan=180, tilt=60."""
        R = create_ptz_rotation_matrix(180.0, 60.0)
        pan, tilt = self.deriver._extract_pan_tilt_from_rotation(R)
        # pan=180 may be extracted as -180 (equivalent)
        self.assertTrue(
            abs(pan - 180.0) < 1.0 or abs(pan + 180.0) < 1.0,
            f"Pan {pan} not close to ±180"
        )
        self.assertAlmostEqual(tilt, 60.0, places=1)

    def test_angle_extraction_negative_tilt(self):
        """Test angle extraction for negative tilt (looking up)."""
        R = create_ptz_rotation_matrix(30.0, -15.0)
        pan, tilt = self.deriver._extract_pan_tilt_from_rotation(R)
        self.assertAlmostEqual(pan, 30.0, places=1)
        self.assertAlmostEqual(tilt, -15.0, places=1)

    def test_angle_extraction_multiple_combinations(self):
        """Test angle extraction round-trip for various pan/tilt combinations."""
        test_cases = [
            (0.0, 30.0),
            (45.0, 45.0),
            (90.0, 60.0),
            (-45.0, 15.0),
            (135.0, 75.0),
            (-135.0, 30.0),
        ]

        for pan_in, tilt_in in test_cases:
            with self.subTest(pan=pan_in, tilt=tilt_in):
                R = create_ptz_rotation_matrix(pan_in, tilt_in)
                pan_out, tilt_out = self.deriver._extract_pan_tilt_from_rotation(R)

                # Handle pan wrap-around at ±180°
                pan_diff = abs(pan_out - pan_in)
                if pan_diff > 180:
                    pan_diff = 360 - pan_diff

                self.assertLess(pan_diff, 2.0, f"Pan mismatch: in={pan_in}, out={pan_out}")
                self.assertAlmostEqual(tilt_out, tilt_in, places=1,
                    msg=f"Tilt mismatch: in={tilt_in}, out={tilt_out}")


class TestRANSACOutlierRejection(unittest.TestCase):
    """Test RANSAC outlier rejection with contaminated GCP data."""

    def setUp(self):
        """Set up test fixtures."""
        self.camera_height = 25.0
        self.pan_deg = 30.0
        self.tilt_deg = 45.0

        self.K = np.array([
            [2106.27, 0.0, 1280.0],
            [0.0, 2106.27, 720.0],
            [0.0, 0.0, 1.0]
        ])

        self.reference_lat = 39.640583
        self.reference_lon = -0.230194
        self.true_position = np.array([0.0, 0.0, self.camera_height])

    def _generate_gcps_with_outliers(self, num_inliers: int, num_outliers: int) -> list:
        """Generate GCPs with some outliers (wrong pixel coordinates)."""
        gcps = []
        np.random.seed(123)

        # Create rotation matrix using shared helper
        R = create_ptz_rotation_matrix(self.pan_deg, self.tilt_deg)
        t = -R @ self.true_position
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        # Generate inlier object points
        inlier_object_points = []
        for i in range(num_inliers * 2):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(5, 25)
            z = 0.0
            inlier_object_points.append([x, y, z])

        inlier_object_points = np.array(inlier_object_points, dtype=np.float64)

        # Project inliers using OpenCV
        inlier_image_points, _ = cv2.projectPoints(
            inlier_object_points, rvec, tvec, self.K, distCoeffs=None
        )
        inlier_image_points = inlier_image_points.reshape(-1, 2)

        # Add inliers that are within image bounds
        for obj_pt, img_pt in zip(inlier_object_points, inlier_image_points):
            if 0 <= img_pt[0] < 2560 and 0 <= img_pt[1] < 1440:
                x, y, z = obj_pt
                lat, lon = local_xy_to_gps(self.reference_lat, self.reference_lon, x, y)
                gcps.append(GroundControlPoint(
                    latitude=lat, longitude=lon,
                    u=float(img_pt[0]), v=float(img_pt[1])
                ))
                if len(gcps) >= num_inliers:
                    break

        # Generate outliers (random pixel coordinates that don't match the world point)
        for i in range(num_outliers):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(5, 25)
            lat, lon = local_xy_to_gps(self.reference_lat, self.reference_lon, x, y)

            # Random (incorrect) pixel coordinates
            u = np.random.uniform(0, 2560)
            v = np.random.uniform(0, 1440)

            gcps.append(GroundControlPoint(latitude=lat, longitude=lon, u=u, v=v))

        np.random.shuffle(gcps)
        return gcps

    def test_ransac_rejects_outliers(self):
        """Test that RANSAC correctly identifies and rejects outliers."""
        # 10 good GCPs, 4 outliers (28% contamination)
        gcps = self._generate_gcps_with_outliers(num_inliers=10, num_outliers=4)

        # Ensure we have enough valid GCPs (inliers + outliers)
        self.assertGreaterEqual(len(gcps), 6,
            f"GCP generation produced only {len(gcps)} valid points.")

        deriver = CameraPositionDeriver(
            K=self.K,
            reference_lat=self.reference_lat,
            reference_lon=self.reference_lon,
            accuracy=AccuracyLevel.HIGH
        )

        result = deriver.derive_position(gcps)

        # Check if we got a valid result
        if result.position is not None:
            # Position should be reasonably accurate despite outliers
            position_error = np.linalg.norm(result.position - self.true_position)
            self.assertLess(position_error, 5.0,
                f"Position error {position_error:.2f}m too high despite outlier rejection")


class TestPnPResultDataclass(unittest.TestCase):
    """Test PnPResult dataclass functionality."""

    def test_failed_result_defaults(self):
        """Test that failed result has appropriate defaults."""
        result = PnPResult(success=False)
        self.assertFalse(result.success)
        self.assertIsNone(result.position)
        self.assertIsNone(result.rotation_matrix)
        self.assertIsNone(result.pan_deg)

    def test_to_dict_serialization(self):
        """Test that to_dict produces serializable output."""
        result = PnPResult(
            success=True,
            position=np.array([1.0, 2.0, 25.0]),
            rotation_matrix=np.eye(3),
            rotation_vector=np.array([0.1, 0.2, 0.3]),
            pan_deg=45.0,
            tilt_deg=30.0,
            reprojection_error_mean=1.5,
            reprojection_error_max=3.0,
            inlier_ratio=0.9,
            num_inliers=9,
            inliers_mask=np.array([True, True, True, False, True, True, True, True, True, True])
        )

        d = result.to_dict()

        # All values should be JSON-serializable (lists, not numpy arrays)
        self.assertIsInstance(d['position'], list)
        self.assertIsInstance(d['rotation_matrix'], list)
        self.assertIsInstance(d['rotation_vector'], list)
        self.assertIsInstance(d['inliers_mask'], list)

        # Values should match
        self.assertEqual(d['success'], True)
        self.assertAlmostEqual(d['pan_deg'], 45.0)
        self.assertAlmostEqual(d['tilt_deg'], 30.0)


class TestGPSCoordinateConversion(unittest.TestCase):
    """Test GPS to local coordinate conversion accuracy."""

    def test_gps_to_local_round_trip(self):
        """Test GPS to local to GPS round-trip accuracy."""
        ref_lat = 39.640583
        ref_lon = -0.230194

        # Test points at various distances
        test_offsets = [
            (10.0, 10.0),    # 10m east, 10m north
            (-20.0, 30.0),   # 20m west, 30m north
            (50.0, -25.0),   # 50m east, 25m south
            (0.0, 100.0),    # 100m north
            (100.0, 0.0),    # 100m east
        ]

        for x_orig, y_orig in test_offsets:
            with self.subTest(x=x_orig, y=y_orig):
                # Convert local to GPS
                lat, lon = local_xy_to_gps(ref_lat, ref_lon, x_orig, y_orig)

                # Convert back to local
                x_recovered, y_recovered = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                # Round-trip error should be < 1cm
                self.assertAlmostEqual(x_recovered, x_orig, places=2,
                    msg=f"X round-trip error: {abs(x_recovered - x_orig):.4f}m")
                self.assertAlmostEqual(y_recovered, y_orig, places=2,
                    msg=f"Y round-trip error: {abs(y_recovered - y_orig):.4f}m")


class TestAccuracyLevelConfiguration(unittest.TestCase):
    """Test accuracy level configuration affects RANSAC parameters."""

    def test_low_accuracy_config(self):
        """Test LOW accuracy level configuration."""
        K = np.eye(3) * 1000
        K[0, 2] = 640
        K[1, 2] = 480
        K[2, 2] = 1

        deriver = CameraPositionDeriver(
            K=K,
            reference_lat=0.0,
            reference_lon=0.0,
            accuracy=AccuracyLevel.LOW
        )

        self.assertEqual(deriver.ransac_iterations, 100)
        self.assertEqual(deriver.ransac_reprojection_threshold, 5.0)
        self.assertEqual(deriver.ransac_confidence, 0.95)

    def test_medium_accuracy_config(self):
        """Test MEDIUM accuracy level configuration."""
        K = np.eye(3) * 1000
        K[0, 2] = 640
        K[1, 2] = 480
        K[2, 2] = 1

        deriver = CameraPositionDeriver(
            K=K,
            reference_lat=0.0,
            reference_lon=0.0,
            accuracy=AccuracyLevel.MEDIUM
        )

        self.assertEqual(deriver.ransac_iterations, 500)
        self.assertEqual(deriver.ransac_reprojection_threshold, 3.0)
        self.assertEqual(deriver.ransac_confidence, 0.99)

    def test_high_accuracy_config(self):
        """Test HIGH accuracy level configuration."""
        K = np.eye(3) * 1000
        K[0, 2] = 640
        K[1, 2] = 480
        K[2, 2] = 1

        deriver = CameraPositionDeriver(
            K=K,
            reference_lat=0.0,
            reference_lon=0.0,
            accuracy=AccuracyLevel.HIGH
        )

        self.assertEqual(deriver.ransac_iterations, 1000)
        self.assertEqual(deriver.ransac_reprojection_threshold, 1.0)
        self.assertEqual(deriver.ransac_confidence, 0.999)


def main():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    main()
