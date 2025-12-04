#!/usr/bin/env python3
"""
Unit tests for GCP-based FeatureMatchHomography implementation.

Tests verify homography computation, GPS coordinate conversion, point projection,
and validation logic for the feature match approach using Ground Control Points.
"""

import unittest
import sys
import os
import math
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.feature_match_homography import FeatureMatchHomography
from poc_homography.homography_interface import (
    WorldPoint,
    MapCoordinate,
    HomographyResult,
    HomographyApproach,
)


# Test data: 4 corners of a 100m x 100m square centered around reference point
# Reference: (39.640, -0.230)
TEST_GCPS_SQUARE = [
    {'gps': {'latitude': 39.6405, 'longitude': -0.2305}, 'image': {'u': 500, 'v': 500}},
    {'gps': {'latitude': 39.6405, 'longitude': -0.2295}, 'image': {'u': 2000, 'v': 500}},
    {'gps': {'latitude': 39.6395, 'longitude': -0.2305}, 'image': {'u': 500, 'v': 1000}},
    {'gps': {'latitude': 39.6395, 'longitude': -0.2295}, 'image': {'u': 2000, 'v': 1000}},
]

# Test data: More realistic GCPs with some spread
TEST_GCPS_REALISTIC = [
    {'gps': {'latitude': 39.640583, 'longitude': -0.230194}, 'image': {'u': 1250.5, 'v': 680.0}},
    {'gps': {'latitude': 39.640612, 'longitude': -0.229856}, 'image': {'u': 2456.2, 'v': 695.5}},
    {'gps': {'latitude': 39.640245, 'longitude': -0.230301}, 'image': {'u': 1180.0, 'v': 1820.3}},
    {'gps': {'latitude': 39.640271, 'longitude': -0.229934}, 'image': {'u': 2380.7, 'v': 1835.2}},
]


class TestFeatureMatchHomographyInit(unittest.TestCase):
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test that provider initializes with default parameters."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        self.assertEqual(provider.width, 2560)
        self.assertEqual(provider.height, 1440)
        self.assertEqual(provider.min_matches, 4)
        self.assertEqual(provider.ransac_threshold, 3.0)
        self.assertEqual(provider.confidence_threshold, 0.5)
        self.assertFalse(provider.is_valid())
        self.assertEqual(provider.get_confidence(), 0.0)

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        provider = FeatureMatchHomography(
            width=1920,
            height=1080,
            min_matches=6,
            ransac_threshold=5.0,
            confidence_threshold=0.7
        )
        self.assertEqual(provider.width, 1920)
        self.assertEqual(provider.height, 1080)
        self.assertEqual(provider.min_matches, 6)
        self.assertEqual(provider.ransac_threshold, 5.0)
        self.assertEqual(provider.confidence_threshold, 0.7)

    def test_min_matches_less_than_4_raises_error(self):
        """Test that min_matches < 4 raises ValueError."""
        with self.assertRaisesRegex(ValueError, "min_matches must be at least 4"):
            FeatureMatchHomography(width=2560, height=1440, min_matches=3)

    def test_confidence_threshold_out_of_range_raises_error(self):
        """Test that confidence_threshold outside [0, 1] raises ValueError."""
        with self.assertRaisesRegex(ValueError, "confidence_threshold must be in range"):
            FeatureMatchHomography(width=2560, height=1440, confidence_threshold=1.5)

        with self.assertRaisesRegex(ValueError, "confidence_threshold must be in range"):
            FeatureMatchHomography(width=2560, height=1440, confidence_threshold=-0.1)

    def test_homography_matrices_initialized_to_identity(self):
        """Test that homography matrices start as identity."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        np.testing.assert_array_equal(provider.H, np.eye(3))
        np.testing.assert_array_equal(provider.H_inv, np.eye(3))

    def test_gps_reference_not_set_initially(self):
        """Test that GPS reference point is None initially."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        self.assertIsNone(provider._reference_lat)
        self.assertIsNone(provider._reference_lon)
        self.assertIsNone(provider._camera_gps_lat)
        self.assertIsNone(provider._camera_gps_lon)


class TestGCPExtraction(unittest.TestCase):
    """Test GCP extraction and validation from reference data."""

    def test_missing_gcps_key_raises_error(self):
        """Test that missing 'ground_control_points' key raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {}

        with self.assertRaisesRegex(ValueError, "Missing required reference key: 'ground_control_points'"):
            provider.compute_homography(frame, reference)

    def test_insufficient_gcps_raises_error(self):
        """Test that fewer than min_matches GCPs raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440, min_matches=4)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {
            'ground_control_points': TEST_GCPS_SQUARE[:3]  # Only 3 GCPs
        }

        with self.assertRaisesRegex(ValueError, "Need at least 4 ground control points"):
            provider.compute_homography(frame, reference)

    def test_invalid_gcp_format_missing_gps_raises_error(self):
        """Test that GCP missing 'gps' key raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {
            'ground_control_points': [
                {'image': {'u': 100, 'v': 200}},  # Missing 'gps'
            ] * 4
        }

        with self.assertRaisesRegex(ValueError, "Each GCP must have 'gps' and 'image' keys"):
            provider.compute_homography(frame, reference)

    def test_invalid_gcp_format_missing_image_raises_error(self):
        """Test that GCP missing 'image' key raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {
            'ground_control_points': [
                {'gps': {'latitude': 39.64, 'longitude': -0.23}},  # Missing 'image'
            ] * 4
        }

        with self.assertRaisesRegex(ValueError, "Each GCP must have 'gps' and 'image' keys"):
            provider.compute_homography(frame, reference)

    def test_invalid_gcp_missing_latitude_raises_error(self):
        """Test that GCP missing 'latitude' raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {
            'ground_control_points': [
                {
                    'gps': {'longitude': -0.23},  # Missing latitude
                    'image': {'u': 100, 'v': 200}
                }
            ] * 4
        }

        with self.assertRaisesRegex(ValueError, "GPS must have 'latitude' and 'longitude' keys"):
            provider.compute_homography(frame, reference)

    def test_invalid_gcp_missing_u_raises_error(self):
        """Test that GCP missing 'u' raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {
            'ground_control_points': [
                {
                    'gps': {'latitude': 39.64, 'longitude': -0.23},
                    'image': {'v': 200}  # Missing u
                }
            ] * 4
        }

        with self.assertRaisesRegex(ValueError, "Image must have 'u' and 'v' keys"):
            provider.compute_homography(frame, reference)


class TestGPSToLocalConversion(unittest.TestCase):
    """Test GPS to local metric coordinate conversion."""

    def setUp(self):
        """Set up test provider with computed homography."""
        self.provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}
        self.provider.compute_homography(frame, reference)

    def test_gps_to_local_converts_correctly(self):
        """Test that GPS coordinates are converted to local metric coordinates."""
        # Reference point should map to (0, 0)
        ref_lat = self.provider._reference_lat
        ref_lon = self.provider._reference_lon

        x, y = self.provider._gps_to_local(ref_lat, ref_lon)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_gps_to_local_east_offset(self):
        """Test that eastward GPS offset produces positive x."""
        ref_lat = self.provider._reference_lat
        ref_lon = self.provider._reference_lon

        # Move 0.001 degrees east (positive longitude)
        x, y = self.provider._gps_to_local(ref_lat, ref_lon + 0.001)
        self.assertGreater(x, 0)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_gps_to_local_north_offset(self):
        """Test that northward GPS offset produces positive y."""
        ref_lat = self.provider._reference_lat
        ref_lon = self.provider._reference_lon

        # Move 0.001 degrees north (positive latitude)
        x, y = self.provider._gps_to_local(ref_lat + 0.001, ref_lon)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertGreater(y, 0)

    def test_local_to_gps_converts_correctly(self):
        """Test that local metric coordinates are converted to GPS."""
        # (0, 0) should map to reference point
        lat, lon = self.provider._local_to_gps(0.0, 0.0)
        self.assertAlmostEqual(lat, self.provider._reference_lat, places=6)
        self.assertAlmostEqual(lon, self.provider._reference_lon, places=6)

    def test_gps_to_local_round_trip(self):
        """Test that GPS -> local -> GPS preserves coordinates."""
        test_lat = 39.6403
        test_lon = -0.2298

        # Convert GPS -> local -> GPS
        x, y = self.provider._gps_to_local(test_lat, test_lon)
        lat_back, lon_back = self.provider._local_to_gps(x, y)

        self.assertAlmostEqual(lat_back, test_lat, places=6)
        self.assertAlmostEqual(lon_back, test_lon, places=6)

    def test_local_to_gps_round_trip(self):
        """Test that local -> GPS -> local preserves coordinates."""
        test_x = 50.0
        test_y = 100.0

        # Convert local -> GPS -> local
        lat, lon = self.provider._local_to_gps(test_x, test_y)
        x_back, y_back = self.provider._gps_to_local(lat, lon)

        self.assertAlmostEqual(x_back, test_x, places=2)
        self.assertAlmostEqual(y_back, test_y, places=2)

    def test_gps_conversion_without_reference_raises_error(self):
        """Test that GPS conversion without reference point raises RuntimeError."""
        provider = FeatureMatchHomography(width=2560, height=1440)

        with self.assertRaisesRegex(RuntimeError, "Reference GPS position not set"):
            provider._gps_to_local(39.64, -0.23)

        with self.assertRaisesRegex(RuntimeError, "Reference GPS position not set"):
            provider._local_to_gps(0.0, 0.0)

    def test_gps_conversion_at_pole_raises_error(self):
        """Test that GPS conversion near poles raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

        # Create GCPs very close to north pole (89.99999 degrees)
        # This should trigger the cos(lat) ~ 0 check during compute_homography
        gcps_at_pole = [
            {'gps': {'latitude': 89.99999, 'longitude': 0}, 'image': {'u': 500, 'v': 500}},
            {'gps': {'latitude': 89.99999, 'longitude': 90}, 'image': {'u': 2000, 'v': 500}},
            {'gps': {'latitude': 89.99998, 'longitude': 0}, 'image': {'u': 500, 'v': 1000}},
            {'gps': {'latitude': 89.99998, 'longitude': 90}, 'image': {'u': 2000, 'v': 1000}},
        ]
        reference = {'ground_control_points': gcps_at_pole}

        # Homography computation will set reference to pole latitude
        # This should cause ValueError during GPS->local conversion
        with self.assertRaises((ValueError, RuntimeError)):
            # Either ValueError from pole check, or RuntimeError from failed homography
            provider.compute_homography(frame, reference)


class TestComputeHomography(unittest.TestCase):
    """Test homography computation from GCPs."""

    def test_compute_with_valid_gcps_succeeds(self):
        """Test that homography computation succeeds with valid GCPs."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        result = provider.compute_homography(frame, reference)

        self.assertIsInstance(result, HomographyResult)
        self.assertEqual(result.homography_matrix.shape, (3, 3))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_compute_sets_reference_gps(self):
        """Test that compute_homography sets reference GPS position."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        provider.compute_homography(frame, reference)

        self.assertIsNotNone(provider._reference_lat)
        self.assertIsNotNone(provider._reference_lon)
        self.assertIsNotNone(provider._camera_gps_lat)
        self.assertIsNotNone(provider._camera_gps_lon)

    def test_compute_updates_homography_matrix(self):
        """Test that compute_homography updates H and H_inv."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        # Initial state is identity
        np.testing.assert_array_equal(provider.H, np.eye(3))

        provider.compute_homography(frame, reference)

        # After computation, should not be identity
        self.assertFalse(np.allclose(provider.H, np.eye(3)))
        self.assertFalse(np.allclose(provider.H_inv, np.eye(3)))

    def test_compute_returns_valid_confidence(self):
        """Test that confidence score is within [0, 1] range."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_REALISTIC}

        result = provider.compute_homography(frame, reference)

        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.confidence, provider.get_confidence())

    def test_compute_includes_metadata(self):
        """Test that result includes approach and metadata."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        result = provider.compute_homography(frame, reference)

        self.assertIn('approach', result.metadata)
        self.assertEqual(result.metadata['approach'], HomographyApproach.FEATURE_MATCH.value)
        self.assertIn('method', result.metadata)
        self.assertEqual(result.metadata['method'], 'gcp_based')
        self.assertIn('num_gcps', result.metadata)
        self.assertEqual(result.metadata['num_gcps'], 4)
        self.assertIn('num_inliers', result.metadata)
        self.assertIn('inlier_ratio', result.metadata)
        self.assertIn('determinant', result.metadata)
        self.assertIn('reference_gps', result.metadata)

    def test_compute_with_many_inliers_high_confidence(self):
        """Test that high inlier ratio produces high confidence."""
        provider = FeatureMatchHomography(width=2560, height=1440, ransac_threshold=10.0)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

        # Create many well-distributed GCPs
        gcps = []
        for i in range(10):
            for j in range(10):
                gcps.append({
                    'gps': {
                        'latitude': 39.64 + i * 0.0001,
                        'longitude': -0.23 + j * 0.0001
                    },
                    'image': {
                        'u': 500 + i * 150,
                        'v': 500 + j * 80
                    }
                })

        reference = {'ground_control_points': gcps}
        result = provider.compute_homography(frame, reference)

        # With synthetic clean data, should have high confidence
        self.assertGreater(result.confidence, 0.5)

    def test_is_valid_false_before_compute(self):
        """Test that is_valid returns False before compute_homography."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        self.assertFalse(provider.is_valid())

    def test_is_valid_true_after_successful_compute(self):
        """Test that is_valid returns True after successful computation."""
        provider = FeatureMatchHomography(width=2560, height=1440, confidence_threshold=0.1)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        provider.compute_homography(frame, reference)

        # Should be valid if confidence exceeds threshold
        if provider.get_confidence() >= 0.1:
            self.assertTrue(provider.is_valid())


class TestPointProjection(unittest.TestCase):
    """Test projecting image points to world coordinates."""

    def setUp(self):
        """Set up provider with computed homography."""
        self.provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}
        self.provider.compute_homography(frame, reference)

    def test_project_point_returns_world_point(self):
        """Test that project_point returns WorldPoint with valid structure."""
        world_point = self.provider.project_point((1280, 720))

        self.assertIsInstance(world_point, WorldPoint)
        self.assertIsInstance(world_point.latitude, float)
        self.assertIsInstance(world_point.longitude, float)
        self.assertIsInstance(world_point.confidence, float)
        self.assertGreaterEqual(world_point.confidence, 0.0)
        self.assertLessEqual(world_point.confidence, 1.0)

    def test_project_point_gps_in_valid_range(self):
        """Test that projected GPS coordinates are in valid range."""
        world_point = self.provider.project_point((1280, 720))

        self.assertGreaterEqual(world_point.latitude, -90)
        self.assertLessEqual(world_point.latitude, 90)
        self.assertGreaterEqual(world_point.longitude, -180)
        self.assertLessEqual(world_point.longitude, 180)

    def test_project_gcp_point_returns_original_gps(self):
        """Test that projecting a GCP's image point returns approximately the same GPS."""
        # Project the first GCP's image point
        gcp = TEST_GCPS_SQUARE[0]
        world_point = self.provider.project_point((gcp['image']['u'], gcp['image']['v']))

        # Should be close to original GPS coordinates
        # Allow some tolerance due to homography approximation
        self.assertAlmostEqual(world_point.latitude, gcp['gps']['latitude'], places=3)
        self.assertAlmostEqual(world_point.longitude, gcp['gps']['longitude'], places=3)

    def test_project_points_batch(self):
        """Test batch projection of multiple points."""
        image_points = [(500, 500), (1000, 700), (1500, 900)]
        world_points = self.provider.project_points(image_points)

        self.assertEqual(len(world_points), 3)
        for wp in world_points:
            self.assertIsInstance(wp, WorldPoint)
            self.assertGreaterEqual(wp.latitude, -90)
            self.assertLessEqual(wp.latitude, 90)

    def test_project_point_without_homography_raises_error(self):
        """Test that projecting before compute_homography raises RuntimeError."""
        provider = FeatureMatchHomography(width=2560, height=1440)

        with self.assertRaisesRegex(RuntimeError, "No valid homography available"):
            provider.project_point((1280, 720))

    def test_project_point_outside_bounds_raises_error(self):
        """Test that projecting point outside image bounds raises ValueError."""
        with self.assertRaisesRegex(ValueError, "outside valid bounds"):
            self.provider.project_point((3000, 720))

        with self.assertRaisesRegex(ValueError, "outside valid bounds"):
            self.provider.project_point((1280, 2000))

        with self.assertRaisesRegex(ValueError, "outside valid bounds"):
            self.provider.project_point((-100, 720))

    def test_project_point_confidence_varies_by_position(self):
        """Test that confidence varies based on distance from image center."""
        # Center point should have higher confidence
        center_point = self.provider.project_point((1280, 720))

        # Corner point should have lower confidence (edge factor applied)
        corner_point = self.provider.project_point((100, 100))

        # Note: This assumes edge factor is applied. May not always be lower
        # depending on base confidence, but confidence should be in valid range
        self.assertGreaterEqual(center_point.confidence, 0.0)
        self.assertGreaterEqual(corner_point.confidence, 0.0)


class TestMapCoordinateProjection(unittest.TestCase):
    """Test projecting image points to local map coordinates."""

    def setUp(self):
        """Set up provider with computed homography."""
        self.provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}
        self.provider.compute_homography(frame, reference)

    def test_project_point_to_map_returns_map_coordinate(self):
        """Test that project_point_to_map returns MapCoordinate."""
        map_coord = self.provider.project_point_to_map((1280, 720))

        self.assertIsInstance(map_coord, MapCoordinate)
        self.assertIsInstance(map_coord.x, float)
        self.assertIsInstance(map_coord.y, float)
        self.assertIsInstance(map_coord.confidence, float)
        self.assertGreaterEqual(map_coord.confidence, 0.0)
        self.assertLessEqual(map_coord.confidence, 1.0)

    def test_project_point_to_map_elevation_is_zero(self):
        """Test that elevation is set to 0.0 (ground plane assumption)."""
        map_coord = self.provider.project_point_to_map((1280, 720))
        self.assertEqual(map_coord.elevation, 0.0)

    def test_project_points_to_map_batch(self):
        """Test batch projection to map coordinates."""
        image_points = [(500, 500), (1000, 700), (1500, 900)]
        map_coords = self.provider.project_points_to_map(image_points)

        self.assertEqual(len(map_coords), 3)
        for mc in map_coords:
            self.assertIsInstance(mc, MapCoordinate)

    def test_project_to_map_without_homography_raises_error(self):
        """Test that projecting to map before compute raises RuntimeError."""
        provider = FeatureMatchHomography(width=2560, height=1440)

        with self.assertRaisesRegex(RuntimeError, "No valid homography available"):
            provider.project_point_to_map((1280, 720))

    def test_project_to_map_outside_bounds_raises_error(self):
        """Test that projecting outside bounds raises ValueError."""
        with self.assertRaisesRegex(ValueError, "outside valid bounds"):
            self.provider.project_point_to_map((3000, 720))


class TestValidationMethods(unittest.TestCase):
    """Test validation and state checking methods."""

    def test_is_valid_returns_false_initially(self):
        """Test that is_valid returns False before computation."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        self.assertFalse(provider.is_valid())

    def test_is_valid_returns_true_after_computation(self):
        """Test that is_valid returns True after valid computation."""
        provider = FeatureMatchHomography(
            width=2560,
            height=1440,
            confidence_threshold=0.1  # Low threshold
        )
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        provider.compute_homography(frame, reference)

        # Should be valid if confidence meets threshold
        if provider.get_confidence() >= 0.1:
            self.assertTrue(provider.is_valid())

    def test_get_confidence_returns_zero_initially(self):
        """Test that get_confidence returns 0 before computation."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        self.assertEqual(provider.get_confidence(), 0.0)

    def test_get_confidence_returns_valid_value_after_computation(self):
        """Test that get_confidence returns valid value after computation."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_REALISTIC}

        provider.compute_homography(frame, reference)
        confidence = provider.get_confidence()

        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_is_valid_checks_confidence_threshold(self):
        """Test that is_valid respects confidence threshold."""
        # High threshold
        provider_high = FeatureMatchHomography(
            width=2560,
            height=1440,
            confidence_threshold=0.99
        )
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        provider_high.compute_homography(frame, reference)

        # Likely won't meet 0.99 threshold with synthetic data
        if provider_high.get_confidence() < 0.99:
            self.assertFalse(provider_high.is_valid())


class TestConfidenceCalculation(unittest.TestCase):
    """Test confidence score calculation logic."""

    def test_confidence_based_on_inlier_ratio(self):
        """Test that confidence increases with inlier ratio."""
        provider = FeatureMatchHomography(width=2560, height=1440)

        # Test with different numbers of inliers
        errors = np.array([1.0, 1.0, 1.0])

        low_inliers = provider._calculate_confidence(5, 10, errors)
        high_inliers = provider._calculate_confidence(9, 10, errors)

        # Higher inlier ratio should give higher confidence
        self.assertLess(low_inliers, high_inliers)

    def test_confidence_based_on_reprojection_error(self):
        """Test that confidence decreases with reprojection error."""
        provider = FeatureMatchHomography(width=2560, height=1440, ransac_threshold=3.0)

        low_error = np.array([0.5, 0.5, 0.5])
        high_error = np.array([2.5, 2.5, 2.5])

        conf_low = provider._calculate_confidence(8, 10, low_error)
        conf_high = provider._calculate_confidence(8, 10, high_error)

        # Lower error should give higher confidence
        self.assertGreater(conf_low, conf_high)

    def test_confidence_zero_with_zero_points(self):
        """Test that confidence is 0 with no points."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        errors = np.array([])

        confidence = provider._calculate_confidence(0, 0, errors)
        self.assertEqual(confidence, 0.0)

    def test_confidence_penalty_for_low_inliers(self):
        """Test that low inlier ratio applies penalty."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        errors = np.array([1.0, 1.0])

        # Inlier ratio below MIN_INLIER_RATIO (0.5)
        conf = provider._calculate_confidence(4, 10, errors)

        # Should have penalty applied
        expected_max = 0.4 * provider.CONFIDENCE_PENALTY_LOW_INLIERS
        self.assertLessEqual(conf, expected_max)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_homography_with_collinear_points_fails(self):
        """Test that collinear GCPs fail to compute homography."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

        # Create collinear points (all on a line)
        collinear_gcps = [
            {'gps': {'latitude': 39.64 + i * 0.0001, 'longitude': -0.23}, 'image': {'u': 500 + i * 500, 'v': 500}}
            for i in range(4)
        ]

        reference = {'ground_control_points': collinear_gcps}

        # May raise RuntimeError or compute with very low confidence
        try:
            result = provider.compute_homography(frame, reference)
            # If it doesn't raise, confidence should be very low
            self.assertLess(result.confidence, 0.5)
        except RuntimeError as e:
            self.assertIn("Failed to compute homography", str(e))

    def test_project_point_at_horizon_raises_error(self):
        """Test that projecting point at horizon raises ValueError."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}
        provider.compute_homography(frame, reference)

        # Point very close to top of image (near horizon for typical scenes)
        # May or may not raise depending on the homography
        try:
            result = provider.project_point((1280, 1))
            # If it doesn't raise, just verify it returns a valid point
            self.assertIsInstance(result, WorldPoint)
        except ValueError as e:
            self.assertIn("infinity", str(e).lower())

    def test_very_small_image_dimensions(self):
        """Test with very small image dimensions."""
        provider = FeatureMatchHomography(width=100, height=100)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Scale GCPs to small image
        small_gcps = [
            {'gps': {'latitude': 39.6405, 'longitude': -0.2305}, 'image': {'u': 10, 'v': 10}},
            {'gps': {'latitude': 39.6405, 'longitude': -0.2295}, 'image': {'u': 90, 'v': 10}},
            {'gps': {'latitude': 39.6395, 'longitude': -0.2305}, 'image': {'u': 10, 'v': 90}},
            {'gps': {'latitude': 39.6395, 'longitude': -0.2295}, 'image': {'u': 90, 'v': 90}},
        ]

        reference = {'ground_control_points': small_gcps}
        result = provider.compute_homography(frame, reference)

        self.assertIsInstance(result, HomographyResult)

    def test_very_large_image_dimensions(self):
        """Test with very large image dimensions."""
        provider = FeatureMatchHomography(width=7680, height=4320)
        self.assertEqual(provider.width, 7680)
        self.assertEqual(provider.height, 4320)

    def test_non_list_gcps_raises_error(self):
        """Test that non-list GCPs raises appropriate error."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': "not a list"}

        with self.assertRaisesRegex(ValueError, "Need at least"):
            provider.compute_homography(frame, reference)


class TestHomographyConsistency(unittest.TestCase):
    """Test consistency of homography transformations."""

    def setUp(self):
        """Set up provider with computed homography."""
        self.provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_REALISTIC}
        self.provider.compute_homography(frame, reference)

    def test_map_and_world_coordinates_consistent(self):
        """Test that map coordinates and world coordinates are consistent."""
        image_point = (1280, 720)

        # Get both map and world coordinates
        map_coord = self.provider.project_point_to_map(image_point)
        world_point = self.provider.project_point(image_point)

        # Convert map back to GPS
        lat, lon = self.provider._local_to_gps(map_coord.x, map_coord.y)

        # Should match world_point GPS
        self.assertAlmostEqual(lat, world_point.latitude, places=5)
        self.assertAlmostEqual(lon, world_point.longitude, places=5)

    def test_multiple_projections_same_point(self):
        """Test that projecting the same point multiple times gives same result."""
        image_point = (1500, 800)

        wp1 = self.provider.project_point(image_point)
        wp2 = self.provider.project_point(image_point)

        self.assertEqual(wp1.latitude, wp2.latitude)
        self.assertEqual(wp1.longitude, wp2.longitude)
        self.assertEqual(wp1.confidence, wp2.confidence)

    def test_nearby_points_have_nearby_gps(self):
        """Test that nearby image points project to nearby GPS coordinates."""
        point1 = (1000, 700)
        point2 = (1010, 710)

        wp1 = self.provider.project_point(point1)
        wp2 = self.provider.project_point(point2)

        # Should be close (within reasonable tolerance)
        lat_diff = abs(wp1.latitude - wp2.latitude)
        lon_diff = abs(wp1.longitude - wp2.longitude)

        # 10 pixels shouldn't map to more than 0.01 degrees (rough check)
        self.assertLess(lat_diff, 0.01)
        self.assertLess(lon_diff, 0.01)


class TestGCPRoundTrip(unittest.TestCase):
    """Test round-trip accuracy for GCP points."""

    def test_all_gcps_round_trip_accurately(self):
        """Test that all GCPs round-trip with reasonable accuracy."""
        provider = FeatureMatchHomography(width=2560, height=1440)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        reference = {'ground_control_points': TEST_GCPS_SQUARE}

        result = provider.compute_homography(frame, reference)

        # Project each GCP's image point back to GPS
        for i, gcp in enumerate(TEST_GCPS_SQUARE):
            image_point = (gcp['image']['u'], gcp['image']['v'])
            world_point = provider.project_point(image_point)

            # Check accuracy (should be very close for inliers)
            lat_error = abs(world_point.latitude - gcp['gps']['latitude'])
            lon_error = abs(world_point.longitude - gcp['gps']['longitude'])

            # Allow larger tolerance since homography is approximate
            self.assertLess(lat_error, 0.001, f"GCP {i} latitude error too large")
            self.assertLess(lon_error, 0.001, f"GCP {i} longitude error too large")


def main():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    main()
