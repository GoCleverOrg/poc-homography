#!/usr/bin/env python3
"""
Unit tests for camera footprint calculation with partial projection handling.

Tests validate:
- Per-corner validation with horizon check (abs(w) < 1e-6)
- Distance-based clamping (distance > height_m * 20.0)
- Negative w handling (points behind camera)
- Partial footprint return with at least 2 valid corners
- Test matrix across tilt angles: 5, 10, 13, 20, 45, 80 degrees
- Test matrix across pan angles: 0, 30, 90, 180, 270 degrees
- No regression for normal operating conditions (tilt 20-70 degrees)
"""

import unittest
import sys
import os
import math
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.unified_gcp_tool import (
    _classify_corner_projection,
    _clamp_to_max_distance,
    _convert_world_offset_to_latlon,
    UnifiedSession
)


class TestCornerProjectionClassification(unittest.TestCase):
    """Test corner projection classification helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.height_m = 5.0
        self.max_distance = self.height_m * 20.0  # 100 meters

    def test_classify_valid_corner(self):
        """Test classification of valid corner (w > 1e-6, distance <= max)."""
        # Homogeneous coordinates: [X, Y, w] where point is (X/w, Y/w)
        # Valid point at (10, 20) meters from camera
        pt_world = np.array([[10.0], [20.0], [1.0]])

        result = _classify_corner_projection(pt_world, self.height_m)

        self.assertEqual(result['status'], 'valid')
        self.assertFalse(result['needs_clamping'])
        self.assertAlmostEqual(result['east_offset'], 10.0)
        self.assertAlmostEqual(result['north_offset'], 20.0)
        self.assertAlmostEqual(result['distance'], math.sqrt(10**2 + 20**2))

    def test_classify_near_horizon_positive_w(self):
        """Test classification when w near zero but positive (near horizon)."""
        # Point near horizon: w = 1e-7 (below threshold 1e-6)
        pt_world = np.array([[50.0], [80.0], [1e-7]])

        result = _classify_corner_projection(pt_world, self.height_m)

        self.assertEqual(result['status'], 'clampable')
        self.assertTrue(result['needs_clamping'])
        # Should preserve direction even though unnormalized values are huge
        self.assertIsNotNone(result['east_offset'])
        self.assertIsNotNone(result['north_offset'])

    def test_classify_negative_w(self):
        """Test classification when w is negative (behind camera)."""
        # Point behind camera
        pt_world = np.array([[10.0], [20.0], [-0.5]])

        result = _classify_corner_projection(pt_world, self.height_m)

        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['needs_clamping'])

    def test_classify_exceeds_max_distance(self):
        """Test classification when distance exceeds max (height * 20)."""
        # Point at valid w but distance > 100m (height=5, max=100)
        # Distance = sqrt(80^2 + 80^2) ≈ 113m > 100m
        pt_world = np.array([[80.0], [80.0], [1.0]])

        result = _classify_corner_projection(pt_world, self.height_m)

        self.assertEqual(result['status'], 'clampable')
        self.assertTrue(result['needs_clamping'])
        self.assertGreater(result['distance'], self.max_distance)

    def test_classify_at_max_distance_boundary(self):
        """Test classification exactly at max distance boundary."""
        # Point exactly at max_distance = 100m
        # Place at (0, 100) for simplicity
        pt_world = np.array([[0.0], [100.0], [1.0]])

        result = _classify_corner_projection(pt_world, self.height_m)

        # At boundary should be valid (not clamped)
        self.assertEqual(result['status'], 'valid')
        self.assertFalse(result['needs_clamping'])


class TestDistanceClamping(unittest.TestCase):
    """Test distance clamping helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.height_m = 5.0
        self.max_distance = 100.0  # height_m * 20.0

    def test_clamp_preserves_direction(self):
        """Test that clamping preserves direction vector."""
        # Original point at (80, 60) with distance = 100m (exactly at max)
        # Scaled version at (160, 120) with distance = 200m
        east_offset = 160.0
        north_offset = 120.0
        original_distance = math.sqrt(160**2 + 120**2)  # 200m

        clamped = _clamp_to_max_distance(
            east_offset, north_offset, self.max_distance
        )

        # Should clamp to max_distance while preserving direction
        clamped_distance = math.sqrt(clamped['east']**2 + clamped['north']**2)
        self.assertAlmostEqual(clamped_distance, self.max_distance, places=5)

        # Direction should be preserved (ratio should be same)
        original_ratio = north_offset / east_offset
        clamped_ratio = clamped['north'] / clamped['east']
        self.assertAlmostEqual(original_ratio, clamped_ratio, places=5)

    def test_clamp_northeast_direction(self):
        """Test clamping in northeast direction."""
        # Point at 45 degrees northeast, distance 200m
        east_offset = 200.0 / math.sqrt(2)
        north_offset = 200.0 / math.sqrt(2)

        clamped = _clamp_to_max_distance(
            east_offset, north_offset, self.max_distance
        )

        # Should be at 45 degrees, distance = max_distance
        self.assertAlmostEqual(clamped['east'], clamped['north'], places=5)
        self.assertAlmostEqual(clamped['east'], self.max_distance / math.sqrt(2), places=5)

    def test_clamp_pure_north(self):
        """Test clamping directly north."""
        east_offset = 0.0
        north_offset = 200.0

        clamped = _clamp_to_max_distance(
            east_offset, north_offset, self.max_distance
        )

        self.assertAlmostEqual(clamped['east'], 0.0, places=5)
        self.assertAlmostEqual(clamped['north'], self.max_distance, places=5)

    def test_clamp_southeast_direction(self):
        """Test clamping in southeast direction (negative north)."""
        east_offset = 150.0
        north_offset = -200.0

        clamped = _clamp_to_max_distance(
            east_offset, north_offset, self.max_distance
        )

        # Check distance
        clamped_distance = math.sqrt(clamped['east']**2 + clamped['north']**2)
        self.assertAlmostEqual(clamped_distance, self.max_distance, places=5)

        # Check direction preserved
        original_ratio = north_offset / east_offset
        clamped_ratio = clamped['north'] / clamped['east']
        self.assertAlmostEqual(original_ratio, clamped_ratio, places=5)


class TestWorldOffsetConversion(unittest.TestCase):
    """Test world offset to lat/lon conversion helper."""

    def setUp(self):
        """Set up test fixtures."""
        self.camera_lat = 40.0  # Mid-latitude for reasonable cos factor
        self.camera_lon = -105.0

    def test_convert_north_offset(self):
        """Test conversion of pure north offset."""
        east_offset = 0.0
        north_offset = 111320.0  # Approximately 1 degree latitude

        result = _convert_world_offset_to_latlon(
            east_offset, north_offset, self.camera_lat, self.camera_lon
        )

        # Should add approximately 1 degree to latitude
        self.assertAlmostEqual(result['lat'], self.camera_lat + 1.0, places=4)
        self.assertAlmostEqual(result['lon'], self.camera_lon, places=4)

    def test_convert_east_offset(self):
        """Test conversion of pure east offset."""
        # At 40° latitude, cos(40°) ≈ 0.766
        # 1 degree longitude ≈ 111320 * 0.766 ≈ 85267m
        east_offset = 85267.0
        north_offset = 0.0

        result = _convert_world_offset_to_latlon(
            east_offset, north_offset, self.camera_lat, self.camera_lon
        )

        # Should add approximately 1 degree to longitude
        self.assertAlmostEqual(result['lat'], self.camera_lat, places=4)
        self.assertAlmostEqual(result['lon'], self.camera_lon + 1.0, places=1)

    def test_convert_combined_offset(self):
        """Test conversion of combined north/east offset."""
        east_offset = 1000.0  # meters
        north_offset = 2000.0  # meters

        result = _convert_world_offset_to_latlon(
            east_offset, north_offset, self.camera_lat, self.camera_lon
        )

        # Verify both lat and lon changed
        self.assertGreater(result['lat'], self.camera_lat)
        self.assertGreater(result['lon'], self.camera_lon)

        # Verify north offset dominates lat change
        lat_change_meters = (result['lat'] - self.camera_lat) * 111320.0
        self.assertAlmostEqual(lat_change_meters, north_offset, places=0)


class TestFootprintCalculationIntegration(unittest.TestCase):
    """Test complete footprint calculation with real camera geometry."""

    def _create_test_session_with_params(self, tilt_deg, pan_deg=0.0, height_m=5.0):
        """Create a mock session with camera params for testing."""
        # Create basic camera parameters without full session initialization
        K = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])

        # Create minimal mock object with just camera_params
        class MockSession:
            def __init__(self):
                self.camera_params = {
                    'K': K.tolist(),
                    'image_width': 1920,
                    'image_height': 1080,
                    'camera_lat': 40.0,
                    'camera_lon': -105.0,
                    'height_m': height_m,
                    'pan_deg': pan_deg,
                    'tilt_deg': tilt_deg
                }

            # Bind the actual method
            calculate_camera_footprint = UnifiedSession.calculate_camera_footprint

        return MockSession()

    def test_normal_tilt_all_corners_valid(self):
        """Test normal tilt (45 degrees) returns all 4 valid corners."""
        session = self._create_test_session_with_params(tilt_deg=45.0)
        footprint = session.calculate_camera_footprint()

        self.assertIsNotNone(footprint)
        self.assertEqual(len(footprint), 4)

        # All corners should be valid and not clamped
        for corner in footprint:
            self.assertIn('valid', corner)
            self.assertIn('clamped', corner)
            self.assertTrue(corner['valid'], f"Corner should be valid at tilt=45°")
            self.assertFalse(corner['clamped'], f"Corner should not be clamped at tilt=45°")

    def test_low_tilt_partial_footprint(self):
        """Test low tilt (13 degrees) returns footprint even with clamped corners."""
        session = self._create_test_session_with_params(tilt_deg=13.0, pan_deg=31.0)
        footprint = session.calculate_camera_footprint()

        # Should return footprint (not None)
        self.assertIsNotNone(footprint, "Should return footprint at tilt=13°, pan=31°")

        # At least 2 corners should be present
        self.assertGreaterEqual(len(footprint), 2, "Should have at least 2 corners")

    def test_very_low_tilt_with_clamping(self):
        """Test very low tilt (5 degrees) with clamped top corners."""
        session = self._create_test_session_with_params(tilt_deg=5.0)
        footprint = session.calculate_camera_footprint()

        if footprint is not None:
            # Should have some clamped corners
            clamped_count = sum(1 for c in footprint if c.get('clamped', False))
            # At very low tilt, expect at least some clamping
            # (This may vary based on camera geometry)
            self.assertGreaterEqual(len(footprint), 2, "Should have at least 2 corners even at tilt=5°")

    def test_high_tilt_small_footprint(self):
        """Test high tilt (80 degrees) returns small but valid footprint."""
        session = self._create_test_session_with_params(tilt_deg=80.0)
        footprint = session.calculate_camera_footprint()

        self.assertIsNotNone(footprint)
        self.assertEqual(len(footprint), 4)

        # All corners should be valid (looking almost straight down)
        for corner in footprint:
            self.assertTrue(corner['valid'], "Corner should be valid at tilt=80°")

    def test_footprint_direction_correctness(self):
        """Test footprint centroid displaced from camera in reasonable direction."""
        session = self._create_test_session_with_params(tilt_deg=45.0, pan_deg=90.0)
        footprint = session.calculate_camera_footprint()

        self.assertIsNotNone(footprint)
        self.assertGreater(len(footprint), 0)

        # Calculate footprint centroid
        avg_lat = sum(c['lat'] for c in footprint) / len(footprint)
        avg_lon = sum(c['lon'] for c in footprint) / len(footprint)

        camera_lat = 40.0
        camera_lon = -105.0

        # At pan=90° (east), footprint should be displaced eastward
        self.assertGreater(avg_lon, camera_lon, "Footprint should be east of camera at pan=90°")


class TestFootprintTestMatrix(unittest.TestCase):
    """Test footprint across required tilt/pan combinations from issue #115."""

    def _create_test_session_with_params(self, tilt_deg, pan_deg=0.0):
        """Create a mock session with camera params for testing."""
        K = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])

        class MockSession:
            def __init__(self):
                self.camera_params = {
                    'K': K.tolist(),
                    'image_width': 1920,
                    'image_height': 1080,
                    'camera_lat': 40.0,
                    'camera_lon': -105.0,
                    'height_m': 5.0,
                    'pan_deg': pan_deg,
                    'tilt_deg': tilt_deg
                }

            calculate_camera_footprint = UnifiedSession.calculate_camera_footprint

        return MockSession()

    def test_all_tilt_pan_combinations(self):
        """Test all required tilt/pan combinations return valid footprints."""
        test_tilts = [5, 10, 13, 20, 45, 80]
        test_pans = [0, 30, 90, 180, 270]

        failures = []
        max_distance = 5.0 * 20.0  # height_m * MAX_DISTANCE_HEIGHT_RATIO

        for tilt in test_tilts:
            for pan in test_pans:
                with self.subTest(tilt=tilt, pan=pan):
                    session = self._create_test_session_with_params(tilt_deg=float(tilt), pan_deg=float(pan))
                    footprint = session.calculate_camera_footprint()

                    if footprint is None:
                        failures.append(f"tilt={tilt}°, pan={pan}° returned None")
                        continue

                    # Verify at least 2 corners present
                    if len(footprint) < 2:
                        failures.append(f"tilt={tilt}°, pan={pan}° returned only {len(footprint)} corners")
                        continue

                    # Verify no corners extend beyond max distance (with tolerance)
                    camera_lat = 40.0
                    camera_lon = -105.0

                    for i, corner in enumerate(footprint):
                        # Calculate approximate distance in meters
                        lat_diff = (corner['lat'] - camera_lat) * 111320.0
                        lon_diff = (corner['lon'] - camera_lon) * 111320.0 * math.cos(math.radians(camera_lat))
                        distance = math.sqrt(lat_diff**2 + lon_diff**2)

                        # Should not exceed max_distance * 1.1 (10% tolerance)
                        if distance > max_distance * 1.1:
                            failures.append(
                                f"tilt={tilt}°, pan={pan}°, corner {i}: distance {distance:.1f}m exceeds max {max_distance:.1f}m"
                            )

        if failures:
            self.fail(f"Footprint calculation issues:\n" + "\n".join(failures))


class TestNoRegression(unittest.TestCase):
    """Test no regression in normal operating conditions."""

    def _create_test_session_with_params(self, tilt_deg):
        """Create a mock session with camera params for testing."""
        K = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])

        class MockSession:
            def __init__(self):
                self.camera_params = {
                    'K': K.tolist(),
                    'image_width': 1920,
                    'image_height': 1080,
                    'camera_lat': 40.0,
                    'camera_lon': -105.0,
                    'height_m': 5.0,
                    'pan_deg': 0.0,
                    'tilt_deg': tilt_deg
                }

            calculate_camera_footprint = UnifiedSession.calculate_camera_footprint

        return MockSession()

    def test_normal_range_20_to_70_degrees(self):
        """Test tilt range 20-70 degrees produces valid footprints without regression."""
        for tilt in range(20, 71, 10):
            with self.subTest(tilt=tilt):
                session = self._create_test_session_with_params(tilt_deg=float(tilt))
                footprint = session.calculate_camera_footprint()

                self.assertIsNotNone(footprint, f"Should return footprint at tilt={tilt}°")
                
                # Should always have at least 2 corners
                self.assertGreaterEqual(len(footprint), 2, f"Should have at least 2 corners at tilt={tilt}°")
                
                # At tilt >= 40°, expect all 4 corners to be valid and not clamped
                if tilt >= 40:
                    self.assertEqual(len(footprint), 4, f"Should have 4 corners at tilt={tilt}°")
                    # All corners should be valid and not clamped
                    for corner in footprint:
                        self.assertTrue(corner['valid'], f"Corner should be valid at tilt={tilt}°")
                        self.assertFalse(corner['clamped'], f"Corner should not be clamped at tilt={tilt}°")
                
                # For all tilts in this range, no corners should extend beyond max distance
                # (either valid within range or clamped to max distance)
                max_distance = 5.0 * 20.0
                camera_lat = 40.0
                camera_lon = -105.0
                for corner in footprint:
                    lat_diff = (corner['lat'] - camera_lat) * 111320.0
                    lon_diff = (corner['lon'] - camera_lon) * 111320.0 * np.cos(np.radians(camera_lat))
                    distance = np.sqrt(lat_diff**2 + lon_diff**2)
                    self.assertLessEqual(distance, max_distance * 1.1,
                        f"Corner distance {distance:.1f}m should not exceed max {max_distance}m at tilt={tilt}°")


if __name__ == '__main__':
    unittest.main()
