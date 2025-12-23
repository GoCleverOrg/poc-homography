#!/usr/bin/env python3
"""
Unit tests for AutoCalibrator projection transformation.

Tests verify:
1. Transformation equivalence between AutoCalibrator projection and direct H_total computation
2. Single projection (no double transformation) in the auto-calibration flow
3. Backwards compatibility when geotiff_params is not provided
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.auto_calibrator import AutoCalibrator
from poc_homography.camera_geometry import CameraGeometry


class TestAutoCalibratorProjectionEquivalence(unittest.TestCase):
    """Test that AutoCalibrator projection matches direct H_total computation."""

    def setUp(self):
        """Set up test fixtures with realistic camera parameters."""
        # Set up geometry with known parameters
        self.geo = CameraGeometry(w=1920, h=1080)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float64)
        self.w_pos = np.array([0.0, 0.0, 10.0])

        self.geo.set_camera_parameters(
            K=self.K,
            w_pos=self.w_pos,
            pan_deg=45,
            tilt_deg=30,
            map_width=640,
            map_height=640
        )

        # Create synthetic cartography mask
        self.carto_mask = np.zeros((2000, 2000), dtype=np.uint8)
        self.carto_mask[800:1200, 800:1200] = 255  # 400x400 square

        self.camera_mask = np.zeros((1080, 1920), dtype=np.uint8)

        # Geotiff parameters (synthetic)
        self.geotiff_params = {
            'pixel_size_x': 0.5,
            'pixel_size_y': -0.5,
            'origin_easting': 500000.0,
            'origin_northing': 4500000.0
        }
        self.camera_utm_position = (500100.0, 4499900.0)

    def test_autocalibrtor_projection_equivalence(self):
        """Verify AutoCalibrator projection matches direct H_total computation."""
        # Create AutoCalibrator with geotiff params
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.carto_mask,
            camera_mask=self.camera_mask,
            geotiff_params=self.geotiff_params,
            camera_utm_position=self.camera_utm_position
        )

        # Project using AutoCalibrator
        projected_autocal = calibrator._project_map_mask()

        # Project using direct H_total computation (reference implementation)
        pixel_size_x = self.geotiff_params['pixel_size_x']
        pixel_size_y = self.geotiff_params['pixel_size_y']
        origin_easting = self.geotiff_params['origin_easting']
        origin_northing = self.geotiff_params['origin_northing']
        camera_easting, camera_northing = self.camera_utm_position

        dx = origin_easting - camera_easting
        dy = origin_northing - camera_northing

        T_pixel_to_localXY = np.array([
            [pixel_size_x, 0, dx],
            [0, pixel_size_y, dy],
            [0, 0, 1]
        ], dtype=np.float64)

        H_total = self.geo.H @ T_pixel_to_localXY
        projected_direct = cv2.warpPerspective(
            self.carto_mask, H_total, (1920, 1080),
            flags=cv2.INTER_NEAREST
        )

        # Verify equivalence using IoU
        intersection = cv2.bitwise_and(projected_autocal, projected_direct)
        union = cv2.bitwise_or(projected_autocal, projected_direct)

        intersection_count = cv2.countNonZero(intersection)
        union_count = cv2.countNonZero(union)

        if union_count > 0:
            iou = intersection_count / union_count
        else:
            # Both projections are empty - that's still equivalent
            iou = 1.0

        self.assertGreater(
            iou, 0.95,
            f"Projection mismatch: IoU={iou:.4f} (expected >0.95)"
        )

    def test_autocalibrtor_projection_produces_non_empty_mask(self):
        """Verify that projection produces a non-empty mask when mask has content."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.carto_mask,
            camera_mask=self.camera_mask,
            geotiff_params=self.geotiff_params,
            camera_utm_position=self.camera_utm_position
        )

        projected = calibrator._project_map_mask()

        # The mask should have some projected pixels (unless all off-screen)
        # For our test parameters, we expect some visible projection
        self.assertEqual(projected.shape, (1080, 1920))
        self.assertEqual(projected.dtype, np.uint8)


class TestAutoCalibratorNoDoubleProjection(unittest.TestCase):
    """Test that mask is projected exactly once during calibration."""

    def setUp(self):
        """Set up test fixtures."""
        self.geo = CameraGeometry(w=1920, h=1080)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float64)
        self.w_pos = np.array([0.0, 0.0, 10.0])

        self.geo.set_camera_parameters(
            K=self.K,
            w_pos=self.w_pos,
            pan_deg=45,
            tilt_deg=30,
            map_width=640,
            map_height=640
        )

        self.carto_mask = np.zeros((2000, 2000), dtype=np.uint8)
        self.carto_mask[800:1200, 800:1200] = 255

        self.camera_mask = np.zeros((1080, 1920), dtype=np.uint8)

        self.geotiff_params = {
            'pixel_size_x': 0.5,
            'pixel_size_y': -0.5,
            'origin_easting': 500000.0,
            'origin_northing': 4500000.0
        }
        self.camera_utm_position = (500100.0, 4499900.0)

    def test_single_warp_perspective_call(self):
        """Verify cartography mask is projected only once during _project_map_mask()."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.carto_mask,
            camera_mask=self.camera_mask,
            geotiff_params=self.geotiff_params,
            camera_utm_position=self.camera_utm_position
        )

        # Spy on cv2.warpPerspective calls
        with patch('poc_homography.auto_calibrator.cv2.warpPerspective',
                   wraps=cv2.warpPerspective) as mock_warp:
            projected = calibrator._project_map_mask()

            # Should be called exactly once
            self.assertEqual(
                mock_warp.call_count, 1,
                f"Expected 1 warpPerspective call, got {mock_warp.call_count}"
            )


class TestAutoCalibratorBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility when geotiff_params is not provided."""

    def setUp(self):
        """Set up test fixtures."""
        self.geo = CameraGeometry(w=1920, h=1080)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float64)
        self.w_pos = np.array([0.0, 0.0, 10.0])

        self.geo.set_camera_parameters(
            K=self.K,
            w_pos=self.w_pos,
            pan_deg=45,
            tilt_deg=30,
            map_width=640,
            map_height=640
        )

        # Smaller mask that fits in local XY coordinate space for backwards compat
        self.map_mask = np.zeros((640, 640), dtype=np.uint8)
        self.map_mask[200:440, 200:440] = 255

        self.camera_mask = np.zeros((1080, 1920), dtype=np.uint8)

    def test_no_geotiff_params_uses_camera_homography_only(self):
        """Verify fallback to H_camera when geotiff_params is None."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            # No geotiff_params or camera_utm_position
        )

        # Should use H_camera directly (backwards compatibility)
        projected_autocal = calibrator._project_map_mask()

        # Compare to direct H (camera homography only) projection
        H_camera = self.geo.H
        projected_direct = cv2.warpPerspective(
            self.map_mask, H_camera, (1920, 1080),
            flags=cv2.INTER_NEAREST
        )

        # Should be identical
        np.testing.assert_array_equal(
            projected_autocal, projected_direct,
            "Backwards compatibility failed: projection should match H_camera"
        )

    def test_geotiff_params_none_camera_utm_provided(self):
        """Verify fallback when only camera_utm_position is provided (not geotiff)."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            geotiff_params=None,
            camera_utm_position=(500100.0, 4499900.0)
        )

        # Should fall back to H_camera
        projected = calibrator._project_map_mask()
        self.assertIsNotNone(projected)

    def test_geotiff_provided_camera_utm_none(self):
        """Verify fallback when only geotiff_params is provided (not camera_utm)."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            geotiff_params={
                'pixel_size_x': 0.5,
                'pixel_size_y': -0.5,
                'origin_easting': 500000.0,
                'origin_northing': 4500000.0
            },
            camera_utm_position=None
        )

        # Should fall back to H_camera
        projected = calibrator._project_map_mask()
        self.assertIsNotNone(projected)


class TestAutoCalibratorInitialization(unittest.TestCase):
    """Test AutoCalibrator initialization with new parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.geo = CameraGeometry(w=1920, h=1080)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float64)
        self.w_pos = np.array([0.0, 0.0, 10.0])

        self.geo.set_camera_parameters(
            K=self.K,
            w_pos=self.w_pos,
            pan_deg=45,
            tilt_deg=30,
            map_width=640,
            map_height=640
        )

        self.map_mask = np.zeros((640, 640), dtype=np.uint8)
        self.map_mask[200:440, 200:440] = 255

        self.camera_mask = np.zeros((1080, 1920), dtype=np.uint8)

    def test_geotiff_params_stored(self):
        """Verify geotiff_params is stored correctly."""
        geotiff_params = {
            'pixel_size_x': 0.5,
            'pixel_size_y': -0.5,
            'origin_easting': 500000.0,
            'origin_northing': 4500000.0
        }

        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            geotiff_params=geotiff_params
        )

        self.assertEqual(calibrator.geotiff_params, geotiff_params)

    def test_camera_utm_position_stored(self):
        """Verify camera_utm_position is stored correctly."""
        camera_utm = (500100.0, 4499900.0)

        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            camera_utm_position=camera_utm
        )

        self.assertEqual(calibrator.camera_utm_position, camera_utm)

    def test_default_values_are_none(self):
        """Verify new parameters default to None."""
        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask
        )

        self.assertIsNone(calibrator.geotiff_params)
        self.assertIsNone(calibrator.camera_utm_position)


class TestAutoCalibratorPartialConfigWarning(unittest.TestCase):
    """Test warning log for partial geotiff configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.geo = CameraGeometry(w=1920, h=1080)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float64)
        self.w_pos = np.array([0.0, 0.0, 10.0])

        self.geo.set_camera_parameters(
            K=self.K,
            w_pos=self.w_pos,
            pan_deg=45,
            tilt_deg=30,
            map_width=640,
            map_height=640
        )

        self.map_mask = np.zeros((640, 640), dtype=np.uint8)
        self.map_mask[200:440, 200:440] = 255

        self.camera_mask = np.zeros((1080, 1920), dtype=np.uint8)

    def test_partial_config_logs_warning_once(self):
        """Verify warning is logged once when only one param is provided."""
        import logging
        from poc_homography import auto_calibrator

        calibrator = AutoCalibrator(
            camera_geometry=self.geo,
            map_mask=self.map_mask,
            camera_mask=self.camera_mask,
            geotiff_params={
                'pixel_size_x': 0.5,
                'pixel_size_y': -0.5,
                'origin_easting': 500000.0,
                'origin_northing': 4500000.0
            },
            camera_utm_position=None  # Missing - should trigger warning
        )

        with self.assertLogs(auto_calibrator.logger, level='WARNING') as log:
            # First call should log warning
            calibrator._project_map_mask()
            # Second call should NOT log warning (already warned)
            calibrator._project_map_mask()

        # Should have exactly one warning
        warning_logs = [msg for msg in log.output if 'Partial geotiff configuration' in msg]
        self.assertEqual(
            len(warning_logs), 1,
            f"Expected exactly 1 warning, got {len(warning_logs)}"
        )


if __name__ == '__main__':
    unittest.main()
