#!/usr/bin/env python3
"""
Unit tests for calibration table functionality in IntrinsicExtrinsicHomography.

Tests zoom-dependent intrinsic parameter interpolation and distortion coefficient
retrieval from calibration tables, with fallback to linear approximation.

Run with: python -m pytest tests/test_calibration_table.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


class TestCalibrationTableInterpolation:
    """Tests for calibration table interpolation logic."""

    def test_exact_zoom_match_returns_calibrated_values(self):
        """When zoom exactly matches a calibrated value, return that entry without interpolation."""
        calibration_table = {
            1.0: {
                "fx": 1825.3,
                "fy": 1823.1,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.341,
                "k2": 0.788,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 9120.5,
                "fy": 9115.2,
                "cx": 1282.1,
                "cy": 721.3,
                "k1": -0.298,
                "k2": 0.654,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Exact match at zoom=1.0
        K = homography.get_intrinsics(zoom_factor=1.0)
        assert K[0, 0] == pytest.approx(1825.3, abs=1e-6)  # fx
        assert K[1, 1] == pytest.approx(1823.1, abs=1e-6)  # fy
        assert K[0, 2] == pytest.approx(1280.0, abs=1e-6)  # cx
        assert K[1, 2] == pytest.approx(720.0, abs=1e-6)  # cy

        # Exact match at zoom=5.0
        K = homography.get_intrinsics(zoom_factor=5.0)
        assert K[0, 0] == pytest.approx(9120.5, abs=1e-6)  # fx
        assert K[1, 1] == pytest.approx(9115.2, abs=1e-6)  # fy
        assert K[0, 2] == pytest.approx(1282.1, abs=1e-6)  # cx
        assert K[1, 2] == pytest.approx(721.3, abs=1e-6)  # cy

    def test_interpolation_between_zoom_levels(self):
        """When zoom is between calibrated values, linearly interpolate fx, fy, cx, cy."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom at midpoint (3.0 is halfway between 1.0 and 5.0)
        K = homography.get_intrinsics(zoom_factor=3.0)

        # Expected: linear interpolation
        # fx: 1000 + (5000 - 1000) * (3.0 - 1.0) / (5.0 - 1.0) = 1000 + 4000 * 0.5 = 3000
        # fy: same as fx
        # cx: 1280 + (1290 - 1280) * 0.5 = 1285
        # cy: 720 + (730 - 720) * 0.5 = 725
        assert K[0, 0] == pytest.approx(3000.0, abs=1e-6)  # fx
        assert K[1, 1] == pytest.approx(3000.0, abs=1e-6)  # fy
        assert K[0, 2] == pytest.approx(1285.0, abs=1e-6)  # cx
        assert K[1, 2] == pytest.approx(725.0, abs=1e-6)  # cy

    def test_zoom_below_minimum_clamps_to_lowest(self):
        """When zoom is below minimum calibrated value, use lowest calibrated zoom (no extrapolation)."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom below minimum (0.5 < 1.0)
        K = homography.get_intrinsics(zoom_factor=0.5)

        # Should use zoom=1.0 values (no extrapolation)
        assert K[0, 0] == pytest.approx(1000.0, abs=1e-6)
        assert K[1, 1] == pytest.approx(1000.0, abs=1e-6)
        assert K[0, 2] == pytest.approx(1280.0, abs=1e-6)
        assert K[1, 2] == pytest.approx(720.0, abs=1e-6)

    def test_zoom_above_maximum_clamps_to_highest(self):
        """When zoom is above maximum calibrated value, use highest calibrated zoom (no extrapolation)."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom above maximum (10.0 > 5.0)
        K = homography.get_intrinsics(zoom_factor=10.0)

        # Should use zoom=5.0 values (no extrapolation)
        assert K[0, 0] == pytest.approx(5000.0, abs=1e-6)
        assert K[1, 1] == pytest.approx(5000.0, abs=1e-6)
        assert K[0, 2] == pytest.approx(1290.0, abs=1e-6)
        assert K[1, 2] == pytest.approx(730.0, abs=1e-6)

    def test_no_calibration_table_uses_linear_approximation(self):
        """When calibration_table is None, fall back to linear focal length approximation."""
        homography = IntrinsicExtrinsicHomography(
            width=2560,
            height=1440,
            sensor_width_mm=6.78,
            base_focal_length_mm=5.9,
            calibration_table=None,  # Explicit None
        )

        K = homography.get_intrinsics(zoom_factor=5.0)

        # Linear approximation: f_mm = 5.9 * 5.0 = 29.5mm
        # f_px = 29.5 * (2560 / 6.78) ≈ 11138.05
        expected_f_px = 5.9 * 5.0 * (2560 / 6.78)

        assert K[0, 0] == pytest.approx(expected_f_px, abs=1e-2)
        assert K[1, 1] == pytest.approx(expected_f_px, abs=1e-2)
        assert K[0, 2] == pytest.approx(1280.0, abs=1e-6)  # cx = width / 2
        assert K[1, 2] == pytest.approx(720.0, abs=1e-6)  # cy = height / 2

    def test_calibration_table_not_provided_defaults_to_none(self):
        """When calibration_table parameter is not provided, it defaults to None (linear approximation)."""
        # Don't pass calibration_table parameter at all
        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, sensor_width_mm=6.78, base_focal_length_mm=5.9
        )

        K = homography.get_intrinsics(zoom_factor=5.0)

        # Should use linear approximation
        expected_f_px = 5.9 * 5.0 * (2560 / 6.78)
        assert K[0, 0] == pytest.approx(expected_f_px, abs=1e-2)


class TestDistortionCoefficientRetrieval:
    """Tests for get_distortion_coefficients() method."""

    def test_exact_zoom_match_returns_distortion_coefficients(self):
        """When zoom exactly matches a calibrated value, return distortion coefficients."""
        calibration_table = {
            1.0: {
                "fx": 1825.3,
                "fy": 1823.1,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.341,
                "k2": 0.788,
                "p1": 0.001,
                "p2": 0.002,
                "k3": -0.5,
            },
            5.0: {
                "fx": 9120.5,
                "fy": 9115.2,
                "cx": 1282.1,
                "cy": 721.3,
                "k1": -0.298,
                "k2": 0.654,
                "p1": 0.003,
                "p2": 0.004,
                "k3": -0.3,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Exact match at zoom=1.0
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=1.0)
        expected = np.array([-0.341, 0.788, 0.001, 0.002, -0.5])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

        # Exact match at zoom=5.0
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=5.0)
        expected = np.array([-0.298, 0.654, 0.003, 0.004, -0.3])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_interpolation_of_distortion_coefficients(self):
        """When zoom is between calibrated values, linearly interpolate distortion coefficients."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.004,
                "p2": 0.002,
                "k3": -0.1,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom at midpoint (3.0)
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=3.0)

        # Expected: linear interpolation at 50% (t=0.5)
        # k1: -0.3 + (-0.2 - (-0.3)) * 0.5 = -0.3 + 0.1 * 0.5 = -0.25
        # k2: 0.7 + (0.6 - 0.7) * 0.5 = 0.7 - 0.05 = 0.65
        # p1: 0.0 + (0.004 - 0.0) * 0.5 = 0.002
        # p2: 0.0 + (0.002 - 0.0) * 0.5 = 0.001
        # k3: 0.0 + (-0.1 - 0.0) * 0.5 = -0.05
        expected = np.array([-0.25, 0.65, 0.002, 0.001, -0.05])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_distortion_coefficients_clamp_below_minimum_zoom(self):
        """When zoom is below minimum, use lowest calibrated zoom's distortion coefficients."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.004,
                "p2": 0.002,
                "k3": -0.1,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom below minimum
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=0.5)
        expected = np.array([-0.3, 0.7, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_distortion_coefficients_clamp_above_maximum_zoom(self):
        """When zoom is above maximum, use highest calibrated zoom's distortion coefficients."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.004,
                "p2": 0.002,
                "k3": -0.1,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Zoom above maximum
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=10.0)
        expected = np.array([-0.2, 0.6, 0.004, 0.002, -0.1])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_no_calibration_table_returns_none(self):
        """When calibration_table is None, return None (no distortion from table)."""
        homography = IntrinsicExtrinsicHomography(width=2560, height=1440, calibration_table=None)

        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=5.0)
        assert dist_coeffs is None


class TestMultipleZoomLevels:
    """Tests for calibration tables with more than 2 zoom levels."""

    def test_three_zoom_levels_interpolation(self):
        """Test interpolation with three calibrated zoom levels."""
        calibration_table = {
            1.0: {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1290.0,
                "cy": 730.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.004,
                "p2": 0.002,
                "k3": -0.1,
            },
            10.0: {
                "fx": 10000.0,
                "fy": 10000.0,
                "cx": 1300.0,
                "cy": 740.0,
                "k1": -0.1,
                "k2": 0.5,
                "p1": 0.008,
                "p2": 0.004,
                "k3": -0.2,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Interpolate between zoom=5.0 and zoom=10.0 at zoom=7.5 (midpoint)
        K = homography.get_intrinsics(zoom_factor=7.5)

        # Expected fx: 5000 + (10000 - 5000) * 0.5 = 7500
        # Expected cx: 1290 + (1300 - 1290) * 0.5 = 1295
        # Expected cy: 730 + (740 - 730) * 0.5 = 735
        assert K[0, 0] == pytest.approx(7500.0, abs=1e-6)
        assert K[0, 2] == pytest.approx(1295.0, abs=1e-6)
        assert K[1, 2] == pytest.approx(735.0, abs=1e-6)

        # Check distortion interpolation
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=7.5)
        # k1: -0.2 + (-0.1 - (-0.2)) * 0.5 = -0.15
        # k2: 0.6 + (0.5 - 0.6) * 0.5 = 0.55
        expected = np.array([-0.15, 0.55, 0.006, 0.003, -0.15])
        np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_six_zoom_levels_realistic_scenario(self):
        """Test realistic scenario with 6 calibrated zoom levels (1x, 5x, 10x, 15x, 20x, 25x)."""
        calibration_table = {
            1.0: {
                "fx": 1825.3,
                "fy": 1823.1,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.341,
                "k2": 0.788,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 9120.5,
                "fy": 9115.2,
                "cx": 1282.1,
                "cy": 721.3,
                "k1": -0.298,
                "k2": 0.654,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
            10.0: {
                "fx": 18240.8,
                "fy": 18230.3,
                "cx": 1284.2,
                "cy": 722.6,
                "k1": -0.255,
                "k2": 0.520,
                "p1": 0.002,
                "p2": 0.001,
                "k3": 0.0,
            },
            15.0: {
                "fx": 27361.1,
                "fy": 27345.4,
                "cx": 1286.3,
                "cy": 723.9,
                "k1": -0.212,
                "k2": 0.386,
                "p1": 0.003,
                "p2": 0.001,
                "k3": 0.0,
            },
            20.0: {
                "fx": 36481.4,
                "fy": 36460.5,
                "cx": 1288.4,
                "cy": 725.2,
                "k1": -0.169,
                "k2": 0.252,
                "p1": 0.004,
                "p2": 0.002,
                "k3": 0.0,
            },
            25.0: {
                "fx": 45601.7,
                "fy": 45575.6,
                "cx": 1290.5,
                "cy": 726.5,
                "k1": -0.126,
                "k2": 0.118,
                "p1": 0.005,
                "p2": 0.002,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Test interpolation at zoom=12.5 (between 10.0 and 15.0)
        K = homography.get_intrinsics(zoom_factor=12.5)

        # t = (12.5 - 10.0) / (15.0 - 10.0) = 0.5
        # Expected fx: 18240.8 + (27361.1 - 18240.8) * 0.5 = 22800.95
        expected_fx = 18240.8 + (27361.1 - 18240.8) * 0.5
        assert K[0, 0] == pytest.approx(expected_fx, abs=1e-2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_zoom_level_in_calibration_table(self):
        """Calibration table with single zoom level uses that value for all zooms."""
        calibration_table = {
            5.0: {
                "fx": 5000.0,
                "fy": 5000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.2,
                "k2": 0.6,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            }
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Any zoom should return the single calibrated value
        for zoom in [0.5, 1.0, 5.0, 10.0, 25.0]:
            K = homography.get_intrinsics(zoom_factor=zoom)
            assert K[0, 0] == pytest.approx(5000.0, abs=1e-6)
            assert K[1, 1] == pytest.approx(5000.0, abs=1e-6)

            dist_coeffs = homography.get_distortion_coefficients(zoom_factor=zoom)
            expected = np.array([-0.2, 0.6, 0.0, 0.0, 0.0])
            np.testing.assert_array_almost_equal(dist_coeffs, expected, decimal=6)

    def test_empty_calibration_table_falls_back_to_linear(self):
        """Empty calibration table {} falls back to linear approximation."""
        homography = IntrinsicExtrinsicHomography(
            width=2560,
            height=1440,
            sensor_width_mm=6.78,
            base_focal_length_mm=5.9,
            calibration_table={},
        )

        K = homography.get_intrinsics(zoom_factor=5.0)

        # Should use linear approximation
        expected_f_px = 5.9 * 5.0 * (2560 / 6.78)
        assert K[0, 0] == pytest.approx(expected_f_px, abs=1e-2)

        # No distortion coefficients available
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=5.0)
        assert dist_coeffs is None

    def test_calibration_table_with_non_numeric_zoom_keys_raises_error(self):
        """Calibration table with non-numeric zoom keys should raise TypeError."""
        calibration_table = {
            "wide": {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.3,
                "k2": 0.7,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            }
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Should raise TypeError when trying to use string as numeric key
        with pytest.raises((TypeError, ValueError)):
            homography.get_intrinsics(zoom_factor=5.0)


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_existing_code_without_calibration_table_still_works(self):
        """Existing code that doesn't use calibration_table continues to work."""
        # Old-style initialization (no calibration_table parameter)
        homography = IntrinsicExtrinsicHomography(
            width=1920, height=1080, sensor_width_mm=7.18, base_focal_length_mm=5.9
        )

        # Should work with linear approximation
        K = homography.get_intrinsics(zoom_factor=10.0)
        expected_f_px = 5.9 * 10.0 * (1920 / 7.18)
        assert K[0, 0] == pytest.approx(expected_f_px, abs=1e-2)

        # get_distortion_coefficients should return None when no calibration table
        dist_coeffs = homography.get_distortion_coefficients(zoom_factor=10.0)
        assert dist_coeffs is None

    def test_compute_homography_still_works_with_calibration_table(self):
        """compute_homography() works correctly when calibration_table is used."""
        calibration_table = {
            1.0: {
                "fx": 1825.3,
                "fy": 1823.1,
                "cx": 1280.0,
                "cy": 720.0,
                "k1": -0.341,
                "k2": 0.788,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
            5.0: {
                "fx": 9120.5,
                "fy": 9115.2,
                "cx": 1282.1,
                "cy": 721.3,
                "k1": -0.298,
                "k2": 0.654,
                "p1": 0.001,
                "p2": 0.0,
                "k3": 0.0,
            },
        }

        homography = IntrinsicExtrinsicHomography(
            width=2560, height=1440, calibration_table=calibration_table
        )

        # Get intrinsics using calibration table
        K = homography.get_intrinsics(zoom_factor=3.0)

        # Create dummy frame (not used for intrinsic/extrinsic approach)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "roll_deg": 0.0,
            "map_width": 640,
            "map_height": 640,
        }

        # Should compute homography without error
        result = homography.compute_homography(frame, reference)

        assert result is not None
        assert result.homography_matrix.shape == (3, 3)
        assert result.confidence >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
