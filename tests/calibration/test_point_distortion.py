"""Tests for distort_points and undistort_points functions.

Tests the bidirectional point transformation functions that convert
between distorted and undistorted coordinate spaces.
"""

from __future__ import annotations

import numpy as np

from poc_homography.calibration.lens_distortion.apply_calibration import (
    distort_points,
    undistort_points,
)

# Standard intrinsics for a 1920x1080 image
DEFAULT_FX = 1500.0
DEFAULT_FY = 1500.0
DEFAULT_CX = 960.0
DEFAULT_CY = 540.0


class TestDistortPoints:
    """Tests for the distort_points function."""

    def test_zero_coefficients_returns_same_points(self):
        """With zero distortion coefficients, points should not change."""
        points = np.array([
            [960.0, 540.0],  # Center
            [0.0, 0.0],      # Corner
            [1920.0, 1080.0],  # Opposite corner
            [1500.0, 800.0],   # Random point
        ])

        result = distort_points(
            points,
            k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(result, points, decimal=10)

    def test_principal_point_unchanged(self):
        """The principal point should remain fixed under any distortion."""
        points = np.array([[DEFAULT_CX, DEFAULT_CY]])

        result = distort_points(
            points,
            k1=-0.2, k2=0.05, k3=-0.001, p1=0.001, p2=-0.001,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(result, points, decimal=10)

    def test_barrel_distortion_moves_points_outward(self):
        """Negative k1 (barrel distortion) should move points toward center."""
        # Point to the right of center
        points = np.array([[1200.0, 540.0]])

        result = distort_points(
            points,
            k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        # With barrel distortion, the distorted point should be closer to center
        original_dist = abs(points[0, 0] - DEFAULT_CX)
        distorted_dist = abs(result[0, 0] - DEFAULT_CX)
        assert distorted_dist < original_dist

    def test_pincushion_distortion_moves_points_outward(self):
        """Positive k1 (pincushion distortion) should move points away from center."""
        # Point to the right of center
        points = np.array([[1200.0, 540.0]])

        result = distort_points(
            points,
            k1=0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        # With pincushion distortion, the distorted point should be farther from center
        original_dist = abs(points[0, 0] - DEFAULT_CX)
        distorted_dist = abs(result[0, 0] - DEFAULT_CX)
        assert distorted_dist > original_dist


class TestRoundtrip:
    """Tests for roundtrip consistency between distort and undistort."""

    def test_roundtrip_distort_then_undistort(self):
        """distort then undistort should return approximately original points."""
        original = np.array([
            [960.0, 540.0],   # Center
            [500.0, 300.0],   # Upper left quadrant
            [1400.0, 800.0],  # Lower right quadrant
            [200.0, 900.0],   # Lower left
        ])

        k1, k2, k3 = -0.15, 0.02, -0.001
        p1, p2 = 0.0005, -0.0003

        distorted = distort_points(
            original,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        recovered = undistort_points(
            distorted,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(recovered, original, decimal=4)

    def test_roundtrip_undistort_then_distort(self):
        """undistort then distort should return approximately original points."""
        original = np.array([
            [960.0, 540.0],   # Center
            [500.0, 300.0],   # Upper left quadrant
            [1400.0, 800.0],  # Lower right quadrant
        ])

        k1, k2, k3 = -0.2, 0.03, 0.0
        p1, p2 = 0.001, 0.001

        undistorted = undistort_points(
            original,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        recovered = distort_points(
            undistorted,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(recovered, original, decimal=4)

    def test_roundtrip_with_typical_coefficients(self):
        """Test roundtrip with coefficients typical of real camera lenses."""
        # Grid of points across the image
        x = np.linspace(100, 1800, 5)
        y = np.linspace(100, 980, 4)
        xx, yy = np.meshgrid(x, y)
        original = np.column_stack([xx.ravel(), yy.ravel()])

        # Typical PTZ camera coefficients
        k1, k2, k3 = -0.08, 0.015, -0.002
        p1, p2 = 0.0002, -0.0001

        distorted = distort_points(
            original,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        recovered = undistort_points(
            distorted,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(recovered, original, decimal=3)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_point(self):
        """Function should work with a single point."""
        point = np.array([[800.0, 400.0]])

        result = distort_points(
            point,
            k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        assert result.shape == (1, 2)

    def test_image_corners(self):
        """Test transformation of image corner points."""
        corners = np.array([
            [0.0, 0.0],         # Top-left
            [1920.0, 0.0],      # Top-right
            [0.0, 1080.0],      # Bottom-left
            [1920.0, 1080.0],   # Bottom-right
        ])

        k1, k2, k3 = -0.1, 0.01, 0.0
        p1, p2 = 0.0, 0.0

        distorted = distort_points(
            corners,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        recovered = undistort_points(
            distorted,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        np.testing.assert_array_almost_equal(recovered, corners, decimal=3)

    def test_horizontal_line_remains_horizontal_at_center(self):
        """A horizontal line through the center should remain horizontal."""
        # Points along y = cy (horizontal through center)
        x_coords = np.linspace(100, 1800, 10)
        points = np.column_stack([x_coords, np.full(10, DEFAULT_CY)])

        result = distort_points(
            points,
            k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        # All y-coordinates should remain at cy (within numerical precision)
        np.testing.assert_array_almost_equal(result[:, 1], np.full(10, DEFAULT_CY), decimal=10)

    def test_vertical_line_remains_vertical_at_center(self):
        """A vertical line through the center should remain vertical."""
        # Points along x = cx (vertical through center)
        y_coords = np.linspace(100, 980, 10)
        points = np.column_stack([np.full(10, DEFAULT_CX), y_coords])

        result = distort_points(
            points,
            k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        # All x-coordinates should remain at cx (within numerical precision)
        np.testing.assert_array_almost_equal(result[:, 0], np.full(10, DEFAULT_CX), decimal=10)


class TestTangentialDistortion:
    """Tests specifically for tangential distortion components."""

    def test_tangential_only_distortion(self):
        """Test with only tangential distortion coefficients."""
        points = np.array([
            [1100.0, 600.0],
            [820.0, 480.0],
        ])

        # Only tangential distortion
        result = distort_points(
            points,
            k1=0.0, k2=0.0, k3=0.0, p1=0.002, p2=-0.001,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )

        # Points should move slightly (tangential causes decentering)
        assert not np.allclose(result, points)

        # Roundtrip should recover original
        recovered = undistort_points(
            result,
            k1=0.0, k2=0.0, k3=0.0, p1=0.002, p2=-0.001,
            fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY
        )
        np.testing.assert_array_almost_equal(recovered, points, decimal=4)
