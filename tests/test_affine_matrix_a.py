#!/usr/bin/env python3
"""
Test suite for affine matrix A implementation in CameraGeometry.

Tests Issue #122: Implement explicit affine map A for reference-to-world transformation.

The A matrix represents the affine transformation from reference image pixels to
world ground plane coordinates. For georeferenced ortho imagery (GeoTIFF), this
transformation consists of:
- Scale: meters per pixel (pixel_size_x, pixel_size_y from GeoTIFF metadata)
- Translation: offset from reference image origin to camera position in world coordinates

Mathematical Formula:
    A = [[pixel_size_x, 0,           t_x],
         [0,            pixel_size_y, t_y],
         [0,            0,            1  ]]

Where:
    t_x = origin_easting - camera_easting
    t_y = origin_northing - camera_northing
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry


class TestAffineMatrixInitialization:
    """Test that A matrix is properly initialized."""

    def test_a_matrix_exists_on_initialization(self):
        """Test that A matrix property exists after CameraGeometry initialization."""
        geo = CameraGeometry(1920, 1080)
        assert hasattr(geo, "A"), "CameraGeometry should have A matrix property"

    def test_a_matrix_defaults_to_identity(self):
        """Test that A matrix defaults to 3x3 identity matrix."""
        geo = CameraGeometry(1920, 1080)
        expected = np.eye(3)
        np.testing.assert_array_equal(
            geo.A, expected, err_msg="A matrix should default to identity matrix"
        )

    def test_a_matrix_is_numpy_array(self):
        """Test that A matrix is a numpy array."""
        geo = CameraGeometry(1920, 1080)
        assert isinstance(geo.A, np.ndarray), "A matrix should be numpy array"

    def test_a_matrix_is_3x3(self):
        """Test that A matrix has correct shape (3x3)."""
        geo = CameraGeometry(1920, 1080)
        assert geo.A.shape == (3, 3), "A matrix should be 3x3"


class TestSetGeotiffParams:
    """Test set_geotiff_params method for computing A matrix."""

    def test_set_geotiff_params_method_exists(self):
        """Test that set_geotiff_params method exists."""
        geo = CameraGeometry(1920, 1080)
        assert hasattr(geo, "set_geotiff_params"), (
            "CameraGeometry should have set_geotiff_params method"
        )

    def test_set_geotiff_params_accepts_valid_inputs(self):
        """Test that set_geotiff_params accepts valid geotiff_params and camera_utm_position."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        # Should not raise any exception
        geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_compute_a_matrix_with_valid_params(self):
        """Test A matrix computation with valid geotiff parameters."""
        geo = CameraGeometry(1920, 1080)

        # Example: pixel size 0.5m, camera offset by (10m East, 20m North) from origin
        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Expected: t_x = origin_easting - camera_easting = 500000 - 500010 = -10
        #           t_y = origin_northing - camera_northing = 4000000 - 4000020 = -20
        expected_A = np.array([[0.5, 0.0, -10.0], [0.0, 0.5, -20.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(
            geo.A,
            expected_A,
            decimal=10,
            err_msg="A matrix should be computed correctly from geotiff params",
        )

    def test_compute_a_matrix_different_pixel_sizes(self):
        """Test A matrix computation with different pixel sizes in X and Y."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.3,
            "pixel_size_y": 0.4,
            "origin_easting": 600000.0,
            "origin_northing": 5000000.0,
        }
        camera_utm_position = (600005.0, 5000008.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # t_x = 600000 - 600005 = -5
        # t_y = 5000000 - 5000008 = -8
        expected_A = np.array([[0.3, 0.0, -5.0], [0.0, 0.4, -8.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=10)

    def test_compute_a_matrix_camera_southwest_of_origin(self):
        """Test A matrix when camera is southwest of origin (positive translations)."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 1.0,
            "pixel_size_y": 1.0,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        # Camera is southwest (smaller easting and northing)
        camera_utm_position = (499990.0, 3999980.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # t_x = 500000 - 499990 = 10
        # t_y = 4000000 - 3999980 = 20
        expected_A = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=10)


class TestSetGeotiffParamsValidation:
    """Test validation and error handling in set_geotiff_params."""

    def test_none_geotiff_params_sets_identity(self):
        """Test that None geotiff_params sets A to identity (backward compatibility)."""
        geo = CameraGeometry(1920, 1080)
        camera_utm_position = (500000.0, 4000000.0)

        geo.set_geotiff_params(None, camera_utm_position)

        expected = np.eye(3)
        np.testing.assert_array_equal(
            geo.A, expected, err_msg="A should be identity when geotiff_params is None"
        )

    def test_none_camera_utm_position_sets_identity(self):
        """Test that None camera_utm_position sets A to identity (backward compatibility)."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }

        geo.set_geotiff_params(geotiff_params, None)

        expected = np.eye(3)
        np.testing.assert_array_equal(
            geo.A, expected, err_msg="A should be identity when camera_utm_position is None"
        )

    def test_both_none_sets_identity(self):
        """Test that both None parameters set A to identity."""
        geo = CameraGeometry(1920, 1080)

        geo.set_geotiff_params(None, None)

        expected = np.eye(3)
        np.testing.assert_array_equal(
            geo.A, expected, err_msg="A should be identity when both params are None"
        )

    def test_missing_required_key_pixel_size_x(self):
        """Test that missing 'pixel_size_x' raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="pixel_size_x"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_missing_required_key_pixel_size_y(self):
        """Test that missing 'pixel_size_y' raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="pixel_size_y"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_missing_required_key_origin_easting(self):
        """Test that missing 'origin_easting' raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {"pixel_size_x": 0.5, "pixel_size_y": 0.5, "origin_northing": 4000000.0}
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="origin_easting"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_missing_required_key_origin_northing(self):
        """Test that missing 'origin_northing' raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {"pixel_size_x": 0.5, "pixel_size_y": 0.5, "origin_easting": 500000.0}
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="origin_northing"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_camera_utm_position_not_tuple(self):
        """Test that camera_utm_position must be a tuple."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }

        # Pass list instead of tuple
        with pytest.raises(TypeError, match="tuple"):
            geo.set_geotiff_params(geotiff_params, [500010.0, 4000020.0])

    def test_camera_utm_position_wrong_length(self):
        """Test that camera_utm_position must have exactly 2 elements."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }

        # Pass 3-element tuple
        with pytest.raises(ValueError, match="2.*elements|2-tuple"):
            geo.set_geotiff_params(geotiff_params, (500010.0, 4000020.0, 0.0))

    def test_pixel_size_x_zero_raises_value_error(self):
        """Test that zero pixel_size_x raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.0,  # Invalid: zero
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="pixel_size_x.*non-zero"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_pixel_size_y_zero_raises_value_error(self):
        """Test that zero pixel_size_y raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.0,  # Invalid: zero
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="pixel_size_y.*non-zero"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_pixel_size_y_negative_is_valid(self):
        """Test that negative pixel_size_y is valid (common in GeoTIFF)."""
        geo = CameraGeometry(1920, 1080)

        # GeoTIFF commonly has negative Y pixel size because image Y goes down
        # while geographic northing goes up
        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": -0.5,  # Valid: negative for GeoTIFF
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        # Should not raise - negative pixel_size_y is valid
        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Verify A matrix was computed with negative Y scale
        assert geo.A[1, 1] == -0.5

    def test_non_numeric_pixel_size_raises_type_error(self):
        """Test that non-numeric pixel_size raises TypeError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": "0.5",  # Invalid: string
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(TypeError, match="pixel_size_x.*numeric"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_nan_pixel_size_raises_value_error(self):
        """Test that NaN pixel_size raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": float("nan"),  # Invalid: NaN
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="pixel_size_x.*finite"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_infinity_origin_raises_value_error(self):
        """Test that infinity origin raises ValueError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": float("inf"),  # Invalid: infinity
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with pytest.raises(ValueError, match="origin_easting.*finite"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_non_numeric_camera_position_raises_type_error(self):
        """Test that non-numeric camera position raises TypeError."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        # First element is a string
        camera_utm_position = ("500010.0", 4000020.0)

        with pytest.raises(TypeError, match="easting.*numeric"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)


class TestSetGeotiffParamsLogging:
    """Test logging behavior of set_geotiff_params."""

    @patch("poc_homography.camera_geometry.logger")
    def test_logs_a_matrix_computation(self, mock_logger):
        """Test that A matrix computation is logged at INFO level."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Verify logger.info was called with A matrix information
        mock_logger.info.assert_called()
        # Check that one of the calls mentions the A matrix
        call_args_list = [str(call) for call in mock_logger.info.call_args_list]
        assert any(
            "A matrix" in str(call) or "affine" in str(call).lower() for call in call_args_list
        ), "Should log A matrix computation at INFO level"


class TestAffineMatrixDocumentation:
    """Test that A matrix has proper documentation."""

    def test_set_geotiff_params_has_docstring(self):
        """Test that set_geotiff_params method has a docstring."""
        geo = CameraGeometry(1920, 1080)
        assert geo.set_geotiff_params.__doc__ is not None, (
            "set_geotiff_params should have a docstring"
        )
        assert len(geo.set_geotiff_params.__doc__) > 50, (
            "set_geotiff_params docstring should be descriptive"
        )

    def test_docstring_mentions_coordinate_system(self):
        """Test that docstring mentions coordinate system or transformation."""
        geo = CameraGeometry(1920, 1080)
        docstring = geo.set_geotiff_params.__doc__.lower()
        assert any(
            keyword in docstring
            for keyword in ["coordinate", "transformation", "affine", "geotiff", "utm"]
        ), "Docstring should explain coordinate transformation purpose"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
