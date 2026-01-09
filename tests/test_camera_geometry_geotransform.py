#!/usr/bin/env python3
"""
Test suite for CameraGeometry with 6-parameter GeoTransform support.

Tests Issue #133: Update CameraGeometry.set_geotiff_params() to accept full
6-parameter GDAL GeoTransform array and build A matrix with rotation terms.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry


class TestSetGeotiffParamsWithGeotransform:
    """Test set_geotiff_params with new geotransform array format."""

    def test_accepts_geotransform_array(self):
        """Test that set_geotiff_params accepts geotransform array."""
        geo = CameraGeometry(1920, 1080)

        # New format: geotransform array
        geotiff_params = {"geotransform": [737575.05, 0.15, 0, 4391595.45, 0, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        # Should not raise any exception
        geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_builds_a_matrix_north_up_raster(self):
        """Test A matrix construction for north-up raster (no rotation)."""
        geo = CameraGeometry(1920, 1080)

        # North-up raster: GT[2]=0, GT[4]=0
        geotiff_params = {"geotransform": [500000.0, 0.5, 0, 4000000.0, 0, -0.5]}
        camera_utm_position = (500010.0, 4000020.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Expected A matrix:
        # GT[1]=0.5, GT[2]=0, t_x = 500000-500010 = -10
        # GT[4]=0, GT[5]=-0.5, t_y = 4000000-4000020 = -20
        expected_A = np.array([[0.5, 0.0, -10.0], [0.0, -0.5, -20.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=10)

    def test_builds_a_matrix_with_rotation(self):
        """Test A matrix construction for rotated raster."""
        geo = CameraGeometry(1920, 1080)

        # Rotated raster (22.5° clockwise)
        geotiff_params = {"geotransform": [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]}
        camera_utm_position = (500000, 4400000)  # At origin for simplicity

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Expected A matrix with rotation terms:
        # t_x = 0, t_y = 0 (camera at origin)
        expected_A = np.array([[0.1387, 0.0574, 0.0], [0.0574, -0.1387, 0.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=4)

    def test_builds_a_matrix_with_rotation_and_translation(self):
        """Test A matrix with both rotation and translation components."""
        geo = CameraGeometry(1920, 1080)

        # Rotated raster with camera offset
        geotiff_params = {"geotransform": [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]}
        camera_utm_position = (500010, 4400020)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Expected A matrix:
        # Rotation: GT[1]=0.1387, GT[2]=0.0574, GT[4]=0.0574, GT[5]=-0.1387
        # Translation: t_x = 500000-500010 = -10, t_y = 4400000-4400020 = -20
        expected_A = np.array([[0.1387, 0.0574, -10.0], [0.0574, -0.1387, -20.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=4)


class TestBackwardCompatibilityOldFormat:
    """Test backward compatibility with old 4-parameter format."""

    def test_accepts_old_format_with_deprecation_warning(self):
        """Test that old 4-parameter format still works with deprecation warning."""
        geo = CameraGeometry(1920, 1080)

        # Old format: separate keys
        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": -0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        # Should emit deprecation warning but still work
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

            # Check that a deprecation warning was issued
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "geotransform" in str(w[0].message).lower()

    def test_old_format_produces_correct_a_matrix(self):
        """Test that old format produces same A matrix as before (no rotation)."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": -0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected deprecation warning
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Should produce same result as old implementation
        expected_A = np.array([[0.5, 0.0, -10.0], [0.0, -0.5, -20.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(geo.A, expected_A, decimal=10)

    def test_old_format_equivalent_to_new_format(self):
        """Test that old format produces same A as new geotransform format (when rotation=0)."""
        geo_old = CameraGeometry(1920, 1080)
        geo_new = CameraGeometry(1920, 1080)

        # Old format
        old_params = {
            "pixel_size_x": 0.15,
            "pixel_size_y": -0.15,
            "origin_easting": 737575.05,
            "origin_northing": 4391595.45,
        }

        # New format (equivalent geotransform with rotation=0)
        new_params = {"geotransform": [737575.05, 0.15, 0, 4391595.45, 0, -0.15]}

        camera_utm_position = (737580.0, 4391600.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geo_old.set_geotiff_params(old_params, camera_utm_position)

        geo_new.set_geotiff_params(new_params, camera_utm_position)

        # Both should produce identical A matrix
        np.testing.assert_array_almost_equal(geo_old.A, geo_new.A, decimal=10)


class TestGeotransformValidation:
    """Test validation for geotransform array."""

    def test_geotransform_must_have_six_elements(self):
        """Test that geotransform must have exactly 6 elements."""
        geo = CameraGeometry(1920, 1080)

        # Invalid: only 4 elements
        geotiff_params = {"geotransform": [737575.05, 0.15, 4391595.45, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        with pytest.raises(ValueError, match="6 elements|length"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_geotransform_elements_must_be_numeric(self):
        """Test that geotransform elements must be numeric."""
        geo = CameraGeometry(1920, 1080)

        # Invalid: contains string
        geotiff_params = {"geotransform": [737575.05, "0.15", 0, 4391595.45, 0, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        with pytest.raises(TypeError, match="numeric"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_geotransform_elements_must_be_finite(self):
        """Test that geotransform elements must be finite."""
        geo = CameraGeometry(1920, 1080)

        # Invalid: contains NaN
        geotiff_params = {"geotransform": [737575.05, 0.15, float("nan"), 4391595.45, 0, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        with pytest.raises(ValueError, match="finite"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_geotransform_pixel_sizes_cannot_be_zero(self):
        """Test that pixel sizes (GT[1] and GT[5]) cannot be zero."""
        geo = CameraGeometry(1920, 1080)

        # Invalid: GT[1]=0 (zero pixel width)
        geotiff_params = {"geotransform": [737575.05, 0, 0, 4391595.45, 0, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        with pytest.raises(ValueError, match="GT\\[1\\].*non-zero|pixel.*width"):
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

    def test_geotransform_allows_negative_pixel_height(self):
        """Test that negative pixel height (GT[5]) is valid."""
        geo = CameraGeometry(1920, 1080)

        # Valid: GT[5] is negative (common for GeoTIFF)
        geotiff_params = {"geotransform": [737575.05, 0.15, 0, 4391595.45, 0, -0.15]}
        camera_utm_position = (737575.0, 4391595.0)

        # Should not raise
        geo.set_geotiff_params(geotiff_params, camera_utm_position)
        assert geo.A[1, 1] < 0  # Verify negative Y scale preserved


class TestGeotransformLogging:
    """Test logging behavior for geotransform-based A matrix computation."""

    def test_logs_geotransform_computation(self):
        """Test that geotransform A matrix computation is logged."""
        from unittest.mock import patch

        geo = CameraGeometry(1920, 1080)

        geotiff_params = {"geotransform": [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]}
        camera_utm_position = (500000, 4400000)

        with patch("poc_homography.camera_geometry.logger") as mock_logger:
            geo.set_geotiff_params(geotiff_params, camera_utm_position)

            # Verify logger.info was called with geotransform information
            mock_logger.info.assert_called()
            call_args_list = [str(call) for call in mock_logger.info.call_args_list]
            assert any(
                "geotransform" in str(call).lower() or "affine" in str(call).lower()
                for call in call_args_list
            ), "Should log geotransform A matrix computation"


class TestGeotransformDocumentation:
    """Test that geotransform format is properly documented."""

    def test_docstring_mentions_geotransform(self):
        """Test that set_geotiff_params docstring mentions geotransform."""
        geo = CameraGeometry(1920, 1080)
        docstring = geo.set_geotiff_params.__doc__

        # Docstring should now mention geotransform format
        assert "geotransform" in docstring.lower(), (
            "Docstring should document geotransform array format"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
