#!/usr/bin/env python3
"""
Test suite for GeoTIFF 6-parameter affine transform implementation.

Tests Issue #133: Fix GeoTIFF pixel→UTM transform to use full 6-parameter affine.

The GDAL GeoTransform standard defines pixel-to-coordinate transformation as:
    Xgeo = GT[0] + P*GT[1] + L*GT[2]
    Ygeo = GT[3] + P*GT[4] + L*GT[5]

Where:
    GT[0]: X-coordinate of upper-left corner (origin easting)
    GT[1]: Pixel width (meters per pixel in X direction)
    GT[2]: Row rotation (typically 0 for north-up images)
    GT[3]: Y-coordinate of upper-left corner (origin northing)
    GT[4]: Column rotation (typically 0 for north-up images)
    GT[5]: Pixel height (meters per pixel in Y direction, typically negative)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.geotiff_utils import apply_geotransform


class TestApplyGeotransform:
    """Test the apply_geotransform utility function."""

    def test_north_up_raster_origin(self):
        """Test north-up raster transform at origin (0, 0)."""
        # North-up raster: no rotation (GT[2]=0, GT[4]=0)
        # Based on real config: origin=(737575.05, 4391595.45), pixel_size=(0.15, -0.15)
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # At pixel (0, 0), should return origin coordinates
        easting, northing = apply_geotransform(0, 0, gt)

        assert easting == pytest.approx(737575.05, abs=0.01), "Easting at origin should match GT[0]"
        assert northing == pytest.approx(4391595.45, abs=0.01), (
            "Northing at origin should match GT[3]"
        )

    def test_north_up_raster_offset_pixel(self):
        """Test north-up raster transform at offset pixel."""
        # North-up raster with 0.15m pixels
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # At pixel (10, 20):
        # Expected easting = 737575.05 + 10*0.15 + 20*0 = 737576.55
        # Expected northing = 4391595.45 + 10*0 + 20*(-0.15) = 4391592.45
        easting, northing = apply_geotransform(10, 20, gt)

        assert easting == pytest.approx(737576.55, abs=0.01), (
            "Easting should offset by px * pixel_size_x"
        )
        assert northing == pytest.approx(4391592.45, abs=0.01), (
            "Northing should offset by py * pixel_size_y (negative for GeoTIFF)"
        )

    def test_north_up_raster_large_offset(self):
        """Test north-up raster transform with large pixel offset."""
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # At pixel (1000, 2000):
        # Expected easting = 737575.05 + 1000*0.15 = 737725.05
        # Expected northing = 4391595.45 + 2000*(-0.15) = 4391295.45
        easting, northing = apply_geotransform(1000, 2000, gt)

        assert easting == pytest.approx(737725.05, abs=0.01)
        assert northing == pytest.approx(4391295.45, abs=0.01)

    def test_rotated_raster_22_5_degrees(self):
        """Test rotated raster (22.5° clockwise) affine transform."""
        # Rotated raster: GT[2] and GT[4] are non-zero
        # 22.5° rotation: cos(22.5°) ≈ 0.9239, sin(22.5°) ≈ 0.3827
        # For 0.15m pixels: 0.15 * cos(22.5°) ≈ 0.1387, 0.15 * sin(22.5°) ≈ 0.0574
        gt = [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]

        # At pixel (0, 0), should return origin
        easting, northing = apply_geotransform(0, 0, gt)
        assert easting == pytest.approx(500000, abs=0.01)
        assert northing == pytest.approx(4400000, abs=0.01)

        # At pixel (100, 0): moves in +X direction (along row)
        # Expected easting = 500000 + 100*0.1387 + 0*0.0574 = 500013.87
        # Expected northing = 4400000 + 100*0.0574 + 0*(-0.1387) = 4400005.74
        easting, northing = apply_geotransform(100, 0, gt)
        assert easting == pytest.approx(500013.87, abs=0.01)
        assert northing == pytest.approx(4400005.74, abs=0.01)

        # At pixel (0, 100): moves in +Y direction (along column)
        # Expected easting = 500000 + 0*0.1387 + 100*0.0574 = 500005.74
        # Expected northing = 4400000 + 0*0.0574 + 100*(-0.1387) = 4399986.13
        easting, northing = apply_geotransform(0, 100, gt)
        assert easting == pytest.approx(500005.74, abs=0.01)
        assert northing == pytest.approx(4399986.13, abs=0.01)

    def test_rotated_raster_combined_offset(self):
        """Test rotated raster with combined pixel offsets."""
        gt = [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]

        # At pixel (50, 75): both X and Y offset with rotation
        # Expected easting = 500000 + 50*0.1387 + 75*0.0574 = 500011.24
        # Expected northing = 4400000 + 50*0.0574 + 75*(-0.1387) = 4399992.4675
        easting, northing = apply_geotransform(50, 75, gt)
        assert easting == pytest.approx(500011.24, abs=0.1), (
            "Rotated raster should handle combined offsets with rotation tolerance"
        )
        assert northing == pytest.approx(4399992.47, abs=0.1), (
            "Rotated raster northing should reflect rotation effects"
        )

    def test_half_pixel_center_offset(self):
        """Test pixel center offset convention (GDAL uses pixel corner)."""
        # GDAL GeoTransform references pixel CORNER (upper-left)
        # To get pixel CENTER coordinates, add 0.5 to px and py
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # Pixel (0, 0) corner
        corner_e, corner_n = apply_geotransform(0, 0, gt)

        # Pixel (0, 0) center: add half pixel offset
        center_e, center_n = apply_geotransform(0.5, 0.5, gt)

        # Center should be 0.5 pixels (0.075m) offset from corner
        assert center_e == pytest.approx(corner_e + 0.075, abs=0.001), (
            "Pixel center should be +0.5 pixels east of corner"
        )
        assert center_n == pytest.approx(corner_n - 0.075, abs=0.001), (
            "Pixel center should be -0.5 pixels north of corner (negative pixel_size_y)"
        )

    def test_negative_pixel_coordinates(self):
        """Test that negative pixel coordinates work correctly."""
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # At pixel (-10, -20): should extrapolate correctly
        # Expected easting = 737575.05 + (-10)*0.15 = 737573.55
        # Expected northing = 4391595.45 + (-20)*(-0.15) = 4391598.45
        easting, northing = apply_geotransform(-10, -20, gt)

        assert easting == pytest.approx(737573.55, abs=0.01)
        assert northing == pytest.approx(4391598.45, abs=0.01)

    def test_fractional_pixel_coordinates(self):
        """Test that fractional pixel coordinates interpolate correctly."""
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]

        # At pixel (10.5, 20.25):
        # Expected easting = 737575.05 + 10.5*0.15 = 737576.625
        # Expected northing = 4391595.45 + 20.25*(-0.15) = 4391592.4125
        easting, northing = apply_geotransform(10.5, 20.25, gt)

        assert easting == pytest.approx(737576.625, abs=0.001)
        assert northing == pytest.approx(4391592.4125, abs=0.001)

    def test_geotransform_with_six_elements(self):
        """Test that geotransform must have exactly 6 elements."""
        # Valid 6-element geotransform
        gt_valid = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]
        easting, northing = apply_geotransform(0, 0, gt_valid)
        assert isinstance(easting, (int, float))
        assert isinstance(northing, (int, float))

    def test_geotransform_return_types(self):
        """Test that apply_geotransform returns float tuple."""
        gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]
        result = apply_geotransform(10, 20, gt)

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-tuple"
        assert isinstance(result[0], (int, float)), "Easting should be numeric"
        assert isinstance(result[1], (int, float)), "Northing should be numeric"


class TestGeotransformEquivalence:
    """Test that 6-parameter transform is equivalent to 4-parameter for north-up rasters."""

    def test_north_up_equivalent_to_simplified_formula(self):
        """Test that north-up raster (rotation=0) matches simplified formula."""
        # Old simplified formula: easting = origin_easting + px*pixel_size_x
        # Should match 6-parameter formula when GT[2]=0 and GT[4]=0

        origin_easting = 737575.05
        origin_northing = 4391595.45
        pixel_size_x = 0.15
        pixel_size_y = -0.15

        # Build 6-parameter geotransform with no rotation
        gt = [origin_easting, pixel_size_x, 0, origin_northing, 0, pixel_size_y]

        # Test multiple pixel coordinates
        test_pixels = [(0, 0), (10, 20), (100, 200), (1000, 2000)]

        for px, py in test_pixels:
            # Simplified formula
            simple_easting = origin_easting + (px * pixel_size_x)
            simple_northing = origin_northing + (py * pixel_size_y)

            # 6-parameter formula
            affine_easting, affine_northing = apply_geotransform(px, py, gt)

            assert affine_easting == pytest.approx(simple_easting, abs=0.001), (
                f"6-param should match simplified for north-up at ({px}, {py})"
            )
            assert affine_northing == pytest.approx(simple_northing, abs=0.001), (
                f"6-param should match simplified for north-up at ({px}, {py})"
            )


class TestGeotransformEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_pixel_offset(self):
        """Test that (0, 0) pixel always returns origin."""
        gt = [500000, 0.15, 0.05, 4400000, 0.05, -0.15]
        easting, northing = apply_geotransform(0, 0, gt)

        assert easting == pytest.approx(gt[0], abs=0.0001)
        assert northing == pytest.approx(gt[3], abs=0.0001)

    def test_large_rotation_values(self):
        """Test with large rotation values (extreme non-north-up case)."""
        # 90° rotation: cos(90°)=0, sin(90°)=1
        # For 0.15m pixels: GT[1]=0, GT[2]=0.15, GT[4]=0.15, GT[5]=0
        gt = [500000, 0, 0.15, 4400000, 0.15, 0]

        # At pixel (100, 0): should only affect Y coordinate
        easting, northing = apply_geotransform(100, 0, gt)
        assert easting == pytest.approx(500000, abs=0.01)
        assert northing == pytest.approx(4400015, abs=0.01)

        # At pixel (0, 100): should only affect X coordinate
        easting, northing = apply_geotransform(0, 100, gt)
        assert easting == pytest.approx(500015, abs=0.01)
        assert northing == pytest.approx(4400000, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
