#!/usr/bin/env python3
"""
Test suite for GeoTIFF 6-parameter affine transform via GeoTiff VO.

Tests Issue #133: Fix GeoTIFF pixel→UTM transform to use full 6-parameter affine.

The GDAL GeoTransform standard defines pixel-to-coordinate transformation as:
    Xgeo = GT[0] + P*GT[1] + L*GT[2]
    Ygeo = GT[3] + P*GT[4] + L*GT[5]
"""

import pytest

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.types import Easting, Meters, Northing, Unitless


def _make_geotiff(gt: list[float], crs: str = "EPSG:25830") -> GeoTiff:
    """Helper to build a GeoTiff VO from a raw 6-element list."""
    return GeoTiff(
        geotransform=GeoTransform(
            origin_easting=Easting(gt[0]),
            pixel_width=Meters(gt[1]),
            row_rotation=Unitless(gt[2]),
            origin_northing=Northing(gt[3]),
            col_rotation=Unitless(gt[4]),
            pixel_height=Meters(gt[5]),
        ),
        crs=crs,
    )


class TestPixelToGeo:
    """Test the GeoTiff.pixel_to_geo method."""

    def test_north_up_raster_origin(self):
        """Test north-up raster transform at origin (0, 0)."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        easting, northing = geotiff.pixel_to_geo(0, 0)

        assert easting == pytest.approx(737575.05, abs=0.01)
        assert northing == pytest.approx(4391595.45, abs=0.01)

    def test_north_up_raster_offset_pixel(self):
        """Test north-up raster transform at offset pixel."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        easting, northing = geotiff.pixel_to_geo(10, 20)

        assert easting == pytest.approx(737576.55, abs=0.01)
        assert northing == pytest.approx(4391592.45, abs=0.01)

    def test_north_up_raster_large_offset(self):
        """Test north-up raster transform with large pixel offset."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        easting, northing = geotiff.pixel_to_geo(1000, 2000)

        assert easting == pytest.approx(737725.05, abs=0.01)
        assert northing == pytest.approx(4391295.45, abs=0.01)

    def test_rotated_raster_22_5_degrees(self):
        """Test rotated raster (22.5° clockwise) affine transform."""
        geotiff = _make_geotiff([500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387])

        easting, northing = geotiff.pixel_to_geo(0, 0)
        assert easting == pytest.approx(500000, abs=0.01)
        assert northing == pytest.approx(4400000, abs=0.01)

        easting, northing = geotiff.pixel_to_geo(100, 0)
        assert easting == pytest.approx(500013.87, abs=0.01)
        assert northing == pytest.approx(4400005.74, abs=0.01)

        easting, northing = geotiff.pixel_to_geo(0, 100)
        assert easting == pytest.approx(500005.74, abs=0.01)
        assert northing == pytest.approx(4399986.13, abs=0.01)

    def test_rotated_raster_combined_offset(self):
        """Test rotated raster with combined pixel offsets."""
        geotiff = _make_geotiff([500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387])

        easting, northing = geotiff.pixel_to_geo(50, 75)

        assert easting == pytest.approx(500011.24, abs=0.1)
        assert northing == pytest.approx(4399992.47, abs=0.1)

    def test_half_pixel_center_offset(self):
        """Test pixel center offset convention (GDAL uses pixel corner)."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        corner_e, corner_n = geotiff.pixel_to_geo(0, 0)
        center_e, center_n = geotiff.pixel_to_geo(0.5, 0.5)

        assert center_e == pytest.approx(corner_e + 0.075, abs=0.001)
        assert center_n == pytest.approx(corner_n - 0.075, abs=0.001)

    def test_negative_pixel_coordinates(self):
        """Test that negative pixel coordinates work correctly."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        easting, northing = geotiff.pixel_to_geo(-10, -20)

        assert easting == pytest.approx(737573.55, abs=0.01)
        assert northing == pytest.approx(4391598.45, abs=0.01)

    def test_fractional_pixel_coordinates(self):
        """Test that fractional pixel coordinates interpolate correctly."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        easting, northing = geotiff.pixel_to_geo(10.5, 20.25)

        assert easting == pytest.approx(737576.625, abs=0.001)
        assert northing == pytest.approx(4391592.4125, abs=0.001)

    def test_return_types(self):
        """Test that pixel_to_geo returns float tuple."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])
        result = geotiff.pixel_to_geo(10, 20)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (int, float))
        assert isinstance(result[1], (int, float))


class TestGeotransformEquivalence:
    """Test that 6-parameter transform is equivalent to 4-parameter for north-up rasters."""

    def test_north_up_equivalent_to_simplified_formula(self):
        """Test that north-up raster (rotation=0) matches simplified formula."""
        origin_easting = 737575.05
        origin_northing = 4391595.45
        pixel_size_x = 0.15
        pixel_size_y = -0.15

        geotiff = _make_geotiff([origin_easting, pixel_size_x, 0, origin_northing, 0, pixel_size_y])

        test_pixels = [(0, 0), (10, 20), (100, 200), (1000, 2000)]

        for px, py in test_pixels:
            simple_easting = origin_easting + (px * pixel_size_x)
            simple_northing = origin_northing + (py * pixel_size_y)

            affine_easting, affine_northing = geotiff.pixel_to_geo(px, py)

            assert affine_easting == pytest.approx(simple_easting, abs=0.001)
            assert affine_northing == pytest.approx(simple_northing, abs=0.001)


class TestGeotransformEdgeCases:
    """Test edge cases."""

    def test_zero_pixel_offset(self):
        """Test that (0, 0) pixel always returns origin."""
        geotiff = _make_geotiff([500000, 0.15, 0.05, 4400000, 0.05, -0.15])
        easting, northing = geotiff.pixel_to_geo(0, 0)

        assert easting == pytest.approx(500000, abs=0.0001)
        assert northing == pytest.approx(4400000, abs=0.0001)

    def test_large_rotation_values(self):
        """Test with large rotation values (extreme non-north-up case)."""
        geotiff = _make_geotiff([500000, 0, 0.15, 4400000, 0.15, 0])

        easting, northing = geotiff.pixel_to_geo(100, 0)
        assert easting == pytest.approx(500000, abs=0.01)
        assert northing == pytest.approx(4400015, abs=0.01)

        easting, northing = geotiff.pixel_to_geo(0, 100)
        assert easting == pytest.approx(500015, abs=0.01)
        assert northing == pytest.approx(4400000, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
