"""
Test suite for GeoTiff value object coordinate transformations.

Tests the pixel_to_geo and geo_to_pixel methods which implement GDAL's
6-parameter affine GeoTransform for converting between pixel coordinates
and geographic/projected coordinates (e.g., UTM).

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

import pytest

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.types import Easting, Meters, Northing, PixelsFloat, Unitless


def make_geotiff(gt_tuple: tuple[float, float, float, float, float, float]) -> GeoTiff:
    """Helper to create GeoTiff from GDAL-style tuple."""
    return GeoTiff(
        geotransform=GeoTransform.from_gdal_tuple(gt_tuple),
        crs="EPSG:25830",
    )


class TestPixelToGeoNorthUp:
    """Test pixel_to_geo for north-up rasters (no rotation)."""

    def test_origin_returns_geotransform_origin(self):
        """At pixel (0, 0), should return origin coordinates."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        easting, northing = geotiff.pixel_to_geo(PixelsFloat(0), PixelsFloat(0))

        assert easting == pytest.approx(737575.05, abs=0.01)
        assert northing == pytest.approx(4391595.45, abs=0.01)

    def test_offset_pixel(self):
        """Test transform at offset pixel (10, 20)."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # Expected: easting = 737575.05 + 10*0.15 = 737576.55
        # Expected: northing = 4391595.45 + 20*(-0.15) = 4391592.45
        easting, northing = geotiff.pixel_to_geo(PixelsFloat(10), PixelsFloat(20))

        assert easting == pytest.approx(737576.55, abs=0.01)
        assert northing == pytest.approx(4391592.45, abs=0.01)

    def test_large_offset(self):
        """Test transform with large pixel offset (1000, 2000)."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # Expected: easting = 737575.05 + 1000*0.15 = 737725.05
        # Expected: northing = 4391595.45 + 2000*(-0.15) = 4391295.45
        easting, northing = geotiff.pixel_to_geo(PixelsFloat(1000), PixelsFloat(2000))

        assert easting == pytest.approx(737725.05, abs=0.01)
        assert northing == pytest.approx(4391295.45, abs=0.01)

    def test_fractional_pixel_coordinates(self):
        """Test that fractional pixel coordinates interpolate correctly."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # At pixel (10.5, 20.25):
        # Expected: easting = 737575.05 + 10.5*0.15 = 737576.625
        # Expected: northing = 4391595.45 + 20.25*(-0.15) = 4391592.4125
        easting, northing = geotiff.pixel_to_geo(PixelsFloat(10.5), PixelsFloat(20.25))

        assert easting == pytest.approx(737576.625, abs=0.001)
        assert northing == pytest.approx(4391592.4125, abs=0.001)

    def test_negative_pixel_coordinates(self):
        """Test that negative pixel coordinates extrapolate correctly."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # At pixel (-10, -20):
        # Expected: easting = 737575.05 + (-10)*0.15 = 737573.55
        # Expected: northing = 4391595.45 + (-20)*(-0.15) = 4391598.45
        easting, northing = geotiff.pixel_to_geo(PixelsFloat(-10), PixelsFloat(-20))

        assert easting == pytest.approx(737573.55, abs=0.01)
        assert northing == pytest.approx(4391598.45, abs=0.01)

    def test_pixel_center_offset_convention(self):
        """Test pixel center offset convention (GDAL uses pixel corner)."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # GDAL GeoTransform references pixel CORNER (upper-left)
        # To get pixel CENTER coordinates, add 0.5 to px and py
        corner_e, corner_n = geotiff.pixel_to_geo(PixelsFloat(0), PixelsFloat(0))
        center_e, center_n = geotiff.pixel_to_geo(PixelsFloat(0.5), PixelsFloat(0.5))

        # Center should be 0.5 pixels (0.075m) offset from corner
        assert center_e == pytest.approx(corner_e + 0.075, abs=0.001)
        assert center_n == pytest.approx(corner_n - 0.075, abs=0.001)


class TestPixelToGeoRotated:
    """Test pixel_to_geo for rotated rasters."""

    def test_rotated_22_5_degrees(self):
        """Test rotated raster (22.5° clockwise) affine transform."""
        # 22.5° rotation: cos(22.5°) ≈ 0.9239, sin(22.5°) ≈ 0.3827
        # For 0.15m pixels: 0.15 * cos(22.5°) ≈ 0.1387, 0.15 * sin(22.5°) ≈ 0.0574
        gt = (500000.0, 0.1387, 0.0574, 4400000.0, 0.0574, -0.1387)
        geotiff = make_geotiff(gt)

        # At pixel (0, 0), should return origin
        e, n = geotiff.pixel_to_geo(PixelsFloat(0), PixelsFloat(0))
        assert e == pytest.approx(500000, abs=0.01)
        assert n == pytest.approx(4400000, abs=0.01)

        # At pixel (100, 0): moves along row
        e, n = geotiff.pixel_to_geo(PixelsFloat(100), PixelsFloat(0))
        assert e == pytest.approx(500013.87, abs=0.01)
        assert n == pytest.approx(4400005.74, abs=0.01)

        # At pixel (0, 100): moves along column
        e, n = geotiff.pixel_to_geo(PixelsFloat(0), PixelsFloat(100))
        assert e == pytest.approx(500005.74, abs=0.01)
        assert n == pytest.approx(4399986.13, abs=0.01)

    def test_rotated_combined_offset(self):
        """Test rotated raster with combined pixel offsets."""
        gt = (500000.0, 0.1387, 0.0574, 4400000.0, 0.0574, -0.1387)
        geotiff = make_geotiff(gt)

        # At pixel (50, 75): both X and Y offset with rotation
        e, n = geotiff.pixel_to_geo(PixelsFloat(50), PixelsFloat(75))
        assert e == pytest.approx(500011.24, abs=0.1)
        assert n == pytest.approx(4399992.47, abs=0.1)

    def test_rotated_90_degrees(self):
        """Test with 90° rotation (extreme non-north-up case)."""
        # 90° rotation: cos(90°)=0, sin(90°)=1
        gt = (500000.0, 0.0, 0.15, 4400000.0, 0.15, 0.0)
        geotiff = make_geotiff(gt)

        # At pixel (100, 0): should only affect Y coordinate
        e, n = geotiff.pixel_to_geo(PixelsFloat(100), PixelsFloat(0))
        assert e == pytest.approx(500000, abs=0.01)
        assert n == pytest.approx(4400015, abs=0.01)

        # At pixel (0, 100): should only affect X coordinate
        e, n = geotiff.pixel_to_geo(PixelsFloat(0), PixelsFloat(100))
        assert e == pytest.approx(500015, abs=0.01)
        assert n == pytest.approx(4400000, abs=0.01)


class TestGeoToPixelRoundtrip:
    """Test geo_to_pixel inverse transformation and roundtrip consistency."""

    def test_roundtrip_north_up(self):
        """Test pixel->geo->pixel roundtrip for north-up raster."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        test_pixels = [
            (0.0, 0.0),
            (10.0, 20.0),
            (100.5, 200.25),
            (1000.0, 2000.0),
        ]

        for px, py in test_pixels:
            easting, northing = geotiff.pixel_to_geo(PixelsFloat(px), PixelsFloat(py))
            px_back, py_back = geotiff.geo_to_pixel(easting, northing)

            assert px_back == pytest.approx(px, abs=0.0001), f"X roundtrip failed for ({px}, {py})"
            assert py_back == pytest.approx(py, abs=0.0001), f"Y roundtrip failed for ({px}, {py})"

    def test_roundtrip_rotated(self):
        """Test pixel->geo->pixel roundtrip for rotated raster."""
        gt = (500000.0, 0.1387, 0.0574, 4400000.0, 0.0574, -0.1387)
        geotiff = make_geotiff(gt)

        test_pixels = [
            (0.0, 0.0),
            (50.0, 75.0),
            (100.0, 100.0),
        ]

        for px, py in test_pixels:
            easting, northing = geotiff.pixel_to_geo(PixelsFloat(px), PixelsFloat(py))
            px_back, py_back = geotiff.geo_to_pixel(easting, northing)

            assert px_back == pytest.approx(px, abs=0.001), f"X roundtrip failed for ({px}, {py})"
            assert py_back == pytest.approx(py, abs=0.001), f"Y roundtrip failed for ({px}, {py})"

    def test_geo_to_pixel_known_values(self):
        """Test geo_to_pixel with known coordinate values."""
        gt = (737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15)
        geotiff = make_geotiff(gt)

        # Origin should map to pixel (0, 0)
        px, py = geotiff.geo_to_pixel(Easting(737575.05), Northing(4391595.45))
        assert px == pytest.approx(0, abs=0.0001)
        assert py == pytest.approx(0, abs=0.0001)

        # 10 pixels east, 20 pixels south
        px, py = geotiff.geo_to_pixel(Easting(737576.55), Northing(4391592.45))
        assert px == pytest.approx(10, abs=0.01)
        assert py == pytest.approx(20, abs=0.01)


class TestNorthUpEquivalence:
    """Test that 6-parameter transform matches simplified formula for north-up rasters."""

    def test_north_up_matches_simplified_formula(self):
        """Test that north-up raster (rotation=0) matches simplified formula."""
        # Old simplified formula: easting = origin_easting + px*pixel_size_x
        # Should match 6-parameter formula when GT[2]=0 and GT[4]=0
        origin_easting = 737575.05
        origin_northing = 4391595.45
        pixel_size_x = 0.15
        pixel_size_y = -0.15

        gt = (origin_easting, pixel_size_x, 0.0, origin_northing, 0.0, pixel_size_y)
        geotiff = make_geotiff(gt)

        test_pixels = [(0, 0), (10, 20), (100, 200), (1000, 2000)]

        for px, py in test_pixels:
            # Simplified formula (what someone might naively implement)
            simple_easting = origin_easting + (px * pixel_size_x)
            simple_northing = origin_northing + (py * pixel_size_y)

            # 6-parameter formula via GeoTiff
            affine_easting, affine_northing = geotiff.pixel_to_geo(PixelsFloat(px), PixelsFloat(py))

            assert affine_easting == pytest.approx(simple_easting, abs=0.001), (
                f"6-param should match simplified for north-up at ({px}, {py})"
            )
            assert affine_northing == pytest.approx(simple_northing, abs=0.001), (
                f"6-param should match simplified for north-up at ({px}, {py})"
            )


class TestGeoTransformProperties:
    """Test GeoTransform value object properties."""

    def test_is_north_up_true(self):
        """Test is_north_up returns True when rotation is zero."""
        gt = GeoTransform(
            origin_easting=Easting(500000.0),
            pixel_width=Meters(0.15),
            row_rotation=Unitless(0.0),
            origin_northing=Northing(4400000.0),
            col_rotation=Unitless(0.0),
            pixel_height=Meters(-0.15),
        )
        assert gt.is_north_up is True

    def test_is_north_up_false(self):
        """Test is_north_up returns False when rotation is non-zero."""
        gt = GeoTransform(
            origin_easting=Easting(500000.0),
            pixel_width=Meters(0.1387),
            row_rotation=Unitless(0.0574),
            origin_northing=Northing(4400000.0),
            col_rotation=Unitless(0.0574),
            pixel_height=Meters(-0.1387),
        )
        assert gt.is_north_up is False

    def test_to_gdal_tuple_roundtrip(self):
        """Test from_gdal_tuple and to_gdal_tuple are inverses."""
        original = (737575.05, 0.15, 0.01, 4391595.45, 0.01, -0.15)
        gt = GeoTransform.from_gdal_tuple(original)
        result = gt.to_gdal_tuple()

        for i in range(6):
            assert result[i] == pytest.approx(original[i], abs=0.0001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
