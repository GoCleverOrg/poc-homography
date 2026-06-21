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


class TestGeoToPixel:
    """Test the GeoTiff.geo_to_pixel inverse affine (world -> map-pixel)."""

    def test_origin_maps_to_zero_pixel(self):
        """The geotransform origin inverts back to pixel (0, 0)."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        px, py = geotiff.geo_to_pixel(Easting(737575.05), Northing(4391595.45))

        assert px == pytest.approx(0.0, abs=1e-9)
        assert py == pytest.approx(0.0, abs=1e-9)

    def test_roundtrip_north_up(self):
        """pixel -> geo -> pixel round-trips under 1e-3 px for a north-up raster."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])

        for px, py in [(0, 0), (10.5, 20.25), (1000, 2000), (-10, -20), (1680.0, 915.0)]:
            easting, northing = geotiff.pixel_to_geo(px, py)
            rx, ry = geotiff.geo_to_pixel(easting, northing)

            assert rx == pytest.approx(px, abs=1e-3)
            assert ry == pytest.approx(py, abs=1e-3)

    def test_roundtrip_rotated(self):
        """pixel -> geo -> pixel round-trips under 1e-3 px for a rotated raster."""
        geotiff = _make_geotiff([500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387])

        for px, py in [(0, 0), (100, 0), (0, 100), (50, 75), (1234.5, 678.25)]:
            easting, northing = geotiff.pixel_to_geo(px, py)
            rx, ry = geotiff.geo_to_pixel(easting, northing)

            assert rx == pytest.approx(px, abs=1e-3)
            assert ry == pytest.approx(py, abs=1e-3)

    def test_degenerate_geotransform_raises(self):
        """A zero-determinant (non-invertible) geotransform raises ValueError."""
        geotiff = _make_geotiff([500000, 0.0, 0.0, 4400000, 0.0, -0.15])

        with pytest.raises(ValueError, match="[Dd]egenerate"):
            geotiff.geo_to_pixel(Easting(500000), Northing(4400000))

    def test_return_types(self):
        """geo_to_pixel returns a float tuple of length 2."""
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15])
        result = geotiff.geo_to_pixel(Easting(737576.55), Northing(4391592.45))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestPixelToLatLon:
    """Test the pixel -> WGS84 lat/lon reprojection (issue #33)."""

    def test_valte_origin_matches_known_gps(self):
        """The Valte GeoTIFF origin reprojects to its documented GPS location.

        The Valte camera's GeoTransform origin (EPSG:25830 UTM zone 30N) sits at
        roughly 39.641 N, -0.231 E — the area the project targets.
        """
        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15], crs="EPSG:25830")

        lat, lon = geotiff.pixel_to_latlon(0, 0)

        assert lat == pytest.approx(39.6412, abs=1e-3)
        assert lon == pytest.approx(-0.2314, abs=1e-3)

    def test_roundtrip_consistency_with_pixel_to_geo(self):
        """lat/lon is the WGS84 reprojection of the same easting/northing."""
        from pyproj import Transformer

        geotiff = _make_geotiff([737575.05, 0.15, 0, 4391595.45, 0, -0.15], crs="EPSG:25830")
        easting, northing = geotiff.pixel_to_geo(120, 340)
        lat, lon = geotiff.pixel_to_latlon(120, 340)

        transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
        exp_lon, exp_lat = transformer.transform(float(easting), float(northing))

        assert lat == pytest.approx(exp_lat, abs=1e-9)
        assert lon == pytest.approx(exp_lon, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
