#!/usr/bin/env python3
"""
Comprehensive tests for GPS and UTM coordinate conversion.

This test suite validates that both GPS (equirectangular) and UTM (pyproj)
coordinate systems work correctly in the GCP processing pipeline.

Tests cover:
1. GPS-only coordinate conversion
2. UTM-only coordinate conversion
3. Mixed GPS and UTM coordinate handling
4. Round-trip conversion accuracy
5. GCPCoordinateConverter unified interface
6. Comparison of GPS vs UTM accuracy
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.coordinate_converter import (
    DEFAULT_UTM_CRS,
    PYPROJ_AVAILABLE,
    GCPCoordinateConverter,
    UTMConverter,
    gps_to_local_xy,
    local_xy_to_gps,
)

# Test data: Valencia area coordinates (EPSG:25830)
VALENCIA_REF_LAT = 39.640472
VALENCIA_REF_LON = -0.230194
VALENCIA_REF_UTM_E = 737612.15
VALENCIA_REF_UTM_N = 4391527.36


class TestEquirectangularProjection:
    """Tests for the basic equirectangular (GPS) projection."""

    def test_gps_to_local_xy_identity(self):
        """Reference point should map to origin."""
        x, y = gps_to_local_xy(
            VALENCIA_REF_LAT, VALENCIA_REF_LON, VALENCIA_REF_LAT, VALENCIA_REF_LON
        )
        assert abs(x) < 0.001
        assert abs(y) < 0.001

    def test_gps_to_local_xy_north_displacement(self):
        """Point north of reference should have positive Y."""
        # 1 degree north (about 111km)
        x, y = gps_to_local_xy(
            VALENCIA_REF_LAT, VALENCIA_REF_LON, VALENCIA_REF_LAT + 1.0, VALENCIA_REF_LON
        )
        assert abs(x) < 10  # Small X drift is acceptable
        assert 110000 < y < 112000  # Approximately 111 km

    def test_gps_to_local_xy_east_displacement(self):
        """Point east of reference should have positive X."""
        x, y = gps_to_local_xy(
            VALENCIA_REF_LAT, VALENCIA_REF_LON, VALENCIA_REF_LAT, VALENCIA_REF_LON + 0.001
        )
        assert x > 0
        assert abs(y) < 1

    def test_gps_round_trip(self):
        """GPS -> local -> GPS should return original coordinates."""
        test_lat = VALENCIA_REF_LAT + 0.001
        test_lon = VALENCIA_REF_LON + 0.001

        x, y = gps_to_local_xy(VALENCIA_REF_LAT, VALENCIA_REF_LON, test_lat, test_lon)
        recovered_lat, recovered_lon = local_xy_to_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON, x, y)

        # Should match to about 0.0001 degrees (~10m)
        assert abs(recovered_lat - test_lat) < 0.0001
        assert abs(recovered_lon - test_lon) < 0.0001


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestUTMProjection:
    """Tests for UTM projection using pyproj."""

    def test_utm_converter_initialization(self):
        """UTMConverter should initialize with default CRS."""
        converter = UTMConverter()
        assert converter.utm_crs == DEFAULT_UTM_CRS

    def test_utm_converter_set_reference_gps(self):
        """Set reference using GPS coordinates."""
        converter = UTMConverter()
        easting, northing = converter.set_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        # Should be reasonable UTM values for Valencia
        assert 700000 < easting < 800000
        assert 4300000 < northing < 4500000

    def test_utm_converter_set_reference_utm(self):
        """Set reference using UTM coordinates directly."""
        converter = UTMConverter()
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        assert converter._ref_easting == VALENCIA_REF_UTM_E
        assert converter._ref_northing == VALENCIA_REF_UTM_N

    def test_utm_gps_to_local_identity(self):
        """Reference point should map to origin."""
        converter = UTMConverter()
        converter.set_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        x, y = converter.gps_to_local_xy(VALENCIA_REF_LAT, VALENCIA_REF_LON)
        assert abs(x) < 0.001
        assert abs(y) < 0.001

    def test_utm_to_local_xy_direct(self):
        """UTM coordinates should convert directly to local."""
        converter = UTMConverter()
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        # Test point 100m east and 50m north
        test_e = VALENCIA_REF_UTM_E + 100
        test_n = VALENCIA_REF_UTM_N + 50

        x, y = converter.utm_to_local_xy(test_e, test_n)
        assert abs(x - 100) < 0.001
        assert abs(y - 50) < 0.001

    def test_utm_round_trip(self):
        """GPS -> UTM -> local -> UTM -> GPS should return original."""
        converter = UTMConverter()
        converter.set_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        test_lat = VALENCIA_REF_LAT + 0.001
        test_lon = VALENCIA_REF_LON + 0.001

        # GPS -> local
        x, y = converter.gps_to_local_xy(test_lat, test_lon)
        # local -> GPS
        recovered_lat, recovered_lon = converter.local_xy_to_gps(x, y)

        # Should match to high precision
        assert abs(recovered_lat - test_lat) < 1e-7
        assert abs(recovered_lon - test_lon) < 1e-7


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestGCPCoordinateConverter:
    """Tests for the unified GCPCoordinateConverter interface."""

    def test_converter_prefers_utm(self):
        """By default, should prefer UTM when available."""
        converter = GCPCoordinateConverter()
        assert converter.using_utm is True
        assert "UTM" in converter.method_name

    def test_converter_with_gps_reference(self):
        """Set reference using GPS coordinates."""
        converter = GCPCoordinateConverter()
        converter.set_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        assert converter._ref_lat == VALENCIA_REF_LAT
        assert converter._ref_lon == VALENCIA_REF_LON
        assert converter._ref_easting is not None

    def test_converter_with_utm_reference(self):
        """Set reference using UTM coordinates."""
        converter = GCPCoordinateConverter()
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        assert converter._ref_easting == VALENCIA_REF_UTM_E
        assert converter._ref_northing == VALENCIA_REF_UTM_N
        assert converter._ref_lat is not None

    def test_convert_point_with_gps_only(self):
        """Convert a point that only has GPS coordinates."""
        converter = GCPCoordinateConverter()
        converter.set_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        point = {"latitude": VALENCIA_REF_LAT + 0.0001, "longitude": VALENCIA_REF_LON + 0.0001}
        x, y = converter.convert_point(point)

        # Should produce reasonable small displacements (~8-11m)
        assert 5 < abs(x) < 15
        assert 5 < abs(y) < 15

    def test_convert_point_with_utm_only(self):
        """Convert a point that only has UTM coordinates."""
        converter = GCPCoordinateConverter()
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        point = {"utm_easting": VALENCIA_REF_UTM_E + 100, "utm_northing": VALENCIA_REF_UTM_N + 50}
        x, y = converter.convert_point(point)

        assert abs(x - 100) < 0.01
        assert abs(y - 50) < 0.01

    def test_convert_point_with_both_prefers_utm(self):
        """When both GPS and UTM are present, prefer UTM."""
        converter = GCPCoordinateConverter()
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        # Point with both coordinate types (UTM is accurate, GPS is slightly off)
        point = {
            "latitude": VALENCIA_REF_LAT + 0.001,  # Different from UTM
            "longitude": VALENCIA_REF_LON + 0.001,
            "utm_easting": VALENCIA_REF_UTM_E + 100,  # Exact offset
            "utm_northing": VALENCIA_REF_UTM_N + 50,
        }
        x, y = converter.convert_point(point)

        # Should use UTM, so exact values
        assert abs(x - 100) < 0.01
        assert abs(y - 50) < 0.01


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestGPSvsUTMAccuracy:
    """Compare accuracy between GPS (equirectangular) and UTM projections."""

    def test_small_distance_similar_results(self):
        """For small distances (<100m), both methods should give similar results."""
        utm_converter = UTMConverter()
        utm_converter.set_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        # Small displacement: ~50m
        test_lat = VALENCIA_REF_LAT + 0.00045  # ~50m north
        test_lon = VALENCIA_REF_LON + 0.00060  # ~50m east

        x_gps, y_gps = gps_to_local_xy(VALENCIA_REF_LAT, VALENCIA_REF_LON, test_lat, test_lon)
        x_utm, y_utm = utm_converter.gps_to_local_xy(test_lat, test_lon)

        # Equirectangular has ~2-3% error even at small distances due to projection
        # approximation. Should be within 5% of each other for small distances
        assert abs(x_gps - x_utm) / max(abs(x_utm), 1) < 0.05
        assert abs(y_gps - y_utm) / max(abs(y_utm), 1) < 0.05

    def test_utm_more_accurate_at_larger_distances(self):
        """For larger distances, UTM should be more accurate than equirectangular."""
        utm_converter = UTMConverter()
        utm_converter.set_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        # Larger displacement: ~1km
        test_lat = VALENCIA_REF_LAT + 0.009  # ~1km north
        test_lon = VALENCIA_REF_LON + 0.012  # ~1km east

        x_gps, y_gps = gps_to_local_xy(VALENCIA_REF_LAT, VALENCIA_REF_LON, test_lat, test_lon)
        x_utm, y_utm = utm_converter.gps_to_local_xy(test_lat, test_lon)

        # GPS will have some scale error, UTM is reference
        # Calculate percentage difference
        x_diff_pct = abs(x_gps - x_utm) / abs(x_utm) * 100
        y_diff_pct = abs(y_gps - y_utm) / abs(y_utm) * 100

        # At 1km scale, expect 1-5% error in equirectangular
        print(f"X difference: {x_diff_pct:.2f}%, Y difference: {y_diff_pct:.2f}%")

        # These should not be exactly zero (equirectangular has inherent error)
        assert x_diff_pct > 0 or y_diff_pct > 0
        # But should be less than 10%
        assert x_diff_pct < 10
        assert y_diff_pct < 10


class TestEquirectangularFallback:
    """Test equirectangular fallback when pyproj is not available."""

    def test_gcp_converter_falls_back_to_equirectangular(self):
        """GCPCoordinateConverter should work even without UTM."""
        # Create converter with prefer_utm=False to simulate no pyproj
        converter = GCPCoordinateConverter(prefer_utm=False)
        converter.set_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        assert converter.using_utm is False
        assert "Equirectangular" in converter.method_name

        point = {"latitude": VALENCIA_REF_LAT + 0.0001, "longitude": VALENCIA_REF_LON + 0.0001}
        x, y = converter.convert_point(point)

        # Should still produce reasonable results
        assert abs(x) > 0
        assert abs(y) > 0


class TestGCPPointStructures:
    """Test handling of various GCP point dictionary structures."""

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_full_gcp_point_structure(self):
        """Test with a complete GCP point structure as used in production."""
        converter = GCPCoordinateConverter()
        # Set reference using UTM to ensure consistency with test UTM values
        converter.set_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        gcp_point = {
            "latitude": VALENCIA_REF_LAT + 0.0001,
            "longitude": VALENCIA_REF_LON + 0.0001,
            "utm_easting": VALENCIA_REF_UTM_E + 10,
            "utm_northing": VALENCIA_REF_UTM_N + 10,
            "utm_crs": "EPSG:25830",
        }

        x, y = converter.convert_point(gcp_point)

        # With UTM present and UTM reference set, should use UTM values exactly
        if converter.using_utm:
            assert abs(x - 10) < 0.01  # Exact 10m
            assert abs(y - 10) < 0.01

    def test_point_missing_both_coordinates_raises(self):
        """Point with neither GPS nor UTM should raise ValueError."""
        converter = GCPCoordinateConverter(prefer_utm=False)
        converter.set_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        invalid_point = {"some_other_field": 123}

        with pytest.raises(ValueError):
            converter.convert_point(invalid_point)


class TestCoordinateConverterEdgeCases:
    """Test edge cases and error handling."""

    def test_reference_not_set_raises(self):
        """Operations without setting reference should raise ValueError."""
        converter = GCPCoordinateConverter(prefer_utm=False)

        with pytest.raises(ValueError):
            converter.gps_to_local(VALENCIA_REF_LAT, VALENCIA_REF_LON)

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_utm_reference_not_set_raises(self):
        """UTM operations without reference should raise ValueError."""
        converter = UTMConverter()

        with pytest.raises(ValueError):
            converter.utm_to_local_xy(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

    def test_polar_region_raises(self):
        """Equirectangular should reject polar coordinates."""
        with pytest.raises(ValueError):
            gps_to_local_xy(89.0, 0, 89.1, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
