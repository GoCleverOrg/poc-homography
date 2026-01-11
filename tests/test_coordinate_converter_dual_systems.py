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
)

# Test data: Valencia area coordinates (EPSG:25830)
VALENCIA_REF_LAT = 39.640472
VALENCIA_REF_LON = -0.230194
VALENCIA_REF_UTM_E = 737612.15
VALENCIA_REF_UTM_N = 4391527.36


# NOTE: TestEquirectangularProjection removed - standalone gps_to_local_xy/local_xy_to_gps
# functions were deleted during MapPoint migration. Use GCPCoordinateConverter instead.


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestUTMProjection:
    """Tests for UTM projection using pyproj."""

    def test_utm_converter_initialization(self):
        """UTMConverter should initialize with default CRS."""
        converter = UTMConverter()
        assert converter.utm_crs == DEFAULT_UTM_CRS

    def test_utm_converter_with_reference_gps(self):
        """Create converter with GPS reference using factory method."""
        converter = UTMConverter.with_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        # Should be reasonable UTM values for Valencia
        assert converter.reference_easting is not None
        assert converter.reference_northing is not None
        assert 700000 < converter.reference_easting < 800000
        assert 4300000 < converter.reference_northing < 4500000

    def test_utm_converter_with_reference_utm(self):
        """Create converter with UTM reference using factory method."""
        converter = UTMConverter.with_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        assert converter._ref_easting == VALENCIA_REF_UTM_E
        assert converter._ref_northing == VALENCIA_REF_UTM_N

    def test_utm_gps_to_local_identity(self):
        """Reference point should map to origin."""
        converter = UTMConverter.with_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        x, y = converter.gps_to_local_xy(VALENCIA_REF_LAT, VALENCIA_REF_LON)
        assert abs(x) < 0.001
        assert abs(y) < 0.001

    def test_utm_to_local_xy_direct(self):
        """UTM coordinates should convert directly to local."""
        converter = UTMConverter.with_reference_utm(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

        # Test point 100m east and 50m north
        test_e = VALENCIA_REF_UTM_E + 100
        test_n = VALENCIA_REF_UTM_N + 50

        x, y = converter.utm_to_local_xy(test_e, test_n)
        assert abs(x - 100) < 0.001
        assert abs(y - 50) < 0.001

    def test_utm_round_trip(self):
        """GPS -> UTM -> local -> UTM -> GPS should return original."""
        converter = UTMConverter.with_reference(VALENCIA_REF_LAT, VALENCIA_REF_LON)

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
        converter = GCPCoordinateConverter.with_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)
        assert converter.using_utm is True

    def test_converter_with_gps_reference(self):
        """Create converter using GPS coordinates factory method."""
        converter = GCPCoordinateConverter.with_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        assert converter._ref_lat == VALENCIA_REF_LAT
        assert converter._ref_lon == VALENCIA_REF_LON
        assert converter._ref_easting is not None

    def test_converter_with_utm_reference(self):
        """Create converter using UTM coordinates factory method."""
        converter = GCPCoordinateConverter.with_reference_utm(
            VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N
        )

        assert converter._ref_easting == VALENCIA_REF_UTM_E
        assert converter._ref_northing == VALENCIA_REF_UTM_N
        assert converter._ref_lat is not None

    def test_convert_point_with_gps_only(self):
        """Convert a point that only has GPS coordinates."""
        converter = GCPCoordinateConverter.with_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        point = {"latitude": VALENCIA_REF_LAT + 0.0001, "longitude": VALENCIA_REF_LON + 0.0001}
        x, y = converter.convert_point(point)

        # Should produce reasonable small displacements (~8-11m)
        assert 5 < abs(x) < 15
        assert 5 < abs(y) < 15

    def test_convert_point_with_utm_only(self):
        """Convert a point that only has UTM coordinates."""
        converter = GCPCoordinateConverter.with_reference_utm(
            VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N
        )

        point = {"utm_easting": VALENCIA_REF_UTM_E + 100, "utm_northing": VALENCIA_REF_UTM_N + 50}
        x, y = converter.convert_point(point)

        assert abs(x - 100) < 0.01
        assert abs(y - 50) < 0.01

    def test_convert_point_with_both_prefers_utm(self):
        """When both GPS and UTM are present, prefer UTM."""
        converter = GCPCoordinateConverter.with_reference_utm(
            VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N
        )

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


# NOTE: TestGPSvsUTMAccuracy removed - it compared deleted standalone gps_to_local_xy
# function against UTM. The standalone equirectangular functions no longer exist.


# NOTE: TestEquirectangularFallback removed - standalone equirectangular projection
# functions were deleted. GCPCoordinateConverter now requires pyproj for GPS conversion.


class TestGCPPointStructures:
    """Test handling of various GCP point dictionary structures."""

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_full_gcp_point_structure(self):
        """Test with a complete GCP point structure as used in production."""
        # Use factory method to create converter with UTM reference
        converter = GCPCoordinateConverter.with_reference_utm(
            VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N
        )

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

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_point_missing_both_coordinates_raises(self):
        """Point with neither GPS nor UTM should raise ValueError."""
        converter = GCPCoordinateConverter.with_reference_gps(VALENCIA_REF_LAT, VALENCIA_REF_LON)

        invalid_point = {"some_other_field": 123}

        with pytest.raises(ValueError):
            converter.convert_point(invalid_point)


class TestCoordinateConverterEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_reference_not_set_raises(self):
        """Operations without setting reference should raise ValueError."""
        converter = GCPCoordinateConverter()

        with pytest.raises(ValueError, match="Reference point not set"):
            converter.gps_to_local(VALENCIA_REF_LAT, VALENCIA_REF_LON)

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
    def test_utm_reference_not_set_raises(self):
        """UTM operations without reference should raise ValueError."""
        converter = UTMConverter()

        with pytest.raises(ValueError, match="Reference point not set"):
            converter.utm_to_local_xy(VALENCIA_REF_UTM_E, VALENCIA_REF_UTM_N)

    # NOTE: test_polar_region_raises removed - tested deleted standalone gps_to_local_xy function


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
