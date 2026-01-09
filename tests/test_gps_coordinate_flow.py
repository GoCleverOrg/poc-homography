#!/usr/bin/env python3
"""
Unit tests for GPS coordinate flow from config through to WorldPoint projection.

Tests verify:
1. GPS parsing from DMS format to decimal degrees
2. Setting GPS position on IntrinsicExtrinsicHomography provider
3. Calling project_point() and getting valid GPS coordinates in WorldPoint
4. GPS coordinate transformation accuracy (round-trip)
"""

import math
import os
import sys

import numpy as np

# pytest is optional - only needed when running with pytest runner
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.gps_distance_calculator import (
    dd_to_dms,
    dms_to_dd,
    gps_to_local_xy,
    haversine_distance,
    local_xy_to_gps,
)
from poc_homography.homography_interface import WorldPoint
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


class TestDMSParsing:
    """Test DMS (Degrees, Minutes, Seconds) to Decimal Degrees conversion."""

    def test_dms_parsing_latitude_north(self):
        """Test DMS parsing for latitude with North direction."""
        # Example from camera_config.py: Valte camera
        dms_str = "39°38'25.7\"N"
        dd = dms_to_dd(dms_str)

        # Manual calculation: 39 + 38/60 + 25.7/3600
        expected = 39 + 38 / 60.0 + 25.7 / 3600.0

        assert abs(dd - expected) < 1e-6, f"Expected {expected}, got {dd}"
        assert dd > 0, "North latitude should be positive"
        assert 39 < dd < 40, f"Latitude {dd} out of expected range"

    def test_dms_parsing_latitude_south(self):
        """Test DMS parsing for latitude with South direction."""
        dms_str = "33°55'12.3\"S"
        dd = dms_to_dd(dms_str)

        # South should be negative
        expected = -(33 + 55 / 60.0 + 12.3 / 3600.0)

        assert abs(dd - expected) < 1e-6, f"Expected {expected}, got {dd}"
        assert dd < 0, "South latitude should be negative"

    def test_dms_parsing_longitude_west(self):
        """Test DMS parsing for longitude with West direction."""
        # Example from camera_config.py: Valte camera
        dms_str = "0°13'48.7\"W"
        dd = dms_to_dd(dms_str)

        # Manual calculation: -(0 + 13/60 + 48.7/3600)
        expected = -(0 + 13 / 60.0 + 48.7 / 3600.0)

        assert abs(dd - expected) < 1e-6, f"Expected {expected}, got {dd}"
        assert dd < 0, "West longitude should be negative"

    def test_dms_parsing_longitude_east(self):
        """Test DMS parsing for longitude with East direction."""
        # Example from camera_config.py: Setram camera
        dms_str = "2°08'31.3\"E"
        dd = dms_to_dd(dms_str)

        # Manual calculation: 2 + 8/60 + 31.3/3600
        expected = 2 + 8 / 60.0 + 31.3 / 3600.0

        assert abs(dd - expected) < 1e-6, f"Expected {expected}, got {dd}"
        assert dd > 0, "East longitude should be positive"
        assert 2 < dd < 3, f"Longitude {dd} out of expected range"

    def test_dms_parsing_zero_degrees(self):
        """Test DMS parsing for coordinates at zero degrees."""
        dms_str_lat = "0°0'0\"N"
        dms_str_lon = "0°0'0\"E"

        lat_dd = dms_to_dd(dms_str_lat)
        lon_dd = dms_to_dd(dms_str_lon)

        assert abs(lat_dd) < 1e-6, f"Expected 0, got {lat_dd}"
        assert abs(lon_dd) < 1e-6, f"Expected 0, got {lon_dd}"

    def test_dms_roundtrip_conversion(self):
        """Test that DMS -> DD -> DMS roundtrip preserves values."""
        original_lat_dms = "39°38'25.7\"N"
        original_lon_dms = "0°13'48.7\"W"

        # Convert to decimal degrees
        lat_dd = dms_to_dd(original_lat_dms)
        lon_dd = dms_to_dd(original_lon_dms)

        # Convert back to DMS
        reconstructed_lat_dms = dd_to_dms(lat_dd, is_latitude=True)
        reconstructed_lon_dms = dd_to_dms(lon_dd, is_latitude=False)

        # Convert reconstructed back to DD to verify accuracy
        lat_dd_2 = dms_to_dd(reconstructed_lat_dms)
        lon_dd_2 = dms_to_dd(reconstructed_lon_dms)

        # Should be within 0.1 arcseconds (due to rounding in dd_to_dms)
        # 0.1 arcseconds ≈ 0.000028 degrees ≈ 3 meters at equator
        assert abs(lat_dd - lat_dd_2) < 0.0001, f"Latitude roundtrip failed: {lat_dd} != {lat_dd_2}"
        assert abs(lon_dd - lon_dd_2) < 0.0001, (
            f"Longitude roundtrip failed: {lon_dd} != {lon_dd_2}"
        )


class TestGPSPositionSetting:
    """Test setting GPS position on IntrinsicExtrinsicHomography provider."""

    def test_set_camera_gps_position_valid(self):
        """Test setting valid GPS coordinates."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Use Valte camera coordinates
        lat_dms = "39°38'25.7\"N"
        lon_dms = "0°13'48.7\"W"

        lat_dd = dms_to_dd(lat_dms)
        lon_dd = dms_to_dd(lon_dms)

        # Should not raise exception
        homography.set_camera_gps_position(lat_dd, lon_dd)

        # Verify internal state
        assert homography._camera_gps_lat == lat_dd
        assert homography._camera_gps_lon == lon_dd

    def test_set_camera_gps_position_invalid_latitude(self):
        """Test that invalid latitude raises ValueError."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Latitude out of range [-90, 90]
        if HAS_PYTEST:
            with pytest.raises(ValueError, match="Latitude must be in range"):
                homography.set_camera_gps_position(91.0, 0.0)

            with pytest.raises(ValueError, match="Latitude must be in range"):
                homography.set_camera_gps_position(-91.0, 0.0)
        else:
            try:
                homography.set_camera_gps_position(91.0, 0.0)
                assert False, "Should have raised ValueError for latitude > 90"
            except ValueError as e:
                assert "Latitude must be in range" in str(e)

            try:
                homography.set_camera_gps_position(-91.0, 0.0)
                assert False, "Should have raised ValueError for latitude < -90"
            except ValueError as e:
                assert "Latitude must be in range" in str(e)

    def test_set_camera_gps_position_invalid_longitude(self):
        """Test that invalid longitude raises ValueError."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Longitude out of range [-180, 180]
        if HAS_PYTEST:
            with pytest.raises(ValueError, match="Longitude must be in range"):
                homography.set_camera_gps_position(0.0, 181.0)

            with pytest.raises(ValueError, match="Longitude must be in range"):
                homography.set_camera_gps_position(0.0, -181.0)
        else:
            try:
                homography.set_camera_gps_position(0.0, 181.0)
                assert False, "Should have raised ValueError for longitude > 180"
            except ValueError as e:
                assert "Longitude must be in range" in str(e)

            try:
                homography.set_camera_gps_position(0.0, -181.0)
                assert False, "Should have raised ValueError for longitude < -180"
            except ValueError as e:
                assert "Longitude must be in range" in str(e)

    def test_project_point_without_gps_position_fails(self):
        """Test that project_point() fails if GPS position not set."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Setup homography without GPS position
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Should raise RuntimeError since GPS position not set
        if HAS_PYTEST:
            with pytest.raises(RuntimeError, match="Camera GPS position must be set"):
                homography.project_point((1280, 1200))
        else:
            try:
                homography.project_point((1280, 1200))
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Camera GPS position must be set" in str(e)


class TestProjectPointGPSFlow:
    """Test that project_point() returns valid WorldPoint with GPS coordinates."""

    def test_project_point_returns_world_point(self):
        """Test that project_point() returns a WorldPoint with GPS coordinates."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Set camera GPS position (Valte camera)
        lat_dms = "39°38'25.7\"N"
        lon_dms = "0°13'48.7\"W"
        lat_dd = dms_to_dd(lat_dms)
        lon_dd = dms_to_dd(lon_dms)
        homography.set_camera_gps_position(lat_dd, lon_dd)

        # Compute homography
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        result = homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Project a point near bottom center (should be close to camera)
        image_point = (1280, 1200)  # Center horizontally, near bottom
        world_point = homography.project_point(image_point)

        # Verify WorldPoint structure
        assert isinstance(world_point, WorldPoint)
        assert hasattr(world_point, "latitude")
        assert hasattr(world_point, "longitude")
        assert hasattr(world_point, "confidence")

        # Verify GPS coordinates are valid
        assert -90 <= world_point.latitude <= 90
        assert -180 <= world_point.longitude <= 180
        assert 0.0 <= world_point.confidence <= 1.0

        # Point should be near camera GPS location
        lat_diff = abs(world_point.latitude - lat_dd)
        lon_diff = abs(world_point.longitude - lon_dd)

        # Should be within ~100 meters (rough check)
        # At this latitude, 1 degree ≈ 111km
        # So 0.001 degrees ≈ 111 meters
        assert lat_diff < 0.001, f"Latitude difference too large: {lat_diff}"
        assert lon_diff < 0.001, f"Longitude difference too large: {lon_diff}"

    def test_project_point_different_locations(self):
        """Test projecting multiple image points to different GPS locations."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Set camera GPS position
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Compute homography with camera at 10m height, looking down at 45°
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 10.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,  # Looking down at 45°
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Project points at different image locations
        # Bottom center should be closest to camera
        point_close = (1280, 1400)
        # Top center should be farther
        point_far = (1280, 400)

        world_close = homography.project_point(point_close)
        world_far = homography.project_point(point_far)

        # Calculate distances from camera
        dist_close = haversine_distance(
            cam_lat, cam_lon, world_close.latitude, world_close.longitude
        )
        dist_far = haversine_distance(cam_lat, cam_lon, world_far.latitude, world_far.longitude)

        # Far point should be farther from camera than close point
        assert dist_far > dist_close, (
            f"Expected far point ({dist_far}m) > close point ({dist_close}m)"
        )

        # Both should be reasonable distances (< 100m for this camera setup)
        assert dist_close < 100, f"Close point too far: {dist_close}m"
        assert dist_far < 100, f"Far point too far: {dist_far}m"

    def test_project_point_confidence_varies(self):
        """Test that confidence varies based on point location in image."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Set camera GPS position
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Compute homography
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Project points at different locations
        center_point = (1280, 720)  # Image center
        edge_point = (2500, 100)  # Near top-right corner

        world_center = homography.project_point(center_point)
        world_edge = homography.project_point(edge_point)

        # Center point should have higher confidence than edge point
        # (based on _calculate_point_confidence implementation)
        assert world_center.confidence > world_edge.confidence, (
            f"Center confidence ({world_center.confidence}) should be > "
            f"edge confidence ({world_edge.confidence})"
        )


class TestGPSCoordinateAccuracy:
    """Test GPS coordinate transformation accuracy and round-trip conversions."""

    def test_gps_roundtrip_at_origin(self):
        """Test round-trip GPS conversion for point at camera origin."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Camera position
        cam_lat = 39.640472  # Decimal degrees
        cam_lon = -0.230194
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Test local coordinates -> GPS -> local coordinates
        x_local, y_local = 0.0, 0.0  # At camera position

        # Convert to GPS
        lat, lon = local_xy_to_gps(cam_lat, cam_lon, x_local, y_local)

        # Should be exactly at camera position
        assert abs(lat - cam_lat) < 1e-9, f"Latitude mismatch: {lat} != {cam_lat}"
        assert abs(lon - cam_lon) < 1e-9, f"Longitude mismatch: {lon} != {cam_lon}"

        # Convert back to local
        x_back, y_back = gps_to_local_xy(cam_lat, cam_lon, lat, lon)

        # Should round-trip to original values
        assert abs(x_back - x_local) < 0.01, f"X roundtrip failed: {x_back} != {x_local}"
        assert abs(y_back - y_local) < 0.01, f"Y roundtrip failed: {y_back} != {y_local}"

    def test_gps_roundtrip_at_offset(self):
        """Test round-trip GPS conversion for point offset from camera."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Camera position
        cam_lat = 39.640472
        cam_lon = -0.230194
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Test point 10m East, 5m North of camera
        x_local, y_local = 10.0, 5.0

        # Convert to GPS
        lat, lon = local_xy_to_gps(cam_lat, cam_lon, x_local, y_local)

        # Convert back to local
        x_back, y_back = gps_to_local_xy(cam_lat, cam_lon, lat, lon)

        # Should round-trip to original values (within cm accuracy)
        assert abs(x_back - x_local) < 0.01, (
            f"X roundtrip failed: {x_back} != {x_local} (error: {abs(x_back - x_local)}m)"
        )
        assert abs(y_back - y_local) < 0.01, (
            f"Y roundtrip failed: {y_back} != {y_local} (error: {abs(y_back - y_local)}m)"
        )

    def test_gps_projection_distance_accuracy(self):
        """Test that projected GPS distances match homography distances."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Camera position (Valte)
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Compute homography
        K = homography.get_intrinsics(zoom_factor=1.0)
        camera_height = 11.3  # Test height value (Valencia region test data)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, camera_height]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Project an image point
        image_point = (1280, 1300)
        world_point = homography.project_point(image_point)

        # Get local coordinates for same point
        map_coord = homography.project_point_to_map(image_point)

        # Calculate distance using local coordinates
        local_distance = math.sqrt(map_coord.x**2 + map_coord.y**2)

        # Calculate distance using GPS coordinates
        gps_distance = haversine_distance(
            cam_lat, cam_lon, world_point.latitude, world_point.longitude
        )

        # Distances should match within 1% (accounting for GPS approximation)
        relative_error = abs(local_distance - gps_distance) / max(local_distance, 0.1)
        assert relative_error < 0.01, (
            f"Distance mismatch: local={local_distance:.2f}m, "
            f"gps={gps_distance:.2f}m, error={relative_error * 100:.1f}%"
        )

    def test_pan_rotation_gps_coordinates(self):
        """Test that pan rotation correctly affects GPS coordinates."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Camera position
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        K = homography.get_intrinsics(zoom_factor=1.0)

        # Test same image point with different pan angles
        image_point = (1280, 1200)  # Center bottom

        # Pan 0°
        reference_0 = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference_0)
        world_0 = homography.project_point(image_point)

        # Pan 90°
        reference_90 = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 90.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference_90)
        world_90 = homography.project_point(image_point)

        # Pan 180°
        reference_180 = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 180.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference_180)
        world_180 = homography.project_point(image_point)

        # Verify all three points have valid GPS coordinates
        assert world_0.confidence > 0, "Pan 0° should have valid confidence"
        assert world_90.confidence > 0, "Pan 90° should have valid confidence"
        assert world_180.confidence > 0, "Pan 180° should have valid confidence"

        # All coordinates should be in valid range
        assert -90 <= world_0.latitude <= 90
        assert -180 <= world_0.longitude <= 180
        assert -90 <= world_90.latitude <= 90
        assert -180 <= world_90.longitude <= 180
        assert -90 <= world_180.latitude <= 90
        assert -180 <= world_180.longitude <= 180

        # Verify that pan rotation changes GPS coordinates significantly
        # Different pan angles should produce different GPS coordinates
        dist_0_to_90 = haversine_distance(
            world_0.latitude, world_0.longitude, world_90.latitude, world_90.longitude
        )
        dist_0_to_180 = haversine_distance(
            world_0.latitude, world_0.longitude, world_180.latitude, world_180.longitude
        )

        # The coordinates should be different (not identical)
        assert dist_0_to_90 > 0.1, (
            f"Pan 0° and 90° should produce different GPS coords: dist={dist_0_to_90}m"
        )
        assert dist_0_to_180 > 0.1, (
            f"Pan 0° and 180° should produce different GPS coords: dist={dist_0_to_180}m"
        )


class TestProjectPointsGPSFlow:
    """Test batch projection of multiple points to GPS coordinates."""

    def test_project_points_batch(self):
        """Test that project_points() returns list of WorldPoints."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Set camera GPS position
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Compute homography
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Project multiple points
        image_points = [
            (1280, 1400),  # Bottom center
            (1280, 1000),  # Middle center
            (1280, 600),  # Top center
        ]

        world_points = homography.project_points(image_points)

        # Verify correct number of results
        assert len(world_points) == len(image_points)

        # Verify all are WorldPoint instances
        for wp in world_points:
            assert isinstance(wp, WorldPoint)
            assert -90 <= wp.latitude <= 90
            assert -180 <= wp.longitude <= 180
            assert 0.0 <= wp.confidence <= 1.0

    def test_project_points_same_as_individual(self):
        """Test that batch projection gives same results as individual projections."""
        homography = IntrinsicExtrinsicHomography(2560, 1440)

        # Set camera GPS position
        cam_lat = dms_to_dd("39°38'25.7\"N")
        cam_lon = dms_to_dd("0°13'48.7\"W")
        homography.set_camera_gps_position(cam_lat, cam_lon)

        # Compute homography
        K = homography.get_intrinsics(zoom_factor=1.0)
        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 45.0,
            "map_width": 640,
            "map_height": 480,
        }
        homography.compute_homography(np.zeros((1440, 2560, 3)), reference)

        # Test points
        image_points = [(1280, 1200), (1000, 800), (1500, 1100)]

        # Batch projection
        batch_results = homography.project_points(image_points)

        # Individual projections
        individual_results = [homography.project_point(pt) for pt in image_points]

        # Compare results
        for batch, individual in zip(batch_results, individual_results):
            assert abs(batch.latitude - individual.latitude) < 1e-9
            assert abs(batch.longitude - individual.longitude) < 1e-9
            assert abs(batch.confidence - individual.confidence) < 1e-9


def run_test_class(test_class, class_name):
    """Run all test methods in a test class."""
    print(f"\n{class_name}")
    print("-" * len(class_name))

    test_instance = test_class()
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        method = getattr(test_instance, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {method_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
            failed += 1

    return passed, failed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GPS COORDINATE FLOW TEST SUITE")
    print("=" * 70)

    test_classes = [
        (TestDMSParsing, "TestDMSParsing"),
        (TestGPSPositionSetting, "TestGPSPositionSetting"),
        (TestProjectPointGPSFlow, "TestProjectPointGPSFlow"),
        (TestGPSCoordinateAccuracy, "TestGPSCoordinateAccuracy"),
        (TestProjectPointsGPSFlow, "TestProjectPointsGPSFlow"),
    ]

    total_passed = 0
    total_failed = 0

    for test_class, class_name in test_classes:
        passed, failed = run_test_class(test_class, class_name)
        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70 + "\n")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
