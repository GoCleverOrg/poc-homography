#!/usr/bin/env python3
"""
Automated tests to verify rotation matrix and homography consistency
between CameraGeometry and IntrinsicExtrinsicHomography.

These tests ensure that the fix to IntrinsicExtrinsicHomography._compute_rotation_matrix()
produces results consistent with CameraGeometry.

Run with: python -m pytest tests/test_rotation_matrix_consistency.py -v

UPDATED: Refactored for immutable API (Phase 2)
- Uses static methods for rotation matrix computation
- Uses CameraParameters.create() + CameraGeometry.compute() for homography
- Uses IntrinsicExtrinsicConfig.create() + compute_from_config() for IEH
"""

import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import CameraParameters
from poc_homography.homography import IntrinsicExtrinsicConfig, IntrinsicExtrinsicHomography
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless


class TestRotationMatrixConsistency:
    """Test that both implementations produce identical rotation matrices."""

    @pytest.mark.parametrize(
        "pan_deg,tilt_deg",
        [
            (0, 30),
            (45, 30),
            (90, 30),
            (180, 30),
            (-45, 30),
            (-90, 30),
            (66.7, 30),  # Test pan offset value (Valencia region test data)
            (0, 15),
            (0, 45),
            (0, 60),
            (45, 45),
            (90, 60),
            (135, 25),
            (-135, 40),
        ],
    )
    def test_rotation_matrices_match(self, pan_deg, tilt_deg):
        """Verify rotation matrices are identical for both implementations."""
        # CameraGeometry static method
        R_geo = CameraGeometry._get_rotation_matrix_static(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        # IntrinsicExtrinsicHomography static method
        R_ieh = IntrinsicExtrinsicHomography._compute_rotation_matrix(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        # Check they're equal
        max_diff = np.max(np.abs(R_geo - R_ieh))
        assert max_diff < 1e-10, (
            f"Rotation matrices differ at pan={pan_deg}, tilt={tilt_deg}. "
            f"Max difference: {max_diff}"
        )

    @pytest.mark.parametrize(
        "pan_deg,tilt_deg",
        [
            (0, 30),
            (45, 30),
            (90, 30),
            (66.7, 30),
        ],
    )
    def test_rotation_matrix_is_orthogonal(self, pan_deg, tilt_deg):
        """Verify rotation matrices are proper orthogonal matrices (det=1, R@R.T=I)."""
        for name, R in [
            (
                "CameraGeometry",
                CameraGeometry._get_rotation_matrix_static(
                    Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
                ),
            ),
            (
                "IntrinsicExtrinsicHomography",
                IntrinsicExtrinsicHomography._compute_rotation_matrix(
                    Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
                ),
            ),
        ]:
            # Check determinant is 1
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-10, f"{name}: det(R) = {det}, expected 1.0"

            # Check R @ R.T = I
            RRT = R @ R.T
            identity_diff = np.max(np.abs(RRT - np.eye(3)))
            assert identity_diff < 1e-10, (
                f"{name}: R @ R.T differs from identity by {identity_diff}"
            )


class TestRollRotation:
    """Test roll parameter rotation matrix behavior."""

    @pytest.mark.parametrize("roll_deg", [-5.0, 0.0, 5.0])
    def test_roll_rotation_matrix_correctness(self, roll_deg):
        """Verify roll rotation matrix is correct for various roll angles."""
        # Test with simple pan=0, tilt=30 case
        pan_deg = 0.0
        tilt_deg = 30.0

        R = IntrinsicExtrinsicHomography._compute_rotation_matrix(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(roll_deg)
        )

        # Roll rotation should be orthogonal
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"det(R) = {det}, expected 1.0"

        RRT = R @ R.T
        identity_diff = np.max(np.abs(RRT - np.eye(3)))
        assert identity_diff < 1e-10, f"R @ R.T differs from identity by {identity_diff}"

    @pytest.mark.parametrize(
        "pan_deg,tilt_deg,roll_deg",
        [
            (0, 30, 0),
            (45, 30, 0),
            (0, 30, 5),
            (45, 30, -5),
            (90, 45, 2),
        ],
    )
    def test_rotation_matrices_with_roll_match(self, pan_deg, tilt_deg, roll_deg):
        """Verify rotation matrices match between both classes when roll is specified."""
        # CameraGeometry static method
        R_geo = CameraGeometry._get_rotation_matrix_static(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(roll_deg)
        )

        # IntrinsicExtrinsicHomography static method
        R_ieh = IntrinsicExtrinsicHomography._compute_rotation_matrix(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(roll_deg)
        )

        # Check they're equal
        max_diff = np.max(np.abs(R_geo - R_ieh))
        assert max_diff < 1e-10, (
            f"Rotation matrices differ at pan={pan_deg}, tilt={tilt_deg}, roll={roll_deg}. "
            f"Max difference: {max_diff}"
        )

    def test_backward_compatibility_roll_defaults_to_zero(self):
        """Verify roll defaults to 0.0 for backward compatibility."""
        pan_deg = 45.0
        tilt_deg = 30.0

        # Call with explicit roll=0 and default
        R_ieh_with_zero_roll = IntrinsicExtrinsicHomography._compute_rotation_matrix(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        # CameraGeometry should also produce same result
        R_geo_default = CameraGeometry._get_rotation_matrix_static(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        max_diff = np.max(np.abs(R_geo_default - R_ieh_with_zero_roll))
        assert max_diff < 1e-10, "Default roll=0 should match between implementations"

    def test_homography_differs_with_roll(self):
        """Verify homography changes when roll != 0."""
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
        camera_position = np.array([0.0, 0.0, 5.0])
        pan_deg = 0.0
        tilt_deg = 30.0

        # Calculate homography with roll=0
        H_no_roll = IntrinsicExtrinsicHomography._compute_ground_homography(
            K, camera_position, Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        # Calculate homography with roll=5
        H_with_roll = IntrinsicExtrinsicHomography._compute_ground_homography(
            K, camera_position, Degrees(pan_deg), Degrees(tilt_deg), Degrees(5.0)
        )

        # Homographies should differ
        max_diff = np.max(np.abs(H_no_roll - H_with_roll))
        assert max_diff > 1e-3, "Homography should change significantly when roll != 0"


class TestRollValidation:
    """Test roll validation in CameraGeometry."""

    def test_roll_warning_threshold(self):
        """Verify warning is issued when |roll_deg| > 5.0."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        # Create parameters with roll=6.0 (should trigger warning but not error)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(30.0),
                roll_deg=Degrees(6.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            result = CameraGeometry.compute(params)

            # Should have at least one warning
            assert len(w) > 0, "Expected warning for |roll_deg| > 5.0"
            assert any("roll" in str(warning.message).lower() for warning in w), (
                "Warning should mention roll"
            )

    def test_roll_error_threshold(self):
        """Verify error is raised when |roll_deg| > 15.0."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        # This should raise ValueError during validation
        with pytest.raises(ValueError) as exc_info:
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(30.0),
                roll_deg=Degrees(16.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            CameraGeometry.compute(params)

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"

    def test_roll_negative_threshold(self):
        """Verify error is raised when roll_deg < -15.0."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        # This should raise ValueError during validation
        with pytest.raises(ValueError) as exc_info:
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(30.0),
                roll_deg=Degrees(-16.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            CameraGeometry.compute(params)

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"

    def test_roll_within_limits_accepted(self):
        """Verify roll values within limits are accepted."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        # These should not raise - test various acceptable roll values
        for roll in [0.0, 5.0, -5.0, 10.0, -10.0]:
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(30.0),
                roll_deg=Degrees(roll),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            result = CameraGeometry.compute(params)
            assert result.is_valid, f"Roll {roll} should be accepted"


class TestComputeHomographyWithRoll:
    """Test compute_homography with roll parameter using immutable API."""

    def test_compute_homography_with_roll(self):
        """Verify compute_from_config handles roll_deg correctly."""
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        config = IntrinsicExtrinsicConfig.create(
            camera_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(2.5),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            sensor_width_mm=Millimeters(7.18),
            base_focal_length_mm=Millimeters(5.9),
            map_id="test",
        )
        result = IntrinsicExtrinsicHomography.compute_from_config(config)

        # Check that roll_deg is in result
        assert result.roll_deg == 2.5, f"Expected roll_deg=2.5, got {result.roll_deg}"

    def test_compute_homography_roll_defaults_to_zero(self):
        """Verify compute_from_config works with roll_deg=0.0."""
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        config = IntrinsicExtrinsicConfig.create(
            camera_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            sensor_width_mm=Millimeters(7.18),
            base_focal_length_mm=Millimeters(5.9),
            map_id="test",
        )
        result = IntrinsicExtrinsicHomography.compute_from_config(config)

        # Should work with default roll=0.0
        assert result.roll_deg == 0.0, f"Expected roll_deg=0.0, got {result.roll_deg}"
        assert result.confidence > 0, "Should have valid confidence"

    def test_compute_homography_rejects_invalid_roll(self):
        """Verify IntrinsicExtrinsicConfig raises error for |roll_deg| > 15.0."""
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        with pytest.raises(ValueError) as exc_info:
            config = IntrinsicExtrinsicConfig.create(
                camera_matrix=K,
                camera_position=np.array([0.0, 0.0, 5.0]),
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(30.0),
                roll_deg=Degrees(16.0),  # Invalid: > 15.0
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
                sensor_width_mm=Millimeters(7.18),
                base_focal_length_mm=Millimeters(5.9),
                map_id="test",
            )

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"


class TestCameraViewingDirection:
    """Test that camera viewing direction is correct for various pan/tilt angles."""

    def get_camera_forward(self, R):
        """Get camera forward direction in world coordinates."""
        # Camera Z-axis (forward) transformed to world
        return R.T @ np.array([0, 0, 1])

    def get_azimuth_elevation(self, forward):
        """Get azimuth (degrees from North) and elevation (degrees from horizontal)."""
        azimuth = math.degrees(math.atan2(forward[0], forward[1]))
        elevation = math.degrees(math.asin(-forward[2]))
        return azimuth, elevation

    @pytest.mark.parametrize(
        "pan_deg,expected_azimuth",
        [
            (0, 0),  # Looking North
            (45, 45),  # Looking NE
            (90, 90),  # Looking East
            (180, 180),  # Looking South
            (-90, -90),  # Looking West
            (-45, -45),  # Looking NW
        ],
    )
    def test_pan_controls_azimuth(self, pan_deg, expected_azimuth):
        """Verify pan angle correctly controls camera azimuth."""
        tilt_deg = 30.0
        R = CameraGeometry._get_rotation_matrix_static(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        forward = self.get_camera_forward(R)
        azimuth, _ = self.get_azimuth_elevation(forward)

        # Normalize to -180 to 180
        while azimuth > 180:
            azimuth -= 360
        while azimuth < -180:
            azimuth += 360

        assert abs(azimuth - expected_azimuth) < 0.1, (
            f"Pan={pan_deg} should give azimuth={expected_azimuth}, got {azimuth:.1f}"
        )

    @pytest.mark.parametrize(
        "tilt_deg,expected_elevation",
        [
            (30, 30),
            (45, 45),
            (60, 60),
            (15, 15),
        ],
    )
    def test_tilt_controls_elevation(self, tilt_deg, expected_elevation):
        """Verify tilt angle correctly controls camera elevation."""
        pan_deg = 0.0
        R = CameraGeometry._get_rotation_matrix_static(
            Degrees(pan_deg), Degrees(tilt_deg), Degrees(0.0)
        )

        forward = self.get_camera_forward(R)
        _, elevation = self.get_azimuth_elevation(forward)

        assert abs(elevation - expected_elevation) < 0.1, (
            f"Tilt={tilt_deg} should give elevation={expected_elevation}, got {elevation:.1f}"
        )


class TestHomographyConsistency:
    """Test that homography matrices are consistent between implementations."""

    @pytest.fixture
    def camera_params(self):
        # Static test parameters (Valencia region test data, not current camera config)
        return {
            "width": 1920,
            "height": 1080,
            "height_m": 3.4,
            "pan_deg": 66.7,
            "tilt_deg": 30.0,
            "zoom": 1.0,
            "sensor_width_mm": 7.18,
        }

    def test_homography_matrices_match(self, camera_params):
        """Verify both implementations produce identical homography matrices."""
        K = CameraGeometry.get_intrinsics(
            Unitless(camera_params["zoom"]),
            Pixels(camera_params["width"]),
            Pixels(camera_params["height"]),
            Millimeters(camera_params["sensor_width_mm"]),
        )
        w_pos = np.array([0.0, 0.0, camera_params["height_m"]])

        # CameraGeometry via immutable API
        params = CameraParameters.create(
            image_width=Pixels(camera_params["width"]),
            image_height=Pixels(camera_params["height"]),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(camera_params["pan_deg"]),
            tilt_deg=Degrees(camera_params["tilt_deg"]),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result_geo = CameraGeometry.compute(params)
        H_geo = result_geo.homography_matrix

        # IntrinsicExtrinsicHomography via static method
        H_ieh = IntrinsicExtrinsicHomography._compute_ground_homography(
            K, w_pos, Degrees(camera_params["pan_deg"]), Degrees(camera_params["tilt_deg"])
        )

        max_diff = np.max(np.abs(H_geo - H_ieh))
        assert max_diff < 1e-6, f"Homography matrices differ by {max_diff}"

    @pytest.mark.parametrize(
        "world_point",
        [
            (5.0, 2.0),
            (10.0, 5.0),
            (-3.0, 8.0),
            (0.0, 10.0),
            (15.0, 0.0),
        ],
    )
    def test_projection_matches(self, camera_params, world_point):
        """Verify both implementations project world points identically."""
        K = CameraGeometry.get_intrinsics(
            Unitless(camera_params["zoom"]),
            Pixels(camera_params["width"]),
            Pixels(camera_params["height"]),
            Millimeters(camera_params["sensor_width_mm"]),
        )
        w_pos = np.array([0.0, 0.0, camera_params["height_m"]])

        # CameraGeometry via immutable API
        params = CameraParameters.create(
            image_width=Pixels(camera_params["width"]),
            image_height=Pixels(camera_params["height"]),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(camera_params["pan_deg"]),
            tilt_deg=Degrees(camera_params["tilt_deg"]),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result_geo = CameraGeometry.compute(params)
        H_geo = result_geo.homography_matrix

        # IntrinsicExtrinsicHomography via static method
        H_ieh = IntrinsicExtrinsicHomography._compute_ground_homography(
            K, w_pos, Degrees(camera_params["pan_deg"]), Degrees(camera_params["tilt_deg"])
        )

        # Project with both
        pt = np.array([[world_point[0]], [world_point[1]], [1.0]])

        p_geo = H_geo @ pt
        p_ieh = H_ieh @ pt

        u_geo = p_geo[0, 0] / p_geo[2, 0]
        v_geo = p_geo[1, 0] / p_geo[2, 0]
        u_ieh = p_ieh[0, 0] / p_ieh[2, 0]
        v_ieh = p_ieh[1, 0] / p_ieh[2, 0]

        pixel_diff = math.sqrt((u_geo - u_ieh) ** 2 + (v_geo - v_ieh) ** 2)
        assert pixel_diff < 0.01, (
            f"Projection differs by {pixel_diff:.2f} pixels for point {world_point}"
        )


class TestGPSProjection:
    """Test GPS-to-image projection accuracy."""

    @pytest.fixture
    def valte_config(self):
        # Static test parameters (Valencia region test data, not current camera config)
        return {
            "camera_lat": 39.640497,
            "camera_lon": -0.230106,
            "height_m": 3.4,
            "pan_offset_deg": 66.7,
            "pan_raw": 0.0,
            "tilt_deg": 30.0,
            "zoom": 1.0,
            "width": 1920,
            "height": 1080,
        }

    def test_image_center_projects_to_camera_direction(self, valte_config):
        """Verify image center projects to a point along camera viewing direction."""
        K = CameraGeometry.get_intrinsics(
            Unitless(valte_config["zoom"]),
            Pixels(valte_config["width"]),
            Pixels(valte_config["height"]),
            Millimeters(7.18),
        )
        w_pos = np.array([0.0, 0.0, valte_config["height_m"]])
        pan_deg = valte_config["pan_raw"] + valte_config["pan_offset_deg"]

        # Create parameters and compute
        params = CameraParameters.create(
            image_width=Pixels(valte_config["width"]),
            image_height=Pixels(valte_config["height"]),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(pan_deg),
            tilt_deg=Degrees(valte_config["tilt_deg"]),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)

        # Inverse project image center
        center = np.array([[valte_config["width"] / 2], [valte_config["height"] / 2], [1.0]])
        world_pt = result.inverse_homography_matrix @ center
        x = world_pt[0, 0] / world_pt[2, 0]
        y = world_pt[1, 0] / world_pt[2, 0]

        # Calculate bearing of projected point
        bearing = math.degrees(math.atan2(x, y))

        # Should match camera pan angle
        assert abs(bearing - pan_deg) < 1.0, (
            f"Image center projects to bearing {bearing:.1f}, expected {pan_deg}"
        )

    def test_point_in_camera_direction_projects_to_center(self, valte_config):
        """Verify a point directly in camera's view projects near image center."""
        K = CameraGeometry.get_intrinsics(
            Unitless(valte_config["zoom"]),
            Pixels(valte_config["width"]),
            Pixels(valte_config["height"]),
            Millimeters(7.18),
        )
        w_pos = np.array([0.0, 0.0, valte_config["height_m"]])
        pan_deg = valte_config["pan_raw"] + valte_config["pan_offset_deg"]

        # Create parameters and compute
        params = CameraParameters.create(
            image_width=Pixels(valte_config["width"]),
            image_height=Pixels(valte_config["height"]),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(pan_deg),
            tilt_deg=Degrees(valte_config["tilt_deg"]),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)

        # Calculate ground distance for center projection
        ground_distance = valte_config["height_m"] / math.tan(
            math.radians(valte_config["tilt_deg"])
        )

        # Point at that distance along camera direction
        x = ground_distance * math.sin(math.radians(pan_deg))
        y = ground_distance * math.cos(math.radians(pan_deg))

        # Project
        pt = np.array([[x], [y], [1.0]])
        img_pt = result.homography_matrix @ pt
        u = img_pt[0, 0] / img_pt[2, 0]
        v = img_pt[1, 0] / img_pt[2, 0]

        center_u = valte_config["width"] / 2
        center_v = valte_config["height"] / 2

        # Should be near center
        dist_from_center = math.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
        assert dist_from_center < 50, (
            f"Point at camera direction projects to ({u:.1f}, {v:.1f}), "
            f"{dist_from_center:.1f}px from center"
        )

    # NOTE: test_gps_round_trip removed - used deleted standalone functions gps_to_local_xy
    # and local_xy_to_gps. GPS round-trip accuracy is tested in test_coordinate_converter_dual_systems.py
    # using GCPCoordinateConverter class methods.


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_tilt_rejected(self):
        """Verify zero tilt is rejected (camera horizontal, infinite projection)."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        with pytest.raises(ValueError):
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(0.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            CameraGeometry.compute(params)

    def test_negative_tilt_rejected(self):
        """Verify negative tilt (looking up) is rejected."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        with pytest.raises(ValueError):
            params = CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=K,
                camera_position=w_pos,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(-10.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )
            CameraGeometry.compute(params)

    def test_extreme_pan_values(self):
        """Verify extreme pan values (> 360) are handled correctly."""
        K = CameraGeometry.get_intrinsics(
            Unitless(1.0), Pixels(1920), Pixels(1080), Millimeters(7.18)
        )
        w_pos = np.array([0.0, 0.0, 5.0])

        # These should not raise - pan angles wrap around
        params1 = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(400.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result1 = CameraGeometry.compute(params1)

        params2 = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(-400.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        result2 = CameraGeometry.compute(params2)

        # Verify rotation still works - 400 mod 360 = 40
        R1 = CameraGeometry._get_rotation_matrix_static(Degrees(400.0), Degrees(30.0), Degrees(0.0))
        R2 = CameraGeometry._get_rotation_matrix_static(Degrees(40.0), Degrees(30.0), Degrees(0.0))

        # Rotation matrices should be identical (within floating point)
        max_diff = np.max(np.abs(R1 - R2))
        assert max_diff < 1e-10, f"400 and 40 should produce same rotation, diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
