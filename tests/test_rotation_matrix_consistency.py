#!/usr/bin/env python3
"""
Automated tests to verify rotation matrix and homography consistency
between CameraGeometry and IntrinsicExtrinsicHomography.

These tests ensure that the fix to IntrinsicExtrinsicHomography._get_rotation_matrix()
produces results consistent with CameraGeometry.

Run with: python -m pytest tests/test_rotation_matrix_consistency.py -v
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry

# NOTE: Removed import of deleted standalone functions gps_to_local_xy, local_xy_to_gps
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


class TestRotationMatrixConsistency:
    """Test that both implementations produce identical rotation matrices."""

    @pytest.fixture
    def geometry_instances(self):
        """Create instances of both geometry classes."""
        geo = CameraGeometry(w=1920, h=1080)
        ieh = IntrinsicExtrinsicHomography(map_id="test_map", width=1920, height=1080)
        return geo, ieh

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
    def test_rotation_matrices_match(self, geometry_instances, pan_deg, tilt_deg):
        """Verify rotation matrices are identical for both implementations."""
        geo, ieh = geometry_instances

        # CameraGeometry
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg
        R_geo = geo._get_rotation_matrix()

        # IntrinsicExtrinsicHomography
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg)

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
    def test_rotation_matrix_is_orthogonal(self, geometry_instances, pan_deg, tilt_deg):
        """Verify rotation matrices are proper orthogonal matrices (det=1, R@R.T=I)."""
        geo, ieh = geometry_instances

        for name, R in [
            ("CameraGeometry", geo._get_rotation_matrix()),
            ("IntrinsicExtrinsicHomography", ieh._get_rotation_matrix(pan_deg, tilt_deg)),
        ]:
            geo.pan_deg = pan_deg
            geo.tilt_deg = tilt_deg
            R = (
                geo._get_rotation_matrix()
                if "Camera" in name
                else ieh._get_rotation_matrix(pan_deg, tilt_deg)
            )

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

    @pytest.fixture
    def geometry_instances(self):
        """Create instances of both geometry classes."""
        geo = CameraGeometry(w=1920, h=1080)
        ieh = IntrinsicExtrinsicHomography(map_id="test_map", width=1920, height=1080)
        return geo, ieh

    @pytest.mark.parametrize("roll_deg", [-5.0, 0.0, 5.0])
    def test_roll_rotation_matrix_correctness(self, geometry_instances, roll_deg):
        """Verify roll rotation matrix is correct for various roll angles."""
        _, ieh = geometry_instances

        # Test with simple pan=0, tilt=30 case
        pan_deg = 0.0
        tilt_deg = 30.0

        R = ieh._get_rotation_matrix(pan_deg, tilt_deg, roll_deg)

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
    def test_rotation_matrices_with_roll_match(
        self, geometry_instances, pan_deg, tilt_deg, roll_deg
    ):
        """Verify rotation matrices match between both classes when roll is specified."""
        geo, ieh = geometry_instances

        # CameraGeometry
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg
        geo.roll_deg = roll_deg
        R_geo = geo._get_rotation_matrix(roll_deg=roll_deg)

        # IntrinsicExtrinsicHomography
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg, roll_deg)

        # Check they're equal
        max_diff = np.max(np.abs(R_geo - R_ieh))
        assert max_diff < 1e-10, (
            f"Rotation matrices differ at pan={pan_deg}, tilt={tilt_deg}, roll={roll_deg}. "
            f"Max difference: {max_diff}"
        )

    def test_backward_compatibility_roll_defaults_to_zero(self, geometry_instances):
        """Verify roll defaults to 0.0 for backward compatibility."""
        geo, ieh = geometry_instances

        pan_deg = 45.0
        tilt_deg = 30.0

        # Call without roll parameter (should default to 0)
        R_ieh_no_roll = ieh._get_rotation_matrix(pan_deg, tilt_deg)
        R_ieh_with_zero_roll = ieh._get_rotation_matrix(pan_deg, tilt_deg, roll_deg=0.0)

        max_diff = np.max(np.abs(R_ieh_no_roll - R_ieh_with_zero_roll))
        assert max_diff < 1e-10, "Roll should default to 0.0"

        # CameraGeometry should also default to 0
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg
        geo.roll_deg = 0.0
        R_geo_default = geo._get_rotation_matrix()

        max_diff = np.max(np.abs(R_geo_default - R_ieh_no_roll))
        assert max_diff < 1e-10, "Default roll=0 should match between implementations"

    def test_homography_differs_with_roll(self, geometry_instances):
        """Verify homography changes when roll != 0."""
        _, ieh = geometry_instances

        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
        camera_position = np.array([0.0, 0.0, 5.0])
        pan_deg = 0.0
        tilt_deg = 30.0

        # Calculate homography with roll=0
        H_no_roll = ieh._calculate_ground_homography(K, camera_position, pan_deg, tilt_deg)

        # Calculate homography with roll=5
        R_with_roll = ieh._get_rotation_matrix(pan_deg, tilt_deg, roll_deg=5.0)

        # Build homography manually with roll
        C = camera_position
        t = -R_with_roll @ C
        r1 = R_with_roll[:, 0]
        r2 = R_with_roll[:, 1]
        H_extrinsic = np.column_stack([r1, r2, t])
        H_with_roll = K @ H_extrinsic
        H_with_roll = H_with_roll / H_with_roll[2, 2]

        # Homographies should differ
        max_diff = np.max(np.abs(H_no_roll - H_with_roll))
        assert max_diff > 1e-3, "Homography should change significantly when roll != 0"


class TestRollValidation:
    """Test roll validation in CameraGeometry."""

    def test_roll_warning_threshold(self):
        """Verify warning is issued when |roll_deg| > 5.0."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        # This should trigger a warning but not raise an error
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=6.0)

            # Should have at least one warning
            assert len(w) > 0, "Expected warning for |roll_deg| > 5.0"
            assert any("roll" in str(warning.message).lower() for warning in w), (
                "Warning should mention roll"
            )

    def test_roll_error_threshold(self):
        """Verify error is raised when |roll_deg| > 15.0."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        # This should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=16.0)

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"

    def test_roll_negative_threshold(self):
        """Verify error is raised when roll_deg < -15.0."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        # This should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=-16.0)

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"

    def test_roll_within_limits_accepted(self):
        """Verify roll values within limits are accepted."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        # These should not raise
        geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=0.0)
        geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=5.0)
        geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=-5.0)
        geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=10.0)
        geo.set_camera_parameters(K, w_pos, 0.0, 30.0, 640, 640, roll_deg=-10.0)


class TestComputeHomographyWithRoll:
    """Test compute_homography with roll parameter."""

    def test_compute_homography_extracts_roll(self):
        """Verify compute_homography extracts roll_deg from reference dict."""
        ieh = IntrinsicExtrinsicHomography(map_id="test_map", width=1920, height=1080)

        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 30.0,
            "roll_deg": 2.5,
            "map_width": 640,
            "map_height": 640,
        }

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ieh.compute_homography(frame, reference)

        # Check that roll_deg is stored
        assert hasattr(ieh, "roll_deg"), "roll_deg should be stored as instance attribute"
        assert ieh.roll_deg == 2.5, f"Expected roll_deg=2.5, got {ieh.roll_deg}"

        # Check that roll_deg is in metadata
        assert "roll_deg" in result.metadata, "roll_deg should be in metadata"
        assert result.metadata["roll_deg"] == 2.5, (
            f"Expected metadata roll_deg=2.5, got {result.metadata['roll_deg']}"
        )

    def test_compute_homography_roll_defaults_to_zero(self):
        """Verify compute_homography defaults roll_deg to 0.0 when not in reference."""
        ieh = IntrinsicExtrinsicHomography(map_id="test_map", width=1920, height=1080)

        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 30.0,
            # Note: no 'roll_deg' key
            "map_width": 640,
            "map_height": 640,
        }

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = ieh.compute_homography(frame, reference)

        # Should default to 0.0
        assert hasattr(ieh, "roll_deg"), "roll_deg should be stored as instance attribute"
        assert ieh.roll_deg == 0.0, f"Expected default roll_deg=0.0, got {ieh.roll_deg}"

        # Check metadata
        assert "roll_deg" in result.metadata, "roll_deg should be in metadata"
        assert result.metadata["roll_deg"] == 0.0, (
            f"Expected metadata roll_deg=0.0, got {result.metadata['roll_deg']}"
        )

    def test_compute_homography_rejects_invalid_roll(self):
        """Verify compute_homography raises error for |roll_deg| > 15.0."""
        ieh = IntrinsicExtrinsicHomography(map_id="test_map", width=1920, height=1080)

        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

        reference = {
            "camera_matrix": K,
            "camera_position": np.array([0.0, 0.0, 5.0]),
            "pan_deg": 0.0,
            "tilt_deg": 30.0,
            "roll_deg": 16.0,  # Invalid: > 15.0
            "map_width": 640,
            "map_height": 640,
        }

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with pytest.raises(ValueError) as exc_info:
            ieh.compute_homography(frame, reference)

        assert "roll" in str(exc_info.value).lower(), "Error message should mention roll"


class TestCameraViewingDirection:
    """Test that camera viewing direction is correct for various pan/tilt angles."""

    @pytest.fixture
    def geo(self):
        return CameraGeometry(w=1920, h=1080)

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
    def test_pan_controls_azimuth(self, geo, pan_deg, expected_azimuth):
        """Verify pan angle correctly controls camera azimuth."""
        tilt_deg = 30.0
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg
        R = geo._get_rotation_matrix()

        forward = self.get_camera_forward(R)
        azimuth, _ = self.get_azimuth_elevation(forward)

        # Normalize to -180 to 180
        while azimuth > 180:
            azimuth -= 360
        while azimuth < -180:
            azimuth += 360

        assert abs(azimuth - expected_azimuth) < 0.1, (
            f"Pan={pan_deg}° should give azimuth={expected_azimuth}°, got {azimuth:.1f}°"
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
    def test_tilt_controls_elevation(self, geo, tilt_deg, expected_elevation):
        """Verify tilt angle correctly controls camera elevation."""
        pan_deg = 0.0
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg
        R = geo._get_rotation_matrix()

        forward = self.get_camera_forward(R)
        _, elevation = self.get_azimuth_elevation(forward)

        assert abs(elevation - expected_elevation) < 0.1, (
            f"Tilt={tilt_deg}° should give elevation={expected_elevation}°, got {elevation:.1f}°"
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
        geo = CameraGeometry(w=camera_params["width"], h=camera_params["height"])
        ieh = IntrinsicExtrinsicHomography(
            map_id="test_map", width=camera_params["width"], height=camera_params["height"]
        )

        K = CameraGeometry.get_intrinsics(
            camera_params["zoom"],
            camera_params["width"],
            camera_params["height"],
            camera_params["sensor_width_mm"],
        )
        w_pos = np.array([0.0, 0.0, camera_params["height_m"]])

        # CameraGeometry
        geo.set_camera_parameters(
            K, w_pos, camera_params["pan_deg"], camera_params["tilt_deg"], 640, 640
        )
        H_geo = geo.H

        # IntrinsicExtrinsicHomography
        H_ieh = ieh._calculate_ground_homography(
            K, w_pos, camera_params["pan_deg"], camera_params["tilt_deg"]
        )
        H_ieh_norm = H_ieh / H_ieh[2, 2]

        max_diff = np.max(np.abs(H_geo - H_ieh_norm))
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
        geo = CameraGeometry(w=camera_params["width"], h=camera_params["height"])
        ieh = IntrinsicExtrinsicHomography(
            map_id="test_map", width=camera_params["width"], height=camera_params["height"]
        )

        K = CameraGeometry.get_intrinsics(
            camera_params["zoom"],
            camera_params["width"],
            camera_params["height"],
            camera_params["sensor_width_mm"],
        )
        w_pos = np.array([0.0, 0.0, camera_params["height_m"]])

        geo.set_camera_parameters(
            K, w_pos, camera_params["pan_deg"], camera_params["tilt_deg"], 640, 640
        )
        H_geo = geo.H

        H_ieh = ieh._calculate_ground_homography(
            K, w_pos, camera_params["pan_deg"], camera_params["tilt_deg"]
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
        geo = CameraGeometry(w=valte_config["width"], h=valte_config["height"])
        K = CameraGeometry.get_intrinsics(
            valte_config["zoom"], valte_config["width"], valte_config["height"], 7.18
        )
        w_pos = np.array([0.0, 0.0, valte_config["height_m"]])
        pan_deg = valte_config["pan_raw"] + valte_config["pan_offset_deg"]

        geo.set_camera_parameters(K, w_pos, pan_deg, valte_config["tilt_deg"], 640, 640)

        # Inverse project image center
        center = np.array([[valte_config["width"] / 2], [valte_config["height"] / 2], [1.0]])
        world_pt = geo.H_inv @ center
        x = world_pt[0, 0] / world_pt[2, 0]
        y = world_pt[1, 0] / world_pt[2, 0]

        # Calculate bearing of projected point
        bearing = math.degrees(math.atan2(x, y))

        # Should match camera pan angle
        assert abs(bearing - pan_deg) < 1.0, (
            f"Image center projects to bearing {bearing:.1f}°, expected {pan_deg}°"
        )

    def test_point_in_camera_direction_projects_to_center(self, valte_config):
        """Verify a point directly in camera's view projects near image center."""
        geo = CameraGeometry(w=valte_config["width"], h=valte_config["height"])
        K = CameraGeometry.get_intrinsics(
            valte_config["zoom"], valte_config["width"], valte_config["height"], 7.18
        )
        w_pos = np.array([0.0, 0.0, valte_config["height_m"]])
        pan_deg = valte_config["pan_raw"] + valte_config["pan_offset_deg"]

        geo.set_camera_parameters(K, w_pos, pan_deg, valte_config["tilt_deg"], 640, 640)

        # Calculate ground distance for center projection
        ground_distance = valte_config["height_m"] / math.tan(
            math.radians(valte_config["tilt_deg"])
        )

        # Point at that distance along camera direction
        x = ground_distance * math.sin(math.radians(pan_deg))
        y = ground_distance * math.cos(math.radians(pan_deg))

        # Project
        pt = np.array([[x], [y], [1.0]])
        img_pt = geo.H @ pt
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
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        with pytest.raises(ValueError):
            geo.set_camera_parameters(K, w_pos, 0.0, 0.0, 640, 640)

    def test_negative_tilt_rejected(self):
        """Verify negative tilt (looking up) is rejected."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        with pytest.raises(ValueError):
            geo.set_camera_parameters(K, w_pos, 0.0, -10.0, 640, 640)

    def test_extreme_pan_values(self):
        """Verify extreme pan values (> 360) are handled correctly."""
        geo = CameraGeometry(w=1920, h=1080)
        K = CameraGeometry.get_intrinsics(1.0, 1920, 1080, 7.18)
        w_pos = np.array([0.0, 0.0, 5.0])

        # These should not raise
        geo.set_camera_parameters(K, w_pos, 400.0, 30.0, 640, 640)  # 400 = 40
        geo.set_camera_parameters(K, w_pos, -400.0, 30.0, 640, 640)  # -400 = -40

        # Verify rotation still works
        geo.pan_deg = 400.0
        geo.tilt_deg = 30.0
        R1 = geo._get_rotation_matrix()

        geo.pan_deg = 40.0  # Equivalent
        R2 = geo._get_rotation_matrix()

        # Rotation matrices should be identical (within floating point)
        # Note: 400 mod 360 = 40
        max_diff = np.max(np.abs(R1 - R2))
        assert max_diff < 1e-10, f"400° and 40° should produce same rotation, diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
