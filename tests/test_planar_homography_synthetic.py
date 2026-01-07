#!/usr/bin/env python3
"""
Synthetic projection tests for planar homography H = K[r1, r2, t].

These tests verify the mathematical correctness of the planar homography formula
by generating known camera poses and ground points, then validating that the
homography produces the expected pixel projections.

The key insight is that we can independently compute the expected projection
using the full 3D projection formula (for Z=0 points) and compare against
the homography-based projection. This provides ground truth verification.

Mathematical Background:
    For ground plane Z=0, the projection simplifies to a planar homography:

    General 3D: p_image = K @ (R @ P_world + t)
    For Z=0:    p_image = K @ [r1, r2, t] @ [X, Y, 1]^T = H @ [X, Y, 1]^T

    where H = K @ [r1, r2, t] is a 3x3 matrix (NOT 3x4 projection matrix).

Run with: python -m pytest tests/test_planar_homography_synthetic.py -v
"""

import numpy as np
import math
import sys
import os
import pytest
from hypothesis import given, strategies as st, assume, settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


# ============================================================================
# Hypothesis Strategies for Camera Parameters
# ============================================================================

@st.composite
def camera_dimensions(draw):
    """Generate valid camera image dimensions."""
    width = draw(st.integers(min_value=640, max_value=1920))
    height = draw(st.integers(min_value=480, max_value=1080))
    return width, height


@st.composite
def zoom_factor(draw):
    """Generate valid zoom factor values."""
    return draw(st.floats(min_value=1.0, max_value=10.0))


@st.composite
def pan_angle(draw):
    """Generate pan angle in degrees."""
    return draw(st.floats(min_value=-180.0, max_value=180.0))


@st.composite
def tilt_angle(draw):
    """
    Generate valid tilt angle in degrees.

    Tilt must be in range (10, 80] to ensure ground plane is visible
    and avoid numerical instability near horizontal or vertical.
    """
    return draw(st.floats(min_value=10.0, max_value=80.0))


@st.composite
def camera_position(draw):
    """
    Generate camera position in world coordinates.

    Position is relative to world origin.
    Z (height) must be > 1m to ensure camera is above ground.
    """
    x = draw(st.floats(min_value=-50.0, max_value=50.0))
    y = draw(st.floats(min_value=-50.0, max_value=50.0))
    z = draw(st.floats(min_value=2.0, max_value=20.0))
    return np.array([x, y, z])


@st.composite
def ground_point(draw, camera_pos, pan_deg, tilt_deg, max_distance=50.0):
    """
    Generate a ground point (Z=0) that is likely visible in the camera's FOV.

    The point is generated in front of the camera based on pan direction,
    with some lateral offset.
    """
    # Direction the camera is looking (based on pan)
    pan_rad = math.radians(pan_deg)

    # Distance from camera on ground (based on height and tilt)
    # At tilt degrees, ground distance is approximately height / tan(tilt)
    tilt_rad = math.radians(tilt_deg)
    if tilt_rad > 0.1:  # Avoid division by very small tilt
        typical_ground_dist = camera_pos[2] / math.tan(tilt_rad)
    else:
        typical_ground_dist = camera_pos[2] * 10

    # Generate point in vicinity of where camera is looking
    # Add some randomness to test various positions
    distance_scale = draw(st.floats(min_value=0.5, max_value=2.0))
    lateral_offset = draw(st.floats(min_value=-0.3, max_value=0.3))

    forward_dist = min(typical_ground_dist * distance_scale, max_distance)

    # Point in world coordinates (relative to camera XY position)
    # Forward direction based on pan (North = 0 degrees, clockwise positive)
    x_offset = forward_dist * math.sin(pan_rad) + lateral_offset * forward_dist * math.cos(pan_rad)
    y_offset = forward_dist * math.cos(pan_rad) - lateral_offset * forward_dist * math.sin(pan_rad)

    # Absolute world position
    x = camera_pos[0] + x_offset
    y = camera_pos[1] + y_offset

    return np.array([x, y, 0.0])  # Z=0 for ground plane


# ============================================================================
# Helper Functions for Independent Projection Computation
# ============================================================================

def compute_rotation_matrix(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """
    Compute rotation matrix from world to camera coordinates.

    This is an independent implementation to verify against the module's version.
    Uses the same convention:
    - World: X=East, Y=North, Z=Up
    - Camera: X=Right, Y=Down, Z=Forward
    """
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)

    # Base transform: world to camera when pan=0, tilt=0
    # Camera looking North, horizontal
    R_base = np.array([
        [1,  0,  0],
        [0,  0, -1],  # World Z (up) -> Camera -Y (down)
        [0,  1,  0]   # World Y (north) -> Camera Z (forward)
    ])

    # Pan rotation around world Z-axis
    Rz_pan = np.array([
        [math.cos(pan_rad), -math.sin(pan_rad), 0],
        [math.sin(pan_rad),  math.cos(pan_rad), 0],
        [0,                  0,                 1]
    ])

    # Tilt rotation around camera X-axis
    Rx_tilt = np.array([
        [1,  0,                 0],
        [0,  math.cos(tilt_rad), -math.sin(tilt_rad)],
        [0,  math.sin(tilt_rad),  math.cos(tilt_rad)]
    ])

    # Full rotation: pan in world, then base, then tilt in camera
    R = Rx_tilt @ R_base @ Rz_pan
    return R


def compute_expected_projection_3d(
    K: np.ndarray,
    R: np.ndarray,
    camera_position: np.ndarray,
    world_point: np.ndarray
) -> tuple:
    """
    Compute expected pixel coordinates using full 3D projection.

    This is the ground truth computation:
    p_image = K @ (R @ P_world + t)
    where t = -R @ C

    Returns:
        tuple: (u, v) pixel coordinates, or None if point is behind camera
    """
    # Translation: world origin in camera frame
    t = -R @ camera_position

    # Transform world point to camera coordinates
    P_camera = R @ world_point + t

    # Check if point is in front of camera (positive Z in camera frame)
    if P_camera[2] <= 0:
        return None  # Point behind camera

    # Project to image
    p_homogeneous = K @ P_camera

    # Convert from homogeneous coordinates
    u = p_homogeneous[0] / p_homogeneous[2]
    v = p_homogeneous[1] / p_homogeneous[2]

    return (u, v)


def project_via_homography(H: np.ndarray, world_point_xy: np.ndarray) -> tuple:
    """
    Project ground point using homography.

    H maps [X, Y, 1] to [u, v, 1] (homogeneous coordinates).

    Args:
        H: 3x3 homography matrix
        world_point_xy: [X, Y] coordinates on ground plane (Z=0)

    Returns:
        tuple: (u, v) pixel coordinates
    """
    # Homogeneous coordinates [X, Y, 1]
    p_world = np.array([world_point_xy[0], world_point_xy[1], 1.0])

    # Apply homography
    p_image = H @ p_world

    # Convert from homogeneous
    u = p_image[0] / p_image[2]
    v = p_image[1] / p_image[2]

    return (u, v)


# ============================================================================
# Property Tests: Synthetic Projection Validation
# ============================================================================

class TestHomographySyntheticProjection:
    """
    Synthetic tests verifying H = K[r1, r2, t] produces correct projections.

    These tests compare homography-based projection against independent
    3D projection computation to verify mathematical correctness.
    """

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=100)
    def test_homography_matches_3d_projection_camera_geometry(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """
        Property: Homography projection matches full 3D projection for Z=0 points.

        This is the core validation: for any ground point (Z=0), the homography
        H = K[r1, r2, t] must produce the same pixel coordinates as the full
        3D projection formula p = K @ (R @ P + t).

        Mathematical verification:
        - Full 3D: p = K @ (R @ [X, Y, 0]^T + t) = K @ (r1*X + r2*Y + t)
        - Homography: p = K @ [r1, r2, t] @ [X, Y, 1]^T = K @ (r1*X + r2*Y + t)
        - These are algebraically identical.
        """
        width, height = dimensions

        # Setup CameraGeometry
        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

        # Get the homography
        H = geo.H
        assert H.shape == (3, 3), f"Homography must be 3x3, got {H.shape}"

        # Compute rotation matrix independently
        R = compute_rotation_matrix(pan_deg, tilt_deg)

        # Generate ground points in camera's field of view
        # Test multiple points to cover different projection angles
        test_points = []

        # Point directly in front of camera
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        if tilt_rad > 0.1:
            forward_dist = pos[2] / math.tan(tilt_rad)
        else:
            forward_dist = pos[2] * 5
        forward_dist = min(forward_dist, 100.0)  # Cap at reasonable distance

        # Generate test points
        for dist_scale in [0.5, 1.0, 1.5]:
            for lateral in [-0.2, 0.0, 0.2]:
                d = forward_dist * dist_scale
                x = pos[0] + d * math.sin(pan_rad) + lateral * d * math.cos(pan_rad)
                y = pos[1] + d * math.cos(pan_rad) - lateral * d * math.sin(pan_rad)
                test_points.append(np.array([x, y, 0.0]))

        for world_point in test_points:
            # Compute expected projection using full 3D formula
            expected = compute_expected_projection_3d(K, R, pos, world_point)

            # Skip points behind camera
            if expected is None:
                continue

            expected_u, expected_v = expected

            # Skip points outside reasonable image bounds (with margin)
            if not (-width < expected_u < 2*width and -height < expected_v < 2*height):
                continue

            # Project using homography
            actual_u, actual_v = project_via_homography(H, world_point[:2])

            # Compare - should be identical within numerical tolerance
            pixel_diff = math.sqrt((actual_u - expected_u)**2 + (actual_v - expected_v)**2)

            assert pixel_diff < 1e-6, (
                f"Homography projection differs from 3D projection by {pixel_diff:.2e} pixels\n"
                f"World point: {world_point}\n"
                f"Expected: ({expected_u:.4f}, {expected_v:.4f})\n"
                f"Actual: ({actual_u:.4f}, {actual_v:.4f})\n"
                f"Camera pos: {pos}, pan: {pan_deg}, tilt: {tilt_deg}"
            )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=100)
    def test_homography_matches_3d_projection_intrinsic_extrinsic(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """
        Property: IntrinsicExtrinsicHomography matches full 3D projection.

        Same validation as above but for the IntrinsicExtrinsicHomography class.
        """
        width, height = dimensions

        # Setup IntrinsicExtrinsicHomography
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)

        # Calculate homography
        H = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)
        assert H.shape == (3, 3), f"Homography must be 3x3, got {H.shape}"

        # Compute rotation matrix independently
        R = compute_rotation_matrix(pan_deg, tilt_deg)

        # Generate test points
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        if tilt_rad > 0.1:
            forward_dist = pos[2] / math.tan(tilt_rad)
        else:
            forward_dist = pos[2] * 5
        forward_dist = min(forward_dist, 100.0)

        test_points = []
        for dist_scale in [0.5, 1.0, 1.5]:
            for lateral in [-0.2, 0.0, 0.2]:
                d = forward_dist * dist_scale
                x = pos[0] + d * math.sin(pan_rad) + lateral * d * math.cos(pan_rad)
                y = pos[1] + d * math.cos(pan_rad) - lateral * d * math.sin(pan_rad)
                test_points.append(np.array([x, y, 0.0]))

        for world_point in test_points:
            expected = compute_expected_projection_3d(K, R, pos, world_point)

            if expected is None:
                continue

            expected_u, expected_v = expected

            if not (-width < expected_u < 2*width and -height < expected_v < 2*height):
                continue

            actual_u, actual_v = project_via_homography(H, world_point[:2])

            pixel_diff = math.sqrt((actual_u - expected_u)**2 + (actual_v - expected_v)**2)

            assert pixel_diff < 1e-6, (
                f"IEH projection differs from 3D projection by {pixel_diff:.2e} pixels\n"
                f"World point: {world_point}\n"
                f"Expected: ({expected_u:.4f}, {expected_v:.4f})\n"
                f"Actual: ({actual_u:.4f}, {actual_v:.4f})"
            )


class TestHomographyShapeAndNormalization:
    """
    Tests verifying homography matrix shape (3x3) and normalization (H[2,2] = 1).
    """

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=50)
    def test_homography_shape_camera_geometry(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """Property: CameraGeometry homography must be 3x3 (not 3x4)."""
        width, height = dimensions

        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

        assert geo.H.shape == (3, 3), (
            f"Homography must be 3x3 (planar), got {geo.H.shape}. "
            f"A 3x4 matrix would be a projection matrix, not a homography."
        )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=50)
    def test_homography_shape_intrinsic_extrinsic(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """Property: IntrinsicExtrinsicHomography must be 3x3 (not 3x4)."""
        width, height = dimensions

        ieh = IntrinsicExtrinsicHomography(width=width, height=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        H = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)

        assert H.shape == (3, 3), (
            f"Homography must be 3x3 (planar), got {H.shape}. "
            f"A 3x4 matrix would be a projection matrix, not a homography."
        )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=50)
    def test_homography_normalized_camera_geometry(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """Property: CameraGeometry homography must be normalized (H[2,2] = 1)."""
        width, height = dimensions

        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

        # Skip if we got identity (normalization failed)
        if np.allclose(geo.H, np.eye(3)):
            return

        assert abs(geo.H[2, 2] - 1.0) < 1e-10, (
            f"Homography should be normalized with H[2,2] = 1, got H[2,2] = {geo.H[2, 2]}"
        )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=50)
    def test_homography_normalized_intrinsic_extrinsic(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """Property: IntrinsicExtrinsicHomography must be normalized (H[2,2] = 1)."""
        width, height = dimensions

        ieh = IntrinsicExtrinsicHomography(width=width, height=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        H = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)

        # Skip if we got identity (normalization failed)
        if np.allclose(H, np.eye(3)):
            return

        assert abs(H[2, 2] - 1.0) < 1e-10, (
            f"Homography should be normalized with H[2,2] = 1, got H[2,2] = {H[2, 2]}"
        )


class TestHomographyConsistencyBetweenModules:
    """
    Tests verifying consistency between CameraGeometry and IntrinsicExtrinsicHomography.
    """

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    @settings(max_examples=100)
    def test_both_modules_produce_identical_homography(
        self, pos, pan_deg, tilt_deg, zoom, dimensions
    ):
        """
        Property: Both modules must produce identical homography matrices.

        This ensures implementation consistency between the two modules.
        """
        width, height = dimensions

        # Setup CameraGeometry
        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)
        H_geo = geo.H

        # Setup IntrinsicExtrinsicHomography
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)
        H_ieh = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)

        # Both should be 3x3
        assert H_geo.shape == (3, 3)
        assert H_ieh.shape == (3, 3)

        # Compare matrices
        max_diff = np.max(np.abs(H_geo - H_ieh))

        assert max_diff < 1e-6, (
            f"Homography matrices differ by up to {max_diff:.2e}\n"
            f"CameraGeometry H:\n{H_geo}\n"
            f"IntrinsicExtrinsicHomography H:\n{H_ieh}"
        )


class TestSpecificCameraConfigurations:
    """
    Tests with specific, known camera configurations for easy debugging.

    These deterministic tests complement the property-based tests with
    concrete examples that are easy to verify manually.
    """

    def test_camera_looking_north_45_degree_tilt(self):
        """
        Test with camera looking north (pan=0) at 45 degree tilt.

        At 45 degrees, ground distance equals camera height.
        """
        width, height = 1920, 1080
        pos = np.array([0.0, 0.0, 10.0])  # 10m high
        pan_deg = 0.0
        tilt_deg = 45.0
        zoom = 1.0

        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

        # Verify shape
        assert geo.H.shape == (3, 3)

        # Verify normalization
        assert abs(geo.H[2, 2] - 1.0) < 1e-10

        # Point directly in front at ground distance = height = 10m
        world_point = np.array([0.0, 10.0, 0.0])

        # Compute expected via 3D projection
        R = compute_rotation_matrix(pan_deg, tilt_deg)
        expected = compute_expected_projection_3d(K, R, pos, world_point)
        assert expected is not None, "Point should be visible"

        # Compare with homography
        actual = project_via_homography(geo.H, world_point[:2])

        pixel_diff = math.sqrt((actual[0] - expected[0])**2 + (actual[1] - expected[1])**2)
        assert pixel_diff < 1e-6

    def test_camera_looking_east_steep_tilt(self):
        """
        Test with camera looking east (pan=90) at steep 60 degree tilt.
        """
        width, height = 1280, 720
        pos = np.array([5.0, 5.0, 15.0])  # 15m high
        pan_deg = 90.0  # Looking east
        tilt_deg = 60.0
        zoom = 2.0

        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom, width, height)
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

        # Verify shape
        assert geo.H.shape == (3, 3)

        # Point to the east of camera
        ground_dist = 15.0 / math.tan(math.radians(60.0))
        world_point = np.array([5.0 + ground_dist, 5.0, 0.0])

        R = compute_rotation_matrix(pan_deg, tilt_deg)
        expected = compute_expected_projection_3d(K, R, pos, world_point)
        assert expected is not None

        actual = project_via_homography(geo.H, world_point[:2])

        pixel_diff = math.sqrt((actual[0] - expected[0])**2 + (actual[1] - expected[1])**2)
        assert pixel_diff < 1e-6

    def test_multiple_ground_points_various_configurations(self):
        """
        Test multiple ground points with various camera configurations.

        This is a comprehensive deterministic test covering different
        heights, pan angles, tilt angles, and ground point positions.
        """
        configurations = [
            # (pos, pan_deg, tilt_deg, zoom)
            (np.array([0, 0, 5]), 0, 30, 1.0),    # Low camera, shallow tilt
            (np.array([0, 0, 20]), 0, 60, 1.0),   # High camera, steep tilt
            (np.array([10, 10, 8]), 45, 45, 2.0), # Offset position, diagonal
            (np.array([-5, 5, 12]), -90, 50, 1.5), # Looking west
            (np.array([0, 0, 10]), 180, 40, 3.0), # Looking south
        ]

        for pos, pan_deg, tilt_deg, zoom in configurations:
            width, height = 1920, 1080

            geo = CameraGeometry(w=width, h=height)
            K = CameraGeometry.get_intrinsics(zoom, width, height)
            geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)

            ieh = IntrinsicExtrinsicHomography(width=width, height=height)
            H_ieh = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)

            # Verify both are 3x3
            assert geo.H.shape == (3, 3)
            assert H_ieh.shape == (3, 3)

            # Verify they match
            assert np.allclose(geo.H, H_ieh, atol=1e-6)

            # Test projection consistency
            R = compute_rotation_matrix(pan_deg, tilt_deg)

            # Generate ground points
            pan_rad = math.radians(pan_deg)
            tilt_rad = math.radians(tilt_deg)
            ground_dist = pos[2] / math.tan(tilt_rad) if tilt_rad > 0.1 else pos[2] * 5

            ground_points = [
                np.array([pos[0] + ground_dist * math.sin(pan_rad),
                          pos[1] + ground_dist * math.cos(pan_rad), 0.0]),
                np.array([pos[0] + 0.5 * ground_dist * math.sin(pan_rad),
                          pos[1] + 0.5 * ground_dist * math.cos(pan_rad), 0.0]),
            ]

            for wp in ground_points:
                expected = compute_expected_projection_3d(K, R, pos, wp)
                if expected is None:
                    continue

                actual = project_via_homography(geo.H, wp[:2])
                pixel_diff = math.sqrt((actual[0] - expected[0])**2 + (actual[1] - expected[1])**2)

                assert pixel_diff < 1e-6, (
                    f"Config: pos={pos}, pan={pan_deg}, tilt={tilt_deg}\n"
                    f"Point: {wp}, diff: {pixel_diff:.2e}"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
