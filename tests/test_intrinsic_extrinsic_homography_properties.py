#!/usr/bin/env python3
"""
Property-based tests for IntrinsicExtrinsicHomography using Hypothesis.

These tests verify mathematical invariants and consistency properties that must
hold across a wide range of input parameters. Property-based testing with
Hypothesis explores the input space more thoroughly than example-based tests,
catching edge cases and validating fundamental mathematical relationships.

Run with: python -m pytest tests/test_intrinsic_extrinsic_homography_properties.py -v
"""

import numpy as np
import math
import sys
import os
import pytest
from hypothesis import given, strategies as st, assume

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


# ============================================================================
# Hypothesis Strategies for Camera Parameters
# ============================================================================

@st.composite
def camera_dimensions(draw):
    """
    Generate valid camera image dimensions.

    Returns:
        Tuple[int, int]: (width, height) in pixels
    """
    # Common camera resolutions
    width = draw(st.integers(min_value=640, max_value=3840))
    height = draw(st.integers(min_value=480, max_value=2160))
    return width, height


@st.composite
def zoom_factor(draw):
    """
    Generate valid zoom factor values.

    Zoom range: [1.0, 25.0] based on Hikvision DS-2DF8425IX-AELW specs

    Returns:
        float: zoom factor
    """
    return draw(st.floats(min_value=1.0, max_value=25.0))


@st.composite
def pan_angle(draw):
    """
    Generate pan angle in degrees.

    Pan can be any angle (wraps around 360 degrees).

    Returns:
        float: pan angle in degrees
    """
    return draw(st.floats(min_value=-180.0, max_value=360.0))


@st.composite
def tilt_angle(draw):
    """
    Generate valid tilt angle in degrees.

    Tilt must be in range (0, 90] for ground plane projection.
    Avoiding very small angles near 0 (near-horizontal) which cause
    numerical instability.

    Returns:
        float: tilt angle in degrees
    """
    # Avoid extreme values near 0 and 90 to prevent numerical issues
    return draw(st.floats(min_value=5.0, max_value=85.0))


@st.composite
def camera_height(draw):
    """
    Generate realistic camera height in meters.

    Typical range: [2.0, 30.0] meters for PTZ security cameras

    Returns:
        float: camera height in meters
    """
    return draw(st.floats(min_value=2.0, max_value=30.0))


@st.composite
def camera_position(draw):
    """
    Generate camera position in world coordinates.

    Position is relative to arbitrary world origin.
    X (East): [-100, 100] meters
    Y (North): [-100, 100] meters
    Z (Up/Height): [2.0, 30.0] meters

    Returns:
        np.ndarray: [X, Y, Z] position in meters
    """
    x = draw(st.floats(min_value=-100.0, max_value=100.0))
    y = draw(st.floats(min_value=-100.0, max_value=100.0))
    z = draw(camera_height())
    return np.array([x, y, z])


@st.composite
def sensor_width(draw):
    """
    Generate sensor width in millimeters.

    Typical range: [3.0, 10.0] mm for PTZ cameras

    Returns:
        float: sensor width in millimeters
    """
    return draw(st.floats(min_value=3.0, max_value=10.0))


@st.composite
def base_focal_length(draw):
    """
    Generate base focal length in millimeters.

    Typical range: [3.0, 10.0] mm at 1x zoom

    Returns:
        float: base focal length in millimeters
    """
    return draw(st.floats(min_value=3.0, max_value=10.0))


# ============================================================================
# Property Tests
# ============================================================================

class TestRotationMatrixProperties:
    """
    Property-based tests for rotation matrix calculation consistency.

    The rotation matrix is a fundamental component shared between
    CameraGeometry and IntrinsicExtrinsicHomography. These properties
    verify that both implementations produce identical results.
    """

    @given(
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        dimensions=camera_dimensions()
    )
    def test_rotation_matrix_consistency(self, pan_deg, tilt_deg, dimensions):
        """
        Property: Rotation matrices must be identical between implementations.

        WHY THIS PROPERTY MUST HOLD:
        Both CameraGeometry and IntrinsicExtrinsicHomography compute rotation
        matrices from pan/tilt angles using the same mathematical formulas.
        Any difference indicates:
        1. Implementation bug in one class
        2. Different coordinate system assumptions
        3. Numerical precision issues

        The rotation matrix defines the camera's orientation in world space
        and is critical for correct homography calculation. Inconsistency
        would lead to different projection results between the two approaches.

        Mathematical basis:
        R = Rx_tilt @ R_base @ Rz_pan
        where:
        - Rz_pan: Rotation around world Z-axis (yaw)
        - R_base: Base transform from world to camera coordinates
        - Rx_tilt: Rotation around camera X-axis (pitch)
        """
        width, height = dimensions

        # Create instances
        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        # Set parameters for CameraGeometry
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg

        # Get rotation matrices
        R_geo = geo._get_rotation_matrix()
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg)

        # Verify matrices are identical (within floating-point tolerance)
        assert np.allclose(R_geo, R_ieh, rtol=1e-10, atol=1e-12), (
            f"Rotation matrices differ at pan={pan_deg:.2f}, tilt={tilt_deg:.2f}\n"
            f"Max difference: {np.max(np.abs(R_geo - R_ieh)):.2e}\n"
            f"CameraGeometry:\n{R_geo}\n"
            f"IntrinsicExtrinsic:\n{R_ieh}"
        )

    @given(
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        dimensions=camera_dimensions()
    )
    def test_rotation_matrix_orthogonality(self, pan_deg, tilt_deg, dimensions):
        """
        Property: Rotation matrices must be orthogonal (R @ R.T = I, det(R) = 1).

        WHY THIS PROPERTY MUST HOLD:
        A rotation matrix represents a pure rotation transformation without
        scaling, shearing, or reflection. Mathematically, rotation matrices
        form the Special Orthogonal group SO(3), which requires:
        1. Orthogonality: R @ R^T = I (columns are orthonormal basis vectors)
        2. Proper rotation: det(R) = +1 (preserves orientation, no reflection)

        Violation indicates:
        1. Implementation error in rotation matrix calculation
        2. Numerical precision degradation
        3. Invalid transformation (e.g., reflection instead of rotation)

        Physical meaning:
        - Orthogonality preserves distances and angles
        - det(R) = 1 ensures right-handed coordinate system
        """
        width, height = dimensions

        # Test both implementations
        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg

        R_geo = geo._get_rotation_matrix()
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg)

        for name, R in [("CameraGeometry", R_geo), ("IntrinsicExtrinsic", R_ieh)]:
            # Check determinant is 1
            det = np.linalg.det(R)
            assert np.isclose(det, 1.0, rtol=1e-10, atol=1e-12), (
                f"{name}: det(R) = {det:.10f}, expected 1.0 "
                f"at pan={pan_deg:.2f}, tilt={tilt_deg:.2f}"
            )

            # Check R @ R.T = I
            RRT = R @ R.T
            identity = np.eye(3)
            max_diff = np.max(np.abs(RRT - identity))
            assert max_diff < 1e-10, (
                f"{name}: R @ R.T differs from identity by {max_diff:.2e} "
                f"at pan={pan_deg:.2f}, tilt={tilt_deg:.2f}\n"
                f"R @ R.T:\n{RRT}"
            )


class TestFocalLengthProperties:
    """
    Property-based tests for focal length calculation.

    The focal length calculation is fundamental to the camera intrinsic matrix
    and must follow a precise linear relationship with zoom factor.
    """

    @given(
        zoom1=zoom_factor(),
        zoom2=zoom_factor(),
        dimensions=camera_dimensions(),
        sensor_w=sensor_width(),
        base_f=base_focal_length()
    )
    def test_focal_length_linearity(self, zoom1, zoom2, dimensions, sensor_w, base_f):
        """
        Property: Focal length must scale linearly with zoom factor.

        WHY THIS PROPERTY MUST HOLD:
        The relationship f_px = base_focal_length_mm * zoom_factor * (width / sensor_width_mm)
        is derived from the pinhole camera model and physical optics. This linear
        relationship holds for:
        1. Optical zoom: Physical lens movement changes focal length linearly
        2. Digital zoom: Effective focal length scales linearly with crop factor

        Mathematically:
        f_px(z) = k * z  where k = base_f_mm * (width_px / sensor_width_mm)

        Therefore:
        f_px(z2) / f_px(z1) = z2 / z1  (exact ratio relationship)

        Violation would indicate:
        1. Non-linear zoom mapping (incorrect physical model)
        2. Arithmetic error in get_intrinsics()
        3. Rounding or quantization issues

        This property is critical because zoom changes are frequent in PTZ
        cameras, and incorrect focal length leads to distorted projections.
        """
        # Skip if zoom values are too close (division by small number)
        assume(abs(zoom2 - zoom1) > 0.1)

        width, height = dimensions
        ieh = IntrinsicExtrinsicHomography(
            width=width,
            height=height,
            sensor_width_mm=sensor_w,
            base_focal_length_mm=base_f
        )

        # Get intrinsic matrices at two different zoom levels
        K1 = ieh.get_intrinsics(zoom_factor=zoom1)
        K2 = ieh.get_intrinsics(zoom_factor=zoom2)

        # Extract focal lengths (K[0,0] = fx)
        f_px_1 = K1[0, 0]
        f_px_2 = K2[0, 0]

        # Verify linear scaling: f2/f1 should equal z2/z1
        focal_ratio = f_px_2 / f_px_1
        zoom_ratio = zoom2 / zoom1

        # Use relative tolerance since we're comparing ratios
        assert np.isclose(focal_ratio, zoom_ratio, rtol=1e-10, atol=1e-12), (
            f"Focal length does not scale linearly with zoom\n"
            f"zoom1={zoom1:.4f}, zoom2={zoom2:.4f}, zoom_ratio={zoom_ratio:.6f}\n"
            f"f_px1={f_px_1:.2f}, f_px2={f_px_2:.2f}, focal_ratio={focal_ratio:.6f}\n"
            f"Difference: {abs(focal_ratio - zoom_ratio):.2e}"
        )

    @given(
        zoom=zoom_factor(),
        dimensions=camera_dimensions(),
        sensor_w=sensor_width(),
        base_f=base_focal_length()
    )
    def test_focal_length_formula(self, zoom, dimensions, sensor_w, base_f):
        """
        Property: Focal length must exactly match the formula.

        WHY THIS PROPERTY MUST HOLD:
        The focal length conversion from millimeters to pixels is defined by:
        f_px = f_mm * (width_px / sensor_width_mm)

        where f_mm = base_focal_length_mm * zoom_factor

        This formula converts physical focal length to pixel units using the
        sensor's physical-to-pixel scale factor. This is a fundamental
        relationship in computer vision that connects:
        1. Physical lens properties (f_mm)
        2. Sensor dimensions (sensor_width_mm)
        3. Image resolution (width_px)
        4. Pixel focal length (f_px) used in projection math

        This exact formula must hold because:
        - It defines the projection matrix K used in all transformations
        - Any deviation causes systematic projection errors
        - The relationship is geometric, not empirical

        Violation would indicate a fundamental implementation error.
        """
        width, height = dimensions
        ieh = IntrinsicExtrinsicHomography(
            width=width,
            height=height,
            sensor_width_mm=sensor_w,
            base_focal_length_mm=base_f
        )

        K = ieh.get_intrinsics(zoom_factor=zoom)
        f_px_actual = K[0, 0]

        # Calculate expected focal length using formula
        f_mm = base_f * zoom
        f_px_expected = f_mm * (width / sensor_w)

        # Verify exact match (within floating-point precision)
        assert np.isclose(f_px_actual, f_px_expected, rtol=1e-12, atol=1e-10), (
            f"Focal length does not match formula\n"
            f"zoom={zoom:.4f}, base_f={base_f:.2f}mm, sensor_w={sensor_w:.2f}mm, width={width}px\n"
            f"f_mm = {f_mm:.4f}mm\n"
            f"f_px_expected = {f_px_expected:.2f}px\n"
            f"f_px_actual = {f_px_actual:.2f}px\n"
            f"Difference: {abs(f_px_actual - f_px_expected):.2e}px"
        )


class TestConfidenceProperties:
    """
    Property-based tests for confidence calculation.

    Confidence values represent the reliability of homography transformations
    and must satisfy mathematical bounds and consistency properties.
    """

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    def test_confidence_bounds(self, pos, pan_deg, tilt_deg, zoom, dimensions):
        """
        Property: Confidence values must be bounded in [0.0, 1.0].

        WHY THIS PROPERTY MUST HOLD:
        Confidence is defined as a probability-like score indicating the
        reliability of the homography transformation. By convention:
        - 0.0 = completely unreliable (singular/degenerate homography)
        - 1.0 = maximum reliability (well-conditioned homography)

        Values outside [0.0, 1.0] are meaningless because:
        1. Confidence > 1.0 has no interpretation (probability-like values max at 1)
        2. Confidence < 0.0 has no interpretation (reliability cannot be negative)

        This bound must hold for ALL valid camera parameters because:
        - Users rely on confidence to decide whether to trust projections
        - Downstream systems may use confidence for weighted averaging
        - Out-of-bounds values indicate implementation bugs

        The confidence calculation uses determinant magnitude, condition
        number, and parameter validity checks, all normalized to [0.0, 1.0].
        """
        width, height = dimensions
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        K = CameraGeometry.get_intrinsics(zoom, width, height)

        # Compute homography
        result = ieh.compute_homography(
            frame=np.zeros((height, width, 3), dtype=np.uint8),  # Dummy frame
            reference={
                'camera_matrix': K,
                'camera_position': pos,
                'pan_deg': pan_deg,
                'tilt_deg': tilt_deg,
                'map_width': 640,
                'map_height': 640
            }
        )

        confidence = result.confidence

        # Verify confidence is in [0.0, 1.0]
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence:.4f} is outside valid range [0.0, 1.0]\n"
            f"Parameters: pan={pan_deg:.2f}, tilt={tilt_deg:.2f}, "
            f"height={pos[2]:.2f}m, zoom={zoom:.2f}"
        )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    def test_point_confidence_bounds(self, pos, pan_deg, tilt_deg, zoom, dimensions):
        """
        Property: Point-specific confidence must also be bounded in [0.0, 1.0].

        WHY THIS PROPERTY MUST HOLD:
        Point confidence adjusts the base homography confidence based on the
        image point's location. Points near image edges have lower confidence
        due to:
        1. Lens distortion effects (stronger at edges)
        2. Perspective distortion (increases with distance from center)
        3. Reduced accuracy of calibration at edges

        The adjustment multiplies base confidence by an edge factor in [0.3, 1.0],
        which should preserve the [0.0, 1.0] bound:
        - base_confidence ∈ [0.0, 1.0]
        - edge_factor ∈ [0.3, 1.0]
        - point_confidence = base_confidence * edge_factor ∈ [0.0, 1.0]

        Violation would indicate:
        1. Edge factor calculation error
        2. Incorrect bounds in edge factor constants
        3. Arithmetic overflow/underflow
        """
        width, height = dimensions

        # Skip very small images where edge detection might be problematic
        assume(width >= 100 and height >= 100)

        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        K = CameraGeometry.get_intrinsics(zoom, width, height)

        # Compute homography
        result = ieh.compute_homography(
            frame=np.zeros((height, width, 3), dtype=np.uint8),
            reference={
                'camera_matrix': K,
                'camera_position': pos,
                'pan_deg': pan_deg,
                'tilt_deg': tilt_deg,
                'map_width': 640,
                'map_height': 640
            }
        )

        # Test points at various locations (center, edges, corners)
        test_points = [
            (width / 2.0, height / 2.0),  # Center
            (0.0, 0.0),                    # Top-left corner
            (width - 1.0, 0.0),            # Top-right corner
            (0.0, height - 1.0),           # Bottom-left corner
            (width - 1.0, height - 1.0),   # Bottom-right corner
            (width / 2.0, 0.0),            # Top edge center
            (width / 2.0, height - 1.0),   # Bottom edge center
            (0.0, height / 2.0),           # Left edge center
            (width - 1.0, height / 2.0),   # Right edge center
        ]

        for u, v in test_points:
            point_confidence = ieh._calculate_point_confidence(
                (u, v),
                result.confidence
            )

            assert 0.0 <= point_confidence <= 1.0, (
                f"Point confidence {point_confidence:.4f} is outside [0.0, 1.0]\n"
                f"Point: ({u:.1f}, {v:.1f}), Base confidence: {result.confidence:.4f}\n"
                f"Image dimensions: {width}x{height}"
            )

            # Additional check: point confidence should not exceed base confidence
            # (edge factor can only reduce confidence, not increase it)
            assert point_confidence <= result.confidence + 1e-10, (
                f"Point confidence {point_confidence:.4f} exceeds base confidence {result.confidence:.4f}\n"
                f"Point: ({u:.1f}, {v:.1f})"
            )


class TestHomographyConsistencyProperties:
    """
    Property-based tests for homography matrix consistency.

    Both CameraGeometry and IntrinsicExtrinsicHomography compute homography
    matrices from the same inputs. These must produce identical results.
    """

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    def test_homography_matrix_consistency(self, pos, pan_deg, tilt_deg, zoom, dimensions):
        """
        Property: Homography matrices must be identical between implementations.

        WHY THIS PROPERTY MUST HOLD:
        Both CameraGeometry and IntrinsicExtrinsicHomography implement the
        same mathematical transformation:
        H = K @ [r1, r2, t]
        where:
        - K is the camera intrinsic matrix
        - r1, r2 are first two columns of rotation matrix R
        - t is translation vector: t = -R @ camera_position

        This formula is derived from projective geometry and defines the
        mapping from world ground plane (Z=0) to image pixels.

        Both implementations must produce identical results because:
        1. They use the same rotation matrix formula (verified by other tests)
        2. They use the same intrinsic matrix K
        3. They use the same camera position and pan/tilt parameters
        4. The homography formula is deterministic

        Any difference indicates:
        1. Implementation bug in one class
        2. Different normalization conventions
        3. Numerical precision issues

        Consistency is critical because:
        - Users may switch between approaches
        - Results must be interchangeable
        - Projections must match for same camera state
        """
        width, height = dimensions

        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        K = CameraGeometry.get_intrinsics(zoom, width, height)

        # CameraGeometry approach
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)
        H_geo = geo.H

        # IntrinsicExtrinsicHomography approach
        H_ieh = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)
        # Normalize to match CameraGeometry convention (H[2,2] = 1)
        H_ieh_norm = H_ieh / H_ieh[2, 2]

        # Verify matrices match
        max_diff = np.max(np.abs(H_geo - H_ieh_norm))

        # Use relative tolerance for large matrix values
        assert np.allclose(H_geo, H_ieh_norm, rtol=1e-9, atol=1e-10), (
            f"Homography matrices differ\n"
            f"Max difference: {max_diff:.2e}\n"
            f"Parameters: pan={pan_deg:.2f}, tilt={tilt_deg:.2f}, "
            f"height={pos[2]:.2f}m, zoom={zoom:.2f}\n"
            f"CameraGeometry H:\n{H_geo}\n"
            f"IntrinsicExtrinsic H:\n{H_ieh_norm}"
        )

    @given(
        pos=camera_position(),
        pan_deg=pan_angle(),
        tilt_deg=tilt_angle(),
        zoom=zoom_factor(),
        dimensions=camera_dimensions()
    )
    def test_projection_consistency(self, pos, pan_deg, tilt_deg, zoom, dimensions):
        """
        Property: World point projections must be identical between implementations.

        WHY THIS PROPERTY MUST HOLD:
        If homography matrices are identical (verified by other test), then
        projecting the same world point must yield identical image coordinates.

        Mathematical verification:
        Given world point P_world = [X, Y, 1]^T and homography H,
        the image point is: p_image = H @ P_world

        After normalization: [u, v] = [p[0]/p[2], p[1]/p[2]]

        This is a direct test of the end-to-end projection pipeline.
        Consistency proves that:
        1. Homography matrices are truly equivalent
        2. Normalization conventions match
        3. No implementation-specific quirks affect results

        We test with multiple world points to cover different projection angles
        and distances from the camera.
        """
        width, height = dimensions

        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        K = CameraGeometry.get_intrinsics(zoom, width, height)

        # Setup both implementations
        geo.set_camera_parameters(K, pos, pan_deg, tilt_deg, 640, 640)
        H_ieh = ieh._calculate_ground_homography(K, pos, pan_deg, tilt_deg)

        # Test several world points at different positions
        world_points = [
            (5.0, 5.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (-5.0, 8.0),
            (3.0, -3.0)
        ]

        for X, Y in world_points:
            # Project with CameraGeometry
            pt = np.array([[X], [Y], [1.0]])
            p_geo = geo.H @ pt
            u_geo = p_geo[0, 0] / p_geo[2, 0]
            v_geo = p_geo[1, 0] / p_geo[2, 0]

            # Project with IntrinsicExtrinsicHomography
            p_ieh = H_ieh @ pt
            u_ieh = p_ieh[0, 0] / p_ieh[2, 0]
            v_ieh = p_ieh[1, 0] / p_ieh[2, 0]

            # Calculate pixel difference
            pixel_diff = math.sqrt((u_geo - u_ieh)**2 + (v_geo - v_ieh)**2)

            # Projections should match within sub-pixel accuracy
            assert pixel_diff < 0.01, (
                f"Projection differs by {pixel_diff:.4f} pixels for world point ({X}, {Y})\n"
                f"CameraGeometry: ({u_geo:.2f}, {v_geo:.2f})\n"
                f"IntrinsicExtrinsic: ({u_ieh:.2f}, {v_ieh:.2f})\n"
                f"Parameters: pan={pan_deg:.2f}, tilt={tilt_deg:.2f}, "
                f"height={pos[2]:.2f}m, zoom={zoom:.2f}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
