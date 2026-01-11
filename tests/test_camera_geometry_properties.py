#!/usr/bin/env python3
"""
Property-based tests for CameraGeometry using Hypothesis.

These tests verify mathematical invariants that MUST hold for all valid inputs:
1. Rotation matrices are orthogonal (R @ R.T = I, det(R) = 1)
2. Homographies are invertible (det(H) != 0, bounded condition number)
3. Projection round-trip consistency (pixel -> world -> pixel)
4. Pan/tilt independence (orthogonal parameter effects)

Property-based testing generates hundreds of random valid inputs to find edge cases
that example-based tests might miss.

Run with: python -m pytest tests/test_camera_geometry_properties.py -v
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import assume, given
from hypothesis import strategies as st

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import CameraParameters
from poc_homography.types import Degrees, Pixels, Unitless

# ============================================================================
# Hypothesis Strategies for Valid Camera Parameters
# ============================================================================


def valid_pan_angle():
    """
    Generate valid pan angles in degrees.

    Pan angle controls horizontal camera rotation (azimuth).
    Valid range: [-180, 180] degrees (full rotation).
    """
    return st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


def valid_tilt_angle():
    """
    Generate valid tilt angles in degrees.

    Tilt angle controls vertical camera rotation (elevation).
    Valid range: (0, 90] degrees (must point downward for ground plane projection).

    We avoid very small tilts (<0.1 deg) and near-horizontal angles (>89.9 deg)
    to prevent numerical instability in homography computation.
    """
    return st.floats(min_value=0.1, max_value=89.9, allow_nan=False, allow_infinity=False)


def valid_camera_height():
    """
    Generate valid camera heights in meters.

    Valid range: [1.0, 50.0] meters (from CameraGeometry validation constants).
    Typical PTZ cameras: 2-30 meters.
    """
    return st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def valid_zoom_factor():
    """
    Generate valid zoom factors.

    Valid range: [1.0, 25.0] (from CameraGeometry.ZOOM_MIN/MAX).
    1.0 = no zoom (widest field of view).
    25.0 = maximum zoom (narrowest field of view).
    """
    return st.floats(min_value=1.0, max_value=25.0, allow_nan=False, allow_infinity=False)


def valid_image_dimensions():
    """
    Generate valid image dimensions (width, height) in pixels.

    Common resolutions: 1920x1080 (Full HD), 1280x720 (HD), 640x480 (VGA).
    """
    return st.sampled_from(
        [
            (1920, 1080),  # Full HD
            (1280, 720),  # HD
            (640, 480),  # VGA
            (1024, 768),  # XGA
        ]
    )


# ============================================================================
# Property 1: Rotation Matrix Orthogonality
# ============================================================================


@given(pan=valid_pan_angle(), tilt=valid_tilt_angle())
def test_rotation_matrix_orthogonality(pan, tilt):
    """
    PROPERTY: Rotation matrices MUST be orthogonal.

    Mathematical invariant:
        R @ R.T = I  (rotation preserves lengths and angles)
        det(R) = 1   (proper rotation, not reflection)

    WHY THIS MATTERS:
    - Orthogonal matrices preserve distances and angles in 3D space
    - Non-orthogonal R would distort world coordinates incorrectly
    - This is a fundamental requirement of rigid body transformations

    VERIFICATION:
    - Compute R @ R.T and verify it equals identity matrix (within numerical tolerance)
    - Compute det(R) and verify it equals 1.0 (within numerical tolerance)

    TOLERANCES:
    - rtol=1e-9, atol=1e-12 for floating-point comparisons
    - Stricter than typical because this is a fundamental mathematical property
    """
    R = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan), tilt_deg=Degrees(tilt), roll_deg=Degrees(0.0)
    )

    # Property 1a: R @ R.T = I (orthogonality)
    RRT = R @ R.T
    identity = np.eye(3)

    assert np.allclose(RRT, identity, rtol=1e-9, atol=1e-12), (
        f"Rotation matrix not orthogonal at pan={pan:.2f} deg, tilt={tilt:.2f} deg\n"
        f"R @ R.T =\n{RRT}\n"
        f"Expected identity:\n{identity}\n"
        f"Max difference: {np.max(np.abs(RRT - identity))}"
    )

    # Property 1b: det(R) = 1.0 (proper rotation)
    det_R = np.linalg.det(R)

    assert np.isclose(det_R, 1.0, rtol=1e-9, atol=1e-12), (
        f"Rotation matrix determinant not 1.0 at pan={pan:.2f} deg, tilt={tilt:.2f} deg\n"
        f"det(R) = {det_R}, expected 1.0\n"
        f"Difference: {abs(det_R - 1.0)}"
    )


# ============================================================================
# Property 2: Homography Invertibility
# ============================================================================


@given(
    pan=valid_pan_angle(),
    tilt=valid_tilt_angle(),
    height=valid_camera_height(),
    zoom=valid_zoom_factor(),
    dims=valid_image_dimensions(),
)
def test_homography_invertibility(pan, tilt, height, zoom, dims):
    """
    PROPERTY: Homography matrices MUST be invertible with bounded condition number.

    Mathematical invariants:
        det(H) != 0  (non-singular matrix)
        cond(H) < 1e10  (numerically stable inverse)

    WHY THIS MATTERS:
    - Homography H maps world coords -> image pixels
    - Inverse H_inv maps image pixels -> world coords
    - Singular H (det=0) means mapping is degenerate and cannot be inverted
    - High condition number means inverse is numerically unstable (small input errors -> large output errors)

    VERIFICATION:
    - Compute det(H) and verify it's not close to zero (|det| > 1e-10)
    - Compute condition number cond(H) and verify it's bounded (< 1e10)
    - These bounds are from CameraGeometry.CONDITION_ERROR threshold

    EDGE CASES HANDLED:
    - High zoom (narrow FOV) -> may increase condition number
    - Low tilt (near horizontal) -> may increase condition number
    - Extreme height ratios -> validated by CameraGeometry height constraints
    """
    width, height_px = dims

    K = CameraGeometry.get_intrinsics(Unitless(zoom), Pixels(width), Pixels(height_px))
    w_pos = np.array([0.0, 0.0, height])

    params = CameraParameters.create(
        image_width=Pixels(width),
        image_height=Pixels(height_px),
        intrinsic_matrix=K,
        camera_position=w_pos,
        pan_deg=Degrees(pan),
        tilt_deg=Degrees(tilt),
        roll_deg=Degrees(0.0),
        map_width=Pixels(640),
        map_height=Pixels(640),
        pixels_per_meter=Unitless(100.0),
    )

    result = CameraGeometry.compute(params)
    H = result.homography_matrix

    # Property 2a: det(H) != 0 (invertible)
    det_H = np.linalg.det(H)

    assert abs(det_H) > 1e-10, (
        f"Homography is singular (det ~ 0) at configuration:\n"
        f"  pan={pan:.2f} deg, tilt={tilt:.2f} deg, height={height:.2f}m, zoom={zoom:.2f}\n"
        f"  det(H) = {det_H:.2e}"
    )

    # Property 2b: Bounded condition number (numerically stable)
    cond_H = np.linalg.cond(H)

    assert cond_H < 1e10, (
        f"Homography condition number too high (numerically unstable) at configuration:\n"
        f"  pan={pan:.2f} deg, tilt={tilt:.2f} deg, height={height:.2f}m, zoom={zoom:.2f}\n"
        f"  cond(H) = {cond_H:.2e}, threshold = 1e10"
    )


# ============================================================================
# Property 3: Projection Round-Trip Consistency
# ============================================================================


@given(
    pan=valid_pan_angle(),
    tilt=valid_tilt_angle(),
    height=valid_camera_height(),
    zoom=valid_zoom_factor(),
    dims=valid_image_dimensions(),
    # Generate random pixel within image bounds
    pixel_u_fraction=st.floats(min_value=0.1, max_value=0.9),
    pixel_v_fraction=st.floats(min_value=0.1, max_value=0.9),
)
def test_projection_round_trip_consistency(
    pan, tilt, height, zoom, dims, pixel_u_fraction, pixel_v_fraction
):
    """
    PROPERTY: Projection round-trip MUST return to original pixel within tolerance.

    Mathematical invariant:
        pixel -> world -> pixel = original pixel (within numerical tolerance)

    WHY THIS MATTERS:
    - H maps world -> image, H_inv maps image -> world
    - Round-trip tests that H and H_inv are true inverses
    - Failure indicates numerical instability or implementation bug
    - Critical for geo-localization accuracy (e.g., object tracking on map)

    VERIFICATION:
    1. Start with pixel (u, v) in image bounds
    2. Project to world: (X_w, Y_w) = H_inv @ [u, v, 1]
    3. Project back to image: (u', v') = H @ [X_w, Y_w, 1]
    4. Verify: |u' - u| < tolerance and |v' - v| < tolerance

    TOLERANCE:
    - pixel_tolerance = 0.5 pixels (sub-pixel accuracy)
    - This is stricter than typical calibration errors (~1-2 pixels)
    - Ensures numerical precision of homography computation

    EDGE CASES HANDLED:
    - Avoid image edges (use 0.1-0.9 of image width/height)
    - Edges may have higher projection error due to lens distortion
    """
    width, height_px = dims

    # Generate pixel coordinates within safe bounds (avoid edges)
    pixel_u = pixel_u_fraction * width
    pixel_v = pixel_v_fraction * height_px

    K = CameraGeometry.get_intrinsics(Unitless(zoom), Pixels(width), Pixels(height_px))
    w_pos = np.array([0.0, 0.0, height])

    params = CameraParameters.create(
        image_width=Pixels(width),
        image_height=Pixels(height_px),
        intrinsic_matrix=K,
        camera_position=w_pos,
        pan_deg=Degrees(pan),
        tilt_deg=Degrees(tilt),
        roll_deg=Degrees(0.0),
        map_width=Pixels(640),
        map_height=Pixels(640),
        pixels_per_meter=Unitless(100.0),
    )

    result = CameraGeometry.compute(params)

    # Forward projection: pixel -> world
    pixel_hom = np.array([[pixel_u], [pixel_v], [1.0]])
    world_hom = result.inverse_homography_matrix @ pixel_hom

    # Normalize world coordinates
    w_scale = world_hom[2, 0]

    # Skip if point is near horizon (w_scale ~ 0)
    assume(abs(w_scale) > 1e-6)

    world_x = world_hom[0, 0] / w_scale
    world_y = world_hom[1, 0] / w_scale

    # Backward projection: world -> pixel
    world_hom_back = np.array([[world_x], [world_y], [1.0]])
    pixel_hom_back = result.homography_matrix @ world_hom_back

    # Normalize pixel coordinates
    pixel_u_back = pixel_hom_back[0, 0] / pixel_hom_back[2, 0]
    pixel_v_back = pixel_hom_back[1, 0] / pixel_hom_back[2, 0]

    # Verify round-trip consistency
    pixel_error_u = abs(pixel_u_back - pixel_u)
    pixel_error_v = abs(pixel_v_back - pixel_v)
    pixel_error_total = math.sqrt(pixel_error_u**2 + pixel_error_v**2)

    pixel_tolerance = 0.5  # Sub-pixel accuracy

    assert pixel_error_total < pixel_tolerance, (
        f"Round-trip projection error too large at configuration:\n"
        f"  pan={pan:.2f} deg, tilt={tilt:.2f} deg, height={height:.2f}m, zoom={zoom:.2f}\n"
        f"  Original pixel: ({pixel_u:.2f}, {pixel_v:.2f})\n"
        f"  World coords: ({world_x:.2f}, {world_y:.2f}) meters\n"
        f"  Back-projected pixel: ({pixel_u_back:.2f}, {pixel_v_back:.2f})\n"
        f"  Error: {pixel_error_total:.4f} pixels (tolerance: {pixel_tolerance} pixels)"
    )


# ============================================================================
# Property 4: Pan/Tilt Independence
# ============================================================================


@given(
    pan1=valid_pan_angle(),
    pan2=valid_pan_angle(),
    tilt_fixed=valid_tilt_angle(),
    height=valid_camera_height(),
)
def test_pan_tilt_independence_pan_only(pan1, pan2, tilt_fixed, height):
    """
    PROPERTY: Changing only PAN does not affect ELEVATION component.

    Mathematical invariant:
        For fixed tilt, camera elevation = arcsin(-forward_z) is constant for all pan angles
        where forward = R.T @ [0, 0, 1] (camera Z-axis in world coords)

    WHY THIS MATTERS:
    - Pan controls horizontal rotation (azimuth), tilt controls vertical rotation (elevation)
    - These should be orthogonal (independent) parameters
    - Coupled pan/tilt means user can't control viewing direction independently
    - Critical for PTZ camera control (users expect independent pan/tilt)

    VERIFICATION:
    1. Compute rotation matrix R at (pan1, tilt_fixed)
    2. Extract camera forward direction in world: forward1 = R.T @ [0, 0, 1]
    3. Compute elevation: elev1 = arcsin(-forward_z)
    4. Repeat for (pan2, tilt_fixed)
    5. Verify: |elev1 - elev2| < tolerance

    TOLERANCE:
    - elevation_tolerance = 0.1 degrees
    - Accounts for floating-point errors in trigonometric calculations

    PHYSICAL INTERPRETATION:
    - Pan rotates camera left/right (changes azimuth)
    - Elevation (how high camera looks) should not change when panning
    """
    # Avoid testing pan angles that are too close (no meaningful difference)
    assume(abs(pan2 - pan1) > 1.0)

    # Configuration 1: pan1, tilt_fixed
    R1 = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan1), tilt_deg=Degrees(tilt_fixed), roll_deg=Degrees(0.0)
    )

    # Camera forward direction in world coordinates
    # Camera Z-axis [0, 0, 1] in camera frame -> world frame via R.T
    forward1 = R1.T @ np.array([0, 0, 1])

    # Elevation = arcsin(-forward_z) (negative because world +Z is up, camera forward has negative Z component)
    elevation1 = math.degrees(math.asin(-forward1[2]))

    # Configuration 2: pan2, tilt_fixed (only pan changed)
    R2 = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan2), tilt_deg=Degrees(tilt_fixed), roll_deg=Degrees(0.0)
    )

    forward2 = R2.T @ np.array([0, 0, 1])
    elevation2 = math.degrees(math.asin(-forward2[2]))

    # Verify elevation unchanged
    elevation_diff = abs(elevation2 - elevation1)
    elevation_tolerance = 0.1  # degrees

    assert elevation_diff < elevation_tolerance, (
        f"Pan change affected elevation (pan/tilt not independent):\n"
        f"  pan1={pan1:.2f} deg, pan2={pan2:.2f} deg, tilt={tilt_fixed:.2f} deg\n"
        f"  elevation1={elevation1:.2f} deg, elevation2={elevation2:.2f} deg\n"
        f"  Difference: {elevation_diff:.4f} deg (tolerance: {elevation_tolerance} deg)"
    )


@given(
    tilt1=valid_tilt_angle(),
    tilt2=valid_tilt_angle(),
    pan_fixed=valid_pan_angle(),
    height=valid_camera_height(),
)
def test_pan_tilt_independence_tilt_only(tilt1, tilt2, pan_fixed, height):
    """
    PROPERTY: Changing only TILT does not affect AZIMUTH component.

    Mathematical invariant:
        For fixed pan, camera azimuth = atan2(forward_x, forward_y) is constant for all tilt angles
        where forward = R.T @ [0, 0, 1] (camera Z-axis in world coords)

    WHY THIS MATTERS:
    - Tilt controls vertical rotation (elevation), pan controls horizontal rotation (azimuth)
    - These should be orthogonal (independent) parameters
    - Coupled pan/tilt means user can't control viewing direction independently

    VERIFICATION:
    1. Compute rotation matrix R at (pan_fixed, tilt1)
    2. Extract camera forward direction in world: forward1 = R.T @ [0, 0, 1]
    3. Compute azimuth: az1 = atan2(forward_x, forward_y)
    4. Repeat for (pan_fixed, tilt2)
    5. Verify: |az1 - az2| < tolerance

    TOLERANCE:
    - azimuth_tolerance = 0.1 degrees
    - Accounts for floating-point errors in trigonometric calculations

    PHYSICAL INTERPRETATION:
    - Tilt rotates camera up/down (changes elevation)
    - Azimuth (which compass direction camera looks) should not change when tilting
    """
    # Avoid testing tilt angles that are too close (no meaningful difference)
    assume(abs(tilt2 - tilt1) > 1.0)

    # Configuration 1: pan_fixed, tilt1
    R1 = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan_fixed), tilt_deg=Degrees(tilt1), roll_deg=Degrees(0.0)
    )

    # Camera forward direction in world coordinates
    forward1 = R1.T @ np.array([0, 0, 1])

    # Azimuth = atan2(East, North) = atan2(X, Y)
    azimuth1 = math.degrees(math.atan2(forward1[0], forward1[1]))

    # Configuration 2: pan_fixed, tilt2 (only tilt changed)
    R2 = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan_fixed), tilt_deg=Degrees(tilt2), roll_deg=Degrees(0.0)
    )

    forward2 = R2.T @ np.array([0, 0, 1])
    azimuth2 = math.degrees(math.atan2(forward2[0], forward2[1]))

    # Verify azimuth unchanged
    azimuth_diff = abs(azimuth2 - azimuth1)

    # Handle wrap-around at +/-180 deg
    if azimuth_diff > 180:
        azimuth_diff = 360 - azimuth_diff

    azimuth_tolerance = 0.1  # degrees

    assert azimuth_diff < azimuth_tolerance, (
        f"Tilt change affected azimuth (pan/tilt not independent):\n"
        f"  pan={pan_fixed:.2f} deg, tilt1={tilt1:.2f} deg, tilt2={tilt2:.2f} deg\n"
        f"  azimuth1={azimuth1:.2f} deg, azimuth2={azimuth2:.2f} deg\n"
        f"  Difference: {azimuth_diff:.4f} deg (tolerance: {azimuth_tolerance} deg)"
    )


# ============================================================================
# Property 5: Rotation Matrix Inverse Equals Transpose
# ============================================================================


@given(pan=valid_pan_angle(), tilt=valid_tilt_angle())
def test_rotation_matrix_inverse_equals_transpose(pan, tilt):
    """
    PROPERTY: For orthogonal matrices, inverse equals transpose (R^-1 = R.T).

    Mathematical invariant:
        R^-1 = R.T  (fundamental property of orthogonal matrices)

    WHY THIS MATTERS:
    - Computing inverse of rotation matrix is expensive (O(n^3))
    - Computing transpose is trivial (O(n^2))
    - If R is orthogonal, we can use transpose instead of inverse (huge performance gain)
    - This property is used throughout camera geometry for coordinate transformations

    VERIFICATION:
    - Compute R_inv = np.linalg.inv(R)
    - Compute R_transpose = R.T
    - Verify: R_inv ~ R_transpose (within numerical tolerance)

    TOLERANCES:
    - rtol=1e-9, atol=1e-12 for floating-point comparisons
    """
    R = CameraGeometry._get_rotation_matrix_static(
        pan_deg=Degrees(pan), tilt_deg=Degrees(tilt), roll_deg=Degrees(0.0)
    )

    # Compute inverse (expensive)
    R_inv = np.linalg.inv(R)

    # Compute transpose (cheap)
    R_transpose = R.T

    # Verify they're equal
    assert np.allclose(R_inv, R_transpose, rtol=1e-9, atol=1e-12), (
        f"Rotation matrix inverse does not equal transpose at pan={pan:.2f} deg, tilt={tilt:.2f} deg\n"
        f"R^-1 =\n{R_inv}\n"
        f"R.T =\n{R_transpose}\n"
        f"Max difference: {np.max(np.abs(R_inv - R_transpose))}"
    )


# ============================================================================
# Property 6: Homography Preserves Line Relationships
# ============================================================================


@given(
    pan=valid_pan_angle(),
    tilt=valid_tilt_angle(),
    height=valid_camera_height(),
    zoom=valid_zoom_factor(),
    dims=valid_image_dimensions(),
)
def test_homography_preserves_collinearity(pan, tilt, height, zoom, dims):
    """
    PROPERTY: Homography MUST preserve collinearity of points.

    Mathematical invariant:
        If points A, B, C are collinear in world, their projections a, b, c are collinear in image
        (and vice versa for inverse projection)

    WHY THIS MATTERS:
    - Homographies are projective transformations that preserve lines
    - Straight roads, building edges, etc. in world must appear straight in image
    - Violation indicates non-projective distortion (e.g., lens distortion not accounted for)

    VERIFICATION:
    1. Create three collinear points in world: A, B, C where B = alpha*A + (1-alpha)*C
    2. Project to image: a = H @ A, b = H @ B, c = H @ C
    3. Verify b lies on line segment ac (within numerical tolerance)

    TOLERANCE:
    - collinearity_tolerance = 0.5 pixels
    - Accounts for numerical errors in homography computation

    EDGE CASES HANDLED:
    - Use points within reasonable distance from camera (10-50m)
    - Avoid extreme angles that might project outside image bounds
    """
    width, height_px = dims

    K = CameraGeometry.get_intrinsics(Unitless(zoom), Pixels(width), Pixels(height_px))
    w_pos = np.array([0.0, 0.0, height])

    params = CameraParameters.create(
        image_width=Pixels(width),
        image_height=Pixels(height_px),
        intrinsic_matrix=K,
        camera_position=w_pos,
        pan_deg=Degrees(pan),
        tilt_deg=Degrees(tilt),
        roll_deg=Degrees(0.0),
        map_width=Pixels(640),
        map_height=Pixels(640),
        pixels_per_meter=Unitless(100.0),
    )

    result = CameraGeometry.compute(params)

    # Create three collinear points in world coordinates (along a line)
    # Point A: 10m in front of camera
    angle_rad = math.radians(pan)
    A_world = np.array([10.0 * math.sin(angle_rad), 10.0 * math.cos(angle_rad), 1.0])

    # Point C: 30m in front of camera (same direction)
    C_world = np.array([30.0 * math.sin(angle_rad), 30.0 * math.cos(angle_rad), 1.0])

    # Point B: midpoint between A and C (guaranteed collinear)
    B_world = (A_world + C_world) / 2.0

    # Project all three points to image
    a_img_hom = result.homography_matrix @ A_world.reshape(-1, 1)
    b_img_hom = result.homography_matrix @ B_world.reshape(-1, 1)
    c_img_hom = result.homography_matrix @ C_world.reshape(-1, 1)

    # Normalize to pixel coordinates
    a_img = np.array([a_img_hom[0, 0] / a_img_hom[2, 0], a_img_hom[1, 0] / a_img_hom[2, 0]])
    b_img = np.array([b_img_hom[0, 0] / b_img_hom[2, 0], b_img_hom[1, 0] / b_img_hom[2, 0]])
    c_img = np.array([c_img_hom[0, 0] / c_img_hom[2, 0], c_img_hom[1, 0] / c_img_hom[2, 0]])

    # Skip if any point projects outside image bounds (not visible)
    assume(0 <= a_img[0] <= width and 0 <= a_img[1] <= height_px)
    assume(0 <= b_img[0] <= width and 0 <= b_img[1] <= height_px)
    assume(0 <= c_img[0] <= width and 0 <= c_img[1] <= height_px)

    # Verify collinearity: distance from b to line ac should be zero
    # Line ac: parametric form p(t) = a + t(c - a)
    # Distance from b to line: ||(b - a) - proj_{(c-a)}(b - a)||

    ac = c_img - a_img
    ab = b_img - a_img

    # Project ab onto ac
    if np.linalg.norm(ac) > 1e-6:  # Avoid division by zero
        projection_length = np.dot(ab, ac) / np.linalg.norm(ac)
        projection = (projection_length / np.linalg.norm(ac)) * ac

        # Perpendicular distance from b to line ac
        perpendicular = ab - projection
        distance_to_line = np.linalg.norm(perpendicular)

        collinearity_tolerance = 0.5  # pixels

        assert distance_to_line < collinearity_tolerance, (
            f"Homography violated collinearity at configuration:\n"
            f"  pan={pan:.2f} deg, tilt={tilt:.2f} deg, height={height:.2f}m, zoom={zoom:.2f}\n"
            f"  Point a: ({a_img[0]:.2f}, {a_img[1]:.2f})\n"
            f"  Point b: ({b_img[0]:.2f}, {b_img[1]:.2f})\n"
            f"  Point c: ({c_img[0]:.2f}, {c_img[1]:.2f})\n"
            f"  Distance from b to line ac: {distance_to_line:.4f} pixels (tolerance: {collinearity_tolerance})"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
