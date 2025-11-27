#!/usr/bin/env python3
"""
Unit tests to verify lens distortion correction reduces projection error.

Tests verify that:
1. Distortion correction is applied correctly before projection
2. Edge points are affected more than center points (radial distortion behavior)
3. Zero distortion yields identical results to no distortion
4. Distortion parameters are stored and applied correctly
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("Warning: pytest not installed. Tests can still run standalone.")

import numpy as np
import cv2
import sys
import os

# Add parent directory to path to import camera_geometry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera_geometry import CameraGeometry


# Test fixtures and helper functions

def get_test_camera_params():
    """
    Returns realistic camera parameters for testing.

    Returns:
        dict: Camera parameters including K, w_pos, and image dimensions
    """
    W_px, H_px = 1920, 1080
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0, W_px=W_px, H_px=H_px)
    w_pos = np.array([0.0, 0.0, 5.0])  # 5m camera height

    return {
        'W_px': W_px,
        'H_px': H_px,
        'K': K,
        'w_pos': w_pos,
        'pan_deg': 0.0,
        'tilt_deg': -45.0,
        'map_width': 640,
        'map_height': 480
    }


def create_distorted_points(points, K, distortion_coeffs):
    """
    Apply lens distortion to ideal (undistorted) points.

    This simulates what a real camera with distortion would capture.
    Uses cv2.projectPoints to apply distortion model.

    Args:
        points: Nx2 array of undistorted image points
        K: 3x3 camera intrinsic matrix
        distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        Nx2 array of distorted points
    """
    # Convert points to normalized camera coordinates
    points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    # Create dummy 3D points (we only care about the 2D projection)
    # Place them at unit depth in camera coordinates
    K_inv = np.linalg.inv(K)
    object_points = []

    for pt in points:
        # Unproject to normalized coordinates then to 3D at depth=1
        pt_hom = np.array([pt[0], pt[1], 1.0])
        pt_normalized = K_inv @ pt_hom
        # Scale to depth 1.0
        object_points.append([pt_normalized[0], pt_normalized[1], 1.0])

    object_points = np.array(object_points, dtype=np.float32)

    # Identity rotation and zero translation (points already in camera frame)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)

    # Project with distortion
    distorted_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        distortion_coeffs
    )

    return distorted_points.reshape(-1, 2)


# Test Cases

def test_undistort_returns_same_when_no_distortion():
    """
    Test that when distortion_coeffs is None or all zeros,
    project_image_to_map returns the same results as without distortion.

    This verifies backward compatibility - existing code without distortion
    correction should behave identically.
    """
    params = get_test_camera_params()

    # Test with None distortion
    geo_none = CameraGeometry(params['W_px'], params['H_px'])
    geo_none.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=None
    )

    # Test with zero distortion coefficients
    geo_zero = CameraGeometry(params['W_px'], params['H_px'])
    zero_dist = np.zeros(5, dtype=np.float32)
    geo_zero.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=zero_dist
    )

    # Test points: center, corners, and edges
    test_points = [
        (params['W_px'] // 2, params['H_px'] // 2),  # Center
        (params['W_px'] // 4, params['H_px'] // 4),  # Near center
        (100, 100),  # Top-left corner
        (params['W_px'] - 100, 100),  # Top-right corner
        (100, params['H_px'] - 100),  # Bottom-left corner
        (params['W_px'] - 100, params['H_px'] - 100),  # Bottom-right corner
    ]

    # Project with None distortion
    result_none = geo_none.project_image_to_map(
        test_points,
        params['map_width'],
        params['map_height']
    )

    # Project with zero distortion
    result_zero = geo_zero.project_image_to_map(
        test_points,
        params['map_width'],
        params['map_height']
    )

    # Results should be identical
    for i, (pt_none, pt_zero) in enumerate(zip(result_none, result_zero)):
        error = np.sqrt((pt_none[0] - pt_zero[0])**2 + (pt_none[1] - pt_zero[1])**2)
        assert error < 1.0, f"Point {i} differs: {pt_none} vs {pt_zero}, error={error:.2f}px"

    print(f"✓ test_undistort_returns_same_when_no_distortion: PASSED")
    print(f"  Verified {len(test_points)} points with None and zero distortion")


def test_undistort_modifies_edge_points_more_than_center():
    """
    Test that radial distortion affects edge points more than center points.

    This is a fundamental property of radial lens distortion:
    - Distortion magnitude increases with distance from optical center
    - Points at image edges should move more than points near center
    """
    params = get_test_camera_params()

    # Use negative k1 for barrel distortion (typical in wide-angle cameras)
    distortion_coeffs = np.array([-0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    geo = CameraGeometry(params['W_px'], params['H_px'])
    geo.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=distortion_coeffs
    )

    cx, cy = params['W_px'] / 2, params['H_px'] / 2

    # Test points at varying distances from center
    points_to_test = [
        # (point, description, expected_category)
        (np.array([[cx, cy]], dtype=np.float32), "Center", "center"),
        (np.array([[cx + 100, cy]], dtype=np.float32), "Near center", "near"),
        (np.array([[cx + 400, cy]], dtype=np.float32), "Mid radius", "mid"),
        (np.array([[cx + 800, cy]], dtype=np.float32), "Far from center", "edge"),
    ]

    corrections = []

    for points, description, category in points_to_test:
        # Apply undistortion
        undistorted = geo._undistort_points(points)

        # Calculate correction magnitude
        correction = np.linalg.norm(undistorted - points)
        corrections.append((correction, description, category))

        print(f"  {description:20s}: correction = {correction:.3f} pixels")

    # Verify that correction increases with distance from center
    # Center should have minimal correction
    assert corrections[0][0] < 2.0, f"Center point has unexpected correction: {corrections[0][0]:.3f}px"

    # Each successive point should have larger correction (or very similar for numerical stability)
    for i in range(len(corrections) - 1):
        curr_correction = corrections[i][0]
        next_correction = corrections[i + 1][0]
        assert next_correction >= curr_correction * 0.9, \
            f"Correction not increasing: {corrections[i][1]}={curr_correction:.3f} vs {corrections[i+1][1]}={next_correction:.3f}"

    # Edge correction should be significantly larger than center
    edge_to_center_ratio = corrections[-1][0] / max(corrections[0][0], 0.1)
    assert edge_to_center_ratio > 5.0, \
        f"Edge correction not significantly larger than center: ratio={edge_to_center_ratio:.2f}"

    print(f"✓ test_undistort_modifies_edge_points_more_than_center: PASSED")
    print(f"  Edge/center correction ratio: {edge_to_center_ratio:.2f}")


def test_distortion_correction_consistency():
    """
    Test that applying distortion then undistorting returns approximately to original.

    This verifies the mathematical consistency of the distortion model:
    undistort(distort(point)) ≈ point
    """
    params = get_test_camera_params()

    # Use realistic distortion coefficients
    distortion_coeffs = np.array([-0.2, 0.05, 0.001, 0.001, 0.0], dtype=np.float32)

    geo = CameraGeometry(params['W_px'], params['H_px'])
    geo.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=distortion_coeffs
    )

    # Original undistorted points
    cx, cy = params['W_px'] / 2, params['H_px'] / 2
    original_points = np.array([
        [cx, cy],  # Center
        [cx + 300, cy + 200],  # Offset from center
        [cx - 400, cy - 300],  # Other quadrant
        [200, 200],  # Corner region
        [params['W_px'] - 200, params['H_px'] - 200],  # Opposite corner
    ], dtype=np.float32)

    # Apply distortion (simulate camera capture)
    distorted_points = create_distorted_points(
        original_points,
        params['K'],
        distortion_coeffs
    )

    # Undistort (what our code does)
    recovered_points = geo._undistort_points(distorted_points)

    # Check round-trip error
    max_error = 0.0
    print(f"  Round-trip distortion test:")

    for i, (orig, dist, recov) in enumerate(zip(original_points, distorted_points, recovered_points)):
        error = np.linalg.norm(orig - recov)
        max_error = max(max_error, error)

        print(f"    Point {i}: original={orig}, distorted={dist}, recovered={recov}, error={error:.3f}px")

        # Allow small numerical error (sub-pixel)
        assert error < 1.0, f"Point {i} round-trip error too large: {error:.3f}px"

    print(f"✓ test_distortion_correction_consistency: PASSED")
    print(f"  Max round-trip error: {max_error:.3f} pixels")


def test_project_image_to_map_with_distortion():
    """
    Test the full pipeline: project points with and without distortion.

    Verifies that:
    1. Results differ when significant distortion is present
    2. Results are identical when distortion is zero
    3. Distortion correction is properly integrated into the projection pipeline
    """
    params = get_test_camera_params()

    # Geometry without distortion
    geo_no_dist = CameraGeometry(params['W_px'], params['H_px'])
    geo_no_dist.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=None
    )

    # Geometry with barrel distortion
    distortion_coeffs = np.array([-0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    geo_with_dist = CameraGeometry(params['W_px'], params['H_px'])
    geo_with_dist.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=distortion_coeffs
    )

    # Geometry with zero distortion (should match no distortion)
    zero_distortion = np.zeros(5, dtype=np.float32)
    geo_zero_dist = CameraGeometry(params['W_px'], params['H_px'])
    geo_zero_dist.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=zero_distortion
    )

    # Test points - focus on areas where distortion is significant
    test_points = [
        (params['W_px'] // 2, params['H_px'] - 200),  # Bottom center
        (200, params['H_px'] - 200),  # Bottom left
        (params['W_px'] - 200, params['H_px'] - 200),  # Bottom right
        (params['W_px'] // 2, 200),  # Top center
    ]

    # Project points
    result_no_dist = geo_no_dist.project_image_to_map(
        test_points,
        params['map_width'],
        params['map_height']
    )

    result_with_dist = geo_with_dist.project_image_to_map(
        test_points,
        params['map_width'],
        params['map_height']
    )

    result_zero_dist = geo_zero_dist.project_image_to_map(
        test_points,
        params['map_width'],
        params['map_height']
    )

    print(f"  Comparing projections with/without distortion:")

    # Check that zero distortion matches no distortion
    for i, (pt_none, pt_zero) in enumerate(zip(result_no_dist, result_zero_dist)):
        error = np.sqrt((pt_none[0] - pt_zero[0])**2 + (pt_none[1] - pt_zero[1])**2)
        assert error < 1.0, f"Zero distortion differs from None: point {i}, error={error:.2f}px"

    print(f"    ✓ Zero distortion matches no distortion")

    # Check that significant distortion produces different results
    significant_difference_count = 0
    for i, (pt_no, pt_with) in enumerate(zip(result_no_dist, result_with_dist)):
        diff = np.sqrt((pt_no[0] - pt_with[0])**2 + (pt_no[1] - pt_with[1])**2)
        print(f"    Point {i}: no_dist={pt_no}, with_dist={pt_with}, diff={diff:.2f}px")

        # Corner points should show significant difference
        if i in [1, 2]:  # Bottom corners
            if diff > 5.0:
                significant_difference_count += 1

    # At least some points should show meaningful difference
    assert significant_difference_count > 0, \
        "Distortion correction should produce different results for edge points"

    print(f"✓ test_project_image_to_map_with_distortion: PASSED")
    print(f"  {significant_difference_count} points showed significant difference (>5px)")


def test_distortion_coefficients_stored_correctly():
    """
    Test that distortion coefficients are stored correctly as instance variable.

    Verifies that set_camera_parameters properly stores distortion_coeffs
    and that they can be accessed later.
    """
    params = get_test_camera_params()

    # Test with None
    geo = CameraGeometry(params['W_px'], params['H_px'])
    geo.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=None
    )

    assert geo.distortion_coeffs is None, "distortion_coeffs should be None when not provided"
    print(f"  ✓ None distortion stored correctly")

    # Test with actual coefficients
    test_coeffs = np.array([-0.2, 0.05, 0.001, -0.002, 0.01], dtype=np.float32)
    geo.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=test_coeffs
    )

    assert geo.distortion_coeffs is not None, "distortion_coeffs should be set"
    np.testing.assert_array_almost_equal(
        geo.distortion_coeffs,
        test_coeffs,
        decimal=6,
        err_msg="Stored distortion coefficients don't match input"
    )

    print(f"  ✓ Distortion coefficients stored correctly: {test_coeffs}")

    # Test with zero coefficients
    zero_coeffs = np.zeros(5, dtype=np.float32)
    geo.set_camera_parameters(
        K=params['K'],
        w_pos=params['w_pos'],
        pan_deg=params['pan_deg'],
        tilt_deg=params['tilt_deg'],
        map_width=params['map_width'],
        map_height=params['map_height'],
        distortion_coeffs=zero_coeffs
    )

    assert geo.distortion_coeffs is not None, "Zero distortion coeffs should still be stored"
    np.testing.assert_array_equal(
        geo.distortion_coeffs,
        zero_coeffs,
        err_msg="Zero distortion coefficients not stored correctly"
    )

    print(f"  ✓ Zero distortion coefficients stored correctly")
    print(f"✓ test_distortion_coefficients_stored_correctly: PASSED")


def test_various_distortion_strengths():
    """
    Test projection behavior with mild, moderate, and strong distortion.

    Verifies that correction magnitude scales appropriately with distortion strength.
    """
    params = get_test_camera_params()

    # Test point at edge (where distortion is most visible)
    test_point = np.array([[params['W_px'] - 300, params['H_px'] - 300]], dtype=np.float32)

    distortion_levels = [
        (np.array([-0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "Mild"),
        (np.array([-0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "Moderate"),
        (np.array([-0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "Strong"),
    ]

    corrections = []
    print(f"  Testing distortion correction magnitude:")

    for dist_coeffs, label in distortion_levels:
        geo = CameraGeometry(params['W_px'], params['H_px'])
        geo.set_camera_parameters(
            K=params['K'],
            w_pos=params['w_pos'],
            pan_deg=params['pan_deg'],
            tilt_deg=params['tilt_deg'],
            map_width=params['map_width'],
            map_height=params['map_height'],
            distortion_coeffs=dist_coeffs
        )

        undistorted = geo._undistort_points(test_point)
        correction = np.linalg.norm(undistorted - test_point)
        corrections.append(correction)

        print(f"    {label:12s} (k1={dist_coeffs[0]:+.2f}): correction = {correction:.2f}px")

    # Verify that correction increases with distortion strength
    assert corrections[0] < corrections[1] < corrections[2], \
        "Correction magnitude should increase with distortion strength"

    # Strong distortion should produce significant correction at edges
    assert corrections[2] > 10.0, \
        f"Strong distortion should produce >10px correction at edges, got {corrections[2]:.2f}px"

    print(f"✓ test_various_distortion_strengths: PASSED")
    print(f"  Correction scaling: mild={corrections[0]:.1f}px, moderate={corrections[1]:.1f}px, strong={corrections[2]:.1f}px")


# Main test runner for standalone execution

if __name__ == "__main__":
    """Run all tests with detailed output."""
    print("\n" + "="*70)
    print("DISTORTION CORRECTION TEST SUITE")
    print("="*70 + "\n")

    tests = [
        test_undistort_returns_same_when_no_distortion,
        test_undistort_modifies_edge_points_more_than_center,
        test_distortion_correction_consistency,
        test_project_image_to_map_with_distortion,
        test_distortion_coefficients_stored_correctly,
        test_various_distortion_strengths,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        print(f"\nRunning: {test_func.__name__}")
        print("-" * 70)
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    sys.exit(0 if failed == 0 else 1)
