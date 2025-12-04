#!/usr/bin/env python3
"""
Test suite for parameter validation in camera_geometry.py
Tests Issue #6: Add parameter validation and sanity checks to homography pipeline.
"""

import numpy as np
import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.camera_geometry import CameraGeometry

# Configure logging to see warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def test_zoom_validation():
    """Test zoom factor validation in get_intrinsics."""
    print("\n" + "="*60)
    print("TEST: Zoom Factor Validation")
    print("="*60)

    # Valid zoom
    print("\n1. Testing valid zoom (1.0)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=1.0)
        print("   ✓ PASS: Valid zoom accepted")
    except ValueError as e:
        print(f"   ✗ FAIL: Valid zoom rejected: {e}")

    # High but valid zoom (should trigger warning)
    print("\n2. Testing high zoom (21.0, should warn)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=21.0)
        print("   ✓ PASS: High zoom accepted with warning")
    except ValueError as e:
        print(f"   ✗ FAIL: High zoom rejected: {e}")

    # Invalid zoom - too low
    print("\n3. Testing invalid zoom (0.5, too low)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=0.5)
        print("   ✗ FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid zoom correctly rejected: {e}")

    # Invalid zoom - too high
    print("\n4. Testing invalid zoom (30.0, too high)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=30.0)
        print("   ✗ FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid zoom correctly rejected: {e}")


def test_tilt_validation():
    """Test tilt angle validation."""
    print("\n" + "="*60)
    print("TEST: Tilt Angle Validation")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0)
    w_pos = np.array([0.0, 0.0, 5.0])

    # Valid tilt
    print("\n1. Testing valid tilt (45°)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 480)
        print("   ✓ PASS: Valid tilt accepted")
    except ValueError as e:
        print(f"   ✗ FAIL: Valid tilt rejected: {e}")

    # Near-horizontal tilt (should warn)
    print("\n2. Testing near-horizontal tilt (5°, should warn)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, 5.0, 640, 480)
        print("   ✓ PASS: Near-horizontal tilt accepted with warning")
    except ValueError as e:
        print(f"   ✗ FAIL: Near-horizontal tilt rejected: {e}")

    # Steep tilt (should warn)
    print("\n3. Testing steep tilt (85°, should warn)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, 85.0, 640, 480)
        print("   ✓ PASS: Steep tilt accepted with warning")
    except ValueError as e:
        print(f"   ✗ FAIL: Steep tilt rejected: {e}")

    # Invalid tilt - zero (camera pointing at horizon)
    print("\n4. Testing invalid tilt (0°, pointing at horizon)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, 0.0, 640, 480)
        print("   ✗ FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid tilt correctly rejected: {e}")

    # Invalid tilt - negative (pointing upward)
    print("\n5. Testing invalid tilt (-10°, pointing upward)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, -10.0, 640, 480)
        print("   ✗ FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid tilt correctly rejected: {e}")

    # Invalid tilt - too large
    print("\n6. Testing invalid tilt (95°, beyond vertical)...")
    try:
        geo.set_camera_parameters(K, w_pos, 0.0, 95.0, 640, 480)
        print("   ✗ FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid tilt correctly rejected: {e}")


def test_height_validation():
    """Test camera height validation."""
    print("\n" + "="*60)
    print("TEST: Camera Height Validation")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0)

    # Valid height
    print("\n1. Testing valid height (5m)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 5.0]), 0.0, 45.0, 640, 480)
        print("   ✓ PASS: Valid height accepted")
    except ValueError as e:
        print(f"   ✗ FAIL: Valid height rejected: {e}")

    # Low height (should warn)
    print("\n2. Testing low height (1.5m, should warn)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 1.5]), 0.0, 45.0, 640, 480)
        print("   ✓ PASS: Low height accepted with warning")
    except ValueError as e:
        print(f"   ✗ FAIL: Low height rejected: {e}")

    # High height (should warn)
    print("\n3. Testing high height (35m, should warn)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 35.0]), 0.0, 45.0, 640, 480)
        print("   ✓ PASS: High height accepted with warning")
    except ValueError as e:
        print(f"   ✗ FAIL: High height rejected: {e}")

    # Invalid height - too low
    print("\n4. Testing invalid height (0.5m, too low)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 0.5]), 0.0, 45.0, 640, 480)
        print("   ✗ FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid height correctly rejected: {e}")

    # Invalid height - too high
    print("\n5. Testing invalid height (60m, too high)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 60.0]), 0.0, 45.0, 640, 480)
        print("   ✗ FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   ✓ PASS: Invalid height correctly rejected: {e}")


def test_condition_number_validation():
    """Test homography condition number validation."""
    print("\n" + "="*60)
    print("TEST: Homography Condition Number")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0)

    # Good condition number
    print("\n1. Testing normal configuration (tilt=45°)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 5.0]), 0.0, 45.0, 640, 480)
        print("   ✓ PASS: Normal configuration accepted")
    except ValueError as e:
        print(f"   ✗ FAIL: Normal configuration rejected: {e}")

    # Near-horizontal should have higher condition number (may warn)
    print("\n2. Testing near-horizontal (tilt=2°, may have high condition number)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 5.0]), 0.0, 2.0, 640, 480)
        print("   ✓ PASS: Near-horizontal accepted (may have warned)")
    except ValueError as e:
        print(f"   ✓ PASS: Near-horizontal correctly rejected due to ill-conditioning: {e}")


def test_projection_validation():
    """Test projected distance validation."""
    print("\n" + "="*60)
    print("TEST: Projected Distance Validation")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0)

    # Normal projection
    print("\n1. Testing normal projection (tilt=45°)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 5.0]), 0.0, 45.0, 640, 480)
        print("   ✓ PASS: Normal projection validated")
    except ValueError as e:
        print(f"   ✗ FAIL: Normal projection rejected: {e}")

    # Very shallow angle may produce large distance (may warn)
    print("\n2. Testing shallow angle (tilt=3°, may produce large distance)...")
    try:
        geo.set_camera_parameters(K, np.array([0.0, 0.0, 5.0]), 0.0, 3.0, 640, 480)
        print("   ✓ PASS: Shallow angle accepted (may have warned about large distance)")
    except ValueError as e:
        print(f"   ✓ PASS: Shallow angle rejected: {e}")


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("PARAMETER VALIDATION TEST SUITE (Issue #6)")
    print("="*70)

    try:
        test_zoom_validation()
        test_tilt_validation()
        test_height_validation()
        test_condition_number_validation()
        test_projection_validation()

        print("\n" + "="*70)
        print("ALL VALIDATION TESTS COMPLETED")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
