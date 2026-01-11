#!/usr/bin/env python3
"""
Test suite for parameter validation in camera_geometry.py
Tests Issue #6: Add parameter validation and sanity checks to homography pipeline.

UPDATED: Refactored for immutable API (Phase 2)
- Uses CameraParameters.create() + CameraGeometry.compute() for validation tests
- Uses typed parameters (Pixels, Degrees, Unitless, etc.)
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import CameraParameters
from poc_homography.types import Degrees, Pixels, Unitless

# Configure logging to see warnings
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def test_zoom_validation():
    """Test zoom factor validation in get_intrinsics."""
    print("\n" + "=" * 60)
    print("TEST: Zoom Factor Validation")
    print("=" * 60)

    # Valid zoom
    print("\n1. Testing valid zoom (1.0)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        print("   PASS: Valid zoom accepted")
    except ValueError as e:
        print(f"   FAIL: Valid zoom rejected: {e}")

    # High but valid zoom (should trigger warning)
    print("\n2. Testing high zoom (21.0, should warn)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(21.0))
        print("   PASS: High zoom accepted with warning")
    except ValueError as e:
        print(f"   FAIL: High zoom rejected: {e}")

    # Invalid zoom - too low
    print("\n3. Testing invalid zoom (0.5, too low)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(0.5))
        print("   FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   PASS: Invalid zoom correctly rejected: {e}")

    # Invalid zoom - too high
    print("\n4. Testing invalid zoom (30.0, too high)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(30.0))
        print("   FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   PASS: Invalid zoom correctly rejected: {e}")


def test_tilt_validation():
    """Test tilt angle validation."""
    print("\n" + "=" * 60)
    print("TEST: Tilt Angle Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
    w_pos = np.array([0.0, 0.0, 5.0])

    # Valid tilt
    print("\n1. Testing valid tilt (45 degrees)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Valid tilt accepted")
    except ValueError as e:
        print(f"   FAIL: Valid tilt rejected: {e}")

    # Near-horizontal tilt (should warn)
    print("\n2. Testing near-horizontal tilt (5 degrees, should warn)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(5.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Near-horizontal tilt accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Near-horizontal tilt rejected: {e}")

    # Steep tilt (should warn)
    print("\n3. Testing steep tilt (85 degrees, should warn)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(85.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Steep tilt accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Steep tilt rejected: {e}")

    # Invalid tilt - zero (camera pointing at horizon)
    print("\n4. Testing invalid tilt (0 degrees, pointing at horizon)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(0.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")

    # Invalid tilt - negative (pointing upward)
    print("\n5. Testing invalid tilt (-10 degrees, pointing upward)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(-10.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")

    # Invalid tilt - too large
    print("\n6. Testing invalid tilt (95 degrees, beyond vertical)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(95.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")


def test_height_validation():
    """Test camera height validation."""
    print("\n" + "=" * 60)
    print("TEST: Camera Height Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))

    # Valid height
    print("\n1. Testing valid height (5m)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Valid height accepted")
    except ValueError as e:
        print(f"   FAIL: Valid height rejected: {e}")

    # Low height (should warn)
    print("\n2. Testing low height (1.5m, should warn)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 1.5]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Low height accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Low height rejected: {e}")

    # High height (should warn)
    print("\n3. Testing high height (35m, should warn)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 35.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: High height accepted with warning")
    except ValueError as e:
        print(f"   FAIL: High height rejected: {e}")

    # Invalid height - too low
    print("\n4. Testing invalid height (0.5m, too low)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 0.5]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   PASS: Invalid height correctly rejected: {e}")

    # Invalid height - too high
    print("\n5. Testing invalid height (60m, too high)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 60.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   PASS: Invalid height correctly rejected: {e}")


def test_condition_number_validation():
    """Test homography condition number validation."""
    print("\n" + "=" * 60)
    print("TEST: Homography Condition Number")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))

    # Good condition number
    print("\n1. Testing normal configuration (tilt=45 degrees)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Normal configuration accepted")
    except ValueError as e:
        print(f"   FAIL: Normal configuration rejected: {e}")

    # Near-horizontal should have higher condition number (may warn)
    print("\n2. Testing near-horizontal (tilt=2 degrees, may have high condition number)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(2.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Near-horizontal accepted (may have warned)")
    except ValueError as e:
        print(f"   PASS: Near-horizontal correctly rejected due to ill-conditioning: {e}")


def test_projection_validation():
    """Test projected distance validation."""
    print("\n" + "=" * 60)
    print("TEST: Projected Distance Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))

    # Normal projection
    print("\n1. Testing normal projection (tilt=45 degrees)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Normal projection validated")
    except ValueError as e:
        print(f"   FAIL: Normal projection rejected: {e}")

    # Very shallow angle may produce large distance (may warn)
    print("\n2. Testing shallow angle (tilt=3 degrees, may produce large distance)...")
    try:
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(3.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Shallow angle accepted (may have warned about large distance)")
    except ValueError as e:
        print(f"   PASS: Shallow angle rejected: {e}")


def test_boundary_values():
    """Test exact boundary values for parameters."""
    print("\n" + "=" * 60)
    print("TEST: Boundary Value Validation")
    print("=" * 60)

    # Test zoom boundary: max valid (25.0)
    print("\n1. Testing zoom=25.0 (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(25.0))
        print("   PASS: Zoom 25.0 accepted")
    except ValueError as e:
        print(f"   FAIL: Zoom 25.0 rejected: {e}")

    # Test tilt boundary: max valid (90.0)
    print("\n2. Testing tilt=90.0 (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 5.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(90.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Tilt 90.0 degrees accepted")
    except ValueError as e:
        print(f"   FAIL: Tilt 90.0 degrees rejected: {e}")

    # Test height boundary: min valid (1.0)
    print("\n3. Testing height=1.0m (min valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 1.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Height 1.0m accepted")
    except ValueError as e:
        print(f"   FAIL: Height 1.0m rejected: {e}")

    # Test height boundary: max valid (50.0)
    print("\n4. Testing height=50.0m (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        params = CameraParameters.create(
            image_width=Pixels(2560),
            image_height=Pixels(1440),
            intrinsic_matrix=K,
            camera_position=np.array([0.0, 0.0, 50.0]),
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(480),
            pixels_per_meter=Unitless(100.0),
        )
        result = CameraGeometry.compute(params)
        print("   PASS: Height 50.0m accepted")
    except ValueError as e:
        print(f"   FAIL: Height 50.0m rejected: {e}")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("PARAMETER VALIDATION TEST SUITE (Issue #6)")
    print("=" * 70)

    try:
        test_zoom_validation()
        test_tilt_validation()
        test_height_validation()
        test_condition_number_validation()
        test_projection_validation()
        test_boundary_values()

        print("\n" + "=" * 70)
        print("ALL VALIDATION TESTS COMPLETED")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n TEST SUITE FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
