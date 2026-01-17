from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.domain.vo import CameraIntrinsics, Orientation, Vector3
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def test_zoom_validation():
    print("\n" + "=" * 60)
    print("TEST: Zoom Factor Validation")
    print("=" * 60)

    print("\n1. Testing valid zoom (1.0)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        print("   PASS: Valid zoom accepted")
    except ValueError as e:
        print(f"   FAIL: Valid zoom rejected: {e}")

    print("\n2. Testing high zoom (21.0, should warn)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(21.0))
        print("   PASS: High zoom accepted with warning")
    except ValueError as e:
        print(f"   FAIL: High zoom rejected: {e}")

    print("\n3. Testing invalid zoom (0.5, too low)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(0.5))
        print("   FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   PASS: Invalid zoom correctly rejected: {e}")

    print("\n4. Testing invalid zoom (30.0, too high)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(30.0))
        print("   FAIL: Invalid zoom accepted")
    except ValueError as e:
        print(f"   PASS: Invalid zoom correctly rejected: {e}")


def test_tilt_validation():
    print("\n" + "=" * 60)
    print("TEST: Tilt Angle Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
    intrinsics = CameraIntrinsics.from_K_matrix(
        K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
    )
    camera_position = Vector3.create(0.0, 0.0, 5.0)

    print("\n1. Testing valid tilt (45 degrees)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Valid tilt accepted")
    except ValueError as e:
        print(f"   FAIL: Valid tilt rejected: {e}")

    print("\n2. Testing near-horizontal tilt (5 degrees, should warn)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(5.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Near-horizontal tilt accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Near-horizontal tilt rejected: {e}")

    print("\n3. Testing steep tilt (85 degrees, should warn)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(85.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Steep tilt accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Steep tilt rejected: {e}")

    print("\n4. Testing invalid tilt (0 degrees, pointing at horizon)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(0.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")

    print("\n5. Testing invalid tilt (-10 degrees, pointing upward)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(-10.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")

    print("\n6. Testing invalid tilt (95 degrees, beyond vertical)...")
    try:
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(95.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   FAIL: Invalid tilt accepted")
    except ValueError as e:
        print(f"   PASS: Invalid tilt correctly rejected: {e}")


def test_height_validation():
    print("\n" + "=" * 60)
    print("TEST: Camera Height Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
    intrinsics = CameraIntrinsics.from_K_matrix(
        K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
    )

    print("\n1. Testing valid height (5m)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Valid height accepted")
    except ValueError as e:
        print(f"   FAIL: Valid height rejected: {e}")

    print("\n2. Testing low height (1.5m, should warn)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 1.5)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Low height accepted with warning")
    except ValueError as e:
        print(f"   FAIL: Low height rejected: {e}")

    print("\n3. Testing high height (35m, should warn)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 35.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: High height accepted with warning")
    except ValueError as e:
        print(f"   FAIL: High height rejected: {e}")

    print("\n4. Testing invalid height (0.5m, too low)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 0.5)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   PASS: Invalid height correctly rejected: {e}")

    print("\n5. Testing invalid height (60m, too high)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 60.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   FAIL: Invalid height accepted")
    except ValueError as e:
        print(f"   PASS: Invalid height correctly rejected: {e}")


def test_condition_number_validation():
    print("\n" + "=" * 60)
    print("TEST: Homography Condition Number")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
    intrinsics = CameraIntrinsics.from_K_matrix(
        K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
    )

    print("\n1. Testing normal configuration (tilt=45 degrees)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Normal configuration accepted")
    except ValueError as e:
        print(f"   FAIL: Normal configuration rejected: {e}")

    print("\n2. Testing near-horizontal (tilt=2 degrees, may have high condition number)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(2.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Near-horizontal accepted (may have warned)")
    except ValueError as e:
        print(f"   PASS: Near-horizontal correctly rejected due to ill-conditioning: {e}")


def test_projection_validation():
    print("\n" + "=" * 60)
    print("TEST: Projected Distance Validation")
    print("=" * 60)

    K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
    intrinsics = CameraIntrinsics.from_K_matrix(
        K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
    )

    print("\n1. Testing normal projection (tilt=45 degrees)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Normal projection validated")
    except ValueError as e:
        print(f"   FAIL: Normal projection rejected: {e}")

    print("\n2. Testing shallow angle (tilt=3 degrees, may produce large distance)...")
    try:
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(3.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Shallow angle accepted (may have warned about large distance)")
    except ValueError as e:
        print(f"   PASS: Shallow angle rejected: {e}")


def test_boundary_values():
    print("\n" + "=" * 60)
    print("TEST: Boundary Value Validation")
    print("=" * 60)

    print("\n1. Testing zoom=25.0 (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(25.0))
        print("   PASS: Zoom 25.0 accepted")
    except ValueError as e:
        print(f"   FAIL: Zoom 25.0 rejected: {e}")

    print("\n2. Testing tilt=90.0 (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        intrinsics = CameraIntrinsics.from_K_matrix(
            K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
        )
        camera_position = Vector3.create(0.0, 0.0, 5.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(90.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Tilt 90.0 degrees accepted")
    except ValueError as e:
        print(f"   FAIL: Tilt 90.0 degrees rejected: {e}")

    print("\n3. Testing height=1.0m (min valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        intrinsics = CameraIntrinsics.from_K_matrix(
            K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
        )
        camera_position = Vector3.create(0.0, 0.0, 1.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Height 1.0m accepted")
    except ValueError as e:
        print(f"   FAIL: Height 1.0m rejected: {e}")

    print("\n4. Testing height=50.0m (max valid boundary)...")
    try:
        K = CameraGeometry.get_intrinsics(zoom_factor=Unitless(1.0))
        intrinsics = CameraIntrinsics.from_K_matrix(
            K, Pixels(2560), Pixels(1440), sensor_width=Millimeters(6.78)
        )
        camera_position = Vector3.create(0.0, 0.0, 50.0)
        orientation = Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))
        homography = CameraGeometry.compute_from_vo(intrinsics, camera_position, orientation)
        print("   PASS: Height 50.0m accepted")
    except ValueError as e:
        print(f"   FAIL: Height 50.0m rejected: {e}")


def main():
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
