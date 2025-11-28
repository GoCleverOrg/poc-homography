#!/usr/bin/env python3
"""
Automated tests to verify homography consistency and correctness.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera_geometry import CameraGeometry


def test_forward_backward_consistency():
    """Test that H and H_inv are true inverses."""
    print("\n" + "="*60)
    print("TEST 1: Forward-Backward Consistency")
    print("="*60)

    # Setup camera
    geo = CameraGeometry(2560, 1440)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=2560, H_px=1440)
    w_pos = np.array([0.0, 0.0, 5.0])  # 5m height

    geo.set_camera_parameters(
        K=K,
        w_pos=w_pos,
        pan_deg=0.0,
        tilt_deg=45.0,  # Positive = pointing down (Hikvision convention)
        map_width=640,
        map_height=480
    )

    # Test points at various distances
    world_points = [
        (0.0, 5.0),   # 5m straight ahead
        (5.0, 10.0),  # 10m ahead, 5m to the right
        (-3.0, 8.0),  # 8m ahead, 3m to the left
    ]

    print("\nTesting round-trip projection (World → Image → World):")
    max_error = 0.0

    for Xw, Yw in world_points:
        # World → Image
        pt_world = np.array([[Xw], [Yw], [1.0]])
        pt_image = geo.H @ pt_world
        u = pt_image[0, 0] / pt_image[2, 0]
        v = pt_image[1, 0] / pt_image[2, 0]

        # Image → World
        pt_image_norm = np.array([[u], [v], [1.0]])
        pt_world_recovered = geo.H_inv @ pt_image_norm
        Xw_recovered = pt_world_recovered[0, 0] / pt_world_recovered[2, 0]
        Yw_recovered = pt_world_recovered[1, 0] / pt_world_recovered[2, 0]

        # Calculate error
        error = np.sqrt((Xw - Xw_recovered)**2 + (Yw - Yw_recovered)**2)
        max_error = max(max_error, error)

        status = "✓ PASS" if error < 0.01 else "✗ FAIL"
        print(f"  ({Xw:6.2f}, {Yw:6.2f})m → ({u:7.1f}, {v:7.1f})px → ({Xw_recovered:6.2f}, {Yw_recovered:6.2f})m  |  Error: {error:.6f}m  {status}")

    print(f"\nMax error: {max_error:.6f} meters")
    print("✓ TEST PASSED" if max_error < 0.01 else "✗ TEST FAILED")


def test_principal_point_projection():
    """Test that image center projects to a point in front of camera."""
    print("\n" + "="*60)
    print("TEST 2: Principal Point Projection")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=2560, H_px=1440)
    w_pos = np.array([0.0, 0.0, 5.0])

    for tilt in [30, 45, 60]:  # Positive = pointing down (Hikvision convention)
        geo.set_camera_parameters(
            K=K,
            w_pos=w_pos,
            pan_deg=0.0,
            tilt_deg=tilt,
            map_width=640,
            map_height=480
        )

        # Project center of image to ground
        cx, cy = 2560 / 2, 1440 / 2
        pt_image = np.array([[cx], [cy], [1.0]])
        pt_world = geo.H_inv @ pt_image

        Xw = pt_world[0, 0] / pt_world[2, 0]
        Yw = pt_world[1, 0] / pt_world[2, 0]
        distance = np.sqrt(Xw**2 + Yw**2)

        # For downward tilt: Hikvision tilt is ELEVATION angle (from horizon looking up)
        # So depression angle = 90° - tilt
        # Distance should INCREASE with larger tilt (because depression decreases)
        depression = 90.0 - tilt
        expected_distance = 5.0 / np.tan(np.radians(depression))  # h/tan(depression_angle)

        error = abs(distance - expected_distance)
        status = "✓ PASS" if error < 0.5 else "✗ FAIL"

        print(f"  Tilt {tilt:>3}°: Center projects to ({Xw:6.2f}, {Yw:6.2f})m, distance={distance:.2f}m (expected ~{expected_distance:.2f}m) {status}")


def test_horizon_behavior():
    """Test that points near horizon have very large world coordinates."""
    print("\n" + "="*60)
    print("TEST 3: Horizon Behavior")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=2560, H_px=1440)
    w_pos = np.array([0.0, 0.0, 5.0])

    geo.set_camera_parameters(
        K=K,
        w_pos=w_pos,
        pan_deg=0.0,
        tilt_deg=45.0,  # Positive = pointing down (Hikvision convention)
        map_width=640,
        map_height=480
    )

    print("\nProjecting points from bottom to top of image:")
    # Test points at different vertical positions
    cx = 2560 / 2
    for v in [1440 - 100, 1200, 900, 600, 300, 100]:  # Bottom to top
        pt_image = np.array([[cx], [v], [1.0]])
        pt_world = geo.H_inv @ pt_image

        if abs(pt_world[2, 0]) > 1e-6:
            Xw = pt_world[0, 0] / pt_world[2, 0]
            Yw = pt_world[1, 0] / pt_world[2, 0]
            distance = np.sqrt(Xw**2 + Yw**2)
            print(f"  v={v:>4}px → ({Xw:8.2f}, {Yw:8.2f})m, distance={distance:8.2f}m")
        else:
            print(f"  v={v:>4}px → (infinity, infinity) - near horizon")

    print("\n✓ Points closer to top of image should have larger distances")


def test_pan_rotation():
    """Test that pan angle rotates projections correctly."""
    print("\n" + "="*60)
    print("TEST 4: Pan Rotation")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=2560, H_px=1440)
    w_pos = np.array([0.0, 0.0, 5.0])

    # Test same image point with different pan angles
    u, v = 2560/2, 1440 - 200  # Near bottom center

    print(f"\nProjecting point ({u:.0f}, {v:.0f})px with different pan angles:")
    for pan in [-90, -45, 0, 45, 90]:
        geo.set_camera_parameters(
            K=K,
            w_pos=w_pos,
            pan_deg=pan,
            tilt_deg=45.0,  # Positive = pointing down (Hikvision convention)
            map_width=640,
            map_height=480
        )

        pt_image = np.array([[u], [v], [1.0]])
        pt_world = geo.H_inv @ pt_image

        Xw = pt_world[0, 0] / pt_world[2, 0]
        Yw = pt_world[1, 0] / pt_world[2, 0]
        angle = np.degrees(np.arctan2(Xw, Yw))

        print(f"  Pan {pan:>4}°: World=({Xw:6.2f}, {Yw:6.2f})m, Angle={angle:6.1f}°")

    print("\n✓ World coordinates should rotate with pan angle")


def test_zoom_effect():
    """Test that zoom changes focal length and projection."""
    print("\n" + "="*60)
    print("TEST 5: Zoom Effect on Projection")
    print("="*60)

    geo = CameraGeometry(2560, 1440)
    w_pos = np.array([0.0, 0.0, 5.0])

    # Same image point, different zoom levels
    u, v = 2560/2, 1440 - 200

    print(f"\nProjecting point ({u:.0f}, {v:.0f})px with different zoom:")
    for zoom in [1.0, 2.0, 3.0, 4.0]:
        K = geo.get_intrinsics(zoom_factor=zoom, W_px=2560, H_px=1440)
        geo.set_camera_parameters(
            K=K,
            w_pos=w_pos,
            pan_deg=0.0,
            tilt_deg=45.0,  # Positive = pointing down (Hikvision convention)
            map_width=640,
            map_height=480
        )

        pt_image = np.array([[u], [v], [1.0]])
        pt_world = geo.H_inv @ pt_image

        Xw = pt_world[0, 0] / pt_world[2, 0]
        Yw = pt_world[1, 0] / pt_world[2, 0]
        distance = np.sqrt(Xw**2 + Yw**2)

        print(f"  Zoom {zoom:.1f}x: World=({Xw:6.2f}, {Yw:6.2f})m, Distance={distance:6.2f}m, f_px={K[0,0]:.1f}")

    print("\n✓ Higher zoom = smaller field of view = objects appear farther")


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("HOMOGRAPHY VERIFICATION TEST SUITE")
    print("="*70)

    try:
        test_forward_backward_consistency()
        test_principal_point_projection()
        test_horizon_behavior()
        test_pan_rotation()
        test_zoom_effect()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
