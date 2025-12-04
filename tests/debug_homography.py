#!/usr/bin/env python3
"""
Debug homography calculations with detailed output.
"""

import sys
import numpy as np
from poc_homography.camera_geometry import CameraGeometry
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ
from poc_homography.camera_config import get_camera_by_name, USERNAME, PASSWORD, CAMERAS


def debug_homography(camera_name: str, height: float = 5.0):
    """Debug homography with actual camera parameters."""

    print("="*70)
    print(f"HOMOGRAPHY DEBUG FOR {camera_name}")
    print("="*70)

    # Get camera info and status
    cam_info = get_camera_by_name(camera_name, CAMERAS)
    if not cam_info:
        print(f"Camera '{camera_name}' not found")
        return

    camera = HikvisionPTZ(
        ip=cam_info["ip"],
        username=USERNAME,
        password=PASSWORD,
        name=cam_info["name"]
    )

    status = camera.get_status()
    print(f"\n1. CAMERA STATUS:")
    print(f"   Pan:  {status['pan']:.2f}°")
    print(f"   Tilt: {status['tilt']:.2f}°")
    print(f"   Zoom: {status['zoom']:.2f}x")

    # Setup geometry
    W_px, H_px = 2560, 1440
    geo = CameraGeometry(W_px, H_px)

    # Calculate intrinsics
    K = geo.get_intrinsics(zoom_factor=status['zoom'], W_px=W_px, H_px=H_px)

    print(f"\n2. INTRINSIC MATRIX K:")
    print(f"   Focal length: {K[0,0]:.2f} pixels")
    print(f"   Principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"   K = \n{K}")

    # Setup world position
    w_pos = np.array([0.0, 0.0, height])

    print(f"\n3. CAMERA WORLD POSITION:")
    print(f"   w_pos = {w_pos} meters")

    # Set parameters
    # Pass tilt directly - the internal _get_rotation_matrix() handles
    # the Hikvision convention (positive = down) conversion
    geo.set_camera_parameters(
        K=K,
        w_pos=w_pos,
        pan_deg=status['pan'],
        tilt_deg=status['tilt'],
        map_width=640,
        map_height=480
    )

    print(f"\n4. ROTATION MATRIX R:")
    R = geo._get_rotation_matrix()
    print(f"{R}")
    print(f"   det(R) = {np.linalg.det(R):.6f} (should be ~1.0)")

    print(f"\n5. TRANSLATION VECTOR t = -R @ w_pos:")
    t = -R @ w_pos
    print(f"   t = {t}")

    print(f"\n6. HOMOGRAPHY MATRIX H:")
    print(f"{geo.H}")
    print(f"   H[2,2] = {geo.H[2,2]:.6f} (normalized to 1.0)")
    print(f"   det(H) = {np.linalg.det(geo.H):.2e}")

    print(f"\n7. TEST PROJECTIONS:")
    print("-"*70)

    # Test specific image points
    test_points = [
        (W_px//2, H_px - 100, "Bottom center (near field)"),
        (W_px//2, H_px//2, "Image center"),
        (W_px//2, 200, "Top center (far field)"),
        (W_px//4, H_px - 100, "Bottom left"),
        (3*W_px//4, H_px - 100, "Bottom right"),
    ]

    for u, v, desc in test_points:
        pt_img = np.array([[u], [v], [1.0]])
        pt_world = geo.H_inv @ pt_img

        # Check for valid projection
        if abs(pt_world[2, 0]) < 1e-6:
            print(f"\n   {desc} ({u}, {v})px:")
            print(f"      → INVALID (near horizon, W≈0)")
            continue

        Xw = pt_world[0, 0] / pt_world[2, 0]
        Yw = pt_world[1, 0] / pt_world[2, 0]
        dist = np.sqrt(Xw**2 + Yw**2)
        angle = np.degrees(np.arctan2(Xw, Yw))

        print(f"\n   {desc} ({u}, {v})px:")
        print(f"      → World: ({Xw:7.2f}, {Yw:7.2f}) meters")
        print(f"      → Distance: {dist:7.2f}m")
        print(f"      → Angle: {angle:6.1f}°")
        print(f"      → Raw homogeneous: [{pt_world[0,0]:.2f}, {pt_world[1,0]:.2f}, {pt_world[2,0]:.6f}]")

    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR CHECK:")
    print("="*70)

    # Calculate expected distance for center point at given tilt
    tilt_rad = np.radians(status['tilt'])
    if status['tilt'] < 0:  # Camera pointing down
        expected_center_dist = height / np.tan(-tilt_rad)
        print(f"\nFor tilt={status['tilt']:.1f}° and height={height}m:")
        print(f"  Expected center distance: ~{expected_center_dist:.2f}m")
        print(f"  (Using formula: distance = height / tan(|tilt|))")

    # Check if H_inv @ H = I
    print(f"\nRound-trip check (H @ H_inv):")
    identity_test = geo.H @ geo.H_inv
    print(identity_test)
    is_identity = np.allclose(identity_test, np.eye(3), atol=1e-4)
    print(f"  Is identity? {is_identity}")

    print("\n" + "="*70)
    print("DIAGNOSTIC HINTS:")
    print("="*70)

    # Analyze results
    center_u, center_v = W_px//2, H_px//2
    pt_center = np.array([[center_u], [center_v], [1.0]])
    pt_world_center = geo.H_inv @ pt_center
    if abs(pt_world_center[2, 0]) > 1e-6:
        Xw_center = pt_world_center[0, 0] / pt_world_center[2, 0]
        Yw_center = pt_world_center[1, 0] / pt_world_center[2, 0]
        actual_center_dist = np.sqrt(Xw_center**2 + Yw_center**2)

        if abs(actual_center_dist - expected_center_dist) / expected_center_dist > 0.2:
            print(f"\n⚠️  CENTER DISTANCE MISMATCH:")
            print(f"    Expected: {expected_center_dist:.2f}m")
            print(f"    Actual:   {actual_center_dist:.2f}m")
            print(f"    Error:    {abs(actual_center_dist - expected_center_dist):.2f}m")

            if actual_center_dist < expected_center_dist * 0.5:
                print(f"\n    Possible causes:")
                print(f"    • Rotation matrix order incorrect")
                print(f"    • Camera coordinate frame wrong")
                print(f"    • Tilt angle sign inverted")
            elif actual_center_dist > expected_center_dist * 2:
                print(f"\n    Possible causes:")
                print(f"    • Height parameter wrong")
                print(f"    • Focal length calculation error")

    if abs(Xw_center) > 1.0:
        print(f"\n⚠️  CENTER X-COORDINATE NOT NEAR ZERO:")
        print(f"    X = {Xw_center:.2f}m (should be ~0)")
        print(f"    Possible causes:")
        print(f"    • Pan angle offset")
        print(f"    • Rotation matrix error")

    print("\n" + "="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_homography.py CAMERA_NAME [HEIGHT]")
        print(f"Available cameras: Valte, Setram")
        sys.exit(1)

    camera_name = sys.argv[1]
    height = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    debug_homography(camera_name, height)


if __name__ == "__main__":
    main()
