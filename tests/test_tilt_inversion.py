#!/usr/bin/env python3
"""
Test if inverting tilt sign fixes the homography.
"""

import sys

import numpy as np

from poc_homography.camera_geometry import CameraGeometry


def test_tilt_inversion(pan_deg, tilt_deg, height=5.0):
    """Test homography with normal and inverted tilt."""

    W_px, H_px = 2560, 1440
    geo = CameraGeometry(W_px, H_px)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=W_px, H_px=H_px)
    w_pos = np.array([0.0, 0.0, height])

    print("=" * 70)
    print("TESTING TILT SIGN CONVENTION")
    print("=" * 70)
    print("\nCamera parameters:")
    print(f"  Pan:  {pan_deg:.1f}°")
    print(f"  Tilt: {tilt_deg:.1f}° (as reported by camera)")
    print(f"  Height: {height}m")

    for sign, desc in [(1, "AS REPORTED"), (-1, "INVERTED")]:
        tilt_test = sign * tilt_deg

        print(f"\n{'=' * 70}")
        print(f"TEST: {desc} (using tilt={tilt_test:.1f}°)")
        print(f"{'=' * 70}")

        geo.set_camera_parameters(
            K=K, w_pos=w_pos, pan_deg=pan_deg, tilt_deg=tilt_test, map_width=640, map_height=480
        )

        # Test image center
        u, v = W_px // 2, H_px // 2
        pt_img = np.array([[u], [v], [1.0]])
        pt_world = geo.H_inv @ pt_img

        if abs(pt_world[2, 0]) < 1e-6:
            print("  Image center: INVALID (near horizon)")
            continue

        Xw = pt_world[0, 0] / pt_world[2, 0]
        Yw = pt_world[1, 0] / pt_world[2, 0]
        dist = np.sqrt(Xw**2 + Yw**2)

        print("\n  Image Center Projection:")
        print(f"    World: ({Xw:7.2f}, {Yw:7.2f}) meters")
        print(f"    Distance: {dist:7.2f}m")

        # Expected distance for downward-pointing camera
        if tilt_test < 0:
            expected = height / np.tan(-np.radians(tilt_test))
            error = abs(dist - expected)
            print(f"    Expected: {expected:.2f}m (for tilt={tilt_test:.1f}°)")
            print(f"    Error: {error:.2f}m ({error / expected * 100:.1f}%)")

        # Check if projection makes sense
        print("\n  Sanity Checks:")
        if Yw > 0:
            print("    ✓ Y > 0: Point is AHEAD of camera")
        else:
            print("    ✗ Y < 0: Point is BEHIND camera (WRONG!)")

        # Test bottom vs top
        bottom_u, bottom_v = W_px // 2, H_px - 100
        top_u, top_v = W_px // 2, 200

        pt_bottom = geo.H_inv @ np.array([[bottom_u], [bottom_v], [1.0]])
        pt_top = geo.H_inv @ np.array([[top_u], [top_v], [1.0]])

        if abs(pt_bottom[2, 0]) > 1e-6 and abs(pt_top[2, 0]) > 1e-6:
            dist_bottom = np.sqrt(
                (pt_bottom[0, 0] / pt_bottom[2, 0]) ** 2 + (pt_bottom[1, 0] / pt_bottom[2, 0]) ** 2
            )
            dist_top = np.sqrt(
                (pt_top[0, 0] / pt_top[2, 0]) ** 2 + (pt_top[1, 0] / pt_top[2, 0]) ** 2
            )

            print(f"    Bottom of image distance: {dist_bottom:.2f}m")
            print(f"    Top of image distance: {dist_top:.2f}m")

            if dist_bottom < dist_top:
                print("    ✓ Bottom < Top: Correct (bottom is closer)")
            else:
                print("    ✗ Bottom > Top: WRONG (bottom should be closer)")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("\nThe test that shows:")
    print("  • Y > 0 (ahead of camera)")
    print("  • Bottom < Top (bottom closer than top)")
    print("  • Distance matches expected value")
    print("\n...is using the CORRECT tilt sign convention.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_tilt_inversion.py PAN TILT [HEIGHT]")
        print("Example: python test_tilt_inversion.py 65.6 34.2 5.0")
        sys.exit(1)

    pan = float(sys.argv[1])
    tilt = float(sys.argv[2])
    height = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    test_tilt_inversion(pan, tilt, height)
