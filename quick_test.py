#!/usr/bin/env python3
"""
Quick homography sanity check - run this first!
Tests the math without needing physical markers or camera access.
"""

import numpy as np
from camera_geometry import CameraGeometry


def quick_test():
    """Quick homography sanity check."""
    print("\n" + "="*70)
    print("QUICK HOMOGRAPHY SANITY CHECK")
    print("="*70)

    # Setup a typical camera configuration
    print("\nSetting up camera geometry:")
    print("  Resolution: 2560x1440")
    print("  Height: 5.0m")
    print("  Pan: 0° (facing forward)")
    print("  Tilt: -45° (pointing down)")
    print("  Zoom: 1.0x")

    geo = CameraGeometry(2560, 1440)
    K = geo.get_intrinsics(zoom_factor=1.0, W_px=2560, H_px=1440)
    w_pos = np.array([0.0, 0.0, 5.0])

    geo.set_camera_parameters(
        K=K,
        w_pos=w_pos,
        pan_deg=0.0,
        tilt_deg=-45.0,
        map_width=640,
        map_height=480
    )

    print("\n" + "-"*70)
    print("TEST 1: Image Center Projection")
    print("-"*70)

    # Test image center
    cx, cy = 2560/2, 1440/2
    pt_image = np.array([[cx], [cy], [1.0]])
    pt_world = geo.H_inv @ pt_image

    Xw = pt_world[0, 0] / pt_world[2, 0]
    Yw = pt_world[1, 0] / pt_world[2, 0]
    distance = np.sqrt(Xw**2 + Yw**2)

    # At 45° tilt and 5m height, center should project to ~5m ahead
    expected = 5.0
    error_pct = abs(distance - expected) / expected * 100

    print(f"\nImage center ({cx:.0f}, {cy:.0f})px projects to:")
    print(f"  World: ({Xw:.2f}, {Yw:.2f}) meters")
    print(f"  Distance: {distance:.2f}m")
    print(f"  Expected: ~{expected:.2f}m (at -45° tilt)")
    print(f"  Error: {error_pct:.1f}%")

    if error_pct < 10:
        print("  ✓ PASS - Center projection looks correct")
    else:
        print("  ✗ FAIL - Center projection seems wrong")

    print("\n" + "-"*70)
    print("TEST 2: Bottom of Image (Near Field)")
    print("-"*70)

    # Test bottom center (near field)
    u, v = 2560/2, 1440 - 100
    pt_image = np.array([[u], [v], [1.0]])
    pt_world = geo.H_inv @ pt_image

    Xw = pt_world[0, 0] / pt_world[2, 0]
    Yw = pt_world[1, 0] / pt_world[2, 0]
    distance = np.sqrt(Xw**2 + Yw**2)

    print(f"\nBottom of image ({u:.0f}, {v:.0f})px projects to:")
    print(f"  World: ({Xw:.2f}, {Yw:.2f}) meters")
    print(f"  Distance: {distance:.2f}m")

    if 1.0 < distance < 5.0:
        print("  ✓ PASS - Bottom projects to near field")
    else:
        print("  ✗ FAIL - Bottom should be closer than center")

    print("\n" + "-"*70)
    print("TEST 3: Top of Image (Far Field)")
    print("-"*70)

    # Test top center (far field / horizon)
    u, v = 2560/2, 100
    pt_image = np.array([[u], [v], [1.0]])
    pt_world = geo.H_inv @ pt_image

    Xw = pt_world[0, 0] / pt_world[2, 0]
    Yw = pt_world[1, 0] / pt_world[2, 0]
    distance = np.sqrt(Xw**2 + Yw**2)

    print(f"\nTop of image ({u:.0f}, {v:.0f})px projects to:")
    print(f"  World: ({Xw:.2f}, {Yw:.2f}) meters")
    print(f"  Distance: {distance:.2f}m")

    if distance > 10.0:
        print("  ✓ PASS - Top projects to far field")
    else:
        print("  ✗ FAIL - Top should be farther than center")

    print("\n" + "-"*70)
    print("TEST 4: Round-Trip Consistency")
    print("-"*70)

    # Test round-trip projection
    test_points = [
        (0.0, 5.0, "5m ahead"),
        (3.0, 10.0, "10m ahead, 3m right"),
        (-2.0, 8.0, "8m ahead, 2m left"),
    ]

    max_error = 0.0
    for Xw_orig, Yw_orig, desc in test_points:
        # World → Image
        pt_world = np.array([[Xw_orig], [Yw_orig], [1.0]])
        pt_image = geo.H @ pt_world
        u = pt_image[0, 0] / pt_image[2, 0]
        v = pt_image[1, 0] / pt_image[2, 0]

        # Image → World
        pt_image_norm = np.array([[u], [v], [1.0]])
        pt_world_recovered = geo.H_inv @ pt_image_norm
        Xw_recovered = pt_world_recovered[0, 0] / pt_world_recovered[2, 0]
        Yw_recovered = pt_world_recovered[1, 0] / pt_world_recovered[2, 0]

        error = np.sqrt((Xw_orig - Xw_recovered)**2 + (Yw_orig - Yw_recovered)**2)
        max_error = max(max_error, error)

        status = "✓" if error < 0.01 else "✗"
        print(f"\n  {desc}:")
        print(f"    Original: ({Xw_orig:.2f}, {Yw_orig:.2f})m")
        print(f"    Recovered: ({Xw_recovered:.6f}, {Yw_recovered:.6f})m")
        print(f"    Error: {error:.6f}m {status}")

    print(f"\n  Maximum round-trip error: {max_error:.6f}m")
    if max_error < 0.01:
        print("  ✓ PASS - Round-trip consistency excellent")
    else:
        print("  ✗ FAIL - Round-trip error too large")

    print("\n" + "-"*70)
    print("TEST 5: Homography Properties")
    print("-"*70)

    det_H = np.linalg.det(geo.H)
    det_H_inv = np.linalg.det(geo.H_inv)
    identity_test = np.allclose(geo.H @ geo.H_inv, np.eye(3), atol=1e-6)

    print(f"\n  det(H) = {det_H:.2e}")
    print(f"  det(H_inv) = {det_H_inv:.2e}")
    print(f"  H @ H_inv = I? {identity_test}")

    checks = []
    checks.append(("det(H) > 0", det_H > 1e-6, "✓" if det_H > 1e-6 else "✗"))
    checks.append(("det(H_inv) > 0", det_H_inv > 1e-6, "✓" if det_H_inv > 1e-6 else "✗"))
    checks.append(("H @ H_inv = I", identity_test, "✓" if identity_test else "✗"))

    print("\n  Checks:")
    for desc, result, symbol in checks:
        print(f"    {symbol} {desc}: {result}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_pass = (
        error_pct < 10 and
        max_error < 0.01 and
        det_H > 1e-6 and
        identity_test
    )

    if all_pass:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYour homography implementation looks correct.")
        print("\nNext steps:")
        print("  1. Run: python tests/test_homography_consistency.py")
        print("  2. Run: python verify_homography.py Valte")
        print("  3. Test with real camera stream and physical markers")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease check:")
        print("  - Rotation matrix order (should be Pan then Tilt)")
        print("  - Homography formula: H = K @ [r1, r2, t]")
        print("  - Camera coordinate system conventions")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    quick_test()
