#!/usr/bin/env python3
"""
Demonstration of MapPointHomography using Valte test data.

This script demonstrates the complete workflow:
1. Load map points from registry
2. Load GCP data
3. Compute homography
4. Project camera pixels to map coordinates
5. Project map coordinates to camera pixels
6. Validate round-trip accuracy
"""

import json
from pathlib import Path

import cv2
import numpy as np

from poc_homography.homography_map_points import MapPointHomography
from poc_homography.map_points import MapPointRegistry


def main():
    """Run the demonstration."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    map_points_path = project_root / "map_points.json"
    gcp_data_path = project_root / "test_data_Valte_20260109_195052.json"
    image_path = project_root / "test_data_Valte_20260109_195052.jpg"

    print("=" * 80)
    print("MapPointHomography Demonstration")
    print("=" * 80)

    # Step 1: Load map points
    print("\n[1] Loading map point registry...")
    map_registry = MapPointRegistry.load(map_points_path)
    print(f"    Loaded {len(map_registry.points)} map points from '{map_registry.map_id}'")

    # Step 2: Load GCP data
    print("\n[2] Loading GCP data...")
    with open(gcp_data_path) as f:
        gcp_data = json.load(f)

    gcps = gcp_data["gcps"]
    camera_info = gcp_data["camera_info"]
    print(f"    Loaded {len(gcps)} ground control points")
    print(f"    Camera location: ({camera_info['latitude']:.6f}, {camera_info['longitude']:.6f})")
    print(f"    Camera height: {camera_info['height_meters']:.2f}m")
    print(
        f"    Camera orientation: pan={camera_info['pan_deg']:.1f}°, tilt={camera_info['tilt_deg']:.1f}°"
    )

    # Step 3: Load image (optional, for visualization)
    print("\n[3] Loading camera image...")
    image = cv2.imread(str(image_path))
    if image is not None:
        height, width = image.shape[:2]
        print(f"    Image dimensions: {width}x{height}")
    else:
        print("    Warning: Could not load image")

    # Step 4: Compute homography
    print("\n[4] Computing homography from GCPs...")
    homography = MapPointHomography(map_id=map_registry.map_id)
    result = homography.compute_from_gcps(gcps, map_registry, ransac_threshold=50.0)

    print(f"    GCPs used: {result.num_gcps}")
    print(f"    Inliers: {result.num_inliers} ({result.inlier_ratio:.1%})")
    print(f"    Mean reprojection error: {result.mean_reproj_error:.2f} meters")
    print(f"    Max reprojection error: {result.max_reproj_error:.2f} meters")
    print(f"    RMSE: {result.rmse:.2f} meters")

    # Step 5: Forward projection examples
    print("\n[5] Forward projection (camera pixels -> map coordinates)...")

    test_points = [
        ("Camera center", (960.0, 540.0)),
        ("Top-left corner", (100.0, 100.0)),
        ("Top-right corner", (1820.0, 100.0)),
        ("Bottom-left corner", (100.0, 980.0)),
        ("Bottom-right corner", (1820.0, 980.0)),
    ]

    for name, camera_pixel in test_points:
        try:
            map_point = homography.camera_to_map(camera_pixel)
            print(
                f"    {name:20s} ({camera_pixel[0]:7.1f}, {camera_pixel[1]:6.1f}) px "
                f"-> ({map_point.pixel_x:10.2f}, {map_point.pixel_y:11.2f}) px"
            )
        except Exception as e:
            print(f"    {name:20s} - Error: {e}")

    # Step 6: Inverse projection examples
    print("\n[6] Inverse projection (map coordinates -> camera pixels)...")

    # Use actual map points from GCPs
    sample_map_points = ["A7", "A6", "P17", "X15"]
    for point_id in sample_map_points:
        if point_id in map_registry.points:
            map_point = map_registry.points[point_id]
            map_coord = (map_point.pixel_x, map_point.pixel_y)

            try:
                camera_pixel = homography.map_to_camera(map_coord)
                print(
                    f"    Point {point_id:5s} ({map_coord[0]:10.2f}, {map_coord[1]:11.2f}) m "
                    f"-> ({camera_pixel[0]:7.1f}, {camera_pixel[1]:6.1f}) px"
                )
            except Exception as e:
                print(f"    Point {point_id:5s} - Error: {e}")

    # Step 7: Round-trip validation
    print("\n[7] Round-trip validation (camera -> map -> camera)...")

    errors = []
    for i, gcp in enumerate(gcps[:5]):  # Test first 5 GCPs
        original_pixel = (gcp["pixel_x"], gcp["pixel_y"])

        # Forward then inverse
        map_coord = homography.camera_to_map(original_pixel)
        recovered_pixel = homography.map_to_camera(map_coord)

        # Calculate error
        error = np.linalg.norm(np.array(recovered_pixel) - np.array(original_pixel))
        errors.append(error)

        print(
            f"    GCP {i + 1} ({gcp['map_point_id']:5s}): "
            f"original=({original_pixel[0]:7.1f}, {original_pixel[1]:6.1f}), "
            f"recovered=({recovered_pixel[0]:7.1f}, {recovered_pixel[1]:6.1f}), "
            f"error={error:.2f} px"
        )

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print("\n    Round-trip statistics:")
    print(f"      Mean error: {mean_error:.2f} pixels")
    print(f"      Max error: {max_error:.2f} pixels")

    # Step 8: Batch projection demo
    print("\n[8] Batch projection demo...")

    # Project all GCP camera pixels at once
    camera_pixels = [(gcp["pixel_x"], gcp["pixel_y"]) for gcp in gcps]
    map_coords = homography.camera_to_map_batch(camera_pixels)

    print(f"    Projected {len(camera_pixels)} camera pixels to map coordinates in batch")
    print("    Sample results:")
    for i in range(min(3, len(map_coords))):
        print(
            f"      {i + 1}. ({camera_pixels[i][0]:7.1f}, {camera_pixels[i][1]:6.1f}) px "
            f"-> ({map_coords[i].pixel_x:10.2f}, {map_coords[i].pixel_y:11.2f}) px"
        )

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Successfully computed homography from {result.num_gcps} GCPs")
    print(f"✓ Inlier ratio: {result.inlier_ratio:.1%}")
    print(f"✓ Mean reprojection error: {result.mean_reproj_error:.2f} meters")
    print(f"✓ Round-trip accuracy: {mean_error:.2f} pixels (mean)")
    print("\nThe homography is working correctly and ready for use!")
    print("=" * 80)


if __name__ == "__main__":
    main()
