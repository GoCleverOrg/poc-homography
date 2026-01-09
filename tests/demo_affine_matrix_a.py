#!/usr/bin/env python3
"""
Demo script to showcase the affine matrix A functionality.

This script demonstrates:
1. Default A matrix (identity)
2. Computing A matrix from GeoTIFF parameters
3. Using A matrix for reference-to-world transformation
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry


def demo_default_a_matrix():
    """Demonstrate default A matrix behavior."""
    print("=" * 70)
    print("DEMO 1: Default A Matrix (Identity)")
    print("=" * 70)

    geo = CameraGeometry(1920, 1080)

    print("\nDefault A matrix (no GeoTIFF parameters):")
    print(geo.A)
    print("\nExpected: 3x3 identity matrix")
    print(f"Is identity: {np.allclose(geo.A, np.eye(3))}")


def demo_compute_a_matrix():
    """Demonstrate computing A matrix from GeoTIFF parameters."""
    print("\n" + "=" * 70)
    print("DEMO 2: Computing A Matrix from GeoTIFF Parameters")
    print("=" * 70)

    geo = CameraGeometry(1920, 1080)

    # Example: Reference ortho image with 0.5m/pixel resolution
    # Origin at UTM (500000, 4000000), camera at (500010, 4000020)
    geotiff_params = {
        "pixel_size_x": 0.5,  # 0.5 meters per pixel in X
        "pixel_size_y": 0.5,  # 0.5 meters per pixel in Y
        "origin_easting": 500000.0,  # UTM easting of image origin
        "origin_northing": 4000000.0,  # UTM northing of image origin
    }
    camera_utm_position = (500010.0, 4000020.0)  # Camera UTM position

    print("\nGeoTIFF Parameters:")
    print(
        f"  Pixel size: ({geotiff_params['pixel_size_x']}, {geotiff_params['pixel_size_y']}) m/pixel"
    )
    print(
        f"  Origin: ({geotiff_params['origin_easting']}, {geotiff_params['origin_northing']}) UTM"
    )
    print(f"  Camera position: {camera_utm_position} UTM")

    geo.set_geotiff_params(geotiff_params, camera_utm_position)

    print("\nComputed A matrix:")
    print(geo.A)

    # Verify the translation components
    t_x = geotiff_params["origin_easting"] - camera_utm_position[0]
    t_y = geotiff_params["origin_northing"] - camera_utm_position[1]
    print("\nTranslation components:")
    print(f"  t_x = {geotiff_params['origin_easting']} - {camera_utm_position[0]} = {t_x} m")
    print(f"  t_y = {geotiff_params['origin_northing']} - {camera_utm_position[1]} = {t_y} m")


def demo_transform_reference_to_world():
    """Demonstrate using A matrix to transform reference pixels to world coordinates."""
    print("\n" + "=" * 70)
    print("DEMO 3: Transform Reference Pixels to World Coordinates")
    print("=" * 70)

    geo = CameraGeometry(1920, 1080)

    # Setup GeoTIFF parameters
    geotiff_params = {
        "pixel_size_x": 0.5,
        "pixel_size_y": 0.5,
        "origin_easting": 500000.0,
        "origin_northing": 4000000.0,
    }
    camera_utm_position = (500010.0, 4000020.0)

    geo.set_geotiff_params(geotiff_params, camera_utm_position)

    # Transform some reference image pixels to world coordinates
    reference_pixels = [
        (0, 0),  # Origin of reference image
        (100, 0),  # 100 pixels to the right
        (0, 100),  # 100 pixels down
        (100, 100),  # Diagonal
    ]

    print("\nTransforming reference pixels to world coordinates:")
    print(f"{'Ref Pixel':<15} {'World UTM':<30} {'Offset from Camera':<25}")
    print("-" * 70)

    for px, py in reference_pixels:
        # Apply A matrix transformation: [X_world, Y_world, 1] = A @ [px, py, 1]
        ref_point = np.array([px, py, 1.0])
        world_point = geo.A @ ref_point

        # Normalize homogeneous coordinates
        x_world = world_point[0] / world_point[2]
        y_world = world_point[1] / world_point[2]

        # Calculate offset from camera position
        dx = x_world - camera_utm_position[0]
        dy = y_world - camera_utm_position[1]

        print(
            f"({px:3d}, {py:3d})      ({x_world:9.1f}, {y_world:9.1f})       ({dx:6.1f}m E, {dy:6.1f}m N)"
        )


def demo_backward_compatibility():
    """Demonstrate backward compatibility with None parameters."""
    print("\n" + "=" * 70)
    print("DEMO 4: Backward Compatibility (None Parameters)")
    print("=" * 70)

    geo = CameraGeometry(1920, 1080)

    print("\nCalling set_geotiff_params(None, None)...")
    geo.set_geotiff_params(None, None)

    print("A matrix after None parameters:")
    print(geo.A)
    print(f"Is identity: {np.allclose(geo.A, np.eye(3))}")

    print("\nThis ensures backward compatibility when GeoTIFF data is not available.")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("AFFINE MATRIX A DEMONSTRATION")
    print("Issue #122: Explicit affine map for reference-to-world transformation")
    print("*" * 70)

    demo_default_a_matrix()
    demo_compute_a_matrix()
    demo_transform_reference_to_world()
    demo_backward_compatibility()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
