#!/usr/bin/env python3
"""
Debug tool to trace Y scaling issues in the coordinate transformation pipeline.

This traces a point through every step of the transformation to identify
where the Y scaling mismatch occurs.
"""

import math
from pyproj import Transformer

# Georeferencing parameters
ORIGIN_EASTING = 737575.05
ORIGIN_NORTHING = 4391595.45
GSD = 0.15  # meters per pixel
UTM_CRS = "EPSG:25830"

# Image dimensions
IMAGE_WIDTH = 1681
IMAGE_HEIGHT = 916


def trace_pixel_to_all(px: float, py: float):
    """Trace a pixel coordinate through all transformations."""

    print(f"\n{'='*70}")
    print(f"TRACING PIXEL ({px}, {py})")
    print(f"{'='*70}")

    # Step 1: Pixel to UTM
    print("\n1. PIXEL → UTM")
    print("-" * 50)

    # Method A: As implemented in extract_kml_points.py
    # easting = origin_easting + (px * gsd)
    # northing = origin_northing + (py * -gsd)  # NEGATIVE because Y is inverted

    easting_a = ORIGIN_EASTING + (px * GSD)
    northing_a = ORIGIN_NORTHING + (py * -GSD)

    print(f"   Method A (current - negative Y GSD):")
    print(f"   easting  = {ORIGIN_EASTING} + ({px} * {GSD}) = {easting_a:.4f}")
    print(f"   northing = {ORIGIN_NORTHING} + ({py} * -{GSD}) = {northing_a:.4f}")

    # Method B: Alternative interpretation
    # What if the origin is at BOTTOM-left, not TOP-left?
    northing_b = ORIGIN_NORTHING - (IMAGE_HEIGHT * GSD) + (py * GSD)

    print(f"\n   Method B (if origin is image extent bottom-left):")
    print(f"   northing = {ORIGIN_NORTHING} - ({IMAGE_HEIGHT} * {GSD}) + ({py} * {GSD})")
    print(f"   northing = {ORIGIN_NORTHING - IMAGE_HEIGHT * GSD:.4f} + {py * GSD:.4f} = {northing_b:.4f}")

    # Method C: What if Y=0 is at bottom of image?
    py_from_bottom = IMAGE_HEIGHT - py
    northing_c = ORIGIN_NORTHING + (py_from_bottom * -GSD)

    print(f"\n   Method C (if pixel Y=0 is at bottom):")
    print(f"   py_from_bottom = {IMAGE_HEIGHT} - {py} = {py_from_bottom}")
    print(f"   northing = {ORIGIN_NORTHING} + ({py_from_bottom} * -{GSD}) = {northing_c:.4f}")

    # Step 2: UTM to GPS
    print("\n2. UTM → GPS (WGS84)")
    print("-" * 50)

    transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)

    lon_a, lat_a = transformer.transform(easting_a, northing_a)
    lon_b, lat_b = transformer.transform(easting_a, northing_b)
    lon_c, lat_c = transformer.transform(easting_a, northing_c)

    print(f"   Method A: ({lat_a:.8f}, {lon_a:.8f})")
    print(f"   Method B: ({lat_b:.8f}, {lon_b:.8f})")
    print(f"   Method C: ({lat_c:.8f}, {lon_c:.8f})")

    # Step 3: Check the reverse transformation
    print("\n3. REVERSE: GPS → UTM → Pixel")
    print("-" * 50)

    reverse_transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

    for method, (lat, lon) in [("A", (lat_a, lon_a)), ("B", (lat_b, lon_b)), ("C", (lat_c, lon_c))]:
        e, n = reverse_transformer.transform(lon, lat)

        # Reverse pixel calculation (Method A formula inverted)
        px_back = (e - ORIGIN_EASTING) / GSD
        py_back = (n - ORIGIN_NORTHING) / -GSD

        print(f"   Method {method}:")
        print(f"     UTM: ({e:.4f}, {n:.4f})")
        print(f"     Pixel: ({px_back:.2f}, {py_back:.2f})")
        print(f"     Error: ({px_back - px:.4f}, {py_back - py:.4f})")

    return easting_a, northing_a, lat_a, lon_a


def check_origin_interpretation():
    """Check what the origin coordinates actually mean."""

    print(f"\n{'='*70}")
    print("ORIGIN INTERPRETATION ANALYSIS")
    print(f"{'='*70}")

    print(f"\nGiven:")
    print(f"  Origin Easting:  {ORIGIN_EASTING}")
    print(f"  Origin Northing: {ORIGIN_NORTHING}")
    print(f"  GSD: {GSD} m/pixel")
    print(f"  Image: {IMAGE_WIDTH} x {IMAGE_HEIGHT} pixels")

    # Calculate image extent
    extent_east = ORIGIN_EASTING + (IMAGE_WIDTH * GSD)
    extent_north_if_topleft = ORIGIN_NORTHING - (IMAGE_HEIGHT * GSD)
    extent_north_if_bottomleft = ORIGIN_NORTHING + (IMAGE_HEIGHT * GSD)

    print(f"\nImage extent (E-W):")
    print(f"  West edge:  E = {ORIGIN_EASTING:.2f}")
    print(f"  East edge:  E = {extent_east:.2f}")
    print(f"  Width: {extent_east - ORIGIN_EASTING:.2f} m")

    print(f"\nImage extent (N-S) - IF origin is TOP-LEFT:")
    print(f"  North edge: N = {ORIGIN_NORTHING:.2f}")
    print(f"  South edge: N = {extent_north_if_topleft:.2f}")
    print(f"  Height: {ORIGIN_NORTHING - extent_north_if_topleft:.2f} m")

    print(f"\nImage extent (N-S) - IF origin is BOTTOM-LEFT:")
    print(f"  South edge: N = {ORIGIN_NORTHING:.2f}")
    print(f"  North edge: N = {extent_north_if_bottomleft:.2f}")
    print(f"  Height: {extent_north_if_bottomleft - ORIGIN_NORTHING:.2f} m")

    # Convert corners to GPS
    transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)

    print(f"\nCorner coordinates (assuming TOP-LEFT origin):")
    corners = [
        (0, 0, "Top-Left"),
        (IMAGE_WIDTH, 0, "Top-Right"),
        (0, IMAGE_HEIGHT, "Bottom-Left"),
        (IMAGE_WIDTH, IMAGE_HEIGHT, "Bottom-Right"),
    ]

    for px, py, name in corners:
        e = ORIGIN_EASTING + (px * GSD)
        n = ORIGIN_NORTHING + (py * -GSD)
        lon, lat = transformer.transform(e, n)
        print(f"  {name:15s} pixel({px:4d},{py:4d}) → UTM({e:.2f}, {n:.2f}) → GPS({lat:.6f}, {lon:.6f})")


def analyze_gsd_sign():
    """Analyze the effect of GSD sign on Y coordinates."""

    print(f"\n{'='*70}")
    print("GSD SIGN ANALYSIS")
    print(f"{'='*70}")

    print("""
In GeoTIFF/raster conventions:
- Origin is typically at TOP-LEFT corner
- X increases to the RIGHT (positive GSD in X)
- Y increases DOWNWARD in pixel space
- But in geographic space, Northing increases UPWARD
- Therefore, GSD in Y should be NEGATIVE

The geotransform is typically:
  [origin_x, pixel_width, 0, origin_y, 0, -pixel_height]

Where:
  - origin_x, origin_y = coordinates of TOP-LEFT corner
  - pixel_width = GSD (positive)
  - pixel_height = GSD (the sign makes Y negative)

Current implementation:
  easting  = origin_easting + px * GSD        (correct)
  northing = origin_northing + py * (-GSD)    (correct if origin is top-left)
""")

    # Test with center point
    px, py = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2

    e = ORIGIN_EASTING + px * GSD
    n_neg = ORIGIN_NORTHING + py * (-GSD)
    n_pos = ORIGIN_NORTHING + py * GSD

    print(f"For center pixel ({px}, {py}):")
    print(f"  With -GSD: northing = {n_neg:.2f}")
    print(f"  With +GSD: northing = {n_pos:.2f}")
    print(f"  Difference: {abs(n_pos - n_neg):.2f} m ({abs(n_pos - n_neg)/GSD:.0f} pixels)")


def check_capture_gcps_local_xy():
    """Check how capture_gcps_web.py computes local XY from GPS."""

    print(f"\n{'='*70}")
    print("CAPTURE_GCPS_WEB.PY LOCAL XY COMPUTATION")
    print(f"{'='*70}")

    # Simulate what capture_gcps_web.py does
    # It converts GPS to local XY relative to camera position

    # Let's use the image center as camera position for this test
    cam_px, cam_py = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
    cam_e = ORIGIN_EASTING + cam_px * GSD
    cam_n = ORIGIN_NORTHING + cam_py * (-GSD)

    transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    cam_lon, cam_lat = transformer.transform(cam_e, cam_n)

    print(f"\nCamera position (image center):")
    print(f"  Pixel: ({cam_px}, {cam_py})")
    print(f"  UTM: E={cam_e:.2f}, N={cam_n:.2f}")
    print(f"  GPS: ({cam_lat:.6f}, {cam_lon:.6f})")

    # Test point: 100 pixels to the right and down from camera
    test_px, test_py = cam_px + 100, cam_py + 100
    test_e = ORIGIN_EASTING + test_px * GSD
    test_n = ORIGIN_NORTHING + test_py * (-GSD)
    test_lon, test_lat = transformer.transform(test_e, test_n)

    print(f"\nTest point (100px right, 100px down from camera):")
    print(f"  Pixel: ({test_px}, {test_py})")
    print(f"  UTM: E={test_e:.2f}, N={test_n:.2f}")
    print(f"  GPS: ({test_lat:.6f}, {test_lon:.6f})")

    # Expected local XY (in UTM space)
    expected_x = test_e - cam_e  # Should be +15m (100 * 0.15)
    expected_y = test_n - cam_n  # Should be -15m (100 * -0.15)

    print(f"\nExpected local XY (UTM difference):")
    print(f"  X = {test_e:.2f} - {cam_e:.2f} = {expected_x:.2f} m")
    print(f"  Y = {test_n:.2f} - {cam_n:.2f} = {expected_y:.2f} m")

    # What equirectangular gives us
    from math import radians, cos
    EARTH_RADIUS = 6371000.0

    delta_lat = radians(test_lat - cam_lat)
    delta_lon = radians(test_lon - cam_lon)
    avg_lat = radians((cam_lat + test_lat) / 2)

    equirect_x = delta_lon * cos(avg_lat) * EARTH_RADIUS
    equirect_y = delta_lat * EARTH_RADIUS

    print(f"\nEquirectangular local XY:")
    print(f"  X = {equirect_x:.2f} m")
    print(f"  Y = {equirect_y:.2f} m")

    print(f"\nDiscrepancy (Equirect - UTM):")
    print(f"  ΔX = {equirect_x - expected_x:.4f} m ({(equirect_x - expected_x) / expected_x * 100:.2f}%)")
    print(f"  ΔY = {equirect_y - expected_y:.4f} m ({(equirect_y - expected_y) / expected_y * 100:.2f}%)")

    # Now check what UTMConverter gives us
    print(f"\n--- Using UTMConverter ---")

    reverse_transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

    # Set camera as reference
    ref_e, ref_n = reverse_transformer.transform(cam_lon, cam_lat)
    print(f"Camera UTM (from GPS): E={ref_e:.4f}, N={ref_n:.4f}")
    print(f"Camera UTM (direct):   E={cam_e:.4f}, N={cam_n:.4f}")
    print(f"Difference: ΔE={ref_e - cam_e:.6f}, ΔN={ref_n - cam_n:.6f}")

    # Convert test point GPS to UTM
    test_e_from_gps, test_n_from_gps = reverse_transformer.transform(test_lon, test_lat)
    print(f"\nTest point UTM (from GPS): E={test_e_from_gps:.4f}, N={test_n_from_gps:.4f}")
    print(f"Test point UTM (direct):   E={test_e:.4f}, N={test_n:.4f}")
    print(f"Difference: ΔE={test_e_from_gps - test_e:.6f}, ΔN={test_n_from_gps - test_n:.6f}")

    # Local XY via UTM from GPS
    utm_local_x = test_e_from_gps - ref_e
    utm_local_y = test_n_from_gps - ref_n

    print(f"\nUTM local XY (from GPS round-trip):")
    print(f"  X = {utm_local_x:.4f} m")
    print(f"  Y = {utm_local_y:.4f} m")

    print(f"\nFinal comparison:")
    print(f"  Expected (direct UTM):  X={expected_x:.4f}, Y={expected_y:.4f}")
    print(f"  UTM from GPS:           X={utm_local_x:.4f}, Y={utm_local_y:.4f}")
    print(f"  Equirectangular:        X={equirect_x:.4f}, Y={equirect_y:.4f}")


if __name__ == '__main__':
    check_origin_interpretation()
    analyze_gsd_sign()

    # Trace specific pixels
    trace_pixel_to_all(0, 0)      # Top-left
    trace_pixel_to_all(840, 458)  # Center
    trace_pixel_to_all(100, 100)  # Near top-left

    check_capture_gcps_local_xy()
