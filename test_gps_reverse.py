#!/usr/bin/env python3
"""
Test reverse GPS calculation (homography X,Y -> GPS coordinates).
"""

from gps_distance_calculator import dms_to_dd, local_xy_to_gps, dd_to_dms, haversine_distance

# Your data
camera_gps = {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
point_gps_actual = {"lat": "39°38'25.6\"N", "lon": "0°13'48.4\"W"}

# Convert camera GPS to decimal degrees
cam_lat_dd = dms_to_dd(camera_gps["lat"])
cam_lon_dd = dms_to_dd(camera_gps["lon"])

print("="*70)
print("REVERSE GPS CALCULATION TEST")
print("="*70)

print(f"\nCamera GPS:")
print(f"  {camera_gps['lat']}, {camera_gps['lon']}")
print(f"  ({cam_lat_dd:.6f}°, {cam_lon_dd:.6f}°)")

print(f"\nActual Point GPS:")
print(f"  {point_gps_actual['lat']}, {point_gps_actual['lon']}")

# Homography reported distance
homography_distance = 3.44  # meters

# Since homography underestimated, let's test different scenarios

print(f"\n{'='*70}")
print("SCENARIO 1: Using homography X, Y from click")
print("="*70)
print(f"\nHomography reported: 3.44m distance")
print("Let's assume homography gave world coords: (?, ?) meters")
print("We need to estimate what X, Y the homography calculated...")
print("\nWithout the actual X, Y from homography, we can't reverse calculate.")
print("But the GPS verifier tool will show estimated GPS in real-time!")

print(f"\n{'='*70}")
print("SCENARIO 2: Calculate expected GPS if distance was correct")
print("="*70)

# Calculate actual GPS distance
pt_lat_dd = dms_to_dd(point_gps_actual["lat"])
pt_lon_dd = dms_to_dd(point_gps_actual["lon"])
actual_distance = haversine_distance(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)

print(f"\nActual GPS distance: {actual_distance:.2f}m")

# The bearing to the point
from gps_distance_calculator import bearing_between_points, gps_to_local_xy
bearing = bearing_between_points(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)
print(f"Bearing: {bearing:.1f}°")

# Get actual local X, Y
x_actual, y_actual = gps_to_local_xy(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)
print(f"Actual local coords: ({x_actual:.2f}, {y_actual:.2f}) meters")

# Test reverse: convert back to GPS
lat_reverse, lon_reverse = local_xy_to_gps(cam_lat_dd, cam_lon_dd, x_actual, y_actual)
lat_reverse_dms = dd_to_dms(lat_reverse, is_latitude=True)
lon_reverse_dms = dd_to_dms(lon_reverse, is_latitude=False)

print(f"\nReverse calculated GPS:")
print(f"  {lat_reverse_dms}, {lon_reverse_dms}")
print(f"  ({lat_reverse:.6f}°, {lon_reverse:.6f}°)")

# Check round-trip error
reverse_distance = haversine_distance(pt_lat_dd, pt_lon_dd, lat_reverse, lon_reverse)
print(f"\nRound-trip error: {reverse_distance:.3f} meters")

print(f"\n{'='*70}")
print("CONCLUSION")
print("="*70)
print("\nThe GPS verifier will:")
print("  1. Calculate world X, Y from image click (homography)")
print("  2. Convert X, Y → estimated GPS coordinates")
print("  3. Show estimated GPS to user")
print("  4. User enters actual GPS")
print("  5. Compare actual vs estimated GPS distance")
print("  6. Calculate correction factor for height")
print("\n✓ Ready to use: python verify_homography_gps.py Valte \"39°38'25.7\\\"N\" \"0°13'48.7\\\"W\"")
print("="*70)
