#!/usr/bin/env python3
"""
Web-based GCP Capture Tool - Interactive Ground Control Point collection.

This utility provides a browser-based interface for capturing Ground Control Points
(GCPs) from a camera frame with:
- High-precision zoom and pan capabilities
- Real-time distribution quality feedback
- Quadrant coverage visualization
- Interactive GPS coordinate entry
- Drag-to-reposition existing GCPs for fine-tuning
- Load existing YAML configs and continue adding points

Usage:
    # Capture from live camera
    python tools/capture_gcps_web.py Valte

    # Use existing frame image
    python tools/capture_gcps_web.py --frame path/to/frame.jpg

    # Load existing YAML and continue editing
    python tools/capture_gcps_web.py --frame path/to/frame.jpg --load existing_gcps.yaml

    # Specify output file
    python tools/capture_gcps_web.py Valte --output my_gcps.yaml
"""

import argparse
import base64
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.server_utils import find_available_port

# Camera config is optional
try:
    from poc_homography.camera_config import (
        get_camera_by_name, get_rtsp_url, USERNAME, PASSWORD, CAMERAS
    )
    CAMERA_CONFIG_AVAILABLE = True
except (ValueError, ImportError) as e:
    CAMERA_CONFIG_AVAILABLE = False
    print(f"Note: Camera config not available ({e})")
    print("  Use --frame to load an existing image file.\n")

# Import the intrinsics utility if available
try:
    from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics
    INTRINSICS_AVAILABLE = True
except ImportError:
    INTRINSICS_AVAILABLE = False

# Import camera defaults from config
try:
    from poc_homography.camera_config import (
        DEFAULT_SENSOR_WIDTH_MM,
        DEFAULT_BASE_FOCAL_LENGTH_MM
    )
except ImportError:
    # Fallback values if config import fails
    DEFAULT_SENSOR_WIDTH_MM = 6.78  # Calculated from 59.8° FOV at 5.9mm focal length
    DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9

# Reprojection error thresholds for feedback
REPROJ_ERROR_GOOD = 5.0  # pixels - considered good fit
REPROJ_ERROR_WARNING = 10.0  # pixels - warning threshold
REPROJ_ERROR_BAD = 20.0  # pixels - likely outlier

# RANSAC threshold used in cv2.findHomography() for outlier detection
# Note: Outlier removal now uses RANSAC's inlier mask directly, not a separate threshold.
# The RANSAC threshold (5.0px) is passed directly to cv2.findHomography() in update_homography().
RANSAC_REPROJ_THRESHOLD = 5.0  # pixels - matches cv2.findHomography() call

# Import for reprojection error calculation
try:
    from poc_homography.coordinate_converter import (
        gps_to_local_xy,
        GCPCoordinateConverter,
        DEFAULT_UTM_CRS
    )
    COORDINATE_CONVERTER_AVAILABLE = True
except ImportError:
    COORDINATE_CONVERTER_AVAILABLE = False
    GCPCoordinateConverter = None
    DEFAULT_UTM_CRS = "EPSG:25830"

# Import for GPS precision analysis and duplicate detection
try:
    from poc_homography.gcp_validation import analyze_gps_precision, detect_duplicate_gcps
    GCP_VALIDATION_AVAILABLE = True
except ImportError:
    GCP_VALIDATION_AVAILABLE = False

# Import for map-first mode camera parameter retrieval
try:
    from poc_homography.gps_distance_calculator import dms_to_dd
    from poc_homography.camera_geometry import CameraGeometry
    GPS_CONVERTER_AVAILABLE = True
except ImportError:
    GPS_CONVERTER_AVAILABLE = False

# Import for height calibration verification
try:
    from poc_homography.height_calibration import HeightCalibrator
    HEIGHT_CALIBRATOR_AVAILABLE = True
except ImportError:
    HEIGHT_CALIBRATOR_AVAILABLE = False

# CameraGeometry is available when GPS_CONVERTER_AVAILABLE is True
GEOMETRY_AVAILABLE = GPS_CONVERTER_AVAILABLE


def parse_kml_points(kml_path: str) -> List[Dict]:
    """
    Parse KML file and extract Point placemarks with GPS and optional UTM coordinates.

    This function parses KML 2.2 compliant files and extracts all Point placemarks,
    ignoring LineString and Polygon geometries. It handles common edge cases like
    missing names, invalid coordinates, and empty files.

    If the KML contains ExtendedData with UTM coordinates (from extract_kml_points.py),
    those are also extracted for more accurate coordinate conversion.

    Args:
        kml_path: Path to the KML file to parse

    Returns:
        List of dictionaries with structure:
        [
            {
                'name': str,
                'latitude': float,
                'longitude': float,
                'utm_easting': float (optional),
                'utm_northing': float (optional),
                'utm_crs': str (optional)
            },
            ...
        ]

    Example:
        >>> points = parse_kml_points('gcps.kml')
        >>> print(points)
        [{'name': 'P#01', 'latitude': 39.640600, 'longitude': -0.230200}]

    Edge cases handled:
        - Missing name element: uses "Unnamed Point N" (1-based index)
        - Missing Point element: skips the placemark
        - Invalid coordinates format: skips the placemark and logs warning
        - Empty KML file: returns empty list
    """
    # KML 2.2 namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Warning: KML file not found: {kml_path}")
        return []
    except ET.ParseError as e:
        print(f"Warning: Failed to parse KML file {kml_path}: {e}")
        return []

    points = []
    unnamed_counter = 1
    has_utm = False

    # Find all Placemark elements (handles both with and without namespace)
    placemarks = root.findall('.//kml:Placemark', ns)
    if not placemarks:
        # Try without namespace (for KML files without proper namespace)
        placemarks = root.findall('.//Placemark')

    for placemark in placemarks:
        # Check if this placemark contains a Point (not LineString or Polygon)
        point_elem = placemark.find('.//kml:Point', ns)
        if point_elem is None:
            point_elem = placemark.find('.//Point')

        if point_elem is None:
            # Skip placemarks without Point geometry (LineString, Polygon, etc.)
            continue

        # Extract name
        name_elem = placemark.find('.//kml:name', ns)
        if name_elem is None:
            name_elem = placemark.find('.//name')

        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip()
        else:
            name = f"Unnamed Point {unnamed_counter}"
            unnamed_counter += 1

        # Extract coordinates
        coords_elem = point_elem.find('.//kml:coordinates', ns)
        if coords_elem is None:
            coords_elem = point_elem.find('.//coordinates')

        if coords_elem is None or not coords_elem.text:
            print(f"Warning: Skipping placemark '{name}' - missing coordinates")
            continue

        # Parse coordinates string: "longitude,latitude,altitude" or "longitude,latitude"
        coords_text = coords_elem.text.strip()
        try:
            parts = coords_text.split(',')
            if len(parts) < 2:
                raise ValueError("Not enough coordinate values")

            longitude = float(parts[0])
            latitude = float(parts[1])
            # Altitude (parts[2]) is ignored if present

            point_data = {
                'name': name,
                'latitude': latitude,
                'longitude': longitude
            }

            # Try to extract UTM coordinates from ExtendedData
            extended_data = placemark.find('.//kml:ExtendedData', ns)
            if extended_data is None:
                extended_data = placemark.find('.//ExtendedData')

            if extended_data is not None:
                # Look for SimpleData elements with UTM coordinates
                simple_data_elems = extended_data.findall('.//kml:SimpleData', ns)
                if not simple_data_elems:
                    simple_data_elems = extended_data.findall('.//SimpleData')

                for sd in simple_data_elems:
                    sd_name = sd.get('name')
                    sd_value = sd.text
                    if sd_name and sd_value:
                        if sd_name == 'utm_easting':
                            point_data['utm_easting'] = float(sd_value)
                            has_utm = True
                        elif sd_name == 'utm_northing':
                            point_data['utm_northing'] = float(sd_value)
                            has_utm = True
                        elif sd_name == 'utm_crs':
                            point_data['utm_crs'] = sd_value

            points.append(point_data)

        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping placemark '{name}' - invalid coordinates format: {coords_text} ({e})")
            continue

    # Debug: Print parsed KML points
    print("\n" + "=" * 60)
    print("KML POINTS PARSED")
    print("=" * 60)
    print(f"File: {kml_path}")
    print(f"Total points found: {len(points)}")
    print(f"UTM coordinates available: {'Yes' if has_utm else 'No'}")
    print("-" * 60)
    for i, p in enumerate(points):
        utm_str = ""
        if 'utm_easting' in p and 'utm_northing' in p:
            utm_str = f" UTM: E={p['utm_easting']:.2f}, N={p['utm_northing']:.2f}"
        print(f"  {i+1}. {p['name']:<20} lat={p['latitude']:.6f}, lon={p['longitude']:.6f}{utm_str}")
    print("=" * 60 + "\n")

    return points


def get_camera_params_for_projection(camera_name: str, image_width: int = None, image_height: int = None) -> Dict:
    """
    Retrieve camera parameters for map-first mode projection.

    This function gathers all parameters needed to project map coordinates to camera view:
    - Camera GPS position (converted from DMS to decimal degrees)
    - Camera height above ground
    - Current PTZ status (pan, tilt, zoom) from live camera
    - Camera intrinsic matrix K computed from current zoom level

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")
        image_width: Width of the actual captured frame in pixels (optional, defaults to 1920)
        image_height: Height of the actual captured frame in pixels (optional, defaults to 1080)

    Returns:
        Dictionary with structure:
        {
            'camera_lat': float,      # decimal degrees
            'camera_lon': float,      # decimal degrees
            'height_m': float,        # meters
            'pan_deg': float,         # degrees
            'tilt_deg': float,        # degrees
            'zoom': float,            # zoom factor
            'K': np.ndarray,          # 3x3 intrinsic matrix
            'image_width': int,       # from parameter or default 1920
            'image_height': int       # from parameter or default 1080
        }

    Raises:
        ValueError: If camera not found or required modules not available
        RuntimeError: If cannot connect to camera or retrieve PTZ status

    Example:
        >>> params = get_camera_params_for_projection("Valte", 1920, 1080)
        >>> print(f"Camera at ({params['camera_lat']:.6f}, {params['camera_lon']:.6f})")
        >>> print(f"Pan: {params['pan_deg']:.1f}°, Tilt: {params['tilt_deg']:.1f}°")
    """
    # Check required modules
    if not CAMERA_CONFIG_AVAILABLE:
        raise ValueError(
            "Camera configuration not available. "
            "Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables."
        )

    if not INTRINSICS_AVAILABLE:
        raise ValueError(
            "Camera intrinsics module not available. "
            "Ensure tools.get_camera_intrinsics is accessible."
        )

    if not GPS_CONVERTER_AVAILABLE:
        raise ValueError(
            "GPS converter modules not available. "
            "Ensure poc_homography.gps_distance_calculator and "
            "poc_homography.camera_geometry are installed."
        )

    # Get camera configuration
    cam_config = get_camera_by_name(camera_name)
    if not cam_config:
        available = [c['name'] for c in CAMERAS]
        raise ValueError(
            f"Camera '{camera_name}' not found. "
            f"Available cameras: {', '.join(available)}"
        )

    # Convert GPS coordinates from DMS to decimal degrees
    try:
        camera_lat = dms_to_dd(cam_config['lat'])
        camera_lon = dms_to_dd(cam_config['lon'])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Failed to parse GPS coordinates for camera '{camera_name}': {e}"
        )

    # Get camera height
    height_m = cam_config.get('height_m', 5.0)

    # Get pan offset (angle from north when pan=0)
    # Positive offset means pan=0 points east of north
    pan_offset_deg = cam_config.get('pan_offset_deg', 0.0)

    # Get current PTZ status from live camera
    try:
        ptz_status = get_ptz_status(cam_config['ip'], USERNAME, PASSWORD, timeout=5.0)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to retrieve PTZ status from camera '{camera_name}' "
            f"at {cam_config['ip']}: {e}\n"
            f"Check that the camera is online and credentials are correct."
        )

    # Extract PTZ values and apply pan offset
    # True bearing = reported_pan + pan_offset
    # This converts camera-relative pan to world bearing (degrees from north)
    pan_deg_raw = ptz_status['pan']
    pan_deg = pan_deg_raw + pan_offset_deg
    tilt_deg = ptz_status['tilt']
    zoom = ptz_status['zoom']

    # Use provided image dimensions or defaults
    if image_width is None:
        image_width = 1920  # Default HD resolution
    if image_height is None:
        image_height = 1080  # Default HD resolution

    # Get camera-specific sensor width or use default
    sensor_width_mm = cam_config.get('sensor_width_mm', DEFAULT_SENSOR_WIDTH_MM)

    # Compute intrinsic matrix from current zoom level
    K = CameraGeometry.get_intrinsics(
        zoom_factor=zoom,
        W_px=image_width,
        H_px=image_height,
        sensor_width_mm=sensor_width_mm
    )

    return {
        'camera_lat': camera_lat,
        'camera_lon': camera_lon,
        'height_m': height_m,
        'pan_deg': pan_deg,
        'pan_deg_raw': pan_deg_raw,
        'pan_offset_deg': pan_offset_deg,
        'tilt_deg': tilt_deg,
        'zoom': zoom,
        'K': K,
        'image_width': image_width,
        'image_height': image_height,
        'sensor_width_mm': sensor_width_mm
    }


def project_gps_to_image(gps_points: List[Dict], camera_params: Dict, camera_name: str = None) -> List[Dict]:
    """
    Project GPS points to image pixel coordinates using camera homography.

    This function converts GPS latitude/longitude coordinates to image pixel locations
    by:
    1. Converting GPS to local XY meters relative to camera position
    2. Initializing CameraGeometry with camera parameters
    3. Applying homography H to project world coordinates to image
    4. Applying lens distortion correction if camera has calibrated coefficients
    5. Filtering points that are behind camera or outside image bounds

    Args:
        gps_points: List of GPS point dictionaries from parse_kml_points()
                   [{'name': str, 'latitude': float, 'longitude': float}, ...]
        camera_params: Camera parameter dictionary from get_camera_params_for_projection()
                      with keys: camera_lat, camera_lon, height_m, pan_deg, tilt_deg,
                      K, image_width, image_height
        camera_name: Optional camera name to load distortion coefficients from config

    Returns:
        List of dictionaries with projected points:
        [
            {
                'name': str,
                'latitude': float,
                'longitude': float,
                'pixel_u': float,        # image x coordinate (horizontal)
                'pixel_v': float,        # image y coordinate (vertical)
                'visible': bool,         # True if in camera view
                'reason': str            # 'visible', 'behind_camera', 'outside_bounds'
            },
            ...
        ]

    Example:
        >>> gps_points = parse_kml_points('gcps.kml')
        >>> camera_params = get_camera_params_for_projection("Valte")
        >>> projected = project_gps_to_image(gps_points, camera_params, camera_name="Valte")
        >>> for pt in projected:
        ...     if pt['visible']:
        ...         print(f"{pt['name']}: ({pt['pixel_u']:.1f}, {pt['pixel_v']:.1f})")
    """
    from math import cos, radians

    # Check required modules
    if not COORDINATE_CONVERTER_AVAILABLE:
        raise ValueError(
            "Coordinate converter not available. "
            "Ensure poc_homography.coordinate_converter is installed."
        )

    if not GPS_CONVERTER_AVAILABLE:
        raise ValueError(
            "GPS converter modules not available. "
            "Ensure poc_homography.camera_geometry is installed."
        )

    # Extract camera parameters
    camera_lat = camera_params['camera_lat']
    camera_lon = camera_params['camera_lon']
    height_m = camera_params['height_m']
    pan_deg = camera_params['pan_deg']
    pan_deg_raw = camera_params.get('pan_deg_raw', pan_deg)
    pan_offset_deg = camera_params.get('pan_offset_deg', 0.0)
    tilt_deg = camera_params['tilt_deg']
    K = camera_params['K']
    image_width = camera_params['image_width']
    image_height = camera_params['image_height']

    # Debug: Print camera parameters
    print("\n" + "=" * 60)
    print("KML POINT PROJECTION DEBUG")
    print("=" * 60)
    print(f"Camera Position: lat={camera_lat:.6f}, lon={camera_lon:.6f}")
    print(f"Camera Height: {height_m:.2f} m")
    if pan_offset_deg != 0:
        print(f"Camera PTZ: pan={pan_deg_raw:.1f}° (raw) + {pan_offset_deg:.1f}° (offset) = {pan_deg:.1f}° (true bearing)")
    else:
        print(f"Camera PTZ: pan={pan_deg:.1f}°")
    print(f"Camera Tilt: {tilt_deg:.1f}°")
    print(f"Image Size: {image_width}x{image_height}")
    print(f"Intrinsic Matrix K:\n{K}")
    print("-" * 60)

    # Initialize CameraGeometry with camera parameters
    geo = CameraGeometry(w=image_width, h=image_height)

    # Camera position in world coordinates (X=0, Y=0 at camera location, Z=height)
    w_pos = np.array([0.0, 0.0, height_m])

    # Set up homography
    geo.set_camera_parameters(
        K=K,
        w_pos=w_pos,
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        map_width=640,  # Default map size (not critical for projection)
        map_height=640
    )

    # Load and apply distortion coefficients from camera config
    distortion_applied = False
    if camera_name and CAMERA_CONFIG_AVAILABLE:
        try:
            cam_config = get_camera_by_name(camera_name)
            if cam_config:
                k1 = cam_config.get('k1', 0.0)
                k2 = cam_config.get('k2', 0.0)
                p1 = cam_config.get('p1', 0.0)
                p2 = cam_config.get('p2', 0.0)
                # Only apply if non-zero coefficients exist
                if k1 != 0.0 or k2 != 0.0 or p1 != 0.0 or p2 != 0.0:
                    geo.set_distortion_coefficients(k1=k1, k2=k2, p1=p1, p2=p2)
                    distortion_applied = True
                    print(f"Distortion coefficients applied: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}")
        except Exception as e:
            print(f"Warning: Could not load distortion coefficients: {e}")

    if not distortion_applied:
        print("Note: No distortion correction applied (no coefficients available)")

    # Debug: Print homography matrix
    print(f"Homography Matrix H:\n{geo.H}")
    print("-" * 60)
    print(f"\n{'Point Name':<20} {'GPS (lat,lon)':<25} {'Local XY (m)':<20} {'Pixel (u,v)':<20} {'W':<10} {'Result':<15}")
    print("-" * 110)

    # Check if any points have UTM coordinates
    has_utm_points = any('utm_easting' in p and 'utm_northing' in p for p in gps_points)

    # If we have UTM points, set up UTM converter with camera position as reference
    utm_converter = None
    if has_utm_points:
        try:
            from poc_homography.coordinate_converter import UTMConverter
            # Get UTM CRS from first point that has it, or use default
            utm_crs = next((p.get('utm_crs', 'EPSG:25830') for p in gps_points if 'utm_crs' in p), 'EPSG:25830')
            utm_converter = UTMConverter(utm_crs)
            utm_converter.set_reference(camera_lat, camera_lon)
            print(f"Using UTM coordinates (CRS: {utm_crs}) for accurate conversion")
        except Exception as e:
            print(f"Warning: Could not initialize UTM converter: {e}")
            print("Falling back to equirectangular projection")
            utm_converter = None

    # Project each GPS point
    projected_points = []

    for point in gps_points:
        gps_lat = point['latitude']
        gps_lon = point['longitude']
        name = point['name']

        # Convert to local XY meters relative to camera
        # Prefer UTM coordinates if available (more accurate)
        if utm_converter and 'utm_easting' in point and 'utm_northing' in point:
            # Use UTM coordinates directly - more accurate
            x_m, y_m = utm_converter.utm_to_local_xy(point['utm_easting'], point['utm_northing'])
        elif utm_converter:
            # UTM converter available but point only has GPS - convert via UTM
            x_m, y_m = utm_converter.gps_to_local_xy(gps_lat, gps_lon)
        else:
            # Fall back to equirectangular projection
            x_m, y_m = gps_to_local_xy(camera_lat, camera_lon, gps_lat, gps_lon)

        # Create homogeneous world coordinate [X, Y, 1]
        world_point = np.array([[x_m], [y_m], [1.0]])

        # Apply homography H to project world -> image
        # H maps [X_world, Y_world, 1] -> [u, v, w]
        image_point_homogeneous = geo.H @ world_point

        # Extract homogeneous coordinates
        u = image_point_homogeneous[0, 0]
        v = image_point_homogeneous[1, 0]
        w = image_point_homogeneous[2, 0]

        # Check if point is behind camera (w <= 0)
        if w <= 0:
            # Debug output for behind_camera
            print(f"{name:<20} ({gps_lat:.6f},{gps_lon:.6f}) ({x_m:>8.1f},{y_m:>8.1f}) {'N/A':<20} {w:<10.3f} BEHIND_CAMERA")
            projected_points.append({
                'name': name,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'pixel_u': None,
                'pixel_v': None,
                'visible': False,
                'reason': 'behind_camera'
            })
            continue

        # Normalize to get pixel coordinates (undistorted pinhole model)
        u_px_undist = u / w
        v_px_undist = v / w

        # Apply lens distortion to get actual camera pixel coordinates
        # The distort_point method applies the forward distortion model
        if distortion_applied:
            u_px, v_px = geo.distort_point(u_px_undist, v_px_undist)
        else:
            u_px, v_px = u_px_undist, v_px_undist

        # Check if point is within image bounds
        if 0 <= u_px < image_width and 0 <= v_px < image_height:
            # Debug output for visible
            print(f"{name:<20} ({gps_lat:.6f},{gps_lon:.6f}) ({x_m:>8.1f},{y_m:>8.1f}) ({u_px:>8.1f},{v_px:>8.1f}) {w:<10.3f} VISIBLE")
            projected_points.append({
                'name': name,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'pixel_u': u_px,
                'pixel_v': v_px,
                'visible': True,
                'reason': 'visible'
            })
        else:
            # Debug output for outside_bounds
            print(f"{name:<20} ({gps_lat:.6f},{gps_lon:.6f}) ({x_m:>8.1f},{y_m:>8.1f}) ({u_px:>8.1f},{v_px:>8.1f}) {w:<10.3f} OUTSIDE_BOUNDS")
            projected_points.append({
                'name': name,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'pixel_u': u_px,
                'pixel_v': v_px,
                'visible': False,
                'reason': 'outside_bounds'
            })

    # Debug summary
    visible_count = sum(1 for p in projected_points if p['visible'])
    behind_count = sum(1 for p in projected_points if p.get('reason') == 'behind_camera')
    outside_count = sum(1 for p in projected_points if p.get('reason') == 'outside_bounds')
    print("-" * 110)
    print(f"SUMMARY: {len(projected_points)} total, {visible_count} visible, {behind_count} behind_camera, {outside_count} outside_bounds")
    print("=" * 60 + "\n")

    return projected_points


def verify_camera_height(kml_points: List[Dict], camera_params: Dict, geo: CameraGeometry, camera_name: str = None) -> Dict:
    """
    Verify camera height accuracy using HeightCalibrator and KML reference points.

    This function estimates the optimal camera height by comparing homography-projected
    distances with actual GPS distances. It helps validate if the configured camera
    height is accurate before projecting GPS points to the image.

    Args:
        kml_points: List of GPS point dictionaries from parse_kml_points()
                   [{'name': str, 'latitude': float, 'longitude': float}, ...]
        camera_params: Camera parameter dictionary from get_camera_params_for_projection()
                      with keys: camera_lat, camera_lon, height_m, pan_deg, tilt_deg,
                      K, image_width, image_height
        geo: Initialized CameraGeometry instance with homography matrix
        camera_name: Optional camera name to load distortion coefficients

    Returns:
        Dictionary with height verification results:
        {
            'configured_height': float,        # from camera_params (meters)
            'estimated_height': float,         # from HeightCalibrator (meters)
            'confidence_interval': Tuple[float, float],  # 95% CI (meters)
            'height_valid': bool,              # True if within 10% of configured
            'height_difference_percent': float, # percentage difference
            'inlier_count': int,               # points used for estimation
            'warning': str or None             # warning message if height invalid
        }

    Example:
        >>> gps_points = parse_kml_points('gcps.kml')
        >>> camera_params = get_camera_params_for_projection("Valte")
        >>> geo = CameraGeometry(w=1920, h=1080)
        >>> geo.set_camera_parameters(...)
        >>> verification = verify_camera_height(gps_points, camera_params, geo, camera_name="Valte")
        >>> if not verification['height_valid']:
        ...     print(f"Warning: {verification['warning']}")
    """
    # Check if HeightCalibrator is available
    if not HEIGHT_CALIBRATOR_AVAILABLE:
        return {
            'configured_height': camera_params['height_m'],
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,  # Assume valid if we can't verify
            'height_difference_percent': 0.0,
            'inlier_count': 0,
            'warning': 'Height calibrator module not available - skipping verification'
        }

    # Check minimum points requirement
    if len(kml_points) < 5:
        return {
            'configured_height': camera_params['height_m'],
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,  # Assume valid if we can't verify
            'height_difference_percent': 0.0,
            'inlier_count': len(kml_points),
            'warning': f'Insufficient points for height verification (need 5, have {len(kml_points)})'
        }

    # Extract camera parameters
    camera_lat = camera_params['camera_lat']
    camera_lon = camera_params['camera_lon']
    height_m = camera_params['height_m']
    image_width = camera_params['image_width']
    image_height = camera_params['image_height']

    # Create camera GPS dict for HeightCalibrator
    camera_gps = {
        'lat': camera_lat,
        'lon': camera_lon
    }

    try:
        # Initialize HeightCalibrator
        calibrator = HeightCalibrator(camera_gps, min_points=5)

        # Project GPS points to get pixel coordinates
        projected = project_gps_to_image(kml_points, camera_params, camera_name=camera_name)

        # Add each visible point to the calibrator
        points_added = 0
        for point in projected:
            # Only use visible points that have valid pixel coordinates
            if point['visible'] and point['pixel_u'] is not None and point['pixel_v'] is not None:
                try:
                    calibrator.add_point(
                        pixel_x=point['pixel_u'],
                        pixel_y=point['pixel_v'],
                        gps_lat=point['latitude'],
                        gps_lon=point['longitude'],
                        current_height=height_m,
                        geo=geo
                    )
                    points_added += 1
                except (ValueError, RuntimeError) as e:
                    # Skip points that can't be added (e.g., near horizon)
                    continue

        # Check if we have enough points after filtering
        if points_added < 5:
            return {
                'configured_height': height_m,
                'estimated_height': None,
                'confidence_interval': (None, None),
                'height_valid': True,  # Assume valid if we can't verify
                'height_difference_percent': 0.0,
                'inlier_count': points_added,
                'warning': f'Insufficient valid points for height verification (need 5, have {points_added})'
            }

        # Optimize height using outlier detection
        result = calibrator.optimize_height_with_outliers(method='mad', threshold=2.5)

        # Calculate percentage difference
        estimated_height = result.estimated_height
        height_diff_percent = abs((estimated_height - height_m) / height_m) * 100.0

        # Check if height is valid (within 10% threshold)
        height_valid = height_diff_percent <= 10.0

        # Generate warning message if height is invalid
        warning = None
        if not height_valid:
            warning = (
                f"Camera height may be inaccurate: configured {height_m:.2f}m, "
                f"estimated {estimated_height:.2f}m ({height_diff_percent:.1f}% difference). "
                f"Consider updating camera height in configuration."
            )

        return {
            'configured_height': height_m,
            'estimated_height': estimated_height,
            'confidence_interval': result.confidence_interval,
            'height_valid': height_valid,
            'height_difference_percent': height_diff_percent,
            'inlier_count': result.inlier_count,
            'warning': warning
        }

    except ValueError as e:
        # Handle calibration errors
        return {
            'configured_height': height_m,
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,  # Assume valid if verification failed
            'height_difference_percent': 0.0,
            'inlier_count': 0,
            'warning': f'Height verification failed: {str(e)}'
        }


def verify_camera_height_with_projected(
    projected_points: List[Dict],
    kml_points: List[Dict],
    camera_params: Dict,
    geo: 'CameraGeometry'
) -> Dict:
    """
    Verify camera height using already-projected points.

    This is an optimized version of verify_camera_height that accepts
    pre-computed projected points to avoid duplicate projection calculations.

    Args:
        projected_points: Already projected points from project_gps_to_image()
        kml_points: Original KML points (for reference)
        camera_params: Camera parameter dictionary
        geo: Initialized CameraGeometry instance

    Returns:
        Same structure as verify_camera_height()
    """
    # Check if HeightCalibrator is available
    if not HEIGHT_CALIBRATOR_AVAILABLE:
        return {
            'configured_height': camera_params['height_m'],
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,
            'height_difference_percent': 0.0,
            'inlier_count': 0,
            'warning': 'Height calibrator module not available - skipping verification'
        }

    # Extract camera parameters
    camera_lat = camera_params['camera_lat']
    camera_lon = camera_params['camera_lon']
    height_m = camera_params['height_m']

    # Create camera GPS dict for HeightCalibrator
    camera_gps = {
        'lat': camera_lat,
        'lon': camera_lon
    }

    # Count visible points
    visible_count = sum(1 for p in projected_points if p['visible'])
    if visible_count < 5:
        return {
            'configured_height': height_m,
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,
            'height_difference_percent': 0.0,
            'inlier_count': visible_count,
            'warning': f'Insufficient valid points for height verification (need 5, have {visible_count})'
        }

    try:
        # Initialize HeightCalibrator
        calibrator = HeightCalibrator(camera_gps, min_points=5)

        # Add each visible point to the calibrator
        points_added = 0
        for point in projected_points:
            if point['visible'] and point['pixel_u'] is not None and point['pixel_v'] is not None:
                try:
                    calibrator.add_point(
                        pixel_x=point['pixel_u'],
                        pixel_y=point['pixel_v'],
                        gps_lat=point['latitude'],
                        gps_lon=point['longitude'],
                        current_height=height_m,
                        geo=geo
                    )
                    points_added += 1
                except (ValueError, RuntimeError):
                    continue

        if points_added < 5:
            return {
                'configured_height': height_m,
                'estimated_height': None,
                'confidence_interval': (None, None),
                'height_valid': True,
                'height_difference_percent': 0.0,
                'inlier_count': points_added,
                'warning': f'Insufficient valid points for height verification (need 5, have {points_added})'
            }

        # Optimize height using outlier detection
        result = calibrator.optimize_height_with_outliers(method='mad', threshold=2.5)

        # Calculate percentage difference
        estimated_height = result.estimated_height
        height_diff_percent = abs((estimated_height - height_m) / height_m) * 100.0

        # Check if height is valid (within 10% threshold)
        height_valid = height_diff_percent <= 10.0

        warning = None
        if not height_valid:
            warning = (
                f"Camera height may be inaccurate: configured {height_m:.2f}m, "
                f"estimated {estimated_height:.2f}m ({height_diff_percent:.1f}% difference)."
            )

        return {
            'configured_height': height_m,
            'estimated_height': estimated_height,
            'confidence_interval': result.confidence_interval,
            'height_valid': height_valid,
            'height_difference_percent': height_diff_percent,
            'inlier_count': result.inlier_count,
            'warning': warning
        }

    except ValueError as e:
        return {
            'configured_height': height_m,
            'estimated_height': None,
            'confidence_interval': (None, None),
            'height_valid': True,
            'height_difference_percent': 0.0,
            'inlier_count': 0,
            'warning': f'Height verification failed: {str(e)}'
        }


class GCPCaptureWebSession:
    """Web-based GCP capture session with distribution feedback."""

    # Distribution thresholds (matching feature_match_homography.py)
    MIN_COVERAGE_RATIO = 0.15
    GOOD_COVERAGE_RATIO = 0.35
    MIN_QUADRANT_COVERAGE = 2
    GOOD_QUADRANT_COVERAGE = 3

    def __init__(
        self,
        frame: np.ndarray,
        camera_name: str = None,
        ptz_status: dict = None,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
        base_focal_length_mm: float = DEFAULT_BASE_FOCAL_LENGTH_MM,
        map_first_mode: bool = False,
        kml_points: List[Dict] = None,
        projected_points: List[Dict] = None,
        kml_file_name: str = None,
        height_verification: Dict = None
    ):
        self.frame = frame
        self.camera_name = camera_name or "Unknown"
        self.ptz_status = ptz_status
        self.sensor_width_mm = sensor_width_mm
        self.base_focal_length_mm = base_focal_length_mm

        self.frame_height, self.frame_width = frame.shape[:2]
        self.gcps: List[Dict] = []
        self.capture_timestamp = datetime.now().isoformat()
        # Coordinate system: 'image_v' (V=0 at top) or None (legacy leaflet_y format)
        self.coordinate_system = 'image_v'

        # Homography and reprojection error state
        self.current_homography = None
        self.reference_lat = None
        self.reference_lon = None
        self.reference_utm_easting = None
        self.reference_utm_northing = None
        self.last_reproj_errors = []  # Per-GCP reprojection errors
        self.inlier_mask = None  # RANSAC inlier mask

        # Coordinate converter for accurate GPS/UTM handling
        self.utm_crs = DEFAULT_UTM_CRS
        self.coord_converter = None
        if COORDINATE_CONVERTER_AVAILABLE and GCPCoordinateConverter:
            self.coord_converter = GCPCoordinateConverter(utm_crs=self.utm_crs)

        # Map-first mode support
        self.map_first_mode = map_first_mode
        self.kml_points = kml_points or []
        self.projected_points = projected_points or []
        self.kml_file_name = kml_file_name
        self.height_verification = height_verification

        # Calculate intrinsics if possible
        self.intrinsics = None
        if INTRINSICS_AVAILABLE and ptz_status:
            self.intrinsics = compute_intrinsics(
                zoom=ptz_status.get('zoom', 1.0),
                image_width=self.frame_width,
                image_height=self.frame_height,
                sensor_width_mm=sensor_width_mm,
                base_focal_length_mm=base_focal_length_mm,
            )

    def calculate_distribution(self) -> Dict:
        """Calculate distribution metrics for current GCPs."""
        n_points = len(self.gcps)

        if n_points < 3:
            return {
                'coverage_ratio': 0.0,
                'quadrants_covered': 0,
                'quadrants': [False, False, False, False],
                'spread_x': 0.0,
                'spread_y': 0.0,
                'distribution_score': 0.0,
                'quality': 'Insufficient',
                'warnings': ['Need at least 3 GCPs for distribution analysis']
            }

        # Extract image points
        image_points = np.array([[gcp['image']['u'], gcp['image']['v']] for gcp in self.gcps])

        # Calculate convex hull coverage
        try:
            hull = cv2.convexHull(image_points.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            image_area = self.frame_width * self.frame_height
            coverage_ratio = hull_area / image_area if image_area > 0 else 0.0
        except Exception:
            coverage_ratio = 0.0

        # Calculate quadrant coverage
        center_u = self.frame_width / 2.0
        center_v = self.frame_height / 2.0
        quadrants = [False, False, False, False]  # TL, TR, BL, BR
        for u, v in image_points:
            q = 0
            if u >= center_u:
                q += 1
            if v >= center_v:
                q += 2
            quadrants[q] = True
        quadrants_covered = sum(quadrants)

        # Calculate spread
        spread_x = np.std(image_points[:, 0]) / self.frame_width if self.frame_width > 0 else 0.0
        spread_y = np.std(image_points[:, 1]) / self.frame_height if self.frame_height > 0 else 0.0

        # Generate warnings
        warnings = []
        if coverage_ratio < self.MIN_COVERAGE_RATIO:
            warnings.append(
                f'GCPs are clustered (coverage {coverage_ratio:.1%} < {self.MIN_COVERAGE_RATIO:.0%}). '
                'Add GCPs in different areas.'
            )
        if quadrants_covered < self.MIN_QUADRANT_COVERAGE:
            warnings.append(
                f'GCPs only cover {quadrants_covered}/4 quadrants. '
                'Add GCPs to cover more of the image.'
            )
        if spread_x < 0.15 or spread_y < 0.15:
            warnings.append('GCPs have low spatial variance. Spread points across the image.')

        # Check for GPS precision issues and duplicates
        if GCP_VALIDATION_AVAILABLE and len(self.gcps) >= 2:
            # Analyze GPS precision
            try:
                precision_result = analyze_gps_precision(self.gcps)
                if precision_result.get('warnings'):
                    warnings.extend(precision_result['warnings'])
            except Exception as e:
                pass  # Don't fail if precision analysis fails

            # Check for duplicate GCPs
            try:
                detect_duplicate_gcps(self.gcps)
            except ValueError as e:
                # ValueError is raised when duplicates are detected
                warnings.append(str(e))

        # Calculate overall score
        coverage_score = min(1.0, coverage_ratio / self.GOOD_COVERAGE_RATIO)
        quadrant_score = quadrants_covered / 4.0
        spread_score = min(1.0, (spread_x + spread_y) / 0.5)
        distribution_score = 0.4 * coverage_score + 0.3 * quadrant_score + 0.3 * spread_score

        # Quality label
        if distribution_score > 0.7:
            quality = 'Good'
        elif distribution_score > 0.5:
            quality = 'Fair'
        else:
            quality = 'Poor'

        return {
            'coverage_ratio': coverage_ratio,
            'quadrants_covered': quadrants_covered,
            'quadrants': quadrants,
            'spread_x': spread_x,
            'spread_y': spread_y,
            'distribution_score': distribution_score,
            'quality': quality,
            'warnings': warnings
        }

    def update_homography(self) -> Dict:
        """
        Compute homography from current GCPs and calculate per-GCP reprojection errors.

        Returns:
            Dictionary with homography quality metrics and per-GCP errors.
        """
        if not COORDINATE_CONVERTER_AVAILABLE:
            return {
                'available': False,
                'message': 'Coordinate converter not available'
            }

        if len(self.gcps) < 4:
            self.current_homography = None
            self.last_reproj_errors = []
            self.inlier_mask = None
            return {
                'available': True,
                'num_gcps': len(self.gcps),
                'errors': [],
                'message': 'Need at least 4 GCPs for homography'
            }

        # Set reference point from camera GPS, UTM centroid, or GPS centroid
        if self.reference_lat is None:
            if self.ptz_status:
                # Try to get camera GPS from config
                try:
                    cam_info = get_camera_by_name(self.camera_name)
                    if cam_info and 'gps' in cam_info:
                        self.reference_lat = cam_info['gps'].get('latitude')
                        self.reference_lon = cam_info['gps'].get('longitude')
                except Exception:
                    pass

            # Fall back to GCP centroid (prefer UTM if available)
            if self.reference_lat is None:
                # Check if GCPs have UTM coordinates
                has_utm = any('utm_easting' in g.get('gps', {}) for g in self.gcps)
                if has_utm:
                    # Use UTM centroid
                    eastings = [g['gps']['utm_easting'] for g in self.gcps if 'utm_easting' in g.get('gps', {})]
                    northings = [g['gps']['utm_northing'] for g in self.gcps if 'utm_northing' in g.get('gps', {})]
                    if eastings and northings:
                        self.reference_utm_easting = sum(eastings) / len(eastings)
                        self.reference_utm_northing = sum(northings) / len(northings)
                        # Also compute GPS reference for compatibility
                        if self.coord_converter:
                            self.coord_converter.set_reference_utm(self.reference_utm_easting, self.reference_utm_northing)
                            self.reference_lat = self.coord_converter._ref_lat
                            self.reference_lon = self.coord_converter._ref_lon

                # Fall back to GPS centroid
                if self.reference_lat is None:
                    lats = [g['gps']['latitude'] for g in self.gcps]
                    lons = [g['gps']['longitude'] for g in self.gcps]
                    self.reference_lat = sum(lats) / len(lats)
                    self.reference_lon = sum(lons) / len(lons)

        # Set up the coordinate converter reference point
        if self.coord_converter and self.reference_lat is not None:
            if self.reference_utm_easting is not None:
                self.coord_converter.set_reference_utm(self.reference_utm_easting, self.reference_utm_northing)
            else:
                self.coord_converter.set_reference_gps(self.reference_lat, self.reference_lon)

        # Extract points
        image_points = []
        local_points = []

        for gcp in self.gcps:
            u, v = gcp['image']['u'], gcp['image']['v']
            image_points.append([u, v])

            # Use the unified converter if available (handles both UTM and GPS)
            if self.coord_converter:
                # Build point dict for converter
                point = {'latitude': gcp['gps']['latitude'], 'longitude': gcp['gps']['longitude']}
                if 'utm_easting' in gcp['gps']:
                    point['utm_easting'] = gcp['gps']['utm_easting']
                    point['utm_northing'] = gcp['gps']['utm_northing']
                x, y = self.coord_converter.convert_point(point)
            else:
                # Fall back to equirectangular
                lat, lon = gcp['gps']['latitude'], gcp['gps']['longitude']
                x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)

            local_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float32)
        local_points = np.array(local_points, dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(local_points, image_points, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

        if H is None:
            self.current_homography = None
            self.last_reproj_errors = []
            self.inlier_mask = None
            return {
                'available': True,
                'num_gcps': len(self.gcps),
                'errors': [],
                'message': 'Failed to compute homography'
            }

        self.current_homography = H
        self.inlier_mask = mask.ravel().tolist() if mask is not None else [1] * len(self.gcps)

        # Calculate reprojection errors for ALL points
        projected = cv2.perspectiveTransform(
            local_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        errors = np.linalg.norm(projected - image_points, axis=1)
        self.last_reproj_errors = errors.tolist()

        # Calculate metrics
        num_inliers = int(np.sum(mask)) if mask is not None else len(self.gcps)
        inlier_errors = errors[mask.ravel() == 1] if mask is not None else errors

        # Build per-GCP error info
        gcp_errors = []
        for i, (gcp, error, is_inlier) in enumerate(zip(self.gcps, errors, self.inlier_mask)):
            status = 'good' if error < REPROJ_ERROR_GOOD else 'warning' if error < REPROJ_ERROR_WARNING else 'bad'
            gcp_errors.append({
                'index': i,
                'description': gcp.get('metadata', {}).get('description', f'GCP {i+1}'),
                'error_px': float(error),
                'is_inlier': bool(is_inlier),
                'status': status
            })

        # Sort by error (highest first) for outlier identification
        sorted_by_error = sorted(gcp_errors, key=lambda x: x['error_px'], reverse=True)

        return {
            'available': True,
            'num_gcps': len(self.gcps),
            'num_inliers': num_inliers,
            'inlier_ratio': num_inliers / len(self.gcps),
            'mean_error_px': float(np.mean(errors)),
            'max_error_px': float(np.max(errors)),
            'inlier_mean_error_px': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0,
            'errors': gcp_errors,
            'outliers': [e for e in sorted_by_error if not e['is_inlier']],
            'worst_gcps': sorted_by_error[:3],
            'thresholds': {
                'good': REPROJ_ERROR_GOOD,
                'warning': REPROJ_ERROR_WARNING,
                'bad': REPROJ_ERROR_BAD,
                'ransac': RANSAC_REPROJ_THRESHOLD
            }
        }

    def get_outliers(self) -> List[int]:
        """
        Get indices of GCPs marked as outliers by RANSAC.

        This method returns GCP indices where RANSAC's inlier mask is False,
        ensuring consistency between what the UI displays as outliers and
        what gets removed. The outlier determination is based on RANSAC's
        geometric consensus algorithm, not a fixed error threshold.

        Returns:
            List of GCP indices that are outliers according to RANSAC.
        """
        if self.inlier_mask is None:
            self.update_homography()

        if self.inlier_mask is None:
            return []

        # Verify mask length matches GCPs (in case of stale data)
        if len(self.inlier_mask) != len(self.gcps):
            self.update_homography()
            if self.inlier_mask is None or len(self.inlier_mask) != len(self.gcps):
                return []

        outliers = []
        for i, is_inlier in enumerate(self.inlier_mask):
            if not is_inlier:
                outliers.append(i)

        return outliers

    def remove_outliers(self) -> Dict:
        """
        Remove all GCPs marked as outliers by RANSAC.

        This removes exactly the GCPs shown as outliers in the UI, ensuring
        consistency between the displayed outlier count and the actual removal.
        The outlier determination uses RANSAC's inlier mask from cv2.findHomography().

        Returns:
            Dictionary with removal results.
        """
        outlier_indices = self.get_outliers()

        if not outlier_indices:
            return {
                'removed_count': 0,
                'removed_indices': [],
                'remaining_gcps': len(self.gcps)
            }

        # Remove in reverse order to preserve indices
        removed_descriptions = []
        for i in sorted(outlier_indices, reverse=True):
            if i < len(self.gcps):
                desc = self.gcps[i].get('metadata', {}).get('description', f'GCP {i+1}')
                removed_descriptions.append(desc)
                self.gcps.pop(i)

        # Recalculate homography
        self.update_homography()

        return {
            'removed_count': len(outlier_indices),
            'removed_indices': outlier_indices,
            'removed_descriptions': removed_descriptions,
            'remaining_gcps': len(self.gcps)
        }

    def predict_new_gcp_error(
        self, u: float, v: float, lat: float, lon: float,
        utm_easting: float = None, utm_northing: float = None
    ) -> Dict:
        """
        Predict reprojection error for a potential new GCP before adding it.

        This helps users identify if a GCP would be an outlier before committing.
        Supports both GPS and UTM coordinates for accurate prediction.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            lat: GPS latitude
            lon: GPS longitude
            utm_easting: Optional UTM easting (preferred when available)
            utm_northing: Optional UTM northing (preferred when available)

        Returns:
            Dictionary with prediction results:
                - available: Whether prediction is available
                - predicted_error_px: Predicted reprojection error
                - status: 'good', 'warning', or 'bad'
                - message: Human-readable assessment
        """
        if not COORDINATE_CONVERTER_AVAILABLE:
            return {
                'available': False,
                'message': 'Coordinate converter not available'
            }

        if self.current_homography is None:
            return {
                'available': False,
                'message': 'Need at least 4 GCPs to predict error'
            }

        if self.reference_lat is None or self.reference_lon is None:
            return {
                'available': False,
                'message': 'Reference point not set'
            }

        try:
            # Convert to local coordinates using the unified converter if available
            if self.coord_converter:
                point = {'latitude': lat, 'longitude': lon}
                if utm_easting is not None and utm_northing is not None:
                    point['utm_easting'] = utm_easting
                    point['utm_northing'] = utm_northing
                x, y = self.coord_converter.convert_point(point)
            else:
                # Fall back to equirectangular
                x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)

            # Project through current homography
            local_pt = np.array([[x, y]], dtype=np.float32)
            projected = cv2.perspectiveTransform(
                local_pt.reshape(-1, 1, 2), self.current_homography
            ).reshape(2)

            # Calculate error
            error = float(np.sqrt((projected[0] - u)**2 + (projected[1] - v)**2))

            # Determine status
            if error < REPROJ_ERROR_GOOD:
                status = 'good'
                message = f'Good fit ({error:.1f}px)'
            elif error < REPROJ_ERROR_WARNING:
                status = 'warning'
                message = f'Moderate error ({error:.1f}px) - consider verifying coordinates'
            else:
                status = 'bad'
                message = f'High error ({error:.1f}px) - likely outlier, verify GPS and pixel position'

            return {
                'available': True,
                'predicted_error_px': error,
                'status': status,
                'message': message,
                'thresholds': {
                    'good': REPROJ_ERROR_GOOD,
                    'warning': REPROJ_ERROR_WARNING,
                    'bad': REPROJ_ERROR_BAD
                }
            }

        except Exception as e:
            return {
                'available': False,
                'message': f'Prediction failed: {str(e)}'
            }

    def add_gcp(
        self, u: float, v: float, lat: float, lon: float,
        description: str = "", accuracy: str = "medium",
        utm_easting: float = None, utm_northing: float = None,
        utm_crs: str = None
    ) -> Dict:
        """
        Add a new GCP with optional UTM coordinates.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            lat: GPS latitude
            lon: GPS longitude
            description: Optional description for the GCP
            accuracy: Accuracy level (low, medium, high)
            utm_easting: Optional UTM easting coordinate
            utm_northing: Optional UTM northing coordinate
            utm_crs: Optional UTM coordinate reference system (e.g., EPSG:25830)

        Returns:
            The created GCP dictionary.
        """
        gcp = {
            'gps': {
                'latitude': lat,
                'longitude': lon,
            },
            'image': {
                'u': u,
                'v': v,
            },
            'metadata': {
                'description': description or f"GCP {len(self.gcps) + 1}",
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat(),
            }
        }

        # Add UTM coordinates if provided
        if utm_easting is not None and utm_northing is not None:
            gcp['gps']['utm_easting'] = utm_easting
            gcp['gps']['utm_northing'] = utm_northing
            gcp['gps']['utm_crs'] = utm_crs or self.utm_crs

        self.gcps.append(gcp)
        return gcp

    def remove_gcp(self, index: int) -> bool:
        """Remove a GCP by index."""
        if 0 <= index < len(self.gcps):
            self.gcps.pop(index)
            return True
        return False

    def update_gcp_position(self, index: int, u: float, v: float) -> bool:
        """Update the pixel position of a GCP (for drag-to-reposition)."""
        if 0 <= index < len(self.gcps):
            self.gcps[index]['image']['u'] = u
            self.gcps[index]['image']['v'] = v
            return True
        return False

    def load_from_yaml(self, yaml_content: str) -> dict:
        """
        Load GCPs from YAML content.

        Returns dict with 'gcps_loaded' count, 'warnings', 'loaded_ptz' position,
        and 'coordinate_system' indicating the V coordinate format.
        """
        warnings = []
        loaded_ptz = None
        loaded_camera_name = None
        coordinate_system = None  # 'image_v' or None (legacy leaflet_y format)
        loaded_image_width = None
        loaded_image_height = None

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            return {'gcps_loaded': 0, 'warnings': [f'YAML parse error: {e}'], 'loaded_ptz': None}

        # Navigate to GCPs
        gcps_data = None
        if 'homography' in data:
            if 'feature_match' in data['homography']:
                fm = data['homography']['feature_match']
                gcps_data = fm.get('ground_control_points', [])

                # Also extract camera context if present
                if 'camera_capture_context' in fm:
                    ctx = fm['camera_capture_context']
                    if ctx.get('camera_name'):
                        loaded_camera_name = ctx['camera_name']
                        self.camera_name = loaded_camera_name
                    if 'ptz_position' in ctx:
                        loaded_ptz = ctx['ptz_position']
                        self.ptz_status = loaded_ptz
                    if ctx.get('capture_timestamp'):
                        self.capture_timestamp = ctx['capture_timestamp']
                    # Check coordinate system - old files won't have this field
                    coordinate_system = ctx.get('coordinate_system')
                    # Get source image dimensions for scaling
                    loaded_image_width = ctx.get('image_width')
                    loaded_image_height = ctx.get('image_height')

        if not gcps_data:
            return {'gcps_loaded': 0, 'warnings': ['No GCPs found in YAML'], 'loaded_ptz': None}

        # Set session coordinate system based on loaded data
        self.coordinate_system = coordinate_system  # None for legacy, 'image_v' for new

        # Warn about legacy format
        if coordinate_system is None:
            warnings.append(
                'Legacy format detected (no coordinate_system flag). '
                'V coordinates are treated as Leaflet Y values. '
                'Save will convert to standard image_v format.'
            )

        # Check for resolution mismatch and calculate scale factors
        scale_u = 1.0
        scale_v = 1.0
        if loaded_image_width and loaded_image_height:
            if loaded_image_width != self.frame_width or loaded_image_height != self.frame_height:
                scale_u = self.frame_width / loaded_image_width
                scale_v = self.frame_height / loaded_image_height
                warnings.append(
                    f'Resolution mismatch: GCPs were captured at {loaded_image_width}x{loaded_image_height}, '
                    f'current frame is {self.frame_width}x{self.frame_height}. '
                    f'Scaling coordinates by ({scale_u:.3f}, {scale_v:.3f}).'
                )

        # Load GCPs
        loaded_count = 0
        for gcp in gcps_data:
            try:
                lat = gcp['gps']['latitude']
                lon = gcp['gps']['longitude']
                u = gcp['image']['u'] * scale_u
                v = gcp['image']['v'] * scale_v
                desc = gcp.get('metadata', {}).get('description', f'GCP {len(self.gcps) + 1}')
                accuracy = gcp.get('metadata', {}).get('accuracy', 'medium')

                # Build GPS dict with latitude/longitude
                gps_data = {'latitude': lat, 'longitude': lon}

                # Preserve UTM coordinates if present
                if 'utm_easting' in gcp['gps']:
                    gps_data['utm_easting'] = gcp['gps']['utm_easting']
                    gps_data['utm_northing'] = gcp['gps']['utm_northing']
                    gps_data['utm_crs'] = gcp['gps'].get('utm_crs', self.utm_crs)

                self.gcps.append({
                    'gps': gps_data,
                    'image': {'u': u, 'v': v},
                    'metadata': {
                        'description': desc,
                        'accuracy': accuracy,
                        'timestamp': gcp.get('metadata', {}).get('timestamp', datetime.now().isoformat())
                    }
                })
                loaded_count += 1
            except (KeyError, TypeError) as e:
                warnings.append(f'Skipped invalid GCP: {e}')

        return {
            'gcps_loaded': loaded_count,
            'warnings': warnings,
            'loaded_ptz': loaded_ptz,
            'loaded_camera_name': loaded_camera_name,
            'coordinate_system': coordinate_system
        }

    def move_camera_to_ptz(self, ptz_position: dict, wait_time: float = 3.0) -> dict:
        """
        Move camera to specified PTZ position.

        Args:
            ptz_position: Dict with 'pan', 'tilt', 'zoom' keys
            wait_time: Seconds to wait for camera to reach position

        Returns:
            Dict with 'success' bool and 'message' string
        """
        import time

        if not CAMERA_CONFIG_AVAILABLE:
            return {'success': False, 'message': 'Camera config not available'}

        cam_info = get_camera_by_name(self.camera_name)
        if not cam_info:
            return {'success': False, 'message': f"Camera '{self.camera_name}' not found in config"}

        try:
            from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

            ptz = HikvisionPTZ(cam_info['ip'], USERNAME, PASSWORD)

            # Move to absolute position
            ptz.send_ptz_return({
                'pan': ptz_position.get('pan', 0),
                'tilt': ptz_position.get('tilt', 0),
                'zoom': ptz_position.get('zoom', 1.0)
            })

            # Wait for camera to reach position
            time.sleep(wait_time)

            # Update session PTZ status
            self.ptz_status = ptz_position

            return {
                'success': True,
                'message': f"Camera moved to P={ptz_position.get('pan', 0):.1f} T={ptz_position.get('tilt', 0):.1f} Z={ptz_position.get('zoom', 1):.1f}x"
            }

        except ImportError as e:
            return {'success': False, 'message': f'PTZ control not available: {e}'}
        except Exception as e:
            return {'success': False, 'message': f'Failed to move camera: {e}'}

    def generate_yaml(self) -> str:
        """Generate YAML config content."""
        mode_description = "map-first mode" if self.map_first_mode else "manual capture"
        lines = [
            "# GCP Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Camera: {self.camera_name}",
            f"# GCPs captured: {len(self.gcps)}",
            f"# Mode: {mode_description}",
            "",
            "homography:",
            "  approach: feature_match",
            "",
            "  feature_match:",
            "    detector: sift",
            "    min_matches: 4",
            "    ransac_threshold: 5.0",
            "",
            "    # Camera Capture Context",
            "    camera_capture_context:",
            f"      camera_name: \"{self.camera_name}\"",
            f"      image_width: {self.frame_width}",
            f"      image_height: {self.frame_height}",
        ]

        # Add camera GPS if available
        if CAMERA_CONFIG_AVAILABLE and GPS_CONVERTER_AVAILABLE:
            try:
                cam_info = get_camera_by_name(self.camera_name)
                if cam_info and 'lat' in cam_info and 'lon' in cam_info:
                    camera_lat = dms_to_dd(cam_info['lat'])
                    camera_lon = dms_to_dd(cam_info['lon'])
                    lines.extend([
                        "      camera_gps:",
                        f"        latitude: {camera_lat}",
                        f"        longitude: {camera_lon}",
                    ])
            except Exception:
                pass

        if self.ptz_status:
            lines.extend([
                "      ptz_position:",
                f"        pan: {self.ptz_status.get('pan', 0):.1f}",
                f"        tilt: {self.ptz_status.get('tilt', 0):.1f}",
                f"        zoom: {self.ptz_status.get('zoom', 1.0):.1f}",
            ])

        if self.intrinsics:
            lines.extend([
                "      intrinsics:",
                f"        focal_length_px: {self.intrinsics['focal_length_px']:.2f}",
                "        principal_point:",
                f"          cx: {self.intrinsics['principal_point']['cx']:.1f}",
                f"          cy: {self.intrinsics['principal_point']['cy']:.1f}",
            ])

        notes = ""
        if self.map_first_mode and self.kml_file_name:
            notes = f"Generated from KML using --map-first mode (source: {self.kml_file_name})"

        lines.extend([
            f"      capture_timestamp: \"{self.capture_timestamp}\"",
            "      # Coordinate system: image_v means V=0 at top (standard image coords)",
            "      # Old files without this field used leaflet_y format (V=0 at bottom)",
            "      coordinate_system: image_v",
            f"      notes: \"{notes}\"",
            "",
            "    # Ground Control Points",
            "    ground_control_points:",
        ])

        for i, gcp in enumerate(self.gcps):
            lat = gcp['gps']['latitude']
            lon = gcp['gps']['longitude']
            u = gcp['image']['u']
            v = gcp['image']['v']

            # Convert V to image_v format if loaded from legacy (leaflet_y) format
            # Legacy: v was stored as leaflet_y (0 at bottom)
            # image_v: v should be image V (0 at top)
            # Conversion: image_v = frame_height - leaflet_y
            if self.coordinate_system is None:  # Legacy format
                v = self.frame_height - v

            desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
            accuracy = gcp.get('metadata', {}).get('accuracy', 'medium')
            timestamp = gcp.get('metadata', {}).get('timestamp', '')
            source = gcp.get('metadata', {}).get('source', 'manual')

            gcp_lines = [
                f"      # GCP {i+1}: {desc}",
                "      - gps:",
                f"          latitude: {lat}",
                f"          longitude: {lon}",
            ]

            # Include UTM coordinates if available
            if 'utm_easting' in gcp['gps'] and 'utm_northing' in gcp['gps']:
                utm_e = gcp['gps']['utm_easting']
                utm_n = gcp['gps']['utm_northing']
                utm_crs = gcp['gps'].get('utm_crs', self.utm_crs)
                gcp_lines.extend([
                    f"          utm_easting: {utm_e}",
                    f"          utm_northing: {utm_n}",
                    f"          utm_crs: \"{utm_crs}\"",
                ])

            gcp_lines.extend([
                "        image:",
                f"          u: {u}",
                f"          v: {v}",
                "        metadata:",
                f"          description: \"{desc}\"",
                f"          accuracy: {accuracy}",
                f"          timestamp: \"{timestamp}\"",
                f"          source: {source}",
                "",
            ])
            lines.extend(gcp_lines)

        return "\n".join(lines)

    def generate_gcps_from_map_first(self, projected_points_data: List[Dict]) -> List[Dict]:
        """
        Generate GCP list from map-first projected points.

        Args:
            projected_points_data: List of projected point dictionaries from JavaScript
                Each should have: {pixel_u, pixel_v, visible, kml_index}

        Returns:
            List of GCP dictionaries suitable for self.gcps
        """
        gcps = []

        for point_data in projected_points_data:
            # Skip non-visible (discarded) points
            if not point_data.get('visible', True):
                continue

            # Get KML data for this point
            kml_index = point_data.get('kml_index', -1)
            if kml_index < 0 or kml_index >= len(self.kml_points):
                continue

            kml_point = self.kml_points[kml_index]

            # Create GCP entry
            gcp = {
                'gps': {
                    'latitude': kml_point['latitude'],
                    'longitude': kml_point['longitude'],
                },
                'image': {
                    'u': float(point_data['pixel_u']),
                    'v': float(point_data['pixel_v']),
                },
                'metadata': {
                    'description': kml_point.get('name', f"Point {kml_index}"),
                    'accuracy': 'medium',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'kml'
                }
            }
            gcps.append(gcp)

        return gcps


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_capture_html(session: GCPCaptureWebSession, frame_path: str) -> str:
    """Generate the HTML interface for GCP capture."""

    gcps_json = json.dumps(session.gcps)
    distribution_json = json.dumps(session.calculate_distribution())
    homography_json = json.dumps(session.update_homography())

    # Map-first mode data
    map_first_mode_json = json.dumps(session.map_first_mode)
    kml_points_json = json.dumps(session.kml_points)
    projected_points_json = json.dumps(session.projected_points)
    kml_file_name_json = json.dumps(session.kml_file_name)
    # Convert numpy types to native Python types for JSON serialization
    height_verification_json = json.dumps(_convert_numpy_types(session.height_verification))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCP Capture - {session.camera_name}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        .header {{
            background: #2d2d2d;
            padding: 12px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 18px;
            font-weight: 500;
        }}

        .header-info {{
            font-size: 12px;
            color: #888;
        }}

        .main-content {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .image-panel {{
            flex: 1;
            position: relative;
            background: #111;
        }}

        #imageMap {{
            width: 100%;
            height: 100%;
            background: #111;
        }}

        .side-panel {{
            width: 350px;
            background: #2d2d2d;
            border-left: 1px solid #444;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            overflow-x: hidden;
        }}

        /* Custom scrollbar for side panel */
        .side-panel::-webkit-scrollbar {{
            width: 8px;
        }}

        .side-panel::-webkit-scrollbar-track {{
            background: #1a1a1a;
        }}

        .side-panel::-webkit-scrollbar-thumb {{
            background: #555;
            border-radius: 4px;
        }}

        .side-panel::-webkit-scrollbar-thumb:hover {{
            background: #777;
        }}

        .panel-section {{
            padding: 15px;
            border-bottom: 1px solid #444;
        }}

        .panel-section h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #fff;
        }}

        /* Distribution Panel */
        .distribution-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }}

        .metric-box {{
            background: #3a3a3a;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 600;
            color: #4CAF50;
        }}

        .metric-value.warning {{
            color: #ff9800;
        }}

        .metric-value.error {{
            color: #f44336;
        }}

        .metric-label {{
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }}

        /* Quadrant visualization */
        .quadrant-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            width: 80px;
            height: 60px;
            margin: 0 auto;
        }}

        .quadrant {{
            background: #444;
            border-radius: 3px;
            transition: background 0.3s;
        }}

        .quadrant.covered {{
            background: #4CAF50;
        }}

        /* Warnings */
        .warnings {{
            margin-top: 10px;
        }}

        .warning-item {{
            background: rgba(255, 152, 0, 0.15);
            border-left: 3px solid #ff9800;
            padding: 8px 10px;
            margin-bottom: 6px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}

        /* Map-First Summary Panel */
        #mapFirstSummary {{
            border-bottom: 1px solid #444;
        }}

        .summary-content {{
            margin-bottom: 10px;
        }}

        .summary-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 12px;
        }}

        .summary-item .label {{
            color: #888;
        }}

        .summary-item .value {{
            color: #e0e0e0;
            font-weight: 600;
        }}

        .summary-item .value.success {{
            color: #4CAF50;
        }}

        .summary-item .value.warning {{
            color: #ff9800;
        }}

        .summary-item .value.error {{
            color: #f44336;
        }}

        #heightWarning {{
            background: rgba(244, 67, 54, 0.15);
            border-left: 3px solid #f44336;
            padding: 8px 10px;
            margin-top: 10px;
            font-size: 11px;
            border-radius: 0 4px 4px 0;
        }}

        /* GCP List */
        .gcp-list {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}

        .gcp-item {{
            background: #3a3a3a;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            position: relative;
        }}

        .gcp-item:hover {{
            background: #454545;
        }}

        .gcp-name {{
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .gcp-coords {{
            font-size: 11px;
            color: #888;
            font-family: monospace;
        }}

        .gcp-delete {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 16px;
            padding: 4px;
        }}

        .gcp-delete:hover {{
            color: #f44336;
        }}

        /* Actions */
        .actions {{
            padding: 15px;
            border-top: 1px solid #444;
            display: flex;
            gap: 10px;
        }}

        .btn {{
            flex: 1;
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .btn-primary {{
            background: #4CAF50;
            color: white;
        }}

        .btn-primary:hover {{
            background: #45a049;
        }}

        .btn-primary:disabled {{
            background: #666;
            cursor: not-allowed;
        }}

        .btn-secondary {{
            background: #555;
            color: white;
        }}

        .btn-secondary:hover {{
            background: #666;
        }}

        .btn-batch {{
            background: #666;
            color: white;
        }}

        .btn-batch.active {{
            background: #ff9800;
            color: black;
        }}

        .btn-batch:hover {{
            background: #777;
        }}

        .btn-batch.active:hover {{
            background: #ffb74d;
        }}

        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 2000;
            justify-content: center;
            align-items: center;
        }}

        .modal.active {{
            display: flex;
        }}

        .modal-content {{
            background: #2d2d2d;
            padding: 25px;
            border-radius: 12px;
            width: 400px;
            max-width: 90%;
        }}

        .modal-header {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
        }}

        .form-group {{
            margin-bottom: 15px;
        }}

        .form-group label {{
            display: block;
            font-size: 12px;
            color: #888;
            margin-bottom: 6px;
        }}

        .form-group input, .form-group select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #555;
            border-radius: 6px;
            background: #3a3a3a;
            color: #e0e0e0;
            font-size: 14px;
        }}

        .form-group input:focus {{
            outline: none;
            border-color: #4CAF50;
        }}

        .form-row {{
            display: flex;
            gap: 10px;
        }}

        .form-row .form-group {{
            flex: 1;
        }}

        .modal-actions {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }}

        /* Crosshair */
        .crosshair {{
            position: absolute;
            top: 50%;
            left: 50%;
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }}

        .crosshair-h, .crosshair-v {{
            position: absolute;
            background: rgba(255, 255, 255, 0.3);
        }}

        .crosshair-h {{
            width: 40px;
            height: 1px;
            left: -20px;
            top: 0;
        }}

        .crosshair-v {{
            width: 1px;
            height: 40px;
            left: 0;
            top: -20px;
        }}

        /* GCP markers on map */
        .gcp-marker {{
            background: #4CAF50;
            border: 2px solid white;
            border-radius: 50%;
            width: 12px;
            height: 12px;
            margin-left: -6px;
            margin-top: -6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}

        .gcp-marker.pending {{
            background: #ff9800;
        }}

        /* Instructions */
        .instructions {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13px;
            z-index: 1000;
            text-align: center;
        }}

        .instructions.capture-mode {{
            background: rgba(76, 175, 80, 0.9);
        }}

        /* Zoom controls */
        .zoom-info {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 1000;
        }}

        /* Leaflet overrides for dark theme */
        .leaflet-container {{
            background: #111;
        }}

        .leaflet-control-zoom a {{
            background: #2d2d2d !important;
            color: #e0e0e0 !important;
            border-color: #444 !important;
        }}

        .leaflet-control-zoom a:hover {{
            background: #3a3a3a !important;
        }}

        /* Homography Quality Panel */
        .homography-panel {{
            margin-bottom: 15px;
        }}

        .error-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}

        .error-good {{ background: #4CAF50; }}
        .error-warning {{ background: #ff9800; }}
        .error-bad {{ background: #f44336; }}

        .gcp-error {{
            font-size: 11px;
            color: #888;
            font-family: monospace;
        }}

        .gcp-error.good {{ color: #4CAF50; }}
        .gcp-error.warning {{ color: #ff9800; }}
        .gcp-error.bad {{ color: #f44336; }}

        .outlier-badge {{
            background: #f44336;
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 8px;
        }}

        .btn-danger {{
            background: #c62828;
            color: white;
        }}

        .btn-danger:hover {{
            background: #b71c1c;
        }}

        .btn-danger:disabled {{
            background: #666;
            cursor: not-allowed;
        }}

        .outlier-summary {{
            background: rgba(244, 67, 54, 0.15);
            border-left: 3px solid #f44336;
            padding: 8px 10px;
            margin-top: 10px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}

        .inlier-stats {{
            background: rgba(76, 175, 80, 0.15);
            border-left: 3px solid #4CAF50;
            padding: 8px 10px;
            margin-top: 10px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}

        /* Map-first marker styles */
        .map-first-marker {{
            position: relative;
        }}

        .map-first-marker .marker-pin {{
            width: 20px;
            height: 20px;
            border-radius: 50% 50% 50% 0;
            background: #2196F3;
            border: 3px solid white;
            position: absolute;
            transform: rotate(-45deg);
            left: 50%;
            top: 50%;
            margin: -15px 0 0 -10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            cursor: grab;
        }}

        .map-first-marker .marker-pin.blue {{
            background: #2196F3;
        }}

        .map-first-marker .marker-label {{
            position: absolute;
            left: 20px;
            top: -10px;
            background: rgba(33, 150, 243, 0.9);
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            pointer-events: none;
        }}

        .map-first-marker:hover .marker-pin {{
            background: #1976D2;
            transform: rotate(-45deg) scale(1.1);
        }}

        .map-first-marker:hover .marker-label {{
            background: rgba(25, 118, 210, 0.95);
        }}

        /* Selected marker pulse animation */
        @keyframes pulse {{
            0% {{
                box-shadow: 0 2px 6px rgba(0,0,0,0.4), 0 0 0 0 rgba(76, 175, 80, 0.7);
            }}
            50% {{
                box-shadow: 0 2px 6px rgba(0,0,0,0.4), 0 0 0 8px rgba(76, 175, 80, 0);
            }}
            100% {{
                box-shadow: 0 2px 6px rgba(0,0,0,0.4), 0 0 0 0 rgba(76, 175, 80, 0);
            }}
        }}

        .map-first-marker.selected {{
            z-index: 1000 !important;
        }}

        /* Map-first tooltip style */
        .map-first-tooltip {{
            background: rgba(33, 150, 243, 0.9) !important;
            border: 1px solid white !important;
            color: white !important;
            font-size: 11px !important;
            padding: 2px 6px !important;
        }}

        /* Context menu styles */
        .context-menu-item {{
            padding: 8px 12px;
            cursor: pointer;
            color: #fff;
            font-size: 13px;
        }}

        .context-menu-item:hover {{
            background: #3a3a3a;
        }}

        .context-menu-item.delete {{
            color: #f44336;
        }}

        .context-menu-item.delete:hover {{
            background: rgba(244, 67, 54, 0.2);
        }}

        /* Selected tooltip style */
        .selected-tooltip {{
            background: rgba(76, 175, 80, 0.95) !important;
            border: 2px solid white !important;
            font-weight: bold !important;
            color: white !important;
        }}

        .selected-tooltip::before {{
            border-top-color: rgba(76, 175, 80, 0.95) !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GCP Capture - {session.camera_name}</h1>
            <div class="header-info">
                Frame: {session.frame_width} x {session.frame_height}
                {f"| PTZ: P={session.ptz_status['pan']:.1f} T={session.ptz_status['tilt']:.1f} Z={session.ptz_status['zoom']:.1f}x" if session.ptz_status else ""}
                <button class="btn btn-batch" id="batchModeBtn" onclick="toggleBatchMode()" style="margin-left: 20px; padding: 4px 12px; font-size: 12px;">Batch Mode: OFF</button>
                <button class="btn btn-secondary" onclick="exportImageWithMarkers()" style="margin-left: 10px; padding: 4px 12px; font-size: 12px;" title="Export frame with markers overlay">Export Image</button>
            </div>
        </div>

        <div class="main-content">
            <div class="image-panel">
                <div id="imageMap"></div>
                <div class="crosshair" id="crosshair">
                    <div class="crosshair-h"></div>
                    <div class="crosshair-v"></div>
                </div>
                <div class="instructions" id="instructions">
                    Scroll to zoom, drag to pan. Click to add GCP.
                </div>
                <div class="zoom-info" id="zoomInfo">Zoom: 1x</div>
            </div>

            <div class="side-panel">
                <!-- Map-First Mode Summary Panel -->
                <div id="mapFirstSummary" class="panel-section" style="display: none;">
                    <h3>Map-First Mode Summary</h3>
                    <div class="summary-content">
                        <div class="summary-item">
                            <span class="label">KML File:</span>
                            <span class="value" id="kmlFileName">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Total KML Points:</span>
                            <span class="value" id="totalKmlPoints">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Visible Points:</span>
                            <span class="value success" id="visiblePoints">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Out of View:</span>
                            <span class="value warning" id="outOfViewPoints">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Discarded:</span>
                            <span class="value error" id="discardedPoints">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Adjusted:</span>
                            <span class="value" id="adjustedPoints" style="color: #ff9800;">0</span>
                        </div>
                    </div>
                    <div id="heightWarning" style="display: none;"></div>
                    <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                        <button class="btn btn-primary" onclick="convertAllToGCPs()" style="flex: 1; min-width: 120px;">
                            Convert to GCPs
                        </button>
                        <button class="btn btn-secondary" onclick="showDriftAnalysis()" style="flex: 1; min-width: 120px;">
                            Analyze Drift
                        </button>
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 8px;">
                        <button class="btn btn-secondary" id="moveAllBtn" onclick="toggleMoveAllMode()" style="flex: 1;">
                            Move All (W)
                        </button>
                        <button class="btn btn-secondary" id="rotateAllBtn" onclick="toggleRotateAllMode()" style="flex: 1;">
                            Rotate All (E)
                        </button>
                        <button class="btn btn-secondary" id="scaleAllBtn" onclick="toggleScaleAllMode()" style="flex: 1;">
                            Scale All (R)
                        </button>
                    </div>
                </div>

                <!-- Drift Analysis Panel -->
                <div id="driftAnalysisPanel" class="panel-section" style="display: none;">
                    <h3>Drift Analysis</h3>
                    <div id="driftResults" style="font-size: 12px;">
                        <p style="color: #888;">Drag KML points to match visible features, then click "Analyze Drift" to diagnose projection errors.</p>
                    </div>
                    <div id="driftRecommendations" style="margin-top: 10px; display: none;">
                        <h4 style="font-size: 13px; color: #fff; margin-bottom: 8px;">Recommendations</h4>
                        <div id="driftRecommendationsContent"></div>
                    </div>
                    <div style="margin-top: 12px;">
                        <button class="btn btn-secondary" onclick="applyCalibration()" id="applyCalibrationBtn" style="width: 100%;" disabled>
                            Apply Suggested Calibration
                        </button>
                    </div>
                </div>

                <div class="panel-section">
                    <h3>Distribution Quality</h3>
                    <div class="distribution-grid">
                        <div class="metric-box">
                            <div class="metric-value" id="scoreValue">0.00</div>
                            <div class="metric-label">Score</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="coverageValue">0%</div>
                            <div class="metric-label">Coverage</div>
                        </div>
                    </div>
                    <div class="metric-box" style="margin-bottom: 15px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 8px;">Quadrant Coverage</div>
                        <div class="quadrant-grid">
                            <div class="quadrant" id="q0"></div>
                            <div class="quadrant" id="q1"></div>
                            <div class="quadrant" id="q2"></div>
                            <div class="quadrant" id="q3"></div>
                        </div>
                    </div>
                    <div class="warnings" id="warnings"></div>
                </div>

                <div class="panel-section">
                    <h3>Homography Quality</h3>
                    <div id="homographyPanel">
                        <div style="color: #666; font-size: 12px;">Need at least 4 GCPs</div>
                    </div>
                </div>

                <div class="panel-section">
                    <h3>GCPs (<span id="gcpCount">0</span>)</h3>
                </div>

                <div class="gcp-list" id="gcpList">
                    <div style="color: #666; text-align: center; padding: 20px;">
                        Click on the image to add GCPs
                    </div>
                </div>

                <div class="actions" style="flex-wrap: wrap;">
                    <input type="file" id="yamlFileInput" accept=".yaml,.yml" style="display: none;" onchange="handleYamlUpload(event)">
                    <button class="btn btn-secondary" onclick="document.getElementById('yamlFileInput').click()" style="flex: 0 0 auto;">Load YAML</button>
                    <button class="btn btn-secondary" onclick="clearAllGCPs()">Clear All</button>
                    <button class="btn btn-danger" id="removeOutliersBtn" onclick="removeOutliers()" disabled>Remove Outliers</button>
                    <button class="btn btn-primary" id="saveBtn" onclick="saveConfig()" disabled>Save YAML</button>
                </div>
                <div style="padding: 10px 15px; font-size: 11px; color: #666; border-top: 1px solid #444;">
                    Tip: Drag markers to fine-tune. Red markers are outliers.
                </div>
            </div>
        </div>
    </div>

    <!-- Add GCP Modal -->
    <div class="modal" id="addGcpModal">
        <div class="modal-content">
            <div class="modal-header">Add Ground Control Point</div>
            <div class="form-group">
                <label>Pixel Position</label>
                <input type="text" id="pixelPos" readonly>
            </div>
            <div class="form-group">
                <label>GPS Coordinates (paste from Google Maps)</label>
                <input type="text" id="gpsInput" placeholder="39.640296, -0.230037 or 39°38'25.7&quot;N 0°13'48.7&quot;W">
                <div style="font-size: 11px; color: #666; margin-top: 4px;">
                    Accepts: "39.640296, -0.230037" or "39°38'25.7"N 0°13'48.7"W"
                </div>
            </div>
            <div id="parsedCoords" style="display: none; background: #3a3a3a; padding: 8px 10px; border-radius: 4px; margin-bottom: 10px; font-size: 12px;">
                <span style="color: #888;">Parsed:</span> <span id="parsedLat"></span>, <span id="parsedLon"></span>
            </div>
            <div id="predictedError" style="display: none; padding: 8px 10px; border-radius: 4px; margin-bottom: 15px; font-size: 12px; border-left: 3px solid #888;">
                <span style="color: #888;">Predicted error:</span> <span id="predictedErrorValue"></span>
                <div id="predictedErrorMessage" style="margin-top: 4px; font-size: 11px;"></div>
            </div>
            <div class="form-group">
                <label>Description (optional)</label>
                <input type="text" id="descInput" placeholder="e.g., Corner of zebra crossing">
            </div>
            <div class="form-group">
                <label>Accuracy</label>
                <select id="accuracyInput">
                    <option value="high">High</option>
                    <option value="medium" selected>Medium</option>
                    <option value="low">Low</option>
                </select>
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary" onclick="confirmAddGCP()">Add GCP</button>
            </div>
        </div>
    </div>

    <!-- Batch Mode Modal -->
    <div class="modal" id="batchModal">
        <div class="modal-content" style="max-width: 700px; max-height: 80vh; overflow-y: auto;">
            <div class="modal-header">Batch Mode - Finalize GCPs</div>
            <div style="margin-bottom: 15px; color: #888; font-size: 13px;">
                <span id="batchPointCount">0</span> points clicked. Assign names and load a KML file with GPS coordinates.
            </div>
            <div style="margin-bottom: 15px;">
                <input type="file" id="kmlFileInput" accept=".kml" style="display: none;" onchange="handleKmlUpload(event)">
                <button class="btn btn-secondary" onclick="document.getElementById('kmlFileInput').click()" style="padding: 6px 12px; font-size: 13px;">
                    Load KML File
                </button>
                <span id="kmlStatus" style="margin-left: 10px; font-size: 12px; color: #888;"></span>
            </div>
            <div id="batchPointsList" style="max-height: 400px; overflow-y: auto;">
                <!-- Points will be listed here -->
            </div>
            <div class="modal-actions" style="margin-top: 15px;">
                <button class="btn btn-secondary" onclick="closeBatchModal()">Cancel</button>
                <button class="btn btn-primary" id="batchConfirmBtn" onclick="confirmBatchGCPs()" disabled>Add All GCPs</button>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const imageWidth = {session.frame_width};
        const imageHeight = {session.frame_height};
        const imagePath = "{frame_path}";

        // State
        let gcps = {gcps_json};
        let distribution = {distribution_json};
        let homography = {homography_json};
        let pendingClick = null;
        let gcpMarkers = [];
        // Coordinate system: 'image_v' = standard (V=0 at top), null = legacy leaflet_y format (V=0 at bottom)
        let coordinateSystem = 'image_v';  // Default for new captures

        // Batch mode state
        let batchMode = false;
        let batchPoints = [];  // Array of {{u, v}} pixel positions
        let batchMarkers = []; // Temporary markers shown during batch mode
        let batchGpsCoords = []; // GPS coords loaded from KML

        // Map-first mode state
        let mapFirstMode = {map_first_mode_json};
        let kmlPoints = {kml_points_json};  // Original KML data with names and GPS
        let projectedPoints = {projected_points_json};  // Projected pixel positions
        let mapFirstMarkers = [];  // Markers for projected points
        let kmlFileName = {kml_file_name_json};  // KML file name
        let heightVerification = {height_verification_json};  // Height verification results

        // Store original projected positions for drift analysis
        let originalProjectedPositions = JSON.parse(JSON.stringify({projected_points_json}));

        // Drift analysis state
        let driftAnalysisResult = null;
        let suggestedCalibration = null;

        // Initialize Leaflet map with simple CRS for image
        // Extended bounds allow dragging markers outside the visible image area
        // IMPORTANT: Disable keyboard navigation to prevent arrow keys from panning
        const map = L.map('imageMap', {{
            crs: L.CRS.Simple,
            minZoom: -3,
            maxZoom: 5,
            zoomSnap: 0.25,
            zoomDelta: 0.5,
            // Don't restrict panning - allow dragging markers anywhere
            maxBoundsViscosity: 0,
            // Disable keyboard navigation - we handle arrow keys ourselves
            keyboard: false
        }});

        // Calculate bounds for the image
        const bounds = [[0, 0], [imageHeight, imageWidth]];
        const imageBounds = L.latLngBounds([[0, 0], [imageHeight, imageWidth]]);

        // Extended bounds for marker dragging (2x image size in each direction)
        const extendedBounds = L.latLngBounds(
            [[-imageHeight, -imageWidth],
             [imageHeight * 2, imageWidth * 2]]
        );

        // Add the image as an overlay
        L.imageOverlay(imagePath, bounds).addTo(map);

        // Fit the map to show the full image, but allow panning beyond
        map.fitBounds(bounds);

        // Set extended max bounds to allow dragging markers outside image
        map.setMaxBounds(extendedBounds);

        // Update zoom info display
        function updateZoomInfo() {{
            const zoom = map.getZoom();
            const scale = Math.pow(2, zoom + 1).toFixed(1);
            document.getElementById('zoomInfo').textContent = `Zoom: ${{scale}}x`;
        }}
        map.on('zoom', updateZoomInfo);
        updateZoomInfo();

        // ============================================================
        // GPS Coordinate Parsing
        // ============================================================

        /**
         * Parse GPS coordinates from various formats:
         * - Decimal: "39.640296, -0.230037"
         * - DMS: "39°38'25.7"N 0°13'48.7"W"
         * Returns {{ lat, lon }} or null if parsing fails
         */
        function parseGPSCoordinates(input) {{
            if (!input || typeof input !== 'string') return null;

            // Clean up input
            input = input.trim();

            // Try decimal format first: "39.640296, -0.230037"
            const decimalMatch = input.match(/^(-?\\d+\\.?\\d*)\\s*[,\\s]\\s*(-?\\d+\\.?\\d*)$/);
            if (decimalMatch) {{
                const lat = parseFloat(decimalMatch[1]);
                const lon = parseFloat(decimalMatch[2]);
                if (!isNaN(lat) && !isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            // Try DMS format: "39°38'25.7"N 0°13'48.7"W"
            // Also handles variations like: 39°38'25.7"N, 0°13'48.7"W
            const dmsPattern = /(-?\\d+)[°]\\s*(\\d+)[′']\\s*(\\d+\\.?\\d*)[″"]?\\s*([NSns])?[,\\s]+(-?\\d+)[°]\\s*(\\d+)[′']\\s*(\\d+\\.?\\d*)[″"]?\\s*([EWew])?/;
            const dmsMatch = input.match(dmsPattern);
            if (dmsMatch) {{
                let latDeg = parseFloat(dmsMatch[1]);
                const latMin = parseFloat(dmsMatch[2]);
                const latSec = parseFloat(dmsMatch[3]);
                const latDir = (dmsMatch[4] || 'N').toUpperCase();

                let lonDeg = parseFloat(dmsMatch[5]);
                const lonMin = parseFloat(dmsMatch[6]);
                const lonSec = parseFloat(dmsMatch[7]);
                const lonDir = (dmsMatch[8] || 'E').toUpperCase();

                // Convert to decimal
                let lat = Math.abs(latDeg) + latMin / 60 + latSec / 3600;
                let lon = Math.abs(lonDeg) + lonMin / 60 + lonSec / 3600;

                // Apply direction
                if (latDir === 'S') lat = -lat;
                if (lonDir === 'W') lon = -lon;

                if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            // Try simpler DMS without seconds: "39°38'N 0°13'W"
            const dmsSimplePattern = /(-?\\d+)[°]\\s*(\\d+\\.?\\d*)[′']\\s*([NSns])?[,\\s]+(-?\\d+)[°]\\s*(\\d+\\.?\\d*)[′']\\s*([EWew])?/;
            const dmsSimpleMatch = input.match(dmsSimplePattern);
            if (dmsSimpleMatch) {{
                let latDeg = parseFloat(dmsSimpleMatch[1]);
                const latMin = parseFloat(dmsSimpleMatch[2]);
                const latDir = (dmsSimpleMatch[3] || 'N').toUpperCase();

                let lonDeg = parseFloat(dmsSimpleMatch[4]);
                const lonMin = parseFloat(dmsSimpleMatch[5]);
                const lonDir = (dmsSimpleMatch[6] || 'E').toUpperCase();

                let lat = Math.abs(latDeg) + latMin / 60;
                let lon = Math.abs(lonDeg) + lonMin / 60;

                if (latDir === 'S') lat = -lat;
                if (lonDir === 'W') lon = -lon;

                if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            return null;
        }}

        // Update parsed coordinates display on input
        function updateParsedDisplay() {{
            const input = document.getElementById('gpsInput').value;
            const parsed = parseGPSCoordinates(input);
            const parsedDiv = document.getElementById('parsedCoords');
            const errorDiv = document.getElementById('predictedError');

            if (parsed) {{
                document.getElementById('parsedLat').textContent = parsed.lat.toFixed(6);
                document.getElementById('parsedLon').textContent = parsed.lon.toFixed(6);
                parsedDiv.style.display = 'block';
                parsedDiv.style.borderLeft = '3px solid #4CAF50';

                // Call predict_error API if we have a pending click position and enough GCPs
                if (pendingClick && gcps.length >= 4) {{
                    fetch('/api/predict_error', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            u: pendingClick.u,
                            v: pendingClick.v,
                            lat: parsed.lat,
                            lon: parsed.lon
                        }})
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        if (data.available) {{
                            const errorValue = document.getElementById('predictedErrorValue');
                            const errorMessage = document.getElementById('predictedErrorMessage');

                            errorValue.textContent = data.predicted_error_px.toFixed(1) + 'px';

                            // Color based on status
                            if (data.status === 'good') {{
                                errorDiv.style.borderLeftColor = '#4CAF50';
                                errorDiv.style.background = 'rgba(76, 175, 80, 0.1)';
                                errorValue.style.color = '#4CAF50';
                            }} else if (data.status === 'warning') {{
                                errorDiv.style.borderLeftColor = '#ff9800';
                                errorDiv.style.background = 'rgba(255, 152, 0, 0.1)';
                                errorValue.style.color = '#ff9800';
                            }} else {{
                                errorDiv.style.borderLeftColor = '#f44336';
                                errorDiv.style.background = 'rgba(244, 67, 54, 0.1)';
                                errorValue.style.color = '#f44336';
                            }}

                            errorMessage.textContent = data.message;
                            errorDiv.style.display = 'block';
                        }} else {{
                            errorDiv.style.display = 'none';
                        }}
                    }})
                    .catch(() => {{
                        errorDiv.style.display = 'none';
                    }});
                }} else {{
                    errorDiv.style.display = 'none';
                }}
            }} else if (input.trim()) {{
                document.getElementById('parsedLat').textContent = '?';
                document.getElementById('parsedLon').textContent = '?';
                parsedDiv.style.display = 'block';
                parsedDiv.style.borderLeft = '3px solid #f44336';
                errorDiv.style.display = 'none';
            }} else {{
                parsedDiv.style.display = 'none';
                errorDiv.style.display = 'none';
            }}
        }}

        // Handle click on map to add GCP
        map.on('click', function(e) {{
            // Convert Leaflet coords to image pixel coords
            // Uses coordinate-system-aware conversion
            const imgCoords = leafletToImage(e.latlng.lat, e.latlng.lng);
            const u = imgCoords.u;
            const v = imgCoords.v;

            // Check bounds
            if (u < 0 || u > imageWidth || v < 0 || v > imageHeight) {{
                return;
            }}

            // Batch mode: just add to list without modal
            if (batchMode) {{
                batchPoints.push({{ u: imgCoords.u, v: imgCoords.v }});
                updateBatchMarkers();
                updateBatchUI();
                return;
            }}

            // Normal mode: Store pending click and show modal
            pendingClick = {{ u: u, v: v }};
            document.getElementById('pixelPos').value = `(${{u.toFixed(1)}}, ${{v.toFixed(1)}})`;
            document.getElementById('gpsInput').value = '';
            document.getElementById('descInput').value = '';
            document.getElementById('parsedCoords').style.display = 'none';
            document.getElementById('addGcpModal').classList.add('active');
            document.getElementById('gpsInput').focus();
        }});

        // Modal functions
        function closeModal() {{
            document.getElementById('addGcpModal').classList.remove('active');
            pendingClick = null;
        }}

        function confirmAddGCP() {{
            if (!pendingClick) return;

            const gpsInput = document.getElementById('gpsInput').value;
            const parsed = parseGPSCoordinates(gpsInput);
            const desc = document.getElementById('descInput').value.trim();
            const accuracy = document.getElementById('accuracyInput').value;

            if (!parsed) {{
                alert('Could not parse GPS coordinates.\\n\\nAccepted formats:\\n• Decimal: 39.640296, -0.230037\\n• DMS: 39°38\\'25.7"N 0°13\\'48.7"W');
                return;
            }}

            const {{ lat, lon }} = parsed;

            // Add GCP via API
            fetch('/api/add_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    u: pendingClick.u,
                    v: pendingClick.v,
                    lat: lat,
                    lon: lon,
                    description: desc,
                    accuracy: accuracy
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
                closeModal();
            }});
        }}

        // Delete GCP
        function deleteGCP(index) {{
            fetch('/api/delete_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ index: index }})
            }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
            }});
        }}

        // Remove all outliers
        function removeOutliers() {{
            if (!homography || !homography.outliers || homography.outliers.length === 0) return;

            const outlierCount = homography.outliers.length;
            const outlierNames = homography.outliers.map(o => o.description).join(', ');

            if (!confirm(`Remove ${{outlierCount}} outlier(s)?\\n\\n${{outlierNames}}`)) return;

            fetch('/api/remove_outliers', {{ method: 'POST' }})
            .then(r => r.json())
            .then(data => {{
                if (data.error) {{
                    alert('Error removing outliers: ' + data.error);
                    return;
                }}
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();

                if (data.removed_count > 0) {{
                    alert(`Removed ${{data.removed_count}} outlier(s):\\n${{data.removed_descriptions.join(', ')}}`);
                }}
            }});
        }}

        // Clear all GCPs
        function clearAllGCPs() {{
            if (gcps.length === 0) return;
            if (!confirm(`Clear all ${{gcps.length}} GCPs?`)) return;

            fetch('/api/clear_gcps', {{ method: 'POST' }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
            }});
        }}

        // Save config
        function saveConfig() {{
            if (mapFirstMode) {{
                // In map-first mode, export projected points with their current positions
                const exportData = projectedPoints.map((point, index) => ({{
                    pixel_u: point.pixel_u,
                    pixel_v: point.pixel_v,
                    visible: point.visible !== false,
                    kml_index: index
                }}));

                fetch('/api/export_map_first', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{projected_points: exportData}})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.success) {{
                        alert(`Map-first GCPs saved successfully!\n\nGCPs exported: ${{data.gcps_saved}}`);

                        // Create download link for YAML
                        const blob = new Blob([data.yaml_content], {{type: 'application/x-yaml'}});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'gcps_map_first.yaml';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    }} else {{
                        alert('Failed to export map-first GCPs');
                    }}
                }})
                .catch(err => {{
                    console.error('Export error:', err);
                    alert('Error exporting map-first GCPs');
                }});
            }} else {{
                // Standard manual capture mode
                window.location.href = '/api/save';
            }}
        }}

        // Export image with markers overlay (and plain frame)
        function exportImageWithMarkers() {{
            const timestamp = Date.now();

            // Load the original frame image
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function() {{
                // First, export the plain frame without markers
                const canvasPlain = document.createElement('canvas');
                canvasPlain.width = imageWidth;
                canvasPlain.height = imageHeight;
                const ctxPlain = canvasPlain.getContext('2d');
                ctxPlain.drawImage(img, 0, 0, imageWidth, imageHeight);

                canvasPlain.toBlob(function(blob) {{
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `frame_${{timestamp}}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}, 'image/png');

                // Then, export markers only (transparent background)
                const canvas = document.createElement('canvas');
                canvas.width = imageWidth;
                canvas.height = imageHeight;
                const ctx = canvas.getContext('2d');
                // Leave background transparent (don't draw the image)

                // Draw markers
                if (mapFirstMode) {{
                    // Draw map-first projected points
                    projectedPoints.forEach((point, i) => {{
                        if (point.visible === false) return;

                        const kmlPoint = kmlPoints[i] || {{}};
                        const label = kmlPoint.name || `P${{i + 1}}`;

                        // Draw marker circle
                        ctx.beginPath();
                        ctx.arc(point.pixel_u, point.pixel_v, 8, 0, 2 * Math.PI);
                        ctx.fillStyle = 'rgba(33, 150, 243, 0.8)';
                        ctx.fill();
                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        // Draw label
                        ctx.font = 'bold 12px Arial';
                        ctx.fillStyle = 'white';
                        ctx.strokeStyle = 'black';
                        ctx.lineWidth = 3;
                        ctx.strokeText(label, point.pixel_u + 12, point.pixel_v + 4);
                        ctx.fillText(label, point.pixel_u + 12, point.pixel_v + 4);
                    }});
                }} else {{
                    // Draw manual GCP markers
                    gcps.forEach((gcp, i) => {{
                        const u = gcp.image.u;
                        const v = gcp.image.v;
                        const label = gcp.metadata?.description || `GCP ${{i + 1}}`;

                        // Draw marker circle
                        ctx.beginPath();
                        ctx.arc(u, v, 8, 0, 2 * Math.PI);
                        ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
                        ctx.fill();
                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        // Draw label
                        ctx.font = 'bold 12px Arial';
                        ctx.fillStyle = 'white';
                        ctx.strokeStyle = 'black';
                        ctx.lineWidth = 3;
                        ctx.strokeText(label, u + 12, v + 4);
                        ctx.fillText(label, u + 12, v + 4);
                    }});
                }}

                // Download image with markers
                canvas.toBlob(function(blob) {{
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `frame_markers_${{timestamp}}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}, 'image/png');
            }};
            img.onerror = function() {{
                alert('Failed to load image for export');
            }};
            img.src = imagePath;
        }}

        // Update UI with current state
        function updateUI() {{
            // Update distribution metrics
            const score = distribution.distribution_score;
            const scoreEl = document.getElementById('scoreValue');
            scoreEl.textContent = score.toFixed(2);
            scoreEl.className = 'metric-value ' + (score > 0.7 ? '' : score > 0.5 ? 'warning' : 'error');

            const coverage = distribution.coverage_ratio * 100;
            const coverageEl = document.getElementById('coverageValue');
            coverageEl.textContent = coverage.toFixed(0) + '%';
            coverageEl.className = 'metric-value ' + (coverage >= 35 ? '' : coverage >= 15 ? 'warning' : 'error');

            // Update quadrant visualization
            for (let i = 0; i < 4; i++) {{
                const el = document.getElementById('q' + i);
                if (distribution.quadrants && distribution.quadrants[i]) {{
                    el.classList.add('covered');
                }} else {{
                    el.classList.remove('covered');
                }}
            }}

            // Update warnings
            const warningsEl = document.getElementById('warnings');
            if (distribution.warnings && distribution.warnings.length > 0) {{
                warningsEl.innerHTML = distribution.warnings
                    .map(w => `<div class="warning-item">${{w}}</div>`)
                    .join('');
            }} else {{
                warningsEl.innerHTML = '';
            }}

            // Update homography panel
            updateHomographyPanel();

            // Update GCP count
            document.getElementById('gcpCount').textContent = gcps.length;

            // Update GCP list with reprojection errors
            const listEl = document.getElementById('gcpList');
            if (gcps.length === 0) {{
                listEl.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">Click on the image to add GCPs</div>';
            }} else {{
                listEl.innerHTML = gcps.map((gcp, i) => {{
                    // Get error info from homography if available
                    let errorHtml = '';
                    let outlierBadge = '';
                    if (homography && homography.errors && homography.errors[i]) {{
                        const errInfo = homography.errors[i];
                        const statusClass = errInfo.status || 'good';
                        errorHtml = `<div class="gcp-error ${{statusClass}}">
                            <span class="error-indicator error-${{statusClass}}"></span>
                            Error: ${{errInfo.error_px.toFixed(1)}}px
                            ${{!errInfo.is_inlier ? ' (outlier)' : ''}}
                        </div>`;
                        if (!errInfo.is_inlier) {{
                            outlierBadge = '<span class="outlier-badge">OUTLIER</span>';
                        }}
                    }}

                    return `
                        <div class="gcp-item" onmouseover="highlightGCP(${{i}})" onmouseout="unhighlightGCP(${{i}})">
                            <button class="gcp-delete" onclick="deleteGCP(${{i}})">&times;</button>
                            <div class="gcp-name">${{gcp.metadata?.description || 'GCP ' + (i+1)}}${{outlierBadge}}</div>
                            <div class="gcp-coords">
                                Pixel: (${{gcp.image.u.toFixed(1)}}, ${{gcp.image.v.toFixed(1)}})<br>
                                GPS: (${{gcp.gps.latitude.toFixed(6)}}, ${{gcp.gps.longitude.toFixed(6)}})
                            </div>
                            ${{errorHtml}}
                        </div>
                    `;
                }}).join('');
            }}

            // Update markers on map
            updateMarkers();

            // Update map-first markers if in map-first mode
            if (mapFirstMode) {{
                renderMapFirstPoints();
                updateMapFirstSummary();

                // Update instructions for map-first mode
                const instructions = document.getElementById('instructions');
                if (rotateAllMode) {{
                    instructions.textContent = 'ROTATE MODE: ← → rotate all points around centroid (Shift = faster) | Enter/Esc = done';
                    instructions.style.background = 'rgba(156, 39, 176, 0.9)';
                }} else if (moveAllMode) {{
                    instructions.textContent = 'MOVE ALL MODE: Arrow keys move all points (Shift = faster) | Enter/Esc = done';
                    instructions.style.background = 'rgba(255, 152, 0, 0.9)';
                }} else if (selectedMapFirstIndices.size > 0) {{
                    const count = selectedMapFirstIndices.size;
                    instructions.textContent = count > 1
                        ? `Arrow keys to move ${{count}} selected points (Shift = faster) | Ctrl+click = add/remove | Esc = deselect`
                        : 'Arrow keys to move selected point (Shift = faster) | Ctrl+click = add to selection | Esc = deselect';
                    instructions.style.background = 'rgba(76, 175, 80, 0.9)';
                }} else {{
                    instructions.textContent = 'Click a KML marker to select it, then use Arrow keys to move';
                    instructions.style.background = 'rgba(33, 150, 243, 0.9)';
                }}
            }}

            // Enable/disable save button
            document.getElementById('saveBtn').disabled = gcps.length < 4;

            // Enable/disable remove outliers button
            const outlierCount = homography && homography.outliers ? homography.outliers.length : 0;
            document.getElementById('removeOutliersBtn').disabled = outlierCount === 0;
            document.getElementById('removeOutliersBtn').textContent = outlierCount > 0
                ? `Remove Outliers (${{outlierCount}})`
                : 'Remove Outliers';
        }}

        // Update the homography quality panel
        function updateHomographyPanel() {{
            const panel = document.getElementById('homographyPanel');

            if (!homography || !homography.available) {{
                panel.innerHTML = '<div style="color: #666; font-size: 12px;">Homography calculation not available</div>';
                return;
            }}

            if (gcps.length < 4) {{
                panel.innerHTML = '<div style="color: #666; font-size: 12px;">Need at least 4 GCPs for homography</div>';
                return;
            }}

            const numInliers = homography.num_inliers || 0;
            const numGcps = homography.num_gcps || 0;
            const inlierRatio = homography.inlier_ratio || 0;
            const meanError = homography.mean_error_px || 0;
            const maxError = homography.max_error_px || 0;
            const outliers = homography.outliers || [];

            // Determine quality status
            let qualityClass = 'good';
            let qualityText = 'Good';
            if (inlierRatio < 0.5 || meanError > 10) {{
                qualityClass = 'bad';
                qualityText = 'Poor';
            }} else if (inlierRatio < 0.7 || meanError > 5) {{
                qualityClass = 'warning';
                qualityText = 'Fair';
            }}

            let html = `
                <div class="distribution-grid" style="margin-bottom: 10px;">
                    <div class="metric-box">
                        <div class="metric-value ${{qualityClass === 'bad' ? 'error' : qualityClass === 'warning' ? 'warning' : ''}}">${{(inlierRatio * 100).toFixed(0)}}%</div>
                        <div class="metric-label">Inlier Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value ${{meanError > 10 ? 'error' : meanError > 5 ? 'warning' : ''}}">${{meanError.toFixed(1)}}px</div>
                        <div class="metric-label">Mean Error</div>
                    </div>
                </div>
            `;

            // Show inlier stats
            html += `
                <div class="inlier-stats">
                    <strong>${{numInliers}}/${{numGcps}}</strong> inliers | Max error: <strong>${{maxError.toFixed(1)}}px</strong>
                </div>
            `;

            // Show outlier summary if any
            if (outliers.length > 0) {{
                html += `
                    <div class="outlier-summary">
                        <strong>${{outliers.length}} outlier${{outliers.length > 1 ? 's' : ''}} detected:</strong><br>
                        ${{outliers.slice(0, 3).map(o => `${{o.description}} (${{o.error_px.toFixed(1)}}px)`).join(', ')}}
                        ${{outliers.length > 3 ? '...' : ''}}
                    </div>
                `;
            }}

            panel.innerHTML = html;
        }}

        // Create a custom draggable icon
        function createGcpIcon(color = '#4CAF50', size = 12) {{
            return L.divIcon({{
                className: 'gcp-drag-marker',
                html: `<div style="
                    width: ${{size}}px;
                    height: ${{size}}px;
                    background: ${{color}};
                    border: 2px solid white;
                    border-radius: 50%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    cursor: grab;
                "></div>`,
                iconSize: [size, size],
                iconAnchor: [size/2, size/2]
            }});
        }}

        // Convert image pixel coords to Leaflet coords
        // For image_v format: Image V (0 at top) -> Leaflet Y (0 at bottom): leaflet_y = imageHeight - v
        // For legacy format (null): V was stored as leaflet_y directly, no conversion needed
        function imageToLeaflet(u, v) {{
            if (coordinateSystem === 'image_v') {{
                return [imageHeight - v, u];  // [lat, lng] = [leaflet_y, x]
            }} else {{
                // Legacy format: v is already leaflet_y
                return [v, u];
            }}
        }}

        // Convert Leaflet coords to image pixel coords
        // Uses current coordinate system to stay consistent with loaded data
        function leafletToImage(lat, lng) {{
            if (coordinateSystem === 'image_v') {{
                return {{ u: lng, v: imageHeight - lat }};
            }} else {{
                // Legacy format: store leaflet_y directly as v
                return {{ u: lng, v: lat }};
            }}
        }}

        // Update markers on map
        function updateMarkers() {{
            // Remove existing markers
            gcpMarkers.forEach(m => map.removeLayer(m));
            gcpMarkers = [];

            // Add new markers (draggable, color-coded by error status)
            gcps.forEach((gcp, i) => {{
                // Convert image coords to Leaflet coords (invert Y)
                const leafletCoords = imageToLeaflet(gcp.image.u, gcp.image.v);

                // Determine marker color based on reprojection error
                let markerColor = '#4CAF50';  // Default: green (good)
                let markerSize = 14;
                let tooltipExtra = '';

                if (homography && homography.errors && homography.errors[i]) {{
                    const errInfo = homography.errors[i];
                    if (!errInfo.is_inlier) {{
                        markerColor = '#f44336';  // Red for outliers
                        markerSize = 16;
                        tooltipExtra = ` - OUTLIER (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else if (errInfo.status === 'bad') {{
                        markerColor = '#f44336';  // Red for high error
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else if (errInfo.status === 'warning') {{
                        markerColor = '#ff9800';  // Orange for medium error
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else {{
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }}
                }}

                const marker = L.marker(leafletCoords, {{
                    icon: createGcpIcon(markerColor, markerSize),
                    draggable: true
                }}).addTo(map);

                // Store index and original color for reference
                marker.gcpIndex = i;
                marker.originalColor = markerColor;
                marker.originalSize = markerSize;

                // CRITICAL: Stop click propagation to prevent map click from firing
                // This allows dragging to work without opening the modal
                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});
                marker.on('mousedown', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                // Add label with error info
                marker.bindTooltip(`${{gcp.metadata?.description || 'GCP ' + (i+1)}}${{tooltipExtra}}`, {{
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10]
                }});

                // Handle drag end - update position
                marker.on('dragend', function(e) {{
                    const newPos = e.target.getLatLng();
                    // Convert Leaflet coords back to image coords (invert Y)
                    const imgCoords = leafletToImage(newPos.lat, newPos.lng);
                    const newU = imgCoords.u;
                    const newV = imgCoords.v;

                    // Check bounds
                    if (newU < 0 || newU > imageWidth || newV < 0 || newV > imageHeight) {{
                        // Revert to original position (convert back to Leaflet coords)
                        e.target.setLatLng(imageToLeaflet(gcp.image.u, gcp.image.v));
                        return;
                    }}

                    // Update on server
                    fetch('/api/update_gcp_position', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            index: i,
                            u: newU,
                            v: newV
                        }})
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        gcps = data.gcps;
                        distribution = data.distribution;
                        updateUI();
                    }});
                }});

                // Visual feedback during drag
                marker.on('dragstart', function() {{
                    marker.setIcon(createGcpIcon('#00ffff', 18));
                }});

                marker.on('drag', function() {{
                    // Update tooltip during drag with image coordinates
                    const pos = marker.getLatLng();
                    const imgCoords = leafletToImage(pos.lat, pos.lng);
                    marker.setTooltipContent(`${{gcp.metadata?.description || 'GCP ' + (i+1)}}<br>(${{imgCoords.u.toFixed(1)}}, ${{imgCoords.v.toFixed(1)}})`);
                }});

                gcpMarkers.push(marker);
            }});
        }}

        // Highlight GCP on hover
        function highlightGCP(index) {{
            if (gcpMarkers[index]) {{
                gcpMarkers[index].setIcon(createGcpIcon('#00ffff', 18));
            }}
        }}

        function unhighlightGCP(index) {{
            if (gcpMarkers[index]) {{
                // Restore to original color based on error status
                const marker = gcpMarkers[index];
                const color = marker.originalColor || '#4CAF50';
                const size = marker.originalSize || 14;
                marker.setIcon(createGcpIcon(color, size));
            }}
        }}

        // Selected map-first points for keyboard movement (Set of indices for multi-select)
        let selectedMapFirstIndices = new Set();

        // Move All mode state
        let moveAllMode = false;

        // Rotate All mode state
        let rotateAllMode = false;

        // Scale All mode state
        let scaleAllMode = false;

        // Render map-first projected points
        function renderMapFirstPoints() {{
            // Remove existing map-first markers
            mapFirstMarkers.forEach(m => map.removeLayer(m));
            mapFirstMarkers = [];

            if (!mapFirstMode || !projectedPoints || projectedPoints.length === 0) {{
                return;
            }}

            // Check if any points are selected
            const hasSelection = selectedMapFirstIndices.size > 0;

            // Add markers for each projected point
            projectedPoints.forEach((point, i) => {{
                if (!point.visible) return;  // Skip discarded points

                // Get corresponding KML point for label
                const kmlPoint = kmlPoints[i] || {{}};
                const label = kmlPoint.name || `Point ${{i + 1}}`;
                const gpsLat = kmlPoint.latitude || 0;
                const gpsLon = kmlPoint.longitude || 0;

                // Convert image coords to Leaflet coords
                const leafletCoords = imageToLeaflet(point.pixel_u, point.pixel_v);

                // Check if this point is selected (multi-select)
                const isSelected = selectedMapFirstIndices.has(i);

                // Check if we should dim this marker (when other points are selected but not this one)
                const shouldDim = (hasSelection && !isSelected);

                // Create marker with custom icon (NOT draggable - use arrow keys instead)
                const marker = L.marker(leafletCoords, {{
                    icon: createMapFirstIcon(label, isSelected),
                    draggable: false,  // Disabled - use arrow keys to move
                    opacity: shouldDim ? 0.15 : 1.0  // Dim non-selected markers when some are selected
                }}).addTo(map);

                // Store reference data
                marker.projectedIndex = i;
                marker.kmlName = label;

                // Calculate drift info
                const orig = originalProjectedPositions[i];
                let driftInfo = '';
                if (orig && orig.pixel_u !== null && orig.pixel_v !== null) {{
                    const dx = point.pixel_u - orig.pixel_u;
                    const dy = point.pixel_v - orig.pixel_v;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > 2) {{
                        driftInfo = `<br><span style="color: #ff9800;">Drift: (${{dx.toFixed(1)}}, ${{dy.toFixed(1)}}) = ${{dist.toFixed(1)}}px</span>`;
                    }}
                }}

                // Add popup with GPS info and drift (no discard button - use context menu)
                const selectionCount = selectedMapFirstIndices.size;
                const popupContent = `
                    <div style="min-width: 220px;">
                        <strong>${{label}}</strong>${{isSelected ? ' <span style="color: #4CAF50;">(SELECTED)</span>' : ''}}<br>
                        GPS: ${{gpsLat.toFixed(6)}}, ${{gpsLon.toFixed(6)}}<br>
                        Image: (${{point.pixel_u.toFixed(1)}}, ${{point.pixel_v.toFixed(1)}})<br>
                        ${{driftInfo}}
                        <div style="margin-top: 8px; font-size: 11px; color: #888;">
                            Click to select/deselect (Ctrl+Click for multi-select)<br>
                            Right-click for options
                        </div>
                    </div>
                `;
                marker.bindPopup(popupContent);

                // Add tooltip with label - permanent so we can see point names
                // Hide tooltips for dimmed markers to reduce clutter
                if (!shouldDim) {{
                    let tooltipText = label;
                    if (isSelected && selectionCount > 1) {{
                        tooltipText = `${{label}} (1 of ${{selectionCount}} selected)`;
                    }} else if (isSelected) {{
                        tooltipText = `${{label}} (SELECTED)`;
                    }}
                    marker.bindTooltip(tooltipText, {{
                        permanent: true,
                        direction: 'right',
                        offset: [10, 0],
                        className: isSelected ? 'selected-tooltip' : 'map-first-tooltip'
                    }});
                }}

                // Handle left click to select/deselect point (Ctrl for multi-select)
                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                    toggleMapFirstPointSelection(i, e.originalEvent.ctrlKey || e.originalEvent.metaKey);
                }});

                // Handle right-click for context menu
                marker.on('contextmenu', function(e) {{
                    L.DomEvent.stopPropagation(e);
                    L.DomEvent.preventDefault(e);
                    showPointContextMenu(e.originalEvent, i);
                }});

                marker.on('mousedown', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                mapFirstMarkers.push(marker);
            }});

            // Update selection indicator
            updateSelectionIndicator();
        }}

        // Show context menu for a point
        function showPointContextMenu(event, pointIndex) {{
            // Remove any existing context menu
            const existingMenu = document.getElementById('pointContextMenu');
            if (existingMenu) existingMenu.remove();

            const isSelected = selectedMapFirstIndices.has(pointIndex);
            const selectionCount = selectedMapFirstIndices.size;
            const kmlPoint = kmlPoints[pointIndex] || {{}};
            const label = kmlPoint.name || `Point ${{pointIndex + 1}}`;

            // Create context menu
            const menu = document.createElement('div');
            menu.id = 'pointContextMenu';
            menu.style.cssText = `
                position: fixed;
                left: ${{event.clientX}}px;
                top: ${{event.clientY}}px;
                background: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 0;
                min-width: 180px;
                z-index: 10001;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;

            let menuItems = '';

            // If this point is not selected, offer to select it
            if (!isSelected) {{
                menuItems += `
                    <div class="context-menu-item" onclick="toggleMapFirstPointSelection(${{pointIndex}}, false); hidePointContextMenu();">
                        Select "${{label}}"
                    </div>
                    <div class="context-menu-item" onclick="toggleMapFirstPointSelection(${{pointIndex}}, true); hidePointContextMenu();">
                        Add to selection
                    </div>
                `;
            }} else {{
                menuItems += `
                    <div class="context-menu-item" onclick="toggleMapFirstPointSelection(${{pointIndex}}, true); hidePointContextMenu();">
                        Deselect "${{label}}"
                    </div>
                `;
            }}

            // Separator
            menuItems += `<div style="border-top: 1px solid #555; margin: 4px 0;"></div>`;

            // Delete options
            if (isSelected && selectionCount > 1) {{
                menuItems += `
                    <div class="context-menu-item delete" onclick="deleteSelectedPoints(); hidePointContextMenu();">
                        Delete ${{selectionCount}} selected points
                    </div>
                `;
            }} else {{
                menuItems += `
                    <div class="context-menu-item delete" onclick="discardMapFirstPoint(${{pointIndex}}); hidePointContextMenu();">
                        Delete "${{label}}"
                    </div>
                `;
            }}

            // Select all option
            menuItems += `
                <div style="border-top: 1px solid #555; margin: 4px 0;"></div>
                <div class="context-menu-item" onclick="selectAllMapFirstPoints(); hidePointContextMenu();">
                    Select all points
                </div>
                <div class="context-menu-item" onclick="clearMapFirstSelection(); hidePointContextMenu();">
                    Clear selection
                </div>
            `;

            menu.innerHTML = menuItems;
            document.body.appendChild(menu);

            // Close menu when clicking elsewhere
            setTimeout(() => {{
                document.addEventListener('click', hidePointContextMenu, {{ once: true }});
            }}, 10);
        }}

        // Hide context menu
        function hidePointContextMenu() {{
            const menu = document.getElementById('pointContextMenu');
            if (menu) menu.remove();
        }}

        // Create map-first icon - uses exact same structure as GCP icons to ensure correct positioning
        function createMapFirstIcon(label, isSelected = false) {{
            const color = isSelected ? '#4CAF50' : '#2196F3';
            const size = isSelected ? 16 : 14;

            // Use exact same icon structure as createGcpIcon
            return L.divIcon({{
                className: 'gcp-drag-marker',
                html: `<div style="
                    width: ${{size}}px;
                    height: ${{size}}px;
                    background: ${{color}};
                    border: 2px solid white;
                    border-radius: 50%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    cursor: pointer;
                "></div>`,
                iconSize: [size, size],
                iconAnchor: [size/2, size/2]
            }});
        }}

        // Toggle selection of a map-first point (multi-select support)
        function toggleMapFirstPointSelection(index, isMultiSelect = false) {{
            if (isMultiSelect) {{
                // Multi-select mode: add/remove from selection
                if (selectedMapFirstIndices.has(index)) {{
                    selectedMapFirstIndices.delete(index);
                }} else {{
                    selectedMapFirstIndices.add(index);
                }}
            }} else {{
                // Single-select mode: clear others and toggle this one
                if (selectedMapFirstIndices.has(index) && selectedMapFirstIndices.size === 1) {{
                    selectedMapFirstIndices.clear();
                }} else {{
                    selectedMapFirstIndices.clear();
                    selectedMapFirstIndices.add(index);
                }}
            }}
            renderMapFirstPoints();
            updateMapFirstSummary();

            // Focus on the map to receive keyboard events
            document.getElementById('imageMap').focus();
        }}

        // Select all visible map-first points
        function selectAllMapFirstPoints() {{
            selectedMapFirstIndices.clear();
            projectedPoints.forEach((point, i) => {{
                if (point.visible !== false && point.reason === 'visible') {{
                    selectedMapFirstIndices.add(i);
                }}
            }});
            renderMapFirstPoints();
            updateMapFirstSummary();
        }}

        // Clear all selections
        function clearMapFirstSelection() {{
            if (selectedMapFirstIndices.size > 0) {{
                selectedMapFirstIndices.clear();
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Delete all selected points
        function deleteSelectedPoints() {{
            if (selectedMapFirstIndices.size === 0) return;

            const count = selectedMapFirstIndices.size;
            if (confirm(`Delete ${{count}} selected point${{count > 1 ? 's' : ''}}?`)) {{
                selectedMapFirstIndices.forEach(index => {{
                    projectedPoints[index].visible = false;
                }});
                selectedMapFirstIndices.clear();
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Legacy function for backward compatibility
        function selectMapFirstPoint(index) {{
            toggleMapFirstPointSelection(index, false);
        }}

        // Deselect all points
        function deselectMapFirstPoint() {{
            clearMapFirstSelection();
        }}

        // Move the selected point(s) with arrow keys
        function moveSelectedPoint(dx, dy) {{
            if (selectedMapFirstIndices.size === 0) return;

            // Move all selected points
            let anyMoved = false;
            selectedMapFirstIndices.forEach(i => {{
                if (projectedPoints[i] && projectedPoints[i].visible !== false) {{
                    projectedPoints[i].pixel_u += dx;
                    projectedPoints[i].pixel_v += dy;
                    anyMoved = true;
                }}
            }});

            if (anyMoved) {{
                // Re-render to update marker positions
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Update selection indicator in the UI
        function updateSelectionIndicator() {{
            const existingIndicator = document.getElementById('selectionIndicator');
            if (existingIndicator) {{
                existingIndicator.remove();
            }}

            if (selectedMapFirstIndices.size > 0 && mapFirstMode) {{
                const selectionCount = selectedMapFirstIndices.size;
                let labelText = '';
                let driftText = '';

                if (selectionCount === 1) {{
                    // Single selection - show point name and drift
                    const index = Array.from(selectedMapFirstIndices)[0];
                    const kmlPoint = kmlPoints[index] || {{}};
                    labelText = `<strong>${{kmlPoint.name || 'Point ' + (index + 1)}}</strong> selected`;

                    // Calculate drift
                    const point = projectedPoints[index];
                    const orig = originalProjectedPositions[index];
                    if (orig && orig.pixel_u !== null && orig.pixel_v !== null) {{
                        const dx = point.pixel_u - orig.pixel_u;
                        const dy = point.pixel_v - orig.pixel_v;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        driftText = ` | Drift: ${{dist.toFixed(0)}}px`;
                    }}
                }} else {{
                    // Multi-selection - show count
                    labelText = `<strong>${{selectionCount}} points</strong> selected`;

                    // Calculate average drift
                    let totalDrift = 0;
                    let driftCount = 0;
                    selectedMapFirstIndices.forEach(index => {{
                        const point = projectedPoints[index];
                        const orig = originalProjectedPositions[index];
                        if (orig && orig.pixel_u !== null && orig.pixel_v !== null) {{
                            const dx = point.pixel_u - orig.pixel_u;
                            const dy = point.pixel_v - orig.pixel_v;
                            totalDrift += Math.sqrt(dx * dx + dy * dy);
                            driftCount++;
                        }}
                    }});
                    if (driftCount > 0) {{
                        driftText = ` | Avg drift: ${{(totalDrift / driftCount).toFixed(0)}}px`;
                    }}
                }}

                const indicator = document.createElement('div');
                indicator.id = 'selectionIndicator';
                indicator.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(76, 175, 80, 0.95);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    z-index: 10000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    display: flex;
                    align-items: center;
                    gap: 15px;
                `;
                indicator.innerHTML = `
                    <span>${{labelText}}${{driftText}}</span>
                    <span style="font-size: 12px; opacity: 0.9;">
                        ← → ↑ ↓ to move | Shift = faster | Esc = deselect
                    </span>
                    <button onclick="clearMapFirstSelection()" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        padding: 4px 10px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">✕</button>
                `;
                document.body.appendChild(indicator);
            }}
        }}

        // Discard a map-first point (now used by context menu only)
        function discardMapFirstPoint(index) {{
            projectedPoints[index].visible = false;
            selectedMapFirstIndices.delete(index);
            renderMapFirstPoints();
            updateMapFirstSummary();
        }}

        // Toggle Move All mode
        function toggleMoveAllMode() {{
            if (moveAllMode) {{
                // Exit Move All mode
                exitMoveAllMode();
            }} else {{
                // Enter Move All mode - exit other modes first
                selectedMapFirstIndices.clear();
                rotateAllMode = false;
                updateRotateAllUI();
                scaleAllMode = false;
                updateScaleAllUI();
                moveAllMode = true;
                updateMoveAllUI();
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Exit Move All mode
        function exitMoveAllMode() {{
            moveAllMode = false;
            updateMoveAllUI();
            renderMapFirstPoints();
            updateMapFirstSummary();
        }}

        // Update Move All button and indicator
        function updateMoveAllUI() {{
            const btn = document.getElementById('moveAllBtn');
            const existingIndicator = document.getElementById('moveAllIndicator');

            if (existingIndicator) {{
                existingIndicator.remove();
            }}

            if (moveAllMode) {{
                btn.textContent = 'Accept Movement';
                btn.style.background = '#4CAF50';
                btn.style.borderColor = '#4CAF50';

                // Show indicator at bottom
                const indicator = document.createElement('div');
                indicator.id = 'moveAllIndicator';
                indicator.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(255, 152, 0, 0.95);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    z-index: 10000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    display: flex;
                    align-items: center;
                    gap: 15px;
                `;

                // Count visible points
                const visibleCount = projectedPoints.filter(p => p.visible !== false && p.reason === 'visible').length;

                indicator.innerHTML = `
                    <span><strong>MOVE ALL MODE</strong> - Moving ${{visibleCount}} points</span>
                    <span style="font-size: 12px; opacity: 0.9;">
                        ← → ↑ ↓ to move | Shift = faster
                    </span>
                    <button onclick="exitMoveAllMode()" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        padding: 4px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Done</button>
                `;
                document.body.appendChild(indicator);
            }} else {{
                btn.textContent = 'Move All (W)';
                btn.style.background = '';
                btn.style.borderColor = '';
            }}
        }}

        // Move all visible points
        function moveAllPoints(dx, dy) {{
            if (!moveAllMode) return;

            let movedCount = 0;
            projectedPoints.forEach((point, i) => {{
                // Only move visible points that are in frame
                if (point.visible !== false && point.reason === 'visible') {{
                    point.pixel_u += dx;
                    point.pixel_v += dy;
                    movedCount++;
                }}
            }});

            if (movedCount > 0) {{
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // ============================================================
        // Rotate All Mode Functions
        // ============================================================

        // Toggle Rotate All mode
        function toggleRotateAllMode() {{
            if (rotateAllMode) {{
                // Exit Rotate All mode
                exitRotateAllMode();
            }} else {{
                // Enter Rotate All mode - exit other modes first
                selectedMapFirstIndices.clear();
                moveAllMode = false;
                updateMoveAllUI();
                scaleAllMode = false;
                updateScaleAllUI();
                rotateAllMode = true;
                updateRotateAllUI();
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Exit Rotate All mode
        function exitRotateAllMode() {{
            rotateAllMode = false;
            updateRotateAllUI();
            renderMapFirstPoints();
            updateMapFirstSummary();
        }}

        // Calculate centroid of all visible points
        function calculateCentroid() {{
            let sumU = 0, sumV = 0, count = 0;

            projectedPoints.forEach((point) => {{
                if (point.visible !== false && point.reason === 'visible') {{
                    sumU += point.pixel_u;
                    sumV += point.pixel_v;
                    count++;
                }}
            }});

            if (count === 0) return null;

            return {{
                u: sumU / count,
                v: sumV / count
            }};
        }}

        // Rotate all visible points around centroid
        function rotateAllPoints(angleDegrees) {{
            if (!rotateAllMode) return;

            const centroid = calculateCentroid();
            if (!centroid) return;

            const angleRadians = angleDegrees * Math.PI / 180;
            const cosA = Math.cos(angleRadians);
            const sinA = Math.sin(angleRadians);

            let rotatedCount = 0;
            projectedPoints.forEach((point) => {{
                if (point.visible !== false && point.reason === 'visible') {{
                    // Translate to origin (centroid)
                    const dx = point.pixel_u - centroid.u;
                    const dy = point.pixel_v - centroid.v;

                    // Rotate
                    const newDx = dx * cosA - dy * sinA;
                    const newDy = dx * sinA + dy * cosA;

                    // Translate back
                    point.pixel_u = centroid.u + newDx;
                    point.pixel_v = centroid.v + newDy;

                    rotatedCount++;
                }}
            }});

            if (rotatedCount > 0) {{
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Update Rotate All button and indicator
        function updateRotateAllUI() {{
            const btn = document.getElementById('rotateAllBtn');
            const existingIndicator = document.getElementById('rotateAllIndicator');

            if (existingIndicator) {{
                existingIndicator.remove();
            }}

            if (rotateAllMode) {{
                btn.textContent = 'Accept Rotation';
                btn.style.background = '#4CAF50';
                btn.style.borderColor = '#4CAF50';

                // Show indicator at bottom
                const indicator = document.createElement('div');
                indicator.id = 'rotateAllIndicator';
                indicator.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(156, 39, 176, 0.95);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    z-index: 10000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    display: flex;
                    align-items: center;
                    gap: 15px;
                `;

                // Count visible points
                const visibleCount = projectedPoints.filter(p => p.visible !== false && p.reason === 'visible').length;

                indicator.innerHTML = `
                    <span><strong>ROTATE MODE</strong> - Rotating ${{visibleCount}} points around centroid</span>
                    <span style="font-size: 12px; opacity: 0.9;">
                        ← → to rotate | Shift = faster (10×)
                    </span>
                    <button onclick="exitRotateAllMode()" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        padding: 4px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Done</button>
                `;
                document.body.appendChild(indicator);
            }} else {{
                btn.textContent = 'Rotate All (E)';
                btn.style.background = '';
                btn.style.borderColor = '';
            }}
        }}

        // ============================================================
        // Scale All Mode Functions
        // ============================================================

        // Toggle Scale All mode
        function toggleScaleAllMode() {{
            if (scaleAllMode) {{
                // Exit Scale All mode
                exitScaleAllMode();
            }} else {{
                // Enter Scale All mode - exit other modes first
                selectedMapFirstIndices.clear();
                moveAllMode = false;
                updateMoveAllUI();
                rotateAllMode = false;
                updateRotateAllUI();
                scaleAllMode = true;
                updateScaleAllUI();
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Exit Scale All mode
        function exitScaleAllMode() {{
            scaleAllMode = false;
            updateScaleAllUI();
            renderMapFirstPoints();
            updateMapFirstSummary();
        }}

        // Scale all visible points around centroid
        function scaleAllPoints(scaleX, scaleY) {{
            if (!scaleAllMode) return;

            const centroid = calculateCentroid();
            if (!centroid) return;

            let scaledCount = 0;
            projectedPoints.forEach((point) => {{
                if (point.visible !== false && point.reason === 'visible') {{
                    // Translate to origin (centroid)
                    const dx = point.pixel_u - centroid.u;
                    const dy = point.pixel_v - centroid.v;

                    // Scale
                    const newDx = dx * scaleX;
                    const newDy = dy * scaleY;

                    // Translate back
                    point.pixel_u = centroid.u + newDx;
                    point.pixel_v = centroid.v + newDy;

                    scaledCount++;
                }}
            }});

            if (scaledCount > 0) {{
                renderMapFirstPoints();
                updateMapFirstSummary();
            }}
        }}

        // Update Scale All button and indicator
        function updateScaleAllUI() {{
            const btn = document.getElementById('scaleAllBtn');
            const existingIndicator = document.getElementById('scaleAllIndicator');

            if (existingIndicator) {{
                existingIndicator.remove();
            }}

            if (scaleAllMode) {{
                btn.textContent = 'Accept Scaling';
                btn.style.background = '#4CAF50';
                btn.style.borderColor = '#4CAF50';

                // Show indicator at bottom
                const indicator = document.createElement('div');
                indicator.id = 'scaleAllIndicator';
                indicator.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(0, 150, 136, 0.95);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    z-index: 10000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    display: flex;
                    align-items: center;
                    gap: 15px;
                `;

                // Count visible points
                const visibleCount = projectedPoints.filter(p => p.visible !== false && p.reason === 'visible').length;

                indicator.innerHTML = `
                    <span><strong>SCALE MODE</strong> - Scaling ${{visibleCount}} points around centroid</span>
                    <span style="font-size: 12px; opacity: 0.9;">
                        ← → horizontal | ↑ ↓ vertical | Shift = 10%
                    </span>
                    <button onclick="exitScaleAllMode()" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        padding: 4px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Done</button>
                `;
                document.body.appendChild(indicator);
            }} else {{
                btn.textContent = 'Scale All (R)';
                btn.style.background = '';
                btn.style.borderColor = '';
            }}
        }}

        // Update the Map-First Mode summary panel
        function updateMapFirstSummary() {{
            if (!mapFirstMode) {{
                document.getElementById('mapFirstSummary').style.display = 'none';
                return;
            }}

            // Show the summary panel
            document.getElementById('mapFirstSummary').style.display = 'block';

            // Update KML file name
            if (kmlFileName) {{
                document.getElementById('kmlFileName').textContent = kmlFileName;
            }}

            // Count points by status
            let visibleCount = 0;
            let outOfViewCount = 0;
            let discardedCount = 0;
            let adjustedCount = 0;

            projectedPoints.forEach((point, i) => {{
                if (point.visible === false) {{
                    discardedCount++;
                }} else if (point.reason === 'visible') {{
                    visibleCount++;
                    // Check if point was adjusted from original position
                    if (originalProjectedPositions[i] &&
                        originalProjectedPositions[i].pixel_u !== null &&
                        originalProjectedPositions[i].pixel_v !== null) {{
                        const dx = point.pixel_u - originalProjectedPositions[i].pixel_u;
                        const dy = point.pixel_v - originalProjectedPositions[i].pixel_v;
                        if (Math.sqrt(dx*dx + dy*dy) > 2) {{  // More than 2 pixels moved
                            adjustedCount++;
                        }}
                    }}
                }} else if (point.reason === 'behind_camera' || point.reason === 'outside_bounds') {{
                    outOfViewCount++;
                }}
            }});

            // Update display
            document.getElementById('totalKmlPoints').textContent = kmlPoints.length;
            document.getElementById('visiblePoints').textContent = visibleCount;
            document.getElementById('outOfViewPoints').textContent = outOfViewCount;
            document.getElementById('discardedPoints').textContent = discardedCount;
            document.getElementById('adjustedPoints').textContent = adjustedCount;

            // Update height verification warning
            const warningEl = document.getElementById('heightWarning');
            if (heightVerification && heightVerification.warning) {{
                warningEl.textContent = heightVerification.warning;
                warningEl.style.display = 'block';
            }} else {{
                warningEl.style.display = 'none';
            }}
        }}

        // Handle YAML file upload
        function handleYamlUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                const content = e.target.result;

                fetch('/api/load_yaml', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ yaml_content: content }})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error loading YAML: ' + data.error);
                        return;
                    }}

                    // Set coordinate system based on loaded data
                    // If null/undefined, it's legacy format (V stored as leaflet_y)
                    coordinateSystem = data.coordinate_system || null;
                    console.log('Loaded coordinate system:', coordinateSystem);

                    gcps = data.gcps;
                    distribution = data.distribution;
                    homography = data.homography;
                    updateUI();

                    let msg = `Loaded ${{data.gcps_loaded}} GCPs`;
                    if (data.warnings && data.warnings.length > 0) {{
                        msg += '\\n\\nWarnings:\\n' + data.warnings.join('\\n');
                    }}

                    // Check if PTZ position differs and live camera is available
                    if (data.has_live_camera && data.loaded_ptz && data.current_ptz) {{
                        const loadedPtz = data.loaded_ptz;
                        const currentPtz = data.current_ptz;

                        // Check if positions differ significantly (threshold: 0.5 deg for pan/tilt, 0.1 for zoom)
                        const panDiff = Math.abs((loadedPtz.pan || 0) - (currentPtz.pan || 0));
                        const tiltDiff = Math.abs((loadedPtz.tilt || 0) - (currentPtz.tilt || 0));
                        const zoomDiff = Math.abs((loadedPtz.zoom || 1) - (currentPtz.zoom || 1));

                        if (panDiff > 0.5 || tiltDiff > 0.5 || zoomDiff > 0.1) {{
                            const moveMsg = `\\n\\nThe YAML was captured at a different PTZ position:\\n` +
                                `  Loaded:  P=${{loadedPtz.pan?.toFixed(1) || '?'}}° T=${{loadedPtz.tilt?.toFixed(1) || '?'}}° Z=${{loadedPtz.zoom?.toFixed(1) || '?'}}x\\n` +
                                `  Current: P=${{currentPtz.pan?.toFixed(1) || '?'}}° T=${{currentPtz.tilt?.toFixed(1) || '?'}}° Z=${{currentPtz.zoom?.toFixed(1) || '?'}}x\\n\\n` +
                                `Move camera to the saved position?`;

                            if (confirm(msg + moveMsg)) {{
                                // Move camera to loaded position
                                moveCameraToPosition(loadedPtz);
                                return;
                            }}
                        }}
                    }}

                    alert(msg);
                }});
            }};
            reader.readAsText(file);

            // Reset input so same file can be loaded again
            event.target.value = '';
        }}

        // Move camera to PTZ position
        function moveCameraToPosition(ptzPosition) {{
            const statusEl = document.getElementById('instructions');
            const originalText = statusEl.textContent;
            const originalBg = statusEl.style.background;

            statusEl.textContent = 'Moving camera to saved position...';
            statusEl.style.background = 'rgba(255, 152, 0, 0.9)';

            fetch('/api/move_camera', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    ptz_position: ptzPosition,
                    wait_time: 3.0
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                statusEl.textContent = originalText;
                statusEl.style.background = originalBg;

                if (data.success) {{
                    alert(data.message + '\\n\\nNote: The displayed frame is from the old position. Reload page for new frame.');
                }} else {{
                    alert('Failed to move camera: ' + data.message);
                }}
            }})
            .catch(err => {{
                statusEl.textContent = originalText;
                statusEl.style.background = originalBg;
                alert('Error moving camera: ' + err);
            }});
        }}

        // ============================================================
        // Batch Mode Functions
        // ============================================================

        function toggleBatchMode() {{
            if (batchMode) {{
                // Turning off batch mode - show finalize modal if there are points
                if (batchPoints.length > 0) {{
                    openBatchModal();
                }} else {{
                    batchMode = false;
                    updateBatchModeUI();
                }}
            }} else {{
                // Turning on batch mode
                batchMode = true;
                batchPoints = [];
                batchGpsCoords = [];
                clearBatchMarkers();
                updateBatchModeUI();
            }}
        }}

        function updateBatchModeUI() {{
            const btn = document.getElementById('batchModeBtn');
            const instructions = document.getElementById('instructions');
            if (batchMode) {{
                btn.textContent = `Batch Mode: ON (${{batchPoints.length}})`;
                btn.classList.add('active');
                instructions.textContent = 'BATCH MODE: Click points to add them. Click button again to finalize.';
                instructions.style.background = 'rgba(255, 152, 0, 0.9)';
                instructions.style.color = 'black';
            }} else {{
                btn.textContent = 'Batch Mode: OFF';
                btn.classList.remove('active');
                instructions.textContent = 'Scroll to zoom, drag to pan. Click to add GCP.';
                instructions.style.background = 'rgba(0,0,0,0.7)';
                instructions.style.color = 'white';
            }}
        }}

        function updateBatchUI() {{
            document.getElementById('batchModeBtn').textContent = `Batch Mode: ON (${{batchPoints.length}})`;
        }}

        function updateBatchMarkers() {{
            // Clear existing batch markers
            clearBatchMarkers();

            // Add numbered markers for batch points
            batchPoints.forEach((pt, i) => {{
                // Convert image coords to Leaflet coords (invert Y)
                const leafletCoords = imageToLeaflet(pt.u, pt.v);
                const marker = L.marker(leafletCoords, {{
                    icon: L.divIcon({{
                        className: 'batch-marker',
                        html: `<div style="
                            width: 24px;
                            height: 24px;
                            background: #ff9800;
                            border: 2px solid white;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 11px;
                            font-weight: bold;
                            color: black;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        ">${{i + 1}}</div>`,
                        iconSize: [24, 24],
                        iconAnchor: [12, 12]
                    }}),
                    draggable: true
                }}).addTo(map);

                // Stop click propagation to prevent adding duplicate points
                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});
                marker.on('mousedown', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                // Allow dragging batch markers to adjust position
                marker.on('dragend', function(e) {{
                    const newPos = e.target.getLatLng();
                    // Convert Leaflet coords back to image coords (coordinate-system aware)
                    const newImgCoords = leafletToImage(newPos.lat, newPos.lng);
                    batchPoints[i].u = newImgCoords.u;
                    batchPoints[i].v = newImgCoords.v;
                }});

                batchMarkers.push(marker);
            }});
        }}

        function clearBatchMarkers() {{
            batchMarkers.forEach(m => map.removeLayer(m));
            batchMarkers = [];
        }}

        function openBatchModal() {{
            document.getElementById('batchPointCount').textContent = batchPoints.length;
            document.getElementById('kmlStatus').textContent = '';
            batchGpsCoords = [];

            // Build the points list
            let html = '<table style="width: 100%; border-collapse: collapse;">';
            html += '<tr style="background: #3a3a3a; font-size: 12px;">';
            html += '<th style="padding: 8px; text-align: left;">#</th>';
            html += '<th style="padding: 8px; text-align: left;">Pixel (u, v)</th>';
            html += '<th style="padding: 8px; text-align: left;">Name</th>';
            html += '<th style="padding: 8px; text-align: left;">GPS</th>';
            html += '<th style="padding: 8px; text-align: center;">Del</th>';
            html += '</tr>';

            batchPoints.forEach((pt, i) => {{
                html += `<tr style="border-bottom: 1px solid #444;">`;
                html += `<td style="padding: 8px; font-size: 12px;">${{i + 1}}</td>`;
                html += `<td style="padding: 8px; font-size: 12px;">(${{pt.u.toFixed(1)}}, ${{pt.v.toFixed(1)}})</td>`;
                html += `<td style="padding: 8px;"><input type="text" id="batchName_${{i}}" placeholder="GCP ${{i + 1}}" style="width: 150px; padding: 4px; font-size: 12px; background: #3a3a3a; border: 1px solid #555; color: white; border-radius: 4px;"></td>`;
                html += `<td style="padding: 8px; font-size: 12px;" id="batchGps_${{i}}">-</td>`;
                html += `<td style="padding: 8px; text-align: center;"><button onclick="removeBatchPoint(${{i}})" style="background: #c62828; color: white; border: none; padding: 2px 8px; cursor: pointer; border-radius: 3px; font-size: 11px;">×</button></td>`;
                html += '</tr>';
            }});
            html += '</table>';

            document.getElementById('batchPointsList').innerHTML = html;
            document.getElementById('batchConfirmBtn').disabled = true;
            document.getElementById('batchModal').classList.add('active');
        }}

        function closeBatchModal() {{
            document.getElementById('batchModal').classList.remove('active');
            // Keep batch mode on, user can continue adding points
        }}

        function removeBatchPoint(index) {{
            batchPoints.splice(index, 1);
            if (batchGpsCoords.length > index) {{
                batchGpsCoords.splice(index, 1);
            }}
            updateBatchMarkers();
            openBatchModal(); // Refresh the modal
        }}

        function handleKmlUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                const content = e.target.result;
                const coords = parseKmlCoordinates(content);

                if (coords.length === 0) {{
                    document.getElementById('kmlStatus').textContent = '❌ No coordinates found in KML';
                    document.getElementById('kmlStatus').style.color = '#f44336';
                    return;
                }}

                if (coords.length !== batchPoints.length) {{
                    document.getElementById('kmlStatus').textContent = `❌ Mismatch: KML has ${{coords.length}} points, you clicked ${{batchPoints.length}}`;
                    document.getElementById('kmlStatus').style.color = '#f44336';
                    alert(`Count mismatch!\\n\\nKML file has ${{coords.length}} coordinates.\\nYou clicked ${{batchPoints.length}} points.\\n\\nPlease ensure the KML has the same number of points in the correct order.`);
                    return;
                }}

                // Assign GPS coords to batch points
                batchGpsCoords = coords;
                document.getElementById('kmlStatus').textContent = `✓ Loaded ${{coords.length}} coordinates`;
                document.getElementById('kmlStatus').style.color = '#4CAF50';

                // Update GPS display in table
                coords.forEach((coord, i) => {{
                    const gpsCell = document.getElementById(`batchGps_${{i}}`);
                    if (gpsCell) {{
                        gpsCell.textContent = `${{coord.lat.toFixed(6)}}, ${{coord.lon.toFixed(6)}}`;
                        gpsCell.style.color = '#4CAF50';
                    }}
                }});

                // Enable confirm button
                document.getElementById('batchConfirmBtn').disabled = false;
            }};
            reader.readAsText(file);

            // Reset input so same file can be loaded again
            event.target.value = '';
        }}

        function parseKmlCoordinates(kmlContent) {{
            const coords = [];

            // Parse KML - look for <coordinates> tags
            // KML format: longitude,latitude,altitude (whitespace separated)
            const coordPattern = /<coordinates>([^<]+)<\\/coordinates>/gi;
            let match;

            while ((match = coordPattern.exec(kmlContent)) !== null) {{
                const coordText = match[1].trim();
                // Split by whitespace (newlines, spaces)
                const points = coordText.split(/\\s+/);

                points.forEach(point => {{
                    if (!point.trim()) return;
                    const parts = point.split(',');
                    if (parts.length >= 2) {{
                        const lon = parseFloat(parts[0]);
                        const lat = parseFloat(parts[1]);
                        if (!isNaN(lat) && !isNaN(lon)) {{
                            coords.push({{ lat, lon }});
                        }}
                    }}
                }});
            }}

            // Also try to parse <Point> elements with single coordinates
            const pointPattern = /<Point>[^<]*<coordinates>([^<]+)<\\/coordinates>[^<]*<\\/Point>/gi;
            while ((match = pointPattern.exec(kmlContent)) !== null) {{
                const coordText = match[1].trim();
                const parts = coordText.split(',');
                if (parts.length >= 2) {{
                    const lon = parseFloat(parts[0]);
                    const lat = parseFloat(parts[1]);
                    if (!isNaN(lat) && !isNaN(lon)) {{
                        // Check if not already added
                        const exists = coords.some(c => c.lat === lat && c.lon === lon);
                        if (!exists) {{
                            coords.push({{ lat, lon }});
                        }}
                    }}
                }}
            }}

            return coords;
        }}

        async function confirmBatchGCPs() {{
            if (batchGpsCoords.length !== batchPoints.length) {{
                alert('Please load a KML file with matching coordinates first.');
                return;
            }}

            // Add all GCPs one by one
            for (let i = 0; i < batchPoints.length; i++) {{
                const pt = batchPoints[i];
                const gps = batchGpsCoords[i];
                const nameInput = document.getElementById(`batchName_${{i}}`);
                const name = nameInput ? nameInput.value.trim() || `GCP ${{i + 1}}` : `GCP ${{i + 1}}`;

                await fetch('/api/add_gcp', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        u: pt.u,
                        v: pt.v,
                        lat: gps.lat,
                        lon: gps.lon,
                        description: name,
                        accuracy: 'medium'
                    }})
                }})
                .then(r => r.json())
                .then(data => {{
                    gcps = data.gcps;
                    distribution = data.distribution;
                }});
            }}

            // Clean up batch mode
            batchMode = false;
            batchPoints = [];
            batchGpsCoords = [];
            clearBatchMarkers();
            updateBatchModeUI();
            updateUI();

            document.getElementById('batchModal').classList.remove('active');
            alert(`Added ${{batchPoints.length || gcps.length}} GCPs successfully!`);
        }}

        // ============================================================
        // Map-First Mode: Convert KML Points to GCPs
        // ============================================================

        function convertAllToGCPs() {{
            if (!mapFirstMode) {{
                alert('This function is only available in map-first mode.');
                return;
            }}

            // Filter visible points
            const visiblePoints = projectedPoints.filter(p => p.visible !== false && p.reason === 'visible');

            if (visiblePoints.length === 0) {{
                alert('No visible points to convert. Make sure KML points are projected within the camera view.');
                return;
            }}

            // Prepare data with current (possibly adjusted) positions
            const convertData = projectedPoints.map((point, index) => ({{
                pixel_u: point.pixel_u,
                pixel_v: point.pixel_v,
                visible: point.visible !== false && point.reason === 'visible',
                kml_index: index
            }}));

            // Call API to convert KML points to GCPs (without saving to file)
            fetch('/api/convert_kml_to_gcps', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{projected_points: convertData}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    // Update local GCPs list and UI
                    gcps = data.gcps || [];
                    distribution = data.distribution || {{}};
                    homography = data.homography || {{}};
                    updateUI();

                    // Show success message (no file download)
                    const count = data.gcps_converted || gcps.length;
                    console.log(`Converted ${{count}} KML points to GCPs`);

                    // Update button to show conversion done
                    const btn = document.querySelector('button[onclick="convertAllToGCPs()"]');
                    if (btn) {{
                        btn.textContent = `Converted (${{count}} GCPs)`;
                        btn.style.background = '#4CAF50';
                        btn.style.borderColor = '#4CAF50';
                    }}
                }} else {{
                    alert('Failed to convert KML points: ' + (data.error || 'Unknown error'));
                }}
            }})
            .catch(err => {{
                console.error('Conversion error:', err);
                alert('Error converting KML points to GCPs: ' + err);
            }});
        }}

        // ============================================================
        // Drift Analysis Functions
        // ============================================================

        function calculateDriftVectors() {{
            const driftData = [];

            projectedPoints.forEach((point, i) => {{
                // Skip non-visible or discarded points
                if (point.visible === false || point.reason !== 'visible') return;

                const original = originalProjectedPositions[i];
                if (!original || original.pixel_u === null || original.pixel_v === null) return;

                const dx = point.pixel_u - original.pixel_u;
                const dy = point.pixel_v - original.pixel_v;
                const distance = Math.sqrt(dx * dx + dy * dy);

                // Only include if actually moved (threshold: 2 pixels)
                if (distance > 2) {{
                    driftData.push({{
                        index: i,
                        name: kmlPoints[i]?.name || `Point ${{i + 1}}`,
                        original_u: original.pixel_u,
                        original_v: original.pixel_v,
                        current_u: point.pixel_u,
                        current_v: point.pixel_v,
                        dx: dx,
                        dy: dy,
                        distance: distance,
                        // Direction in degrees (0 = right, 90 = down)
                        direction: Math.atan2(dy, dx) * 180 / Math.PI
                    }});
                }}
            }});

            return driftData;
        }}

        function analyzeDriftPattern(driftData) {{
            if (driftData.length < 2) {{
                return {{
                    pattern: 'insufficient_data',
                    message: 'Need at least 2 adjusted points for drift analysis.',
                    suggestions: []
                }};
            }}

            // Calculate statistics
            const dxValues = driftData.map(d => d.dx);
            const dyValues = driftData.map(d => d.dy);
            const distances = driftData.map(d => d.distance);

            const meanDx = dxValues.reduce((a, b) => a + b, 0) / dxValues.length;
            const meanDy = dyValues.reduce((a, b) => a + b, 0) / dyValues.length;
            const meanDistance = distances.reduce((a, b) => a + b, 0) / distances.length;

            const stdDx = Math.sqrt(dxValues.reduce((sum, x) => sum + Math.pow(x - meanDx, 2), 0) / dxValues.length);
            const stdDy = Math.sqrt(dyValues.reduce((sum, y) => sum + Math.pow(y - meanDy, 2), 0) / dyValues.length);
            const stdDistance = Math.sqrt(distances.reduce((sum, d) => sum + Math.pow(d - meanDistance, 2), 0) / distances.length);

            // Coefficient of variation (how consistent is the drift?)
            const cvDx = stdDx / Math.abs(meanDx) || Infinity;
            const cvDy = stdDy / Math.abs(meanDy) || Infinity;

            const result = {{
                stats: {{
                    mean_dx: meanDx,
                    mean_dy: meanDy,
                    mean_distance: meanDistance,
                    std_dx: stdDx,
                    std_dy: stdDy,
                    std_distance: stdDistance,
                    num_points: driftData.length
                }},
                pattern: 'unknown',
                message: '',
                suggestions: [],
                confidence: 'low'
            }};

            // Analyze pattern
            const isConsistentDx = cvDx < 0.5;  // CV < 50%
            const isConsistentDy = cvDy < 0.5;

            // Check for uniform translation (all points shifted same direction)
            if (isConsistentDx && isConsistentDy && stdDistance < meanDistance * 0.3) {{
                result.pattern = 'uniform_translation';
                result.message = `All points shifted uniformly by (~${{meanDx.toFixed(1)}}px, ~${{meanDy.toFixed(1)}}px)`;
                result.confidence = 'high';

                // This suggests camera GPS position error or pan offset error
                if (Math.abs(meanDx) > Math.abs(meanDy) * 2) {{
                    result.suggestions.push({{
                        type: 'pan_offset',
                        message: `Horizontal drift suggests pan offset error. Consider adjusting pan_offset_deg.`,
                        estimated_correction: meanDx > 0 ? 'Increase pan_offset_deg' : 'Decrease pan_offset_deg'
                    }});
                }} else if (Math.abs(meanDy) > Math.abs(meanDx) * 2) {{
                    result.suggestions.push({{
                        type: 'tilt_or_height',
                        message: `Vertical drift suggests tilt or height error.`,
                        estimated_correction: meanDy > 0 ? 'Height may be too low or tilt too high' : 'Height may be too high or tilt too low'
                    }});
                }} else {{
                    result.suggestions.push({{
                        type: 'camera_position',
                        message: `Combined drift suggests camera GPS position error.`,
                        estimated_correction: `Check camera lat/lon configuration`
                    }});
                }}
            }}
            // Check for radial pattern (scaling from center)
            else {{
                // Calculate if points move radially from image center
                const centerU = imageWidth / 2;
                const centerV = imageHeight / 2;

                let radialScore = 0;
                driftData.forEach(d => {{
                    const fromCenterU = d.original_u - centerU;
                    const fromCenterV = d.original_v - centerV;
                    const toPointU = d.current_u - d.original_u;
                    const toPointV = d.current_v - d.original_v;

                    // Dot product to check if drift is along radial direction
                    const dot = fromCenterU * toPointU + fromCenterV * toPointV;
                    const radialMag = Math.sqrt(fromCenterU*fromCenterU + fromCenterV*fromCenterV);
                    const driftMag = Math.sqrt(toPointU*toPointU + toPointV*toPointV);

                    if (radialMag > 10 && driftMag > 2) {{
                        radialScore += dot / (radialMag * driftMag);  // Cosine similarity
                    }}
                }});

                radialScore /= driftData.length;

                if (Math.abs(radialScore) > 0.6) {{
                    result.pattern = 'radial_scaling';
                    const direction = radialScore > 0 ? 'outward' : 'inward';
                    result.message = `Points drift ${{direction}} from image center (scale factor error)`;
                    result.confidence = 'medium';
                    result.suggestions.push({{
                        type: 'height',
                        message: `Radial ${{direction}} drift strongly suggests camera height error.`,
                        estimated_correction: radialScore > 0
                            ? `Height appears LOWER than configured. Try increasing height_m.`
                            : `Height appears HIGHER than configured. Try decreasing height_m.`
                    }});
                }} else {{
                    result.pattern = 'mixed';
                    result.message = `Mixed drift pattern - multiple error sources likely`;
                    result.confidence = 'low';
                    result.suggestions.push({{
                        type: 'multiple',
                        message: `Complex drift pattern may indicate multiple calibration errors.`,
                        estimated_correction: `Check height, pan_offset, and camera GPS coordinates.`
                    }});
                }}
            }}

            // Estimate height correction if we have enough data
            if (driftData.length >= 3 && heightVerification) {{
                const configuredHeight = heightVerification.configured_height || 5.0;

                // Use radial drift to estimate scale factor
                let scaleSamples = [];
                const centerU = imageWidth / 2;
                const centerV = imageHeight / 2;

                driftData.forEach(d => {{
                    const origDist = Math.sqrt(Math.pow(d.original_u - centerU, 2) + Math.pow(d.original_v - centerV, 2));
                    const currDist = Math.sqrt(Math.pow(d.current_u - centerU, 2) + Math.pow(d.current_v - centerV, 2));

                    if (origDist > 50) {{  // Only use points reasonably far from center
                        scaleSamples.push(currDist / origDist);
                    }}
                }});

                if (scaleSamples.length >= 2) {{
                    const avgScale = scaleSamples.reduce((a, b) => a + b, 0) / scaleSamples.length;
                    const estimatedHeight = configuredHeight / avgScale;

                    if (Math.abs(avgScale - 1.0) > 0.05) {{  // More than 5% scale change
                        result.estimatedHeight = estimatedHeight;
                        result.suggestions.push({{
                            type: 'height_estimate',
                            message: `Based on point adjustments, estimated true height: ${{estimatedHeight.toFixed(2)}}m (configured: ${{configuredHeight.toFixed(2)}}m)`,
                            estimated_correction: `Set height_m to approximately ${{estimatedHeight.toFixed(1)}}m`
                        }});
                    }}
                }}
            }}

            // Estimate camera GPS correction from uniform translation drift
            if (heightVerification && heightVerification.camera_lat && heightVerification.camera_lon &&
                driftData.length >= 2 && (result.pattern === 'uniform_translation' || Math.abs(meanDx) > 5 || Math.abs(meanDy) > 5)) {{

                const configuredHeight = heightVerification.configured_height || 5.0;
                const tiltDeg = heightVerification.tilt_deg || 45;
                const panDeg = heightVerification.pan_deg || 0;
                const cameraLat = heightVerification.camera_lat;
                const cameraLon = heightVerification.camera_lon;
                const imgWidth = heightVerification.image_width || imageWidth;
                const imgHeight = heightVerification.image_height || imageHeight;

                // Calculate approximate meters per pixel at scene center
                // Ground distance from camera = height / tan(tilt)
                // For a typical PTZ camera with ~60° horizontal FOV
                const tiltRad = tiltDeg * Math.PI / 180;
                const groundDistanceAtCenter = configuredHeight / Math.tan(tiltRad);

                // Approximate horizontal FOV based on typical PTZ camera
                const hFovDeg = 60;  // Typical PTZ horizontal FOV at 1x zoom
                const hFovRad = hFovDeg * Math.PI / 180;
                const viewWidthAtCenter = 2 * groundDistanceAtCenter * Math.tan(hFovRad / 2);

                // Meters per pixel
                const metersPerPixelX = viewWidthAtCenter / imgWidth;
                const metersPerPixelY = metersPerPixelX;  // Approximate square pixels

                // Convert pixel drift to meters in camera coordinate system
                // Note: positive dx means points need to move right in image,
                // which means the projected points are too far left,
                // which means camera thinks it's further right than it is
                // So we need to move camera LEFT (subtract from position in that direction)
                const driftMetersRight = -meanDx * metersPerPixelX;  // Camera needs to move this much right
                const driftMetersForward = -meanDy * metersPerPixelY;  // Camera needs to move this much forward

                // Rotate by pan angle to get East/North offset
                // Pan=0 means camera pointing North, pan increases clockwise
                const panRad = panDeg * Math.PI / 180;
                const driftEast = driftMetersRight * Math.cos(panRad) + driftMetersForward * Math.sin(panRad);
                const driftNorth = -driftMetersRight * Math.sin(panRad) + driftMetersForward * Math.cos(panRad);

                // Convert to lat/lon offset
                // 1 degree latitude ≈ 111,111 meters
                // 1 degree longitude ≈ 111,111 * cos(latitude) meters
                const metersPerDegreeLat = 111111;
                const metersPerDegreeLon = 111111 * Math.cos(cameraLat * Math.PI / 180);

                const driftLat = driftNorth / metersPerDegreeLat;
                const driftLon = driftEast / metersPerDegreeLon;

                // Calculate suggested camera GPS
                const suggestedLat = cameraLat + driftLat;
                const suggestedLon = cameraLon + driftLon;

                // Only suggest if drift is significant (> 0.5 meters)
                const totalDriftMeters = Math.sqrt(driftEast * driftEast + driftNorth * driftNorth);

                if (totalDriftMeters > 0.5) {{
                    result.estimatedCameraGPS = {{
                        latitude: suggestedLat,
                        longitude: suggestedLon,
                        currentLatitude: cameraLat,
                        currentLongitude: cameraLon,
                        driftEast: driftEast,
                        driftNorth: driftNorth,
                        driftMeters: totalDriftMeters,
                        metersPerPixel: metersPerPixelX
                    }};

                    result.suggestions.push({{
                        type: 'camera_gps_estimate',
                        message: `Based on uniform drift of ${{totalDriftMeters.toFixed(1)}}m (${{driftEast >= 0 ? '+' : ''}}${{driftEast.toFixed(1)}}m E, ${{driftNorth >= 0 ? '+' : ''}}${{driftNorth.toFixed(1)}}m N), estimated camera GPS:`,
                        estimated_correction: `Suggested: ${{suggestedLat.toFixed(6)}}, ${{suggestedLon.toFixed(6)}}\\nCurrent: ${{cameraLat.toFixed(6)}}, ${{cameraLon.toFixed(6)}}`
                    }});
                }}
            }}

            return result;
        }}

        function showDriftAnalysis() {{
            if (!mapFirstMode) {{
                alert('Drift analysis is only available in map-first mode.');
                return;
            }}

            const driftData = calculateDriftVectors();
            const analysis = analyzeDriftPattern(driftData);

            driftAnalysisResult = analysis;
            suggestedCalibration = analysis.estimatedHeight || null;

            // Show the drift analysis panel
            document.getElementById('driftAnalysisPanel').style.display = 'block';

            const resultsEl = document.getElementById('driftResults');
            const recommendationsEl = document.getElementById('driftRecommendations');
            const recommendationsContent = document.getElementById('driftRecommendationsContent');

            if (driftData.length === 0) {{
                resultsEl.innerHTML = `
                    <p style="color: #ff9800;">No adjusted points detected.</p>
                    <p style="color: #888; margin-top: 8px;">Drag KML markers to match visible features in the image, then click "Analyze Drift" again.</p>
                `;
                recommendationsEl.style.display = 'none';
                return;
            }}

            // Show statistics
            const stats = analysis.stats;
            let html = `
                <div style="margin-bottom: 12px;">
                    <div style="font-weight: 600; color: #fff; margin-bottom: 6px;">
                        Pattern: <span style="color: ${{analysis.confidence === 'high' ? '#4CAF50' : analysis.confidence === 'medium' ? '#ff9800' : '#f44336'}};">
                        ${{analysis.pattern.replace(/_/g, ' ')}}</span>
                        <span style="font-size: 11px; color: #888;">(${{analysis.confidence}} confidence)</span>
                    </div>
                    <div style="color: #ccc;">${{analysis.message}}</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px; background: #3a3a3a; padding: 10px; border-radius: 4px;">
                    <div>Points adjusted: <strong>${{stats.num_points}}</strong></div>
                    <div>Mean distance: <strong>${{stats.mean_distance.toFixed(1)}}px</strong></div>
                    <div>Mean Δx: <strong>${{stats.mean_dx.toFixed(1)}}px</strong></div>
                    <div>Mean Δy: <strong>${{stats.mean_dy.toFixed(1)}}px</strong></div>
                    <div>Std Δx: <strong>${{stats.std_dx.toFixed(1)}}px</strong></div>
                    <div>Std Δy: <strong>${{stats.std_dy.toFixed(1)}}px</strong></div>
                </div>
            `;

            // Show per-point drift table
            if (driftData.length <= 10) {{
                html += `
                    <div style="margin-top: 12px; font-size: 11px;">
                        <div style="font-weight: 600; margin-bottom: 4px;">Individual Point Drift:</div>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="color: #888;">
                                <th style="text-align: left; padding: 2px 4px;">Point</th>
                                <th style="text-align: right; padding: 2px 4px;">Δx</th>
                                <th style="text-align: right; padding: 2px 4px;">Δy</th>
                                <th style="text-align: right; padding: 2px 4px;">Dist</th>
                            </tr>
                `;
                driftData.forEach(d => {{
                    html += `
                        <tr>
                            <td style="padding: 2px 4px;">${{d.name}}</td>
                            <td style="text-align: right; padding: 2px 4px; color: ${{d.dx > 0 ? '#4CAF50' : '#f44336'}};">${{d.dx.toFixed(1)}}</td>
                            <td style="text-align: right; padding: 2px 4px; color: ${{d.dy > 0 ? '#4CAF50' : '#f44336'}};">${{d.dy.toFixed(1)}}</td>
                            <td style="text-align: right; padding: 2px 4px;">${{d.distance.toFixed(1)}}</td>
                        </tr>
                    `;
                }});
                html += `</table></div>`;
            }}

            resultsEl.innerHTML = html;

            // Show recommendations
            if (analysis.suggestions.length > 0) {{
                recommendationsEl.style.display = 'block';
                let recHtml = '';
                analysis.suggestions.forEach(s => {{
                    // Choose border color based on suggestion type
                    let borderColor = '#ff9800';  // Default orange
                    if (s.type === 'height_estimate') borderColor = '#4CAF50';  // Green
                    if (s.type === 'camera_gps_estimate') borderColor = '#2196F3';  // Blue

                    // Handle multi-line corrections (especially for GPS coordinates)
                    const correctionHtml = s.estimated_correction.replace(/\\n/g, '<br>');

                    recHtml += `
                        <div style="background: #3a3a3a; padding: 10px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid ${{borderColor}};">
                            <div style="font-weight: 500; color: #fff; margin-bottom: 4px;">${{s.type.replace(/_/g, ' ').toUpperCase()}}</div>
                            <div style="font-size: 12px; color: #ccc;">${{s.message}}</div>
                            <div style="font-size: 11px; color: #4CAF50; margin-top: 4px; font-family: monospace;">${{correctionHtml}}</div>
                        </div>
                    `;
                }});
                recommendationsContent.innerHTML = recHtml;

                // Enable calibration button if we have height estimate
                if (analysis.estimatedHeight) {{
                    document.getElementById('applyCalibrationBtn').disabled = false;
                }}
            }} else {{
                recommendationsEl.style.display = 'none';
            }}
        }}

        function applyCalibration() {{
            if (!suggestedCalibration) {{
                alert('No calibration suggestion available. Run drift analysis first.');
                return;
            }}

            const currentHeight = heightVerification?.configured_height || 5.0;
            const newHeight = suggestedCalibration;

            if (!confirm(`Apply calibration?\n\nCurrent height: ${{currentHeight.toFixed(2)}}m\nSuggested height: ${{newHeight.toFixed(2)}}m\n\nThis will update the camera configuration.`)) {{
                return;
            }}

            // Send calibration request to server
            fetch('/api/apply_calibration', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    estimated_height: newHeight,
                    drift_analysis: driftAnalysisResult
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    alert(`Calibration applied!\n\nNew height: ${{newHeight.toFixed(2)}}m\n\nNote: To see the effect, reload the page with the new camera parameters.`);

                    // Update local height verification
                    if (heightVerification) {{
                        heightVerification.configured_height = newHeight;
                    }}
                }} else {{
                    alert('Failed to apply calibration: ' + (data.error || 'Unknown error'));
                }}
            }})
            .catch(err => {{
                console.error('Calibration error:', err);
                alert('Error applying calibration: ' + err);
            }});
        }}

        // Handle keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Escape key - close modals, exit modes, or deselect point
            if (e.key === 'Escape') {{
                if (scaleAllMode) {{
                    exitScaleAllMode();
                    e.preventDefault();
                    return;
                }}
                if (rotateAllMode) {{
                    exitRotateAllMode();
                    e.preventDefault();
                    return;
                }}
                if (moveAllMode) {{
                    exitMoveAllMode();
                    e.preventDefault();
                    return;
                }}
                if (selectedMapFirstIndices.size > 0) {{
                    clearMapFirstSelection();
                    e.preventDefault();
                    return;
                }}
                closeModal();
                closeBatchModal();
            }}

            // Enter key - confirm modal or accept current mode
            if (e.key === 'Enter') {{
                if (scaleAllMode) {{
                    exitScaleAllMode();
                    e.preventDefault();
                    return;
                }}
                if (rotateAllMode) {{
                    exitRotateAllMode();
                    e.preventDefault();
                    return;
                }}
                if (moveAllMode) {{
                    exitMoveAllMode();
                    e.preventDefault();
                    return;
                }}
                if (document.getElementById('addGcpModal').classList.contains('active')) {{
                    confirmAddGCP();
                }}
            }}

            // W/E/R keys - toggle Move/Rotate/Scale All modes
            // Skip if user is typing in an input field
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {{
                if (e.key === 'w' || e.key === 'W') {{
                    toggleMoveAllMode();
                    e.preventDefault();
                    return;
                }}
                if (e.key === 'e' || e.key === 'E') {{
                    toggleRotateAllMode();
                    e.preventDefault();
                    return;
                }}
                if (e.key === 'r' || e.key === 'R') {{
                    toggleScaleAllMode();
                    e.preventDefault();
                    return;
                }}
            }}

            // Arrow keys - handle different modes
            if (mapFirstMode) {{
                // Rotation speed: normal = 0.5 degrees, shift = 5 degrees
                const rotationSpeed = e.shiftKey ? 5 : 0.5;
                // Movement speed: normal = 1px, shift = 10px
                const moveSpeed = e.shiftKey ? 10 : 1;
                // Scale speed: normal = 1% (0.01), shift = 10% (0.10)
                const scaleSpeed = e.shiftKey ? 0.10 : 0.01;

                // Scale All mode - arrow keys scale around centroid
                if (scaleAllMode) {{
                    switch (e.key) {{
                        case 'ArrowLeft':
                            scaleAllPoints(1 - scaleSpeed, 1.0);
                            e.preventDefault();
                            break;
                        case 'ArrowRight':
                            scaleAllPoints(1 + scaleSpeed, 1.0);
                            e.preventDefault();
                            break;
                        case 'ArrowUp':
                            scaleAllPoints(1.0, 1 - scaleSpeed);
                            e.preventDefault();
                            break;
                        case 'ArrowDown':
                            scaleAllPoints(1.0, 1 + scaleSpeed);
                            e.preventDefault();
                            break;
                    }}
                }}
                // Rotate All mode - left/right arrows rotate around centroid
                else if (rotateAllMode) {{
                    switch (e.key) {{
                        case 'ArrowLeft':
                            rotateAllPoints(-rotationSpeed);
                            e.preventDefault();
                            break;
                        case 'ArrowRight':
                            rotateAllPoints(rotationSpeed);
                            e.preventDefault();
                            break;
                    }}
                }}
                // Move All mode - move all visible points
                else if (moveAllMode) {{
                    switch (e.key) {{
                        case 'ArrowLeft':
                            moveAllPoints(-moveSpeed, 0);
                            e.preventDefault();
                            break;
                        case 'ArrowRight':
                            moveAllPoints(moveSpeed, 0);
                            e.preventDefault();
                            break;
                        case 'ArrowUp':
                            moveAllPoints(0, -moveSpeed);
                            e.preventDefault();
                            break;
                        case 'ArrowDown':
                            moveAllPoints(0, moveSpeed);
                            e.preventDefault();
                            break;
                    }}
                }}
                // Point(s) selected - move selected points
                else if (selectedMapFirstIndices.size > 0) {{
                    switch (e.key) {{
                        case 'ArrowLeft':
                            moveSelectedPoint(-moveSpeed, 0);
                            e.preventDefault();
                            break;
                        case 'ArrowRight':
                            moveSelectedPoint(moveSpeed, 0);
                            e.preventDefault();
                            break;
                        case 'ArrowUp':
                            moveSelectedPoint(0, -moveSpeed);
                            e.preventDefault();
                            break;
                        case 'ArrowDown':
                            moveSelectedPoint(0, moveSpeed);
                            e.preventDefault();
                            break;
                    }}
                }}
            }}
        }});

        // Add event listener for GPS input to show real-time parsing
        document.getElementById('gpsInput').addEventListener('input', updateParsedDisplay);

        // Initial UI update
        updateUI();
    </script>
</body>
</html>
"""
    return html


class GCPCaptureHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for GCP capture interface."""

    session: GCPCaptureWebSession = None
    output_path: str = None
    temp_dir: str = None
    frame_filename: str = None
    has_live_camera: bool = False  # True if using live camera, False if using static frame

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            # Serve main HTML
            html = generate_capture_html(self.session, f'/{self.frame_filename}')
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

        elif parsed.path == f'/{self.frame_filename}':
            # Serve the frame image
            frame_path = os.path.join(self.temp_dir, self.frame_filename)
            with open(frame_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif parsed.path == '/api/save':
            # Generate and serve YAML
            yaml_content = self.session.generate_yaml()

            # Save to file if output path specified
            if self.output_path:
                with open(self.output_path, 'w') as f:
                    f.write(yaml_content)
                print(f"\nSaved configuration to: {self.output_path}")
                print(f"  GCPs: {len(self.session.gcps)}")

            # Serve as download
            filename = os.path.basename(self.output_path) if self.output_path else f"gcps_{self.session.camera_name}.yaml"
            self.send_response(200)
            self.send_header('Content-type', 'application/x-yaml')
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(yaml_content.encode())

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        if parsed.path == '/api/add_gcp':
            data = json.loads(post_data)
            self.session.add_gcp(
                u=data['u'],
                v=data['v'],
                lat=data['lat'],
                lon=data['lon'],
                description=data.get('description', ''),
                accuracy=data.get('accuracy', 'medium'),
                utm_easting=data.get('utm_easting'),
                utm_northing=data.get('utm_northing'),
                utm_crs=data.get('utm_crs')
            )
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/delete_gcp':
            data = json.loads(post_data)
            self.session.remove_gcp(data['index'])
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/clear_gcps':
            self.session.gcps.clear()
            self.session.current_homography = None
            self.session.last_reproj_errors = []
            self.session.inlier_mask = None
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/update_gcp_position':
            data = json.loads(post_data)
            success = self.session.update_gcp_position(
                index=data['index'],
                u=data['u'],
                v=data['v']
            )
            self.send_json_response({
                'success': success,
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/remove_outliers':
            result = self.session.remove_outliers()
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography(),
                'removed_count': result['removed_count'],
                'removed_indices': result['removed_indices'],
                'removed_descriptions': result.get('removed_descriptions', []),
                'remaining_gcps': result['remaining_gcps']
            })

        elif parsed.path == '/api/predict_error':
            data = json.loads(post_data)
            result = self.session.predict_new_gcp_error(
                u=data['u'],
                v=data['v'],
                lat=data['lat'],
                lon=data['lon'],
                utm_easting=data.get('utm_easting'),
                utm_northing=data.get('utm_northing')
            )
            self.send_json_response(result)

        elif parsed.path == '/api/load_yaml':
            data = json.loads(post_data)
            # Store current PTZ before loading
            current_ptz = self.session.ptz_status.copy() if self.session.ptz_status else None
            result = self.session.load_from_yaml(data['yaml_content'])
            self.send_json_response({
                'success': True,
                'gcps_loaded': result['gcps_loaded'],
                'warnings': result['warnings'],
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography(),
                'loaded_ptz': result.get('loaded_ptz'),
                'loaded_camera_name': result.get('loaded_camera_name'),
                'current_ptz': current_ptz,
                'has_live_camera': self.has_live_camera,
                'coordinate_system': result.get('coordinate_system')
            })

        elif parsed.path == '/api/move_camera':
            data = json.loads(post_data)
            ptz_position = data.get('ptz_position', {})
            wait_time = data.get('wait_time', 3.0)
            result = self.session.move_camera_to_ptz(ptz_position, wait_time)
            self.send_json_response({
                'success': result['success'],
                'message': result['message'],
                'ptz_status': self.session.ptz_status
            })

        elif parsed.path == '/api/convert_kml_to_gcps':
            # Convert KML points to GCPs without saving to file
            try:
                data = json.loads(post_data)
                projected_points_data = data.get('projected_points', [])

                if not projected_points_data:
                    self.send_json_response({
                        'success': False,
                        'error': 'No projected points provided for conversion'
                    })
                    return

                # Generate GCPs from projected points
                self.session.gcps = self.session.generate_gcps_from_map_first(projected_points_data)

                if not self.session.gcps:
                    self.send_json_response({
                        'success': False,
                        'error': 'No visible GCPs to convert. All points may have been discarded.'
                    })
                    return

                # Update homography with new GCPs
                homography_result = self.session.update_homography()

                self.send_json_response({
                    'success': True,
                    'gcps_converted': len(self.session.gcps),
                    'gcps': self.session.gcps,
                    'distribution': self.session.calculate_distribution(),
                    'homography': homography_result
                })
            except json.JSONDecodeError as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid JSON data: {str(e)}'
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Conversion failed: {str(e)}'
                })

        elif parsed.path == '/api/export_map_first':
            # Export map-first mode GCPs
            try:
                data = json.loads(post_data)
                projected_points_data = data.get('projected_points', [])

                if not projected_points_data:
                    self.send_json_response({
                        'success': False,
                        'error': 'No projected points provided for export'
                    })
                    return

                # Generate GCPs from projected points
                self.session.gcps = self.session.generate_gcps_from_map_first(projected_points_data)

                if not self.session.gcps:
                    self.send_json_response({
                        'success': False,
                        'error': 'No visible GCPs to export. All points may have been discarded.'
                    })
                    return

                # Generate YAML content
                yaml_content = self.session.generate_yaml()

                # Save to file if output path specified
                if self.output_path:
                    try:
                        with open(self.output_path, 'w') as f:
                            f.write(yaml_content)
                        print(f"\nSaved map-first configuration to: {self.output_path}")
                        print(f"  GCPs: {len(self.session.gcps)}")
                        print(f"  Source: {self.session.kml_file_name}")
                    except IOError as e:
                        self.send_json_response({
                            'success': False,
                            'error': f'Failed to save file: {str(e)}'
                        })
                        return

                self.send_json_response({
                    'success': True,
                    'gcps_saved': len(self.session.gcps),
                    'yaml_content': yaml_content,
                    'gcps': self.session.gcps,
                    'distribution': self.session.calculate_distribution()
                })
            except json.JSONDecodeError as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid JSON data: {str(e)}'
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Export failed: {str(e)}'
                })

        elif parsed.path == '/api/apply_calibration':
            # Apply calibration from drift analysis
            try:
                data = json.loads(post_data)
                estimated_height = data.get('estimated_height')
                drift_analysis = data.get('drift_analysis', {})

                if estimated_height is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'No estimated height provided'
                    })
                    return

                # Log the calibration
                print("\n" + "=" * 60)
                print("CALIBRATION APPLIED FROM DRIFT ANALYSIS")
                print("=" * 60)
                print(f"Camera: {self.session.camera_name}")
                print(f"Previous height: {self.session.height_verification.get('configured_height', 'unknown')}m")
                print(f"New estimated height: {estimated_height:.2f}m")
                if drift_analysis:
                    print(f"Pattern detected: {drift_analysis.get('pattern', 'unknown')}")
                    print(f"Confidence: {drift_analysis.get('confidence', 'unknown')}")
                    if drift_analysis.get('stats'):
                        stats = drift_analysis['stats']
                        print(f"Points analyzed: {stats.get('num_points', 0)}")
                        print(f"Mean drift: ({stats.get('mean_dx', 0):.1f}, {stats.get('mean_dy', 0):.1f}) px")
                print("=" * 60)

                # Save calibration to a file for reference
                calibration_file = f"calibration_{self.session.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                calibration_data = {
                    'camera_name': self.session.camera_name,
                    'timestamp': datetime.now().isoformat(),
                    'previous_height': self.session.height_verification.get('configured_height') if self.session.height_verification else None,
                    'estimated_height': float(estimated_height),
                    'drift_analysis': {
                        'pattern': drift_analysis.get('pattern'),
                        'confidence': drift_analysis.get('confidence'),
                        'stats': drift_analysis.get('stats'),
                        'suggestions': drift_analysis.get('suggestions', [])
                    }
                }

                try:
                    with open(calibration_file, 'w') as f:
                        yaml.dump(calibration_data, f, default_flow_style=False)
                    print(f"Calibration saved to: {calibration_file}")
                except IOError as e:
                    print(f"Warning: Could not save calibration file: {e}")

                # Update the session's height verification
                if self.session.height_verification:
                    self.session.height_verification['previous_height'] = self.session.height_verification.get('configured_height')
                    self.session.height_verification['configured_height'] = estimated_height
                    self.session.height_verification['calibrated'] = True
                    self.session.height_verification['calibration_timestamp'] = datetime.now().isoformat()

                self.send_json_response({
                    'success': True,
                    'message': f'Calibration applied. New height: {estimated_height:.2f}m',
                    'calibration_file': calibration_file,
                    'estimated_height': estimated_height
                })

            except json.JSONDecodeError as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid JSON data: {str(e)}'
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Calibration failed: {str(e)}'
                })

        else:
            self.send_error(404)

    def send_json_response(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)


def start_capture_server(
    session: GCPCaptureWebSession,
    output_path: str = None,
    port: int = 8765,
    auto_open: bool = True,
    has_live_camera: bool = False
):
    """Start the web server for GCP capture."""

    # Create temp directory for frame
    temp_dir = tempfile.mkdtemp(prefix='gcp_capture_')
    frame_filename = 'frame.jpg'
    frame_path = os.path.join(temp_dir, frame_filename)
    cv2.imwrite(frame_path, session.frame)

    # Configure handler
    GCPCaptureHandler.session = session
    GCPCaptureHandler.output_path = output_path
    GCPCaptureHandler.temp_dir = temp_dir
    GCPCaptureHandler.frame_filename = frame_filename
    GCPCaptureHandler.has_live_camera = has_live_camera

    # Find available port
    port = find_available_port(start_port=port, max_attempts=10)
    server = socketserver.TCPServer(("", port), GCPCaptureHandler)

    url = f"http://localhost:{port}"
    print(f"\nGCP Capture server running at: {url}")
    print("Press Ctrl+C to stop\n")

    if auto_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
        # Cleanup temp files
        try:
            os.remove(frame_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


def grab_frame_from_camera(camera_name: str, wait_time: float = 2.0) -> tuple:
    """
    Grab a frame from a camera and get PTZ status.

    Returns:
        Tuple of (frame, ptz_status)
    """
    if not CAMERA_CONFIG_AVAILABLE:
        raise RuntimeError(
            "Camera config not available. Set CAMERA_USERNAME and CAMERA_PASSWORD "
            "environment variables, or use --frame to load an existing image."
        )

    cam_info = get_camera_by_name(camera_name)
    if not cam_info:
        available = [c['name'] for c in CAMERAS]
        raise ValueError(
            f"Camera '{camera_name}' not found. Available: {', '.join(available)}"
        )

    rtsp_url = get_rtsp_url(camera_name)

    print(f"Connecting to camera '{camera_name}'...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to connect to camera: {rtsp_url}")

    # Get PTZ status
    ptz_status = None
    if INTRINSICS_AVAILABLE:
        try:
            ptz_status = get_ptz_status(cam_info['ip'], USERNAME, PASSWORD)
            print(f"PTZ: pan={ptz_status['pan']:.1f}, tilt={ptz_status['tilt']:.1f}, zoom={ptz_status['zoom']:.1f}x")
        except Exception as e:
            print(f"Warning: Could not get PTZ status: {e}")

    # Grab frame
    print("Grabbing frame...")
    import time
    time.sleep(wait_time)  # Wait for camera to stabilize

    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    cap.release()

    if frame is None:
        raise RuntimeError("Failed to grab frame from camera")

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")
    return frame, ptz_status


def main():
    parser = argparse.ArgumentParser(
        description='Web-based GCP Capture Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'camera',
        nargs='?',
        type=str,
        help='Camera name (e.g., Valte, Setram)'
    )
    parser.add_argument(
        '--frame', '-f',
        type=str,
        help='Path to an existing frame image (skips camera connection)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output YAML file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (filename auto-generated)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not automatically open browser'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    parser.add_argument(
        '--map-first',
        type=str,
        metavar='KML_FILE',
        help='Load GPS points from KML file and project onto camera image (GPS-to-image workflow)'
    )

    args = parser.parse_args()

    if args.map_first:
        # Validate KML file exists
        if not os.path.exists(args.map_first):
            print(f"Error: KML file not found: {args.map_first}")
            sys.exit(1)

        # Camera name is required in map-first mode
        if not args.camera:
            print("Error: Camera name is required when using --map-first mode")
            parser.print_help()
            sys.exit(1)

    if args.list_cameras:
        if CAMERA_CONFIG_AVAILABLE:
            print("Available cameras:")
            for cam in CAMERAS:
                print(f"  - {cam['name']} ({cam['ip']})")
        else:
            print("Camera config not available.")
        sys.exit(0)

    # Get frame
    frame = None
    ptz_status = None
    camera_name = args.camera or "Unknown"
    has_live_camera = False

    if args.frame:
        # Load from file
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"Error: Could not load image from {args.frame}")
            sys.exit(1)
        print(f"Loaded frame from {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
        camera_name = Path(args.frame).stem

    elif args.camera:
        # Grab from camera
        try:
            frame, ptz_status = grab_frame_from_camera(args.camera)
            camera_name = args.camera
            has_live_camera = True
        except (RuntimeError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        print("\nError: Either camera name or --frame must be specified.")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None and args.output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(Path(args.output_dir) / f"gcps_{camera_name}_{timestamp}.yaml")
    elif output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"gcps_{camera_name}_{timestamp}.yaml"

    # Handle map-first mode
    map_first_mode = False
    kml_points = None
    projected_points = None
    kml_file_name = None
    height_verification = None

    if args.map_first:
        map_first_mode = True
        kml_file_name = Path(args.map_first).name

        # Parse KML points
        try:
            kml_points = parse_kml_points(args.map_first)
        except Exception as e:
            print(f"Error: Failed to parse KML file: {e}")
            sys.exit(1)

        if not kml_points:
            print(f"Error: No valid GPS points found in KML file: {args.map_first}")
            sys.exit(1)

        print(f"Loaded {len(kml_points)} points from KML file")

        # Get camera parameters for projection
        if not CAMERA_CONFIG_AVAILABLE:
            print("Error: Camera config not available for map-first mode")
            print("  Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.")
            sys.exit(1)

        # Get actual frame dimensions for accurate projection
        frame_height, frame_width = frame.shape[:2]

        try:
            camera_params = get_camera_params_for_projection(
                camera_name,
                image_width=frame_width,
                image_height=frame_height
            )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Project GPS points to image first (needed by both height verification and display)
        try:
            projected_points = project_gps_to_image(kml_points, camera_params, camera_name=camera_name)
        except ValueError as e:
            print(f"Error: Failed to project GPS points: {e}")
            sys.exit(1)

        # Verify camera height if possible (using already projected points)
        if GEOMETRY_AVAILABLE:
            try:
                from poc_homography.camera_geometry import CameraGeometry
                # Initialize CameraGeometry with image dimensions
                geo = CameraGeometry(
                    w=camera_params['image_width'],
                    h=camera_params['image_height']
                )
                # Camera position in world coordinates (X=0, Y=0 at camera location, Z=height)
                w_pos = np.array([0.0, 0.0, camera_params['height_m']])
                # Set up homography with camera parameters
                geo.set_camera_parameters(
                    K=camera_params['K'],
                    w_pos=w_pos,
                    pan_deg=camera_params['pan_deg'],
                    tilt_deg=camera_params['tilt_deg'],
                    map_width=640,
                    map_height=640
                )
                # Pass already projected points to avoid duplicate projection
                height_verification = verify_camera_height_with_projected(
                    projected_points, kml_points, camera_params, geo
                )
                if height_verification.get('warning'):
                    print(f"Height verification warning: {height_verification['warning']}")
                # Add camera GPS and parameters for drift-based GPS estimation
                height_verification['camera_lat'] = camera_params['camera_lat']
                height_verification['camera_lon'] = camera_params['camera_lon']
                height_verification['tilt_deg'] = camera_params['tilt_deg']
                height_verification['pan_deg'] = camera_params['pan_deg']
                height_verification['image_width'] = camera_params['image_width']
                height_verification['image_height'] = camera_params['image_height']
            except Exception as e:
                print(f"Warning: Height verification failed: {e}")
                height_verification = {
                    'configured_height': camera_params['height_m'],
                    'camera_lat': camera_params['camera_lat'],
                    'camera_lon': camera_params['camera_lon'],
                    'tilt_deg': camera_params['tilt_deg'],
                    'pan_deg': camera_params['pan_deg'],
                    'image_width': camera_params['image_width'],
                    'image_height': camera_params['image_height'],
                    'warning': f'Verification failed: {str(e)}'
                }

        # Count visible points
        visible_count = sum(1 for p in projected_points if p['visible'] and p['reason'] == 'visible')
        print(f"Projected {visible_count} visible points onto image")

        if visible_count == 0:
            print("Warning: No KML points are visible in the camera view.")
            print("  All points are either behind the camera or outside image bounds.")
            print("  The UI will show all points for reference, but none can be exported.")

    # Create session and start server
    session = GCPCaptureWebSession(
        frame=frame,
        camera_name=camera_name,
        ptz_status=ptz_status,
        map_first_mode=map_first_mode,
        kml_points=kml_points or [],
        projected_points=projected_points or [],
        kml_file_name=kml_file_name,
        height_verification=height_verification
    )

    start_capture_server(
        session=session,
        output_path=output_path,
        port=args.port,
        auto_open=not args.no_open,
        has_live_camera=has_live_camera
    )


if __name__ == '__main__':
    main()
