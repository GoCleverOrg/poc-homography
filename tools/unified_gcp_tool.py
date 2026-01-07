#!/usr/bin/env python3
"""
Unified GCP Tool - Combines KML extraction and GCP capture in a two-tab interface.

This tool provides a unified workflow for:
1. Extracting KML points from georeferenced images (Tab 1)
2. Capturing GCPs with map-first mode (Tab 2)

The two tabs share a unified session with automatic point synchronization.

Usage:
    python tools/unified_gcp_tool.py <image> --camera <camera_name> [--port <port>] [--kml <kml_file>]

Examples:
    # Basic usage
    python tools/unified_gcp_tool.py frame.jpg --camera Valte

    # Custom port
    python tools/unified_gcp_tool.py frame.jpg --camera Valte --port 8080

    # Pre-load KML points on startup
    python tools/unified_gcp_tool.py frame.jpg --camera Valte --kml exported_points.kml
"""

import argparse
import base64
import http.server
import json
import os
import re
import socketserver
import sys
import tempfile
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
from pyproj import Transformer

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.server_utils import find_available_port
from poc_homography.camera_config import get_camera_by_name, get_camera_configs
from poc_homography.geotiff_utils import apply_geotransform

# Import for map-first mode projection
try:
    from poc_homography.camera_geometry import CameraGeometry
    from poc_homography.gps_distance_calculator import dms_to_dd
    from poc_homography.coordinate_converter import UTMConverter
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

# Import GCP calibrator for reprojection error minimization
try:
    from poc_homography.gcp_calibrator import GCPCalibrator, CalibrationResult
    CALIBRATOR_AVAILABLE = True
except ImportError:
    CALIBRATOR_AVAILABLE = False

# Import intrinsics utility
try:
    from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics
    INTRINSICS_AVAILABLE = True
except ImportError:
    INTRINSICS_AVAILABLE = False

# SAM3 detection prompts for ground markings
# Testing showed "ground markings" detects both road lines AND parking spot lines with
# best overall results: 12.12% coverage, 0.699 confidence on sample cartography images.
# See tools/test_sam3_prompts.py for the testing script.
DEFAULT_SAM3_PROMPT_CARTOGRAPHY = "ground markings"

# For camera frames, "road marking" works better than "ground markings" which often
# detects 0 features on live camera imagery. The singular form appears to be more
# specific and reliable for real-world camera footage.
DEFAULT_SAM3_PROMPT_CAMERA = "road marking"

# Valid preprocessing options (CLAHE is default - provides ~3% confidence boost)
VALID_PREPROCESSING_TYPES = ('none', 'clahe')

def apply_preprocessing(frame, preprocessing_type):
    """
    Apply preprocessing to frame for SAM3 detection.

    Args:
        frame: BGR image as numpy array
        preprocessing_type: One of 'none', 'clahe', or None

    Returns:
        Preprocessed BGR image as numpy array
    """
    if preprocessing_type == 'none' or preprocessing_type is None:
        return frame
    elif preprocessing_type == 'clahe':
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Testing showed CLAHE improves SAM3 confidence by ~3% on road marking detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        # Unknown type, return original frame
        return frame


# Import camera defaults
try:
    from poc_homography.camera_config import (
        DEFAULT_SENSOR_WIDTH_MM,
        DEFAULT_BASE_FOCAL_LENGTH_MM,
        get_rtsp_url,
        USERNAME,
        PASSWORD
    )
except ImportError:
    DEFAULT_SENSOR_WIDTH_MM = 6.78
    DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9
    USERNAME = None
    PASSWORD = None



# Helper functions for camera footprint calculation with partial projection handling

def _compute_ray_ground_direction(u: float, v: float, K: np.ndarray, R: np.ndarray) -> dict:
    """
    Compute the direction a camera ray points on the ground plane (XY).

    When homography projection fails (w <= 0), we need to compute the ray
    direction directly from camera geometry to get the correct ground direction.

    Args:
        u, v: Image pixel coordinates
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)

    Returns:
        Dictionary with 'east' and 'north' components (unit vector on ground plane)
    """
    # Convert pixel to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    pixel = np.array([u, v, 1.0])
    ray_cam = K_inv @ pixel  # Ray direction in camera frame

    # Transform to world frame (R is world-to-camera, so R.T is camera-to-world)
    ray_world = R.T @ ray_cam

    # Extract XY components (ground plane direction)
    east = ray_world[0]
    north = ray_world[1]

    # Normalize to unit vector
    length = math.sqrt(east**2 + north**2)
    if length > 1e-10:
        east /= length
        north /= length

    return {'east': east, 'north': north}


def _classify_corner_projection(pt_world: np.ndarray, height_m: float) -> dict:
    """
    Classify a corner projection based on homogeneous coordinates.

    Args:
        pt_world: Homogeneous world coordinates [X, Y, w] from inverse homography
        height_m: Camera height in meters

    Returns:
        Dictionary with:
        - status: 'valid' or 'clampable'
        - needs_clamping: Boolean indicating if distance clamping needed
        - east_offset: X coordinate in meters (unnormalized if clampable)
        - north_offset: Y coordinate in meters (unnormalized if clampable)
        - distance: Distance from camera (meters, inf if clampable)
    """
    w = pt_world[2, 0]
    max_distance = height_m * 20.0  # MAX_DISTANCE_HEIGHT_RATIO from CameraGeometry

    # Check for w near zero or negative (near/beyond horizon)
    # When w <= 0, the point projects at or beyond the horizon
    # We use the unnormalized direction and clamp to max distance
    if abs(w) < 1e-6 or w < 0:
        # Use unnormalized coordinates for direction
        # If w < 0, negate to get forward direction (point is "behind" camera plane)
        sign = 1.0 if w >= 0 else -1.0
        return {
            'status': 'clampable',
            'needs_clamping': True,
            'east_offset': pt_world[0, 0] * sign,  # Unnormalized, corrected direction
            'north_offset': pt_world[1, 0] * sign,
            'distance': float('inf')
        }
    
    # Valid w - normalize coordinates
    east_offset = pt_world[0, 0] / w
    north_offset = pt_world[1, 0] / w
    distance = math.sqrt(east_offset**2 + north_offset**2)
    
    # Check if distance exceeds maximum
    if distance > max_distance:
        return {
            'status': 'clampable',
            'needs_clamping': True,
            'east_offset': east_offset,
            'north_offset': north_offset,
            'distance': distance
        }
    
    # Valid projection within reasonable distance
    return {
        'status': 'valid',
        'needs_clamping': False,
        'east_offset': east_offset,
        'north_offset': north_offset,
        'distance': distance
    }


def _clamp_to_max_distance(east_offset: float, north_offset: float, max_distance: float) -> dict:
    """
    Clamp world coordinates to maximum distance while preserving direction.
    
    Args:
        east_offset: East offset in meters (may exceed max_distance)
        north_offset: North offset in meters (may exceed max_distance)
        max_distance: Maximum allowed distance in meters
        
    Returns:
        Dictionary with clamped 'east' and 'north' coordinates
    """
    current_distance = math.sqrt(east_offset**2 + north_offset**2)
    
    if current_distance == 0:
        # Degenerate case - camera directly overhead
        return {'east': 0.0, 'north': max_distance}
    
    # Scale to max_distance while preserving direction
    scale = max_distance / current_distance
    return {
        'east': east_offset * scale,
        'north': north_offset * scale
    }


def _convert_world_offset_to_latlon(east_offset: float, north_offset: float, 
                                     camera_lat: float, camera_lon: float) -> dict:
    """
    Convert world offset (meters) to lat/lon coordinates.
    
    Args:
        east_offset: East offset from camera in meters
        north_offset: North offset from camera in meters
        camera_lat: Camera latitude in decimal degrees
        camera_lon: Camera longitude in decimal degrees
        
    Returns:
        Dictionary with 'lat' and 'lon' keys
    """
    # Use approximate conversion: 1 degree lat ≈ 111320m
    lat_offset_deg = north_offset / 111320.0
    lon_offset_deg = east_offset / (111320.0 * np.cos(np.radians(camera_lat)))
    
    return {
        'lat': camera_lat + lat_offset_deg,
        'lon': camera_lon + lon_offset_deg
    }


class UnifiedSession:
    """
    Unified session combining KML extractor and GCP capture functionality.

    This class manages the shared state between the two tabs:
    - Camera configuration and frame image
    - Georeferencing parameters (origin, pixel size, CRS)
    - KML points (single source of truth)
    - Coordinate conversion methods
    """

    def __init__(self, image_path: str, camera_config: dict, geotiff_params: dict):
        """
        Initialize unified session.

        Args:
            image_path: Path to the frame image
            camera_config: Camera configuration dictionary
            geotiff_params: Georeferencing parameters with keys:
                - origin_easting: UTM easting of top-left pixel
                - origin_northing: UTM northing of top-left pixel
                - pixel_size_x: Pixel size in X direction (meters)
                - pixel_size_y: Pixel size in Y direction (meters, typically negative)
                - utm_crs: UTM coordinate reference system (e.g., "EPSG:25830")
        """
        self.image_path = Path(image_path)
        self.camera_config = camera_config
        self.camera_name = camera_config['name']
        # Normalize geotiff_params to internal format with geotransform array
        if 'geotransform' in geotiff_params:
            # New format: already has geotransform
            self.geotiff_params = geotiff_params
        else:
            # Legacy format: convert to geotransform format
            self.geotiff_params = {
                'geotransform': [
                    geotiff_params['origin_easting'],
                    geotiff_params['pixel_size_x'],
                    0,  # row_rotation (assumed 0)
                    geotiff_params['origin_northing'],
                    0,  # col_rotation (assumed 0)
                    geotiff_params['pixel_size_y']
                ],
                'utm_crs': geotiff_params['utm_crs']
            }

        # Load cartography frame (georeferenced image for Tab 1)
        self.cartography_frame = cv2.imread(str(image_path))
        if self.cartography_frame is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Legacy alias for compatibility
        self.frame = self.cartography_frame
        self.frame_height, self.frame_width = self.cartography_frame.shape[:2]

        # Camera frame (live camera capture for Tab 2)
        self.camera_frame = None
        self.projected_points = []

        # Initialize camera_params from config (available immediately for visualization)
        # These are default/static params; live capture will update pan/tilt/zoom
        self._init_camera_params_from_config()

        # SAM3 feature detection masks
        self.cartography_mask = None
        self.camera_mask = None
        self.projected_cartography_mask = None  # Cartography mask projected to camera coords
        self.projected_mask_offset = None  # CSS offset (left, top) for positioning mask overlay
        self.feature_mask_metadata = None

        # FROZEN GCP observations for calibration
        # These are captured ONCE when SAM3 detects features, then used for all calibrations
        # Format: {'point_name': {'pixel_u': float, 'pixel_v': float}, ...}
        # This prevents re-matching on every calibration which causes drift
        self.frozen_gcp_observations = None

        # KML points storage (single source of truth)
        self.points = []

        # Coordinate transformers
        self.utm_crs = geotiff_params['utm_crs']
        self.transformer_utm_to_gps = Transformer.from_crs(
            self.utm_crs, "EPSG:4326", always_xy=True
        )
        self.transformer_gps_to_utm = Transformer.from_crs(
            "EPSG:4326", self.utm_crs, always_xy=True
        )

    def pixel_to_utm(self, px: float, py: float) -> tuple:
        """
        Convert pixel coordinates to UTM using 6-parameter affine geotransform.

        Uses the GDAL GeoTransform formula:
            easting = GT[0] + px*GT[1] + py*GT[2]
            northing = GT[3] + px*GT[4] + py*GT[5]
        """
        easting, northing = apply_geotransform(px, py, self.geotiff_params['geotransform'])
        return easting, northing

    def pixel_to_latlon(self, px: float, py: float) -> tuple:
        """Convert pixel coordinates to lat/lon."""
        easting, northing = self.pixel_to_utm(px, py)
        lon, lat = self.transformer_utm_to_gps.transform(easting, northing)
        return lat, lon

    def latlon_to_utm(self, lat: float, lon: float) -> tuple:
        """Convert lat/lon to UTM."""
        easting, northing = self.transformer_gps_to_utm.transform(lon, lat)
        return easting, northing

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple:
        """
        Convert lat/lon to pixel coordinates using inverse affine transform.

        For north-up images (GT[2]=0, GT[4]=0), this simplifies to:
            px = (easting - GT[0]) / GT[1]
            py = (northing - GT[3]) / GT[5]

        For rotated images, we need to solve the 2x2 system.
        """
        easting, northing = self.latlon_to_utm(lat, lon)
        gt = self.geotiff_params['geotransform']

        # Inverse affine transform
        det = gt[1] * gt[5] - gt[2] * gt[4]
        if abs(det) < 1e-10:
            raise ValueError("Geotransform matrix is singular (cannot invert)")

        de = easting - gt[0]
        dn = northing - gt[3]

        px = (gt[5] * de - gt[2] * dn) / det
        py = (-gt[4] * de + gt[1] * dn) / det

        return px, py

    def _init_camera_params_from_config(self):
        """Initialize camera_params from camera_config for immediate visualization."""
        from poc_homography.gps_distance_calculator import dms_to_dd

        # Parse camera lat/lon from DMS format
        camera_lat = dms_to_dd(self.camera_config['lat'])
        camera_lon = dms_to_dd(self.camera_config['lon'])

        # Get height from config
        height_m = self.camera_config.get('height_m', 5.0)

        # Default pan/tilt/zoom (will be updated when camera frame is captured)
        # Use values that produce a visible footprint on the cartography
        # - Low zoom for wide FOV
        # - Moderate tilt to look at a reasonable distance
        default_pan = self.camera_config.get('pan_offset_deg', 0.0)
        default_tilt = 25.0  # Look further out (ground dist ≈ height / tan(25°) ≈ 10m)
        default_zoom = 1.0   # Widest FOV (~60° horizontal)

        # Compute intrinsic matrix K
        image_width = 1920
        image_height = 1080
        K = CameraGeometry.get_intrinsics(
            zoom_factor=default_zoom,
            W_px=image_width,
            H_px=image_height,
            sensor_width_mm=self.camera_config.get('sensor_width_mm', 6.78)
        )

        self.camera_params = {
            'camera_lat': camera_lat,
            'camera_lon': camera_lon,
            'height_m': height_m,
            'pan_deg': default_pan,
            'tilt_deg': default_tilt,
            'zoom': default_zoom,
            'image_width': image_width,
            'image_height': image_height,
            'K': K
        }

    def add_point(self, px: float, py: float, name: str, category: str):
        """Add a KML point."""
        # Calculate UTM first, then derive lat/lon (avoid redundant pixel_to_utm call)
        easting, northing = self.pixel_to_utm(px, py)
        lon, lat = self.transformer_utm_to_gps.transform(easting, northing)
        self.points.append({
            "name": name,
            "category": category,
            "pixel_x": px,
            "pixel_y": py,
            "easting": easting,
            "northing": northing,
            "lat": lat,
            "lon": lon
        })
        return self.points[-1]

    def delete_point(self, index: int):
        """Delete a point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            return True
        return False

    def export_kml(self) -> str:
        """Export points to KML format string."""
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Reference Points</name>
    <description>Extracted from {}</description>

    <Schema name="GCPData" id="GCPData">
        <SimpleField type="string" name="category"/>
        <SimpleField type="float" name="pixel_x"/>
        <SimpleField type="float" name="pixel_y"/>
        <SimpleField type="float" name="utm_easting"/>
        <SimpleField type="float" name="utm_northing"/>
        <SimpleField type="string" name="utm_crs"/>
    </Schema>

    <Style id="zebra">
        <IconStyle><color>ff0000ff</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="arrow">
        <IconStyle><color>ff00ff00</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="parking">
        <IconStyle><color>ffff0000</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="other">
        <IconStyle><color>ff00ffff</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon>
        </IconStyle>
    </Style>
'''.format(self.image_path.name)

        for pt in self.points:
            style = pt["category"].lower().replace(" ", "_")
            if style not in ["zebra", "arrow", "parking"]:
                style = "other"

            kml_content += '''
    <Placemark>
        <name>{name}</name>
        <description>Category: {category}
Pixel: ({pixel_x:.1f}, {pixel_y:.1f})
UTM: E {easting:.2f}, N {northing:.2f}
CRS: {crs}</description>
        <styleUrl>#{style}</styleUrl>
        <ExtendedData>
            <SchemaData schemaUrl="#GCPData">
                <SimpleData name="category">{category}</SimpleData>
                <SimpleData name="pixel_x">{pixel_x:.2f}</SimpleData>
                <SimpleData name="pixel_y">{pixel_y:.2f}</SimpleData>
                <SimpleData name="utm_easting">{easting:.4f}</SimpleData>
                <SimpleData name="utm_northing">{northing:.4f}</SimpleData>
                <SimpleData name="utm_crs">{crs}</SimpleData>
            </SchemaData>
        </ExtendedData>
        <Point>
            <coordinates>{lon:.8f},{lat:.8f},0</coordinates>
        </Point>
    </Placemark>
'''.format(
                name=pt["name"],
                category=pt["category"],
                pixel_x=pt["pixel_x"],
                pixel_y=pt["pixel_y"],
                easting=pt["easting"],
                northing=pt["northing"],
                crs=self.utm_crs,
                style=style,
                lon=pt["lon"],
                lat=pt["lat"]
            )

        kml_content += '''
</Document>
</kml>'''
        return kml_content

    def parse_kml(self, kml_text: str) -> list:
        """Parse KML file and extract points with pixel coordinates."""
        # Remove namespace for easier parsing
        kml_text = re.sub(r'\sxmlns="[^"]+"', '', kml_text, count=1)

        root = ET.fromstring(kml_text)
        points = []

        for placemark in root.iter('Placemark'):
            name_elem = placemark.find('name')
            name = name_elem.text if name_elem is not None else f"Point_{len(points)+1}"

            # Try to extract category from styleUrl or description
            style_elem = placemark.find('styleUrl')
            desc_elem = placemark.find('description')

            category = 'other'
            if style_elem is not None:
                style = style_elem.text.replace('#', '')
                if style in ['zebra', 'arrow', 'parking']:
                    category = style

            # Also check description for category
            if desc_elem is not None and desc_elem.text:
                desc_lower = desc_elem.text.lower()
                if 'category: zebra' in desc_lower:
                    category = 'zebra'
                elif 'category: arrow' in desc_lower:
                    category = 'arrow'
                elif 'category: parking' in desc_lower:
                    category = 'parking'

            # Get coordinates
            coords_elem = placemark.find('.//coordinates')
            if coords_elem is not None and coords_elem.text:
                coords_text = coords_elem.text.strip()
                parts = coords_text.split(',')
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    px, py = self.latlon_to_pixel(lat, lon)
                    points.append({
                        'name': name,
                        'category': category,
                        'px': px,
                        'py': py,
                        'lat': lat,
                        'lon': lon
                    })

        return points

    def project_points_to_image(self) -> List[Dict]:
        """
        Project KML points to camera frame using actual camera projection.

        Returns list of dictionaries with projected pixel coordinates.
        """
        if not GEOMETRY_AVAILABLE or self.camera_params is None:
            return []

        # Import the projection function
        try:
            from tools.capture_gcps_web import project_gps_to_image
        except ImportError:
            print("Warning: Could not import project_gps_to_image")
            return []

        # Convert points to format expected by project_gps_to_image
        gps_points = []
        for pt in self.points:
            gps_points.append({
                'name': pt['name'],
                'latitude': pt['lat'],
                'longitude': pt['lon'],
                'utm_easting': pt['easting'],
                'utm_northing': pt['northing'],
                'utm_crs': self.utm_crs
            })

        # Project points using camera geometry
        projected = project_gps_to_image(
            gps_points,
            self.camera_params,
            camera_name=self.camera_name
        )

        # Add category information
        for i, proj_pt in enumerate(projected):
            if i < len(self.points):
                proj_pt['category'] = self.points[i]['category']

        return projected

    def calculate_camera_footprint(self) -> Optional[List[Dict]]:
        """
        Calculate the camera's visible ground footprint by projecting image frame corners.

        Projects the four corners of the camera frame (0,0), (w,0), (w,h), (0,h) through
        the inverse homography to get world coordinates, then converts to lat/lon.
        
        Handles partial projections when some corners are near/beyond horizon or exceed
        reasonable distance limits. Uses per-corner validation instead of all-or-nothing.

        Returns:
            List of 4 dicts (one per corner) with keys:
            - 'lat': Latitude in decimal degrees
            - 'lon': Longitude in decimal degrees
            - 'valid': Boolean indicating if corner projected within max distance
            - 'clamped': Boolean indicating if coordinates were clamped to max distance

            Returns None if camera_params are not available.
            Corners near/beyond horizon are clamped to max_distance (height_m * 20).
        """
        if not GEOMETRY_AVAILABLE or self.camera_params is None:
            return None

        try:
            # Get camera geometry from camera_params
            K = self.camera_params.get('K')
            if K is None:
                return None

            # Convert K to numpy array if it's a list
            if isinstance(K, list):
                K = np.array(K)

            image_width = self.camera_params.get('image_width', 1920)
            image_height = self.camera_params.get('image_height', 1080)
            camera_lat = self.camera_params.get('camera_lat')
            camera_lon = self.camera_params.get('camera_lon')
            height_m = self.camera_params.get('height_m', 5.0)
            pan_deg = self.camera_params.get('pan_deg', 0.0)
            tilt_deg = self.camera_params.get('tilt_deg', 0.0)

            if camera_lat is None or camera_lon is None:
                return None

            # Create CameraGeometry instance and compute homography
            geo = CameraGeometry(image_width, image_height)

            # Use set_camera_parameters() to properly compute homography
            # Camera is at origin in local coordinate frame, height is Z component
            w_pos = np.array([0.0, 0.0, height_m])
            geo.set_camera_parameters(
                K=K,
                w_pos=w_pos,
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
                map_width=640,  # Map dimensions for visualization (not used for footprint calc)
                map_height=640
            )

            # Define the four corners of the camera frame
            corners = [
                (0, 0),                          # top-left
                (image_width - 1, 0),            # top-right
                (image_width - 1, image_height - 1),  # bottom-right
                (0, image_height - 1)            # bottom-left
            ]

            max_distance = height_m * 20.0  # MAX_DISTANCE_HEIGHT_RATIO from CameraGeometry
            footprint = []

            # Get rotation matrix for ray direction computation (needed for invalid corners)
            R = geo._get_rotation_matrix()

            for u, v in corners:
                # Project pixel to world coordinates (relative to camera position)
                pt_img = np.array([[u], [v], [1.0]])
                pt_world = geo.H_inv @ pt_img

                w = pt_world[2, 0]

                # Check if homography projection is valid
                if w > 1e-6:
                    # Valid projection - use homography result
                    east_offset = pt_world[0, 0] / w
                    north_offset = pt_world[1, 0] / w
                    distance = math.sqrt(east_offset**2 + north_offset**2)

                    if distance <= max_distance:
                        # Valid corner within reasonable distance
                        clamped = False
                    else:
                        # Valid direction but too far - clamp to max_distance
                        clamped_coords = _clamp_to_max_distance(east_offset, north_offset, max_distance)
                        east_offset = clamped_coords['east']
                        north_offset = clamped_coords['north']
                        clamped = True
                else:
                    # Invalid projection (w <= 0) - compute ray direction directly
                    # This happens when the pixel ray doesn't intersect the ground plane
                    ray_dir = _compute_ray_ground_direction(u, v, K, R)
                    # Scale unit direction to max_distance
                    east_offset = ray_dir['east'] * max_distance
                    north_offset = ray_dir['north'] * max_distance
                    clamped = True

                # Convert world offset to lat/lon
                latlon = _convert_world_offset_to_latlon(
                    east_offset, north_offset, camera_lat, camera_lon
                )

                footprint.append({
                    'lat': float(latlon['lat']),
                    'lon': float(latlon['lon']),
                    'valid': bool(w > 1e-6),
                    'clamped': bool(clamped)
                })

            return footprint

        except Exception as e:
            print(f"Warning: Failed to calculate camera footprint: {e}")
            return None

    def project_cartography_mask_to_camera(self) -> Optional[np.ndarray]:
        """
        Project the cartography mask from Tab 1 onto the camera frame coordinates.

        This transforms the cartography mask (in cartography image pixel coordinates)
        to camera image coordinates using the coordinate transformation chain:
        1. Cartography pixels → UTM coordinates (using geotiff_params)
        2. UTM coordinates → World XY (relative to camera position)
        3. World XY → Camera image pixels (using camera homography H)

        The transformation is computed by finding the homography between cartography
        pixels and camera pixels, then applying cv2.warpPerspective to transform
        the entire mask image efficiently.

        Returns:
            Binary mask in camera image coordinates, or None if projection fails
            or required data is not available.
        """
        if not GEOMETRY_AVAILABLE:
            print("Warning: Geometry modules not available for mask projection")
            return None

        if self.cartography_mask is None:
            return None

        if self.camera_params is None or self.camera_frame is None:
            return None

        try:
            # Get camera frame dimensions
            cam_height, cam_width = self.camera_frame.shape[:2]

            # Set up UTM converter with camera position as reference
            utm_converter = UTMConverter(self.utm_crs)
            camera_lat = self.camera_params['camera_lat']
            camera_lon = self.camera_params['camera_lon']
            utm_converter.set_reference(camera_lat, camera_lon)

            # Set up CameraGeometry for projection
            geo = CameraGeometry(w=cam_width, h=cam_height)
            height_m = self.camera_params['height_m']
            pan_deg = self.camera_params['pan_deg']
            tilt_deg = self.camera_params['tilt_deg']
            K = self.camera_params['K']

            # Camera position in world coordinates (X=0, Y=0 at camera location, Z=height)
            w_pos = np.array([0.0, 0.0, height_m])

            geo.set_camera_parameters(
                K=K,
                w_pos=w_pos,
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
                map_width=640,
                map_height=640
            )

            # Load and apply distortion coefficients from camera config
            # Issue #135: Always work in distorted image space with original K matrix
            # Distortion is applied via CameraGeometry for homography-based projection
            distortion_applied = False
            if self.camera_name:
                try:
                    cam_config = get_camera_by_name(self.camera_name)
                    if cam_config:
                        k1 = cam_config.get('k1', 0.0)
                        k2 = cam_config.get('k2', 0.0)
                        p1 = cam_config.get('p1', 0.0)
                        p2 = cam_config.get('p2', 0.0)
                        k3 = cam_config.get('k3', 0.0)
                        # Only apply if non-zero coefficients exist
                        if k1 != 0.0 or k2 != 0.0 or p1 != 0.0 or p2 != 0.0 or k3 != 0.0:
                            geo.set_distortion_coefficients(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
                            distortion_applied = True
                except Exception as e:
                    print(f"Warning: Could not load distortion coefficients: {e}")

            # =================================================================
            # COMPOSE THE HOMOGRAPHY MATHEMATICALLY (not from sampled points)
            # =================================================================
            # The transformation chain is:
            #   Cartography pixels → UTM → Local XY → Camera pixels
            #
            # This can be expressed as: H_total = H_camera @ A
            # where A is the affine transformation matrix (CameraGeometry.A)
            # =================================================================

            # Get camera's UTM coordinates
            camera_easting, camera_northing = utm_converter.gps_to_utm(camera_lat, camera_lon)
            print(f"Camera UTM: E={camera_easting:.2f}, N={camera_northing:.2f}")


            # Set geotiff parameters to compute the A matrix
            # A matrix transforms reference image pixels to world ground plane coordinates
            geo.set_geotiff_params(self.geotiff_params, (camera_easting, camera_northing))

            # A matrix computed during set_geotiff_params call above
            print(f"A matrix (reference pixel to world):")
            print(f"  pixel_size: ({geo.A[0,0]}, {geo.A[1,1]})")
            print(f"  translation: ({geo.A[0,2]:.2f}, {geo.A[1,2]:.2f})")

            # Compose the total homography: H_total = H_camera @ A
            # This maps cartography pixels directly to camera pixels
            H_total = geo.H @ geo.A
            print(f"H_camera (geo.H):")
            print(geo.H)
            print(f"H_total (composed):")
            print(H_total)

            # Get mask dimensions
            carto_height, carto_width = self.cartography_mask.shape[:2]

            # Find bounding box of non-zero pixels in the mask
            non_zero_coords = cv2.findNonZero(self.cartography_mask)
            if non_zero_coords is None:
                print("Warning: Mask has no non-zero pixels")
                return None

            bbox_x, bbox_y, w_box, h_box = cv2.boundingRect(non_zero_coords)
            print(f"Mask bounding box: x={bbox_x}, y={bbox_y}, w={w_box}, h={h_box}")

            # Project the 4 corners of the mask bounding box to determine output canvas size
            carto_corners = np.array([
                [bbox_x, bbox_y],                          # top-left
                [bbox_x + w_box, bbox_y],                  # top-right
                [bbox_x + w_box, bbox_y + h_box],          # bottom-right
                [bbox_x, bbox_y + h_box]                   # bottom-left
            ], dtype=np.float32)

            # Project corners using H_total
            camera_corners = []
            for px, py in carto_corners:
                pt_h = np.array([px, py, 1.0])
                proj = H_total @ pt_h
                if proj[2] != 0:
                    u, v = proj[0] / proj[2], proj[1] / proj[2]
                else:
                    u, v = 0, 0
                camera_corners.append((u, v))

            print(f"Projected corners:")
            for i, (c, p) in enumerate(zip(carto_corners, camera_corners)):
                print(f"  {i}: carto ({c[0]:.0f}, {c[1]:.0f}) -> camera ({p[0]:.1f}, {p[1]:.1f})")

            # Calculate canvas size to fit both camera frame and projected mask
            all_x = [c[0] for c in camera_corners] + [0, cam_width]
            all_y = [c[1] for c in camera_corners] + [0, cam_height]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Clamp to reasonable bounds (10x frame size max)
            max_extent = max(cam_width, cam_height) * 10
            min_x = max(min_x, -max_extent)
            max_x = min(max_x, max_extent)
            min_y = max(min_y, -max_extent)
            max_y = min(max_y, max_extent)

            offset_x = int(-min_x) if min_x < 0 else 0
            offset_y = int(-min_y) if min_y < 0 else 0
            canvas_width = int(max_x - min_x) + 1
            canvas_height = int(max_y - min_y) + 1

            print(f"  Canvas size: {canvas_width}x{canvas_height}, offset: ({offset_x}, {offset_y})")

            # Adjust H_total to account for the canvas offset
            # We need to translate the output by (offset_x, offset_y)
            T_offset = np.array([
                [1, 0, offset_x],
                [0, 1, offset_y],
                [0, 0, 1]
            ], dtype=np.float64)

            H_final = T_offset @ H_total

            # Apply the transformation to the mask
            projected_mask_full = cv2.warpPerspective(
                self.cartography_mask,
                H_final,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,  # Use nearest neighbor for binary mask
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Debug: save full canvas mask
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, 'debug_projected_mask_full.png'), projected_mask_full)
            print(f"  Debug: Saved full canvas mask to {debug_dir}/debug_projected_mask_full.png")

            # Return the FULL projected mask (not cropped) so it can be displayed as overlay
            # The offset tells JavaScript where the camera frame sits within this larger canvas
            # A negative offset means the mask extends BEFORE the camera frame origin
            # So to position the mask, JavaScript should use: left = -offset_x, top = -offset_y
            print(f"  Full canvas non-zero pixels: {cv2.countNonZero(projected_mask_full)}")
            print(f"  Camera frame is at offset ({offset_x}, {offset_y}) within canvas")

            # Store the full mask and offset for the overlay
            self.projected_cartography_mask = projected_mask_full
            self.projected_mask_offset = (-offset_x, -offset_y)  # CSS position relative to camera frame
            print(f"Successfully projected cartography mask to camera coordinates")
            print(f"  Mask size: {canvas_width}x{canvas_height}, CSS offset: ({-offset_x}, {-offset_y})")
            return projected_mask_full

        except Exception as e:
            print(f"Warning: Failed to project cartography mask: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_calibration(self, observed_pixels: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Run GCP-based calibration to optimize camera parameters.

        Uses reprojection error minimization to find optimal camera parameter
        adjustments (pan, tilt, position) that minimize the difference between
        observed and predicted GCP pixel locations.

        Args:
            observed_pixels: Optional list of observed pixel locations. If not provided,
                           uses frozen observations or SAM3-detected centroids from camera_mask.
                           Raises ValueError if no real observations are available.
                           Format: [{'name': str, 'pixel_u': float, 'pixel_v': float}, ...]

        Returns:
            Dictionary with calibration results:
            - 'success': bool
            - 'optimized_params': list of 6 floats [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
            - 'initial_error': float (RMS error in pixels before calibration)
            - 'final_error': float (RMS error in pixels after calibration)
            - 'improvement_percent': float
            - 'num_inliers': int
            - 'num_outliers': int
            - 'inlier_ratio': float
            - 'per_gcp_errors': list of per-point errors
            - 'convergence_info': dict with optimizer details
            - 'suggested_params': dict with suggested new camera parameters

            Returns None if dependencies unavailable (calibrator, geometry, points).

        Raises:
            ValueError: If no real observations are available (no SAM3 mask, no frozen
                       observations, no explicit observations), or if fewer than 6 valid
                       GCPs with real observations are available.
        """
        if not CALIBRATOR_AVAILABLE:
            print("GCPCalibrator not available - cannot run calibration")
            return None

        if not GEOMETRY_AVAILABLE or self.camera_params is None:
            print("Camera geometry not available - cannot run calibration")
            return None

        if not self.points:
            print("No KML points loaded - cannot run calibration")
            return None

        # Get projected points if not already computed
        if not hasattr(self, 'projected_points') or not self.projected_points:
            self.projected_points = self.project_points_to_image()

        if not self.projected_points:
            print("No projected points available - cannot run calibration")
            return None

        # Determine observed pixel locations
        # PRIORITY: provided observations > frozen observations > extract from mask
        # Raises ValueError if no observation source is available
        if observed_pixels is not None:
            # Use explicitly provided observed pixels
            obs_by_name = {p['name']: p for p in observed_pixels}
            print("Using explicitly provided observed pixel locations")
        elif self.frozen_gcp_observations is not None:
            # Use FROZEN observations (captured when SAM3 first ran)
            # This is the correct approach - observations should not change with params
            obs_by_name = self.frozen_gcp_observations
            print(f"Using FROZEN observations ({len(obs_by_name)} points) - these won't drift!")
        elif self.camera_mask is not None:
            # Fallback: Extract centroids from SAM3 mask (will be frozen after this)
            print("WARNING: No frozen observations - extracting from mask and freezing now")
            obs_by_name = self._extract_mask_centroids_for_calibration()
            if obs_by_name:
                # Freeze these observations for future calibrations
                self.frozen_gcp_observations = obs_by_name
                print(f"Froze {len(obs_by_name)} observations for future calibrations")
            else:
                raise ValueError(
                    "Failed to extract GCP observations from SAM3 mask. "
                    "The mask exists but no centroids could be detected. "
                    "Try re-running SAM3 feature detection with different parameters."
                )
        else:
            raise ValueError(
                "Calibration requires real image observations. "
                "No SAM3 mask, frozen observations, or explicit observations provided. "
                "Run SAM3 feature detection or provide manual annotations before calibration."
            )

        # Build GCP list for calibrator - ONLY use visible points
        gcps = []
        skipped_not_visible = 0
        skipped_no_gps = 0
        skipped_no_pixel = 0

        for proj_pt in self.projected_points:
            # Only use points that are explicitly marked as visible
            # Points not visible in camera view should NOT be used for calibration
            if not proj_pt.get('visible', False):
                skipped_not_visible += 1
                continue

            name = proj_pt.get('name', '')
            gps_lat = proj_pt.get('latitude')
            gps_lon = proj_pt.get('longitude')

            if gps_lat is None or gps_lon is None:
                skipped_no_gps += 1
                continue

            # Get observed pixel location - ONLY use GCPs with real observations
            # obs_by_name is guaranteed to exist (we raise ValueError earlier if not)
            if name not in obs_by_name:
                # Skip GCPs without real observations - never use projected pixels
                skipped_no_pixel += 1
                continue

            obs = obs_by_name[name]
            pixel_u = obs.get('pixel_u')
            pixel_v = obs.get('pixel_v')

            if pixel_u is None or pixel_v is None:
                skipped_no_pixel += 1
                continue

            # Get UTM coordinates if available (more accurate than GPS conversion)
            utm_easting = proj_pt.get('utm_easting')
            utm_northing = proj_pt.get('utm_northing')

            gcps.append({
                'gps': {'latitude': gps_lat, 'longitude': gps_lon},
                'utm': {'easting': utm_easting, 'northing': utm_northing} if utm_easting and utm_northing else None,
                'image': {'u': pixel_u, 'v': pixel_v},
                '_name': name
            })

        # DEBUG: Print first GCP's coordinates for verification
        if gcps:
            g0 = gcps[0]
            print(f"\n--- DEBUG: First GCP coordinates ---")
            print(f"Name: {g0.get('_name')}")
            print(f"GPS: lat={g0['gps']['latitude']:.6f}, lon={g0['gps']['longitude']:.6f}")
            if g0.get('utm'):
                print(f"UTM: E={g0['utm']['easting']:.2f}, N={g0['utm']['northing']:.2f}")
            else:
                print("UTM: None")
            print(f"Image: u={g0['image']['u']:.1f}, v={g0['image']['v']:.1f}")
            print("---")

        print(f"\n--- GCP FILTERING SUMMARY ---")
        print(f"Total projected points: {len(self.projected_points)}")
        print(f"Valid GCPs with real observations: {len(gcps)}")
        print(f"Skipped: {skipped_not_visible} not visible, {skipped_no_gps} no GPS, {skipped_no_pixel} no observation")
        print(f"----------------------------\n")

        MINIMUM_GCP_COUNT = 6  # Industry standard for 6-parameter optimization
        if len(gcps) < MINIMUM_GCP_COUNT:
            raise ValueError(
                f"Insufficient GCP correspondences for calibration: "
                f"{len(gcps)} provided, minimum {MINIMUM_GCP_COUNT} required with real observations. "
                f"Add more Ground Control Points or ensure all points have SAM3 observations."
            )

        # Create CameraGeometry for calibrator
        try:
            K = self.camera_params.get('K')
            if isinstance(K, list):
                K = np.array(K)

            image_width = self.camera_params.get('image_width', 1920)
            image_height = self.camera_params.get('image_height', 1080)
            height_m = self.camera_params.get('height_m', 5.0)
            pan_deg = self.camera_params.get('pan_deg', 0.0)
            tilt_deg = self.camera_params.get('tilt_deg', 45.0)

            geo = CameraGeometry(image_width, image_height)
            geo.set_camera_parameters(
                K=K,
                w_pos=np.array([0, 0, height_m]),
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
                map_width=640,
                map_height=640
            )

            # Apply distortion coefficients to match the projection model
            # Issue #135: Always work in distorted image space with original K matrix
            # CameraGeometry applies distortion for homography-based projection
            if self.camera_name:
                try:
                    cam_config = get_camera_by_name(self.camera_name)
                    if cam_config:
                        k1 = cam_config.get('k1', 0.0)
                        k2 = cam_config.get('k2', 0.0)
                        p1 = cam_config.get('p1', 0.0)
                        p2 = cam_config.get('p2', 0.0)
                        k3 = cam_config.get('k3', 0.0)
                        if k1 != 0.0 or k2 != 0.0 or p1 != 0.0 or p2 != 0.0 or k3 != 0.0:
                            geo.set_distortion_coefficients(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
                            print(f"Calibrator using distortion: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
                except Exception as e:
                    print(f"Warning: Could not load distortion coefficients for calibrator: {e}")

            # Run calibration with dynamic bounds based on current height
            # CameraGeometry requires height in [1.0, 50.0]m
            # So ΔZ bounds must keep (height_m + ΔZ) within [1.0, 50.0]
            min_height = 1.0
            max_height = 50.0
            z_lower_bound = max(-5.0, min_height - height_m)  # Don't go below 1.0m
            z_upper_bound = min(5.0, max_height - height_m)   # Don't go above 50.0m

            calibration_bounds = {
                'pan': (-10.0, 10.0),
                'tilt': (-10.0, 10.0),
                'roll': (-5.0, 5.0),
                'X': (-5.0, 5.0),
                'Y': (-5.0, 5.0),
                'Z': (z_lower_bound, z_upper_bound)
            }

            # Get camera GPS position for coordinate reference
            camera_lat = self.camera_params.get('camera_lat')
            camera_lon = self.camera_params.get('camera_lon')

            print(f"Calibration: {len(gcps)} GCPs, camera at ({camera_lat:.6f}, {camera_lon:.6f}), height={height_m}m")
            print(f"Calibration bounds for Z: ({z_lower_bound:.2f}, {z_upper_bound:.2f})")

            # Use adaptive loss_scale based on median error to avoid all-outlier situation
            # Start with a large scale (50px) for initial calibration, can be refined later
            # This ensures the optimizer has meaningful gradients even with large initial errors
            loss_scale = 50.0  # Large enough to include most points initially

            calibrator = GCPCalibrator(
                camera_geometry=geo,
                gcps=gcps,
                loss_function='huber',
                loss_scale=loss_scale,
                reference_lat=camera_lat,  # CRITICAL: Use camera position as reference
                reference_lon=camera_lon,  # so world coordinates have camera at origin
                utm_crs=self.utm_crs        # Use same UTM CRS as projection for consistency
            )

            result = calibrator.calibrate(bounds=calibration_bounds)

            # DEBUG: Print calibration details
            print(f"\n{'='*60}")
            print("CALIBRATION DEBUG")
            print(f"{'='*60}")
            print(f"Current params: pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°, height={height_m:.2f}m")
            print(f"Initial RMS error: {result.initial_error:.2f} px")
            print(f"Final RMS error: {result.final_error:.2f} px")
            print(f"Optimized deltas: Δpan={result.optimized_params[0]:.4f}°, Δtilt={result.optimized_params[1]:.4f}°, ΔZ={result.optimized_params[5]:.4f}m")
            print(f"Convergence: {result.convergence_info.get('message', 'N/A')}")

            # Show per-GCP errors with names, sorted by error (worst first)
            if result.per_gcp_errors and len(gcps) == len(result.per_gcp_errors):
                gcp_errors = [(gcps[i].get('_name', f'GCP{i}'), result.per_gcp_errors[i]) for i in range(len(gcps))]
                gcp_errors.sort(key=lambda x: x[1], reverse=True)
                print(f"\nPer-GCP errors (worst first):")
                for name, error in gcp_errors[:10]:  # Show top 10 worst
                    status = "⚠️ BAD" if error > 50 else "⚡ HIGH" if error > 30 else "✓"
                    print(f"  {status} {name}: {error:.1f}px")
                if len(gcp_errors) > 10:
                    print(f"  ... and {len(gcp_errors) - 10} more")
            print(f"{'='*60}\n")

            # Calculate improvement
            if result.initial_error > 0:
                improvement = (result.initial_error - result.final_error) / result.initial_error * 100
            else:
                improvement = 0.0

            # Compute suggested new parameters
            new_pan = pan_deg + result.optimized_params[0]
            new_tilt = tilt_deg + result.optimized_params[1]
            new_height = height_m + result.optimized_params[5]

            print(f"Suggested new params: pan={new_pan:.2f}°, tilt={new_tilt:.2f}°, height={new_height:.2f}m")

            suggested_params = {
                'pan_deg': new_pan,
                'tilt_deg': new_tilt,
                'height_m': new_height,
                # Position adjustments (X, Y) would need coordinate conversion
            }

            return {
                'success': True,
                'optimized_params': result.optimized_params.tolist(),
                'initial_error': result.initial_error,
                'final_error': result.final_error,
                'improvement_percent': improvement,
                'num_inliers': result.num_inliers,
                'num_outliers': result.num_outliers,
                'inlier_ratio': result.inlier_ratio,
                'per_gcp_errors': result.per_gcp_errors,
                'convergence_info': result.convergence_info,
                'suggested_params': suggested_params,
                'num_gcps_used': len(gcps)
            }

        except Exception as e:
            print(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def run_pnp_calibration(self) -> Optional[Dict]:
        """
        Run PnP-based calibration to directly solve for camera pose.

        Unlike iterative homography refinement, PnP directly solves for camera
        rotation and translation given 3D-2D point correspondences.

        This is more robust when initial camera parameters are significantly off.

        Returns:
            Dictionary with calibration results including new camera parameters.
        """
        if not GEOMETRY_AVAILABLE or self.camera_params is None:
            print("Camera geometry not available - cannot run PnP calibration")
            return None

        if not self.points:
            print("No KML points loaded - cannot run PnP calibration")
            return None

        # Get observed pixel locations (from manual matches or frozen observations)
        if self.frozen_gcp_observations is None:
            print("ERROR: No frozen observations! Run SAM3 detection or manual matching first.")
            return None

        obs_by_name = self.frozen_gcp_observations
        print(f"PnP Calibration: Using {len(obs_by_name)} frozen observations")

        # Get camera intrinsics
        K = self.camera_params.get('K')
        if K is None:
            print("ERROR: No camera intrinsics matrix K")
            return None
        K = np.array(K)

        # Get distortion coefficients from camera_params (set during frame capture)
        # Issue #135: Always use distortion coefficients with cv2.solvePnP/projectPoints
        # This ensures calibration operates in distorted image space with original K
        dist_coeffs = np.zeros(5)  # Default: [k1, k2, p1, p2, k3]
        if 'dist_coeffs' in self.camera_params:
            dist_coeffs = np.array(self.camera_params['dist_coeffs'])
        elif self.camera_name:
            try:
                cam_config = get_camera_by_name(self.camera_name)
                if cam_config:
                    k1 = cam_config.get('k1', 0.0)
                    k2 = cam_config.get('k2', 0.0)
                    p1 = cam_config.get('p1', 0.0)
                    p2 = cam_config.get('p2', 0.0)
                    k3 = cam_config.get('k3', 0.0)
                    dist_coeffs = np.array([k1, k2, p1, p2, k3])
            except Exception as e:
                print(f"Warning: Could not load distortion coefficients for PnP: {e}")

        print(f"Using distortion coefficients: {dist_coeffs}")

        # Get camera position as reference for UTM conversion
        camera_lat = self.camera_params.get('camera_lat')
        camera_lon = self.camera_params.get('camera_lon')
        camera_height = self.camera_params.get('height_m', 5.0)

        if camera_lat is None or camera_lon is None:
            print("ERROR: No camera GPS position")
            return None

        # Build 3D world points and 2D image points
        world_points_3d = []
        image_points_2d = []
        point_names = []

        # Set up UTM converter if available
        utm_converter = None
        if self.utm_crs and UTMConverter:
            try:
                utm_converter = UTMConverter(self.utm_crs)
                utm_converter.set_reference(camera_lat, camera_lon)
            except Exception as e:
                print(f"Warning: Could not set up UTM converter: {e}")

        # Get projected points for UTM coordinates
        if not hasattr(self, 'projected_points') or not self.projected_points:
            self.projected_points = self.project_points_to_image()

        for proj_pt in self.projected_points:
            name = proj_pt.get('name', '')

            # Skip if no observation for this point
            if name not in obs_by_name:
                continue

            # Get observed 2D pixel location
            obs = obs_by_name[name]
            u = obs.get('pixel_u')
            v = obs.get('pixel_v')
            if u is None or v is None:
                continue

            # Get 3D world coordinates
            # Priority: UTM coordinates (more accurate)
            utm_easting = proj_pt.get('utm_easting')
            utm_northing = proj_pt.get('utm_northing')

            if utm_converter and utm_easting is not None and utm_northing is not None:
                # Convert UTM to local XY (camera at origin)
                x, y = utm_converter.utm_to_local_xy(utm_easting, utm_northing)
            else:
                # Fall back to GPS conversion
                gps_lat = proj_pt.get('latitude')
                gps_lon = proj_pt.get('longitude')
                if gps_lat is None or gps_lon is None:
                    continue

                # Simple equirectangular approximation
                R_EARTH = 6371000
                ref_lat_rad = math.radians(camera_lat)
                x = (gps_lon - camera_lon) * math.radians(1) * R_EARTH * math.cos(ref_lat_rad)
                y = (gps_lat - camera_lat) * math.radians(1) * R_EARTH

            # Z coordinate: ground level relative to camera
            # Ground is at -camera_height (camera is at Z=0 looking down)
            # But for PnP, we use camera-centric coordinates where camera is at origin
            # So ground points have Z = -height (below camera)
            z = -camera_height  # Ground plane is below camera

            world_points_3d.append([x, y, z])
            image_points_2d.append([u, v])
            point_names.append(name)

        if len(world_points_3d) < 4:
            print(f"ERROR: Need at least 4 point correspondences, have {len(world_points_3d)}")
            return None

        print(f"\nPnP Input:")
        print(f"  - {len(world_points_3d)} point correspondences")
        print(f"  - Camera height: {camera_height}m")
        print(f"  - Intrinsics K:\n{K}")
        print(f"  - Distortion: {dist_coeffs}")

        # Convert to numpy arrays
        world_points_3d = np.array(world_points_3d, dtype=np.float64)
        image_points_2d = np.array(image_points_2d, dtype=np.float64)

        # Debug: print first few points
        print(f"\nFirst 3 correspondences:")
        for i in range(min(3, len(point_names))):
            print(f"  {point_names[i]}: 3D=({world_points_3d[i][0]:.2f}, {world_points_3d[i][1]:.2f}, {world_points_3d[i][2]:.2f}) -> 2D=({image_points_2d[i][0]:.1f}, {image_points_2d[i][1]:.1f})")

        try:
            # For coplanar points (all on ground plane), standard PnP can fail
            # Try multiple solvers in order of preference

            success = False
            rvec = None
            tvec = None
            inliers = None

            # Method 1: Try IPPE first (best for coplanar points, needs 4+ points)
            if len(world_points_3d) >= 4:
                try:
                    print("Trying IPPE solver (optimized for coplanar points)...")
                    success, rvec, tvec = cv2.solvePnP(
                        world_points_3d,
                        image_points_2d,
                        K,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE
                    )
                    if success:
                        print("  IPPE solver succeeded")
                        inliers = np.arange(len(world_points_3d)).reshape(-1, 1)
                except Exception as e:
                    print(f"  IPPE solver failed: {e}")
                    success = False

            # Method 2: Try EPNP (works with 4+ points, handles some coplanar cases)
            if not success and len(world_points_3d) >= 4:
                try:
                    print("Trying EPNP solver...")
                    success, rvec, tvec = cv2.solvePnP(
                        world_points_3d,
                        image_points_2d,
                        K,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    if success:
                        print("  EPNP solver succeeded")
                        inliers = np.arange(len(world_points_3d)).reshape(-1, 1)
                except Exception as e:
                    print(f"  EPNP solver failed: {e}")
                    success = False

            # Method 3: Try RANSAC with more lenient threshold
            if not success:
                try:
                    print("Trying RANSAC with ITERATIVE solver...")
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        world_points_3d,
                        image_points_2d,
                        K,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=30.0,  # More lenient for noisy matches
                        confidence=0.99,
                        iterationsCount=2000
                    )
                    if success:
                        print("  RANSAC+ITERATIVE succeeded")
                except Exception as e:
                    print(f"  RANSAC+ITERATIVE failed: {e}")
                    success = False

            # Method 4: Try AP3P (needs exactly 4 points, very robust)
            if not success and len(world_points_3d) >= 4:
                try:
                    print("Trying AP3P solver...")
                    success, rvec, tvec = cv2.solvePnP(
                        world_points_3d[:4],  # Use first 4 points
                        image_points_2d[:4],
                        K,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_AP3P
                    )
                    if success:
                        print("  AP3P solver succeeded (using first 4 points)")
                        inliers = np.arange(4).reshape(-1, 1)
                except Exception as e:
                    print(f"  AP3P solver failed: {e}")
                    success = False

            if not success:
                print("\nERROR: All PnP solvers failed!")
                print("Possible causes:")
                print("  - Inconsistent point matches (verify manually)")
                print("  - Wrong camera intrinsics (check focal length)")
                print("  - Points too close together or collinear")
                return None

            num_inliers = len(inliers) if inliers is not None else 0
            print(f"\nPnP Solution found: {num_inliers}/{len(world_points_3d)} inliers")

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            print(f"\nRotation matrix R:\n{R}")
            print(f"Translation vector t: {tvec.flatten()}")

            # Extract pan, tilt, roll from rotation matrix
            # The rotation matrix R transforms from world to camera coordinates
            # R = Rz(pan) @ Rx(tilt) @ Ry(roll) approximately
            #
            # For a camera looking at ground:
            # - Pan (yaw): rotation around vertical axis (Z in world)
            # - Tilt (pitch): rotation around horizontal axis (X in camera)
            # - Roll: rotation around optical axis

            # Extract Euler angles (assuming ZYX or similar convention)
            # This depends on your coordinate system conventions
            #
            # Standard decomposition for R = Rz @ Ry @ Rx:
            # tilt (x) = atan2(-R[1,2], R[2,2])
            # pan (y) = asin(R[0,2])
            # roll (z) = atan2(-R[0,1], R[0,0])

            # Alternative: use cv2.decomposeProjectionMatrix or direct extraction
            # Let's use a robust method

            # For camera pointing roughly downward:
            # The camera's Z-axis (optical axis) in world coords is R[:,2]
            # Pan = angle of projection onto XY plane
            # Tilt = angle from horizontal

            optical_axis = R[2, :]  # Third row of R is camera Z in world frame
            print(f"Optical axis in world frame: {optical_axis}")

            # Pan: angle in XY plane (from Y axis, positive = clockwise from above)
            pan_rad = math.atan2(optical_axis[0], optical_axis[1])
            pan_deg = math.degrees(pan_rad)

            # Tilt: angle from horizontal (0 = horizontal, 90 = straight down)
            horizontal_component = math.sqrt(optical_axis[0]**2 + optical_axis[1]**2)
            tilt_rad = math.atan2(-optical_axis[2], horizontal_component)
            tilt_deg = math.degrees(tilt_rad)

            # Roll: rotation around optical axis
            # This is trickier - we need to look at the camera's X-axis
            camera_x_in_world = R[0, :]  # First row of R
            # Project onto plane perpendicular to optical axis
            # For simplicity, assume roll is small and extract from R
            roll_rad = math.atan2(R[2, 0], R[2, 1]) - pan_rad
            roll_deg = math.degrees(roll_rad)

            # Translation gives camera position in world frame
            # tvec is the position of world origin in camera frame
            # Camera position in world = -R^T @ tvec
            camera_pos_world = -R.T @ tvec.flatten()

            print(f"\n=== DEBUG PnP ===")
            print(f"camera_pos_world = ({camera_pos_world[0]:.2f}, {camera_pos_world[1]:.2f}, {camera_pos_world[2]:.2f})")
            print(f"optical_axis (R[2,:]) = ({optical_axis[0]:.4f}, {optical_axis[1]:.4f}, {optical_axis[2]:.4f})")
            print(f"Initial pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}° (before flip check)")

            # Handle planar PnP ambiguity: when all GCPs are on the ground plane (Z=0),
            # solvePnP may find an equivalent solution with camera below the plane.
            # For PTZ cameras, we know the camera is always above ground, so ensure Z > 0.
            position_was_flipped = False
            if camera_pos_world[2] < 0:
                print(f"\nPnP returned camera below ground (Z={camera_pos_world[2]:.2f}m), flipping to above-ground solution")
                # Flip the solution: negate Z and adjust rotation accordingly
                camera_pos_world[2] = -camera_pos_world[2]
                # The rotation matrix needs to be adjusted for the flipped coordinate
                # This is equivalent to applying a reflection across the XY plane
                R = R @ np.diag([1.0, 1.0, -1.0])
                position_was_flipped = True
                # Recompute optical axis after rotation adjustment
                optical_axis = R[2, :]
                # Recompute pan and tilt
                pan_rad = math.atan2(optical_axis[0], optical_axis[1])
                pan_deg = math.degrees(pan_rad)
                horizontal_component = math.sqrt(optical_axis[0]**2 + optical_axis[1]**2)
                tilt_rad = math.atan2(-optical_axis[2], horizontal_component)
                tilt_deg = math.degrees(tilt_rad)

            # Note: When we flip R with R @ diag([1,1,-1]), the optical_axis[2] sign is
            # already inverted, so the recomputed tilt has the correct sign.
            # No additional negation needed here.
            if position_was_flipped:
                print(f"After flip, tilt={tilt_deg:.1f}° (recomputed from flipped R)")

            print(f"\nExtracted parameters:")
            print(f"  Pan: {pan_deg:.2f}°")
            print(f"  Tilt: {tilt_deg:.2f}°")
            print(f"  Roll: {roll_deg:.2f}°")
            print(f"  Camera position (local XY): ({camera_pos_world[0]:.2f}, {camera_pos_world[1]:.2f}, {camera_pos_world[2]:.2f})m")

            # For coplanar points (all on ground plane), the Z/height from PnP is UNRELIABLE
            # because PnP can't distinguish height from focal length scaling
            # Keep the original camera height instead
            pnp_derived_height = -camera_pos_world[2]  # What PnP thinks
            print(f"  PnP derived height: {pnp_derived_height:.2f}m (UNRELIABLE for coplanar points)")
            print(f"  Keeping original height: {camera_height:.2f}m")
            pnp_height = camera_height  # Keep original height

            # Compute reprojection error
            projected_pts, _ = cv2.projectPoints(
                world_points_3d, rvec, tvec, K, dist_coeffs
            )
            projected_pts = projected_pts.reshape(-1, 2)

            errors = np.linalg.norm(image_points_2d - projected_pts, axis=1)
            rms_error = np.sqrt(np.mean(errors**2))
            max_error = np.max(errors)

            print(f"\nReprojection errors:")
            print(f"  RMS: {rms_error:.2f}px")
            print(f"  Max: {max_error:.2f}px")
            print(f"  Per-point: {[f'{e:.1f}' for e in errors[:10]]}{'...' if len(errors) > 10 else ''}")

            # Prepare suggested parameters
            # Note: We need to convert local XY offset to GPS offset
            delta_x = camera_pos_world[0]
            delta_y = camera_pos_world[1]

            # Convert XY offset to lat/lon offset
            R_EARTH = 6371000
            ref_lat_rad = math.radians(camera_lat)
            delta_lon = delta_x / (R_EARTH * math.cos(ref_lat_rad) * math.radians(1))
            delta_lat = delta_y / (R_EARTH * math.radians(1))

            new_camera_lat = camera_lat + delta_lat
            new_camera_lon = camera_lon + delta_lon

            return {
                'success': True,
                'method': 'PnP',
                'rms_error': rms_error,
                'max_error': max_error,
                'num_inliers': num_inliers,
                'num_points': len(world_points_3d),
                'per_point_errors': errors.tolist(),
                'rotation_matrix': R.tolist(),
                'translation_vector': tvec.flatten().tolist(),
                'suggested_params': {
                    'pan_deg': pan_deg,
                    'tilt_deg': tilt_deg,
                    'roll_deg': roll_deg,
                    'height_m': pnp_height,
                    'camera_lat': new_camera_lat,
                    'camera_lon': new_camera_lon,
                    'delta_lat': delta_lat,
                    'delta_lon': delta_lon
                },
                'current_params': {
                    'pan_deg': self.camera_params.get('pan_deg'),
                    'tilt_deg': self.camera_params.get('tilt_deg'),
                    'height_m': camera_height,
                    'camera_lat': camera_lat,
                    'camera_lon': camera_lon
                }
            }

        except Exception as e:
            print(f"PnP calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_mask_vertices_for_calibration(self) -> Optional[Dict[str, Dict]]:
        """
        Extract polygon VERTICES from camera_mask and match to nearest projected points.

        For line features (parking spots, road markings), GCPs are typically at
        corners/endpoints, not centroids. This extracts vertices from contours.

        Returns:
            Dictionary mapping point names to observed pixel locations:
            {'point_name': {'pixel_u': float, 'pixel_v': float}, ...}
            Returns None if extraction fails.
        """
        if self.camera_mask is None:
            return None

        if not self.projected_points:
            return None

        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(
                self.camera_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("No contours found in mask")
                return None

            # Extract vertices from all contours
            all_vertices = []
            for contour in contours:
                # Filter out very small contours (noise)
                area = cv2.contourArea(contour)
                if area < 10:
                    continue

                # Approximate contour to polygon to get clean vertices
                # epsilon controls simplification: smaller = more vertices
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.01 * perimeter  # 1% of perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Extract each vertex
                for point in approx:
                    x, y = point[0]
                    all_vertices.append({'x': float(x), 'y': float(y)})

            if not all_vertices:
                print("No vertices extracted from contours")
                return None

            print(f"Extracted {len(all_vertices)} vertices from {len(contours)} contours")

            # Match each projected point to nearest vertex
            result = {}
            used_vertices = set()  # Track which vertices are already matched

            for proj_pt in self.projected_points:
                if not proj_pt.get('visible', False):
                    continue

                name = proj_pt.get('name', '')
                pred_u = proj_pt.get('pixel_u')
                pred_v = proj_pt.get('pixel_v')

                if pred_u is None or pred_v is None:
                    continue

                # Find nearest vertex that hasn't been used yet
                min_dist = float('inf')
                nearest_idx = None
                for i, vertex in enumerate(all_vertices):
                    if i in used_vertices:
                        continue
                    dist = math.sqrt((vertex['x'] - pred_u)**2 + (vertex['y'] - pred_v)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = i

                # Only match if reasonably close
                # Use tighter threshold (40px) to avoid bad matches
                # Matches at 50-100px are likely wrong and will hurt calibration
                MAX_MATCH_DISTANCE = 40.0
                if nearest_idx is not None and min_dist < MAX_MATCH_DISTANCE:
                    vertex = all_vertices[nearest_idx]
                    result[name] = {
                        'pixel_u': vertex['x'],
                        'pixel_v': vertex['y'],
                        'match_distance': min_dist
                    }
                    used_vertices.add(nearest_idx)
                    match_quality = "GOOD" if min_dist < 20 else "OK" if min_dist < 30 else "WEAK"
                    print(f"  [{match_quality}] '{name}' -> vertex ({vertex['x']:.1f}, {vertex['y']:.1f}), dist={min_dist:.1f}px")
                else:
                    print(f"  [SKIP] '{name}' - nearest vertex at {min_dist:.1f}px (>{MAX_MATCH_DISTANCE}px)")

            # Report match quality statistics
            if result:
                distances = [r['match_distance'] for r in result.values()]
                good_count = sum(1 for d in distances if d < 20)
                ok_count = sum(1 for d in distances if 20 <= d < 30)
                weak_count = sum(1 for d in distances if d >= 30)
                print(f"\nMatch quality: {good_count} GOOD (<20px), {ok_count} OK (20-30px), {weak_count} WEAK (30-{MAX_MATCH_DISTANCE}px)")
                print(f"Average match distance: {np.mean(distances):.1f}px")
                print(f"Total matched: {len(result)} points")

            return result if result else None

        except Exception as e:
            print(f"Failed to extract mask vertices: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Keep old method name as alias for backwards compatibility
    def _extract_mask_centroids_for_calibration(self) -> Optional[Dict[str, Dict]]:
        """Alias for _extract_mask_vertices_for_calibration (uses vertices, not centroids)."""
        return self._extract_mask_vertices_for_calibration()

    def freeze_gcp_observations(self) -> int:
        """
        Capture and freeze GCP observations from current SAM3 mask.

        This should be called ONCE when SAM3 detection runs on the GCP tab.
        The frozen observations are then used for ALL subsequent calibrations,
        preventing the drift caused by re-matching on every calibration.

        Returns:
            Number of observations frozen, or 0 if failed.
        """
        if self.camera_mask is None:
            print("Cannot freeze observations: no camera mask available")
            return 0

        if not self.projected_points:
            print("Cannot freeze observations: no projected points available")
            return 0

        # Extract centroids from mask
        observations = self._extract_mask_centroids_for_calibration()

        if not observations:
            print("Cannot freeze observations: no centroids matched to projected points")
            return 0

        # Store as frozen observations
        self.frozen_gcp_observations = observations

        print(f"\n{'='*60}")
        print("FROZEN GCP OBSERVATIONS")
        print(f"{'='*60}")
        print(f"Captured {len(observations)} observation(s) from SAM3 detection")
        for name, obs in list(observations.items())[:5]:
            print(f"  {name}: ({obs['pixel_u']:.1f}, {obs['pixel_v']:.1f})")
        if len(observations) > 5:
            print(f"  ... and {len(observations) - 5} more")
        print("These observations are now FROZEN and will be used for all calibrations.")
        print("Camera parameter changes will NOT affect these observed positions.")
        print(f"{'='*60}\n")

        return len(observations)

    def clear_frozen_observations(self):
        """Clear frozen observations, allowing re-capture."""
        self.frozen_gcp_observations = None
        print("Frozen GCP observations cleared. Run SAM3 detection to capture new observations.")

    def get_detected_vertices(self) -> Optional[List[Dict]]:
        """
        Get all detected vertices from the camera mask.

        Returns a list of vertices that can be used for manual matching.
        Each vertex has an index, x, y coordinates.

        Returns:
            List of vertices: [{'idx': int, 'x': float, 'y': float}, ...]
            Returns None if no mask is available.
        """
        if self.camera_mask is None:
            return None

        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(
                self.camera_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Extract vertices from all contours
            all_vertices = []
            vertex_idx = 0
            for contour in contours:
                # Filter out very small contours (noise)
                area = cv2.contourArea(contour)
                if area < 10:
                    continue

                # Approximate contour to polygon to get clean vertices
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.01 * perimeter  # 1% of perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Extract each vertex
                for point in approx:
                    x, y = point[0]
                    all_vertices.append({
                        'idx': vertex_idx,
                        'x': float(x),
                        'y': float(y)
                    })
                    vertex_idx += 1

            return all_vertices if all_vertices else None

        except Exception as e:
            print(f"Failed to get detected vertices: {e}")
            return None

    def set_manual_matches(self, matches: Dict[str, Dict]) -> int:
        """
        Set manual matches from user selection.

        Args:
            matches: Dictionary mapping GCP names to pixel coordinates:
                     {'point_name': {'pixel_u': float, 'pixel_v': float}, ...}

        Returns:
            Number of matches stored.
        """
        if not matches:
            return 0

        # Store as frozen observations
        self.frozen_gcp_observations = matches

        print(f"\n{'='*60}")
        print("MANUAL GCP MATCHES STORED")
        print(f"{'='*60}")
        print(f"Stored {len(matches)} manual match(es)")
        for name, obs in list(matches.items())[:5]:
            print(f"  {name}: ({obs['pixel_u']:.1f}, {obs['pixel_v']:.1f})")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")
        print("These observations will be used for all calibrations.")
        print(f"{'='*60}\n")

        return len(matches)


class UnifiedHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for unified two-tab interface."""

    session: UnifiedSession = None
    temp_dir: str = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            # Serve main two-tab HTML
            html = generate_unified_html(self.session)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

        elif self.path == '/frame.jpg':
            # Serve frame image
            frame_path = os.path.join(self.temp_dir, 'frame.jpg')
            with open(frame_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        else:
            self.send_error(404)

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError) as e:
            self.send_json_response({'success': False, 'error': f'Invalid request: {e}'})
            return

        if self.path == '/api/add_point':
            # Add KML point (Tab 1)
            point = self.session.add_point(
                post_data['px'],
                post_data['py'],
                post_data['name'],
                post_data['category']
            )
            self.send_json_response({
                'success': True,
                'point': point,
                'total_points': len(self.session.points)
            })

        elif self.path == '/api/delete_point':
            # Delete point
            success = self.session.delete_point(post_data['index'])
            self.send_json_response({
                'success': success,
                'total_points': len(self.session.points)
            })

        elif self.path == '/api/get_points':
            # Get current points list
            self.send_json_response({
                'points': self.session.points,
                'total_points': len(self.session.points)
            })

        elif self.path == '/api/export_kml':
            # Export current points to KML
            kml_content = self.session.export_kml()
            output_path = str(self.session.image_path.with_suffix('.kml'))

            try:
                with open(output_path, 'w') as f:
                    f.write(kml_content)

                self.send_json_response({
                    'success': True,
                    'path': output_path,
                    'count': len(self.session.points)
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/import_kml':
            # Import KML file
            try:
                kml_text = post_data.get('kml', '')
                points = self.session.parse_kml(kml_text)

                # Clear existing points and add imported ones
                self.session.points = []
                for p in points:
                    self.session.add_point(p['px'], p['py'], p['name'], p['category'])

                self.send_json_response({
                    'success': True,
                    'points': points,
                    'total_points': len(self.session.points)
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/switch_to_gcp':
            # Switch to GCP Capture tab
            # Auto-save KML to temp file
            temp_kml_path = os.path.join(self.temp_dir, 'auto_save.kml')
            kml_content = self.session.export_kml()

            try:
                with open(temp_kml_path, 'w') as f:
                    f.write(kml_content)
            except Exception as e:
                print(f"Warning: Could not auto-save KML: {e}")

            # Capture camera frame if not already done
            if self.session.camera_frame is None:
                try:
                    self.capture_camera_frame()
                except Exception as e:
                    self.send_json_response({
                        'success': False,
                        'error': f'Failed to capture camera frame: {str(e)}'
                    })
                    return

            # Project points for GCP tab
            projected_points = self.session.project_points_to_image()
            self.session.projected_points = projected_points

            # Encode camera frame as base64
            camera_frame_b64 = None
            if self.session.camera_frame is not None:
                success, buffer = cv2.imencode('.jpg', self.session.camera_frame)
                if success:
                    camera_frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Project cartography mask to camera coordinates if available (Bug 3 fix - improved logging)
            projected_mask_b64 = None
            projected_mask_available = False
            projection_error = None
            if self.session.cartography_mask is not None:
                print(f"Cartography mask exists, attempting projection...")
                # Debug: save original cartography mask
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, 'debug_cartography_mask.png'), self.session.cartography_mask)
                print(f"Debug: Saved cartography mask to {debug_dir}/debug_cartography_mask.png")

                projected_mask = self.session.project_cartography_mask_to_camera()
                if projected_mask is not None:
                    # Debug: save projected mask
                    cv2.imwrite(os.path.join(debug_dir, 'debug_projected_mask.png'), projected_mask)
                    print(f"Debug: Saved projected mask to {debug_dir}/debug_projected_mask.png")
                    print(f"Debug: Projected mask shape: {projected_mask.shape}, non-zero pixels: {cv2.countNonZero(projected_mask)}")

                    success, mask_buffer = cv2.imencode('.png', projected_mask)
                    if success:
                        projected_mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
                        projected_mask_available = True
                        print(f"Mask projection successful, base64 length: {len(projected_mask_b64)}")
                    else:
                        print("Warning: Failed to encode projected mask as PNG")
                        projection_error = "encoding_failed"
                else:
                    print("Warning: Mask projection failed unexpectedly")
                    projection_error = "projection_failed"
            else:
                print("No cartography mask available for projection")

            # Run initial calibration to get quality metrics
            calibration_result = None
            calibration_formatted = None
            calibration_available = CALIBRATOR_AVAILABLE
            if CALIBRATOR_AVAILABLE and len(projected_points) >= 4:
                try:
                    calibration_result = self.session.run_calibration()
                    if calibration_result and calibration_result.get('success', False):
                        print(f"Initial calibration: RMS error = {calibration_result.get('final_error', 0):.2f}px")
                        # Format for frontend
                        calibration_formatted = self._format_calibration_result(calibration_result)
                except Exception as e:
                    print(f"Initial calibration failed: {e}")

            self.send_json_response({
                'success': True,
                'projected_points': projected_points,
                'camera_frame': camera_frame_b64,
                'camera_params': self.camera_params_to_dict(),
                'kml_saved_to': temp_kml_path,
                'total_points': len(self.session.points),
                'projected_mask': projected_mask_b64,
                'projected_mask_available': projected_mask_available,
                'projected_mask_offset': self.session.projected_mask_offset,
                'projection_error': projection_error,
                'calibration_available': calibration_available,
                'calibration': calibration_formatted
            })

        elif self.path == '/api/run_calibration':
            # Run GCP-based calibration with optional observed pixel locations
            try:
                observed_pixels = post_data.get('observed_pixels', None)
                calibration_result = self.session.run_calibration(observed_pixels)

                if calibration_result is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'Calibration could not be performed (check console for details)'
                    })
                elif not calibration_result.get('success', False):
                    self.send_json_response({
                        'success': False,
                        'error': calibration_result.get('error', 'Calibration failed')
                    })
                else:
                    calibration_formatted = self._format_calibration_result(calibration_result)
                    self.send_json_response({
                        'success': True,
                        'calibration': calibration_formatted
                    })

            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/run_pnp_calibration':
            # Run PnP-based calibration (direct pose estimation)
            try:
                pnp_result = self.session.run_pnp_calibration()

                if pnp_result is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'PnP calibration could not be performed (check console for details)'
                    })
                elif not pnp_result.get('success', False):
                    self.send_json_response({
                        'success': False,
                        'error': pnp_result.get('error', 'PnP calibration failed')
                    })
                else:
                    # Format result for frontend
                    suggested = pnp_result.get('suggested_params', {})
                    current = pnp_result.get('current_params', {})

                    self.send_json_response({
                        'success': True,
                        'method': 'PnP',
                        'rms_error': pnp_result.get('rms_error', 0),
                        'max_error': pnp_result.get('max_error', 0),
                        'num_inliers': pnp_result.get('num_inliers', 0),
                        'num_points': pnp_result.get('num_points', 0),
                        'suggested_params': {
                            'pan': suggested.get('pan_deg'),
                            'tilt': suggested.get('tilt_deg'),
                            'roll': suggested.get('roll_deg'),
                            'height': suggested.get('height_m'),
                            'camera_lat': suggested.get('camera_lat'),
                            'camera_lon': suggested.get('camera_lon')
                        },
                        'current_params': {
                            'pan': current.get('pan_deg'),
                            'tilt': current.get('tilt_deg'),
                            'height': current.get('height_m'),
                            'camera_lat': current.get('camera_lat'),
                            'camera_lon': current.get('camera_lon')
                        },
                        'changes': {
                            'pan': (suggested.get('pan_deg', 0) or 0) - (current.get('pan_deg', 0) or 0),
                            'tilt': (suggested.get('tilt_deg', 0) or 0) - (current.get('tilt_deg', 0) or 0),
                            'height': (suggested.get('height_m', 0) or 0) - (current.get('height_m', 0) or 0),
                            'lat': suggested.get('delta_lat', 0),
                            'lon': suggested.get('delta_lon', 0)
                        }
                    })

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/capture_frame':
            # Capture new camera frame
            try:
                self.capture_camera_frame()

                # Re-project points with new frame
                projected_points = self.session.project_points_to_image()
                self.session.projected_points = projected_points

                # Encode camera frame as base64
                success, buffer = cv2.imencode('.jpg', self.session.camera_frame)
                if not success:
                    self.send_json_response({
                        'success': False,
                        'error': 'Failed to encode camera frame'
                    })
                    return

                camera_frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Re-project cartography mask with new camera params
                projected_mask_b64 = None
                projected_mask_available = False
                if self.session.cartography_mask is not None:
                    projected_mask = self.session.project_cartography_mask_to_camera()
                    if projected_mask is not None:
                        success_mask, mask_buffer = cv2.imencode('.png', projected_mask)
                        if success_mask:
                            projected_mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
                            projected_mask_available = True

                self.send_json_response({
                    'success': True,
                    'camera_frame': camera_frame_b64,
                    'camera_params': self.camera_params_to_dict(),
                    'projected_points': projected_points,
                    'projected_mask': projected_mask_b64,
                    'projected_mask_available': projected_mask_available,
                    'projected_mask_offset': self.session.projected_mask_offset
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/detect_features':
            # SAM3 feature detection endpoint
            try:
                method = post_data.get('method', 'sam3')
                tab = post_data.get('tab', 'kml')  # 'kml' or 'gcp'

                if method != 'sam3':
                    self.send_json_response({
                        'success': False,
                        'error': f'Unknown detection method: {method}'
                    })
                    return

                # Get API key from environment
                api_key = os.environ.get('ROBOFLOW_API_KEY', '')
                if not api_key:
                    self.send_json_response({
                        'success': False,
                        'error': 'ROBOFLOW_API_KEY environment variable not set'
                    })
                    return

                # Get the appropriate frame based on tab
                if tab == 'gcp':
                    frame = self.session.camera_frame
                else:
                    frame = self.session.cartography_frame

                if frame is None:
                    self.send_json_response({
                        'success': False,
                        'error': f'No {tab} frame available'
                    })
                    return

                # Get and validate preprocessing option
                preprocessing = post_data.get('preprocessing', 'none')
                if preprocessing not in VALID_PREPROCESSING_TYPES:
                    preprocessing = 'none'  # Default to none for invalid values

                # Apply preprocessing to a copy of the frame (preserve original)
                processed_frame = apply_preprocessing(frame.copy(), preprocessing)
                
                # Encode processed frame to base64
                success, buffer = cv2.imencode('.jpg', processed_frame)
                if not success:
                    self.send_json_response({
                        'success': False,
                        'error': 'Failed to encode frame'
                    })
                    return
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Use default prompt or custom prompt (different defaults for camera vs cartography)
                default_prompt = DEFAULT_SAM3_PROMPT_CAMERA if tab == 'gcp' else DEFAULT_SAM3_PROMPT_CARTOGRAPHY
                prompt = post_data.get('prompt', default_prompt)

                # Call Roboflow SAM3 API
                api_url = f"https://serverless.roboflow.com/sam3/concept_segment?api_key={api_key}"

                request_body = {
                    "format": "polygon",
                    "image": {
                        "type": "base64",
                        "value": image_base64
                    },
                    "prompts": [
                        {"type": "text", "text": prompt}
                    ]
                }

                headers = {'Content-Type': 'application/json'}
                response = requests.post(api_url, json=request_body, headers=headers, timeout=120)

                if response.status_code != 200:
                    self.send_json_response({
                        'success': False,
                        'error': f'SAM3 API request failed: {response.status_code}'
                    })
                    return

                # Parse response
                try:
                    api_response = response.json()
                except json.JSONDecodeError:
                    self.send_json_response({
                        'success': False,
                        'error': 'Invalid JSON response from SAM3 API'
                    })
                    return

                # Create binary mask
                frame_height, frame_width = frame.shape[:2]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                prompt_results = api_response.get('prompt_results', [])
                total_predictions = 0
                total_polygons = 0
                confidence_scores = []

                for prompt_result in prompt_results:
                    predictions = prompt_result.get('predictions', [])
                    total_predictions += len(predictions)

                    for prediction in predictions:
                        confidence = prediction.get('confidence', 0)
                        confidence_scores.append(confidence)
                        masks = prediction.get('masks', [])

                        for polygon in masks:
                            if isinstance(polygon, list) and len(polygon) >= 3:
                                pts = np.array([[int(pt[0]), int(pt[1])] for pt in polygon if len(pt) >= 2], dtype=np.int32)
                                if len(pts) >= 3:
                                    cv2.fillPoly(mask, [pts], 255)
                                    total_polygons += 1

                # Store mask
                if tab == 'gcp':
                    self.session.camera_mask = mask
                    # CRITICAL: Freeze GCP observations when SAM3 detects features on camera
                    # This captures where features ACTUALLY are in the image
                    num_frozen = self.session.freeze_gcp_observations()
                    print(f"Froze {num_frozen} GCP observations from SAM3 detection")
                else:
                    self.session.cartography_mask = mask

                self.session.feature_mask_metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt,
                    'tab': tab,
                    'total_predictions': total_predictions,
                    'total_polygons': total_polygons,
                    'confidence_scores': confidence_scores
                }

                # Encode mask as base64
                success, mask_buffer = cv2.imencode('.png', mask)
                if not success:
                    self.send_json_response({
                        'success': False,
                        'error': 'Failed to encode mask'
                    })
                    return
                mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

                # Include frozen observations count for GCP tab
                frozen_count = len(self.session.frozen_gcp_observations) if self.session.frozen_gcp_observations else 0

                self.send_json_response({
                    'success': True,
                    'mask': mask_base64,
                    'metadata': {
                        'total_predictions': total_predictions,
                        'total_polygons': total_polygons,
                        'confidence_range': {
                            'min': min(confidence_scores) if confidence_scores else 0,
                            'max': max(confidence_scores) if confidence_scores else 0
                        },
                        'frozen_observations': frozen_count if tab == 'gcp' else None
                    }
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/update_camera_params':
            # Update camera parameters and reproject
            try:
                if self.session.camera_params is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'No camera parameters available'
                    })
                    return

                # Update parameters from request
                if 'pan_deg' in post_data:
                    self.session.camera_params['pan_deg'] = float(post_data['pan_deg'])
                if 'tilt_deg' in post_data:
                    self.session.camera_params['tilt_deg'] = float(post_data['tilt_deg'])
                if 'zoom' in post_data:
                    self.session.camera_params['zoom'] = float(post_data['zoom'])
                    # Recompute intrinsics with new zoom
                    sensor_width_mm = self.session.camera_params.get('sensor_width_mm', DEFAULT_SENSOR_WIDTH_MM)
                    K = CameraGeometry.get_intrinsics(
                        zoom_factor=self.session.camera_params['zoom'],
                        W_px=self.session.camera_params['image_width'],
                        H_px=self.session.camera_params['image_height'],
                        sensor_width_mm=sensor_width_mm
                    )
                    self.session.camera_params['K'] = K
                if 'height_m' in post_data:
                    self.session.camera_params['height_m'] = float(post_data['height_m'])
                if 'camera_lat' in post_data:
                    self.session.camera_params['camera_lat'] = float(post_data['camera_lat'])
                if 'camera_lon' in post_data:
                    self.session.camera_params['camera_lon'] = float(post_data['camera_lon'])

                # Reproject points
                projected_points = self.session.project_points_to_image()
                self.session.projected_points = projected_points

                # Re-project cartography mask with updated camera params
                projected_mask_b64 = None
                projected_mask_available = False
                if self.session.cartography_mask is not None:
                    projected_mask = self.session.project_cartography_mask_to_camera()
                    if projected_mask is not None:
                        success_mask, mask_buffer = cv2.imencode('.png', projected_mask)
                        if success_mask:
                            projected_mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
                            projected_mask_available = True

                self.send_json_response({
                    'success': True,
                    'camera_params': self.camera_params_to_dict(),
                    'projected_points': projected_points,
                    'projected_mask': projected_mask_b64,
                    'projected_mask_available': projected_mask_available,
                    'projected_mask_offset': self.session.projected_mask_offset
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/get_vertices':
            # Get all detected vertices for manual matching
            try:
                vertices = self.session.get_detected_vertices()
                if vertices is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'No detected features. Run SAM3 detection first.',
                        'vertices': []
                    })
                    return

                # Also return projected points for matching
                projected_points = self.session.projected_points or []
                visible_points = [p for p in projected_points if p.get('visible', False)]

                self.send_json_response({
                    'success': True,
                    'vertices': vertices,
                    'vertex_count': len(vertices),
                    'projected_points': visible_points,
                    'point_count': len(visible_points)
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/set_manual_matches':
            # Save manual matches from user
            try:
                matches = post_data.get('matches', {})
                if not matches:
                    self.send_json_response({
                        'success': False,
                        'error': 'No matches provided'
                    })
                    return

                # Convert matches format if needed
                # Expected format: {'point_name': {'pixel_u': float, 'pixel_v': float}, ...}
                formatted_matches = {}
                for name, coords in matches.items():
                    formatted_matches[name] = {
                        'pixel_u': float(coords.get('pixel_u', coords.get('x', 0))),
                        'pixel_v': float(coords.get('pixel_v', coords.get('y', 0)))
                    }

                num_matches = self.session.set_manual_matches(formatted_matches)

                self.send_json_response({
                    'success': True,
                    'num_matches': num_matches,
                    'message': f'Stored {num_matches} manual matches'
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        elif self.path == '/api/clear_matches':
            # Clear frozen observations
            try:
                self.session.clear_frozen_observations()
                self.send_json_response({
                    'success': True,
                    'message': 'Cleared all frozen observations'
                })
            except Exception as e:
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                })

        else:
            self.send_error(404)

    def send_json_response(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def capture_camera_frame(self):
        """Capture frame from live camera and get camera parameters."""
        if not GEOMETRY_AVAILABLE or not INTRINSICS_AVAILABLE:
            raise RuntimeError("Camera geometry or intrinsics modules not available")

        # Import helper functions
        try:
            from tools.capture_gcps_web import grab_frame_from_camera, get_camera_params_for_projection
        except ImportError as e:
            raise RuntimeError(f"Could not import camera functions: {e}")

        # Grab frame from camera
        frame, ptz_status = grab_frame_from_camera(self.session.camera_name, wait_time=2.0)

        # Get camera parameters for projection
        frame_height, frame_width = frame.shape[:2]
        camera_params = get_camera_params_for_projection(
            self.session.camera_name,
            image_width=frame_width,
            image_height=frame_height
        )

        # Store distortion coefficients for use in calibration (no frame undistortion)
        # Issue #135: Work in distorted image space for consistent calibration
        K = camera_params.get('K')
        if K is not None:
            # Get distortion coefficients from camera config
            dist_coeffs = np.zeros(5)  # [k1, k2, p1, p2, k3]
            try:
                cam_config = get_camera_by_name(self.session.camera_name)
                if cam_config:
                    k1 = cam_config.get('k1', 0.0)
                    k2 = cam_config.get('k2', 0.0)
                    p1 = cam_config.get('p1', 0.0)
                    p2 = cam_config.get('p2', 0.0)
                    k3 = cam_config.get('k3', 0.0)
                    dist_coeffs = np.array([k1, k2, p1, p2, k3])
            except Exception as e:
                print(f"Warning: Could not load distortion coefficients: {e}")

            if np.any(dist_coeffs != 0):
                print(f"Distortion coefficients loaded: k1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}, p1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}, k3={dist_coeffs[4]:.6f}")
                print("Frame NOT undistorted - calibration will use distorted image space with cv2.projectPoints/solvePnP")
            else:
                print("No distortion coefficients - using zero distortion")

            camera_params['dist_coeffs'] = dist_coeffs.tolist()

        # Always set undistorted=False regardless of K availability
        # Issue #135: Calibration works in distorted image space
        camera_params['undistorted'] = False

        self.session.camera_frame = frame
        self.session.camera_params = camera_params

        print(f"Captured camera frame: {frame_width}x{frame_height}")
        print(f"Camera params: pan={camera_params['pan_deg']:.1f}°, tilt={camera_params['tilt_deg']:.1f}°, zoom={camera_params['zoom']:.1f}x")

    def camera_params_to_dict(self) -> dict:
        """Convert camera parameters to JSON-serializable dict including footprint."""
        if self.session.camera_params is None:
            return {}

        params = self.session.camera_params.copy()

        # Convert numpy arrays to lists
        if 'K' in params and isinstance(params['K'], np.ndarray):
            params['K'] = params['K'].tolist()

        # Calculate camera footprint
        footprint = self.session.calculate_camera_footprint()

        return {
            'camera_lat': params.get('camera_lat', 0),
            'camera_lon': params.get('camera_lon', 0),
            'height_m': params.get('height_m', 0),
            'pan_deg': params.get('pan_deg', 0),
            'tilt_deg': params.get('tilt_deg', 0),
            'zoom': params.get('zoom', 1),
            'image_width': params.get('image_width', 0),
            'image_height': params.get('image_height', 0),
            'camera_footprint': footprint
        }

    def _format_calibration_result(self, calibration_result: dict) -> dict:
        """Format calibration result for frontend consumption."""
        if not calibration_result or not calibration_result.get('success', False):
            return None

        sp = calibration_result.get('suggested_params', {})
        current_pan = self.session.camera_params.get('pan_deg', 0) if self.session.camera_params else 0
        current_tilt = self.session.camera_params.get('tilt_deg', 0) if self.session.camera_params else 0
        current_height = self.session.camera_params.get('height_m', 0) if self.session.camera_params else 0

        per_gcp_errors = calibration_result.get('per_gcp_errors', [0])
        max_error = max(per_gcp_errors) if per_gcp_errors else 0

        # Check if using frozen observations
        has_frozen = self.session.frozen_gcp_observations is not None
        frozen_count = len(self.session.frozen_gcp_observations) if has_frozen else 0

        return {
            'rms_error': calibration_result.get('final_error', 0),
            'max_error': max_error,
            'num_points': calibration_result.get('num_gcps_used', 0),
            'num_inliers': calibration_result.get('num_inliers', 0),
            'initial_error': calibration_result.get('initial_error', 0),
            'improvement_percent': calibration_result.get('improvement_percent', 0),
            'has_frozen_observations': has_frozen,
            'frozen_observation_count': frozen_count,
            'suggested_params': {
                'pan': sp.get('pan_deg'),
                'tilt': sp.get('tilt_deg'),
                'height': sp.get('height_m'),
                'delta_pan': sp.get('pan_deg', current_pan) - current_pan if sp.get('pan_deg') is not None else None,
                'delta_tilt': sp.get('tilt_deg', current_tilt) - current_tilt if sp.get('tilt_deg') is not None else None,
                'delta_height': sp.get('height_m', current_height) - current_height if sp.get('height_m') is not None else None,
            } if sp else None
        }


def generate_unified_html(session: UnifiedSession) -> str:
    """Generate the unified two-tab HTML interface."""

    # Encode image as base64
    with open(session.image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    # Determine MIME type
    suffix = session.image_path.suffix.lower()
    mime_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff'
    }.get(suffix, 'image/jpeg')

    # Prepare config for JavaScript
    config = session.geotiff_params

    # Prepare initial camera params for JavaScript
    initial_camera_params = None
    if session.camera_params:
        initial_camera_params = {
            'camera_lat': session.camera_params.get('camera_lat'),
            'camera_lon': session.camera_params.get('camera_lon'),
            'height_m': session.camera_params.get('height_m'),
            'pan_deg': session.camera_params.get('pan_deg'),
            'tilt_deg': session.camera_params.get('tilt_deg'),
            'zoom': session.camera_params.get('zoom'),
            'image_width': session.camera_params.get('image_width'),
            'image_height': session.camera_params.get('image_height'),
            'camera_footprint': session.calculate_camera_footprint()
        }

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Unified GCP Tool - {session.camera_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }}

        /* Tab Navigation */
        .tab-container {{
            background: #0f3460;
            border-bottom: 2px solid #e94560;
        }}
        .tab-nav {{
            display: flex;
            padding: 0 20px;
        }}
        .tab-button {{
            padding: 15px 30px;
            background: transparent;
            border: none;
            color: #aaa;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        .tab-button:hover:not(.disabled) {{
            color: #fff;
            background: rgba(255,255,255,0.05);
        }}
        .tab-button.active {{
            color: #e94560;
            border-bottom-color: #e94560;
        }}
        .tab-button.disabled {{
            color: #555;
            cursor: not-allowed;
            opacity: 0.5;
        }}
        .tab-status {{
            font-size: 11px;
            color: #888;
            margin-left: 8px;
        }}

        /* Main Container */
        .container {{
            display: flex;
            height: calc(100vh - 60px);
        }}

        /* Tab Content */
        .tab-content {{
            display: none;
            width: 100%;
            height: 100%;
        }}
        .tab-content.active {{
            display: flex;
        }}

        /* Common Layout */
        .image-panel {{
            flex: 1;
            overflow: auto;
            position: relative;
            background: #16213e;
        }}
        .sidebar {{
            width: 350px;
            background: #0f3460;
            padding: 15px;
            overflow-y: auto;
            border-left: 1px solid #333;
        }}

        /* Image Container */
        #image-container {{
            position: relative;
            display: inline-block;
            cursor: crosshair;
        }}
        #gcp-image-container {{
            position: relative;
            display: inline-block;
            overflow: visible;
        }}
        #main-image {{
            display: block;
            max-width: none;
        }}
        #gcp-image {{
            display: block;
            max-width: none;
        }}

        /* Markers */
        .marker {{
            position: absolute;
            padding: 2px 5px;
            border-radius: 3px;
            border: 2px solid white;
            cursor: pointer;
            font-size: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            white-space: nowrap;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.8);
            z-index: 10;
        }}
        .marker.zebra {{ background: #e94560; }}
        .marker.arrow {{ background: #0ead69; }}
        .marker.parking {{ background: #3498db; }}
        .marker.other {{ background: #f39c12; }}

        .point-dot {{
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            border: 2px solid white;
            transform: translate(-50%, -50%);
            z-index: 5;
        }}
        .point-dot.zebra {{ background: #e94560; }}
        .point-dot.arrow {{ background: #0ead69; }}
        .point-dot.parking {{ background: #3498db; }}
        .point-dot.other {{ background: #f39c12; }}

        /* GCP Marker (larger for Tab 2) */
        .gcp-marker {{
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 3px solid #4CAF50;
            background: rgba(76, 175, 80, 0.3);
            transform: translate(-50%, -50%);
            z-index: 8;
        }}

        /* GCP Label */
        .gcp-label {{
            position: absolute;
            font-size: 10px;
            font-weight: bold;
            color: #fff;
            background: rgba(0, 0, 0, 0.7);
            padding: 1px 4px;
            border-radius: 3px;
            white-space: nowrap;
            z-index: 9;
            pointer-events: none;
            text-shadow: 1px 1px 1px #000;
        }}

        /* Sidebar Components */
        h2 {{
            margin-bottom: 15px;
            color: #e94560;
            font-size: 18px;
        }}
        h3 {{
            margin: 15px 0 10px 0;
            color: #0ead69;
            font-size: 14px;
        }}

        .controls {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            margin: 10px 0 5px;
            font-size: 12px;
            color: #aaa;
        }}
        select, input, button {{
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #333;
            border-radius: 4px;
            background: #16213e;
            color: #eee;
        }}
        button {{
            background: #e94560;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #c73e54;
        }}
        button.secondary {{
            background: #0ead69;
        }}
        button.secondary:hover {{
            background: #0c9a5c;
        }}
        button:disabled {{
            background: #555;
            cursor: not-allowed;
            opacity: 0.5;
        }}

        /* Calibration Panel */
        .calibration-panel {{
            background: #16213e;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .calibration-metric {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }}
        .calibration-metric:last-of-type {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #aaa;
            font-size: 12px;
        }}
        .metric-value {{
            color: #0ead69;
            font-weight: bold;
        }}
        .metric-value.error {{
            color: #e94560;
        }}
        .metric-value.warning {{
            color: #ffa500;
        }}
        .calibration-suggestion-header {{
            color: #aaa;
            font-size: 12px;
            margin-bottom: 5px;
            border-top: 1px solid #333;
            padding-top: 8px;
        }}
        .suggestion-item {{
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            font-size: 12px;
            color: #e0e0e0;
        }}

        /* Point List */
        .point-list {{
            margin-top: 20px;
        }}
        .point-item {{
            background: #16213e;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .point-item .info {{
            flex: 1;
        }}
        .point-item .name {{
            font-weight: bold;
            color: #e94560;
        }}
        .point-item .coords {{
            color: #888;
            font-size: 11px;
        }}
        .point-item .delete {{
            background: #c0392b;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }}

        /* Category Filters */
        .category-filters {{
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}
        .category-btn {{
            padding: 5px 10px;
            border: 2px solid;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
            transition: opacity 0.2s;
        }}
        .category-btn.zebra {{
            background: #e94560;
            border-color: #e94560;
            color: white;
        }}
        .category-btn.arrow {{
            background: #0ead69;
            border-color: #0ead69;
            color: white;
        }}
        .category-btn.parking {{
            background: #3498db;
            border-color: #3498db;
            color: white;
        }}
        .category-btn.other {{
            background: #f39c12;
            border-color: #f39c12;
            color: white;
        }}
        .category-btn.hidden {{
            background: transparent;
            opacity: 0.5;
        }}

        /* Instructions */
        .instructions {{
            background: #16213e;
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            color: #aaa;
            margin-bottom: 15px;
        }}

        /* Status Bar */
        .status {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
        }}

        /* Zoom Controls */
        .zoom-controls {{
            position: fixed;
            bottom: 20px;
            right: 370px;
            display: flex;
            gap: 5px;
            z-index: 1000;
        }}
        .zoom-controls button {{
            width: auto;
            padding: 8px 15px;
        }}

        /* GCP Tab specific */
        #gcp-points-list {{
            max-height: 300px;
            overflow-y: auto;
        }}

        /* Mask overlay */
        .mask-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            opacity: 0.5;
            mix-blend-mode: multiply;
        }}

        /* Projected mask overlay (map mask projected onto camera view) */
        .mask-overlay.projected {{
            opacity: 0.6;
            mix-blend-mode: normal;
            filter: sepia(100%) saturate(500%) hue-rotate(70deg) brightness(1.2);
        }}

        /* Camera mask overlay (camera-detected features) */
        .mask-overlay.camera {{
            opacity: 0.5;
            mix-blend-mode: multiply;
        }}

        /* Camera visualization */
        .camera-position-dot {{
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #4A90E2;
            border: 3px solid white;
            transform: translate(-50%, -50%);
            z-index: 6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            pointer-events: none;
        }}

        .camera-footprint-polygon {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 3;
        }}

        /* Camera toggle button */
        .camera-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #16213e;
            border: 1px solid #333;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 10px;
        }}
        .camera-toggle input {{
            width: auto;
            margin: 0;
        }}
        .camera-toggle label {{
            margin: 0;
            cursor: pointer;
            color: #4A90E2;
            font-weight: bold;
        }}
        .camera-toggle.disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <!-- Tab Navigation -->
    <div class="tab-container">
        <div class="tab-nav">
            <button class="tab-button active" onclick="switchTab('kml')" id="tab-btn-kml">
                KML Extractor
                <span class="tab-status" id="kml-tab-status">0 points</span>
            </button>
            <button class="tab-button disabled" onclick="switchTab('gcp')" id="tab-btn-gcp">
                GCP Capture
                <span class="tab-status">(disabled until points added)</span>
            </button>
        </div>
    </div>

    <!-- Main Container -->
    <div class="container">
        <!-- Tab 1: KML Extractor -->
        <div class="tab-content active" id="tab-kml">
            <div class="image-panel" id="kml-image-panel">
                <div id="image-container">
                    <img id="main-image" src="data:{mime_type};base64,{img_data}">
                </div>
            </div>

            <div class="sidebar">
                <h2>KML Point Extractor</h2>

                <div class="instructions">
                    <strong>Instructions:</strong><br>
                    1. Select a category below<br>
                    2. Click on image to place points<br>
                    3. Work left to right<br>
                    4. Switch to GCP Capture tab when ready
                </div>

                <div class="controls">
                    <label>Category:</label>
                    <select id="category">
                        <option value="zebra">Zebra Crossing Corner</option>
                        <option value="arrow">Arrow Tip</option>
                        <option value="parking">Parking Spot Corner</option>
                        <option value="other">Other</option>
                    </select>

                    <label>Point Name (auto-increments):</label>
                    <input type="text" id="point-name" placeholder="e.g., Z1, A1, P1">

                    <h3>Camera Visualization</h3>
                    <div class="camera-toggle" id="camera-toggle-container">
                        <input type="checkbox" id="camera-overlay-toggle" checked>
                        <label for="camera-overlay-toggle">Show Camera View</label>
                    </div>
                    <div id="camera-status" style="font-size: 11px; color: #888; margin-bottom: 10px;">
                        Shows camera position and estimated field of view
                    </div>

                    <h3>SAM3 Feature Detection</h3>
                    <label>Preprocessing:</label>
                    <select id="sam3-preprocessing-kml">
                        <option value="none">None</option>
                        <option value="clahe" selected>CLAHE (Recommended)</option>
                    </select>
                    
                    <label>Prompt:</label>
                    <input type="text" id="sam3-prompt-kml" placeholder="ground markings" value="ground markings">
                    <button onclick="detectFeatures('kml')" class="secondary">Detect Features</button>
                    <button onclick="toggleMask('kml')" id="toggle-mask-kml-btn" style="display: none;">Toggle Mask</button>

                    <h3>Export/Import</h3>
                    <button onclick="exportKML()" class="secondary">Export KML</button>

                    <label>Import KML:</label>
                    <input type="file" id="kml-file" accept=".kml" onchange="importKML(this)">

                    <button onclick="clearAll()">Clear All Points</button>
                </div>

                <div class="point-list">
                    <h3>Points (<span id="point-count">0</span>)</h3>
                    <div class="category-filters" id="category-filters">
                        <div class="category-btn zebra" onclick="toggleCategory('zebra')" data-category="zebra">Zebra</div>
                        <div class="category-btn arrow" onclick="toggleCategory('arrow')" data-category="arrow">Arrow</div>
                        <div class="category-btn parking" onclick="toggleCategory('parking')" data-category="parking">Parking</div>
                        <div class="category-btn other" onclick="toggleCategory('other')" data-category="other">Other</div>
                    </div>
                    <div id="points-container"></div>
                </div>
            </div>
        </div>

        <!-- Tab 2: GCP Capture -->
        <div class="tab-content" id="tab-gcp">
            <div class="image-panel" id="gcp-image-panel">
                <div id="gcp-image-container">
                    <img id="gcp-image" src="data:{mime_type};base64,{img_data}">
                </div>
            </div>

            <div class="sidebar">
                <h2>GCP Capture</h2>

                <div class="instructions">
                    <strong>Map-First Mode:</strong><br>
                    KML points from Tab 1 are projected onto the camera frame.
                    Use camera controls to adjust projection.
                </div>

                <div class="controls">
                    <h3>Camera Frame</h3>
                    <button onclick="captureNewFrame()" class="secondary">Capture New Frame</button>

                    <h3>Camera Parameters</h3>
                    <div id="camera-params-display" style="font-size: 11px; color: #aaa; margin-bottom: 10px;">
                        <div>Lat: <span id="param-lat">--</span>&deg;</div>
                        <div>Lon: <span id="param-lon">--</span>&deg;</div>
                        <div>Pan: <span id="param-pan">--</span>&deg;</div>
                        <div>Tilt: <span id="param-tilt">--</span>&deg;</div>
                        <div>Height: <span id="param-height">--</span>m</div>
                    </div>

                    <label>Arrow key modes (1=Lat/Lon, 2=Pan, 3=Tilt, 4=Height):</label>
                    <div id="latlon-mode-indicator" style="font-size: 11px; color: #888; margin-bottom: 5px; display: none;">
                        Arrow key mode: <span id="latlon-mode-status">OFF</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-bottom: 10px;">
                        <button onclick="adjustParam('camera_lat', -0.000001)">Lat -</button>
                        <button onclick="adjustParam('camera_lat', 0.000001)">Lat +</button>
                        <button onclick="adjustParam('camera_lon', -0.000001)">Lon -</button>
                        <button onclick="adjustParam('camera_lon', 0.000001)">Lon +</button>
                        <button onclick="adjustParam('pan', -1)">Pan -</button>
                        <button onclick="adjustParam('pan', 1)">Pan +</button>
                        <button onclick="adjustParam('tilt', -0.5)">Tilt -</button>
                        <button onclick="adjustParam('tilt', 0.5)">Tilt +</button>
                        <button onclick="adjustParam('height', -0.01)">Height -</button>
                        <button onclick="adjustParam('height', 0.01)">Height +</button>
                    </div>

                    <h3>Map Mask Projection</h3>
                    <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
                        Projects road markings from cartography map onto camera view
                    </div>
                    <button onclick="toggleProjectedMask()" id="toggle-projected-mask-btn" style="display: none;">Show Map Mask</button>

                    <h3>SAM3 Feature Detection</h3>
                    <label>Preprocessing:</label>
                    <select id="sam3-preprocessing-gcp">
                        <option value="none">None</option>
                        <option value="clahe" selected>CLAHE (Recommended)</option>
                    </select>
                    
                    <label>Prompt:</label>
                    <input type="text" id="sam3-prompt-gcp" placeholder="road marking" value="road marking">
                    <button onclick="detectFeatures('gcp')" class="secondary">Detect Features</button>
                    <button onclick="toggleMask('gcp')" id="toggle-mask-gcp-btn" style="display: none;">Toggle Camera Mask</button>
                    <button onclick="startManualMatching()" id="manual-match-btn" style="display: none;" class="secondary">Match Features Manually</button>

                    <h3>Calibration Quality</h3>
                    <div id="calibration-status" class="calibration-panel">
                        <div id="calibration-not-available" style="display: none; color: #999;">
                            Calibration not available (need 4+ visible GCPs)
                        </div>
                        <div id="calibration-results" style="display: none;">
                            <div class="calibration-metric">
                                <span class="metric-label">RMS Error:</span>
                                <span id="calibration-rms-error" class="metric-value">--</span>
                            </div>
                            <div class="calibration-metric">
                                <span class="metric-label">Max Error:</span>
                                <span id="calibration-max-error" class="metric-value">--</span>
                            </div>
                            <div class="calibration-metric">
                                <span class="metric-label">GCP Points:</span>
                                <span id="calibration-num-points" class="metric-value">--</span>
                            </div>
                            <div id="calibration-obs-status" style="display: none; margin-top: 8px; font-size: 11px;"></div>
                            <div id="calibration-suggestions" style="display: none; margin-top: 10px;"></div>
                        </div>
                        <button onclick="runCalibration()" class="secondary" id="recalibrate-btn">Re-run Calibration</button>
                        <button onclick="runPnPCalibration()" class="secondary" id="pnp-calibrate-btn" style="background: #0ead69;">Run PnP Calibration</button>
                        <div id="pnp-results" style="display: none; margin-top: 10px; padding: 10px; background: rgba(14, 173, 105, 0.1); border-radius: 4px;"></div>
                    </div>

                    <h3>Projected Points (<span id="gcp-point-count">0</span>)</h3>
                    <div id="gcp-points-list"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="status" id="status">Ready</div>

    <div class="zoom-controls">
        <button onclick="zoom(0.8)">-</button>
        <button onclick="zoom(1.25)">+</button>
        <button onclick="resetZoom()">Reset</button>
    </div>

    <script>
        const config = {json.dumps(config)};
        let points = [];
        let kmlZoom = 1;  // Independent zoom for Tab 1 (KML Extractor)
        let gcpZoom = 1;  // Independent zoom for Tab 2 (GCP Capture)
        let counters = {{ zebra: 1, arrow: 1, parking: 1, other: 1 }};
        let categoryVisibility = {{ zebra: true, arrow: true, parking: true, other: true }};
        let currentTab = 'kml';
        let projectedPoints = [];
        let cameraParams = {json.dumps(initial_camera_params) if initial_camera_params else 'null'};
        let maskVisible = {{ kml: false, gcp: false }};
        let maskData = {{ kml: null, gcp: null }};
        let projectedMaskVisible = false;
        let projectedMaskData = null;
        let projectedMaskOffset = null;  // [left, top] CSS offset for positioning mask overlay
        let cameraOverlayVisible = true;  // Camera visualization toggle state
        let activeMode = null;  // Arrow key mode: 'latlon', 'pan', 'tilt', 'height', or null

        const img = document.getElementById('main-image');
        const gcpImg = document.getElementById('gcp-image');
        const container = document.getElementById('image-container');
        const gcpContainer = document.getElementById('gcp-image-container');

        // Initialize camera visualization when image loads
        img.onload = function() {{
            if (cameraParams) {{
                enableCameraToggle();
                updateCameraVisualization();
                centerOnCamera();
            }}
        }};

        // Also check if image is already loaded (base64 inline images load synchronously)
        if (img.complete && img.naturalWidth > 0) {{
            if (cameraParams) {{
                enableCameraToggle();
                updateCameraVisualization();
                centerOnCamera();
            }}
        }}

        // Initialize
        updatePointName();
        document.getElementById('category').addEventListener('change', updatePointName);

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Only process if not typing in input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (currentTab === 'gcp') {{
                // Keys 1-4: Toggle arrow key modes
                if (e.key >= '1' && e.key <= '4') {{
                    const modes = ['latlon', 'pan', 'tilt', 'height'];
                    const modeNames = ['Lat/Lon', 'Pan', 'Tilt', 'Height'];
                    const selectedMode = modes[parseInt(e.key) - 1];
                    const modeName = modeNames[parseInt(e.key) - 1];

                    // Toggle mode: if same mode, turn off; otherwise switch to new mode
                    if (activeMode === selectedMode) {{
                        activeMode = null;
                        document.getElementById('latlon-mode-indicator').style.display = 'none';
                        updateStatus(`${{modeName}} arrow key mode: OFF`);
                    }} else {{
                        activeMode = selectedMode;
                        document.getElementById('latlon-mode-indicator').style.display = 'block';
                        document.getElementById('latlon-mode-status').textContent = modeName;
                        document.getElementById('latlon-mode-status').style.color = '#0f0';
                        updateStatus(`${{modeName}} arrow key mode: ON`);
                    }}
                    return;
                }}

                // Arrow keys: Control parameter based on active mode
                if (activeMode && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {{
                    e.preventDefault();  // Prevent page scrolling

                    // Define step sizes for each mode
                    const stepSizes = {{
                        latlon: e.shiftKey ? 0.00001 : 0.000001,
                        pan: e.shiftKey ? 10 : 1,
                        tilt: e.shiftKey ? 5 : 0.5,
                        height: e.shiftKey ? 0.1 : 0.01  // 10cm / 1cm
                    }};
                    const step = stepSizes[activeMode];

                    // Map mode to parameter names
                    const paramMap = {{
                        latlon: {{ up: 'camera_lat', down: 'camera_lat', left: 'camera_lon', right: 'camera_lon' }},
                        pan: {{ up: 'pan', down: 'pan', left: 'pan', right: 'pan' }},
                        tilt: {{ up: 'tilt', down: 'tilt', left: 'tilt', right: 'tilt' }},
                        height: {{ up: 'height', down: 'height', left: 'height', right: 'height' }}
                    }};

                    // Determine direction (positive or negative)
                    const isPositive = (e.key === 'ArrowUp' || e.key === 'ArrowRight');
                    const delta = isPositive ? step : -step;

                    // For latlon mode, use different params for different arrow directions
                    if (activeMode === 'latlon') {{
                        if (e.key === 'ArrowUp') adjustParam('camera_lat', step);
                        else if (e.key === 'ArrowDown') adjustParam('camera_lat', -step);
                        else if (e.key === 'ArrowRight') adjustParam('camera_lon', step);
                        else if (e.key === 'ArrowLeft') adjustParam('camera_lon', -step);
                    }} else {{
                        // For pan/tilt/height, up/right = increase, down/left = decrease
                        const param = paramMap[activeMode].up;
                        adjustParam(param, delta);
                    }}
                }}
            }}
        }});

        // Tab switching
        function switchTab(tabName) {{
            const kmlBtn = document.getElementById('tab-btn-kml');
            const gcpBtn = document.getElementById('tab-btn-gcp');
            const kmlTab = document.getElementById('tab-kml');
            const gcpTab = document.getElementById('tab-gcp');

            if (tabName === 'gcp' && gcpBtn.classList.contains('disabled')) {{
                updateStatus('Add at least 1 point before switching to GCP Capture tab');
                return;
            }}

            // Auto-save when switching from KML to GCP
            if (currentTab === 'kml' && tabName === 'gcp') {{
                autoSaveAndProject();
            }}

            currentTab = tabName;

            // Update tab buttons
            kmlBtn.classList.toggle('active', tabName === 'kml');
            gcpBtn.classList.toggle('active', tabName === 'gcp');

            // Update tab content
            kmlTab.classList.toggle('active', tabName === 'kml');
            gcpTab.classList.toggle('active', tabName === 'gcp');

            updateStatus(`Switched to ${{tabName === 'kml' ? 'KML Extractor' : 'GCP Capture'}} tab`);
        }}

        // Auto-save and project points when switching to GCP tab
        function autoSaveAndProject() {{
            if (points.length === 0) return;

            updateStatus('Capturing camera frame and projecting points...');

            fetch('/api/switch_to_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ points: points }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    projectedPoints = data.projected_points;
                    cameraParams = data.camera_params;

                    // Update camera frame if provided
                    if (data.camera_frame) {{
                        // Set up onload to calculate fit-to-view zoom after image loads
                        gcpImg.onload = function() {{
                            // Calculate zoom to fit image within panel (Bug 2 fix)
                            const gcpPanel = document.getElementById('gcp-image-panel');
                            if (gcpPanel && gcpImg.naturalWidth > 0) {{
                                const panelWidth = gcpPanel.clientWidth;
                                const panelHeight = gcpPanel.clientHeight;
                                const imgWidth = gcpImg.naturalWidth;
                                const imgHeight = gcpImg.naturalHeight;

                                // Calculate zoom to fit both dimensions
                                const zoomX = panelWidth / imgWidth;
                                const zoomY = panelHeight / imgHeight;
                                gcpZoom = Math.min(zoomX, zoomY, 1);  // Don't zoom in beyond 1

                                gcpImg.style.width = (imgWidth * gcpZoom) + 'px';

                                // Reset scroll position
                                gcpPanel.scrollTop = 0;
                                gcpPanel.scrollLeft = 0;
                            }}

                            // Update view with new zoom
                            updateGCPView();
                            if (projectedMaskVisible) showProjectedMask();

                            gcpImg.onload = null;
                        }};
                        gcpImg.src = 'data:image/jpeg;base64,' + data.camera_frame;
                    }}

                    // Handle projected mask data
                    console.log('Projected mask response:', {{
                        available: data.projected_mask_available,
                        hasData: !!data.projected_mask,
                        dataLength: data.projected_mask ? data.projected_mask.length : 0,
                        offset: data.projected_mask_offset,
                        error: data.projection_error
                    }});
                    const maskBtn = document.getElementById('toggle-projected-mask-btn');
                    if (data.projected_mask_available && data.projected_mask) {{
                        projectedMaskData = data.projected_mask;
                        projectedMaskOffset = data.projected_mask_offset;
                        if (maskBtn) {{
                            maskBtn.style.display = 'block';
                            console.log('Map Mask button set to visible');
                        }}
                    }} else {{
                        projectedMaskData = null;
                        projectedMaskOffset = null;
                        if (maskBtn) {{
                            maskBtn.style.display = 'none';
                        }}
                        if (data.projection_error) {{
                            console.warn('Mask projection failed:', data.projection_error);
                        }}
                    }}

                    updateCameraParamsDisplay();

                    // Update calibration display
                    updateCalibrationDisplay(data.calibration, data.calibration_available);

                    // Enable and update camera visualization on Tab 1
                    enableCameraToggle();
                    updateCameraVisualization();

                    // If no camera frame update, apply fit-to-view zoom now
                    if (!data.camera_frame) {{
                        fitGcpToView();
                    }}

                    updateStatus(`Captured frame and projected ${{data.projected_points.length}} points`);
                }} else {{
                    updateStatus('Error: ' + data.error);
                    alert('Failed to switch to GCP tab: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Auto-save failed:', err);
                updateStatus('Auto-save failed: ' + err.message);
            }});
        }}

        // Update GCP view with projected points
        function updateGCPView() {{
            const listContainer = document.getElementById('gcp-points-list');

            // Clear existing markers and labels
            document.querySelectorAll('.gcp-marker').forEach(m => m.remove());
            document.querySelectorAll('.gcp-label').forEach(l => l.remove());

            if (projectedPoints.length === 0) {{
                listContainer.innerHTML = '<div style="color: #888; font-size: 12px;">No points to display</div>';
                return;
            }}

            // Draw markers with labels
            projectedPoints.forEach((pt, idx) => {{
                if (pt.visible) {{
                    const marker = document.createElement('div');
                    marker.className = 'gcp-marker';
                    marker.style.left = (pt.pixel_u * gcpZoom) + 'px';
                    marker.style.top = (pt.pixel_v * gcpZoom) + 'px';
                    marker.title = pt.name;
                    marker.dataset.name = pt.name;
                    gcpContainer.appendChild(marker);

                    // Add label showing the point name
                    const label = document.createElement('div');
                    label.className = 'gcp-label';
                    label.textContent = pt.name;
                    label.style.left = (pt.pixel_u * gcpZoom + 12) + 'px';
                    label.style.top = (pt.pixel_v * gcpZoom - 6) + 'px';
                    gcpContainer.appendChild(label);
                }}
            }});

            // Update list
            listContainer.innerHTML = projectedPoints.map((pt, i) => `
                <div class="point-item">
                    <div class="info">
                        <div class="name">${{i+1}}. ${{pt.name}} (${{pt.category || 'point'}})</div>
                        <div class="coords">
                            ${{pt.visible ?
                                `Pixel: (${{pt.pixel_u.toFixed(1)}}, ${{pt.pixel_v.toFixed(1)}})` :
                                `Not visible (${{pt.reason || 'unknown'}})`
                            }}
                        </div>
                        <div class="coords">GPS: ${{pt.latitude.toFixed(6)}}, ${{pt.longitude.toFixed(6)}}</div>
                    </div>
                </div>
            `).join('');

            document.getElementById('gcp-point-count').textContent = projectedPoints.filter(p => p.visible).length;
        }}

        // KML Extractor functionality
        function updatePointName() {{
            const cat = document.getElementById('category').value;
            const prefix = {{ zebra: 'Z', arrow: 'A', parking: 'P', other: 'X' }}[cat];
            document.getElementById('point-name').value = prefix + counters[cat];
        }}

        container.addEventListener('click', function(e) {{
            if (e.target !== img) return;

            const rect = img.getBoundingClientRect();
            const px = (e.clientX - rect.left) / kmlZoom;
            const py = (e.clientY - rect.top) / kmlZoom;

            const category = document.getElementById('category').value;
            const name = document.getElementById('point-name').value || (category + '_' + points.length);

            addPoint(px, py, name, category);

            // Increment counter
            counters[category]++;
            updatePointName();
        }});

        function pixelToUTM(px, py) {{
            // Apply GDAL 6-parameter affine geotransform
            // easting = GT[0] + px*GT[1] + py*GT[2]
            // northing = GT[3] + px*GT[4] + py*GT[5]
            const gt = config.geotransform;
            const easting = gt[0] + px * gt[1] + py * gt[2];
            const northing = gt[3] + px * gt[4] + py * gt[5];
            return {{ easting, northing }};
        }}

        function addPoint(px, py, name, category) {{
            fetch('/api/add_point', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ px, py, name, category }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    points.push(data.point);
                    redrawMarkers();
                    updatePointsList();
                    updateTabStatus();

                    const utm = pixelToUTM(px, py);
                    updateStatus('Added: ' + name + ' at E:' + utm.easting.toFixed(2) + ' N:' + utm.northing.toFixed(2));
                }}
            }});
        }}

        function updatePointsList() {{
            const container = document.getElementById('points-container');
            container.innerHTML = points.map((p, i) => `
                <div class="point-item">
                    <div class="info">
                        <div class="name">${{i+1}}. ${{p.name}} (${{p.category}})</div>
                        <div class="coords">Pixel: (${{p.pixel_x.toFixed(1)}}, ${{p.pixel_y.toFixed(1)}})</div>
                        <div class="coords">UTM: E ${{p.easting.toFixed(2)}}, N ${{p.northing.toFixed(2)}}</div>
                    </div>
                    <div class="delete" onclick="deletePoint(${{i}})">X</div>
                </div>
            `).join('');
            document.getElementById('point-count').textContent = points.length;
        }}

        function deletePoint(index) {{
            fetch('/api/delete_point', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ index }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    points.splice(index, 1);
                    redrawMarkers();
                    updatePointsList();
                    updateTabStatus();
                }}
            }});
        }}

        function redrawMarkers() {{
            // Remove all existing markers and dots
            document.querySelectorAll('.marker, .point-dot').forEach(m => m.remove());

            // Get only visible points
            const visiblePoints = points.filter(p => categoryVisibility[p.category]);

            visiblePoints.forEach((p) => {{
                const pointX = p.pixel_x * kmlZoom;
                const pointY = p.pixel_y * kmlZoom;

                // Draw dot
                const dot = document.createElement('div');
                dot.className = 'point-dot ' + p.category;
                dot.style.left = pointX + 'px';
                dot.style.top = pointY + 'px';
                container.appendChild(dot);

                // Draw label
                const marker = document.createElement('div');
                marker.className = 'marker ' + p.category;
                marker.style.left = pointX + 'px';
                marker.style.top = pointY + 'px';
                marker.textContent = p.name;
                marker.onclick = (e) => {{ e.stopPropagation(); }};
                container.appendChild(marker);
            }});
        }}

        function toggleCategory(category) {{
            categoryVisibility[category] = !categoryVisibility[category];

            const btn = document.querySelector(`.category-btn[data-category="${{category}}"]`);
            if (categoryVisibility[category]) {{
                btn.classList.remove('hidden');
            }} else {{
                btn.classList.add('hidden');
            }}

            redrawMarkers();
            updateStatus(category + ' labels ' + (categoryVisibility[category] ? 'shown' : 'hidden'));
        }}

        function updateTabStatus() {{
            const statusEl = document.getElementById('kml-tab-status');
            const gcpBtn = document.getElementById('tab-btn-gcp');

            statusEl.textContent = points.length + ' points';

            // Enable GCP tab when at least 1 point exists
            if (points.length > 0) {{
                gcpBtn.classList.remove('disabled');
                gcpBtn.querySelector('.tab-status').textContent = 'ready';
            }} else {{
                gcpBtn.classList.add('disabled');
                gcpBtn.querySelector('.tab-status').textContent = '(disabled until points added)';
            }}
        }}

        // Load existing points from server on page load
        function loadExistingPoints() {{
            fetch('/api/get_points', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.points && data.points.length > 0) {{
                    points = data.points;
                    redrawMarkers();
                    updatePointsList();
                    updateTabStatus();
                    updateStatus('Loaded ' + points.length + ' pre-existing points');
                }}
            }})
            .catch(err => console.error('Failed to load existing points:', err));
        }}

        // Load existing points on page load
        loadExistingPoints();

        function zoom(factor) {{
            // Each tab has independent zoom (Bug 2 fix)
            if (currentTab === 'kml') {{
                kmlZoom *= factor;
                img.style.width = (img.naturalWidth * kmlZoom) + 'px';
                redrawMarkers();
                if (maskVisible.kml) showMask('kml');
                updateCameraVisualization();
            }} else {{
                gcpZoom *= factor;
                gcpImg.style.width = (gcpImg.naturalWidth * gcpZoom) + 'px';
                updateGCPView();
                if (maskVisible.gcp) showMask('gcp');
                if (projectedMaskVisible) showProjectedMask();
            }}
        }}

        function resetZoom() {{
            // Reset zoom for current tab only (Bug 2 fix)
            if (currentTab === 'kml') {{
                kmlZoom = 1;
                img.style.width = '';
                redrawMarkers();
                if (maskVisible.kml) showMask('kml');
                updateCameraVisualization();
            }} else {{
                fitGcpToView();
            }}
        }}

        function fitGcpToView() {{
            // Calculate zoom to fit GCP image within panel (Bug 2 fix)
            const gcpPanel = document.getElementById('gcp-image-panel');
            if (gcpPanel && gcpImg.naturalWidth > 0) {{
                const panelWidth = gcpPanel.clientWidth;
                const panelHeight = gcpPanel.clientHeight;
                const imgWidth = gcpImg.naturalWidth;
                const imgHeight = gcpImg.naturalHeight;

                // Calculate zoom to fit both dimensions
                const zoomX = panelWidth / imgWidth;
                const zoomY = panelHeight / imgHeight;
                gcpZoom = Math.min(zoomX, zoomY, 1);  // Don't zoom in beyond 1

                gcpImg.style.width = (imgWidth * gcpZoom) + 'px';

                // Reset scroll position
                gcpPanel.scrollTop = 0;
                gcpPanel.scrollLeft = 0;
            }} else {{
                gcpZoom = 1;
                gcpImg.style.width = '';
            }}

            updateGCPView();
            if (maskVisible.gcp) showMask('gcp');
            if (projectedMaskVisible) showProjectedMask();
        }}

        function clearAll() {{
            if (confirm('Clear all points?')) {{
                points = [];
                counters = {{ zebra: 1, arrow: 1, parking: 1, other: 1 }};
                redrawMarkers();
                updatePointsList();
                updatePointName();
                updateTabStatus();
            }}
        }}

        function exportKML() {{
            if (points.length === 0) {{
                alert('No points to export!');
                return;
            }}

            fetch('/api/export_kml', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ points: points }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    updateStatus('Exported to: ' + data.path);
                    alert('KML saved to: ' + data.path);
                }}
            }});
        }}

        function importKML(input) {{
            if (!input.files || !input.files[0]) return;

            const file = input.files[0];
            const reader = new FileReader();

            reader.onload = function(e) {{
                const kmlText = e.target.result;

                fetch('/api/import_kml', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ kml: kmlText }})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.success) {{
                        // Clear existing points
                        points = [];
                        counters = {{ zebra: 1, arrow: 1, parking: 1, other: 1 }};

                        // Add imported points
                        data.points.forEach(p => {{
                            const utm = pixelToUTM(p.px, p.py);
                            points.push({{
                                name: p.name,
                                category: p.category,
                                pixel_x: p.px,
                                pixel_y: p.py,
                                easting: utm.easting,
                                northing: utm.northing,
                                lat: p.lat,
                                lon: p.lon
                            }});

                            // Update counters
                            const cat = p.category;
                            const prefix = {{ zebra: 'Z', arrow: 'A', parking: 'P', other: 'X' }}[cat] || 'X';
                            const match = p.name.match(new RegExp('^' + prefix + '(\\\\d+)$'));
                            if (match) {{
                                counters[cat] = Math.max(counters[cat], parseInt(match[1]) + 1);
                            }}
                        }});

                        redrawMarkers();
                        updatePointsList();
                        updatePointName();
                        updateTabStatus();
                        updateStatus('Imported ' + data.points.length + ' points from KML');
                        alert('Imported ' + data.points.length + ' points');
                    }} else {{
                        alert('Error importing KML: ' + data.error);
                    }}
                }});
            }};

            reader.readAsText(file);
            input.value = ''; // Reset file input
        }}

        function updateStatus(msg) {{
            document.getElementById('status').textContent = msg;
        }}

        // New GCP Tab Functions

        function captureNewFrame() {{
            updateStatus('Capturing new camera frame...');

            fetch('/api/capture_frame', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    gcpImg.src = 'data:image/jpeg;base64,' + data.camera_frame;
                    cameraParams = data.camera_params;
                    projectedPoints = data.projected_points;

                    // Update projected mask (consistent with switch_to_gcp handler)
                    console.log('Capture frame - projected mask response:', {{
                        available: data.projected_mask_available,
                        hasData: !!data.projected_mask,
                        dataLength: data.projected_mask ? data.projected_mask.length : 0,
                        offset: data.projected_mask_offset
                    }});
                    const captureMaskBtn = document.getElementById('toggle-projected-mask-btn');
                    if (data.projected_mask_available && data.projected_mask) {{
                        projectedMaskData = data.projected_mask;
                        projectedMaskOffset = data.projected_mask_offset;
                        if (captureMaskBtn) {{
                            captureMaskBtn.style.display = 'block';
                        }}
                        // If mask was visible, refresh it with new data
                        if (projectedMaskVisible) {{
                            showProjectedMask();
                        }}
                    }} else {{
                        projectedMaskData = null;
                        projectedMaskOffset = null;
                        hideProjectedMask();
                        if (captureMaskBtn) {{
                            captureMaskBtn.style.display = 'none';
                        }}
                    }}

                    updateCameraParamsDisplay();
                    updateGCPView();

                    // Update camera visualization on Tab 1
                    enableCameraToggle();
                    updateCameraVisualization();

                    // Auto-trigger SAM3 detection for newly captured GCP frame
                    detectFeatures('gcp');

                    updateStatus('Camera frame captured successfully');
                }} else {{
                    updateStatus('Error: ' + data.error);
                    alert('Failed to capture frame: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Capture failed:', err);
                updateStatus('Capture failed: ' + err.message);
            }});
        }}

        function updateCameraParamsDisplay() {{
            if (!cameraParams) return;

            document.getElementById('param-lat').textContent = cameraParams.camera_lat.toFixed(6);
            document.getElementById('param-lon').textContent = cameraParams.camera_lon.toFixed(6);
            document.getElementById('param-pan').textContent = cameraParams.pan_deg.toFixed(1);
            document.getElementById('param-tilt').textContent = cameraParams.tilt_deg.toFixed(1);
            document.getElementById('param-height').textContent = cameraParams.height_m.toFixed(1);
        }}

        // Calibration display and control functions
        let calibrationResult = null;

        function updateCalibrationDisplay(calibration, available) {{
            const resultsDiv = document.getElementById('calibration-results');
            const notAvailableDiv = document.getElementById('calibration-not-available');
            const calibrateBtn = document.querySelector('#calibration-status button');

            if (!available) {{
                if (resultsDiv) resultsDiv.style.display = 'none';
                if (notAvailableDiv) {{
                    notAvailableDiv.style.display = 'block';
                    notAvailableDiv.textContent = 'Calibrator not available (missing dependencies)';
                }}
                if (calibrateBtn) calibrateBtn.style.display = 'none';
                return;
            }}

            if (!calibration) {{
                if (resultsDiv) resultsDiv.style.display = 'none';
                if (notAvailableDiv) {{
                    notAvailableDiv.style.display = 'block';
                    notAvailableDiv.textContent = 'Need at least 4 visible GCP points for calibration';
                }}
                if (calibrateBtn) calibrateBtn.style.display = 'block';
                return;
            }}

            calibrationResult = calibration;

            if (notAvailableDiv) notAvailableDiv.style.display = 'none';
            if (resultsDiv) resultsDiv.style.display = 'block';
            if (calibrateBtn) calibrateBtn.style.display = 'block';

            // Update metrics
            document.getElementById('calibration-rms-error').textContent =
                calibration.rms_error.toFixed(2) + ' px';
            document.getElementById('calibration-max-error').textContent =
                calibration.max_error.toFixed(2) + ' px';
            document.getElementById('calibration-num-points').textContent =
                calibration.num_points;

            // Show observation status (frozen vs fallback)
            const obsStatusEl = document.getElementById('calibration-obs-status');
            if (obsStatusEl) {{
                if (calibration.has_frozen_observations) {{
                    obsStatusEl.innerHTML = `<span style="color: #4CAF50;">✓ Using ${{calibration.frozen_observation_count}} frozen observations</span>`;
                }} else {{
                    obsStatusEl.innerHTML = `<span style="color: #f44336;">⚠ No frozen observations - run SAM3 first!</span>`;
                }}
                obsStatusEl.style.display = 'block';
            }}

            // Update suggested adjustments if available
            const suggestionsDiv = document.getElementById('calibration-suggestions');
            if (calibration.suggested_params && suggestionsDiv) {{
                const sp = calibration.suggested_params;
                let suggestionsHtml = '<div class="calibration-suggestion-header">Suggested Adjustments:</div>';

                // Show delta values with +/- formatting
                if (sp.delta_pan !== undefined) {{
                    const sign = sp.delta_pan >= 0 ? '+' : '';
                    suggestionsHtml += `<div class="suggestion-item">Pan: ${{sign}}${{sp.delta_pan.toFixed(2)}}°</div>`;
                }}
                if (sp.delta_tilt !== undefined) {{
                    const sign = sp.delta_tilt >= 0 ? '+' : '';
                    suggestionsHtml += `<div class="suggestion-item">Tilt: ${{sign}}${{sp.delta_tilt.toFixed(2)}}°</div>`;
                }}
                if (sp.delta_height !== undefined) {{
                    const sign = sp.delta_height >= 0 ? '+' : '';
                    suggestionsHtml += `<div class="suggestion-item">Height: ${{sign}}${{sp.delta_height.toFixed(2)}}m</div>`;
                }}

                suggestionsHtml += '<button onclick="applyCalibration()" class="primary" style="margin-top: 8px;">Apply Suggestions</button>';
                suggestionsDiv.innerHTML = suggestionsHtml;
                suggestionsDiv.style.display = 'block';
            }} else if (suggestionsDiv) {{
                suggestionsDiv.style.display = 'none';
            }}

            // Set color based on RMS error quality
            const rmsEl = document.getElementById('calibration-rms-error');
            if (calibration.rms_error < 5) {{
                rmsEl.style.color = '#4CAF50';  // Green - excellent
            }} else if (calibration.rms_error < 15) {{
                rmsEl.style.color = '#FF9800';  // Orange - acceptable
            }} else {{
                rmsEl.style.color = '#f44336';  // Red - needs improvement
            }}
        }}

        function runCalibration() {{
            updateStatus('Running GCP calibration...');

            fetch('/api/run_calibration', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    updateCalibrationDisplay(data.calibration, true);
                    updateStatus('Calibration complete: RMS error = ' + data.calibration.rms_error.toFixed(2) + ' px');
                }} else {{
                    updateStatus('Calibration failed: ' + data.error);
                    alert('Calibration failed: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Calibration failed:', err);
                updateStatus('Calibration failed: ' + err.message);
            }});
        }}

        let pnpResult = null;

        function runPnPCalibration() {{
            updateStatus('Running PnP calibration (direct pose estimation)...');

            fetch('/api/run_pnp_calibration', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    pnpResult = data;
                    const resultsDiv = document.getElementById('pnp-results');
                    resultsDiv.style.display = 'block';

                    const sp = data.suggested_params;
                    const cp = data.current_params;
                    const ch = data.changes;

                    resultsDiv.innerHTML = `
                        <h4 style="margin: 0 0 8px 0; color: #0ead69;">PnP Calibration Results</h4>
                        <div style="font-size: 11px;">
                            <div><strong>RMS Error:</strong> ${{data.rms_error.toFixed(2)}}px (${{data.num_inliers}}/${{data.num_points}} inliers)</div>
                            <div style="margin-top: 6px;"><strong>Suggested Parameters:</strong></div>
                            <div>Pan: ${{sp.pan?.toFixed(2) || '--'}}° (Δ${{ch.pan >= 0 ? '+' : ''}}${{ch.pan?.toFixed(2) || '0'}}°)</div>
                            <div>Tilt: ${{sp.tilt?.toFixed(2) || '--'}}° (Δ${{ch.tilt >= 0 ? '+' : ''}}${{ch.tilt?.toFixed(2) || '0'}}°)</div>
                            <div>Height: ${{sp.height?.toFixed(2) || '--'}}m (Δ${{ch.height >= 0 ? '+' : ''}}${{ch.height?.toFixed(2) || '0'}}m)</div>
                            <div>Lat: ${{sp.camera_lat?.toFixed(6) || '--'}}°</div>
                            <div>Lon: ${{sp.camera_lon?.toFixed(6) || '--'}}°</div>
                        </div>
                        <button onclick="applyPnPCalibration()" class="primary" style="margin-top: 8px;">Apply PnP Results</button>
                    `;

                    updateStatus('PnP calibration complete: RMS error = ' + data.rms_error.toFixed(2) + ' px');
                }} else {{
                    updateStatus('PnP calibration failed: ' + data.error);
                    alert('PnP calibration failed: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('PnP calibration failed:', err);
                updateStatus('PnP calibration failed: ' + err.message);
            }});
        }}

        function applyPnPCalibration() {{
            if (!pnpResult || !pnpResult.suggested_params) {{
                alert('No PnP calibration results available');
                return;
            }}

            const sp = pnpResult.suggested_params;

            // Build update payload with new absolute values from PnP
            const updateData = {{}};
            if (sp.pan !== undefined && sp.pan !== null) updateData.pan_deg = sp.pan;
            if (sp.tilt !== undefined && sp.tilt !== null) updateData.tilt_deg = sp.tilt;
            if (sp.height !== undefined && sp.height !== null) updateData.height_m = sp.height;
            if (sp.camera_lat !== undefined && sp.camera_lat !== null) updateData.camera_lat = sp.camera_lat;
            if (sp.camera_lon !== undefined && sp.camera_lon !== null) updateData.camera_lon = sp.camera_lon;

            updateStatus('Applying PnP calibration results...');

            fetch('/api/update_camera_params', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(updateData)
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    cameraParams = data.camera_params;
                    projectedPoints = data.projected_points;

                    // Update projected mask if available
                    if (data.projected_mask_available && data.projected_mask) {{
                        projectedMaskData = data.projected_mask;
                        projectedMaskOffset = data.projected_mask_offset;
                        if (projectedMaskVisible) {{
                            showProjectedMask();
                        }}
                    }}

                    updateCameraParamsDisplay();
                    updateGCPView();
                    updateCameraVisualization();

                    // Hide PnP results panel
                    document.getElementById('pnp-results').style.display = 'none';

                    updateStatus('PnP calibration applied successfully!');
                }} else {{
                    updateStatus('Error applying PnP calibration: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Apply PnP calibration failed:', err);
                updateStatus('Apply PnP calibration failed: ' + err.message);
            }});
        }}

        function applyCalibration() {{
            if (!calibrationResult || !calibrationResult.suggested_params) {{
                alert('No calibration suggestions available');
                return;
            }}

            const sp = calibrationResult.suggested_params;

            // Build update payload with new absolute values
            const updateData = {{}};
            if (sp.pan !== undefined) updateData.pan_deg = sp.pan;
            if (sp.tilt !== undefined) updateData.tilt_deg = sp.tilt;
            if (sp.height !== undefined) updateData.height_m = sp.height;

            updateStatus('Applying calibration suggestions...');

            fetch('/api/update_camera_params', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(updateData)
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    cameraParams = data.camera_params;
                    projectedPoints = data.projected_points;

                    // Update projected mask if available
                    if (data.projected_mask_available && data.projected_mask) {{
                        projectedMaskData = data.projected_mask;
                        projectedMaskOffset = data.projected_mask_offset;
                        if (projectedMaskVisible) {{
                            showProjectedMask();
                        }}
                    }}

                    updateCameraParamsDisplay();
                    updateGCPView();
                    updateCameraVisualization();

                    // Re-run calibration to show new error metrics
                    runCalibration();

                    updateStatus('Calibration suggestions applied');
                }} else {{
                    updateStatus('Error applying calibration: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Apply calibration failed:', err);
                updateStatus('Apply calibration failed: ' + err.message);
            }});
        }}

        // Manual matching state
        let matchingMode = false;
        let detectedVertices = [];
        let manualMatches = {{}};  // GCP name -> vertex coords
        let selectedGCP = null;
        let vertexOverlay = null;

        function startManualMatching() {{
            updateStatus('Loading detected vertices...');

            fetch('/api/get_vertices', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    detectedVertices = data.vertices;
                    matchingMode = true;
                    manualMatches = {{}};
                    selectedGCP = null;

                    // Show vertices on canvas
                    createVertexOverlay();
                    drawVertices();

                    // Update button text
                    const btn = document.getElementById('manual-match-btn');
                    btn.textContent = 'Finish Matching';
                    btn.onclick = finishManualMatching;

                    updateStatus(`Manual matching mode: ${{detectedVertices.length}} vertices available. Click a GCP marker, then click a vertex.`);

                    // Show matching instructions
                    alert(`Manual Matching Mode\\n\\n` +
                          `${{detectedVertices.length}} vertices detected.\\n` +
                          `${{data.point_count}} GCP points available.\\n\\n` +
                          `Instructions:\\n` +
                          `1. Click on a GCP marker (colored circle)\\n` +
                          `2. Click on the corresponding vertex (small green dot)\\n` +
                          `3. Repeat for all GCPs you want to match\\n` +
                          `4. Click "Finish Matching" when done`);
                }} else {{
                    updateStatus('Error: ' + data.error);
                    alert('Could not load vertices: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Get vertices failed:', err);
                updateStatus('Get vertices failed: ' + err.message);
            }});
        }}

        function createVertexOverlay() {{
            // Remove existing overlay if any
            if (vertexOverlay) {{
                vertexOverlay.remove();
            }}

            // Create canvas overlay for drawing vertices
            const gcpContainer = document.getElementById('gcp-image-container');
            const gcpImg = document.getElementById('gcp-image');

            vertexOverlay = document.createElement('canvas');
            vertexOverlay.id = 'vertex-overlay';
            vertexOverlay.width = gcpImg.naturalWidth;
            vertexOverlay.height = gcpImg.naturalHeight;
            vertexOverlay.style.position = 'absolute';
            vertexOverlay.style.left = '0';
            vertexOverlay.style.top = '0';
            vertexOverlay.style.width = gcpImg.clientWidth + 'px';
            vertexOverlay.style.height = gcpImg.clientHeight + 'px';
            vertexOverlay.style.pointerEvents = 'auto';
            vertexOverlay.style.cursor = 'crosshair';
            vertexOverlay.style.zIndex = '50';

            gcpContainer.style.position = 'relative';
            gcpContainer.appendChild(vertexOverlay);

            // Add click handler for vertex selection
            vertexOverlay.addEventListener('click', handleMatchingClick);
        }}

        function drawVertices() {{
            if (!vertexOverlay) return;

            const ctx = vertexOverlay.getContext('2d');
            ctx.clearRect(0, 0, vertexOverlay.width, vertexOverlay.height);

            // Draw all vertices as small dots
            ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            ctx.strokeStyle = 'rgba(0, 100, 0, 1)';

            for (const v of detectedVertices) {{
                ctx.beginPath();
                ctx.arc(v.x, v.y, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }}

            // Highlight already matched vertices
            ctx.fillStyle = 'rgba(255, 200, 0, 0.9)';
            ctx.strokeStyle = 'rgba(200, 150, 0, 1)';
            ctx.lineWidth = 2;

            for (const [gcpName, coords] of Object.entries(manualMatches)) {{
                ctx.beginPath();
                ctx.arc(coords.pixel_u, coords.pixel_v, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Draw line to corresponding GCP
                const gcp = projectedPoints.find(p => p.name === gcpName);
                if (gcp) {{
                    ctx.strokeStyle = 'rgba(255, 200, 0, 0.6)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(coords.pixel_u, coords.pixel_v);
                    ctx.lineTo(gcp.pixel_u, gcp.pixel_v);
                    ctx.stroke();
                }}
            }}

            // Highlight selected GCP if any
            if (selectedGCP) {{
                const gcp = projectedPoints.find(p => p.name === selectedGCP);
                if (gcp) {{
                    ctx.strokeStyle = 'rgba(255, 0, 255, 1)';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(gcp.pixel_u, gcp.pixel_v, 15, 0, Math.PI * 2);
                    ctx.stroke();
                }}
            }}
        }}

        function handleMatchingClick(e) {{
            if (!matchingMode) return;

            const rect = vertexOverlay.getBoundingClientRect();
            const scaleX = vertexOverlay.width / rect.width;
            const scaleY = vertexOverlay.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            // Check if clicking on a GCP marker (within 15px radius)
            const clickedGCP = projectedPoints.find(p => {{
                if (!p.visible) return false;
                const dx = p.pixel_u - x;
                const dy = p.pixel_v - y;
                return Math.sqrt(dx*dx + dy*dy) < 15;
            }});

            if (clickedGCP) {{
                // Select this GCP
                selectedGCP = clickedGCP.name;
                updateStatus(`Selected GCP: ${{selectedGCP}}. Now click on the corresponding vertex.`);
                drawVertices();  // Redraw with highlight
                return;
            }}

            // Check if clicking on a vertex (within 8px radius)
            if (selectedGCP) {{
                const clickedVertex = detectedVertices.find(v => {{
                    const dx = v.x - x;
                    const dy = v.y - y;
                    return Math.sqrt(dx*dx + dy*dy) < 8;
                }});

                if (clickedVertex) {{
                    // Match GCP to vertex
                    manualMatches[selectedGCP] = {{
                        pixel_u: clickedVertex.x,
                        pixel_v: clickedVertex.y
                    }};
                    updateStatus(`Matched '${{selectedGCP}}' to vertex at (${{clickedVertex.x.toFixed(0)}}, ${{clickedVertex.y.toFixed(0)}}). ${{Object.keys(manualMatches).length}} matches total.`);
                    selectedGCP = null;
                    drawVertices();  // Redraw with match
                }} else {{
                    updateStatus('No vertex found near click. Try clicking closer to a green dot.');
                }}
            }} else {{
                updateStatus('Click on a GCP marker first (colored circle), then click on a vertex.');
            }}
        }}

        function finishManualMatching() {{
            if (Object.keys(manualMatches).length === 0) {{
                if (!confirm('No matches made. Exit matching mode anyway?')) {{
                    return;
                }}
            }}

            matchingMode = false;

            // Remove vertex overlay
            if (vertexOverlay) {{
                vertexOverlay.remove();
                vertexOverlay = null;
            }}

            // Reset button
            const btn = document.getElementById('manual-match-btn');
            btn.textContent = 'Match Features Manually';
            btn.onclick = startManualMatching;

            // Save matches if any
            if (Object.keys(manualMatches).length > 0) {{
                updateStatus(`Saving ${{Object.keys(manualMatches).length}} manual matches...`);

                fetch('/api/set_manual_matches', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ matches: manualMatches }})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.success) {{
                        updateStatus(`Saved ${{data.num_matches}} matches. Running calibration...`);
                        runCalibration();
                    }} else {{
                        updateStatus('Error saving matches: ' + data.error);
                    }}
                }})
                .catch(err => {{
                    console.error('Save matches failed:', err);
                    updateStatus('Save matches failed: ' + err.message);
                }});
            }} else {{
                updateStatus('Matching mode cancelled. No matches saved.');
            }}
        }}

        function adjustParam(param, delta) {{
            if (!cameraParams) {{
                updateStatus('No camera parameters available');
                return;
            }}

            // Map parameter names to their storage keys
            function getParamKey(p) {{
                if (p === 'height') return 'height_m';
                if (p === 'camera_lat' || p === 'camera_lon') return p;
                return p + '_deg';  // pan, tilt
            }}

            const paramKey = getParamKey(param);
            const newValue = cameraParams[paramKey] + delta;

            // Use appropriate precision for display
            const precision = (param === 'camera_lat' || param === 'camera_lon') ? 6 : 1;
            updateStatus(`Adjusting ${{param}}: ${{newValue.toFixed(precision)}}...`);

            const updateData = {{}};
            updateData[paramKey] = newValue;

            fetch('/api/update_camera_params', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(updateData)
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    cameraParams = data.camera_params;
                    projectedPoints = data.projected_points;

                    // Update projected mask
                    if (data.projected_mask_available && data.projected_mask) {{
                        projectedMaskData = data.projected_mask;
                        projectedMaskOffset = data.projected_mask_offset;
                        // If mask was visible, refresh it with new data
                        if (projectedMaskVisible) {{
                            showProjectedMask();
                        }}
                    }}

                    updateCameraParamsDisplay();
                    updateGCPView();

                    // Update camera visualization on Tab 1 (real-time sync)
                    updateCameraVisualization();

                    updateStatus(`${{param}} adjusted to ${{newValue.toFixed(1)}}`);
                }} else {{
                    updateStatus('Error: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Adjustment failed:', err);
                updateStatus('Adjustment failed: ' + err.message);
            }});
        }}

        function detectFeatures(tab) {{
            const promptInput = document.getElementById('sam3-prompt-' + tab);
            const defaultPrompt = tab === 'gcp' ? 'road marking' : 'ground markings';
            const prompt = promptInput.value.trim() || defaultPrompt;

            updateStatus('Detecting features with SAM3...');

            fetch('/api/detect_features', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    method: 'sam3',
                    prompt: prompt,
                    preprocessing: document.getElementById('sam3-preprocessing-' + tab).value,
                    tab: tab
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    maskData[tab] = data.mask;

                    // Show toggle button
                    document.getElementById('toggle-mask-' + tab + '-btn').style.display = 'block';

                    // Show manual matching button for GCP tab
                    if (tab === 'gcp') {{
                        document.getElementById('manual-match-btn').style.display = 'block';
                    }}

                    // Auto-show mask
                    maskVisible[tab] = true;
                    showMask(tab);

                    const meta = data.metadata;
                    updateStatus(`Detected ${{meta.total_predictions}} features (${{meta.total_polygons}} polygons)`);
                }} else {{
                    updateStatus('Error: ' + data.error);
                    alert('Feature detection failed: ' + data.error);
                }}
            }})
            .catch(err => {{
                console.error('Detection failed:', err);
                updateStatus('Detection failed: ' + err.message);
            }});
        }}

        function toggleMask(tab) {{
            maskVisible[tab] = !maskVisible[tab];

            if (maskVisible[tab]) {{
                showMask(tab);
            }} else {{
                hideMask(tab);
            }}

            document.getElementById('toggle-mask-' + tab + '-btn').textContent =
                maskVisible[tab] ? 'Hide Mask' : 'Show Mask';
        }}

        function showMask(tab) {{
            const targetContainer = tab === 'gcp' ? gcpContainer : container;
            const targetImg = tab === 'gcp' ? gcpImg : img;

            // Remove existing mask (not the projected one)
            const existingMask = targetContainer.querySelector('.mask-overlay:not(.projected)');
            if (existingMask) existingMask.remove();

            if (!maskData[tab]) return;

            // Create mask overlay with camera class for GCP tab
            const maskImg = document.createElement('img');
            maskImg.className = 'mask-overlay' + (tab === 'gcp' ? ' camera' : '');
            maskImg.src = 'data:image/png;base64,' + maskData[tab];
            const tabZoom = tab === 'gcp' ? gcpZoom : kmlZoom;
            maskImg.style.width = (targetImg.naturalWidth * tabZoom) + 'px';
            maskImg.style.height = (targetImg.naturalHeight * tabZoom) + 'px';

            targetContainer.appendChild(maskImg);
        }}

        function hideMask(tab) {{
            const targetContainer = tab === 'gcp' ? gcpContainer : container;
            const maskOverlay = targetContainer.querySelector('.mask-overlay:not(.projected)');
            if (maskOverlay) maskOverlay.remove();
        }}

        // Projected mask functions (Map mask projected onto camera view)
        function toggleProjectedMask() {{
            projectedMaskVisible = !projectedMaskVisible;

            if (projectedMaskVisible) {{
                showProjectedMask();
            }} else {{
                hideProjectedMask();
            }}

            document.getElementById('toggle-projected-mask-btn').textContent =
                projectedMaskVisible ? 'Hide Map Mask' : 'Show Map Mask';
        }}

        function showProjectedMask() {{
            // Remove existing projected mask
            const existingMask = gcpContainer.querySelector('.mask-overlay.projected');
            if (existingMask) existingMask.remove();

            if (!projectedMaskData) {{
                console.warn('showProjectedMask: No mask data available');
                return;
            }}

            console.log('showProjectedMask: Creating mask overlay');
            console.log('  gcpImg dimensions:', gcpImg.naturalWidth, 'x', gcpImg.naturalHeight);
            console.log('  gcpZoom:', gcpZoom);
            console.log('  projectedMaskOffset:', projectedMaskOffset);

            // Create projected mask overlay
            const maskImg = document.createElement('img');
            maskImg.className = 'mask-overlay projected';

            maskImg.onload = function() {{
                console.log('Projected mask loaded:', maskImg.naturalWidth, 'x', maskImg.naturalHeight);

                // Scale mask by zoom
                const scaledMaskWidth = maskImg.naturalWidth * gcpZoom;
                const scaledMaskHeight = maskImg.naturalHeight * gcpZoom;

                maskImg.style.width = scaledMaskWidth + 'px';
                maskImg.style.height = scaledMaskHeight + 'px';

                // The offset tells us where camera (0,0) is within the mask canvas
                // projectedMaskOffset = (-offset_x, -offset_y) where offset_x/y is camera position in mask
                // So to position the mask relative to camera at (0,0), we use these values directly
                if (projectedMaskOffset) {{
                    const offsetLeft = projectedMaskOffset[0] * gcpZoom;
                    const offsetTop = projectedMaskOffset[1] * gcpZoom;

                    maskImg.style.left = offsetLeft + 'px';
                    maskImg.style.top = offsetTop + 'px';

                    console.log('  Mask positioned at: (' + offsetLeft + ', ' + offsetTop + ')');
                }}

                console.log('  Final mask dimensions:', scaledMaskWidth + 'x' + scaledMaskHeight);
            }};

            maskImg.onerror = function() {{
                console.error('Failed to load projected mask image');
            }};

            maskImg.src = 'data:image/png;base64,' + projectedMaskData;

            gcpContainer.appendChild(maskImg);
            console.log('Projected mask appended to container');
        }}

        function hideProjectedMask() {{
            const maskOverlay = gcpContainer.querySelector('.mask-overlay.projected');
            if (maskOverlay) maskOverlay.remove();
        }}

        // Camera Visualization Functions

        function latLonToPixelProper(lat, lon) {{
            // Convert lat/lon to UTM first, then to pixel
            // UTM conversion using pyproj-like calculation (simplified)
            // Uses the existing config georeferencing parameters

            // Simple approximation for UTM zone 30N (EPSG:25830)
            // More accurate: use proper UTM conversion
            const latRad = lat * Math.PI / 180;
            const lonRad = lon * Math.PI / 180;

            // UTM Zone 30N central meridian is -3°
            const centralMeridian = -3 * Math.PI / 180;

            // WGS84 parameters
            const a = 6378137.0;  // semi-major axis
            const f = 1/298.257223563;  // flattening
            const k0 = 0.9996;  // scale factor

            const e2 = 2*f - f*f;
            const e_prime2 = e2 / (1 - e2);

            const N = a / Math.sqrt(1 - e2 * Math.sin(latRad) * Math.sin(latRad));
            const T = Math.tan(latRad) * Math.tan(latRad);
            const C = e_prime2 * Math.cos(latRad) * Math.cos(latRad);
            const A = Math.cos(latRad) * (lonRad - centralMeridian);

            const M = a * ((1 - e2/4 - 3*e2*e2/64 - 5*e2*e2*e2/256) * latRad
                        - (3*e2/8 + 3*e2*e2/32 + 45*e2*e2*e2/1024) * Math.sin(2*latRad)
                        + (15*e2*e2/256 + 45*e2*e2*e2/1024) * Math.sin(4*latRad)
                        - (35*e2*e2*e2/3072) * Math.sin(6*latRad));

            const easting = k0 * N * (A + (1-T+C)*A*A*A/6
                        + (5-18*T+T*T+72*C-58*e_prime2)*A*A*A*A*A/120) + 500000;

            const northing = k0 * (M + N * Math.tan(latRad) * (A*A/2
                        + (5-T+9*C+4*C*C)*A*A*A*A/24
                        + (61-58*T+T*T+600*C-330*e_prime2)*A*A*A*A*A*A/720));

            // Convert UTM to pixel coordinates using inverse affine transform
            const gt = config.geotransform;
            const det = gt[1] * gt[5] - gt[2] * gt[4];
            if (Math.abs(det) < 1e-10) {{
                console.error('Geotransform matrix is singular');
                return {{ x: 0, y: 0 }};
            }}
            const de = easting - gt[0];
            const dn = northing - gt[3];
            const pixelX = (gt[5] * de - gt[2] * dn) / det;
            const pixelY = (-gt[4] * de + gt[1] * dn) / det;

            return {{ x: pixelX, y: pixelY }};
        }}

        // Center KML tab viewport on camera position
        function centerOnCamera() {{
            if (!cameraParams || !cameraParams.camera_lat || !cameraParams.camera_lon) {{
                return; // No camera position available
            }}

            const panel = document.getElementById('kml-image-panel');
            if (!panel) return;

            const cameraPixel = latLonToPixelProper(cameraParams.camera_lat, cameraParams.camera_lon);

            // Apply current zoom
            const scaledX = cameraPixel.x * kmlZoom;
            const scaledY = cameraPixel.y * kmlZoom;

            // Calculate center offset
            const scrollLeft = scaledX - (panel.clientWidth / 2);
            const scrollTop = scaledY - (panel.clientHeight / 2);

            // Apply scroll position
            panel.scrollLeft = Math.max(0, scrollLeft);
            panel.scrollTop = Math.max(0, scrollTop);
        }}

        function updateCameraVisualization() {{
            // Remove existing camera visualization elements
            const existingDot = container.querySelector('.camera-position-dot');
            const existingPolygon = container.querySelector('.camera-footprint-polygon');
            if (existingDot) existingDot.remove();
            if (existingPolygon) existingPolygon.remove();

            // Check if camera params are available
            if (!cameraParams || !cameraOverlayVisible) {{
                return;
            }}

            const cameraLat = cameraParams.camera_lat;
            const cameraLon = cameraParams.camera_lon;
            const footprint = cameraParams.camera_footprint;

            if (!cameraLat || !cameraLon) {{
                return;
            }}

            // Draw camera position dot
            const cameraPixel = latLonToPixelProper(cameraLat, cameraLon);
            const dotX = cameraPixel.x * kmlZoom;
            const dotY = cameraPixel.y * kmlZoom;

            const dot = document.createElement('div');
            dot.className = 'camera-position-dot';
            dot.style.left = dotX + 'px';
            dot.style.top = dotY + 'px';
            dot.title = `Camera: ${{cameraLat.toFixed(6)}}, ${{cameraLon.toFixed(6)}}`;
            container.appendChild(dot);

            // Draw camera footprint polygon if available
            if (footprint && footprint.length === 4) {{
                // Convert footprint corners to pixel coordinates
                const pixelCorners = footprint.map(corner => {{
                    const pixel = latLonToPixelProper(corner.lat, corner.lon);
                    return {{ x: pixel.x * kmlZoom, y: pixel.y * kmlZoom }};
                }});

                // Validate all coordinates are finite (not NaN or Infinity)
                const allValid = pixelCorners.every(p =>
                    Number.isFinite(p.x) && Number.isFinite(p.y)
                );

                if (allValid) {{
                    // Create SVG for polygon
                    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                    svg.setAttribute('class', 'camera-footprint-polygon');
                    svg.style.width = (img.naturalWidth * kmlZoom) + 'px';
                    svg.style.height = (img.naturalHeight * kmlZoom) + 'px';

                    const points = pixelCorners.map(p => `${{p.x}},${{p.y}}`).join(' ');

                    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    polygon.setAttribute('points', points);
                    polygon.setAttribute('fill', 'rgba(74, 144, 226, 0.15)');
                    polygon.setAttribute('stroke', '#4A90E2');
                    polygon.setAttribute('stroke-width', '3');

                    svg.appendChild(polygon);
                    container.appendChild(svg);
                }}
            }}
        }}

        function toggleCameraOverlay() {{
            const checkbox = document.getElementById('camera-overlay-toggle');
            cameraOverlayVisible = checkbox.checked;
            updateCameraVisualization();
            updateStatus('Camera visualization ' + (cameraOverlayVisible ? 'shown' : 'hidden'));
        }}

        function enableCameraToggle() {{
            // Update status when live camera data is captured
            const statusDiv = document.getElementById('camera-status');
            statusDiv.textContent = 'Camera position and footprint visible on map';
        }}

        // Attach toggle event listener immediately
        document.getElementById('camera-overlay-toggle').addEventListener('change', toggleCameraOverlay);
    </script>
</body>
</html>'''


def run_server(session: UnifiedSession, port: int = 8765):
    """Run the unified web server."""
    import atexit
    import shutil

    # Create temp directory for serving frame
    temp_dir = tempfile.mkdtemp(prefix='unified_gcp_')
    frame_path = os.path.join(temp_dir, 'frame.jpg')
    cv2.imwrite(frame_path, session.frame)

    # Register cleanup handler for temp directory
    def cleanup_temp():
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    atexit.register(cleanup_temp)

    # Set up handler
    UnifiedHTTPHandler.session = session
    UnifiedHTTPHandler.temp_dir = temp_dir

    # Pre-capture camera frame before browser opens (Bug 1 fix)
    # This eliminates delay when user first switches to GCP Capture tab
    if GEOMETRY_AVAILABLE and INTRINSICS_AVAILABLE:
        try:
            from tools.capture_gcps_web import grab_frame_from_camera, get_camera_params_for_projection
            print("Pre-capturing camera frame...")
            frame, ptz_status = grab_frame_from_camera(session.camera_name, wait_time=2.0)
            frame_height, frame_width = frame.shape[:2]
            camera_params = get_camera_params_for_projection(
                session.camera_name,
                image_width=frame_width,
                image_height=frame_height
            )

            # Store distortion coefficients for use in calibration (no frame undistortion)
            # Issue #135: Work in distorted image space for consistent calibration
            K = camera_params.get('K')
            if K is not None:
                # Get distortion coefficients from camera config
                dist_coeffs = np.zeros(5)  # [k1, k2, p1, p2, k3]
                try:
                    cam_config = get_camera_by_name(session.camera_name)
                    if cam_config:
                        k1 = cam_config.get('k1', 0.0)
                        k2 = cam_config.get('k2', 0.0)
                        p1 = cam_config.get('p1', 0.0)
                        p2 = cam_config.get('p2', 0.0)
                        k3 = cam_config.get('k3', 0.0)
                        dist_coeffs = np.array([k1, k2, p1, p2, k3])
                except Exception as e:
                    print(f"Warning: Could not load distortion coefficients: {e}")

                if np.any(dist_coeffs != 0):
                    print(f"Distortion coefficients loaded: k1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}, p1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}, k3={dist_coeffs[4]:.6f}")
                    print("Frame NOT undistorted - calibration will use distorted image space")
                else:
                    print("No distortion coefficients - using zero distortion")

                camera_params['dist_coeffs'] = dist_coeffs.tolist()

            # Always set undistorted=False regardless of K availability
            # Issue #135: Calibration works in distorted image space
            camera_params['undistorted'] = False

            session.camera_frame = frame
            session.camera_params = camera_params
            print(f"Camera frame captured: {frame_width}x{frame_height}")
            print(f"Camera params: pan={camera_params['pan_deg']:.1f}°, tilt={camera_params['tilt_deg']:.1f}°, zoom={camera_params['zoom']:.1f}x")
        except Exception as e:
            print(f"Warning: Failed to pre-capture camera frame: {e}")
            import traceback
            traceback.print_exc()
            print("Camera frame will be captured when switching to GCP Capture tab.")

    # Find available port
    port = find_available_port(start_port=port, max_attempts=10)

    with socketserver.TCPServer(("", port), UnifiedHTTPHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n{'='*60}")
        print(f"Unified GCP Tool running at: {url}")
        print(f"Camera: {session.camera_name}")
        print(f"Image: {session.image_path}")
        print(f"Size: {session.frame_width}x{session.frame_height}")
        print(f"CRS: {session.utm_crs}")
        gt = session.geotiff_params['geotransform']
        print(f"Geotransform: {gt}")
        print(f"GSD: {abs(gt[1])}m")
        print(f"{'='*60}")
        print("\nTab 1: KML Extractor - Click to add points")
        print("Tab 2: GCP Capture - View projected points (enabled after adding points)")
        print("\nPress Ctrl+C to stop\n")

        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            cleanup_temp()


def main():
    parser = argparse.ArgumentParser(
        description='Unified GCP Tool - KML Extraction + GCP Capture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'image',
        help='Path to the georeferenced image'
    )
    parser.add_argument(
        '--camera',
        type=str,
        required=True,
        help='Camera name to load configuration from (e.g., Valte)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    parser.add_argument(
        '--kml',
        type=str,
        help='Path to KML file to pre-load points on startup'
    )

    args = parser.parse_args()

    # Validate image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Validate KML file exists (if provided)
    if args.kml and not os.path.exists(args.kml):
        print(f"Error: KML file not found: {args.kml}")
        sys.exit(1)

    # Load camera configuration
    camera_config = get_camera_by_name(args.camera)
    if camera_config is None:
        print(f"Error: Camera '{args.camera}' not found in configuration.")
        available = [c['name'] for c in get_camera_configs()]
        print(f"Available cameras: {', '.join(available)}")
        sys.exit(1)

    # Check for geotiff_params
    if 'geotiff_params' not in camera_config:
        print(f"Error: Camera '{args.camera}' does not have 'geotiff_params' defined.")
        print(f"Please update the camera configuration in poc_homography/camera_config.py")
        sys.exit(1)

    geotiff_params = camera_config['geotiff_params']

    print(f"Loaded configuration for camera: {args.camera}")
    print(f"  UTM CRS: {geotiff_params['utm_crs']}")
    if 'geotransform' in geotiff_params:
        gt = geotiff_params['geotransform']
        print(f"  Geotransform: {gt}")
        print(f"  Origin: E {gt[0]}, N {gt[3]}")
        print(f"  Pixel size: {gt[1]} x {gt[5]} m")
    else:
        print(f"  Origin: E {geotiff_params['origin_easting']}, N {geotiff_params['origin_northing']}")
        print(f"  Pixel size: {geotiff_params['pixel_size_x']} x {geotiff_params['pixel_size_y']} m")

    # Create unified session
    try:
        session = UnifiedSession(args.image, camera_config, geotiff_params)
    except Exception as e:
        print(f"Error creating session: {e}")
        sys.exit(1)

    # Pre-load KML points if --kml argument provided
    if args.kml:
        try:
            with open(args.kml, 'r', encoding='utf-8') as f:
                kml_text = f.read()
            points = session.parse_kml(kml_text)
            for p in points:
                session.add_point(p['px'], p['py'], p['name'], p['category'])
            if len(points) == 0:
                print(f"Warning: KML file contains no valid points: {args.kml}")
            else:
                print(f"Pre-loaded {len(points)} points from KML: {args.kml}")
        except ET.ParseError as e:
            print(f"Error: Malformed KML file: {args.kml}")
            print(f"  XML parsing error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading KML file: {args.kml}")
            print(f"  {e}")
            sys.exit(1)

    # Run server
    run_server(session, args.port)


if __name__ == '__main__':
    main()
