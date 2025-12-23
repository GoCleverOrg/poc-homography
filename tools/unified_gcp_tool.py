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

# Import for map-first mode projection
try:
    from poc_homography.camera_geometry import CameraGeometry
    from poc_homography.gps_distance_calculator import dms_to_dd
    from poc_homography.coordinate_converter import UTMConverter
    from poc_homography.auto_calibrator import AutoCalibrator
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

# Import intrinsics utility
try:
    from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics
    INTRINSICS_AVAILABLE = True
except ImportError:
    INTRINSICS_AVAILABLE = False

# SAM3 detection prompt for ground markings
# Testing showed "ground markings" detects both road lines AND parking spot lines with
# best overall results: 12.12% coverage, 0.699 confidence on sample cartography images.
# See tools/test_sam3_prompts.py for the testing script.
DEFAULT_SAM3_PROMPT = "ground markings"

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
        self.geotiff_params = geotiff_params

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
        """Convert pixel coordinates to UTM."""
        easting = self.geotiff_params['origin_easting'] + (px * self.geotiff_params['pixel_size_x'])
        northing = self.geotiff_params['origin_northing'] + (py * self.geotiff_params['pixel_size_y'])
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
        """Convert lat/lon to pixel coordinates."""
        easting, northing = self.latlon_to_utm(lat, lon)
        px = (easting - self.geotiff_params['origin_easting']) / self.geotiff_params['pixel_size_x']
        py = (northing - self.geotiff_params['origin_northing']) / self.geotiff_params['pixel_size_y']
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

        Returns:
            List of 4 dicts with 'lat' and 'lon' keys representing footprint corners,
            ordered: top-left, top-right, bottom-right, bottom-left.
            Returns None if camera_params are not available or projection fails.
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

            footprint = []
            for u, v in corners:
                # Project pixel to world coordinates (relative to camera position)
                pt_img = np.array([[u], [v], [1.0]])
                pt_world = geo.H_inv @ pt_img

                # Check for valid projection (not near horizon)
                if abs(pt_world[2, 0]) < 1e-6:
                    # Point is near horizon, projection is invalid
                    return None

                # Normalize homogeneous coordinates
                # H_inv returns world coordinates: X=East, Y=North (pan is already in the homography)
                # Bug fix: Removed erroneous pan rotation that was double-applying pan
                east_offset = pt_world[0, 0] / pt_world[2, 0]   # X = East
                north_offset = pt_world[1, 0] / pt_world[2, 0]  # Y = North

                # Convert world offset (meters) to lat/lon
                # Use approximate conversion: 1 degree lat ≈ 111320m
                lat_offset_deg = north_offset / 111320.0
                lon_offset_deg = east_offset / (111320.0 * np.cos(np.radians(camera_lat)))

                corner_lat = camera_lat + lat_offset_deg
                corner_lon = camera_lon + lon_offset_deg

                footprint.append({
                    'lat': corner_lat,
                    'lon': corner_lon
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
            distortion_applied = False
            if self.camera_name:
                try:
                    cam_config = get_camera_by_name(self.camera_name)
                    if cam_config:
                        k1 = cam_config.get('k1', 0.0)
                        k2 = cam_config.get('k2', 0.0)
                        p1 = cam_config.get('p1', 0.0)
                        p2 = cam_config.get('p2', 0.0)
                        # Only apply if non-zero coefficients exist
                        if k1 != 0.0 or k2 != 0.0 or p1 != 0.0 or p2 != 0.0:
                            geo.set_distortion_coefficients(k1=k1, k2=k2, p1=p1, p2=p2)
                            distortion_applied = True
                except Exception as e:
                    print(f"Warning: Could not load distortion coefficients: {e}")

            # =================================================================
            # COMPOSE THE HOMOGRAPHY MATHEMATICALLY (not from sampled points)
            # =================================================================
            # The transformation chain is:
            #   Cartography pixels → UTM → Local XY → Camera pixels
            #
            # This can be expressed as: H_total = H_camera @ T_pixel_to_localXY
            # where T_pixel_to_localXY is an affine transformation (3x3 matrix)
            # =================================================================

            # Get camera's UTM coordinates
            camera_easting, camera_northing = utm_converter.gps_to_utm(camera_lat, camera_lon)
            print(f"Camera UTM: E={camera_easting:.2f}, N={camera_northing:.2f}")

            # Build T_pixel_to_localXY: transforms cartography pixels to local XY
            # Cartography pixel (px, py) → Local XY (x, y):
            #   easting = origin_easting + px * pixel_size_x
            #   northing = origin_northing + py * pixel_size_y
            #   x = easting - camera_easting
            #   y = northing - camera_northing
            #
            # Combined:
            #   x = pixel_size_x * px + (origin_easting - camera_easting)
            #   y = pixel_size_y * py + (origin_northing - camera_northing)
            #
            # In homogeneous coordinates:
            #   [x]   [pixel_size_x  0            dx] [px]
            #   [y] = [0             pixel_size_y dy] [py]
            #   [1]   [0             0            1 ] [1 ]
            # where dx = origin_easting - camera_easting
            #       dy = origin_northing - camera_northing

            pixel_size_x = self.geotiff_params['pixel_size_x']
            pixel_size_y = self.geotiff_params['pixel_size_y']
            origin_easting = self.geotiff_params['origin_easting']
            origin_northing = self.geotiff_params['origin_northing']

            dx = origin_easting - camera_easting
            dy = origin_northing - camera_northing

            T_pixel_to_localXY = np.array([
                [pixel_size_x, 0,            dx],
                [0,            pixel_size_y, dy],
                [0,            0,            1 ]
            ], dtype=np.float64)

            print(f"T_pixel_to_localXY:")
            print(f"  pixel_size: ({pixel_size_x}, {pixel_size_y})")
            print(f"  translation: ({dx:.2f}, {dy:.2f})")

            # Compose the total homography: H_total = H_camera @ T_pixel_to_localXY
            # This maps cartography pixels directly to camera pixels
            H_total = geo.H @ T_pixel_to_localXY

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
                'projection_error': projection_error
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

                # Use default prompt or custom prompt
                prompt = post_data.get('prompt', DEFAULT_SAM3_PROMPT)

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

                self.send_json_response({
                    'success': True,
                    'mask': mask_base64,
                    'metadata': {
                        'total_predictions': total_predictions,
                        'total_polygons': total_polygons,
                        'confidence_range': {
                            'min': min(confidence_scores) if confidence_scores else 0,
                            'max': max(confidence_scores) if confidence_scores else 0
                        }
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

        elif self.path == '/api/auto_calibrate':
            # Auto-calibration via mask matching
            try:
                # Validate cartography mask exists (use original unprojected mask)
                if self.session.cartography_mask is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'Cartography mask not detected. Please detect cartography mask first.'
                    })
                    return

                if self.session.camera_mask is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'Camera mask not detected. Please detect camera features first.'
                    })
                    return

                if self.session.camera_params is None:
                    self.send_json_response({
                        'success': False,
                        'error': 'Camera parameters not available.'
                    })
                    return

                if not GEOMETRY_AVAILABLE:
                    self.send_json_response({
                        'success': False,
                        'error': 'AutoCalibrator not available. Camera geometry module required.'
                    })
                    return

                # Create CameraGeometry from session camera_params
                # Camera position is at origin (0, 0) in local coordinate system
                image_width = self.session.camera_params.get('image_width', 1920)
                image_height = self.session.camera_params.get('image_height', 1080)
                geo = CameraGeometry(image_width, image_height)
                geo.set_camera_parameters(
                    K=self.session.camera_params['K'],
                    w_pos=np.array([
                        0.0,
                        0.0,
                        self.session.camera_params['height_m']
                    ]),
                    pan_deg=self.session.camera_params['pan_deg'],
                    tilt_deg=self.session.camera_params['tilt_deg'],
                    map_width=640,
                    map_height=640
                )

                # Store status messages
                status_messages = []
                def status_callback(msg):
                    status_messages.append(msg)
                    print(f"[AutoCalibrator] {msg}")

                # Get camera UTM coordinates for geotiff transformation
                camera_lat = self.session.camera_params['camera_lat']
                camera_lon = self.session.camera_params['camera_lon']
                utm_converter = UTMConverter(self.session.utm_crs)
                utm_converter.set_reference(camera_lat, camera_lon)
                camera_easting, camera_northing = utm_converter.gps_to_utm(camera_lat, camera_lon)

                # Create AutoCalibrator and run optimization
                # Pass original unprojected mask + geotiff params for complete transformation
                calibrator = AutoCalibrator(
                    camera_geometry=geo,
                    map_mask=self.session.cartography_mask,  # Original unprojected mask
                    camera_mask=self.session.camera_mask,
                    callback=status_callback,
                    geotiff_params=self.session.geotiff_params,  # Enable geotiff transformation
                    camera_utm_position=(camera_easting, camera_northing)  # For T_pixel_to_localXY
                )

                result = calibrator.run()

                # Helper function to convert numpy arrays to JSON-serializable format
                def serialize_params(params):
                    """Convert numpy arrays in params dict to lists for JSON serialization."""
                    serialized = {}
                    for key, value in params.items():
                        if isinstance(value, np.ndarray):
                            serialized[key] = value.tolist()
                        elif isinstance(value, (np.float32, np.float64)):
                            serialized[key] = float(value)
                        elif isinstance(value, (np.int32, np.int64)):
                            serialized[key] = int(value)
                        else:
                            serialized[key] = value
                    return serialized

                # Format response
                response = {
                    'success': result.success,
                    'initial_correlation': float(result.initial_correlation),
                    'final_correlation': float(result.final_correlation),
                    'improvement': float(result.improvement),
                    'relative_improvement': float(result.relative_improvement),
                    'elapsed_time': float(result.elapsed_time),
                    'steps_completed': result.steps_completed,
                    'timeout_reached': result.timeout_reached,
                    'message': result.message,
                    'original_params': serialize_params(result.original_params),
                    'optimized_params': serialize_params(result.optimized_params),
                    'status_log': status_messages
                }

                self.send_json_response(response)

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Auto-calibration error: {error_details}")
                self.send_json_response({
                    'success': False,
                    'error': f'Auto-calibration failed: {str(e)}'
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
        self.session.camera_frame = frame

        # Get camera parameters for projection
        frame_height, frame_width = frame.shape[:2]
        camera_params = get_camera_params_for_projection(
            self.session.camera_name,
            image_width=frame_width,
            image_height=frame_height
        )
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

                    <h3>Auto-Calibration</h3>
                    <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
                        Automatically calibrate camera parameters by matching masks
                    </div>
                    <button onclick="autoCalibrate()" id="auto-calibrate-btn" style="display: none;">Auto-Calibrate</button>
                    <div id="auto-calibrate-status" style="font-size: 11px; color: #666; margin-top: 5px;"></div>

                    <h3>SAM3 Feature Detection</h3>
                    <label>Preprocessing:</label>
                    <select id="sam3-preprocessing-gcp">
                        <option value="none">None</option>
                        <option value="clahe" selected>CLAHE (Recommended)</option>
                    </select>
                    
                    <label>Prompt:</label>
                    <input type="text" id="sam3-prompt-gcp" placeholder="ground markings" value="ground markings">
                    <button onclick="detectFeatures('gcp')" class="secondary">Detect Features</button>
                    <button onclick="toggleMask('gcp')" id="toggle-mask-gcp-btn" style="display: none;">Toggle Camera Mask</button>

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
        let kmlAutoDetected = false;  // Guard flag: auto-detection already triggered for KML tab
        let gcpAutoDetected = false;  // Guard flag: auto-detection already triggered for GCP tab

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
            // Auto-trigger SAM3 detection for KML tab
            if (!kmlAutoDetected) {{
                kmlAutoDetected = true;
                detectFeatures('kml');
            }}
        }};

        // Also check if image is already loaded (base64 inline images load synchronously)
        if (img.complete && img.naturalWidth > 0) {{
            if (cameraParams) {{
                enableCameraToggle();
                updateCameraVisualization();
                centerOnCamera();
            }}
            // Auto-trigger SAM3 detection for KML tab (synchronous load)
            if (!kmlAutoDetected) {{
                kmlAutoDetected = true;
                detectFeatures('kml');
            }}
        }}

        // Initialize
        updatePointName();
        document.getElementById('category').addEventListener('change', updatePointName);
        updateAutoCalibrateBtnVisibility();  // Set initial auto-calibrate button state

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


                            // Auto-trigger SAM3 detection for GCP tab
                            if (!gcpAutoDetected) {{
                                gcpAutoDetected = true;
                                detectFeatures('gcp');
                            }}
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

                    // Update auto-calibrate button visibility when projected mask changes
                    updateAutoCalibrateBtnVisibility();

                    updateCameraParamsDisplay();

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

            // Clear existing markers
            document.querySelectorAll('.gcp-marker').forEach(m => m.remove());

            if (projectedPoints.length === 0) {{
                listContainer.innerHTML = '<div style="color: #888; font-size: 12px;">No points to display</div>';
                return;
            }}

            // Draw markers
            projectedPoints.forEach(pt => {{
                if (pt.visible) {{
                    const marker = document.createElement('div');
                    marker.className = 'gcp-marker';
                    marker.style.left = (pt.pixel_u * gcpZoom) + 'px';
                    marker.style.top = (pt.pixel_v * gcpZoom) + 'px';
                    marker.title = pt.name;
                    gcpContainer.appendChild(marker);
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
            const easting = config.origin_easting + (px * config.pixel_size_x);
            const northing = config.origin_northing + (py * config.pixel_size_y);
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
            const prompt = promptInput.value.trim() || 'ground markings';

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

                    // Auto-show mask
                    maskVisible[tab] = true;
                    showMask(tab);

                    // Update auto-calibrate button visibility when camera mask is detected
                    if (tab === 'gcp') {{
                        updateAutoCalibrateBtnVisibility();
                    }}

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

        function updateAutoCalibrateBtnVisibility() {{
            // Show auto-calibrate button when BOTH masks are available:
            // 1. projectedMaskData (map mask projected to camera)
            // 2. maskData.gcp (camera mask detected via SAM3)
            const btn = document.getElementById('auto-calibrate-btn');
            if (btn) {{
                const bothMasksAvailable = projectedMaskData && maskData.gcp;
                btn.style.display = bothMasksAvailable ? 'block' : 'none';
                console.log('Auto-calibrate button visibility:', {{
                    projectedMaskData: !!projectedMaskData,
                    cameraMask: !!maskData.gcp,
                    visible: bothMasksAvailable
                }});
            }}
        }}

        function autoCalibrate() {{
            const btn = document.getElementById('auto-calibrate-btn');
            const statusDiv = document.getElementById('auto-calibrate-status');

            // Disable button and show status
            btn.disabled = true;
            statusDiv.textContent = 'Starting auto-calibration...';
            updateStatus('Running auto-calibration...');

            fetch('/api/auto_calibrate', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{}})
            }})
            .then(r => r.json())
            .then(data => {{
                btn.disabled = false;
                statusDiv.textContent = '';

                if (data.success) {{
                    const before = (data.initial_correlation * 100).toFixed(1);
                    const after = (data.final_correlation * 100).toFixed(1);
                    const improvement = (data.relative_improvement * 100).toFixed(1);

                    // Show confirmation dialog
                    let message = `Auto-calibration Complete\\n\\n`;
                    message += `Correlation Score:\\n`;
                    message += `  Before: ${{before}}%\\n`;
                    message += `  After: ${{after}}%\\n`;
                    message += `  Improvement: ${{improvement > 0 ? '+' : ''}}${{improvement}}%\\n\\n`;
                    message += `Time: ${{data.elapsed_time.toFixed(1)}}s\\n`;
                    message += `Steps: ${{data.steps_completed}}/5\\n\\n`;

                    if (data.improvement < 0) {{
                        message += `WARNING: Correlation decreased!\\n\\n`;
                    }}

                    message += `Accept new parameters?`;

                    const accepted = confirm(message);

                    if (accepted) {{
                        // Update camera params with optimized values
                        const optimized = data.optimized_params;

                        // Send update request with new parameters
                        fetch('/api/update_camera_params', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                pan_deg: optimized.pan_deg,
                                tilt_deg: optimized.tilt_deg,
                                height_m: optimized.height_m
                            }})
                        }})
                        .then(r => r.json())
                        .then(updateData => {{
                            if (updateData.success) {{
                                cameraParams = updateData.camera_params;
                                projectedPoints = updateData.projected_points;

                                // Update projected mask
                                if (updateData.projected_mask_available && updateData.projected_mask) {{
                                    projectedMaskData = updateData.projected_mask;
                                    projectedMaskOffset = updateData.projected_mask_offset;
                                    if (projectedMaskVisible) {{
                                        showProjectedMask();
                                    }}
                                }}

                                updateCameraParamsDisplay();
                                updateGCPView();
                                updateCameraVisualization();

                                statusDiv.textContent = `Applied: ${{improvement > 0 ? '+' : ''}}${{improvement}}% improvement`;
                                statusDiv.style.color = improvement > 0 ? '#0a0' : '#c60';
                                updateStatus(`Auto-calibration accepted (improvement: ${{improvement > 0 ? '+' : ''}}${{improvement}}%)`);
                            }} else {{
                                alert('Failed to apply parameters: ' + updateData.error);
                                updateStatus('Error applying calibration');
                            }}
                        }})
                        .catch(err => {{
                            console.error('Failed to apply parameters:', err);
                            alert('Failed to apply parameters: ' + err.message);
                        }});
                    }} else {{
                        statusDiv.textContent = 'Calibration rejected';
                        statusDiv.style.color = '#888';
                        updateStatus('Auto-calibration rejected by user');
                    }}
                }} else {{
                    const errorMsg = data.error || data.message || 'Unknown error';
                    statusDiv.textContent = 'Error: ' + errorMsg;
                    statusDiv.style.color = '#c60';
                    updateStatus('Auto-calibration failed: ' + errorMsg);
                    alert('Auto-calibration failed:\\n' + errorMsg);
                }}
            }})
            .catch(err => {{
                btn.disabled = false;
                statusDiv.textContent = 'Error: ' + err.message;
                statusDiv.style.color = '#c60';
                console.error('Auto-calibration failed:', err);
                updateStatus('Auto-calibration failed: ' + err.message);
                alert('Auto-calibration failed:\\n' + err.message);
            }});
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

            // Convert UTM to pixel coordinates
            const pixelX = (easting - config.origin_easting) / config.pixel_size_x;
            const pixelY = (northing - config.origin_northing) / config.pixel_size_y;

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
            session.camera_frame = frame
            frame_height, frame_width = frame.shape[:2]
            camera_params = get_camera_params_for_projection(
                session.camera_name,
                image_width=frame_width,
                image_height=frame_height
            )
            session.camera_params = camera_params
            print(f"Camera frame captured: {frame_width}x{frame_height}")
            print(f"Camera params: pan={camera_params['pan_deg']:.1f}°, tilt={camera_params['tilt_deg']:.1f}°, zoom={camera_params['zoom']:.1f}x")
        except Exception as e:
            print(f"Warning: Failed to pre-capture camera frame: {e}")
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
        print(f"Origin: E {session.geotiff_params['origin_easting']}, N {session.geotiff_params['origin_northing']}")
        print(f"GSD: {abs(session.geotiff_params['pixel_size_x'])}m")
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
