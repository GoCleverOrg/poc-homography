#!/usr/bin/env python3
"""
Unified GCP Tool - Combines KML extraction and GCP capture in a two-tab interface.

This tool provides a unified workflow for:
1. Extracting KML points from georeferenced images (Tab 1)
2. Capturing GCPs with map-first mode (Tab 2)

The two tabs share a unified session with automatic point synchronization.

Usage:
    python tools/unified_gcp_tool.py <image> --camera <camera_name> [--port <port>]

Examples:
    # Basic usage
    python tools/unified_gcp_tool.py frame.jpg --camera Valte

    # Custom port
    python tools/unified_gcp_tool.py frame.jpg --camera Valte --port 8080
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
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

# Import intrinsics utility
try:
    from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics
    INTRINSICS_AVAILABLE = True
except ImportError:
    INTRINSICS_AVAILABLE = False


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

        # Load frame image
        self.frame = cv2.imread(str(image_path))
        if self.frame is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.frame_height, self.frame_width = self.frame.shape[:2]

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

    def add_point(self, px: float, py: float, name: str, category: str):
        """Add a KML point."""
        lat, lon = self.pixel_to_latlon(px, py)
        easting, northing = self.pixel_to_utm(px, py)
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
        Project KML points to image coordinates for GCP Capture tab.

        Returns list of dictionaries with projected pixel coordinates.
        """
        if not GEOMETRY_AVAILABLE:
            return []

        projected = []
        for pt in self.points:
            # Points already have pixel coordinates from KML extraction
            projected.append({
                'name': pt['name'],
                'category': pt['category'],
                'pixel_u': pt['pixel_x'],
                'pixel_v': pt['pixel_y'],
                'latitude': pt['lat'],
                'longitude': pt['lon'],
                'visible': True
            })

        return projected


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
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))

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

            # Project points for GCP tab
            projected_points = self.session.project_points_to_image()

            self.send_json_response({
                'success': True,
                'projected_points': projected_points,
                'kml_saved_to': temp_kml_path,
                'total_points': len(self.session.points)
            })

        else:
            self.send_error(404)

    def send_json_response(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


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
        #main-image {{
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
                    KML points from Tab 1 are projected onto the image.
                    These markers show where your GPS points appear in the camera view.
                </div>

                <div class="controls">
                    <h3>Projected Points</h3>
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
        let currentZoom = 1;
        let counters = {{ zebra: 1, arrow: 1, parking: 1, other: 1 }};
        let categoryVisibility = {{ zebra: true, arrow: true, parking: true, other: true }};
        let currentTab = 'kml';
        let projectedPoints = [];

        const img = document.getElementById('main-image');
        const gcpImg = document.getElementById('gcp-image');
        const container = document.getElementById('image-container');
        const gcpContainer = document.getElementById('gcp-image-container');

        // Initialize
        updatePointName();
        document.getElementById('category').addEventListener('change', updatePointName);

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

            fetch('/api/switch_to_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ points: points }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    projectedPoints = data.projected_points;
                    updateGCPView();
                    updateStatus(`Auto-saved ${{points.length}} points and projected to image`);
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
                    marker.style.left = (pt.pixel_u * currentZoom) + 'px';
                    marker.style.top = (pt.pixel_v * currentZoom) + 'px';
                    marker.title = pt.name;
                    gcpContainer.appendChild(marker);
                }}
            }});

            // Update list
            listContainer.innerHTML = projectedPoints.map((pt, i) => `
                <div class="point-item">
                    <div class="info">
                        <div class="name">${{i+1}}. ${{pt.name}} (${{pt.category || 'point'}})</div>
                        <div class="coords">Pixel: (${{pt.pixel_u.toFixed(1)}}, ${{pt.pixel_v.toFixed(1)}})</div>
                        <div class="coords">GPS: ${{pt.latitude.toFixed(6)}}, ${{pt.longitude.toFixed(6)}}</div>
                    </div>
                </div>
            `).join('');
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
            const px = (e.clientX - rect.left) / currentZoom;
            const py = (e.clientY - rect.top) / currentZoom;

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
                const pointX = p.pixel_x * currentZoom;
                const pointY = p.pixel_y * currentZoom;

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

        function zoom(factor) {{
            currentZoom *= factor;
            img.style.width = (img.naturalWidth * currentZoom) + 'px';
            gcpImg.style.width = (gcpImg.naturalWidth * currentZoom) + 'px';
            redrawMarkers();
            updateGCPView();
        }}

        function resetZoom() {{
            currentZoom = 1;
            img.style.width = '';
            gcpImg.style.width = '';
            redrawMarkers();
            updateGCPView();
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
    </script>
</body>
</html>'''


def run_server(session: UnifiedSession, port: int = 8765):
    """Run the unified web server."""

    # Create temp directory for serving frame
    temp_dir = tempfile.mkdtemp(prefix='unified_gcp_')
    frame_path = os.path.join(temp_dir, 'frame.jpg')
    cv2.imwrite(frame_path, session.frame)

    # Set up handler
    UnifiedHTTPHandler.session = session
    UnifiedHTTPHandler.temp_dir = temp_dir

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

    args = parser.parse_args()

    # Validate image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
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

    # Run server
    run_server(session, args.port)


if __name__ == '__main__':
    main()
