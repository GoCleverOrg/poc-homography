#!/usr/bin/env python3
"""
Map Debug Server Module for POC Homography Project.

This module provides a lightweight web server for visualizing GCP validation results
with side-by-side camera frame and satellite map views. Uses Python's built-in
http.server module for zero-dependency deployment.

Example Usage:
    from poc_homography.map_debug_server import start_server

    # Start server with auto-open browser
    start_server(
        output_dir='output',
        camera_frame_path='output/annotated_frame.jpg',
        kml_path='output/gcp_validation.kml',
        camera_gps={'latitude': 39.640500, 'longitude': -0.230000},
        gcps=[...],
        validation_results={...},
        auto_open=True
    )

Environment Variables:
    GOOGLE_MAPS_API_KEY: Optional API key for Google Maps satellite layer
"""

import http.server
import os
import shutil
import socket
import socketserver
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port.

    Attempts to bind to consecutive ports starting from start_port until
    an available port is found or max_attempts is reached.

    Args:
        start_port: Port number to start searching from (default: 8080)
        max_attempts: Maximum number of ports to try (default: 10)

    Returns:
        First available port number

    Raises:
        RuntimeError: If no available port found within max_attempts

    Example:
        >>> port = find_available_port(8080, 10)
        >>> print(f"Found available port: {port}")
        Found available port: 8080
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            # Port is in use, try next
            continue

    raise RuntimeError(
        f"Could not find an available port in range "
        f"{start_port}-{start_port + max_attempts - 1}"
    )


def generate_html(
    camera_frame_path: str,
    kml_path: str,
    camera_gps: Dict[str, float],
    gcps: List[Dict[str, Any]],
    validation_results: Dict[str, Any]
) -> str:
    """
    Generate self-contained HTML string for the debug visualization.

    Creates an HTML page with side-by-side camera frame and map views.
    The camera frame includes canvas overlay with GCP markers, and the map
    uses Leaflet.js with ESRI World Imagery tiles.

    Args:
        camera_frame_path: Relative path to camera frame image (from output_dir)
        kml_path: Relative path to KML file (from output_dir)
        camera_gps: Camera GPS position {'latitude': float, 'longitude': float}
        gcps: List of GCP dictionaries with structure:
            {
                'gps': {'latitude': float, 'longitude': float},
                'image': {'u': float, 'v': float},
                'metadata': {'description': str}
            }
        validation_results: Dictionary containing:
            {
                'details': [
                    {
                        'projected_gps': (lat, lon),
                        'error_meters': float
                    },
                    ...
                ]
            }

    Returns:
        Complete HTML string ready to be written to file

    Example:
        >>> html = generate_html(
        ...     'annotated_frame.jpg',
        ...     'gcp_validation.kml',
        ...     {'latitude': 39.640500, 'longitude': -0.230000},
        ...     gcps=[...],
        ...     validation_results={...}
        ... )
    """
    # Check for Google Maps API key
    google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')

    # Prepare GCP data for JavaScript
    gcp_data = []
    for i, gcp in enumerate(gcps):
        gps = gcp.get('gps', {})
        image = gcp.get('image', {})
        metadata = gcp.get('metadata', {})

        gcp_entry = {
            'name': metadata.get('description', f'GCP {i+1}'),
            'original_gps': {
                'lat': gps.get('latitude', 0),
                'lon': gps.get('longitude', 0)
            },
            'pixel': {
                'u': image.get('u', 0),
                'v': image.get('v', 0)
            }
        }

        # Add projected GPS if available
        if validation_results and 'details' in validation_results:
            details = validation_results['details']
            if i < len(details) and 'projected_gps' in details[i]:
                proj_lat, proj_lon = details[i]['projected_gps']
                gcp_entry['projected_gps'] = {
                    'lat': proj_lat,
                    'lon': proj_lon
                }
                gcp_entry['error_meters'] = details[i].get('error_meters', 0)

        gcp_data.append(gcp_entry)

    # Convert data to JSON for embedding in HTML
    gcp_data_json = json.dumps(gcp_data)
    camera_gps_json = json.dumps(camera_gps)

    # Generate HTML with embedded data
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCP Map Debug Visualization</title>

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""/>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            height: 100vh;
            flex-direction: column;
        }}

        .header {{
            background: #2d2d2d;
            padding: 15px 20px;
            border-bottom: 2px solid #404040;
        }}

        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            color: #ffffff;
        }}

        .content {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #252525;
        }}

        .panel-header {{
            background: #2d2d2d;
            padding: 12px 20px;
            border-bottom: 1px solid #404040;
            font-size: 16px;
            font-weight: 500;
        }}

        .panel-content {{
            flex: 1;
            overflow: auto;
            position: relative;
        }}

        .divider {{
            width: 2px;
            background: #404040;
        }}

        /* Camera frame styling */
        .image-container {{
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1a1a1a;
        }}

        #cameraFrame {{
            max-width: 100%;
            max-height: 100%;
            display: block;
        }}

        #gcpCanvas {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }}

        /* Map styling */
        #map {{
            width: 100%;
            height: 100%;
        }}

        /* Legend styling */
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: rgba(45, 45, 45, 0.95);
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}

        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 14px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 13px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid #404040;
        }}

        .legend-line {{
            width: 20px;
            height: 3px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GCP Map Debug Visualization</h1>
        </div>

        <div class="content">
            <!-- Left Panel: Camera Frame with GCPs -->
            <div class="panel">
                <div class="panel-header">Camera Frame with GCPs</div>
                <div class="panel-content">
                    <div class="image-container">
                        <img id="cameraFrame" src="{camera_frame_path}" alt="Camera Frame">
                        <canvas id="gcpCanvas"></canvas>
                    </div>
                </div>
            </div>

            <div class="divider"></div>

            <!-- Right Panel: Satellite Map -->
            <div class="panel">
                <div class="panel-header">Satellite Map</div>
                <div class="panel-content">
                    <div id="map"></div>
                    <div class="legend">
                        <div class="legend-title">Legend</div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #00ff00;"></div>
                            <span>Original GCP</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #0000ff;"></div>
                            <span>Projected GCP</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line" style="background: #ffff00;"></div>
                            <span>Error Line</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
            crossorigin=""></script>

    <!-- Leaflet Omnivore for KML loading -->
    <script src='https://api.tiles.mapbox.com/mapbox.js/plugins/leaflet-omnivore/v0.3.1/leaflet-omnivore.min.js'></script>

    <script>
        // Embedded data
        const gcpData = {gcp_data_json};
        const cameraGPS = {camera_gps_json};
        const kmlPath = '{kml_path}';
        const googleMapsApiKey = '{google_maps_api_key}';

        // Initialize map
        const map = L.map('map');

        // ESRI World Imagery base layer
        const esriSatellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
                maxZoom: 19
            }}
        );

        esriSatellite.addTo(map);

        // Optional Google Maps layer
        const baseLayers = {{
            "ESRI Satellite": esriSatellite
        }};

        if (googleMapsApiKey) {{
            const googleSatellite = L.tileLayer(
                'https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key=' + googleMapsApiKey,
                {{
                    attribution: '&copy; Google Maps',
                    maxZoom: 20
                }}
            );
            baseLayers["Google Satellite"] = googleSatellite;
        }}

        // Add layer control if multiple base layers
        if (Object.keys(baseLayers).length > 1) {{
            L.control.layers(baseLayers).addTo(map);
        }}

        // Load KML file
        const kmlLayer = omnivore.kml(kmlPath);

        kmlLayer.on('ready', function() {{
            // Fit map to KML bounds
            map.fitBounds(kmlLayer.getBounds());
        }});

        kmlLayer.on('error', function(e) {{
            console.error('Error loading KML:', e);
        }});

        kmlLayer.addTo(map);

        // Set initial view to camera position
        map.setView([cameraGPS.latitude, cameraGPS.longitude], 18);

        // Canvas overlay for GCP markers on camera frame
        const img = document.getElementById('cameraFrame');
        const canvas = document.getElementById('gcpCanvas');
        const ctx = canvas.getContext('2d');

        // Extract P#XX from name (e.g., "P#00 - description" → "P#00")
        function getShortLabel(name, index) {{
            const match = name.match(/^P#\\d+/);
            return match ? match[0] : `GCP ${{index + 1}}`;
        }}

        function drawGCPMarkers() {{
            // Get actual displayed image dimensions
            const imgRect = img.getBoundingClientRect();
            const displayWidth = img.width;
            const displayHeight = img.height;

            // Get natural (original) image dimensions
            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;

            // Set canvas size to match displayed image
            canvas.width = displayWidth;
            canvas.height = displayHeight;

            // Calculate scale factor
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw GCP markers
            gcpData.forEach((gcp, index) => {{
                const u = gcp.pixel.u * scaleX;
                const v = gcp.pixel.v * scaleY;

                // Draw error line if projected GPS available
                if (gcp.projected_gps) {{
                    const projU = gcp.pixel.u * scaleX;  // Same pixel coords for line
                    const projV = gcp.pixel.v * scaleY;

                    ctx.strokeStyle = '#ffff00';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(u, v);
                    ctx.lineTo(projU, projV);
                    ctx.stroke();

                    // Draw projected marker (blue)
                    ctx.fillStyle = '#0000ff';
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(projU, projV, 3, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }}

                // Draw original marker (green)
                ctx.fillStyle = '#00ff00';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(u, v, 4, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();

                // Draw label (P#XX format only)
                const label = getShortLabel(gcp.name, index);
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 3;
                ctx.font = 'bold 7px sans-serif';
                ctx.strokeText(label, u + 12, v + 5);
                ctx.fillText(label, u + 12, v + 5);
            }});
        }}

        // Draw markers when image loads
        img.addEventListener('load', drawGCPMarkers);

        // Redraw on window resize
        window.addEventListener('resize', drawGCPMarkers);

        // Draw immediately if image is already loaded
        if (img.complete) {{
            drawGCPMarkers();
        }}
    </script>
</body>
</html>
"""

    return html


def start_server(
    output_dir: str,
    camera_frame_path: str,
    kml_path: str,
    camera_gps: Dict[str, float],
    gcps: List[Dict[str, Any]],
    validation_results: Dict[str, Any],
    auto_open: bool = True
) -> None:
    """
    Start the map debug visualization web server.

    Creates a lightweight HTTP server serving the debug visualization HTML page.
    Automatically finds an available port, generates HTML with embedded data,
    and optionally opens the browser.

    Args:
        output_dir: Directory where HTML and assets will be served from
        camera_frame_path: Absolute path to camera frame image
        kml_path: Absolute path to KML file
        camera_gps: Camera GPS position {'latitude': float, 'longitude': float}
        gcps: List of GCP dictionaries
        validation_results: Validation results dictionary
        auto_open: If True, automatically open browser (default: True)

    Raises:
        RuntimeError: If no available port found
        IOError: If unable to write HTML file or copy assets

    Example:
        >>> start_server(
        ...     output_dir='output',
        ...     camera_frame_path='output/annotated_frame.jpg',
        ...     kml_path='output/gcp_validation.kml',
        ...     camera_gps={'latitude': 39.640500, 'longitude': -0.230000},
        ...     gcps=[...],
        ...     validation_results={...},
        ...     auto_open=True
        ... )
        Server running at http://localhost:8080
        Press Ctrl+C to stop
    """
    # Find available port
    try:
        port = find_available_port()
    except RuntimeError as e:
        print(f"Error: {e}")
        raise

    # Convert paths to Path objects
    output_path = Path(output_dir).resolve()
    camera_frame_src = Path(camera_frame_path).resolve()
    kml_src = Path(kml_path).resolve()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine relative paths for HTML
    camera_frame_filename = camera_frame_src.name
    kml_filename = kml_src.name

    # Copy camera frame to output dir if not already there
    camera_frame_dest = output_path / camera_frame_filename
    if camera_frame_src != camera_frame_dest:
        if camera_frame_src.exists():
            shutil.copy2(camera_frame_src, camera_frame_dest)
            print(f"Copied camera frame to {camera_frame_dest}")
        else:
            raise IOError(f"Camera frame not found: {camera_frame_src}")

    # Copy KML file to output dir if not already there
    kml_dest = output_path / kml_filename
    if kml_src != kml_dest:
        if kml_src.exists():
            shutil.copy2(kml_src, kml_dest)
            print(f"Copied KML file to {kml_dest}")
        else:
            raise IOError(f"KML file not found: {kml_src}")

    # Generate HTML
    html_content = generate_html(
        camera_frame_filename,  # Use relative path in HTML
        kml_filename,           # Use relative path in HTML
        camera_gps,
        gcps,
        validation_results
    )

    # Write HTML to output directory
    html_path = output_path / 'index.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated HTML at {html_path}")

    # Create custom HTTP request handler that serves from output_dir
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_path), **kwargs)

        def log_message(self, format, *args):
            """Override to customize logging."""
            print(f"[{self.address_string()}] {format % args}")

    # Create server
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        server_url = f"http://localhost:{port}"
        print(f"\nServer running at {server_url}")
        print("Press Ctrl+C to stop\n")

        # Auto-open browser
        if auto_open:
            try:
                webbrowser.open(server_url)
                print(f"Opened browser to {server_url}")
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please open {server_url} manually")

        # Run server until keyboard interrupt
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()
            print("Server stopped.")


if __name__ == '__main__':
    """
    Example usage and test of the map debug server.
    """
    # Create sample data
    sample_gcps = [
        {
            'gps': {'latitude': 39.640600, 'longitude': -0.230200},
            'image': {'u': 400.0, 'v': 300.0},
            'metadata': {'description': 'P#01'}
        },
        {
            'gps': {'latitude': 39.640620, 'longitude': -0.229800},
            'image': {'u': 2100.0, 'v': 320.0},
            'metadata': {'description': 'P#02'}
        },
        {
            'gps': {'latitude': 39.640400, 'longitude': -0.230000},
            'image': {'u': 1280.0, 'v': 720.0},
            'metadata': {'description': 'P#03'}
        },
    ]

    sample_results = {
        'details': [
            {
                'projected_gps': (39.640605, -0.230205),
                'error_meters': 0.56
            },
            {
                'projected_gps': (39.640618, -0.229805),
                'error_meters': 0.45
            },
            {
                'projected_gps': (39.640398, -0.230002),
                'error_meters': 0.23
            },
        ]
    }

    sample_camera = {
        'latitude': 39.640500,
        'longitude': -0.230000
    }

    # Get project root
    module_dir = Path(__file__).parent
    project_root = module_dir.parent
    output_dir = project_root / 'output'

    # Use example KML file if it exists
    example_kml = output_dir / 'example_gcp_validation.kml'

    if not example_kml.exists():
        print("Error: Example KML file not found.")
        print(f"Please run kml_generator.py first to create {example_kml}")
        exit(1)

    # Create a dummy camera frame for testing
    import cv2
    import numpy as np

    dummy_frame_path = output_dir / 'dummy_frame.jpg'
    if not dummy_frame_path.exists():
        # Create a simple test image
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = (50, 50, 50)  # Dark gray background

        # Draw grid
        for i in range(0, 1920, 100):
            cv2.line(frame, (i, 0), (i, 1080), (100, 100, 100), 1)
        for i in range(0, 1080, 100):
            cv2.line(frame, (0, i), (1920, i), (100, 100, 100), 1)

        cv2.imwrite(str(dummy_frame_path), frame)
        print(f"Created dummy frame at {dummy_frame_path}")

    # Start server
    print("Starting map debug server with example data...")
    start_server(
        output_dir=str(output_dir),
        camera_frame_path=str(dummy_frame_path),
        kml_path=str(example_kml),
        camera_gps=sample_camera,
        gcps=sample_gcps,
        validation_results=sample_results,
        auto_open=True
    )
