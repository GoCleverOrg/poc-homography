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

from poc_homography.satellite_layers import generate_satellite_layers_js


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
    validation_results: Dict[str, Any],
    homography_matrix: Optional[List[List[float]]] = None
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
                        'projected_pixel': (u, v),
                        'error_meters': float
                    },
                    ...
                ]
            }
        homography_matrix: Optional 3x3 inverse homography matrix (image -> local metric)
            for interactive projection. If provided, enables click-to-project feature.

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
    # Check for Google Maps API key and generate satellite layers
    google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    satellite_layers_js = generate_satellite_layers_js(
        google_api_key=google_maps_api_key if google_maps_api_key else None,
        default_layer='google'
    )

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

        # Add projected GPS and projected pixel if available
        if validation_results and 'details' in validation_results:
            details = validation_results['details']
            if i < len(details) and 'projected_gps' in details[i]:
                proj_lat, proj_lon = details[i]['projected_gps']
                gcp_entry['projected_gps'] = {
                    'lat': proj_lat,
                    'lon': proj_lon
                }
                gcp_entry['error_meters'] = details[i].get('error_meters', 0)

                # Add projected pixel coordinates if available
                if 'projected_pixel' in details[i]:
                    proj_u, proj_v = details[i]['projected_pixel']
                    gcp_entry['projected_pixel'] = {
                        'u': proj_u,
                        'v': proj_v
                    }

        gcp_data.append(gcp_entry)

    # Extract homography stats for header display
    confidence = validation_results.get('confidence', 0) if validation_results else 0
    inliers = validation_results.get('inliers', 0) if validation_results else 0
    outliers = validation_results.get('outliers', 0) if validation_results else 0
    total_gcps = validation_results.get('gcps_tested', len(gcps)) if validation_results else len(gcps)

    # GPS error stats (in meters)
    mean_gps_error = validation_results.get('mean_error_m', 0) if validation_results else 0
    min_gps_error = validation_results.get('min_error_m', 0) if validation_results else 0
    max_gps_error = validation_results.get('max_error_m', 0) if validation_results else 0

    # Pixel reprojection error stats
    reproj = validation_results.get('reprojection_error', {}) if validation_results else {}
    mean_px_error = reproj.get('mean_px', 0) or 0
    min_px_error = reproj.get('min_px', 0) or 0
    max_px_error = reproj.get('max_px', 0) or 0

    # Determine CSS classes for color coding
    conf_class = 'good' if confidence >= 0.7 else ('warn' if confidence >= 0.5 else 'bad')
    mean_gps_class = 'good' if mean_gps_error < 1.0 else ('warn' if mean_gps_error < 3.0 else 'bad')
    mean_px_class = 'good' if mean_px_error < 5.0 else ('warn' if mean_px_error < 10.0 else 'bad')

    # Convert data to JSON for embedding in HTML
    gcp_data_json = json.dumps(gcp_data)
    camera_gps_json = json.dumps(camera_gps)
    homography_json = json.dumps(homography_matrix) if homography_matrix else 'null'

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
            display: flex;
            justify-content: space-between;
            align-items: center;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .control-btn.header-btn {{
            position: static;
            padding: 6px 12px;
            font-size: 12px;
        }}

        .header-stats {{
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #b0b0b0;
        }}

        .header-stats .stat-group {{
            display: flex;
            gap: 12px;
        }}

        .header-stats .stat {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .header-stats .stat-label {{
            color: #888;
        }}

        .header-stats .stat-value {{
            color: #e0e0e0;
            font-weight: 500;
        }}

        .header-stats .stat-value.good {{
            color: #4CAF50;
        }}

        .header-stats .stat-value.warn {{
            color: #FFC107;
        }}

        .header-stats .stat-value.bad {{
            color: #f44336;
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
            cursor: move;
            user-select: none;
        }}

        .legend:active {{
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        }}

        .legend.top-right {{
            top: 50px;
            bottom: auto;
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

        .legend-item svg {{
            margin-right: 8px;
            flex-shrink: 0;
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

        /* Remove default Leaflet marker styling for custom icons */
        .custom-marker {{
            background: transparent;
            border: none;
        }}

        /* Tooltip styling - use fixed positioning to render on top of all panels */
        .gcp-tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 10000;
            white-space: nowrap;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            border: 1px solid #555;
        }}

        .gcp-tooltip .tooltip-title {{
            font-weight: bold;
            margin-bottom: 4px;
            color: #ffd700;
        }}

        .gcp-tooltip .tooltip-row {{
            margin: 2px 0;
        }}

        /* Interactive canvas cursor */
        #gcpCanvas.interactive {{
            pointer-events: auto;
            cursor: crosshair;
        }}

        /* Click instruction */
        .click-instruction {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(45, 45, 45, 0.9);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            color: #aaa;
            z-index: 1000;
        }}

        .click-instruction.active {{
            color: #4CAF50;
        }}

        /* Control buttons */
        .control-btn {{
            position: absolute;
            background: rgba(45, 45, 45, 0.95);
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            z-index: 1001;
            transition: all 0.2s ease;
        }}

        .control-btn:hover {{
            background: rgba(60, 60, 60, 0.95);
            border-color: #777;
        }}

        .control-btn.active {{
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
            color: #4CAF50;
        }}

        .control-btn.reset {{
            top: 10px;
            right: 10px;
        }}

        .control-btn.zoom-mode {{
            top: 10px;
            left: 10px;
        }}

        .control-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GCP Map Debug Visualization</h1>
            <div class="header-stats">
                <div class="stat-group">
                    <div class="stat">
                        <span class="stat-label">Confidence:</span>
                        <span class="stat-value {conf_class}">{confidence:.1%}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Points:</span>
                        <span class="stat-value">{inliers}/{total_gcps}</span>
                        <span class="stat-label">({outliers} outliers)</span>
                    </div>
                </div>
                <div class="stat-group">
                    <div class="stat">
                        <span class="stat-label">GPS Error:</span>
                        <span class="stat-value {mean_gps_class}">{mean_gps_error:.2f}m</span>
                        <span class="stat-label">(min: {min_gps_error:.2f}m, max: {max_gps_error:.2f}m)</span>
                    </div>
                </div>
                <div class="stat-group">
                    <div class="stat">
                        <span class="stat-label">Pixel Error:</span>
                        <span class="stat-value {mean_px_class}">{mean_px_error:.1f}px</span>
                        <span class="stat-label">(min: {min_px_error:.1f}px, max: {max_px_error:.1f}px)</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="content">
            <!-- Left Panel: Camera Frame with GCPs -->
            <div class="panel">
                <div class="panel-header">Camera Frame with GCPs</div>
                <div class="panel-content">
                    <div class="image-container">
                        <img id="cameraFrame" src="{camera_frame_path}" alt="Camera Frame">
                        <canvas id="gcpCanvas"></canvas>
                        <div id="leftTooltip" class="gcp-tooltip" style="display: none;"></div>
                        <div id="clickInstruction" class="click-instruction">Click to project point to map</div>
                        <button id="zoomModeBtn" class="control-btn zoom-mode">Precision Zoom: OFF</button>
                    </div>
                    <div class="legend top-right">
                        <div class="legend-title">Legend</div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #00ff00;"></div>
                            <span>Original GCP</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff0000;"></div>
                            <span>Projected GCP</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffd700;"></div>
                            <span>Accurate GCP (&lt;5px)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line" style="background: #ffff00;"></div>
                            <span>Error Line</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="divider"></div>

            <!-- Right Panel: Satellite Map -->
            <div class="panel">
                <div class="panel-header">
                    Satellite Map
                    <button id="resetViewBtn" class="control-btn reset header-btn">Reset View</button>
                </div>
                <div class="panel-content">
                    <div id="map"></div>
                    <div id="rightTooltip" class="gcp-tooltip" style="display: none;"></div>
                    <div class="legend">
                        <div class="legend-title">Legend</div>
                        <div class="legend-item">
                            <svg width="20" height="20" viewBox="0 0 24 24">
                                <polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9"
                                         fill="#00ff00" stroke="white" stroke-width="1.5"/>
                            </svg>
                            <span>Original GCP</span>
                        </div>
                        <div class="legend-item">
                            <svg width="20" height="20" viewBox="0 0 24 24">
                                <polygon points="12,2 22,12 12,22 2,12"
                                         fill="#ff0000" stroke="white" stroke-width="1.5"/>
                            </svg>
                            <span>Projected GCP</span>
                        </div>
                        <div class="legend-item">
                            <svg width="20" height="20" viewBox="0 0 24 24">
                                <polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9"
                                         fill="#ffd700" stroke="white" stroke-width="1.5"/>
                            </svg>
                            <span>Accurate GCP (&lt;50cm)</span>
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

    <script>
        // Embedded data
        const gcpData = {gcp_data_json};
        const cameraGPS = {camera_gps_json};
        const homographyMatrix = {homography_json};
        const EARTH_RADIUS_M = 6371000;

        // Interactive projection state
        let clickedMarkers = [];  // Store markers from clicks
        let hoveredGcpIndex = null;  // Currently hovered GCP index
        let hoveredClickedIndex = null;  // Currently hovered clicked point index
        let savedMapView = null;  // Store map view before zoom
        let initialMapView = null;  // Store initial map view for reset
        let precisionZoomEnabled = false;  // Precision zoom mode toggle

        // Make legends draggable
        function makeDraggable(element) {{
            let isDragging = false;
            let startX, startY, startLeft, startTop;

            element.addEventListener('mousedown', function(e) {{
                // Ignore if clicking on interactive elements
                if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;

                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;

                const rect = element.getBoundingClientRect();
                const parentRect = element.parentElement.getBoundingClientRect();
                startLeft = rect.left - parentRect.left;
                startTop = rect.top - parentRect.top;

                // Reset right/bottom positioning to use left/top
                element.style.right = 'auto';
                element.style.bottom = 'auto';
                element.style.left = startLeft + 'px';
                element.style.top = startTop + 'px';

                e.preventDefault();
            }});

            document.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;

                const dx = e.clientX - startX;
                const dy = e.clientY - startY;

                element.style.left = (startLeft + dx) + 'px';
                element.style.top = (startTop + dy) + 'px';
            }});

            document.addEventListener('mouseup', function() {{
                isDragging = false;
            }});
        }}

        // Initialize draggable legends after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('.legend').forEach(makeDraggable);
        }});

        // Initialize map with high max zoom for over-zooming
        const map = L.map('map', {{
            maxZoom: 23  // Allow over-zooming for precision inspection
        }});

        // Satellite layer configuration (from shared module)
        {satellite_layers_js}

        // Haversine distance function
        function haversineDistance(lat1, lon1, lat2, lon2) {{
            const R = 6371000; // Earth radius in meters
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                      Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}

        // Create custom SVG icons
        function createStarIcon(color, size = 24, opacity = 1) {{
            const svg = `
                <svg width="${{size}}" height="${{size}}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9"
                             fill="${{color}}" fill-opacity="${{opacity}}" stroke="white" stroke-width="1.5" stroke-opacity="${{opacity}}"/>
                </svg>`;
            return L.divIcon({{
                html: svg,
                className: 'custom-marker',
                iconSize: [size, size],
                iconAnchor: [size/2, size/2],
                popupAnchor: [0, -size/2]
            }});
        }}

        function createDiamondIcon(color, size = 20, opacity = 1) {{
            const svg = `
                <svg width="${{size}}" height="${{size}}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <polygon points="12,2 22,12 12,22 2,12"
                             fill="${{color}}" fill-opacity="${{opacity}}" stroke="white" stroke-width="1.5" stroke-opacity="${{opacity}}"/>
                </svg>`;
            return L.divIcon({{
                html: svg,
                className: 'custom-marker',
                iconSize: [size, size],
                iconAnchor: [size/2, size/2],
                popupAnchor: [0, -size/2]
            }});
        }}

        // Calculate bounds from GCP data
        function calculateGCPBounds() {{
            const latLngs = [];
            gcpData.forEach(gcp => {{
                latLngs.push([gcp.original_gps.lat, gcp.original_gps.lon]);
                if (gcp.projected_gps) {{
                    latLngs.push([gcp.projected_gps.lat, gcp.projected_gps.lon]);
                }}
            }});
            if (latLngs.length > 0) {{
                return L.latLngBounds(latLngs);
            }}
            return null;
        }}

        // Fit map to GCP bounds
        const bounds = calculateGCPBounds();
        if (bounds) {{
            map.fitBounds(bounds, {{ padding: [50, 50] }});
        }} else {{
            // Fallback to camera position
            map.setView([cameraGPS.latitude, cameraGPS.longitude], 18);
        }}

        // Store initial map view after a short delay to ensure map has settled
        setTimeout(() => {{
            initialMapView = {{
                center: map.getCenter(),
                zoom: map.getZoom()
            }};
        }}, 100);

        // Canvas overlay for GCP markers on camera frame
        const img = document.getElementById('cameraFrame');
        const canvas = document.getElementById('gcpCanvas');
        const ctx = canvas.getContext('2d');

        function drawGCPMarkers() {{
            // Get actual displayed image dimensions
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

                if (gcp.projected_pixel) {{
                    const projU = gcp.projected_pixel.u * scaleX;
                    const projV = gcp.projected_pixel.v * scaleY;

                    // Calculate distance between original and projected pixels
                    const dx = u - projU;
                    const dy = v - projV;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const threshold = 5;

                    if (distance <= threshold) {{
                        // Draw single gold marker at original position
                        ctx.fillStyle = '#ffd700';  // Gold
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(u, v, 4, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                    }} else {{
                        // Draw error line (yellow)
                        ctx.strokeStyle = '#ffff00';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(u, v);
                        ctx.lineTo(projU, projV);
                        ctx.stroke();

                        // Draw projected marker (red)
                        ctx.fillStyle = '#ff0000';
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(projU, projV, 3, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();

                        // Draw original marker (green)
                        ctx.fillStyle = '#00ff00';
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(u, v, 4, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                    }}
                }} else {{
                    // No projected pixel - draw only original marker (green)
                    ctx.fillStyle = '#00ff00';
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(u, v, 4, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }}
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

        // ============================================================
        // Interactive Projection Functions
        // ============================================================

        // Project image point to GPS using homography matrix
        function projectImageToGPS(u, v) {{
            if (!homographyMatrix) return null;

            // Apply inverse homography: H_inv * [u, v, 1]^T
            const H = homographyMatrix;
            const w = H[2][0] * u + H[2][1] * v + H[2][2];

            if (Math.abs(w) < 1e-10) {{
                return null; // Point at infinity
            }}

            const x_local = (H[0][0] * u + H[0][1] * v + H[0][2]) / w;
            const y_local = (H[1][0] * u + H[1][1] * v + H[1][2]) / w;

            // Convert local metric to GPS (equirectangular projection)
            const ref_lat_rad = cameraGPS.latitude * Math.PI / 180;
            const delta_lat_rad = y_local / EARTH_RADIUS_M;
            const delta_lon_rad = x_local / (EARTH_RADIUS_M * Math.cos(ref_lat_rad));

            const lat = cameraGPS.latitude + delta_lat_rad * 180 / Math.PI;
            const lon = cameraGPS.longitude + delta_lon_rad * 180 / Math.PI;

            return {{ lat, lon, x_local, y_local }};
        }}

        // Store map markers for hover highlighting
        const mapMarkers = [];

        // Redraw map markers and store references
        function drawMapMarkersWithRefs() {{
            const threshold = 0.5; // meters (50cm)

            gcpData.forEach((gcp, index) => {{
                const origLat = gcp.original_gps.lat;
                const origLon = gcp.original_gps.lon;

                if (gcp.projected_gps) {{
                    const projLat = gcp.projected_gps.lat;
                    const projLon = gcp.projected_gps.lon;
                    const distance = gcp.error_meters ?? haversineDistance(origLat, origLon, projLat, projLon);

                    if (distance <= threshold) {{
                        const marker = L.marker([origLat, origLon], {{
                            icon: createStarIcon('#ffd700', 28)
                        }}).addTo(map).bindPopup(`${{gcp.name}}<br>Accurate: ${{distance.toFixed(2)}}m`);
                        mapMarkers.push({{ marker, index, type: 'accurate', iconType: 'star', color: '#ffd700', size: 28 }});
                        addMarkerHoverHandlers(marker, index);
                    }} else {{
                        L.polyline([[origLat, origLon], [projLat, projLon]], {{
                            color: '#ffff00',
                            weight: 2
                        }}).addTo(map);

                        const projMarker = L.marker([projLat, projLon], {{
                            icon: createDiamondIcon('#ff0000', 20)
                        }}).addTo(map).bindPopup(`${{gcp.name}} (projected)<br>Error: ${{distance.toFixed(2)}}m`);
                        mapMarkers.push({{ marker: projMarker, index, type: 'projected', iconType: 'diamond', color: '#ff0000', size: 20 }});
                        addMarkerHoverHandlers(projMarker, index);

                        const origMarker = L.marker([origLat, origLon], {{
                            icon: createStarIcon('#00ff00', 24)
                        }}).addTo(map).bindPopup(`${{gcp.name}} (original)`);
                        mapMarkers.push({{ marker: origMarker, index, type: 'original', iconType: 'star', color: '#00ff00', size: 24 }});
                        addMarkerHoverHandlers(origMarker, index);
                    }}
                }} else {{
                    const marker = L.marker([origLat, origLon], {{
                        icon: createStarIcon('#00ff00', 24)
                    }}).addTo(map).bindPopup(`${{gcp.name}}`);
                    mapMarkers.push({{ marker, index, type: 'original', iconType: 'star', color: '#00ff00', size: 24 }});
                    addMarkerHoverHandlers(marker, index);
                }}
            }});
        }}

        // Add hover handlers to map markers
        function addMarkerHoverHandlers(marker, index) {{
            marker.on('mouseover', () => highlightGCP(index));
            marker.on('mouseout', () => unhighlightGCP());
        }}

        // Highlight GCP on both panels
        function highlightGCP(index) {{
            hoveredGcpIndex = index;
            const gcp = gcpData[index];
            if (!gcp) return;

            // Show left tooltip (pixel error)
            const leftTooltip = document.getElementById('leftTooltip');

            // Calculate distance from camera to this GCP
            const distToCamera = haversineDistance(
                cameraGPS.latitude, cameraGPS.longitude,
                gcp.original_gps.lat, gcp.original_gps.lon
            );

            if (gcp.projected_pixel) {{
                const dx = gcp.pixel.u - gcp.projected_pixel.u;
                const dy = gcp.pixel.v - gcp.projected_pixel.v;
                const pixelError = Math.sqrt(dx * dx + dy * dy);
                leftTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Pixel: (${{gcp.pixel.u.toFixed(1)}}, ${{gcp.pixel.v.toFixed(1)}})</div>
                    <div class="tooltip-row">Pixel Error: ${{pixelError.toFixed(2)}}px</div>
                    <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
                `;
            }} else {{
                leftTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Pixel: (${{gcp.pixel.u.toFixed(1)}}, ${{gcp.pixel.v.toFixed(1)}})</div>
                    <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
                `;
            }}
            leftTooltip.style.display = 'block';

            // Position left tooltip using viewport coordinates (fixed positioning)
            const displayWidth = img.width;
            const displayHeight = img.height;
            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;
            const u = gcp.pixel.u * scaleX;
            const v = gcp.pixel.v * scaleY;

            // Get viewport position of the GCP marker
            const imgRect = img.getBoundingClientRect();
            const tooltipX = imgRect.left + u + 15;
            const tooltipY = imgRect.top + v - 30;

            // Ensure tooltip doesn't go off-screen
            const tooltipRect = leftTooltip.getBoundingClientRect();
            const maxX = window.innerWidth - tooltipRect.width - 10;
            const maxY = window.innerHeight - tooltipRect.height - 10;

            leftTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
            leftTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';

            // Only zoom when precision mode is enabled
            if (precisionZoomEnabled) {{
                // Save current map view before zooming
                if (!savedMapView) {{
                    savedMapView = {{
                        center: map.getCenter(),
                        zoom: map.getZoom()
                    }};
                }}

                // Zoom to the GCP location for precision inspection
                const zoomTarget = L.latLng(gcp.original_gps.lat, gcp.original_gps.lon);
                map.setView(zoomTarget, 22, {{ animate: true, duration: 0.3 }});

                // Change map marker icons to cyan with 35% opacity for transparency
                mapMarkers.forEach(m => {{
                    if (m.index === index) {{
                        // Semi-transparent (35% opacity) cyan marker for accuracy testing
                        const cyanIcon = m.iconType === 'star'
                            ? createStarIcon('#00ffff', m.size * 1.5, 0.35)
                            : createDiamondIcon('#00ffff', m.size * 1.5, 0.35);
                        m.marker.setIcon(cyanIcon);
                    }}
                }});
            }} else {{
                // Just highlight without zoom - use cyan with full opacity
                mapMarkers.forEach(m => {{
                    if (m.index === index) {{
                        const cyanIcon = m.iconType === 'star'
                            ? createStarIcon('#00ffff', m.size * 1.3, 1)
                            : createDiamondIcon('#00ffff', m.size * 1.3, 1);
                        m.marker.setIcon(cyanIcon);
                    }}
                }});
            }}

            // Show right tooltip (GPS error) - positioned after zoom completes
            const rightTooltip = document.getElementById('rightTooltip');
            if (gcp.projected_gps) {{
                const errorMeters = gcp.error_meters ?? haversineDistance(
                    gcp.original_gps.lat, gcp.original_gps.lon,
                    gcp.projected_gps.lat, gcp.projected_gps.lon
                );
                rightTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Original: (${{gcp.original_gps.lat.toFixed(6)}}, ${{gcp.original_gps.lon.toFixed(6)}})</div>
                    <div class="tooltip-row">GPS Error: ${{errorMeters.toFixed(2)}}m</div>
                    <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
                `;
            }} else {{
                rightTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">GPS: (${{gcp.original_gps.lat.toFixed(6)}}, ${{gcp.original_gps.lon.toFixed(6)}})</div>
                    <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
                `;
            }}
            rightTooltip.style.display = 'block';

            // Position right tooltip using viewport coordinates (fixed positioning)
            const positionRightTooltip = () => {{
                const mapContainer = document.getElementById('map');
                const mapRect = mapContainer.getBoundingClientRect();
                const markerLatLng = L.latLng(gcp.original_gps.lat, gcp.original_gps.lon);
                const markerPoint = map.latLngToContainerPoint(markerLatLng);

                // Convert to viewport coordinates
                const tooltipX = mapRect.left + markerPoint.x + 20;
                const tooltipY = mapRect.top + markerPoint.y - 20;

                // Ensure tooltip doesn't go off-screen
                const tooltipRect = rightTooltip.getBoundingClientRect();
                const maxX = window.innerWidth - tooltipRect.width - 10;
                const maxY = window.innerHeight - tooltipRect.height - 10;

                rightTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
                rightTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';
            }};

            if (precisionZoomEnabled) {{
                // Wait for zoom animation to complete
                setTimeout(positionRightTooltip, 350);
            }} else {{
                // Position immediately
                positionRightTooltip();
            }}

            // Redraw canvas with highlight
            drawGCPMarkers();
        }}

        // Remove highlight
        function unhighlightGCP() {{
            hoveredGcpIndex = null;
            document.getElementById('leftTooltip').style.display = 'none';
            document.getElementById('rightTooltip').style.display = 'none';

            // Restore original map view (zoom out) only if we saved it
            if (savedMapView && precisionZoomEnabled) {{
                map.setView(savedMapView.center, savedMapView.zoom, {{ animate: true, duration: 0.3 }});
            }}
            savedMapView = null;

            // Restore original map marker icons with full opacity
            mapMarkers.forEach(m => {{
                const originalIcon = m.iconType === 'star'
                    ? createStarIcon(m.color, m.size, 1)
                    : createDiamondIcon(m.color, m.size, 1);
                m.marker.setIcon(originalIcon);
            }});

            drawGCPMarkers();
        }}

        // Reset map to initial view (failsafe button)
        function resetMapView() {{
            // Clear any hover state
            hoveredGcpIndex = null;
            hoveredClickedIndex = null;
            savedMapView = null;
            document.getElementById('leftTooltip').style.display = 'none';
            document.getElementById('rightTooltip').style.display = 'none';

            // Restore all GCP markers to original state
            mapMarkers.forEach(m => {{
                const originalIcon = m.iconType === 'star'
                    ? createStarIcon(m.color, m.size, 1)
                    : createDiamondIcon(m.color, m.size, 1);
                m.marker.setIcon(originalIcon);
            }});

            // Restore all clicked markers to original state
            clickedMarkers.forEach(cm => {{
                cm.marker.setIcon(createDiamondIcon('#ff00ff', 22, 1));
            }});

            // Reset to initial view
            if (initialMapView) {{
                map.setView(initialMapView.center, initialMapView.zoom, {{ animate: true, duration: 0.3 }});
            }} else {{
                // Fallback: recalculate bounds
                const bounds = calculateGCPBounds();
                if (bounds) {{
                    map.fitBounds(bounds, {{ padding: [50, 50] }});
                }} else {{
                    map.setView([cameraGPS.latitude, cameraGPS.longitude], 18);
                }}
            }}

            drawGCPMarkers();
        }}

        // Toggle precision zoom mode
        function togglePrecisionZoom() {{
            precisionZoomEnabled = !precisionZoomEnabled;
            const btn = document.getElementById('zoomModeBtn');
            if (precisionZoomEnabled) {{
                btn.textContent = 'Precision Zoom: ON';
                btn.classList.add('active');
            }} else {{
                btn.textContent = 'Precision Zoom: OFF';
                btn.classList.remove('active');
                // Reset map if we're turning off precision mode
                if (savedMapView) {{
                    map.setView(savedMapView.center, savedMapView.zoom, {{ animate: true, duration: 0.3 }});
                    savedMapView = null;
                }}
            }}
        }}

        // Highlight clicked point on both panels
        function highlightClickedPoint(index) {{
            hoveredClickedIndex = index;
            const cm = clickedMarkers[index];
            if (!cm) return;

            // Calculate distance from camera to this clicked point
            const distToCamera = haversineDistance(
                cameraGPS.latitude, cameraGPS.longitude,
                cm.lat, cm.lon
            );

            // Show left tooltip
            const leftTooltip = document.getElementById('leftTooltip');
            leftTooltip.innerHTML = `
                <div class="tooltip-title">Clicked Point</div>
                <div class="tooltip-row">Pixel: (${{cm.origU.toFixed(1)}}, ${{cm.origV.toFixed(1)}})</div>
                <div class="tooltip-row">GPS: (${{cm.lat.toFixed(6)}}, ${{cm.lon.toFixed(6)}})</div>
                <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
            `;
            leftTooltip.style.display = 'block';

            // Position left tooltip using viewport coordinates (fixed positioning)
            const imgRect = img.getBoundingClientRect();
            const tooltipX = imgRect.left + cm.displayU + 15;
            const tooltipY = imgRect.top + cm.displayV - 30;

            // Ensure tooltip doesn't go off-screen
            const tooltipRect = leftTooltip.getBoundingClientRect();
            const maxX = window.innerWidth - tooltipRect.width - 10;
            const maxY = window.innerHeight - tooltipRect.height - 10;

            leftTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
            leftTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';

            // Only zoom when precision mode is enabled
            if (precisionZoomEnabled) {{
                // Save current map view before zooming
                if (!savedMapView) {{
                    savedMapView = {{
                        center: map.getCenter(),
                        zoom: map.getZoom()
                    }};
                }}

                // Zoom to the clicked point location
                const zoomTarget = L.latLng(cm.lat, cm.lon);
                map.setView(zoomTarget, 22, {{ animate: true, duration: 0.3 }});

                // Change marker to semi-transparent cyan
                cm.marker.setIcon(createDiamondIcon('#00ffff', 30, 0.35));
            }} else {{
                // Just highlight without zoom
                cm.marker.setIcon(createDiamondIcon('#00ffff', 28, 1));
            }}

            // Show right tooltip
            const rightTooltip = document.getElementById('rightTooltip');
            rightTooltip.innerHTML = `
                <div class="tooltip-title">Clicked Point</div>
                <div class="tooltip-row">GPS: (${{cm.lat.toFixed(6)}}, ${{cm.lon.toFixed(6)}})</div>
                <div class="tooltip-row">Distance to camera: ${{distToCamera.toFixed(1)}}m</div>
            `;
            rightTooltip.style.display = 'block';

            // Position right tooltip using viewport coordinates (fixed positioning)
            const positionRightTooltip = () => {{
                const mapContainer = document.getElementById('map');
                const mapRect = mapContainer.getBoundingClientRect();
                const markerLatLng = L.latLng(cm.lat, cm.lon);
                const markerPoint = map.latLngToContainerPoint(markerLatLng);

                // Convert to viewport coordinates
                const tooltipX = mapRect.left + markerPoint.x + 20;
                const tooltipY = mapRect.top + markerPoint.y - 20;

                // Ensure tooltip doesn't go off-screen
                const tooltipRect = rightTooltip.getBoundingClientRect();
                const maxX = window.innerWidth - tooltipRect.width - 10;
                const maxY = window.innerHeight - tooltipRect.height - 10;

                rightTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
                rightTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';
            }};

            if (precisionZoomEnabled) {{
                setTimeout(positionRightTooltip, 350);
            }} else {{
                positionRightTooltip();
            }}

            drawGCPMarkers();
        }}

        // Remove clicked point highlight
        function unhighlightClickedPoint() {{
            if (hoveredClickedIndex === null) return;

            const cm = clickedMarkers[hoveredClickedIndex];
            hoveredClickedIndex = null;

            document.getElementById('leftTooltip').style.display = 'none';
            document.getElementById('rightTooltip').style.display = 'none';

            // Restore original map view if precision mode is on
            if (savedMapView && precisionZoomEnabled) {{
                map.setView(savedMapView.center, savedMapView.zoom, {{ animate: true, duration: 0.3 }});
            }}
            savedMapView = null;

            // Restore marker to original magenta
            if (cm) {{
                cm.marker.setIcon(createDiamondIcon('#ff00ff', 22, 1));
            }}

            drawGCPMarkers();
        }}

        // Override drawGCPMarkers to support hover highlighting
        const originalDrawGCPMarkers = drawGCPMarkers;
        drawGCPMarkers = function() {{
            const displayWidth = img.width;
            const displayHeight = img.height;
            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;

            canvas.width = displayWidth;
            canvas.height = displayHeight;
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            gcpData.forEach((gcp, index) => {{
                const u = gcp.pixel.u * scaleX;
                const v = gcp.pixel.v * scaleY;
                const isHovered = hoveredGcpIndex === index;
                const highlightScale = isHovered ? 1.5 : 1;
                const highlightColor = isHovered ? '#00ffff' : null;

                if (gcp.projected_pixel) {{
                    const projU = gcp.projected_pixel.u * scaleX;
                    const projV = gcp.projected_pixel.v * scaleY;
                    const dx = u - projU;
                    const dy = v - projV;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const threshold = 5;

                    if (distance <= threshold) {{
                        ctx.fillStyle = highlightColor || '#ffd700';
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                    }} else {{
                        ctx.strokeStyle = highlightColor || '#ffff00';
                        ctx.lineWidth = isHovered ? 3 : 2;
                        ctx.beginPath();
                        ctx.moveTo(u, v);
                        ctx.lineTo(projU, projV);
                        ctx.stroke();

                        ctx.fillStyle = highlightColor || '#ff0000';
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(projU, projV, 3 * highlightScale, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();

                        ctx.fillStyle = highlightColor || '#00ff00';
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                    }}
                }} else {{
                    ctx.fillStyle = highlightColor || '#00ff00';
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }}
            }});

            // Draw clicked markers (magenta, or cyan if hovered)
            clickedMarkers.forEach((cm, idx) => {{
                const isHovered = hoveredClickedIndex === idx;
                ctx.fillStyle = isHovered ? '#00ffff' : '#ff00ff';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(cm.displayU, cm.displayV, isHovered ? 9 : 6, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }});
        }};

        // ============================================================
        // Click to Project
        // ============================================================

        // Enable interactive mode if homography is available
        if (homographyMatrix) {{
            canvas.classList.add('interactive');
            document.getElementById('clickInstruction').classList.add('active');

            canvas.addEventListener('click', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const displayX = e.clientX - rect.left;
                const displayY = e.clientY - rect.top;

                // Convert to original image coordinates
                const displayWidth = img.width;
                const displayHeight = img.height;
                const naturalWidth = img.naturalWidth;
                const naturalHeight = img.naturalHeight;
                const scaleX = naturalWidth / displayWidth;
                const scaleY = naturalHeight / displayHeight;
                const origU = displayX * scaleX;
                const origV = displayY * scaleY;

                // Project to GPS
                const result = projectImageToGPS(origU, origV);
                if (result) {{
                    // Add marker to map
                    const marker = L.marker([result.lat, result.lon], {{
                        icon: createDiamondIcon('#ff00ff', 22)
                    }}).addTo(map).bindPopup(
                        `Clicked Point<br>Pixel: (${{origU.toFixed(1)}}, ${{origV.toFixed(1)}})<br>GPS: (${{result.lat.toFixed(6)}}, ${{result.lon.toFixed(6)}})`
                    );

                    const clickedIndex = clickedMarkers.length;
                    clickedMarkers.push({{
                        displayU: displayX,
                        displayV: displayY,
                        origU, origV,
                        lat: result.lat,
                        lon: result.lon,
                        marker
                    }});

                    // Add hover handlers to clicked marker
                    marker.on('mouseover', () => highlightClickedPoint(clickedIndex));
                    marker.on('mouseout', () => unhighlightClickedPoint());

                    // Pan map to show the point
                    map.panTo([result.lat, result.lon]);

                    // Redraw canvas
                    drawGCPMarkers();

                    console.log(`Projected: (${{origU.toFixed(1)}}, ${{origV.toFixed(1)}}) -> (${{result.lat.toFixed(6)}}, ${{result.lon.toFixed(6)}})`);
                }}
            }});

            // Add hover detection on canvas for GCPs and clicked points
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const displayX = e.clientX - rect.left;
                const displayY = e.clientY - rect.top;

                const displayWidth = img.width;
                const displayHeight = img.height;
                const naturalWidth = img.naturalWidth;
                const naturalHeight = img.naturalHeight;
                const scaleX = displayWidth / naturalWidth;
                const scaleY = displayHeight / naturalHeight;

                // Check if hovering over a clicked point first (they're drawn on top)
                let foundClickedIndex = -1;
                clickedMarkers.forEach((cm, index) => {{
                    const dist = Math.sqrt((displayX - cm.displayU) ** 2 + (displayY - cm.displayV) ** 2);
                    if (dist < 15) {{
                        foundClickedIndex = index;
                    }}
                }});

                if (foundClickedIndex >= 0) {{
                    // Unhighlight GCP if we're now on a clicked point
                    if (hoveredGcpIndex !== null) {{
                        unhighlightGCP();
                    }}
                    if (hoveredClickedIndex !== foundClickedIndex) {{
                        highlightClickedPoint(foundClickedIndex);
                    }}
                    return;
                }}

                // Check if hovering over a GCP
                let foundGcpIndex = -1;
                gcpData.forEach((gcp, index) => {{
                    const u = gcp.pixel.u * scaleX;
                    const v = gcp.pixel.v * scaleY;
                    const dist = Math.sqrt((displayX - u) ** 2 + (displayY - v) ** 2);
                    if (dist < 15) {{
                        foundGcpIndex = index;
                    }}
                }});

                if (foundGcpIndex >= 0) {{
                    // Unhighlight clicked point if we're now on a GCP
                    if (hoveredClickedIndex !== null) {{
                        unhighlightClickedPoint();
                    }}
                    if (hoveredGcpIndex !== foundGcpIndex) {{
                        highlightGCP(foundGcpIndex);
                    }}
                }} else {{
                    // Not hovering over anything - unhighlight both
                    if (hoveredGcpIndex !== null) {{
                        unhighlightGCP();
                    }}
                    if (hoveredClickedIndex !== null) {{
                        unhighlightClickedPoint();
                    }}
                }}
            }});
        }} else {{
            document.getElementById('clickInstruction').textContent = 'Interactive mode disabled (no homography)';
        }}

        // ============================================================
        // Control Button Event Listeners
        // ============================================================

        // Precision Zoom Mode toggle button
        document.getElementById('zoomModeBtn').addEventListener('click', togglePrecisionZoom);

        // Reset View button
        document.getElementById('resetViewBtn').addEventListener('click', resetMapView);

        // Replace drawMapMarkers with version that stores refs
        drawMapMarkersWithRefs();
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
    homography_matrix: Optional[List[List[float]]] = None,
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
        homography_matrix: Optional 3x3 inverse homography matrix for interactive projection
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
        validation_results,
        homography_matrix
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
                'projected_pixel': (402.3, 301.5),
                'error_meters': 0.56
            },
            {
                'projected_gps': (39.640618, -0.229805),
                'projected_pixel': (2098.7, 318.2),
                'error_meters': 0.45
            },
            {
                'projected_gps': (39.640398, -0.230002),
                'projected_pixel': (1281.1, 719.8),
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
