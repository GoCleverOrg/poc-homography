#!/usr/bin/env python3
"""
Map Debug Server Module for POC Homography Project.

This module provides a lightweight web server for visualizing GCP validation results
with side-by-side camera frame and map image views. Uses Python's built-in
http.server module for zero-dependency deployment.

Example Usage:
    from poc_homography.map_debug_server import start_server

    # Start server with auto-open browser
    start_server(
        output_dir='output',
        camera_frame_path='output/annotated_frame.jpg',
        map_image_path='output/map_image.jpg',
        gcps=[...],
        validation_results={...},
        auto_open=True
    )
"""

from __future__ import annotations

import http.server
import json
import shutil
import socketserver
import webbrowser
from pathlib import Path
from typing import Any

from poc_homography.server_utils import find_available_port


def generate_html(
    camera_frame_path: str,
    map_image_path: str,
    gcps: list[dict[str, Any]],
    validation_results: dict[str, Any],
    homography_matrix: list[list[float]] | None = None,
) -> str:
    """
    Generate self-contained HTML string for the debug visualization.

    Creates an HTML page with side-by-side camera frame and map image views.
    Both panels include canvas overlays with GCP markers showing pixel coordinates.

    Args:
        camera_frame_path: Relative path to camera frame image (from output_dir)
        map_image_path: Relative path to map image (from output_dir)
        gcps: List of GCP dictionaries with structure:
            {
                'map': {'pixel_x': float, 'pixel_y': float},
                'image': {'u': float, 'v': float},
                'metadata': {'description': str}
            }
        validation_results: Dictionary containing:
            {
                'details': [
                    {
                        'projected_map': (pixel_x, pixel_y),
                        'projected_pixel': (u, v),
                        'error_pixels': float
                    },
                    ...
                ]
            }
        homography_matrix: Optional 3x3 inverse homography matrix (image -> map)
            for interactive projection. If provided, enables click-to-project feature.

    Returns:
        Complete HTML string ready to be written to file

    Example:
        >>> html = generate_html(
        ...     'annotated_frame.jpg',
        ...     'map_image.jpg',
        ...     gcps=[...],
        ...     validation_results={...}
        ... )
    """
    # Prepare GCP data for JavaScript
    gcp_data = []
    for i, gcp in enumerate(gcps):
        map_coords = gcp.get("map", {})
        image = gcp.get("image", {})
        metadata = gcp.get("metadata", {})

        gcp_entry = {
            "name": metadata.get("description", f"GCP {i + 1}"),
            "original_map": {"x": map_coords.get("pixel_x", 0), "y": map_coords.get("pixel_y", 0)},
            "pixel": {"u": image.get("u", 0), "v": image.get("v", 0)},
        }

        # Add projected map coordinates and projected pixel if available
        if validation_results and "details" in validation_results:
            details = validation_results["details"]
            if i < len(details) and "projected_map" in details[i]:
                proj_x, proj_y = details[i]["projected_map"]
                gcp_entry["projected_map"] = {"x": proj_x, "y": proj_y}
                gcp_entry["error_pixels"] = details[i].get("error_pixels", 0)

                # Add projected pixel coordinates if available
                if "projected_pixel" in details[i]:
                    proj_u, proj_v = details[i]["projected_pixel"]
                    gcp_entry["projected_pixel"] = {"u": proj_u, "v": proj_v}

        gcp_data.append(gcp_entry)

    # Extract homography stats for header display
    confidence = validation_results.get("confidence", 0) if validation_results else 0
    inliers = validation_results.get("inliers", 0) if validation_results else 0
    outliers = validation_results.get("outliers", 0) if validation_results else 0
    total_gcps = (
        validation_results.get("gcps_tested", len(gcps)) if validation_results else len(gcps)
    )

    # Map pixel error stats
    mean_map_error = validation_results.get("mean_error_px", 0) if validation_results else 0
    min_map_error = validation_results.get("min_error_px", 0) if validation_results else 0
    max_map_error = validation_results.get("max_error_px", 0) if validation_results else 0

    # Pixel reprojection error stats
    reproj = validation_results.get("reprojection_error", {}) if validation_results else {}
    mean_px_error = reproj.get("mean_px", 0) or 0
    min_px_error = reproj.get("min_px", 0) or 0
    max_px_error = reproj.get("max_px", 0) or 0

    # Determine CSS classes for color coding
    conf_class = "good" if confidence >= 0.7 else ("warn" if confidence >= 0.5 else "bad")
    mean_map_class = (
        "good" if mean_map_error < 5.0 else ("warn" if mean_map_error < 10.0 else "bad")
    )
    mean_px_class = "good" if mean_px_error < 5.0 else ("warn" if mean_px_error < 10.0 else "bad")

    # Convert data to JSON for embedding in HTML
    gcp_data_json = json.dumps(gcp_data)
    homography_json = json.dumps(homography_matrix) if homography_matrix else "null"

    # Generate HTML with embedded data
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCP Map Debug Visualization</title>

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

        /* Image container styling (for both camera and map) */
        .image-container {{
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1a1a1a;
        }}

        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            display: block;
        }}

        .image-container canvas {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
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

        /* Tooltip styling */
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
        .interactive {{
            pointer-events: auto !important;
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
            background: rgba(45, 45, 45, 0.95);
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .control-btn:hover {{
            background: rgba(60, 60, 60, 0.95);
            border-color: #777;
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
                        <span class="stat-label">Map Error:</span>
                        <span class="stat-value {mean_map_class}">{mean_map_error:.2f}px</span>
                        <span class="stat-label">(min: {min_map_error:.2f}px, max: {max_map_error:.2f}px)</span>
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

            <!-- Right Panel: Map Image -->
            <div class="panel">
                <div class="panel-header">Map Image</div>
                <div class="panel-content">
                    <div class="image-container">
                        <img id="mapImage" src="{map_image_path}" alt="Map Image">
                        <canvas id="mapCanvas"></canvas>
                        <div id="rightTooltip" class="gcp-tooltip" style="display: none;"></div>
                    </div>
                    <div class="legend">
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
        </div>
    </div>

    <script>
        // Embedded data
        const gcpData = {gcp_data_json};
        const homographyMatrix = {homography_json};

        // Interactive projection state
        let clickedMarkers = [];  // Store clicked markers
        let hoveredGcpIndex = null;  // Currently hovered GCP index
        let hoveredClickedIndex = null;  // Currently hovered clicked point index

        // Make legends draggable
        function makeDraggable(element) {{
            let isDragging = false;
            let startX, startY, startLeft, startTop;

            element.addEventListener('mousedown', function(e) {{
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;

                const rect = element.getBoundingClientRect();
                const parentRect = element.parentElement.getBoundingClientRect();
                startLeft = rect.left - parentRect.left;
                startTop = rect.top - parentRect.top;

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

        // ============================================================
        // Camera Frame Canvas
        // ============================================================

        const cameraImg = document.getElementById('cameraFrame');
        const gcpCanvas = document.getElementById('gcpCanvas');
        const gcpCtx = gcpCanvas.getContext('2d');

        function drawCameraMarkers() {{
            const displayWidth = cameraImg.width;
            const displayHeight = cameraImg.height;
            const naturalWidth = cameraImg.naturalWidth;
            const naturalHeight = cameraImg.naturalHeight;

            gcpCanvas.width = displayWidth;
            gcpCanvas.height = displayHeight;

            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;

            gcpCtx.clearRect(0, 0, gcpCanvas.width, gcpCanvas.height);

            // Draw GCP markers
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
                        // Draw single gold marker
                        gcpCtx.fillStyle = highlightColor || '#ffd700';
                        gcpCtx.strokeStyle = '#ffffff';
                        gcpCtx.lineWidth = 2;
                        gcpCtx.beginPath();
                        gcpCtx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                        gcpCtx.fill();
                        gcpCtx.stroke();
                    }} else {{
                        // Draw error line
                        gcpCtx.strokeStyle = highlightColor || '#ffff00';
                        gcpCtx.lineWidth = isHovered ? 3 : 2;
                        gcpCtx.beginPath();
                        gcpCtx.moveTo(u, v);
                        gcpCtx.lineTo(projU, projV);
                        gcpCtx.stroke();

                        // Draw projected marker (red)
                        gcpCtx.fillStyle = highlightColor || '#ff0000';
                        gcpCtx.strokeStyle = '#ffffff';
                        gcpCtx.lineWidth = 2;
                        gcpCtx.beginPath();
                        gcpCtx.arc(projU, projV, 3 * highlightScale, 0, 2 * Math.PI);
                        gcpCtx.fill();
                        gcpCtx.stroke();

                        // Draw original marker (green)
                        gcpCtx.fillStyle = highlightColor || '#00ff00';
                        gcpCtx.strokeStyle = '#ffffff';
                        gcpCtx.lineWidth = 2;
                        gcpCtx.beginPath();
                        gcpCtx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                        gcpCtx.fill();
                        gcpCtx.stroke();
                    }}
                }} else {{
                    // No projected pixel - draw only original marker
                    gcpCtx.fillStyle = highlightColor || '#00ff00';
                    gcpCtx.strokeStyle = '#ffffff';
                    gcpCtx.lineWidth = 2;
                    gcpCtx.beginPath();
                    gcpCtx.arc(u, v, 4 * highlightScale, 0, 2 * Math.PI);
                    gcpCtx.fill();
                    gcpCtx.stroke();
                }}
            }});

            // Draw clicked markers
            clickedMarkers.forEach((cm, idx) => {{
                const isHovered = hoveredClickedIndex === idx;
                gcpCtx.fillStyle = isHovered ? '#00ffff' : '#ff00ff';
                gcpCtx.strokeStyle = '#ffffff';
                gcpCtx.lineWidth = 2;
                gcpCtx.beginPath();
                gcpCtx.arc(cm.displayU, cm.displayV, isHovered ? 9 : 6, 0, 2 * Math.PI);
                gcpCtx.fill();
                gcpCtx.stroke();
            }});
        }}

        cameraImg.addEventListener('load', drawCameraMarkers);
        window.addEventListener('resize', drawCameraMarkers);
        if (cameraImg.complete) drawCameraMarkers();

        // ============================================================
        // Map Image Canvas
        // ============================================================

        const mapImg = document.getElementById('mapImage');
        const mapCanvas = document.getElementById('mapCanvas');
        const mapCtx = mapCanvas.getContext('2d');

        function drawMapMarkers() {{
            const displayWidth = mapImg.width;
            const displayHeight = mapImg.height;
            const naturalWidth = mapImg.naturalWidth;
            const naturalHeight = mapImg.naturalHeight;

            mapCanvas.width = displayWidth;
            mapCanvas.height = displayHeight;

            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;

            mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);

            // Draw GCP markers
            gcpData.forEach((gcp, index) => {{
                const x = gcp.original_map.x * scaleX;
                const y = gcp.original_map.y * scaleY;
                const isHovered = hoveredGcpIndex === index;
                const highlightScale = isHovered ? 1.5 : 1;
                const highlightColor = isHovered ? '#00ffff' : null;

                if (gcp.projected_map) {{
                    const projX = gcp.projected_map.x * scaleX;
                    const projY = gcp.projected_map.y * scaleY;

                    const dx = x - projX;
                    const dy = y - projY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const threshold = 5;

                    if (distance <= threshold) {{
                        // Draw single gold marker
                        mapCtx.fillStyle = highlightColor || '#ffd700';
                        mapCtx.strokeStyle = '#ffffff';
                        mapCtx.lineWidth = 2;
                        mapCtx.beginPath();
                        mapCtx.arc(x, y, 4 * highlightScale, 0, 2 * Math.PI);
                        mapCtx.fill();
                        mapCtx.stroke();
                    }} else {{
                        // Draw error line
                        mapCtx.strokeStyle = highlightColor || '#ffff00';
                        mapCtx.lineWidth = isHovered ? 3 : 2;
                        mapCtx.beginPath();
                        mapCtx.moveTo(x, y);
                        mapCtx.lineTo(projX, projY);
                        mapCtx.stroke();

                        // Draw projected marker (red)
                        mapCtx.fillStyle = highlightColor || '#ff0000';
                        mapCtx.strokeStyle = '#ffffff';
                        mapCtx.lineWidth = 2;
                        mapCtx.beginPath();
                        mapCtx.arc(projX, projY, 3 * highlightScale, 0, 2 * Math.PI);
                        mapCtx.fill();
                        mapCtx.stroke();

                        // Draw original marker (green)
                        mapCtx.fillStyle = highlightColor || '#00ff00';
                        mapCtx.strokeStyle = '#ffffff';
                        mapCtx.lineWidth = 2;
                        mapCtx.beginPath();
                        mapCtx.arc(x, y, 4 * highlightScale, 0, 2 * Math.PI);
                        mapCtx.fill();
                        mapCtx.stroke();
                    }}
                }} else {{
                    // No projected map - draw only original marker
                    mapCtx.fillStyle = highlightColor || '#00ff00';
                    mapCtx.strokeStyle = '#ffffff';
                    mapCtx.lineWidth = 2;
                    mapCtx.beginPath();
                    mapCtx.arc(x, y, 4 * highlightScale, 0, 2 * Math.PI);
                    mapCtx.fill();
                    mapCtx.stroke();
                }}
            }});

            // Draw clicked markers
            clickedMarkers.forEach((cm, idx) => {{
                const isHovered = hoveredClickedIndex === idx;
                mapCtx.fillStyle = isHovered ? '#00ffff' : '#ff00ff';
                mapCtx.strokeStyle = '#ffffff';
                mapCtx.lineWidth = 2;
                mapCtx.beginPath();
                mapCtx.arc(cm.mapX, cm.mapY, isHovered ? 9 : 6, 0, 2 * Math.PI);
                mapCtx.fill();
                mapCtx.stroke();
            }});
        }}

        mapImg.addEventListener('load', drawMapMarkers);
        window.addEventListener('resize', drawMapMarkers);
        if (mapImg.complete) drawMapMarkers();

        // ============================================================
        // Interactive Projection
        // ============================================================

        function projectImageToMap(u, v) {{
            if (!homographyMatrix) return null;

            const H = homographyMatrix;
            const w = H[2][0] * u + H[2][1] * v + H[2][2];

            if (Math.abs(w) < 1e-10) return null;

            const map_x = (H[0][0] * u + H[0][1] * v + H[0][2]) / w;
            const map_y = (H[1][0] * u + H[1][1] * v + H[1][2]) / w;

            return {{ map_x, map_y }};
        }}

        // Highlight GCP on both panels
        function highlightGCP(index) {{
            hoveredGcpIndex = index;
            const gcp = gcpData[index];
            if (!gcp) return;

            // Show left tooltip (pixel info)
            const leftTooltip = document.getElementById('leftTooltip');

            if (gcp.projected_pixel) {{
                const dx = gcp.pixel.u - gcp.projected_pixel.u;
                const dy = gcp.pixel.v - gcp.projected_pixel.v;
                const pixelError = Math.sqrt(dx * dx + dy * dy);
                leftTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Pixel: (${{gcp.pixel.u.toFixed(1)}}, ${{gcp.pixel.v.toFixed(1)}})</div>
                    <div class="tooltip-row">Pixel Error: ${{pixelError.toFixed(2)}}px</div>
                `;
            }} else {{
                leftTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Pixel: (${{gcp.pixel.u.toFixed(1)}}, ${{gcp.pixel.v.toFixed(1)}})</div>
                `;
            }}
            leftTooltip.style.display = 'block';

            // Position left tooltip
            const displayWidth = cameraImg.width;
            const displayHeight = cameraImg.height;
            const naturalWidth = cameraImg.naturalWidth;
            const naturalHeight = cameraImg.naturalHeight;
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;
            const u = gcp.pixel.u * scaleX;
            const v = gcp.pixel.v * scaleY;

            const imgRect = cameraImg.getBoundingClientRect();
            const tooltipX = imgRect.left + u + 15;
            const tooltipY = imgRect.top + v - 30;

            const tooltipRect = leftTooltip.getBoundingClientRect();
            const maxX = window.innerWidth - tooltipRect.width - 10;
            const maxY = window.innerHeight - tooltipRect.height - 10;

            leftTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
            leftTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';

            // Show right tooltip (map info)
            const rightTooltip = document.getElementById('rightTooltip');
            if (gcp.projected_map) {{
                const dx = gcp.original_map.x - gcp.projected_map.x;
                const dy = gcp.original_map.y - gcp.projected_map.y;
                const mapError = Math.sqrt(dx * dx + dy * dy);
                rightTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Map: (${{gcp.original_map.x.toFixed(1)}}, ${{gcp.original_map.y.toFixed(1)}})</div>
                    <div class="tooltip-row">Map Error: ${{mapError.toFixed(2)}}px</div>
                `;
            }} else {{
                rightTooltip.innerHTML = `
                    <div class="tooltip-title">${{gcp.name}}</div>
                    <div class="tooltip-row">Map: (${{gcp.original_map.x.toFixed(1)}}, ${{gcp.original_map.y.toFixed(1)}})</div>
                `;
            }}
            rightTooltip.style.display = 'block';

            // Position right tooltip
            const mapImgRect = mapImg.getBoundingClientRect();
            const mapDisplayWidth = mapImg.width;
            const mapDisplayHeight = mapImg.height;
            const mapNaturalWidth = mapImg.naturalWidth;
            const mapNaturalHeight = mapImg.naturalHeight;
            const mapScaleX = mapDisplayWidth / mapNaturalWidth;
            const mapScaleY = mapDisplayHeight / mapNaturalHeight;
            const mapX = gcp.original_map.x * mapScaleX;
            const mapY = gcp.original_map.y * mapScaleY;

            const mapTooltipX = mapImgRect.left + mapX + 15;
            const mapTooltipY = mapImgRect.top + mapY - 30;

            const rightTooltipRect = rightTooltip.getBoundingClientRect();
            const rightMaxX = window.innerWidth - rightTooltipRect.width - 10;
            const rightMaxY = window.innerHeight - rightTooltipRect.height - 10;

            rightTooltip.style.left = Math.min(mapTooltipX, rightMaxX) + 'px';
            rightTooltip.style.top = Math.max(10, Math.min(mapTooltipY, rightMaxY)) + 'px';

            drawCameraMarkers();
            drawMapMarkers();
        }}

        function unhighlightGCP() {{
            hoveredGcpIndex = null;
            document.getElementById('leftTooltip').style.display = 'none';
            document.getElementById('rightTooltip').style.display = 'none';
            drawCameraMarkers();
            drawMapMarkers();
        }}

        // Hover detection on camera canvas
        gcpCanvas.addEventListener('mousemove', (e) => {{
            const rect = gcpCanvas.getBoundingClientRect();
            const displayX = e.clientX - rect.left;
            const displayY = e.clientY - rect.top;

            const displayWidth = cameraImg.width;
            const displayHeight = cameraImg.height;
            const naturalWidth = cameraImg.naturalWidth;
            const naturalHeight = cameraImg.naturalHeight;
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayHeight / naturalHeight;

            // Check clicked points first
            let foundClickedIndex = -1;
            clickedMarkers.forEach((cm, index) => {{
                const dist = Math.sqrt((displayX - cm.displayU) ** 2 + (displayY - cm.displayV) ** 2);
                if (dist < 15) foundClickedIndex = index;
            }});

            if (foundClickedIndex >= 0) {{
                if (hoveredGcpIndex !== null) unhighlightGCP();
                if (hoveredClickedIndex !== foundClickedIndex) highlightClickedPoint(foundClickedIndex);
                return;
            }}

            // Check GCPs
            let foundGcpIndex = -1;
            gcpData.forEach((gcp, index) => {{
                const u = gcp.pixel.u * scaleX;
                const v = gcp.pixel.v * scaleY;
                const dist = Math.sqrt((displayX - u) ** 2 + (displayY - v) ** 2);
                if (dist < 15) foundGcpIndex = index;
            }});

            if (foundGcpIndex >= 0) {{
                if (hoveredClickedIndex !== null) unhighlightClickedPoint();
                if (hoveredGcpIndex !== foundGcpIndex) highlightGCP(foundGcpIndex);
            }} else {{
                if (hoveredGcpIndex !== null) unhighlightGCP();
                if (hoveredClickedIndex !== null) unhighlightClickedPoint();
            }}
        }});

        // Hover detection on map canvas
        mapCanvas.addEventListener('mousemove', (e) => {{
            const rect = mapCanvas.getBoundingClientRect();
            const displayX = e.clientX - rect.left;
            const displayY = e.clientY - rect.top;

            const displayWidth = mapImg.width;
            const naturalWidth = mapImg.naturalWidth;
            const scaleX = displayWidth / naturalWidth;
            const scaleY = displayWidth / naturalWidth;

            // Check clicked points first
            let foundClickedIndex = -1;
            clickedMarkers.forEach((cm, index) => {{
                const dist = Math.sqrt((displayX - cm.mapX) ** 2 + (displayY - cm.mapY) ** 2);
                if (dist < 15) foundClickedIndex = index;
            }});

            if (foundClickedIndex >= 0) {{
                if (hoveredGcpIndex !== null) unhighlightGCP();
                if (hoveredClickedIndex !== foundClickedIndex) highlightClickedPoint(foundClickedIndex);
                return;
            }}

            // Check GCPs
            let foundGcpIndex = -1;
            gcpData.forEach((gcp, index) => {{
                const x = gcp.original_map.x * scaleX;
                const y = gcp.original_map.y * scaleY;
                const dist = Math.sqrt((displayX - x) ** 2 + (displayY - y) ** 2);
                if (dist < 15) foundGcpIndex = index;
            }});

            if (foundGcpIndex >= 0) {{
                if (hoveredClickedIndex !== null) unhighlightClickedPoint();
                if (hoveredGcpIndex !== foundGcpIndex) highlightGCP(foundGcpIndex);
            }} else {{
                if (hoveredGcpIndex !== null) unhighlightGCP();
                if (hoveredClickedIndex !== null) unhighlightClickedPoint();
            }}
        }});

        function highlightClickedPoint(index) {{
            hoveredClickedIndex = index;
            const cm = clickedMarkers[index];
            if (!cm) return;

            const leftTooltip = document.getElementById('leftTooltip');
            leftTooltip.innerHTML = `
                <div class="tooltip-title">Clicked Point</div>
                <div class="tooltip-row">Pixel: (${{cm.origU.toFixed(1)}}, ${{cm.origV.toFixed(1)}})</div>
                <div class="tooltip-row">Map: (${{cm.map_x.toFixed(1)}}, ${{cm.map_y.toFixed(1)}})</div>
            `;
            leftTooltip.style.display = 'block';

            const imgRect = cameraImg.getBoundingClientRect();
            const tooltipX = imgRect.left + cm.displayU + 15;
            const tooltipY = imgRect.top + cm.displayV - 30;

            const tooltipRect = leftTooltip.getBoundingClientRect();
            const maxX = window.innerWidth - tooltipRect.width - 10;
            const maxY = window.innerHeight - tooltipRect.height - 10;

            leftTooltip.style.left = Math.min(tooltipX, maxX) + 'px';
            leftTooltip.style.top = Math.max(10, Math.min(tooltipY, maxY)) + 'px';

            const rightTooltip = document.getElementById('rightTooltip');
            rightTooltip.innerHTML = `
                <div class="tooltip-title">Clicked Point</div>
                <div class="tooltip-row">Map: (${{cm.map_x.toFixed(1)}}, ${{cm.map_y.toFixed(1)}})</div>
            `;
            rightTooltip.style.display = 'block';

            const mapImgRect = mapImg.getBoundingClientRect();
            const mapTooltipX = mapImgRect.left + cm.mapX + 15;
            const mapTooltipY = mapImgRect.top + cm.mapY - 30;

            const rightTooltipRect = rightTooltip.getBoundingClientRect();
            const rightMaxX = window.innerWidth - rightTooltipRect.width - 10;
            const rightMaxY = window.innerHeight - rightTooltipRect.height - 10;

            rightTooltip.style.left = Math.min(mapTooltipX, rightMaxX) + 'px';
            rightTooltip.style.top = Math.max(10, Math.min(mapTooltipY, rightMaxY)) + 'px';

            drawCameraMarkers();
            drawMapMarkers();
        }}

        function unhighlightClickedPoint() {{
            hoveredClickedIndex = null;
            document.getElementById('leftTooltip').style.display = 'none';
            document.getElementById('rightTooltip').style.display = 'none';
            drawCameraMarkers();
            drawMapMarkers();
        }}

        // Click to project
        if (homographyMatrix) {{
            gcpCanvas.classList.add('interactive');
            document.getElementById('clickInstruction').classList.add('active');

            gcpCanvas.addEventListener('click', (e) => {{
                const rect = gcpCanvas.getBoundingClientRect();
                const displayX = e.clientX - rect.left;
                const displayY = e.clientY - rect.top;

                const displayWidth = cameraImg.width;
                const displayHeight = cameraImg.height;
                const naturalWidth = cameraImg.naturalWidth;
                const naturalHeight = cameraImg.naturalHeight;
                const scaleX = naturalWidth / displayWidth;
                const scaleY = naturalHeight / displayHeight;
                const origU = displayX * scaleX;
                const origV = displayY * scaleY;

                const result = projectImageToMap(origU, origV);
                if (result) {{
                    const mapDisplayWidth = mapImg.width;
                    const mapNaturalWidth = mapImg.naturalWidth;
                    const mapScaleX = mapDisplayWidth / mapNaturalWidth;
                    const mapScaleY = mapDisplayWidth / mapNaturalWidth;

                    clickedMarkers.push({{
                        displayU: displayX,
                        displayV: displayY,
                        origU, origV,
                        map_x: result.map_x,
                        map_y: result.map_y,
                        mapX: result.map_x * mapScaleX,
                        mapY: result.map_y * mapScaleY
                    }});

                    drawCameraMarkers();
                    drawMapMarkers();

                    console.log(`Projected: (${{origU.toFixed(1)}}, ${{origV.toFixed(1)}}) -> (${{result.map_x.toFixed(1)}}, ${{result.map_y.toFixed(1)}})`);
                }}
            }});
        }} else {{
            document.getElementById('clickInstruction').textContent = 'Interactive mode disabled (no homography)';
        }}
    </script>
</body>
</html>
"""

    return html


def start_server(
    output_dir: str,
    camera_frame_path: str,
    map_image_path: str,
    gcps: list[dict[str, Any]],
    validation_results: dict[str, Any],
    homography_matrix: list[list[float]] | None = None,
    auto_open: bool = True,
) -> None:
    """
    Start the map debug visualization web server.

    Creates a lightweight HTTP server serving the debug visualization HTML page.
    Automatically finds an available port, generates HTML with embedded data,
    and optionally opens the browser.

    Args:
        output_dir: Directory where HTML and assets will be served from
        camera_frame_path: Absolute path to camera frame image
        map_image_path: Absolute path to map image
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
        ...     map_image_path='output/map_image.jpg',
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
    map_image_src = Path(map_image_path).resolve()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine relative paths for HTML
    camera_frame_filename = camera_frame_src.name
    map_image_filename = map_image_src.name

    # Copy camera frame to output dir if not already there
    camera_frame_dest = output_path / camera_frame_filename
    if camera_frame_src != camera_frame_dest:
        if camera_frame_src.exists():
            shutil.copy2(camera_frame_src, camera_frame_dest)
            print(f"Copied camera frame to {camera_frame_dest}")
        else:
            raise OSError(f"Camera frame not found: {camera_frame_src}")

    # Copy map image to output dir if not already there
    map_image_dest = output_path / map_image_filename
    if map_image_src != map_image_dest:
        if map_image_src.exists():
            shutil.copy2(map_image_src, map_image_dest)
            print(f"Copied map image to {map_image_dest}")
        else:
            raise OSError(f"Map image not found: {map_image_src}")

    # Generate HTML
    html_content = generate_html(
        camera_frame_filename,  # Use relative path in HTML
        map_image_filename,  # Use relative path in HTML
        gcps,
        validation_results,
        homography_matrix,
    )

    # Write HTML to output directory
    html_path = output_path / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
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


if __name__ == "__main__":
    """
    Example usage and test of the map debug server.
    """
    # Create sample data
    sample_gcps = [
        {
            "map": {"pixel_x": 1234.5, "pixel_y": 5678.9},
            "image": {"u": 400.0, "v": 300.0},
            "metadata": {"description": "P#01"},
        },
        {
            "map": {"pixel_x": 2345.6, "pixel_y": 6789.0},
            "image": {"u": 2100.0, "v": 320.0},
            "metadata": {"description": "P#02"},
        },
        {
            "map": {"pixel_x": 3456.7, "pixel_y": 7890.1},
            "image": {"u": 1280.0, "v": 720.0},
            "metadata": {"description": "P#03"},
        },
    ]

    sample_results = {
        "details": [
            {
                "projected_map": (1236.8, 5680.4),
                "projected_pixel": (402.3, 301.5),
                "error_pixels": 2.8,
            },
            {
                "projected_map": (2347.3, 6790.5),
                "projected_pixel": (2098.7, 318.2),
                "error_pixels": 2.1,
            },
            {
                "projected_map": (3457.8, 7891.0),
                "projected_pixel": (1281.1, 719.8),
                "error_pixels": 1.4,
            },
        ]
    }

    # Get project root
    module_dir = Path(__file__).parent
    project_root = module_dir.parent
    output_dir = project_root / "output"

    # Create a dummy camera frame for testing
    import cv2
    import numpy as np

    dummy_frame_path = output_dir / "dummy_frame.jpg"
    dummy_map_path = output_dir / "dummy_map.jpg"

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

    if not dummy_map_path.exists():
        # Create a simple test map
        map_img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        map_img[:, :] = (40, 60, 40)  # Dark greenish background

        # Draw grid
        for i in range(0, 3000, 200):
            cv2.line(map_img, (i, 0), (i, 2000), (60, 90, 60), 1)
        for i in range(0, 2000, 200):
            cv2.line(map_img, (0, i), (3000, i), (60, 90, 60), 1)

        cv2.imwrite(str(dummy_map_path), map_img)
        print(f"Created dummy map at {dummy_map_path}")

    # Start server
    print("Starting map debug server with example data...")
    start_server(
        output_dir=str(output_dir),
        camera_frame_path=str(dummy_frame_path),
        map_image_path=str(dummy_map_path),
        gcps=sample_gcps,
        validation_results=sample_results,
        auto_open=True,
    )
