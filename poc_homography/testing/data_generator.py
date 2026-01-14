"""
Test data generator for camera calibration GCPs.

This module captures full-resolution camera frames from PTZ cameras and enables
interactive marking of Ground Control Points (GCPs) with map point references.
Automatically fetches camera parameters (pan/tilt/zoom, GPS, height) and exports
test data in JSON format.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

from poc_homography.application import ApplicationContext
from poc_homography.camera.intrinsics import get_ptz_status
from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.server_utils import find_available_port


def fetch_ptz_status(ip_address: str, username: str, password: str) -> dict[str, float]:
    """
    Fetch current PTZ status from camera.

    Args:
        ip_address: Camera IP address
        username: Camera username
        password: Camera password

    Returns:
        Dictionary with pan_deg, tilt_deg, zoom_level

    Raises:
        RuntimeError: If PTZ status cannot be fetched
    """
    try:
        ptz_data = get_ptz_status(ip_address, username, password, timeout=10.0)

        return {
            "pan_deg": float(ptz_data.pan),
            "tilt_deg": float(ptz_data.tilt),
            "zoom_level": float(ptz_data.zoom),
        }
    except RuntimeError as e:
        raise RuntimeError(f"Failed to fetch PTZ status: {e}") from e


def capture_frame_from_rtsp(rtsp_url: str, timeout_sec: float = 10.0) -> str:
    """
    Capture a single frame from RTSP camera stream.

    Args:
        rtsp_url: Full RTSP URL for the camera stream
        timeout_sec: Timeout for capture operation in seconds

    Returns:
        Path to saved frame image file

    Raises:
        RuntimeError: If frame capture fails
    """
    print("Connecting to RTSP stream...")

    # Open video capture
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay

    # Set timeout by reading with retries
    import time

    start_time = time.time()
    frame = None

    while time.time() - start_time < timeout_sec:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    cap.release()

    if frame is None:
        raise RuntimeError(
            f"Failed to capture frame within {timeout_sec}s timeout. "
            f"Please check camera connectivity and RTSP stream availability."
        )

    # Save frame to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]} pixels")
    print(f"Saved to: {temp_path}")

    return temp_path


def generate_json_output(
    camera_info: dict[str, float],
    gcps: list[dict[str, Any]],
    camera_name: str,
    output_path: str | None = None,
    frame_path: str | None = None,
) -> dict[str, str | None]:
    """
    Generate JSON output file with camera info and GCPs, and copy the frame image.

    Args:
        camera_info: Dictionary with camera parameters
        gcps: List of GCP dictionaries
        camera_name: Name of the camera
        output_path: Optional custom output path for JSON
        frame_path: Optional path to the captured frame image

    Returns:
        Dictionary with 'json_path' and 'image_path' keys
    """
    # Generate default filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"test_data_{camera_name}_{timestamp}.json"

    # Construct output data
    data = {"camera_info": camera_info, "gcps": gcps}

    # Write JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    result: dict[str, str | None] = {"json_path": output_path, "image_path": None}

    # Copy frame image with matching filename
    if frame_path and os.path.exists(frame_path):
        # Replace .json with .jpg for the image filename
        image_output_path = output_path.rsplit(".json", 1)[0] + ".jpg"
        shutil.copy2(frame_path, image_output_path)
        result["image_path"] = image_output_path

    return result


def load_map_points(map_points_path: str | Path) -> dict[str, GroundControlPoint]:
    """
    Load map points from YAML file.

    Args:
        map_points_path: Path to map points file (.yaml or .yml)

    Returns:
        Dictionary mapping GCP name to GroundControlPoint entity.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not .yaml or .yml
        yaml.YAMLError: If YAML is invalid
        KeyError: If required keys are missing
    """
    import yaml

    file_path = Path(map_points_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Map points file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return {}

    map_id = data.get("map_id", file_path.stem)
    points_data = data.get("points", [])

    gcps: dict[str, GroundControlPoint] = {}
    for point_data in points_data:
        name = str(point_data["id"])
        pixel_x = float(point_data["pixel_x"])
        pixel_y = float(point_data["pixel_y"])

        pixel_point = PixelPoint(_x=pixel_x, _y=pixel_y)
        map_point = MapPoint(map_id=map_id, pixel_point=pixel_point)
        gcp = GroundControlPoint(id=name, name=name, map_point=map_point)
        gcps[name] = gcp

    return gcps


def convert_map_points_to_list(registry: dict[str, GroundControlPoint]) -> list[dict[str, Any]]:
    """
    Convert GCP dictionary to list format for web interface.

    Args:
        registry: Dictionary of GroundControlPoint entities.

    Returns:
        List of dictionaries with pixel_x, pixel_y, map_id keys
    """
    return [
        {
            "pixel_x": float(gcp.map_point.pixel_point.x),
            "pixel_y": float(gcp.map_point.pixel_point.y),
            "map_id": gcp.map_id,
        }
        for gcp in registry.values()
    ]


def create_html_interface() -> str:
    """
    Create HTML interface for interactive GCP marking.

    Returns:
        HTML string with embedded JavaScript
    """
    return """<!DOCTYPE html>
<html>
<head>
    <title>Test Data Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        #canvas-container {
            position: relative;
            display: inline-block;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #image-canvas {
            border: 1px solid #ccc;
            cursor: crosshair;
        }
        #controls {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .params-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }
        .param-group label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }
        .param-group input {
            width: 100%;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        button {
            padding: 10px 20px;
            margin-right: 10px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        #export-btn {
            background-color: #4CAF50;
            color: white;
        }
        #export-btn:hover {
            background-color: #45a049;
        }
        #clear-btn {
            background-color: #f44336;
            color: white;
        }
        #clear-btn:hover {
            background-color: #da190b;
        }
        #gcp-list {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .gcp-item {
            padding: 10px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-left: 3px solid #4CAF50;
            cursor: pointer;
        }
        .gcp-item:hover {
            background-color: #e8f5e9;
        }
        .gcp-item.selected {
            background-color: #c8e6c9;
            border-left-color: #2196F3;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: white;
            margin: 15% auto;
            padding: 20px;
            border-radius: 5px;
            width: 400px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .modal-content h3 {
            margin-top: 0;
        }
        .modal-content input {
            width: 100%;
            padding: 8px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .modal-buttons {
            margin-top: 15px;
            text-align: right;
        }
        .modal-buttons button {
            padding: 8px 16px;
        }
    </style>
</head>
<body>
    <h1>Test Data Generator - <span id="camera-name"></span></h1>

    <div id="canvas-container">
        <canvas id="image-canvas"></canvas>
    </div>

    <div id="controls">
        <h3>Camera Parameters</h3>
        <div class="params-grid">
            <div class="param-group">
                <label>Latitude (deg):</label>
                <input type="number" id="cam-lat" step="any" />
            </div>
            <div class="param-group">
                <label>Longitude (deg):</label>
                <input type="number" id="cam-lon" step="any" />
            </div>
            <div class="param-group">
                <label>Height (m):</label>
                <input type="number" id="cam-height" step="any" />
            </div>
            <div class="param-group">
                <label>Pan (deg):</label>
                <input type="number" id="cam-pan" step="any" />
            </div>
            <div class="param-group">
                <label>Tilt (deg):</label>
                <input type="number" id="cam-tilt" step="any" />
            </div>
            <div class="param-group">
                <label>Zoom:</label>
                <input type="number" id="cam-zoom" step="any" />
            </div>
        </div>

        <div style="margin-top: 20px;">
            <button id="export-btn">Export JSON</button>
            <button id="clear-btn">Clear All GCPs</button>
            <span id="status-msg" style="margin-left: 20px; color: #666;"></span>
        </div>
    </div>

    <div id="gcp-list">
        <h3>Ground Control Points (<span id="gcp-count">0</span>)</h3>
        <div id="gcp-items"></div>
    </div>

    <!-- Modal for selecting map points -->
    <div id="gps-modal" class="modal">
        <div class="modal-content">
            <h3 id="modal-title">Select Map Point</h3>

            <!-- Map point search -->
            <div id="map-point-search-container" style="margin-bottom: 15px; padding: 10px; background: #f0f7ff; border-radius: 5px;">
                <label style="font-weight: bold; color: #1976D2;">Search Map Points:</label>
                <input type="text" id="map-point-search" placeholder="Type to filter by ID..." style="width: 100%; padding: 8px; margin: 5px 0; border: 2px solid #1976D2; border-radius: 3px;" />
                <div id="map-point-results" style="max-height: 150px; overflow-y: auto; border: 1px solid #ddd; border-radius: 3px; background: white;"></div>
            </div>

            <div class="modal-buttons">
                <button id="modal-cancel" style="background-color: #999; color: white;">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let gcps = [];
        let selectedGcpIndex = null;
        let pendingPixelCoords = null;
        let isDragging = false;
        let dragGcpIndex = null;
        let cameraInfo = {};
        let cameraName = '';
        let mapPoints = [];  // Map points with {id, pixel_x, pixel_y, map_id}

        // Canvas and image
        const canvas = document.getElementById('image-canvas');
        const ctx = canvas.getContext('2d');
        let img = new Image();

        // Load initial data
        fetch('/api/init')
            .then(r => r.json())
            .then(data => {
                cameraInfo = data.camera_info;
                cameraName = data.camera_name;
                mapPoints = data.map_points || [];

                document.getElementById('camera-name').textContent = cameraName;
                document.getElementById('cam-lat').value = cameraInfo.latitude;
                document.getElementById('cam-lon').value = cameraInfo.longitude;
                document.getElementById('cam-height').value = cameraInfo.height_meters;
                document.getElementById('cam-pan').value = cameraInfo.pan_deg;
                document.getElementById('cam-tilt').value = cameraInfo.tilt_deg;
                document.getElementById('cam-zoom').value = cameraInfo.zoom_level;

                // Load image
                img.src = '/api/image';
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    drawCanvas();
                };
            });

        // Map point search functionality
        function filterMapPoints(query) {
            const resultsContainer = document.getElementById('map-point-results');
            resultsContainer.innerHTML = '';

            if (!query || query.length < 1) {
                // Show all points when query is empty
                const matches = mapPoints.slice(0, 20);  // Show first 20
                displayMapPointMatches(matches);
                return;
            }

            const lowerQuery = query.toLowerCase();
            const matches = mapPoints.filter(p =>
                p.id.toLowerCase().includes(lowerQuery)
            ).slice(0, 20);  // Limit to 20 results

            displayMapPointMatches(matches);
        }

        function displayMapPointMatches(matches) {
            const resultsContainer = document.getElementById('map-point-results');
            resultsContainer.innerHTML = '';

            if (matches.length === 0) {
                resultsContainer.innerHTML = '<div style="padding: 8px; color: #999;">No matches found</div>';
                return;
            }

            matches.forEach(point => {
                const div = document.createElement('div');
                div.style.cssText = 'padding: 8px; cursor: pointer; border-bottom: 1px solid #eee;';
                div.innerHTML = `<strong>${point.id}</strong><br><small style="color: #666;">Pixel: (${point.pixel_x.toFixed(1)}, ${point.pixel_y.toFixed(1)})</small>`;
                div.onmouseover = () => div.style.backgroundColor = '#e3f2fd';
                div.onmouseout = () => div.style.backgroundColor = 'white';
                div.onclick = () => selectMapPoint(point);
                resultsContainer.appendChild(div);
            });
        }

        function selectMapPoint(point) {
            document.getElementById('map-point-search').value = point.id;
            document.getElementById('map-point-results').innerHTML = '';

            // Add or update GCP
            if (selectedGcpIndex !== null) {
                // Edit existing GCP
                gcps[selectedGcpIndex].map_point_id = point.id;
            } else {
                // Add new GCP
                gcps.push({
                    pixel_x: pendingPixelCoords.x,
                    pixel_y: pendingPixelCoords.y,
                    map_point_id: point.id
                });
            }

            updateGcpList();
            drawCanvas();
            hideGpsModal();
            selectedGcpIndex = null;
        }

        // Set up map point search event listener
        document.getElementById('map-point-search').addEventListener('input', (e) => {
            filterMapPoints(e.target.value);
        });

        // Show all map points when search is focused
        document.getElementById('map-point-search').addEventListener('focus', (e) => {
            filterMapPoints(e.target.value);
        });

        function drawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            // Draw GCP markers
            gcps.forEach((gcp, idx) => {
                const isSelected = idx === selectedGcpIndex;
                const size = isSelected ? 10 : 8;
                const color = isSelected ? '#2196F3' : '#4CAF50';

                // Draw crosshair
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(gcp.pixel_x - size, gcp.pixel_y);
                ctx.lineTo(gcp.pixel_x + size, gcp.pixel_y);
                ctx.moveTo(gcp.pixel_x, gcp.pixel_y - size);
                ctx.lineTo(gcp.pixel_x, gcp.pixel_y + size);
                ctx.stroke();

                // Draw circle
                ctx.beginPath();
                ctx.arc(gcp.pixel_x, gcp.pixel_y, size, 0, 2 * Math.PI);
                ctx.stroke();

                // Draw label
                ctx.fillStyle = color;
                ctx.font = 'bold 12px Arial';
                ctx.fillText(`#${idx + 1}`, gcp.pixel_x + size + 3, gcp.pixel_y - size);
            });
        }

        function showGpsModal(pixelX, pixelY, existingGcp = null) {
            pendingPixelCoords = { x: pixelX, y: pixelY };

            const modal = document.getElementById('gps-modal');
            const title = document.getElementById('modal-title');
            const mapPointSearch = document.getElementById('map-point-search');
            const mapPointResults = document.getElementById('map-point-results');

            // Clear map point search
            mapPointSearch.value = '';
            mapPointResults.innerHTML = '';

            if (existingGcp) {
                title.textContent = 'Edit Map Point';
                if (existingGcp.map_point_id) {
                    mapPointSearch.value = existingGcp.map_point_id;
                }
            } else {
                title.textContent = 'Select Map Point';
            }

            modal.style.display = 'block';
            mapPointSearch.focus();
        }

        function hideGpsModal() {
            document.getElementById('gps-modal').style.display = 'none';
            document.getElementById('map-point-search').value = '';
            document.getElementById('map-point-results').innerHTML = '';
            pendingPixelCoords = null;
        }

        function updateGcpList() {
            const container = document.getElementById('gcp-items');
            const count = document.getElementById('gcp-count');

            count.textContent = gcps.length;
            container.innerHTML = '';

            gcps.forEach((gcp, idx) => {
                const div = document.createElement('div');
                div.className = 'gcp-item';
                if (idx === selectedGcpIndex) {
                    div.classList.add('selected');
                }
                div.innerHTML = `
                    <strong>GCP #${idx + 1}</strong><br>
                    Pixel: (${gcp.pixel_x.toFixed(1)}, ${gcp.pixel_y.toFixed(1)})<br>
                    Map Point: ${gcp.map_point_id || 'None'}
                `;
                div.onclick = () => selectGcp(idx);
                container.appendChild(div);
            });
        }

        function selectGcp(idx) {
            selectedGcpIndex = idx;
            updateGcpList();
            drawCanvas();
        }

        function deleteSelectedGcp() {
            if (selectedGcpIndex !== null) {
                gcps.splice(selectedGcpIndex, 1);
                selectedGcpIndex = null;
                updateGcpList();
                drawCanvas();
            }
        }

        function exportJson() {
            // Get camera parameters from form
            const exportData = {
                camera_info: {
                    latitude: parseFloat(document.getElementById('cam-lat').value),
                    longitude: parseFloat(document.getElementById('cam-lon').value),
                    height_meters: parseFloat(document.getElementById('cam-height').value),
                    pan_deg: parseFloat(document.getElementById('cam-pan').value),
                    tilt_deg: parseFloat(document.getElementById('cam-tilt').value),
                    zoom_level: parseFloat(document.getElementById('cam-zoom').value)
                },
                gcps: gcps
            };

            fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(exportData)
            })
            .then(r => {
                if (!r.ok) {
                    throw new Error(`HTTP error! status: ${r.status}`);
                }
                return r.json();
            })
            .then(data => {
                let msg = `Exported: ${data.json_path}`;
                if (data.image_path) {
                    msg += ` + ${data.image_path}`;
                }
                document.getElementById('status-msg').textContent = msg;
                document.getElementById('status-msg').style.color = '#4CAF50';
                setTimeout(() => {
                    document.getElementById('status-msg').textContent = '';
                }, 8000);
            })
            .catch(err => {
                document.getElementById('status-msg').textContent = `Export failed: ${err.message}`;
                document.getElementById('status-msg').style.color = '#f44336';
                console.error('Export error:', err);
            });
        }

        // Event listeners
        canvas.addEventListener('click', (e) => {
            if (isDragging) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Check if clicked on existing marker
            let clickedGcp = null;
            let clickedIdx = null;

            for (let i = 0; i < gcps.length; i++) {
                const gcp = gcps[i];
                const dist = Math.sqrt(Math.pow(x - gcp.pixel_x, 2) + Math.pow(y - gcp.pixel_y, 2));
                if (dist < 15) {
                    clickedGcp = gcp;
                    clickedIdx = i;
                    break;
                }
            }

            if (clickedGcp) {
                selectedGcpIndex = clickedIdx;
                updateGcpList();
                drawCanvas();
                showGpsModal(clickedGcp.pixel_x, clickedGcp.pixel_y, clickedGcp);
            } else {
                showGpsModal(x, y);
            }
        });

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Check if mouse down on marker
            for (let i = 0; i < gcps.length; i++) {
                const gcp = gcps[i];
                const dist = Math.sqrt(Math.pow(x - gcp.pixel_x, 2) + Math.pow(y - gcp.pixel_y, 2));
                if (dist < 15) {
                    isDragging = true;
                    dragGcpIndex = i;
                    canvas.style.cursor = 'move';
                    break;
                }
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            gcps[dragGcpIndex].pixel_x = x;
            gcps[dragGcpIndex].pixel_y = y;

            updateGcpList();
            drawCanvas();
        });

        canvas.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                dragGcpIndex = null;
                canvas.style.cursor = 'crosshair';
            }
        });

        document.getElementById('modal-cancel').addEventListener('click', hideGpsModal);

        document.getElementById('export-btn').addEventListener('click', exportJson);

        document.getElementById('clear-btn').addEventListener('click', () => {
            if (confirm('Clear all GCPs?')) {
                gcps = [];
                selectedGcpIndex = null;
                updateGcpList();
                drawCanvas();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && selectedGcpIndex !== null) {
                deleteSelectedGcp();
            }
            if (e.key === 'Escape') {
                hideGpsModal();
            }
        });
    </script>
</body>
</html>"""


# Global state for server
class ServerState:
    """Server state container."""

    frame_path: str | None = None
    camera_info: dict[str, float] = {}
    camera_name: str | None = None
    output_path: str | None = None
    map_points: list[dict[str, Any]] = []


SERVER_STATE = ServerState()


class RequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for test data generator web interface."""

    def log_message(self, _format: str, *_args: Any) -> None:
        """Suppress default logging."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/":
            # Serve main HTML interface
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(create_html_interface().encode())

        elif parsed_url.path == "/api/init":
            # Serve initial camera info and map points
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            data = {
                "camera_info": SERVER_STATE.camera_info,
                "camera_name": SERVER_STATE.camera_name,
                "map_points": SERVER_STATE.map_points,
            }
            self.wfile.write(json.dumps(data).encode())

        elif parsed_url.path == "/api/image":
            # Serve captured frame image
            if SERVER_STATE.frame_path:
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                with open(SERVER_STATE.frame_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/api/export":
            # Handle JSON export
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())

            # Generate JSON output and copy image
            result = generate_json_output(
                camera_info=data["camera_info"],
                gcps=data["gcps"],
                camera_name=SERVER_STATE.camera_name or "unknown",
                output_path=SERVER_STATE.output_path,
                frame_path=SERVER_STATE.frame_path,
            )

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_data_generator(
    camera_name: str,
    output_path: str | None = None,
    map_points_path: Path | None = None,
) -> None:
    """
    Run the test data generator for a camera.

    Args:
        camera_name: Name of the camera
        output_path: Optional custom output path for JSON
        map_points_path: Path to map points JSON file

    Raises:
        RuntimeError: If camera operations fail
        ValueError: If camera not found
    """
    # Get camera config from repository
    ctx = ApplicationContext.default()

    all_configs = ctx.repo_camera_config.get_all()
    camera_config = next((c for c in all_configs if c.name == camera_name), None)

    if not camera_config:
        available = [c.name for c in all_configs]
        raise ValueError(
            f"Camera '{camera_name}' not found in configuration. Available: {', '.join(available)}"
        )

    if not camera_config.ip_address:
        raise ValueError(f"Camera '{camera_name}' has no IP address configured")

    # Get calibration data
    calibration = ctx.repo_camera_calibration.get(camera_config.id)

    print(f"=== Test Data Generator for {camera_name} ===\n")

    # Step 1: Extract camera parameters
    print("1. Extracting camera parameters...")
    height_m = float(calibration.height) if calibration else 5.0
    position_x = float(calibration.position.x) if calibration else None
    position_y = float(calibration.position.y) if calibration else None

    camera_params: dict[str, float | None] = {
        "height_meters": height_m,
        "position_x": position_x,
        "position_y": position_y,
        "latitude": None,
        "longitude": None,
    }
    print(f"   Height: {height_m} m")
    if position_x is not None and position_y is not None:
        print(f"   Position: ({position_x:.1f}, {position_y:.1f})")

    # Get credentials from camera config
    username = camera_config.credential.username
    password = camera_config.credential.password

    # Step 2: Fetch PTZ status
    print("2. Fetching PTZ status...")
    try:
        ptz_status = fetch_ptz_status(camera_config.ip_address, username, password)
        print(f"   Pan: {ptz_status['pan_deg']:.1f}deg")
        print(f"   Tilt: {ptz_status['tilt_deg']:.1f}deg")
        print(f"   Zoom: {ptz_status['zoom_level']:.1f}x")
    except RuntimeError as e:
        print(f"   Warning: {e}")
        print("   Using default values (manual entry required)")
        ptz_status = {"pan_deg": 0.0, "tilt_deg": 0.0, "zoom_level": 1.0}

    # Combine camera info
    camera_info = {**camera_params, **ptz_status}

    # Step 3: Capture frame
    print("3. Capturing frame from camera...")
    try:
        rtsp_url = camera_config.rtsp_url(stream_type="main")
        frame_path = capture_frame_from_rtsp(rtsp_url, timeout_sec=10.0)
    except RuntimeError as e:
        print(f"   Error: {e}")
        raise

    # Step 4: Load map points
    map_points: list[dict[str, Any]] = []
    if map_points_path:
        print(f"4. Loading map points from {map_points_path}...")
        try:
            registry = load_map_points(map_points_path)
            map_points = convert_map_points_to_list(registry)
            map_id = next(iter(registry.values())).map_id if registry else "unknown"
            print(f"   Loaded {len(map_points)} points from map '{map_id}'")
        except FileNotFoundError:
            print(f"   Warning: Map points file not found: {map_points_path}")
            print("   Continuing without map points (manual coordinate entry required)")
        except Exception as e:
            print(f"   Warning: Failed to load map points: {e}")
            print("   Continuing without map points")
    else:
        print("4. No map points file specified - skipping")

    # Step 5: Start web server
    print("\n5. Starting web server...")

    # Store state for server
    SERVER_STATE.frame_path = frame_path
    SERVER_STATE.camera_info = camera_info
    SERVER_STATE.camera_name = camera_name
    SERVER_STATE.output_path = output_path
    SERVER_STATE.map_points = map_points

    # Find available port
    port = find_available_port(start_port=8080, max_attempts=10)

    server = http.server.HTTPServer(("localhost", port), RequestHandler)

    print(f"   Server running at http://localhost:{port}")
    print("\n=== Opening browser... ===")
    print("Mark GCP points by clicking on the image.")
    print("Press Ctrl+C to stop the server.\n")

    # Open browser
    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()

        # Clean up temp file
        if frame_path and os.path.exists(frame_path):
            os.unlink(frame_path)

        print("Done!")
