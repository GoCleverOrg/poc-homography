"""
Standalone test data generator tool for camera calibration GCPs.

This tool captures full-resolution camera frames from PTZ cameras and enables
interactive marking of Ground Control Points (GCPs) with map point references.
Automatically fetches camera parameters (pan/tilt/zoom, GPS, height) and exports
test data in JSON format.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.get_camera_intrinsics import get_ptz_status

from poc_homography.camera_config import (
    CAMERAS,
    PASSWORD,
    USERNAME,
    get_camera_by_name,
    get_rtsp_url,
)
from poc_homography.gps_distance_calculator import dms_to_dd
from poc_homography.map_points import MapPointRegistry
from poc_homography.server_utils import find_available_port


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Command-line arguments (typically sys.argv[1:])

    Returns:
        Parsed arguments namespace with fields:
        - camera_name: str or None
        - output: str or None
        - list_cameras: bool
        - map_points: str (path to map points JSON file)

    Raises:
        SystemExit: If arguments are invalid (argparse behavior)
    """
    parser = argparse.ArgumentParser(
        description="Test data generator for camera calibration GCPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "camera_name", type=str, nargs="?", help="Camera name (e.g., Valte, Setram)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: test_data_{camera}_{timestamp}.json)",
    )

    parser.add_argument(
        "--list-cameras", action="store_true", help="List available cameras and exit"
    )

    parser.add_argument(
        "--map-points",
        type=str,
        default="map_points.json",
        help="Path to map points JSON file (default: map_points.json)",
    )

    args = parser.parse_args(argv)

    # Validate that camera_name is provided if not listing cameras
    if not args.list_cameras and not args.camera_name:
        parser.error("camera_name is required unless --list-cameras is specified")

    return args


def validate_camera_name(camera_name: str, cameras: list[dict]) -> None:
    """
    Validate that camera name exists in camera list.

    Args:
        camera_name: Name of camera to validate
        cameras: List of camera configuration dicts with 'name' field

    Raises:
        ValueError: If camera name not found in cameras list
    """
    available_names = [cam["name"] for cam in cameras]

    if camera_name not in available_names:
        raise ValueError(
            f"Camera '{camera_name}' not found. Available: {', '.join(available_names)}"
        )


def validate_gps_ranges(latitude: float, longitude: float) -> None:
    """
    Validate that GPS coordinates are within valid ranges.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees

    Raises:
        ValueError: If coordinates are outside valid ranges
    """
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"Latitude must be between -90 and 90 degrees, got {latitude}")

    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"Longitude must be between -180 and 180 degrees, got {longitude}")


def load_map_points(map_points_path: str | Path) -> MapPointRegistry:
    """
    Load map points from JSON file using MapPointRegistry.

    Args:
        map_points_path: Path to map points JSON file

    Returns:
        MapPointRegistry containing loaded map points

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        KeyError: If required keys are missing
        ValueError: If data format is invalid
    """
    return MapPointRegistry.load(map_points_path)


def convert_map_points_to_list(registry: MapPointRegistry) -> list[dict]:
    """
    Convert MapPointRegistry to list format for web interface.

    Args:
        registry: MapPointRegistry to convert

    Returns:
        List of dictionaries with id, pixel_x, pixel_y, map_id keys
    """
    return [point.to_dict() for point in registry.points.values()]


def convert_gps_coordinates(lat_dms: str, lon_dms: str) -> tuple[float, float]:
    """
    Convert GPS coordinates from DMS format to decimal degrees with validation.

    Args:
        lat_dms: Latitude in DMS format (e.g., "39°38'25.72\"N")
        lon_dms: Longitude in DMS format (e.g., "0°13'48.63\"W")

    Returns:
        Tuple of (latitude, longitude) in decimal degrees

    Raises:
        ValueError: If converted coordinates are outside valid ranges
    """
    # Convert using existing dms_to_dd function
    latitude = dms_to_dd(lat_dms)
    longitude = dms_to_dd(lon_dms)

    # Validate ranges
    validate_gps_ranges(latitude, longitude)

    return (latitude, longitude)


def extract_camera_parameters(camera_config: dict) -> dict:
    """
    Extract camera GPS position and height from camera config.

    Args:
        camera_config: Camera configuration dict with lat, lon, height_m fields

    Returns:
        Dictionary with latitude, longitude, height_meters (all in decimal degrees/meters)

    Raises:
        ValueError: If GPS conversion fails
    """
    lat_dms = camera_config["lat"]
    lon_dms = camera_config["lon"]
    height_m = camera_config["height_m"]

    # Convert DMS to decimal degrees
    latitude, longitude = convert_gps_coordinates(lat_dms, lon_dms)

    return {"latitude": latitude, "longitude": longitude, "height_meters": height_m}


def fetch_ptz_status(camera_config: dict) -> dict:
    """
    Fetch current PTZ status from camera.

    Args:
        camera_config: Camera configuration with IP address

    Returns:
        Dictionary with pan_deg, tilt_deg, zoom_level

    Raises:
        RuntimeError: If PTZ status cannot be fetched
    """
    ip = camera_config["ip"]

    try:
        ptz_data = get_ptz_status(ip, USERNAME, PASSWORD, timeout=10.0)

        return {
            "pan_deg": ptz_data["pan"],
            "tilt_deg": ptz_data["tilt"],
            "zoom_level": ptz_data["zoom"],
        }
    except RuntimeError as e:
        raise RuntimeError(f"Failed to fetch PTZ status: {e}")


def capture_frame_from_rtsp(camera_name: str, timeout_sec: float = 10.0) -> str:
    """
    Capture a single frame from RTSP camera stream.

    Args:
        camera_name: Name of the camera
        timeout_sec: Timeout for capture operation in seconds

    Returns:
        Path to saved frame image file

    Raises:
        RuntimeError: If frame capture fails
    """
    # Get RTSP URL
    rtsp_url = get_rtsp_url(camera_name, stream_type="main")

    if not rtsp_url:
        raise RuntimeError(f"Could not get RTSP URL for camera {camera_name}")

    print(f"Connecting to RTSP stream: {rtsp_url}")

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
            f"Failed to capture frame from camera {camera_name} within {timeout_sec}s timeout. "
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
    camera_info: dict,
    gcps: list[dict],
    camera_name: str,
    output_path: str | None = None,
    frame_path: str | None = None,
) -> dict[str, str]:
    """
    Generate JSON output file with camera info and GCPs, and copy the frame image.

    Args:
        camera_info: Dictionary with camera parameters (latitude, longitude, height_meters, pan_deg, tilt_deg, zoom_level)
        gcps: List of GCP dictionaries with pixel_x, pixel_y, and either map_point_id or latitude/longitude
        camera_name: Name of the camera
        output_path: Optional custom output path for JSON
        frame_path: Optional path to the captured frame image

    Returns:
        Dictionary with 'json_path' and 'image_path' keys
    """
    import shutil

    # Generate default filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"test_data_{camera_name}_{timestamp}.json"

    # Construct output data
    data = {"camera_info": camera_info, "gcps": gcps}

    # Write JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    result = {"json_path": output_path, "image_path": None}

    # Copy frame image with matching filename
    if frame_path and os.path.exists(frame_path):
        # Replace .json with .jpg for the image filename
        image_output_path = output_path.rsplit(".json", 1)[0] + ".jpg"
        shutil.copy2(frame_path, image_output_path)
        result["image_path"] = image_output_path

    return result


# Global variables for server state
SERVER_STATE = {
    "frame_path": None,
    "camera_info": {},
    "camera_name": None,
    "output_path": None,
    "map_points": [],  # List of {id, pixel_x, pixel_y, map_id} from map points file
}


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


class RequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for test data generator web interface."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
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
                "camera_info": SERVER_STATE["camera_info"],
                "camera_name": SERVER_STATE["camera_name"],
                "map_points": SERVER_STATE["map_points"],
            }
            self.wfile.write(json.dumps(data).encode())

        elif parsed_url.path == "/api/image":
            # Serve captured frame image
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            with open(SERVER_STATE["frame_path"], "rb") as f:
                self.wfile.write(f.read())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
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
                camera_name=SERVER_STATE["camera_name"],
                output_path=SERVER_STATE["output_path"],
                frame_path=SERVER_STATE["frame_path"],
            )

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    """Main entry point."""
    args = parse_arguments(sys.argv[1:])

    # Handle --list-cameras
    if args.list_cameras:
        print("Available cameras:")
        for cam in CAMERAS:
            print(f"  - {cam['name']} ({cam['ip']})")
        sys.exit(0)

    # Validate camera name
    try:
        validate_camera_name(args.camera_name, CAMERAS)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get camera config
    camera_config = get_camera_by_name(args.camera_name)

    print(f"=== Test Data Generator for {args.camera_name} ===\n")

    # Step 1: Extract camera parameters
    print("1. Extracting camera parameters...")
    try:
        camera_params = extract_camera_parameters(camera_config)
        print(f"   GPS: {camera_params['latitude']:.6f}, {camera_params['longitude']:.6f}")
        print(f"   Height: {camera_params['height_meters']} m")
    except Exception as e:
        print(f"   Error: {e}")
        sys.exit(1)

    # Step 2: Fetch PTZ status
    print("2. Fetching PTZ status...")
    try:
        ptz_status = fetch_ptz_status(camera_config)
        print(f"   Pan: {ptz_status['pan_deg']:.1f}°")
        print(f"   Tilt: {ptz_status['tilt_deg']:.1f}°")
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
        frame_path = capture_frame_from_rtsp(args.camera_name, timeout_sec=10.0)
    except RuntimeError as e:
        print(f"   Error: {e}")
        sys.exit(1)

    # Step 4: Load map points
    map_points = []
    print(f"4. Loading map points from {args.map_points}...")
    try:
        registry = load_map_points(args.map_points)
        map_points = convert_map_points_to_list(registry)
        print(f"   Loaded {len(map_points)} points from map '{registry.map_id}'")
    except FileNotFoundError:
        print(f"   Warning: Map points file not found: {args.map_points}")
        print("   Continuing without map points (manual coordinate entry required)")
    except Exception as e:
        print(f"   Warning: Failed to load map points: {e}")
        print("   Continuing without map points")

    # Step 5: Start web server
    print("\n5. Starting web server...")

    # Store state for server
    SERVER_STATE["frame_path"] = frame_path
    SERVER_STATE["camera_info"] = camera_info
    SERVER_STATE["camera_name"] = args.camera_name
    SERVER_STATE["output_path"] = args.output
    SERVER_STATE["map_points"] = map_points

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
        if os.path.exists(frame_path):
            os.unlink(frame_path)

        print("Done!")


if __name__ == "__main__":
    main()
