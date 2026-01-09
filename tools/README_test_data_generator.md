# Test Data Generator Tool

## Overview

The Test Data Generator is a standalone web-based tool for capturing full-resolution camera frames from PTZ cameras and interactively generating test data for camera calibration. It enables users to click points on captured images to create Ground Control Points (GCPs) with GPS coordinates.

## Features

- **Automatic Frame Capture**: Captures full-resolution frames from RTSP camera streams
- **Auto-Fetch Camera Parameters**: Automatically retrieves:
  - GPS position (latitude, longitude) from camera config
  - Camera height from config
  - Current PTZ status (pan, tilt, zoom) from camera API
- **Interactive Web Interface**:
  - Click on image to mark GCP points
  - Modal dialog for entering GPS coordinates
  - Drag markers to adjust pixel positions
  - Edit GPS coordinates by clicking existing markers
  - Delete markers with Delete key or button
- **Manual Parameter Override**: All camera parameters editable in UI if auto-fetch fails
- **JSON Export**: Exports data in simple JSON format with camera info and GCPs

## Installation

No additional installation required beyond project dependencies:
- opencv-python (cv2)
- Standard Python 3.9+ libraries

## Usage

### List Available Cameras

```bash
python tools/cli/test_data_generator.py --list-cameras
```

Output:
```
Available cameras:
  - Valte (10.207.99.178)
  - Setram (10.237.100.15)
```

### Generate Test Data for Camera

```bash
# Basic usage - auto-generate output filename
python tools/cli/test_data_generator.py Valte

# Specify custom output path
python tools/cli/test_data_generator.py Valte --output /path/to/output.json
```

### Workflow

1. **Tool starts and auto-fetches**:
   - Camera GPS position and height from `camera_config.py`
   - Current PTZ status (pan/tilt/zoom) from camera API
   - Captures full-resolution frame from RTSP stream

2. **Browser opens automatically** with interactive interface showing:
   - Captured camera frame
   - Camera parameters (all editable)
   - Empty GCP list

3. **Mark GCP points**:
   - Click on image → modal appears
   - Enter latitude and longitude in decimal degrees
   - Marker appears with crosshair and label
   - Repeat for each GCP point

4. **Edit GCP points**:
   - **Drag marker**: Mouse down on marker → drag to new position
   - **Edit GPS**: Click on existing marker → modal opens with current coordinates
   - **Delete**: Select marker → press Delete key

5. **Export JSON**:
   - Click "Export JSON" button
   - File saved with format: `test_data_{camera_name}_{timestamp}.json`
   - Status message shows export path

## Output Format

```json
{
  "camera_info": {
    "latitude": 39.640477,
    "longitude": -0.230175,
    "height_meters": 11.3,
    "pan_deg": 93.2,
    "tilt_deg": 6.7,
    "zoom_level": 1.0
  },
  "gcps": [
    {
      "pixel_x": 904.2,
      "pixel_y": 612.6,
      "latitude": 39.640264,
      "longitude": -0.229967
    },
    {
      "pixel_x": 1205.8,
      "pixel_y": 823.4,
      "latitude": 39.640156,
      "longitude": -0.229845
    }
  ]
}
```

## Coordinate Systems

- **Pixel Coordinates**: Origin at top-left (standard image coordinates)
  - `pixel_x`: Horizontal position (0 = left edge)
  - `pixel_y`: Vertical position (0 = top edge)

- **GPS Coordinates**: Decimal degrees
  - `latitude`: -90 to +90 (negative = South, positive = North)
  - `longitude`: -180 to +180 (negative = West, positive = East)

## Error Handling

### Frame Capture Timeout

If RTSP stream is unreachable or times out (10 seconds):

```
Error: Failed to capture frame from camera Valte within 10s timeout.
Please check camera connectivity and RTSP stream availability.
```

**Solution**: Verify camera is online and RTSP stream is accessible.

### PTZ Status Fetch Failure

If PTZ API is unreachable:

```
Warning: Failed to fetch PTZ status: Failed to connect to camera: ...
Using default values (manual entry required)
```

**Solution**: PTZ parameters default to (0.0, 0.0, 1.0). Edit manually in web interface.

### Invalid Camera Name

```
Error: Camera 'InvalidCam' not found. Available: Valte, Setram
```

**Solution**: Use `--list-cameras` to see available cameras.

## Architecture

### Backend (Python)

- **CLI Argument Parsing**: `parse_arguments()`, `validate_camera_name()`
- **GPS Conversion**: `convert_gps_coordinates()` wraps `dms_to_dd()` with validation
- **Camera Parameters**: `extract_camera_parameters()` extracts GPS/height from config
- **PTZ Status**: `fetch_ptz_status()` calls camera API via `get_ptz_status()`
- **Frame Capture**: `capture_frame_from_rtsp()` uses OpenCV to capture single frame
- **JSON Export**: `generate_json_output()` creates timestamped JSON file
- **Web Server**: `RequestHandler` serves HTML interface and handles API endpoints

### Frontend (JavaScript)

- **Canvas Drawing**: Displays image with interactive GCP markers
- **Modal Dialog**: GPS coordinate entry with validation
- **Drag-and-Drop**: Mouse event handlers for marker repositioning
- **API Calls**:
  - `GET /api/init`: Load camera info
  - `GET /api/image`: Load captured frame
  - `POST /api/export`: Export JSON with GCPs

### Web Server Pattern

Follows existing pattern from `tools/capture_gcps_web.py`:
- Single-file distribution (HTML/CSS/JS embedded in Python)
- `http.server.HTTPServer` with custom request handler
- Auto-finds available port using `find_available_port()`
- Auto-opens browser to `http://localhost:{port}`

## Testing

Unit tests cover backend logic:

```bash
# Run all tests
python3 -m pytest tests/test_test_data_generator.py -v

# Run specific test class
python3 -m pytest tests/test_test_data_generator.py::TestCLIArgumentParsing -v
python3 -m pytest tests/test_test_data_generator.py::TestGPSCoordinateConversion -v
```

Test coverage:
- CLI argument parsing and validation (7 tests)
- GPS coordinate conversion and range validation (6 tests)

UI interactions tested manually.

## Dependencies

From existing codebase:
- `poc_homography.gps_distance_calculator.dms_to_dd`: DMS to decimal conversion
- `poc_homography.camera_config`: Camera configurations and RTSP URLs
- `poc_homography.server_utils.find_available_port`: Port finding utility
- `tools.get_camera_intrinsics.get_ptz_status`: PTZ API interaction

## Comparison with capture_gcps_web.py

| Feature | test_data_generator.py | capture_gcps_web.py |
|---------|------------------------|---------------------|
| Purpose | Generate test data for calibration | Collect GCPs for production calibration |
| Output Format | Simple JSON | Complex YAML with intrinsics/homography |
| Frame Source | RTSP capture only | RTSP or existing image file |
| GCP Source | Manual GPS entry | Manual or KML import |
| Camera Parameters | Auto-fetch with manual override | Auto-fetch with complex calibration |
| Complexity | Minimal, focused on data collection | Full-featured with validation/optimization |
| Use Case | Quick test data generation | Production calibration workflow |

## Example Session

```bash
$ python tools/cli/test_data_generator.py Valte

=== Test Data Generator for Valte ===

1. Extracting camera parameters...
   GPS: 39.640477, -0.230175
   Height: 4.71 m
2. Fetching PTZ status...
   Pan: 93.2°
   Tilt: 6.7°
   Zoom: 1.0x
3. Capturing frame from camera...
Connecting to RTSP stream: rtsp://...
Frame captured: 1920x1080 pixels
Saved to: /tmp/tmp8x3fy2jk.jpg

4. Starting web server...
   Server running at http://localhost:8080

=== Opening browser... ===
Mark GCP points by clicking on the image.
Press Ctrl+C to stop the server.

^C
Shutting down server...
Done!
```

## Future Enhancements

Potential improvements:
- Load existing JSON to continue editing
- Import GCPs from KML files
- Visual validation (show GCP distribution on map)
- Batch mode (capture multiple cameras sequentially)
- Integration with existing calibration workflows

## License

Part of the POC Homography project. See project LICENSE for details.
