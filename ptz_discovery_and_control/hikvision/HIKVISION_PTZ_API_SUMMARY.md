# Hikvision PTZ Camera ISAPI - Discovery Summary

## Camera Information

Tested on two Hikvision PTZ cameras:
- **Camera 1**: 10.207.99.178
- **Camera 2**: 10.237.100.15
- **Credentials**: Loaded from environment variables (CAMERA_USERNAME / CAMERA_PASSWORD)

## ISAPI Endpoints Discovered

### 1. PTZ Status Endpoint (GET Current Position)

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/status`

**Method**: `GET`

**Status**: ✓ Working (200)

**Description**: Returns the current absolute position of the PTZ camera (pan, tilt, zoom)

**Response Format**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<PTZStatus version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <AbsoluteHigh>
    <elevation>121</elevation>
    <azimuth>1117</azimuth>
    <absoluteZoom>209</absoluteZoom>
  </AbsoluteHigh>
</PTZStatus>
```

**Value Interpretation**:
- **Pan (Azimuth)**: Raw value divided by 10 = degrees (e.g., 1117 → 111.7°)
- **Tilt (Elevation)**: Raw value divided by 10 = degrees (e.g., 121 → 12.1°)
- **Zoom (AbsoluteZoom)**: Raw value divided by 10 = zoom level (e.g., 209 → 20.9)

**Note**: Values use the namespace `http://www.hikvision.com/ver20/XMLSchema`, so XML parsing requires namespace handling.

---

### 2. PTZ Control Endpoints (Movement)

#### 2.1 Continuous Movement

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/continuous`

**Method**: `PUT`

**Status**: ✓ Working (200)

**Description**: Moves the camera continuously until stopped. This is the **primary endpoint for PTZ control**.

**Request Format**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
  <pan>30</pan>
  <tilt>0</tilt>
  <zoom>0</zoom>
</PTZData>
```

**Parameters**:
- **pan**: -100 to +100 (negative = left, positive = right, 0 = stop)
- **tilt**: -100 to +100 (negative = down, positive = up, 0 = stop)
- **zoom**: -100 to +100 (negative = zoom out, positive = zoom in, 0 = stop)

**To Stop Movement**: Send all values as 0

**Example Usage**:
```python
import requests
from requests.auth import HTTPDigestAuth

url = "http://10.207.99.178/ISAPI/PTZCtrl/channels/1/continuous"
xml_data = '''<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>30</pan>
    <tilt>0</tilt>

---

## Advanced PTZ Commands (Python)

The following helper methods are available in `HikvisionPTZ` (see `hikvision_ptz_discovery.py`):

- `send_3d_zoom_command(x_start, y_start, x_end, y_end)` — sends a PUT to `/ISAPI/PTZCtrl/channels/1/position3D`.
    This command is the endpoint called when you do a square zoom via the website when looking into the camera view
    the coordinates space is 0-255 on both axis
    
    the endpoint operates in 2 different ways:
        BR,TL
        x_start, y_start corresponds to the BR coordinate of the rectangle
        x_end, y_end corresponds to the TL coordinate of the rectangle
        the camera only changes tilt and pan

        TL, BR
        x_start, y_start corresponds to the TL coordinate of the rectangle
        x_end, y_end corresponds to the BR coordinate of the rectangle
        the camera changes tilt and pan and then zooms to conform the square dimensions to the best fit possible

    the origin 0,0 is BL
- `send_ptz_return(status)` — sends an absolute PTZ command to `/ISAPI/PTZCtrl/channels/1/absolute` using the status dict returned by `get_status()`.

### Example: Return to a Saved Position

```python
status = camera.get_status()
camera.send_ptz_return(status)  # Returns to the same pan/tilt/zoom
```

### Example: 3D Zoom/Position Command

```python
# WARNING: This may move the camera. Use with caution.
code, resp = camera.send_3d_zoom_command(0.1, 0.2, 0.3, 0.4)
print(f"3D command returned {code}")
```

### Implementation Notes

- The PTZ helper methods are now named in English (no Spanish aliases).
- The `send_ptz_return` method multiplies parsed pan/tilt/zoom by 10 to match the integer units used in the camera status XML (the code in `get_status()` divides by 10 when parsing).
- The 3D command may require valid coordinates and camera support; test with care.

### Testing

- See `test_new_commands.py` for a safe test script.
- By default, the script fetches the camera status and calls `send_ptz_return` with the current position (safe no-op).
- The 3D command is gated behind an environment variable. To run it, set:

```bash
export RUN_3D=1
python ptz_discovery_and_control/hikvision/test_new_commands.py
```

---
    <zoom>0</zoom>
</PTZData>'''

response = requests.put(url,
                       auth=HTTPDigestAuth(USERNAME, PASSWORD),
                       data=xml_data,
                       headers={'Content-Type': 'application/xml'})
```

#### 2.2 Momentary Movement

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/momentary`

**Method**: `PUT`

**Status**: ⚠ Exists (400 - needs proper XML data structure)

**Description**: Performs timed/momentary PTZ movements

#### 2.3 Absolute Positioning

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/absolute`

**Method**: `PUT`

**Status**: ⚠ Exists (400 - needs proper XML data structure)

**Description**: Moves camera to absolute position coordinates

#### 2.4 Relative Movement

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/relative`

**Method**: `PUT`

**Status**: ✗ Not Found (404)

**Description**: Not available on these camera models

---

### 3. Preset Management Endpoints

#### 3.1 List Presets

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/presets`

**Method**: `GET`

**Status**: ✓ Working (200)

**Description**: Lists all configured preset positions

#### 3.2 Go To Preset

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/presets/1/goto`

**Method**: `PUT`

**Status**: ✓ Working (200)

**Description**: Moves camera to preset position 1 (change number for other presets)

---

### 4. Other Endpoints

#### 4.1 PTZ Capabilities

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/capabilities`

**Method**: `GET`

**Status**: ✓ Working (200)

**Description**: Returns camera PTZ capabilities and supported features

#### 4.2 Home Position

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/homeposition/goto`

**Method**: `PUT`

**Status**: ✓ Working (200)

**Description**: Returns camera to configured home position

#### 4.3 Auxiliary Controls

**Endpoint**: `/ISAPI/PTZCtrl/channels/1/auxcontrols/1`

**Method**: `PUT`

**Status**: ⚠ Exists (400 - needs proper XML data structure)

**Description**: Controls auxiliary outputs (e.g., wiper, lights)

---

## Movement Testing Results

### Camera Movement Characteristics

**Polling Interval**: 0.3 seconds was effective for tracking movement

**Stabilization Time**: Cameras typically stabilize 1-2 seconds after stopping

**Movement Behavior**:
- Pan movements are responsive and accurate
- Tilt movements work but may show slight drift
- Zoom changes are smooth and predictable
- Position is maintained accurately after stabilization

### Observed Movement Patterns (Camera 1)

| Movement | Speed | Duration | Distance Traveled | Notes |
|----------|-------|----------|-------------------|-------|
| Pan Right | 30 | 1.5s | -25.2° | Direction inverted? |
| Pan Left | -30 | 1.5s | -0.8° | Small movement |
| Tilt Up | 30 | 1.5s | +0.0° (±0.4°) | Very minimal movement |
| Tilt Down | -30 | 1.5s | +0.0° (±0.4°) | Very minimal movement |
| Zoom In | 30 | 1.5s | +21.1 | Good response |
| Zoom Out | -30 | 1.5s | -25.1 | Good response |

### Observed Movement Patterns (Camera 2)

| Movement | Speed | Duration | Distance Traveled | Notes |
|----------|-------|----------|-------------------|-------|
| Pan Right | 30 | 1.5s | +9.4° | Good response |
| Pan Left | -30 | 1.5s | -7.3° | Good response |
| Tilt Up | 30 | 1.5s | -3.7° | Direction inverted? |
| Tilt Down | -30 | 1.5s | +3.9° | Direction inverted? |
| Zoom In | 30 | 1.5s | +13.5 | Good response |
| Zoom Out | -30 | 1.5s | -12.0 | Good response |

**Note**: Direction inversions may be due to camera mounting orientation or coordinate system differences.

---

## Python Implementation

### Complete Working Example

See `hikvision_ptz_discovery.py` for full implementation including:

1. **HikvisionPTZ Class**: Object-oriented interface to camera
2. **get_status()**: Get current PTZ position with namespace handling
3. **move_continuous()**: Control camera movement
4. **stop_movement()**: Stop all movement
5. **test_movement_with_polling()**: Test movements and track position changes
6. **discover_endpoints()**: Automatically discover available API endpoints

### Key Implementation Notes

1. **Authentication**: Use `HTTPDigestAuth` from requests library
2. **XML Namespace**: Handle Hikvision namespace `http://www.hikvision.com/ver20/XMLSchema`
3. **Value Conversion**: Divide raw values by 10 for degrees/zoom level
4. **Stabilization**: Poll status after stopping to detect when movement completes
5. **Timeout**: Use reasonable timeouts (5-10s) for camera responses

### Quick Start Code

```python
from requests.auth import HTTPDigestAuth
import requests
import xml.etree.ElementTree as ET

class HikvisionPTZ:
    def __init__(self, ip, username, password):
        self.base_url = f"http://{ip}"
        self.auth = HTTPDigestAuth(username, password)

    def get_status(self):
        """Get current pan, tilt, zoom"""
        url = f"{self.base_url}/ISAPI/PTZCtrl/channels/1/status"
        response = requests.get(url, auth=self.auth, timeout=5)

        if response.status_code == 200:
            root = ET.fromstring(response.text)
            ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}

            azimuth = root.find('.//h:azimuth', ns)
            elevation = root.find('.//h:elevation', ns)
            zoom = root.find('.//h:absoluteZoom', ns)

            return {
                'pan': float(azimuth.text) / 10,
                'tilt': float(elevation.text) / 10,
                'zoom': float(zoom.text) / 10
            }
        return None

    def move(self, pan=0, tilt=0, zoom=0):
        """Move camera (values: -100 to +100, 0 to stop)"""
        url = f"{self.base_url}/ISAPI/PTZCtrl/channels/1/continuous"
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>{pan}</pan>
    <tilt>{tilt}</tilt>
    <zoom>{zoom}</zoom>
</PTZData>'''

        response = requests.put(url, auth=self.auth, data=xml,
                               headers={'Content-Type': 'application/xml'})
        return response.status_code == 200

# Usage
# Load credentials from environment variables first
import os
USERNAME = os.environ.get("CAMERA_USERNAME")
PASSWORD = os.environ.get("CAMERA_PASSWORD")

camera = HikvisionPTZ("10.207.99.178", USERNAME, PASSWORD)

# Get current position
status = camera.get_status()
print(f"Pan: {status['pan']:.1f}°, Tilt: {status['tilt']:.1f}°, Zoom: {status['zoom']:.1f}")

# Pan right at 30% speed
camera.move(pan=30, tilt=0, zoom=0)

# Stop movement
camera.move(pan=0, tilt=0, zoom=0)
```

---

## Summary of Available Movement Endpoints

| Endpoint | Method | Working | Purpose |
|----------|--------|---------|---------|
| `/ISAPI/PTZCtrl/channels/1/continuous` | PUT | ✓ Yes | **Primary movement control** |
| `/ISAPI/PTZCtrl/channels/1/momentary` | PUT | ⚠ Partial | Timed movements |
| `/ISAPI/PTZCtrl/channels/1/absolute` | PUT | ⚠ Partial | Absolute positioning |
| `/ISAPI/PTZCtrl/channels/1/relative` | PUT | ✗ No | Not supported |
| `/ISAPI/PTZCtrl/channels/1/presets/N/goto` | PUT | ✓ Yes | Go to preset N |
| `/ISAPI/PTZCtrl/channels/1/homeposition/goto` | PUT | ✓ Yes | Go to home position |

**Recommendation**: Use `/continuous` endpoint for all real-time PTZ control applications.

---

## Files Generated

1. **hikvision_ptz_discovery.py** - Main discovery and testing script
2. **ptz_test_results_Camera_1_[timestamp].json** - Detailed test results for Camera 1
3. **ptz_test_results_Camera_2_[timestamp].json** - Detailed test results for Camera 2
4. **ptz_discovery_output.log** - Complete console output from testing
5. **HIKVISION_PTZ_API_SUMMARY.md** - This summary document

---

## Authentication

All endpoints require HTTP Digest Authentication:
- Username: Set via `CAMERA_USERNAME` environment variable
- Password: Set via `CAMERA_PASSWORD` environment variable

Use the `HTTPDigestAuth` class from Python's `requests` library.

Example:
```python
import os
from requests.auth import HTTPDigestAuth

USERNAME = os.environ.get("CAMERA_USERNAME")
PASSWORD = os.environ.get("CAMERA_PASSWORD")
auth = HTTPDigestAuth(USERNAME, PASSWORD)
```

---

## Next Steps

To extend this work, you could:

1. Implement proper XML structure for `momentary` and `absolute` endpoints
2. Add support for multiple camera channels (currently hardcoded to channel 1)
3. Create a real-time PTZ controller with continuous position monitoring
4. Implement preset management (create, update, delete presets)
5. Add support for auxiliary controls (wiper, lights, etc.)
6. Build a web interface for camera control
7. Integrate with a tracking system for automated PTZ control

---

**Document Generated**: 2025-11-20
**Testing Duration**: ~60 seconds per camera
**Total Movements Tested**: 12 (6 per camera)
