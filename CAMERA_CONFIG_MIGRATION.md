# Camera Configuration Migration

## Summary

Centralized all camera configurations into `camera_config.py` for easier management and consistency across all verification tools.

## Changes Made

### New File: `camera_config.py`

Central configuration file containing:
- **Camera definitions** with GPS coordinates and default heights
- **Credentials** (USERNAME, PASSWORD)
- **Helper functions**:
  - `get_camera_by_name(name)` - Get camera config
  - `get_camera_gps(name)` - Get camera GPS coords
  - `get_rtsp_url(name)` - Generate RTSP URL

### Updated Files:

1. **main.py**
   - Imports from `camera_config`
   - Removed duplicate camera definitions
   - Uses `get_rtsp_url()` helper

2. **verify_homography_gps.py**
   - GPS coordinates loaded automatically from config
   - Simplified usage: `python verify_homography_gps.py Valte`
   - No need to pass GPS coordinates as arguments

3. **verify_homography.py**
   - Uses centralized config
   - Cleaner imports

4. **debug_homography.py**
   - Removed duplicate camera definitions
   - Uses centralized config

5. **check_camera_tilt.py**
   - Uses centralized config

---

## Benefits

### Before (Scattered Configuration):
```python
# In main.py
CAMERAS = [...]
USERNAME = "admin"
PASSWORD = "..."

# In debug_homography.py
CAMERAS = [...]  # Duplicated!
USERNAME = "admin"  # Duplicated!

# In verify_homography_gps.py
python verify_homography_gps.py Valte "39°38'25.7\"N" "0°13'48.7\"W" 5.0  # Long!
```

### After (Centralized):
```python
# In all files
from camera_config import get_camera_by_name, get_rtsp_url, ...

# Usage
python verify_homography_gps.py Valte  # Simple!
```

---

## Camera Configuration Format

```python
CAMERAS = [
    {
        "ip": "10.207.99.178",
        "name": "Valte",
        "lat": "39°38'25.7\"N",
        "lon": "0°13'48.7\"W",
        "height_m": 5.0,  # Default height, calibrate with GPS
        "description": "Valte camera location"
    },
    # ... more cameras
]
```

---

## Usage Examples

### View Configuration:
```bash
python camera_config.py
```

### Verify Homography (Now Simpler!):
```bash
# Before:
python verify_homography_gps.py Valte "39°38'25.7\"N" "0°13'48.7\"W" 5.0

# After:
python verify_homography_gps.py Valte          # Uses config GPS & height
python verify_homography_gps.py Valte 11.3     # Custom height
```

### Check All Cameras:
```bash
python check_camera_tilt.py  # Automatically checks all cameras in config
```

### Debug Homography:
```bash
python debug_homography.py Valte  # Uses config automatically
```

---

## Adding New Cameras

Edit `camera_config.py`:

```python
CAMERAS = [
    # ... existing cameras ...
    {
        "ip": "10.x.x.x",
        "name": "NewCamera",
        "lat": "XX°XX'XX.X\"N",
        "lon": "X°XX'XX.X\"W",
        "height_m": 5.0,  # Measure or calibrate with GPS
        "description": "New camera location"
    },
]
```

All tools will automatically detect the new camera!

---

## Updating Camera Heights

After GPS calibration, update the default height in `camera_config.py`:

```python
{
    "name": "Valte",
    "height_m": 11.3,  # Updated from GPS calibration
    # ...
}
```

Now all tools will use the calibrated height by default.

---

## API Reference

### `get_camera_by_name(camera_name: str) -> dict`
Get full camera configuration.

```python
cam = get_camera_by_name("Valte")
# Returns: {"ip": "...", "name": "...", "lat": "...", ...}
```

### `get_camera_gps(camera_name: str) -> dict`
Get GPS coordinates only.

```python
gps = get_camera_gps("Valte")
# Returns: {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
```

### `get_rtsp_url(camera_name: str, stream_type: str = "main") -> str`
Generate RTSP URL.

```python
url = get_rtsp_url("Valte")
# Returns: "rtsp://admin:pass@10.207.99.178:554/Streaming/Channels/101"

url = get_rtsp_url("Valte", "sub")
# Returns: "rtsp://admin:pass@10.207.99.178:554/Streaming/Channels/102"
```

---

## Migration Checklist

- [x] Created `camera_config.py`
- [x] Updated `main.py`
- [x] Updated `verify_homography_gps.py`
- [x] Updated `verify_homography.py`
- [x] Updated `debug_homography.py`
- [x] Updated `check_camera_tilt.py`
- [x] Tested configuration loading
- [x] Simplified command-line usage

---

## Backward Compatibility

The new system maintains compatibility:
- All existing scripts still work
- Command-line interfaces simplified (fewer args needed)
- GPS coordinates loaded automatically
- Optional arguments still supported for overrides

---

## Next Steps

1. **Test the simplified commands**:
   ```bash
   python verify_homography_gps.py Valte
   ```

2. **Update camera heights** after GPS calibration in `camera_config.py`

3. **Add new cameras** as needed to the central config

Configuration is now centralized and consistent across all tools! 🎉
