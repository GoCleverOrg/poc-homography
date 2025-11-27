# Camera Calibrations Directory

This directory stores camera calibration data files in JSON format.

## Purpose

Calibration files contain camera intrinsic parameters (camera matrix K) and distortion coefficients for each camera, allowing for accurate lens distortion correction in the homography pipeline.

## File Naming Convention

Calibration files follow the pattern:
```
{camera_name}_calibration.json
```

Examples:
- `Valte_calibration.json`
- `Setram_calibration.json`

## File Format

Each calibration file contains:

```json
{
  "camera_name": "Valte",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coeffs": [k1, k2, p1, p2, k3],
  "image_size": [width, height],
  "rms_error": 0.345,
  "num_images": 15,
  "pattern_size": [9, 6],
  "square_size_mm": 25.0,
  "calibration_date": "2025-11-27T12:34:56.789012"
}
```

## Creating Calibrations

To create a new calibration file:

1. **Using calibrate_camera.py (Interactive Capture)**:
   ```bash
   python calibrate_camera.py --camera Valte --mode capture --images ./calibration_images
   ```

2. **Using calibrate_camera.py (Process Existing Images)**:
   ```bash
   python calibrate_camera.py --images ./calibration_images --output calibrations/Valte_calibration.json
   ```

3. **Programmatically using calibration_storage.py**:
   ```python
   from calibration_storage import save_calibration
   import numpy as np

   save_calibration(
       camera_name="Valte",
       camera_matrix=K,
       distortion_coeffs=dist,
       image_size=(1920, 1080),
       rms_error=0.345,
       num_images=15,
       pattern_size=(9, 6),
       square_size_mm=25.0
   )
   ```

## Loading Calibrations

Calibrations are automatically loaded by `camera_config.py`:

```python
from camera_config import get_camera_distortion, get_camera_calibration

# Get distortion coefficients only
dist = get_camera_distortion("Valte")

# Get full calibration (camera matrix + distortion)
calib = get_camera_calibration("Valte")
if calib:
    K, dist = calib
```

Or use `calibration_storage.py` directly:

```python
from calibration_storage import load_calibration

calib_data = load_calibration("Valte")
if calib_data:
    K = calib_data['camera_matrix']
    dist = calib_data['distortion_coeffs']
    rms_error = calib_data['rms_error']
```

## Listing Available Calibrations

```python
from calibration_storage import list_calibrations

cameras = list_calibrations()
print(f"Calibrated cameras: {cameras}")
```

Or run the module directly:
```bash
python calibration_storage.py
```

## Important Notes

- **Machine-Specific**: Calibration data is specific to each camera's physical characteristics and should not be committed to git
- **Excluded from Git**: This directory is included in `.gitignore` to prevent accidental commits
- **Storage Location**: Calibrations are stored locally and persist across sessions
- **Override Behavior**: Calibrations in config files (camera_config.py) take precedence over stored calibrations
