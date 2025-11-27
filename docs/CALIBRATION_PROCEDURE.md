# Camera Calibration Procedure

## Overview

Camera calibration is essential for accurate spatial measurements and projections in computer vision applications. This guide covers the complete workflow for calibrating PTZ cameras to correct lens distortion and compute intrinsic parameters.

### Why Calibration is Important

Real camera lenses introduce distortion that causes straight lines in the world to appear curved in images. This is particularly pronounced at image edges and in wide-angle lenses. Without calibration:

- Projected coordinates can be off by 1-5% at image edges
- Homography-based ground plane projections become inaccurate
- Spatial measurements and object tracking suffer from systematic errors

### What is Lens Distortion?

Lens distortion is an optical aberration that causes deviations from ideal pinhole camera projection:

- **Radial distortion**: Points are displaced radially from the image center. Barrel distortion (bulging outward) and pincushion distortion (pinching inward) are common types.
- **Tangential distortion**: Occurs when the lens is not perfectly parallel to the image sensor, causing asymmetric displacement.

### Expected Improvements

After calibration, you can expect:
- **1-5% improvement** in projection accuracy at image edges
- **Sub-pixel accuracy** for central image regions (< 0.5 pixels RMS error)
- More reliable homography transformations for ground plane mapping
- Better spatial consistency across the entire field of view

---

## Prerequisites

### Hardware Requirements

1. **Checkerboard Pattern**: A high-quality printed checkerboard calibration pattern
   - **Recommended size**: 10x7 squares (9x6 internal corners)
   - **Square size**: 25mm x 25mm (configurable)
   - **Print quality**: High-resolution printer on stiff, flat paper or cardboard

2. **Pattern Mounting**:
   - Mount pattern on a rigid, flat surface (foam board, cardboard, or acrylic)
   - Ensure pattern is perfectly flat with no warping or bending
   - Avoid glossy surfaces that create reflections

### Print Instructions

To generate a checkerboard pattern:

```bash
# Using Python with OpenCV (recommended)
python -c "import cv2; import numpy as np; \
pattern = np.zeros((700, 1000), dtype=np.uint8); \
for i in range(7): \
    for j in range(10): \
        if (i+j) % 2 == 0: \
            pattern[i*100:(i+1)*100, j*100:(j+1)*100] = 255; \
cv2.imwrite('checkerboard_10x7.png', pattern)"

# Then print at actual size (250mm x 175mm for 25mm squares)
```

Alternatively, use online checkerboard generators or the provided pattern templates.

### Software Requirements

- Python 3.8+
- OpenCV with checkerboard detection support
- NumPy
- Access to camera RTSP stream (for capture mode)

---

## Step-by-Step Calibration Guide

### Method 1: Interactive Capture Mode (Recommended)

Use this method to capture calibration images directly from the camera's RTSP stream.

#### Step 1: Prepare the Environment

1. **Lighting**: Ensure even, bright lighting without glare on the checkerboard
2. **Camera Setup**: Position the camera at its typical operating height and orientation
3. **Checkerboard Mounting**: Secure the checkerboard on a flat, rigid surface

#### Step 2: Run the Calibration Tool

```bash
# Basic capture mode (default 9x6 pattern, 25mm squares)
python calibrate_camera.py --camera Valte --mode capture --images ./calibration_images

# Custom pattern size and square dimensions
python calibrate_camera.py --camera Setram --mode capture \
    --pattern 7x5 --square-size 30.0 --images ./calibration_images
```

**Parameters**:
- `--camera`: Camera name from `camera_config.py` (e.g., "Valte", "Setram")
- `--mode capture`: Interactive capture from RTSP stream
- `--images`: Directory to save captured images
- `--pattern`: Checkerboard pattern as `WIDTHxHEIGHT` internal corners (default: 9x6)
- `--square-size`: Size of each square in millimeters (default: 25.0)
- `--min-images`: Minimum images required (default: 10)

#### Step 3: Capture Calibration Images

The tool will open a live video feed with real-time corner detection:

**Interactive Controls**:
- **Press 'c'**: Capture image when corners are detected (green overlay)
- **Press 'q'**: Quit capture mode and proceed to calibration

**Tips for Good Coverage**:

1. **Vary Position**: Move the checkerboard to different image regions
   - Top-left, top-right, bottom-left, bottom-right corners
   - Center of the frame
   - Edges at multiple locations

2. **Vary Orientation**: Rotate the checkerboard
   - Horizontal orientation
   - Vertical orientation
   - Diagonal orientations (~45 degrees)

3. **Vary Distance**: Capture at different depths
   - Close to camera (fills ~60% of frame)
   - Medium distance (fills ~40% of frame)
   - Far from camera (fills ~20% of frame)

4. **Aim for 15-20 images**: More images = better calibration
   - Minimum: 10 images (required)
   - Recommended: 15-20 images
   - Diminishing returns beyond 30 images

**Quality Indicators**:
- Green overlay = corners detected successfully
- Red text = no corners detected, adjust position/lighting
- Clear, sharp checkerboard edges in captured images

#### Step 4: Review Calibration Results

After capturing sufficient images and pressing 'q', the tool will automatically:

1. **Process images**: Detect and refine corner positions
2. **Compute calibration**: Calculate camera matrix and distortion coefficients
3. **Display results**: Show calibration quality metrics
4. **Save to file**: Store results in JSON format

**Example Output**:
```
======================================================================
CALIBRATION SUCCESSFUL
======================================================================

Calibration Quality:
  RMS reprojection error: 0.342 pixels
  Mean error per image:   0.338 pixels
  Quality assessment:     Excellent

Camera Matrix (K):
[[2567.8  0.0     1280.0]
 [0.0     2567.8  720.0 ]
 [0.0     0.0     1.0   ]]

Distortion Coefficients [k1, k2, p1, p2, k3]:
[-0.2841  0.0923  0.0001  -0.0002  -0.0147]

Derived Parameters:
  Focal length (fx, fy): (2567.80, 2567.80) pixels
  Field of view (x, y):  (28.45, 16.14) degrees
  Principal point (cx, cy): (1280.00, 720.00)
```

---

### Method 2: Directory Mode (Process Existing Images)

Use this method if you already have calibration images.

```bash
# Process pre-captured images
python calibrate_camera.py --images ./calibration_images --output calibration.json

# Custom pattern
python calibrate_camera.py --images ./my_images \
    --pattern 8x6 --square-size 30.0 --output calibration_custom.json
```

The tool will process all images in the directory and perform calibration.

---

## Interpreting Calibration Results

### RMS Reprojection Error

The RMS (Root Mean Square) reprojection error measures how accurately the calibrated model can reproject 3D points back onto the 2D image plane.

**Quality Assessment**:
- **< 0.5 pixels**: Excellent - Professional-grade calibration
- **0.5 - 1.0 pixels**: Good - Suitable for most applications
- **1.0 - 2.0 pixels**: Acceptable - May be adequate depending on use case
- **> 2.0 pixels**: Poor - Consider recalibrating with better images

**Factors Affecting RMS Error**:
- Image coverage and diversity
- Checkerboard flatness and print quality
- Lighting conditions and image sharpness
- Lens quality and camera stability

### Camera Matrix (K)

The intrinsic camera matrix contains:
- **fx, fy**: Focal lengths in pixels (typically similar for square pixels)
- **cx, cy**: Principal point (optical center, usually near image center)

### Distortion Coefficients

Five coefficients model lens distortion:
- **k1, k2, k3**: Radial distortion (most significant: k1)
- **p1, p2**: Tangential distortion (typically small)

**Typical Values**:
- Wide-angle lenses: k1 < -0.2 (barrel distortion)
- Telephoto lenses: k1 > 0 (pincushion distortion)
- k2, k3: Usually smaller in magnitude than k1

---

## Using Calibration Results

### Automatic Storage

Calibration results are automatically saved to two locations:

1. **Standalone file**: `calibration_<camera_name>.json` (in current directory)
2. **Persistent storage**: `calibrations/<camera_name>_calibration.json`

### Integration with Application

The calibration data is automatically loaded by the camera geometry pipeline:

```python
from camera_config import get_camera_by_name
from camera_geometry import CameraGeometry

# Get camera configuration (automatically loads calibration)
camera_info = get_camera_by_name("Valte")

# Initialize geometry with distortion correction
geometry = CameraGeometry(w=1920, h=1080)
geometry.set_camera_parameters(
    K=camera_matrix,
    w_pos=np.array([0, 0, 5.0]),
    pan_deg=0,
    tilt_deg=-15,
    map_width=640,
    map_height=640,
    distortion_coeffs=distortion_coeffs  # Loaded automatically
)

# Projections now use undistorted coordinates
world_points = geometry.project_image_to_map([(100, 200), (300, 400)], 640, 640)
```

### Verify Calibration is Active

Check if a camera has calibration data:

```python
from calibration_storage import has_calibration, get_calibration_info

# Check if calibration exists
if has_calibration("Valte"):
    info = get_calibration_info("Valte")
    print(f"Calibration date: {info['calibration_date']}")
    print(f"RMS error: {info['rms_error']:.3f} pixels")
    print(f"Images used: {info['num_images']}")
else:
    print("Camera not calibrated - using zero distortion")
```

### List All Calibrations

```python
from calibration_storage import list_calibrations

calibrated_cameras = list_calibrations()
print(f"Calibrated cameras: {calibrated_cameras}")
```

---

## Troubleshooting

### Issue: Corners Not Detected

**Symptoms**: Red text "No corners detected" during capture

**Solutions**:
1. **Improve Lighting**: Ensure even, bright lighting without shadows
2. **Reduce Glare**: Avoid glossy surfaces or direct reflections
3. **Check Focus**: Ensure camera is focused properly
4. **Adjust Distance**: Move checkerboard closer or farther from camera
5. **Verify Pattern**: Confirm checkerboard has correct number of squares
6. **Flatten Pattern**: Ensure checkerboard is completely flat

### Issue: High RMS Error (> 1.0 pixels)

**Symptoms**: Poor quality assessment after calibration

**Solutions**:
1. **Improve Image Coverage**: Capture more images at image corners and edges
2. **Check Pattern Flatness**: Ensure checkerboard is not warped or bent
3. **Increase Image Count**: Capture 20-30 images instead of minimum 10
4. **Verify Print Quality**: Use high-quality printer with sharp edges
5. **Stabilize Camera**: Ensure camera is not moving during capture
6. **Check Lens Quality**: Poor lens quality may limit achievable accuracy

### Issue: Calibration Fails (Insufficient Images)

**Symptoms**: Error message "Insufficient images for calibration"

**Solutions**:
1. **Capture More Images**: Minimum 10 required, 15-20 recommended
2. **Verify Corner Detection**: Ensure corners are detected in captured images
3. **Adjust Pattern Parameters**: Verify `--pattern` matches actual checkerboard
4. **Check Image Directory**: Ensure images are saved correctly

### Issue: Inconsistent Results Between Calibrations

**Symptoms**: Calibration parameters vary significantly between runs

**Solutions**:
1. **Use Same Pattern**: Always use identical checkerboard dimensions
2. **Consistent Environment**: Maintain similar lighting and camera position
3. **More Images**: Increase sample size for statistical stability
4. **Fixed Camera Settings**: Lock zoom, focus, and exposure during capture

### Issue: Distortion Correction Not Applied

**Symptoms**: Projections still show edge errors after calibration

**Solutions**:
1. **Verify File Location**: Check calibration saved to `calibrations/` directory
2. **Check Camera Name**: Ensure camera name matches exactly (case-sensitive)
3. **Reload Configuration**: Restart application to load new calibration
4. **Examine Coefficients**: Verify distortion coefficients are non-zero
5. **Test Integration**: Run `test_distortion_integration.py` to verify

---

## When to Recalibrate

Recalibration is recommended when:

1. **Camera Hardware Changes**:
   - Lens replaced or adjusted
   - Zoom level changed (requires new calibration per zoom level)
   - Camera resolution changed

2. **Physical Damage**:
   - Camera dropped or impacted
   - Lens scratched or damaged
   - Mounting bracket adjusted

3. **Regular Maintenance**:
   - Every 6-12 months for critical applications
   - After extended outdoor exposure (temperature, moisture)

4. **Performance Degradation**:
   - Noticeable projection errors at image edges
   - GPS validation shows systematic biases
   - Tracking accuracy decreases over time

---

## Advanced Topics

### Multiple Zoom Levels

Different zoom levels require separate calibrations:

```bash
# Calibrate at zoom 1x
python calibrate_camera.py --camera Valte --mode capture \
    --images ./calibration_1x --output calibration_Valte_1x.json

# Calibrate at zoom 2x
python calibrate_camera.py --camera Valte --mode capture \
    --images ./calibration_2x --output calibration_Valte_2x.json
```

Store multiple calibrations and load based on current zoom setting.

### Custom Pattern Sizes

For larger or smaller checkerboards:

```bash
# Large pattern: 12x9 squares (11x8 internal corners), 40mm squares
python calibrate_camera.py --camera Valte --mode capture \
    --pattern 11x8 --square-size 40.0 --images ./calibration_images

# Small pattern: 6x4 squares (5x3 internal corners), 15mm squares
python calibrate_camera.py --camera Valte --mode capture \
    --pattern 5x3 --square-size 15.0 --images ./calibration_images
```

**Larger patterns**: Better for distant calibration, requires more space
**Smaller patterns**: Better for close-range, easier to position

### Validating Calibration Quality

Test calibration accuracy with GPS validation:

```bash
# Test projection accuracy using known GPS points
python verify_homography_gps.py --camera Valte

# Visual distortion correction test
python example_use_calibration.py
```

Compare projection errors before and after calibration to quantify improvement.

---

## Quick Reference

### Complete Calibration Workflow

```bash
# 1. Prepare checkerboard (9x6 internal corners, 25mm squares)
# 2. Capture calibration images
python calibrate_camera.py --camera Valte --mode capture --images ./calibration_images

# 3. Verify results
python -c "from calibration_storage import get_calibration_info; \
info = get_calibration_info('Valte'); \
print(f'RMS Error: {info[\"rms_error\"]:.3f} pixels')"

# 4. Test integration
python test_distortion_integration.py
```

### File Locations

- **Calibration tool**: `calibrate_camera.py`
- **Storage module**: `calibration_storage.py`
- **Camera config**: `camera_config.py`
- **Geometry pipeline**: `camera_geometry.py`
- **Calibration files**: `calibrations/<camera_name>_calibration.json`

### Key Commands

```bash
# Interactive capture (RTSP)
python calibrate_camera.py --camera <name> --mode capture --images <dir>

# Process existing images
python calibrate_camera.py --images <dir> --output <file.json>

# List calibrated cameras
python calibration_storage.py

# Delete calibration
python -c "from calibration_storage import delete_calibration; delete_calibration('<name>')"
```

---

## Support and Resources

For issues or questions:
1. Check troubleshooting section above
2. Review calibration tool help: `python calibrate_camera.py --help`
3. Examine example usage: `example_use_calibration.py`
4. Consult technical documentation in `docs/` directory

**Related Documentation**:
- `CAMERA_CONFIG_MIGRATION.md` - Camera configuration guide
- `GPS_VALIDATION_GUIDE.md` - GPS-based accuracy validation
- `docs/homography_verification_guide.md` - Homography testing procedures
