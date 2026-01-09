# GCP Capture Workflow for Accurate Homography Calibration

This document describes the recommended workflow for capturing Ground Control Points (GCPs) that will produce accurate camera-to-world coordinate transformations.

## Overview

GCPs are pairs of coordinates:
- **GPS coordinates** (latitude, longitude) - where a point is in the real world
- **Pixel coordinates** (u, v) - where that same point appears in the camera image

The accuracy of your homography depends entirely on the accuracy of these coordinate pairs.

## Prerequisites

1. **Camera calibration values** in `camera_config.py`:
   - Camera GPS position (lat, lon)
   - Camera height (height_m)
   - Pan offset (pan_offset_deg)
   - Tilt offset (tilt_offset_deg)
   - Lens distortion (k1, k2)

2. **Tools**:
   - `tools/cli/capture_gcps_web.py` - Web-based GCP capture tool
   - `tools/cli/verify_gcp_gps.py` - GPS verification map tool
   - `tools/cli/validate_camera_model.py` - Projection error validation

## Workflow

### Step 1: Identify Reference Points

Choose reference points that are:
- **Clearly visible** in the camera image
- **Precisely locatable** on satellite imagery (Google Earth, etc.)
- **At ground level** (not elevated structures)
- **Distributed across the image** (not clustered in one area)

Good reference points:
- Zebra crossing stripe corners
- Road marking intersections
- Manhole covers
- Building corners (at ground level)
- Curb intersections

Poor reference points:
- Tree canopy centers (GPS ≠ trunk position)
- Moving objects
- Points only visible from specific angles
- Points at varying elevations

### Step 2: Collect GPS Coordinates

**Option A: Google Earth (Recommended for accuracy)**

1. Open Google Earth Pro (desktop version for higher precision)
2. Navigate to your camera's location
3. Zoom in until you can clearly see reference points
4. Click on each reference point and note the coordinates
5. Use decimal degrees format (e.g., 39.640264, -0.229967)

**Option B: KML from Geotiff**

1. If using a georeferenced map (geotiff), extract points using the KML tool
2. **Important**: Verify KML coordinates match Google Earth!
   - Different geotiffs may have different geo-registration
   - Always cross-check a few points

**Option C: RTK GPS Survey**

1. Use RTK GPS equipment for centimeter-level accuracy
2. Mark each point and record coordinates
3. This is the most accurate method for critical applications

### Step 3: Verify GPS Coordinates

Before capturing GCPs, verify your GPS coordinates are accurate:

```bash
# Create a verification map
python tools/cli/verify_gcp_gps.py --gcps your_gcps.yaml --camera Valte

# Opens browser with GCPs plotted on satellite imagery
# Switch to "Satellite" view to check alignment
```

**Check for:**
- Points should overlay exactly on the features they represent
- No systematic offset (all points shifted in same direction)
- Consistent accuracy across near and far points

### Step 4: Capture GCPs

Use the web capture tool:

```bash
python tools/cli/capture_gcps_web.py \
    --camera Valte \
    --map-image your_map.png \
    --kml your_reference.kml
```

**Capture process:**
1. Click on a reference point in the camera image
2. Click on the corresponding point in the map
3. The tool shows the GPS coordinate - verify it matches your expected value
4. Repeat for 10-20 well-distributed points

**Tips:**
- Start with points near the image center (most accurate)
- Include points at varying distances from camera
- Include points on both left and right sides
- Avoid points at extreme edges of the image

### Step 5: Validate Projection Accuracy

After capturing GCPs, validate the projection model:

```bash
python tools/cli/validate_camera_model.py --camera Valte --gcps your_gcps.yaml
```

**Interpreting results:**
- `< 5px error`: Excellent - GCP is highly accurate
- `5-15px error`: Good - acceptable for most applications
- `15-30px error`: Fair - consider rechecking GPS accuracy
- `> 30px error`: Poor - GPS coordinate likely wrong

### Step 6: Iterate and Refine

If errors are high:

1. **Check individual GCP errors** - identify outliers
2. **Verify GPS coordinates** - cross-check with Google Earth
3. **Remove bad GCPs** - delete points with consistently high error
4. **Re-run validation** - confirm improvement

## Common Issues and Solutions

### Issue: All GCPs have systematic offset

**Symptom**: Projected points are all shifted in the same direction

**Causes**:
- Pan offset needs adjustment
- Camera GPS position is incorrect
- Using wrong KML source

**Solution**:
```bash
# Run parameter sweep to find optimal offset
python tools/cli/validate_camera_model.py --camera Valte --gcps your_gcps.yaml --sweep
```

### Issue: Near points accurate, far points have large errors

**Symptom**: Error increases with distance from camera

**Causes**:
- Tilt angle is incorrect
- Camera height is wrong

**Solution**:
- Verify tilt_offset_deg in camera config
- Re-measure camera height

### Issue: Legacy coordinate system (V-flip)

**Symptom**: ~500px vertical offset for all points

**Cause**: Old GCP files used leaflet_y coordinates (V=0 at bottom) instead of standard image coordinates (V=0 at top)

**Solution**: The tools automatically detect and convert legacy format. If capturing new GCPs, always use `coordinate_system: image_v`.

### Issue: KML coordinates don't match reality

**Symptom**: Points visually misaligned on verification map

**Cause**: Geotiff has incorrect geo-registration

**Solution**:
1. Verify geotiff alignment using known landmarks
2. Consider using Google Earth directly for GPS coordinates
3. Use multiple sources and cross-check

## File Formats

### GCP YAML Format (capture tool output)

```yaml
homography:
  feature_match:
    camera_capture_context:
      camera_name: "Valte"
      image_width: 1920
      image_height: 1080
      ptz_position:
        pan: 93.2
        tilt: 6.7
        zoom: 1.0
      coordinate_system: image_v  # V=0 at top (standard)

    ground_control_points:
      - gps:
          latitude: 39.640264
          longitude: -0.229967
        image:
          u: 904.2
          v: 612.6  # Standard image coordinates
        metadata:
          description: "zebra crossing corner"
```

### Simple GCP Format (for calibration tools)

```yaml
gcps:
  - lat: 39.640264
    lon: -0.229967
    pixel_u: 904.2
    pixel_v: 612.6
    pan_raw: 93.2
    tilt_deg: 6.7
    zoom: 1.0
```

## Quality Checklist

Before using GCPs for calibration:

- [ ] At least 10 GCPs distributed across the image
- [ ] GPS coordinates verified against satellite imagery
- [ ] No systematic offset visible on verification map
- [ ] Mean projection error < 15px
- [ ] No individual GCP with error > 30px
- [ ] GCPs captured at the correct PTZ position
- [ ] Coordinate system is `image_v` (not legacy format)

## Tools Reference

| Tool | Purpose |
|------|---------|
| `capture_gcps_web.py` | Capture GCPs with web interface |
| `verify_gcp_gps.py` | Generate map to verify GPS accuracy |
| `validate_camera_model.py` | Test projection accuracy |
| `comprehensive_calibration.py` | Optimize camera parameters |

## Calibration Parameters

After capturing accurate GCPs, these parameters can be optimized:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `height_m` | Camera height above ground | 3-10m |
| `pan_offset_deg` | Pan angle when camera reports 0 | 0-360° |
| `tilt_offset_deg` | Correction for reported tilt | -2° to +2° |
| `k1` | Primary radial distortion | -0.5 to 0 |
| `k2` | Secondary radial distortion | 0 to 1 |
