# Homography Verification Guide

## Overview
This guide explains how to verify that your homography calculations are correct using your live camera stream.

## Prerequisites
- Live RTSP camera stream access
- Camera mounted at known height
- Ability to place physical markers in the scene (optional but helpful)

---

## Method 1: Interactive Point Verification ⭐ RECOMMENDED

### What You Need:
- Physical markers (cones, boxes, tape marks) at known distances
- Measuring tape or laser distance meter
- The verification script

### Steps:

1. **Place Physical Markers**
   ```
   Example setup:
   - Marker A: 5m straight ahead from camera
   - Marker B: 10m straight ahead from camera
   - Marker C: 5m ahead, 3m to the right
   - Marker D: 8m ahead, 2m to the left
   ```

2. **Run the Interactive Verifier**
   ```bash
   python verify_homography.py Valte
   ```

3. **Click on Markers**
   - Click on the **BASE** of each marker (where it touches the ground)
   - The tool will show you the computed world coordinates

4. **Compare Results**
   ```
   Expected vs Computed:
   Marker A: Expected (0.0, 5.0)m   → Got (0.1, 5.2)m  ✓ Good (error < 0.5m)
   Marker B: Expected (0.0, 10.0)m  → Got (0.2, 9.8)m  ✓ Good
   Marker C: Expected (3.0, 5.0)m   → Got (2.9, 5.1)m  ✓ Good
   ```

5. **Acceptable Error Margins**
   - **Excellent**: < 0.3m error at 10m distance
   - **Good**: < 0.5m error at 10m distance
   - **Acceptable**: < 1.0m error at 10m distance
   - **Poor**: > 1.0m error → Check camera parameters

---

## Method 2: Automated Consistency Tests

Run mathematical consistency checks:

```bash
python tests/test_homography_consistency.py
```

### What It Tests:

1. **Round-Trip Consistency**
   - Projects world → image → world
   - Should recover original coordinates within 1cm

2. **Principal Point Test**
   - Image center should project to expected distance
   - Expected distance = height / tan(|tilt|)
   - Example: 5m height at -45° tilt → ~5m ahead

3. **Horizon Behavior**
   - Points near top of image should project far away
   - Points near bottom should be close

4. **Pan/Tilt/Zoom Effects**
   - Pan should rotate world coordinates
   - Tilt should change forward distance
   - Zoom should scale the field of view

---

## Method 3: Visual Cross-Check

### Parallel Lines Test
Real-world parallel lines (road edges, building sides) should:
- Converge to vanishing point in image
- Remain parallel in top-down view

### Distance Ladder Test
Place markers at 5m, 10m, 15m, 20m:
- Click each marker in the interactive tool
- Verify computed distances match actual

---

## Common Issues and Fixes

### Issue 1: All Distances Are Wrong by a Scale Factor
**Symptom**: Computed 10m when actual is 15m (consistent 1.5x error)

**Likely Cause**: Incorrect camera height
```python
# Fix: Adjust height parameter
height=7.5  # Try different values until scale matches
```

### Issue 2: Forward Distance OK, But Left/Right Wrong
**Symptom**: Y-coordinate correct, X-coordinate off

**Likely Cause**: Pan angle incorrect
```python
# Fix: Calibrate pan zero position
pan_deg=5.0  # Add offset to pan
```

### Issue 3: All Points Project Too Far/Too Close
**Symptom**: 5m marker projects to 15m

**Likely Cause**: Tilt angle incorrect
```python
# Fix: Verify tilt angle from camera
# Pass tilt directly from camera - positive = pointing down (Hikvision convention)
# The internal _get_rotation_matrix() handles the conversion
tilt_deg=40.0  # Try adjusting ±5°
```

### Issue 4: Homography Determinant Near Zero
**Symptom**: Warning about singular homography

**Likely Cause**: Camera pointing at or above horizon
```python
# Fix: Camera must point downward
# Ensure tilt > 10° (positive = down in Hikvision convention)
```

---

## Calibration Procedure

If your measurements are consistently off:

### Step 1: Verify Camera Height
```python
# Measure actual camera height with tape measure
ACTUAL_HEIGHT = 5.2  # meters (not 5.0)
```

### Step 2: Calibrate Tilt Offset
```python
# Place marker at known distance
# Adjust tilt until distance matches
for tilt_offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
    tilt_deg = camera.get_status()["tilt"] + tilt_offset
    # Test and find best match
```

### Step 3: Calibrate Pan Offset
```python
# Place marker directly ahead
# Adjust pan until X-coordinate ≈ 0
for pan_offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
    pan_deg = camera.get_status()["pan"] + pan_offset
    # Test and find best match
```

---

## Expected Accuracy

With proper calibration:

| Distance | Typical Error | Notes |
|----------|---------------|-------|
| 0-5m     | ± 0.2m        | Very accurate |
| 5-10m    | ± 0.5m        | Good accuracy |
| 10-20m   | ± 1.0m        | Acceptable |
| 20-50m   | ± 2-5m        | Decreases with distance |
| >50m     | ± 10m+        | Unreliable |

**Note**: Accuracy degrades with distance due to:
1. Pixel resolution limits
2. Lens distortion (not corrected)
3. Small angle errors amplified at distance

---

## Quick Verification Checklist

- [ ] Camera height measured accurately
- [ ] Camera pointing downward (tilt < -10°)
- [ ] Test point at 5m → computes 4.5-5.5m ✓
- [ ] Test point at 10m → computes 9.0-11.0m ✓
- [ ] Point ahead → X-coordinate ≈ 0 ✓
- [ ] Point to right → X-coordinate > 0 ✓
- [ ] Point to left → X-coordinate < 0 ✓
- [ ] Round-trip error < 0.01m ✓
- [ ] Homography det(H) > 1e-6 ✓

---

## Advanced: Lens Distortion

This implementation **does not correct lens distortion**. For better accuracy:

1. Calibrate camera with checkerboard pattern
2. Obtain distortion coefficients (k1, k2, p1, p2)
3. Undistort images before applying homography

Lens distortion typically causes:
- Straight lines to appear curved
- Distance errors increasing toward image edges
- Radial distortion (barrel or pincushion effect)

For most surveillance applications, uncorrected homography is sufficient for distances < 20m.

---

## Troubleshooting Commands

```bash
# Run automated tests
python tests/test_homography_consistency.py

# Interactive verification
python verify_homography.py Valte

# Test with different heights
python verify_homography.py Valte 6.5  # 6.5m height

# Check camera status
python -c "
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ
cam = HikvisionPTZ('10.207.99.178', 'admin', 'CameraLab01*', 'Valte')
print(cam.get_status())
"
```

---

## Real-World Example

**Scenario**: Surveillance camera monitoring a parking lot

**Setup**:
- Camera height: 5.0m
- Tilt: -45°
- Pan: 0° (facing North)
- Zoom: 1.0x

**Verification**:
1. Placed cone at parking space 10m away
2. Clicked on cone base in stream
3. Tool computed: (0.2m, 9.8m)
4. Error: 0.28m at 10m distance
5. ✓ Accuracy confirmed: 2.8% error

**Result**: Homography verified and working correctly!
