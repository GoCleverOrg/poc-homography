# GPS Validation Guide for Homography

## Your Test Results

**Camera GPS:** 39°38'25.7"N, 0°13'48.7"W
**Point GPS:** 39°38'25.6"N, 0°13'48.4"W

**Actual Distance (GPS):** 7.78 meters
**Homography Distance:** 3.44 meters
**Error:** -4.34m (underestimate by 56%)

**Conclusion:** Camera height is set too LOW (current: 5.0m, likely actual: ~11m)

---

## Quick Distance Check

```bash
# Calculate actual GPS distance between two points
python gps_distance_calculator.py "39°38'25.7\"N" "0°13'48.7\"W" "39°38'25.6\"N" "0°13'48.4\"W" 3.44
```

**Output:**
- GPS Distance (Haversine): 7.78m
- Homography Distance: 3.44m
- Error: -4.34m (-55.8%)
- Assessment: 🟠 POOR

---

## Interactive GPS Validation (RECOMMENDED)

```bash
python verify_homography_gps.py Valte "39°38'25.7\"N" "0°13'48.7\"W" 5.0
```

### What It Does:

1. **Opens live camera stream**
2. **Click on a ground point**
3. **Shows ESTIMATED GPS** from homography (NEW!)
   ```
   🌍 ESTIMATED GPS (from homography):
     Latitude:  39°38'25.6"N (39.640444°)
     Longitude: 0°13'48.3"W (-0.230083°)
   ```
4. **Enter ACTUAL GPS** (from phone/map/etc.)
5. **Compares and shows error**
6. **Suggests better camera height**

### Workflow:

```
1. Click object in video
   ↓
2. Tool shows: "Homography estimates GPS: 39°38'25.6"N, 0°13'48.3"W"
   ↓
3. You check on phone/map: Actual GPS is 39°38'25.6"N, 0°13'48.4"W
   ↓
4. Enter actual GPS
   ↓
5. Tool shows:
      GPS Distance:    7.78m
      Homography:      3.44m
      Error:           -4.34m
      Suggested height: 11.3m
   ↓
6. Press 'r' to recalibrate with new height
   ↓
7. Repeat with more points to verify
```

---

## Understanding the Error

### Why Homography Underestimated:

Your homography gave **3.44m** but actual is **7.78m**.

**Scale factor:** 7.78 / 3.44 = **2.26x**

This means:
- Camera height is likely **2.26x higher** than current setting
- Current: 5.0m → Estimated actual: **~11.3m**

### Height Formula:

```
correct_height = current_height * (gps_distance / homography_distance)
correct_height = 5.0 * (7.78 / 3.44) = 11.3m
```

---

## Calibration Workflow

### Step 1: Collect Multiple GPS Points

```bash
python verify_homography_gps.py Valte "39°38'25.7\"N" "0°13'48.7\"W" 5.0
```

Click 3-5 different ground points at various distances:
- Near point (~5-10m)
- Medium point (~10-20m)
- Far point (~20-30m)

For each:
1. Click in video
2. Note estimated GPS
3. Measure actual GPS (smartphone, map, etc.)
4. Enter actual GPS
5. Tool calculates height suggestion

### Step 2: Recalibrate

After collecting points, press **'r'** to auto-recalibrate:
- Tool averages all height suggestions
- Recomputes homography with new height
- Clears points for fresh validation

### Step 3: Verify

Click new points and check if errors are now < 1m.

---

## GPS Coordinate Sources

### Option 1: Smartphone GPS
1. Walk to the point
2. Use GPS Status app / Google Maps
3. Long-press location to get coordinates
4. Copy GPS coordinates

### Option 2: Google Maps (Desktop)
1. Right-click on point in Google Maps
2. Select "What's here?"
3. Copy coordinates from bottom panel

### Option 3: Satellite Imagery with Known Features
1. Use Google Earth Pro
2. Place pins on known features
3. Right-click → Properties → Get coordinates

---

## Expected Accuracy After Calibration

| Distance | Expected Error | Status |
|----------|----------------|--------|
| 0-10m | ±0.5m | 🟢 Excellent |
| 10-20m | ±1.0m | 🟢 Good |
| 20-50m | ±2-5m | 🟡 Acceptable |
| >50m | ±10m+ | 🔴 Poor |

---

## Troubleshooting

### "All distances are scaled wrong"
**Cause:** Camera height incorrect
**Fix:** Use GPS validation to find correct height

### "Distances correct in one direction, wrong in another"
**Cause:** Ground not flat / elevation changes
**Fix:** Homography assumes flat ground (Z=0)
**Solution:** Use only points on same elevation as camera base

### "Close points accurate, far points wrong"
**Cause:** Lens distortion or ground curvature
**Fix:** Calibrate on medium-distance points (10-20m)

### "GPS coordinates don't match on map"
**Cause:** GPS accuracy varies (±3-10m typical)
**Fix:** Use multiple points and average

---

## Tools Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `gps_distance_calculator.py` | Calculate GPS distance | Quick distance check |
| `verify_homography_gps.py` | Interactive GPS validation | Full calibration workflow |
| `test_gps_reverse.py` | Test GPS calculations | Verify GPS conversion math |
| `debug_homography.py` | Debug homography math | Troubleshoot projection issues |
| `check_camera_tilt.py` | Check camera orientation | Verify cameras point down |

---

## Real-World Example

**Your Data:**
```
Camera: Valte
Camera GPS: 39°38'25.7"N, 0°13'48.7"W
Camera Height (initial): 5.0m

Point clicked:
  Image: (1280, 800)px
  Homography: 3.44m

Estimated GPS (from homography):
  39°38'25.6"N, 0°13'48.3"W

Actual GPS (from phone):
  39°38'25.6"N, 0°13'48.4"W

Actual distance: 7.78m
Error: -4.34m (-56%)

Suggested new height: 11.3m
```

**Action:** Measure actual camera height with tape measure or laser. If close to 11m, use that value. Otherwise, use GPS-calibrated value.

---

## Limitations

1. **GPS Accuracy:** ±3-10m typical for smartphones
2. **Ground Plane Assumption:** Only works if all points are at same elevation (flat ground)
3. **Distance Range:** Best results 5-50m from camera
4. **Lens Distortion:** Not corrected (affects edges of image)

---

## Next Steps

1. ✅ Run `python verify_homography_gps.py Valte "39°38'25.7\"N" "0°13'48.7\"W" 5.0`
2. ✅ Click 3-5 known points
3. ✅ Enter their GPS coordinates
4. ✅ Let tool suggest height correction
5. ✅ Recalibrate with 'r'
6. ✅ Verify with new points (error < 1m)
7. ✅ Update height in main.py permanently

**Target:** Get all errors < 1m for points within 20m of camera.
