# Homography Verification Tools

Quick guide to verify your homography is working correctly.

## 🚀 Quick Start

### 1. Run Automated Tests (30 seconds)
```bash
python tests/test_homography_consistency.py
```
**What it does**: Mathematical consistency checks
**Expected result**: All tests pass with errors < 0.01m

### 2. Interactive Verification (5 minutes)
```bash
python verify_homography.py Valte
```
**What it does**: Click points in live stream to see world coordinates
**What you need**: Markers at known distances (5m, 10m, etc.)

### 3. Run Your Application
```bash
python main.py
```
**What it does**: Process live stream with homography-projected annotations

---

## 📊 Verification Methods

| Method | Time | Accuracy | Tools Needed |
|--------|------|----------|--------------|
| **Automated Tests** | 30s | Mathematical only | None |
| **Interactive Clicking** | 5min | ±0.5m | Physical markers |
| **Visual Inspection** | 2min | Qualitative | None |

---

## ✅ Quick Verification Checklist

Run automated tests:
```bash
python tests/test_homography_consistency.py
```

Look for:
- ✓ Round-trip error < 0.01m
- ✓ Principal point projects ahead (not behind)
- ✓ Horizon at top of image
- ✓ det(H) > 1e-6

---

## 🎯 Interactive Verification Example

```bash
# Start interactive verifier
python verify_homography.py Valte

# Place physical marker 10m ahead
# Click on marker base in video window
# Check output:
📍 Point 1:
  Image: (1280, 1200) pixels
  World: (0.15, 9.92) meters
  Distance from camera: 9.93m
  ✓ Expected ~10m → Got 9.93m (0.7% error)
```

---

## 🔧 Common Adjustments

### Camera Height Wrong?
```python
# In main.py, adjust:
CAMERA_HEIGHT_M = 5.5  # Measure actual height
```

### Tilt Angle Off?
```python
# Add calibration offset:
tilt=status["tilt"] + TILT_OFFSET  # Try ±3° adjustment
```

### Pan Angle Off?
```python
# Add calibration offset:
pan=status["pan"] + PAN_OFFSET  # Try ±5° adjustment
```

---

## 📖 Full Documentation

See [docs/homography_verification_guide.md](docs/homography_verification_guide.md) for:
- Detailed verification procedures
- Troubleshooting common issues
- Calibration procedures
- Expected accuracy metrics
- Real-world examples

---

## 🐛 Troubleshooting

### Homography is singular (det ≈ 0)
**Cause**: Camera pointing at or above horizon
**Fix**: Ensure tilt < -10° (pointing down)

### Distances all wrong by same factor
**Cause**: Incorrect camera height
**Fix**: Measure and update actual height

### Forward OK, left/right wrong
**Cause**: Pan angle offset
**Fix**: Calibrate pan zero position

### Round-trip error > 0.1m
**Cause**: Implementation bug
**Fix**: Check rotation matrix order (should be Pan then Tilt)

---

## 📝 Files Overview

```
verify_homography.py              # Interactive verification tool
tests/test_homography_consistency.py  # Automated math tests
docs/homography_verification_guide.md # Detailed guide
camera_geometry.py                # Core homography implementation
main.py                          # Main application
```

---

## 🎓 Understanding the Output

When you click a point in `verify_homography.py`:

```
📍 Point 1:
  Image: (1280, 1200) pixels     ← Where you clicked
  World: (0.15, 9.92) meters     ← Computed ground position
  Distance from camera: 9.93m    ← Euclidean distance
  Angle from camera: 0.9°        ← Direction (0°=ahead)
```

**Coordinate System**:
- X: East (+) / West (-)
- Y: North/Forward (+) / South/Backward (-)
- Origin: Camera position
- Ground plane: Z = 0

---

## ⚡ Expected Performance

| Distance Range | Accuracy | Confidence |
|----------------|----------|------------|
| 0-10m | ±0.5m | High |
| 10-20m | ±1.0m | Medium |
| 20-50m | ±2-5m | Low |
| >50m | ±10m+ | Very Low |

Factors affecting accuracy:
- Camera height measurement error
- Pan/tilt angle errors
- Lens distortion (not corrected)
- Ground plane assumption violations

---

## 🚀 Next Steps

1. Run automated tests → Verify math is correct
2. Run interactive tool → Verify real-world accuracy
3. Adjust parameters if needed → Calibrate for your setup
4. Run main application → Process live streams!

Happy verifying! 🎉
