# Map Point Homography - Quick Start Guide

## Installation

No installation needed - already part of the `poc_homography` package.

## 5-Minute Quick Start

### 1. Import

```python
from poc_homography.homography_map_points import MapPointHomography
from poc_homography.map_points import MapPointRegistry
```

### 2. Load Map Points

```python
# Load from JSON file
registry = MapPointRegistry.load("map_points.json")
```

### 3. Define Ground Control Points (GCPs)

```python
gcps = [
    {"pixel_x": 798.0, "pixel_y": 578.4, "map_point_id": "A7"},
    {"pixel_x": 1082.0, "pixel_y": 390.4, "map_point_id": "A6"},
    {"pixel_x": 408, "pixel_y": 776.1, "map_point_id": "X15"},
    {"pixel_x": 568, "pixel_y": 846.1, "map_point_id": "X16"},
    # Need at least 4 GCPs, 10-20 recommended
]
```

### 4. Compute Homography

```python
homography = MapPointHomography()
result = homography.compute_from_gcps(gcps, registry)

# Check quality
print(f"Inliers: {result.inlier_ratio:.1%}")
print(f"Mean error: {result.mean_reproj_error:.2f} meters")
```

### 5. Use It!

```python
# Camera pixel to map coordinate
map_coord = homography.camera_to_map((960, 540))
print(f"Map: ({map_coord[0]:.2f}, {map_coord[1]:.2f}) meters")

# Map coordinate to camera pixel
camera_pixel = homography.map_to_camera((251500, -360500))
print(f"Camera: ({camera_pixel[0]:.1f}, {camera_pixel[1]:.1f}) pixels")
```

## Common Patterns

### Batch Processing

```python
# Multiple camera pixels at once
camera_pixels = [(800, 580), (1082, 390), (408, 776)]
map_coords = homography.camera_to_map_batch(camera_pixels)

# Multiple map coordinates at once
map_coords = [(251200, -360700), (251400, -360680)]
camera_pixels = homography.map_to_camera_batch(map_coords)
```

### Error Handling

```python
try:
    result = homography.compute_from_gcps(gcps, registry)
except ValueError as e:
    print(f"Invalid data: {e}")
except RuntimeError as e:
    print(f"Computation failed: {e}")
```

### Quality Checking

```python
result = homography.compute_from_gcps(gcps, registry)

# Check if good enough
if result.inlier_ratio < 0.7:
    print("Warning: Low inlier ratio")
if result.mean_reproj_error > 20.0:
    print("Warning: High reprojection error")
```

## Test Your Setup

```bash
# Run tests
python3 -m pytest tests/test_homography_map_points*.py -v

# Run demo
python3 -m examples.demo_map_point_homography
```

Expected output:
```
✓ 37 tests passing
✓ Demo shows successful homography computation
```

## Troubleshooting

### "Need at least 4 GCPs"
- Add more GCPs to your dataset
- Minimum: 4, Recommended: 10-20

### "Map point not found in registry"
- Check that all `map_point_id` values exist in registry
- Verify you loaded the correct registry file

### "Inlier ratio too low"
- Check GCP quality (are pixels/map coordinates accurate?)
- Adjust RANSAC threshold: `compute_from_gcps(gcps, registry, ransac_threshold=100.0)`
- Remove obvious outliers from GCPs

### High reprojection errors
- Normal range: 5-20 meters for real-world data
- If >50 meters: Check coordinate system consistency (are map points in UTM?)
- Verify camera pixels are in correct image coordinate system

## Data Format

### GCP Format
```json
{
  "pixel_x": 798.0,      // Camera pixel X (0-image_width)
  "pixel_y": 578.4,      // Camera pixel Y (0-image_height)
  "map_point_id": "A7"   // Reference to map point in registry
}
```

### Map Point Format
```json
{
  "id": "A7",
  "pixel_x": 251207.03,     // Actually UTM easting in meters
  "pixel_y": -360705.00,    // Actually UTM northing in meters
  "map_id": "map_valte"
}
```

Note: Despite being named `pixel_x`/`pixel_y`, map points store UTM coordinates.

## Performance Tips

1. **Use batch operations** for multiple points (10-100x faster)
2. **Compute once**, project many times
3. **Cache results** if using same homography repeatedly
4. **Tune RANSAC** threshold based on your data quality

## Next Steps

- Read full documentation: `docs/MAP_POINT_HOMOGRAPHY.md`
- Explore examples: `examples/demo_map_point_homography.py`
- Check tests for more usage patterns: `tests/test_homography_map_points*.py`

## Quick Reference

| Operation | Method | Input | Output |
|-----------|--------|-------|--------|
| Compute homography | `compute_from_gcps(gcps, registry)` | GCP list + registry | `HomographyResult` |
| Camera → Map | `camera_to_map((x, y))` | Camera pixel tuple | Map coord tuple |
| Map → Camera | `map_to_camera((e, n))` | Map coord tuple | Camera pixel tuple |
| Batch Camera → Map | `camera_to_map_batch(pixels)` | List of pixel tuples | List of coord tuples |
| Batch Map → Camera | `map_to_camera_batch(coords)` | List of coord tuples | List of pixel tuples |
| Check validity | `is_valid()` | - | Boolean |
| Get quality | `get_result()` | - | `HomographyResult` or None |
| Get matrix | `get_homography_matrix()` | - | 3x3 numpy array |
| Get inverse | `get_inverse_matrix()` | - | 3x3 numpy array |

## Coordinate Systems

```
Camera Pixels              Map Coordinates (UTM)
┌─────────────┐
│ (0,0)       │            Northing (Y)
│      (u,v)  │            ↑
│             │            │
│         (W,H)│           └──→ Easting (X)
└─────────────┘
Origin: Top-left          Origin: UTM zone reference
Units: Pixels             Units: Meters
Range: 0-1920, 0-1080     Range: 250000-252000, -361000--359000
```

## Support

- Documentation: `docs/MAP_POINT_HOMOGRAPHY.md`
- Tests: `tests/test_homography_map_points*.py`
- Example: `examples/demo_map_point_homography.py`
