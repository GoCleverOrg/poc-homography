# Map Point Homography System

## Overview

The Map Point Homography system provides coordinate transformation between camera image pixels and map coordinates (UTM) using reference map points instead of GPS coordinates. This approach is particularly useful when:

- Working with local map coordinate systems
- Avoiding GPS coordinate transformations
- Using fixed reference points on a map
- Needing high precision in a local area

## Architecture

### Components

1. **MapPoint** (`poc_homography.map_points.map_point`)
   - Represents a reference point with pixel coordinates on a map
   - Actually stores UTM easting/northing in the `pixel_x`/`pixel_y` fields (naming issue)

2. **MapPointRegistry** (`poc_homography.map_points.map_point_registry`)
   - Manages collections of map points
   - Provides JSON serialization/deserialization
   - Supports loading from files

3. **MapPointHomography** (`poc_homography.homography_map_points`)
   - Computes homography transformation from GCPs
   - Provides bidirectional coordinate transformation
   - Validates quality and accuracy

### Coordinate Systems

- **Camera Pixels**: `(u, v)` in image space, origin at top-left
- **Map Coordinates**: `(easting, northing)` in UTM meters

### Data Flow

```
┌─────────────────┐
│  Map Points     │
│  (UTM coords)   │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
┌────────▼────────┐  ┌──────▼──────────┐
│  GCPs (camera   │  │  MapPoint       │
│  pixels +       │  │  Registry       │
│  map_point_id)  │  │                 │
└────────┬────────┘  └──────┬──────────┘
         │                  │
         └─────────┬────────┘
                   │
         ┌─────────▼─────────┐
         │  MapPoint         │
         │  Homography       │
         │  (cv2.find        │
         │   Homography)     │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  3x3 Homography   │
         │  Matrix (H)       │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────────────┐
         │  Coordinate Transform     │
         │  • camera_to_map()        │
         │  • map_to_camera()        │
         │  • Batch operations       │
         └───────────────────────────┘
```

## Usage

### Basic Example

```python
from poc_homography.homography_map_points import MapPointHomography
from poc_homography.map_points import MapPointRegistry

# Load map points
registry = MapPointRegistry.load("map_points.json")

# Define GCPs (camera pixels with map point references)
gcps = [
    {"pixel_x": 798.0, "pixel_y": 578.4, "map_point_id": "A7"},
    {"pixel_x": 1082.0, "pixel_y": 390.4, "map_point_id": "A6"},
    {"pixel_x": 408, "pixel_y": 776.1, "map_point_id": "X15"},
    # ... more GCPs (minimum 4 required)
]

# Compute homography
homography = MapPointHomography()
result = homography.compute_from_gcps(gcps, registry)

print(f"Computed homography with {result.num_inliers} inliers")
print(f"Mean error: {result.mean_reproj_error:.2f} meters")

# Project camera pixel to map coordinate
map_coord = homography.camera_to_map((960, 540))  # Camera center
print(f"Map coordinate: ({map_coord[0]:.2f}, {map_coord[1]:.2f}) meters")

# Project map coordinate to camera pixel
camera_pixel = homography.map_to_camera((251500.0, -360500.0))
print(f"Camera pixel: ({camera_pixel[0]:.1f}, {camera_pixel[1]:.1f})")
```

### Batch Operations

```python
# Project multiple camera pixels at once
camera_pixels = [(800, 580), (1082, 390), (408, 776)]
map_coords = homography.camera_to_map_batch(camera_pixels)

# Project multiple map coordinates at once
map_coords = [(251200, -360700), (251400, -360680)]
camera_pixels = homography.map_to_camera_batch(map_coords)
```

### Quality Metrics

```python
result = homography.compute_from_gcps(gcps, registry)

# Access quality metrics
print(f"Total GCPs: {result.num_gcps}")
print(f"Inliers: {result.num_inliers} ({result.inlier_ratio:.1%})")
print(f"Mean reprojection error: {result.mean_reproj_error:.2f} m")
print(f"Max reprojection error: {result.max_reproj_error:.2f} m")
print(f"RMSE: {result.rmse:.2f} m")
```

## Test Data

The system includes comprehensive test data:

### Files

1. **`map_points.json`**: 96 map points with UTM coordinates
   - Valencia area (UTM zone 30N)
   - Categories: zebra crossings (Z), arrows (A), parking (P), other (X)

2. **`test_data_Valte_20260109_195052.json`**: GCP data
   - 16 ground control points
   - Camera info (lat/lon, height, pan, tilt)
   - Links camera pixels to map point IDs

3. **`test_data_Valte_20260109_195052.jpg`**: Camera image (1920x1080)

### Example GCP

```json
{
  "pixel_x": 798.0078125,
  "pixel_y": 578.3828125,
  "map_point_id": "A7"
}
```

Corresponding map point:

```json
{
  "id": "A7",
  "pixel_x": 251207.02772919316,
  "pixel_y": -360705.00421358267,
  "map_id": "map_valte"
}
```

Note: `pixel_x`/`pixel_y` in MapPoint actually contain UTM easting/northing in meters.

## Test Coverage

### Test Suite 1: `test_homography_map_points.py`

Low-level tests of homography computation using OpenCV directly:

- Map point registry loading
- GCP correspondence extraction
- Homography matrix computation
- Forward projection (camera → map)
- Inverse projection (map → camera)
- Round-trip validation
- Reprojection error metrics

**18 tests** - All passing

### Test Suite 2: `test_homography_map_points_integration.py`

High-level tests of `MapPointHomography` class:

- Initialization and validation
- Error handling (insufficient GCPs, missing map points)
- Quality metrics validation
- Forward/inverse projection
- Batch operations
- Matrix retrieval
- Round-trip consistency

**19 tests** - All passing

### Test Results

```
Homography Quality Metrics (from real data):
- Inlier ratio: 93.8% (15/16 GCPs)
- Mean reprojection error: 6.38 meters
- Max reprojection error: 19.45 meters
- RMSE: 8.67 meters
- Round-trip accuracy: 0.03 pixels (mean)
```

## Performance Characteristics

### Accuracy

- **Forward projection**: Mean error ~6-12 meters in map space
- **Inverse projection**: Mean error ~10-20 pixels in image space
- **Round-trip**: Mean error <0.1 pixels (excellent consistency)

### Requirements

- **Minimum GCPs**: 4 (more is better, 10-20 recommended)
- **Inlier ratio**: >50% (>70% recommended)
- **RANSAC threshold**: 50 meters (adjustable based on data quality)

### Limitations

1. **Planar assumption**: Assumes ground plane (Z=0)
2. **Scale differences**: Large coordinate scale differences between pixels (0-2000) and UTM meters (250000-252000) can affect precision
3. **Outliers**: RANSAC helps but quality depends on GCP accuracy

## Running the Demo

```bash
# From project root
python3 -m examples.demo_map_point_homography
```

Expected output:
- Loads 96 map points
- Computes homography from 16 GCPs
- Shows forward/inverse projections
- Validates round-trip accuracy
- Reports quality metrics

## API Reference

### MapPointHomography

#### Methods

**`compute_from_gcps(gcps, map_registry, ransac_threshold=50.0, min_inlier_ratio=0.5)`**
- Computes homography from ground control points
- Returns `HomographyResult` with quality metrics
- Raises `ValueError` if insufficient quality

**`camera_to_map(camera_pixel)`**
- Transforms single camera pixel `(x, y)` to map coordinate `(easting, northing)`
- Returns `tuple[float, float]`

**`map_to_camera(map_coord)`**
- Transforms single map coordinate `(easting, northing)` to camera pixel `(x, y)`
- Returns `tuple[float, float]`

**`camera_to_map_batch(camera_pixels)`**
- Transforms list of camera pixels to map coordinates
- Returns `list[tuple[float, float]]`

**`map_to_camera_batch(map_coords)`**
- Transforms list of map coordinates to camera pixels
- Returns `list[tuple[float, float]]`

**`is_valid()`**
- Returns `bool` indicating if homography is computed and valid

**`get_result()`**
- Returns `Optional[HomographyResult]` with quality metrics

**`get_homography_matrix()`**
- Returns `np.ndarray` (3x3) forward transformation matrix

**`get_inverse_matrix()`**
- Returns `np.ndarray` (3x3) inverse transformation matrix

### HomographyResult

#### Attributes

- `homography_matrix`: 3x3 forward transformation matrix
- `inverse_matrix`: 3x3 inverse transformation matrix
- `num_gcps`: Total number of GCPs used
- `num_inliers`: Number of inlier GCPs after RANSAC
- `inlier_ratio`: Ratio of inliers (0.0 to 1.0)
- `mean_reproj_error`: Mean reprojection error in meters
- `max_reproj_error`: Maximum reprojection error in meters
- `rmse`: Root mean square error in meters

## TDD Approach

This system was developed following Test-Driven Development:

1. **RED**: Created comprehensive test suite that initially failed
2. **GREEN**: Implemented `MapPointHomography` class to make tests pass
3. **REFACTOR**: Optimized and documented code

### Benefits

- **High confidence**: 37 passing tests validate correctness
- **Clear requirements**: Tests document expected behavior
- **Regression prevention**: Tests catch breaking changes
- **Living documentation**: Tests show usage examples

## Integration with Existing System

The Map Point Homography system complements the existing homography providers:

- **GPS-based homography**: Uses lat/lon coordinates (existing)
- **Map point homography**: Uses UTM/map coordinates (new)

Both approaches can coexist. Choose based on:
- GPS-based: Multi-camera systems, large areas, geo-referencing
- Map point: Single camera, local area, fixed map reference

## Future Enhancements

1. **HomographyProvider integration**: Implement `HomographyProvider` interface
2. **Automatic GCP detection**: Find correspondences from image features
3. **Non-planar correction**: Handle terrain elevation
4. **Adaptive RANSAC**: Auto-tune threshold based on data
5. **Quality visualization**: Overlay reprojection errors on image

## References

- OpenCV `findHomography`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
- RANSAC algorithm: https://en.wikipedia.org/wiki/Random_sample_consensus
- Homography estimation: Multiple View Geometry in Computer Vision (Hartley & Zisserman)
