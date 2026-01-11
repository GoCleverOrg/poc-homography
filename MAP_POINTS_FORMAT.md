# Map Points Format

This document describes the `map_points.json` format for representing reference points using pixel coordinates only, without geographic (lat/lon) information.

## Overview

The Map Points format is designed to gradually phase out KML and geographic coordinates from the codebase. Instead of storing points with lat/lon values that require georeferencing transformations, we store points directly in pixel coordinates relative to a specific map image.

## Format Structure

```json
{
  "map_id": "map_valte",
  "points": [
    {
      "id": "Z1",
      "pixel_x": 251246.7732343846,
      "pixel_y": -360159.9084134213
    },
    {
      "id": "Z2",
      "pixel_x": 251255.1234567890,
      "pixel_y": -360170.9876543210
    }
  ]
}
```

### Fields

- **map_id** (string): Identifier for the map these points belong to (e.g., "map_valte" for Valencia Terminal)
- **points** (array): List of map point objects
  - **id** (string): Unique identifier for the point (e.g., "Z1", "P5", "A3") - managed by the registry as dictionary keys
  - **pixel_x** (float): X coordinate in pixels (column)
  - **pixel_y** (float): Y coordinate in pixels (row)

**Note:** The `id` field appears in the JSON format as part of each point object, but it is managed by the MapPointRegistry as dictionary keys. Individual MapPoint instances do not store their own IDs.

## Python API

### MapPoint Data Structure

MapPoint is a frozen dataclass that contains only the pixel coordinates. The ID and map association are managed externally by MapPointRegistry.

```python
from poc_homography.map_points import MapPoint

# MapPoint only stores pixel coordinates
point = MapPoint(pixel_x=100.5, pixel_y=200.3)

# Access coordinates
print(f"X: {point.pixel_x}, Y: {point.pixel_y}")

# Access as PixelPoint
pixel = point.pixel  # Returns PixelPoint(100.5, 200.3)
```

### Loading Map Points

```python
from poc_homography.map_points import MapPointRegistry

# Load from JSON file
registry = MapPointRegistry.load("map_points.json")

# Access the map ID
print(f"Map: {registry.map_id}")

# Access points by ID (stored as dictionary keys)
point = registry.points["Z1"]
print(f"Point Z1 at ({point.pixel_x}, {point.pixel_y})")

# Iterate over all points
for point_id, point in registry.points.items():
    print(f"{point_id}: ({point.pixel_x}, {point.pixel_y})")
```

### Creating Map Points

```python
from poc_homography.map_points import MapPoint, MapPointRegistry

# Create individual points (without IDs)
point1 = MapPoint(pixel_x=100.5, pixel_y=200.3)
point2 = MapPoint(pixel_x=150.8, pixel_y=210.1)

# Create registry (IDs are dictionary keys)
registry = MapPointRegistry(
    map_id="map_valte",
    points={"P1": point1, "P2": point2}
)

# Save to JSON
registry.save("my_points.json")
```

## Calibration Data Model

### New Annotation and CaptureContext

The calibration system now uses structured dataclasses for capturing ground control point (GCP) observations:

**Annotation**: Links a GCP to its observed pixel location in a camera image.
```python
from poc_homography.calibration.annotation import Annotation
from poc_homography.pixel_point import PixelPoint

annotation = Annotation(
    gcp_id="Z1",
    pixel=PixelPoint(x=960.0, y=540.0)
)
```

**CaptureContext**: Camera state when a calibration frame was captured.
```python
from poc_homography.calibration.annotation import CaptureContext

context = CaptureContext(
    camera="Valte",
    pan_raw=0.0,
    tilt_deg=30.0,
    zoom=1.0
)
```

### New YAML Format

The calibration tools now use a new YAML format that separates capture context from annotations:

```yaml
capture:
  context:
    camera: Valte
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
  annotations:
    - gcp_id: Z1
      pixel:
        x: 960.0
        y: 540.0
    - gcp_id: Z2
      pixel:
        x: 1100.0
        y: 620.0
```

**Key features:**
- `capture.context` contains camera state (camera name, PTZ values)
- `capture.annotations` contains list of GCP observations
- Each annotation uses `gcp_id` (references map_points.json IDs)
- Pixel coordinates use `x` and `y` (not `pixel_u` and `pixel_v`)

### Legacy YAML Format (Deprecated)

The old format is still supported for backward compatibility, but emits deprecation warnings:

```yaml
gcps:
  - map_point_id: Z1
    pixel_u: 960
    pixel_v: 540
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
```

**Deprecated features:**
- `gcps` top-level key (use `capture` instead)
- `map_point_id` field (use `gcp_id` instead)
- `pixel_u` and `pixel_v` fields (use `pixel.x` and `pixel.y` instead)
- PTZ values repeated for each GCP (use shared `capture.context` instead)

## Converting from KML

The `tools/convert_kml_to_map_points.py` script converts existing KML files to the map_points format:

```bash
python tools/convert_kml_to_map_points.py \
  --kml Cartografia_valencia_recreated.kml \
  --output map_points.json \
  --map-id map_valte \
  --crs EPSG:25830 \
  --geotransform 725140.0 0.05 0.0 4373490.0 0.0 -0.05
```

### Conversion Process

1. Parse KML file to extract lat/lon coordinates
2. Use georeferencing configuration (CRS + geotransform) to convert lat/lon to pixel coordinates
3. Extract only the pixel coordinates and point IDs
4. Write to JSON in the map_points format

## Migration Strategy

The goal is to gradually remove KML and geographic coordinate dependencies:

### Current State (KML-based)
- Points stored as KML with lat/lon coordinates
- Requires GDAL, pyproj, and georeferencing configuration
- Complex coordinate transformations needed

### Target State (Map Points)
- Points stored as JSON with pixel coordinates
- No geographic dependencies
- Direct pixel-based operations

### Migration Steps

1. ✅ Create map_points data structures (MapPoint, MapPointRegistry)
2. ✅ Refactor MapPoint to remove id and map_id fields (managed by registry)
3. ✅ Create Annotation and CaptureContext dataclasses for calibration
4. ✅ Update calibration tools to use new YAML format
5. ✅ Convert existing KML to map_points.json
6. ✅ Update CLI calibration tools to use Map Points
7. 🔄 Migrate workflows to use map_points by default
8. 🔄 Remove KML dependencies where no longer needed

## CLI Tools Using Map Points

The following calibration tools in `tools/` use the Map Points system:

### calibrate_projection.py

Calibrates pan_offset and height parameters using a single reference point.

```bash
python tools/calibrate_projection.py Valte Z1 960 540 0.0 30.0 1.0 --map-points map_points.json
```

Arguments:
- `Valte` - Camera name
- `Z1` - Map Point ID (from map_points.json)
- `960 540` - Image pixel coordinates where the point appears
- `0.0 30.0 1.0` - PTZ values (pan_raw, tilt, zoom)
- `--map-points` - Path to map_points.json (default: `map_points.json`)

### comprehensive_calibration.py

Runs scipy optimization to calibrate all camera parameters using multiple GCPs.

**New YAML format** (recommended):
```yaml
capture:
  context:
    camera: Valte
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
  annotations:
    - gcp_id: Z1
      pixel:
        x: 960.0
        y: 540.0
    - gcp_id: Z2
      pixel:
        x: 1100.0
        y: 620.0
```

**Legacy YAML format** (deprecated, emits warnings):
```yaml
gcps:
  - map_point_id: Z1
    pixel_u: 960
    pixel_v: 540
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
```

### validate_camera_model.py

Validates camera model accuracy by comparing projected vs actual pixel positions.

Uses the same new YAML format as comprehensive_calibration.py:

```yaml
capture:
  context:
    camera: Valte
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
  annotations:
    - gcp_id: Z1
      pixel:
        x: 960.0
        y: 540.0
```

### interactive_calibration.py

Interactive GUI for manual calibration. User clicks on known features in camera image and enters Map Point IDs.

## Point Categories

The original KML had categories (zebra, arrow, parking, other) which were used for visualization styling. In the map_points format, categories are not stored - points are identified only by their ID prefix:

- **Z** prefix: Zebra crossings (48 points)
- **P** prefix: Parking spaces (18 points)
- **A** prefix: Arrows (7 points)
- **X** prefix: Other features (23 points)

If category information is needed, it can be derived from the ID prefix or stored in a separate configuration file.

## File Location

The canonical map_points file for Valencia Terminal is:
- `/Users/nuno.monteiro/Dev/SmartTerminal/poc-homography/map_points.json`

## Advantages over KML

1. **Simpler format**: JSON instead of XML
2. **No geographic dependencies**: No need for CRS, geotransforms, or projection libraries
3. **Direct pixel access**: No coordinate transformations needed
4. **Easier to edit**: Standard JSON tools vs specialized GIS software
5. **Smaller file size**: Less metadata and structure overhead
6. **Language agnostic**: Any language can parse JSON easily

## Data Model Design Principles

### Separation of Concerns

- **MapPoint**: Pure data structure containing only pixel coordinates
- **MapPointRegistry**: Manages IDs and map association as dictionary keys
- **Annotation**: Links GCP IDs to observed pixel locations
- **CaptureContext**: Camera state information separate from observations

### Frozen Dataclasses

All data structures use `@dataclass(frozen=True)` for immutability:
- Prevents accidental modifications
- Enables safe caching with `@cache` decorators
- Clearer value semantics

### JSON Serialization

All dataclasses provide `to_dict()` and `from_dict()` methods for JSON serialization without external dependencies.

## See Also

- `poc_homography/map_points/` - Python implementation
- `poc_homography/calibration/annotation.py` - Annotation and CaptureContext dataclasses
- `tools/convert_kml_to_map_points.py` - Conversion tool
- `Cartografia_valencia_recreated.kml` - Original KML file (legacy)
