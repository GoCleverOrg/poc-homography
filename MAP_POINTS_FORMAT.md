# Map Points Format

This document describes the new `map_points.json` format for representing reference points using pixel coordinates only, without geographic (lat/lon) information.

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
      "pixel_y": -360159.9084134213,
      "map_id": "map_valte"
    },
    ...
  ]
}
```

### Fields

- **map_id** (string): Identifier for the map these points belong to (e.g., "map_valte" for Valencia Terminal)
- **points** (array): List of map point objects
  - **id** (string): Unique identifier for the point (e.g., "Z1", "P5", "A3")
  - **pixel_x** (float): X coordinate in pixels (column)
  - **pixel_y** (float): Y coordinate in pixels (row)
  - **map_id** (string): Reference back to the parent map identifier

## Python API

### Loading Map Points

```python
from poc_homography.map_points import MapPointRegistry

# Load from JSON file
registry = MapPointRegistry.load("map_points.json")

# Access points by ID
point = registry.points["Z1"]
print(f"Point {point.id} at ({point.pixel_x}, {point.pixel_y})")
```

### Creating Map Points

```python
from poc_homography.map_points import MapPoint, MapPointRegistry

# Create individual points
point1 = MapPoint(id="P1", pixel_x=100.5, pixel_y=200.3, map_id="map_valte")
point2 = MapPoint(id="P2", pixel_x=150.8, pixel_y=210.1, map_id="map_valte")

# Create registry
registry = MapPointRegistry(
    map_id="map_valte",
    points={"P1": point1, "P2": point2}
)

# Save to JSON
registry.save("my_points.json")
```

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
2. ✅ Convert existing KML to map_points.json
3. 🔄 Update tools to support both KML and map_points formats
4. 🔄 Migrate workflows to use map_points by default
5. 🔄 Remove KML dependencies where no longer needed

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

## See Also

- `poc_homography/map_points/` - Python implementation
- `tools/convert_kml_to_map_points.py` - Conversion tool
- `Cartografia_valencia_recreated.kml` - Original KML file (legacy)
