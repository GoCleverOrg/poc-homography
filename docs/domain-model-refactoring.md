# Domain Model Refactoring Plan

## Executive Summary

This document describes the domain model for a PTZ camera homography system that projects camera pixels onto georeferenced maps. The system needs to compute where a camera is looking on a map, considering camera intrinsics (lens/sensor), extrinsics (position/orientation), and the map's coordinate system.

---

## 1. Domain Overview

### 1.1 Problem Domain

We have PTZ (Pan-Tilt-Zoom) cameras installed at fixed positions overlooking areas represented by georeferenced maps. The system must:

1. **Project camera pixels → map coordinates**: Given a pixel in a camera image, determine where it falls on the map
2. **Project map coordinates → camera pixels**: Given a point on the map, determine where it appears in the camera image
3. **Handle dynamic PTZ state**: As the camera pans/tilts/zooms, the projection changes
4. **Support calibration**: Installation parameters need to be refined through calibration

### 1.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Intrinsics** | Internal camera properties: focal length, sensor size, principal point, distortion |
| **Extrinsics** | Camera pose in world: position (x, y, height) and orientation (yaw, pitch, roll) |
| **Homography** | 3×3 matrix that maps points between two planes (camera image ↔ map) |
| **GeoTransform** | 6-parameter affine transform for pixel ↔ geographic coordinate conversion |

---

## 2. Current State Analysis

### 2.1 Existing Domain Classes

Located in `poc_homography/domain/`:

| Class | Location | Type | Status |
|-------|----------|------|--------|
| `PixelPoint` | `domain/vo/pixel_point.py` | VO | ✅ Done |
| `MapPoint` | `domain/vo/map_point.py` | VO | ✅ Done |
| `CameraIntrinsics` | `domain/vo/camera_intrinsics.py` | VO | ✅ Done |
| `PTZState` | `domain/vo/ptz_state.py` | VO | ✅ Renamed from CameraState |
| `Annotation` | `domain/entities/annotation.py` | Entity | ✅ Moved to entities/ |
| `GroundControlPoint` | `domain/entities/ground_control_point.py` | Entity | ✅ Moved to entities/ |

### 2.2 Existing Infrastructure (Dict-Based)

**`camera_config.py`** contains camera configurations as dictionaries:

```python
{
    "ip": "10.207.99.178",
    "name": "Valte",
    "model": "DS-2DF8425IX-AELW(T5)",

    # GPS position (NOT map pixel position)
    "lat": "39°38'25.72\"N",
    "lon": "0°13'48.63\"W",

    # Installation parameters
    "height_m": 4.71,
    "pan_offset_deg": 51.7,      # Base yaw
    "tilt_offset_deg": -0.25,    # Base pitch
    # roll: MISSING

    # Intrinsics
    "sensor_width_mm": 6.78,
    "base_focal_length_mm": 5.9,
    "k1": -0.341052,  # Distortion
    "k2": 0.787571,
    "p1": 0.0,
    "p2": 0.0,
    "calibration_table": None,

    # Map reference (embedded, should be separate Map entity)
    "geotiff_params": {
        "geotransform": [737575.05, 0.15, 0, 4391595.45, 0, -0.15],
        "utm_crs": "EPSG:25830",
    },
}
```

**`geotiff_utils.py`** has coordinate transformation:

```python
def apply_geotransform(px, py, gt) -> tuple[float, float]:
    """Convert pixel coords to geographic coords using 6-param affine."""
```

### 2.3 Gaps in Current Model

| Gap | Description |
|-----|-------------|
| No `Map` Entity | GeoTiff params embedded in camera config |
| No `Camera` Entity | Camera is a dict, not a proper entity |
| No `CameraInstallation` VO | Installation params scattered in dict |
| Position is GPS, not MapPoint | Camera position should be map pixel coords |
| No roll parameter | Installation roll angle missing |
| No tilt sign convention | Hardcoded, should be per-camera config |
| No `CameraOrientation` VO | Final orientation computation not modeled |
| No `OrientationService` | Rotation composition logic not abstracted |

---

## 3. Target Domain Model

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTITIES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────────────────────────────┐  │
│  │       Map        │         │                 Camera                    │  │
│  ├──────────────────┤         ├──────────────────────────────────────────┤  │
│  │ id: str          │◄────────│ id: str                                  │  │
│  │ photo_path: Path │         │ installation: CameraInstallation         │  │
│  │ geotiff: GeoTiff │         │ intrinsics: CameraIntrinsics             │  │
│  └──────────────────┘         │ ptz_state: PTZState (mutable)            │  │
│                               └──────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            VALUE OBJECTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   PixelPoint    │  │    MapPoint     │  │      CameraIntrinsics       │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────────────────┤  │
│  │ x: PixelsFloat  │  │ map_id: str     │  │ sensor_width: mm            │  │
│  │ y: PixelsFloat  │  │ pixel: Pixel    │  │ base_focal_length: mm       │  │
│  │ ─────────────── │  │     Point       │  │ image_width/height: px      │  │
│  │ pixels_x: int   │  └─────────────────┘  │ focal_length: mm            │  │
│  │ pixels_y: int   │                       │ ─────────────────────────── │  │
│  └─────────────────┘                       │ focal_length_px (virtual)   │  │
│                                            │ cx, cy (virtual)            │  │
│  ┌─────────────────┐  ┌─────────────────┐  │ K matrix (virtual)          │  │
│  │    PTZState     │  │    GeoTiff      │  └─────────────────────────────┘  │
│  ├─────────────────┤  ├─────────────────┤                                   │
│  │ pan: Degrees    │  │ geotransform:   │  ┌─────────────────────────────┐  │
│  │ tilt: Degrees   │  │   float[6]      │  │    CameraInstallation       │  │
│  │ zoom: Unitless  │  │ crs: str        │  ├─────────────────────────────┤  │
│  └─────────────────┘  │ ─────────────── │  │ map_id: str                 │  │
│                       │ pixel_width_m   │  │ position: MapPoint          │  │
│  ┌─────────────────┐  │ pixel_height_m  │  │ height: Meters              │  │
│  │ BaseOrientation │  │ is_north_up     │  │ base_orientation:           │  │
│  ├─────────────────┤  └─────────────────┘  │   BaseOrientation           │  │
│  │ yaw: Degrees    │                       │ tilt_convention: enum       │  │
│  │ pitch: Degrees  │                       │ ─────────────────────────── │  │
│  │ roll: Degrees   │                       │ distortion (virtual?)       │  │
│  └─────────────────┘                       └─────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              SERVICES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      OrientationService                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ compute_orientation(installation, ptz_state) -> FinalOrientation    │    │
│  │                                                                      │    │
│  │ Strategies:                                                          │    │
│  │   - AdditiveStrategy (simple angle addition, small angles)          │    │
│  │   - RotationMatrixStrategy (proper SO(3) composition)               │    │
│  │   - QuaternionStrategy (for interpolation use cases)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      HomographyService                               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ compute_homography(intrinsics, orientation, height) -> Homography   │    │
│  │ project_to_map(pixel, homography) -> MapPoint                       │    │
│  │ project_to_camera(map_point, homography) -> PixelPoint              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CoordinateTransformService                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ pixel_to_utm(pixel, geotiff) -> UTMCoord                            │    │
│  │ utm_to_pixel(utm, geotiff) -> PixelPoint                            │    │
│  │ pixel_to_gps(pixel, geotiff) -> GPSCoord                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Value Objects Detail

#### 3.2.1 `PixelPoint` (Done)

```python
@dataclass(frozen=True)
class PixelPoint:
    _x: float
    _y: float

    # Virtual properties
    x -> PixelsFloat
    y -> PixelsFloat
    pixels_x -> Pixels (int)
    pixels_y -> Pixels (int)
```

#### 3.2.2 `MapPoint` (Done)

```python
@dataclass(frozen=True)
class MapPoint:
    map_id: str
    pixel_point: PixelPoint
```

#### 3.2.3 `CameraIntrinsics` (Done)

```python
@dataclass(frozen=True)
class CameraIntrinsics:
    # Core fields
    sensor_width: Millimeters
    base_focal_length: Millimeters
    image_width: Pixels
    image_height: Pixels
    focal_length: Millimeters  # zoom-adjusted

    # Virtual properties
    focal_length_px -> PixelsFloat
    cx -> PixelsFloat
    cy -> PixelsFloat
    K -> NDArray[float64]  # 3x3 intrinsic matrix
```

#### 3.2.4 `PTZState` (Rename from CameraState)

```python
@dataclass(frozen=True)
class PTZState:
    """Dynamic state reported by PTZ hardware."""
    pan: Degrees    # Relative to camera home position
    tilt: Degrees   # Relative to camera home position
    zoom: Unitless  # Affects intrinsics, not extrinsics
```

#### 3.2.5 `GeoTiff` (New)

```python
@dataclass(frozen=True)
class GeoTiff:
    """GeoTIFF metadata for pixel ↔ geographic coordinate transforms."""
    geotransform: tuple[float, float, float, float, float, float]
    crs: str  # e.g., "EPSG:25830"

    # Virtual properties
    origin_easting -> float   # GT[0]
    origin_northing -> float  # GT[3]
    pixel_width -> float      # GT[1] (meters/pixel)
    pixel_height -> float     # GT[5] (negative for north-up)
    row_rotation -> float     # GT[2]
    col_rotation -> float     # GT[4]
    is_north_up -> bool       # True if GT[2]==0 and GT[4]==0
```

#### 3.2.6 `BaseOrientation` (New)

```python
@dataclass(frozen=True)
class BaseOrientation:
    """Camera orientation at installation (when PTZ reports 0,0)."""
    yaw: Degrees    # Azimuth from map-north (or defined reference)
    pitch: Degrees  # Elevation angle
    roll: Degrees   # Rotation around optical axis
```

#### 3.2.7 `TiltConvention` (New)

```python
class TiltConvention(Enum):
    """Convention for tilt angle sign."""
    POSITIVE_UP = "positive_up"      # Positive tilt = camera looks up
    POSITIVE_DOWN = "positive_down"  # Positive tilt = camera looks down (Hikvision)
```

#### 3.2.8 `CameraInstallation` (New)

```python
@dataclass(frozen=True)
class CameraInstallation:
    """Fixed installation parameters (don't change after calibration)."""
    map_id: str                        # Which map this camera is on
    position: MapPoint                  # Position on map (pixel coords)
    height: Meters                      # Height above ground
    base_orientation: BaseOrientation   # Orientation at PTZ home
    tilt_convention: TiltConvention     # How to interpret tilt sign

    # Distortion coefficients (could be separate VO)
    k1: Unitless = Unitless(0.0)
    k2: Unitless = Unitless(0.0)
    p1: Unitless = Unitless(0.0)
    p2: Unitless = Unitless(0.0)
```

#### 3.2.9 `Photo` (New)

```python
@dataclass(frozen=True)
class Photo:
    """An image file with its dimensions (cached at load time)."""
    path: Path
    width: Pixels
    height: Pixels
```

#### 3.2.10 `FinalOrientation` (New - Service Result)

```python
@dataclass(frozen=True)
class FinalOrientation:
    """Computed camera orientation in world coordinates."""
    yaw: Degrees
    pitch: Degrees
    roll: Degrees

    # Virtual properties
    rotation_matrix -> NDArray[float64]  # 3x3 rotation matrix R
```

### 3.3 Entities Detail

#### 3.3.1 `Map` (New)

```python
@dataclass
class Map:
    """A georeferenced map image."""
    id: str                  # Unique identifier (e.g., "valte_site")
    photo: Photo             # Map image with cached dimensions
    geotiff: GeoTiff         # Coordinate transformation metadata
```

#### 3.3.2 `Camera` (New)

```python
@dataclass
class Camera:
    """A PTZ camera with installation and current state."""
    id: str                              # Unique identifier (e.g., "valte")
    name: str                            # Human-readable name
    installation: CameraInstallation     # Fixed after calibration
    intrinsics: CameraIntrinsics         # Sensor specs (fixed)
    ptz_state: PTZState                  # Current PTZ state (mutable)

    # Optional metadata
    model: str | None = None
    ip_address: str | None = None

    # Calibration table for zoom-dependent intrinsics
    calibration_table: dict[float, dict] | None = None
```

### 3.4 Services Detail

#### 3.4.1 `OrientationService`

Computes final camera orientation from installation + PTZ state.

```python
class OrientationStrategy(Protocol):
    def compute(
        self,
        base: BaseOrientation,
        ptz: PTZState,
        tilt_convention: TiltConvention,
    ) -> FinalOrientation: ...

class AdditiveOrientationStrategy:
    """Simple angle addition (valid for small angles, no roll)."""

class RotationMatrixStrategy:
    """Proper SO(3) rotation composition."""

class OrientationService:
    def __init__(self, strategy: OrientationStrategy): ...

    def compute_orientation(
        self,
        installation: CameraInstallation,
        ptz_state: PTZState,
    ) -> FinalOrientation: ...
```

### 3.5 Repositories

Repository interfaces define data access abstractions, keeping the domain
layer independent of infrastructure concerns (file I/O, databases, etc.).

**Naming convention**: All repository interfaces use the suffix `Repository`
(e.g., `MapRepository`, `CameraRepository`).

#### 3.5.1 `MapRepository`

```python
class MapRepository(Protocol):
    """Repository interface for Map entities."""

    def get(self, map_id: str) -> Map | None:
        """Retrieve a map by its ID."""
        ...

    def get_all(self) -> list[Map]:
        """Retrieve all available maps."""
        ...

    def exists(self, map_id: str) -> bool:
        """Check if a map exists."""
        ...
```

#### 3.5.2 `GroundControlPointRepository`

```python
class GroundControlPointRepository(Protocol):
    """Repository interface for GroundControlPoint entities."""

    def get(self, gcp_id: str) -> GroundControlPoint | None:
        """Retrieve a GCP by its ID (format: 'map_id/name')."""
        ...

    def get_by_map(self, map_id: str) -> list[GroundControlPoint]:
        """Retrieve all GCPs for a specific map."""
        ...

    def save(self, gcp: GroundControlPoint) -> None:
        """Save a GCP (create or update)."""
        ...

    def delete(self, gcp_id: str) -> bool:
        """Delete a GCP by its ID."""
        ...

    def exists(self, gcp_id: str) -> bool:
        """Check if a GCP exists."""
        ...
```

---

## 4. Coordinate System Conventions

### 4.1 Map Coordinate System

| Property | Convention |
|----------|------------|
| Origin | Top-left corner of map image |
| X-axis | Positive rightward (East for north-up maps) |
| Y-axis | Positive downward (South for north-up maps) |
| Units | Pixels |

### 4.2 World/Geographic Coordinate System

| Property | Convention |
|----------|------------|
| Origin | Defined by GeoTiff (typically UTM zone origin) |
| X-axis (Easting) | Positive eastward |
| Y-axis (Northing) | Positive northward |
| Z-axis | Positive upward (height above ground) |
| Units | Meters |

### 4.3 Camera Orientation Convention

| Angle | Definition | Positive Direction |
|-------|------------|-------------------|
| **Yaw** | Rotation around vertical (Z) axis | Clockwise from North (0° = North, 90° = East) |
| **Pitch** | Rotation around lateral (X) axis | Configurable via `TiltConvention` |
| **Roll** | Rotation around optical (Y) axis | Clockwise when looking along optical axis |

### 4.4 PTZ to World Transform

```
Final_Yaw   = Base_Yaw + PTZ_Pan
Final_Pitch = Base_Pitch + PTZ_Tilt * sign(tilt_convention)
Final_Roll  = Base_Roll  (PTZ doesn't change roll)
```

---

## 5. Migration Plan

### Phase 1: Core Value Objects

**Status**: ✅ Complete

- [x] `PixelPoint` - done (with typed properties: x, y, pixels_x, pixels_y)
- [x] `MapPoint` - done (with map_id and pixel_point)
- [x] `CameraIntrinsics` - done (with virtual properties: focal_length_px, cx, cy, K matrix)
- [x] `PTZState` - done (renamed from CameraState)
- [x] `GeoTiff` - done (with pixel_to_geo, geo_to_pixel methods)
- [x] `BaseOrientation` - done (yaw, pitch, roll)
- [x] `TiltConvention` - done (enum with sign property)

### Phase 2: Composite Value Objects

**Status**: ✅ Complete

- [x] `CameraInstallation` - done (compose position + orientation + distortion)
- [x] `FinalOrientation` - done (with rotation_matrix property)

### Phase 3: Entities

**Status**: ✅ Complete

- [x] `Annotation` - moved to entities/ directory
- [x] `GroundControlPoint` - moved to entities/ directory
- [x] `Map` - done (id, name, photo_path, geotiff)
- [x] `Camera` - done (installation + intrinsics + ptz_state)

### Phase 4: Services

- [ ] `OrientationService` with strategy pattern
- [ ] `CoordinateTransformService` - wrap geotiff_utils
- [ ] Refactor existing `HomographyService` to use new VOs

### Phase 5: Migration of Existing Code

- [ ] Update `camera_config.py` to use new entities
- [ ] Update `camera_geometry.py` to use new VOs
- [ ] Update `homography/` modules
- [ ] Update tests

### Phase 6: Configuration Migration

- [x] Create `data/maps/` for Map YAML files
- [ ] Create `data/cameras/` for Camera YAML files
- [x] Create `data/gcps/` for GCP YAML files
- [x] Add repository interfaces (MapRepository, GroundControlPointRepository)
- [x] Implement YAML repositories (YamlMapRepository, YamlGroundControlPointRepository)

---

## 6. File Structure

```
poc_homography/
├── domain/
│   ├── __init__.py             ✅
│   ├── vo/
│   │   ├── __init__.py         ✅
│   │   ├── pixel_point.py      ✅
│   │   ├── map_point.py        ✅
│   │   ├── camera_intrinsics.py ✅
│   │   ├── ptz_state.py        ✅ (renamed from camera_state.py)
│   │   ├── geotiff.py          ✅
│   │   ├── base_orientation.py ✅
│   │   ├── final_orientation.py ✅
│   │   ├── camera_installation.py ✅
│   │   └── photo.py            ✅
│   ├── entities/
│   │   ├── __init__.py         ✅
│   │   ├── map.py              ✅
│   │   ├── camera.py           ✅
│   │   ├── annotation.py       ✅ (moved from domain/)
│   │   └── ground_control_point.py ✅ (moved from domain/)
│   ├── enums/
│   │   ├── __init__.py         ✅
│   │   └── tilt_convention.py  ✅
│   └── repositories/
│       ├── __init__.py         ✅
│       ├── map_repository.py   ✅
│       └── ground_control_point_repository.py ✅
├── infrastructure/
│   ├── __init__.py             ✅
│   └── repositories/
│       ├── __init__.py         ✅
│       ├── yaml_map_repository.py ✅
│       └── yaml_ground_control_point_repository.py ✅
├── services/
│   ├── __init__.py
│   ├── orientation_service.py  (new)
│   ├── coordinate_transform_service.py (new, wrap geotiff_utils)
│   └── homography_service.py   (refactor existing)
├── data/
│   ├── maps/
│   │   └── valte.yaml          ✅
│   ├── gcps/
│   │   └── valte.yaml          ✅
│   └── cameras/
│       └── (pending)
└── ...
```

---

## 7. Open Questions

1. **Distortion coefficients**: Should they be part of `CameraInstallation` or separate `LensDistortion` VO?

2. **Calibration table**: How to model zoom-dependent intrinsics? Separate VO or part of `Camera` entity?

3. **GPS coordinates**: Do we still need GPS lat/lon for cameras, or is MapPoint sufficient?

4. ~~**Map loading**: Should `Map` entity load image dimensions from file, or require them as input?~~ **Resolved**: `Map.get_dimensions()` method computes from photo file on demand.

5. **Mutability**: `Camera.ptz_state` needs to be mutable. Use `@dataclass` without `frozen=True`, or separate mutable state?

---

## 8. References

- GDAL GeoTransform: https://gdal.org/tutorials/geotransforms_tut.html
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Rotation Matrix Conventions: https://en.wikipedia.org/wiki/Rotation_matrix
