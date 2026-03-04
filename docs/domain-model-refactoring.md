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
| `CameraIntrinsics` | ~~`domain/vo/camera_intrinsics.py`~~ | VO | 🗑️ Deleted (vestigial, replaced by `camera/intrinsics.py`) |
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

### 3.1 Data Lifecycle Separation

Camera-related data has **three distinct lifecycles** that must be handled separately:

| Data Type | Contents | Lifecycle | Persistence | Repository |
|-----------|----------|-----------|-------------|------------|
| **Configuration** | map_id, name, spec, ip_address | Rarely changes (camera registration) | YAML | `CameraConfigRepository` |
| **Calibration** | position, height, orientation, distortion | Changes during calibration | YAML | `CameraCalibrationRepository` |
| **Hardware State** | PTZ state (pan, tilt, zoom) | Changes constantly | None (transient) | None (passed as argument) |

```
┌───────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐
│   CameraConfig    │  │   CameraCalibration     │  │    PTZState     │
├───────────────────┤  ├─────────────────────────┤  ├─────────────────┤
│ map_id            │  │ camera_id               │  │ pan_raw         │
│ name              │  │ position: PixelPoint    │  │ tilt_deg        │
│ spec: CameraSpec  │  │ height: Meters          │  │ zoom            │
│ ip_address        │  │ base_orientation        │  └─────────────────┘
├───────────────────┤  │ distortion              │   ⚡ Transient
│ id (computed)     │  └─────────────────────────┘   From hardware API
└───────────────────┘   🔧 Refined during            No persistence
 📋 Set once            calibration
 Rarely changes
```

**Key Design Decisions:**

1. **Three separate concerns** - Config, Calibration, and PTZ State are distinct
2. **Two repositories** - `CameraConfigRepository` and `CameraCalibrationRepository`
3. **PTZState passed as argument** to domain services that need it (not stored)
4. **All VOs/entities immutable** - no mutable state, no update methods
5. **Domain Services** combine all three pieces for computations
6. **CameraSnapshot** (optional) - immutable point-in-time combination for convenience

### 3.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTITIES (Immutable)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌────────────────────┐  ┌─────────────────────────┐  │
│  │       Map        │  │    CameraConfig    │  │   CameraCalibration     │  │
│  ├──────────────────┤  ├────────────────────┤  ├─────────────────────────┤  │
│  │ id: str          │◄─│ map_id: str        │◄─│ camera_id: str          │  │
│  │ photo: Photo     │  │ name: str          │  │ position: PixelPoint    │  │
│  │ geotiff: GeoTiff │  │ id (computed)      │  │ height: Meters          │  │
│  └──────────────────┘  │ spec: CameraSpec   │  │ base_orientation        │  │
│                        │ ip_address         │  │ distortion              │  │
│                        └────────────────────┘  └─────────────────────────┘  │
│                         📋 Set once             🔧 Refined during           │
│                         (registration)          calibration                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       VALUE OBJECTS (All Immutable)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PERSISTED                                   TRANSIENT                       │
│  ┌─────────────────┐  ┌──────────────┐       ┌─────────────────────────┐    │
│  │   Orientation   │  │ LensDistort. │       │       PTZState          │    │
│  ├─────────────────┤  ├──────────────┤       ├─────────────────────────┤    │
│  │ yaw: Degrees    │  │ k1, k2       │       │ pan_raw: float          │    │
│  │ pitch: Degrees  │  │ p1, p2       │       │ tilt_deg: float         │    │
│  │ roll: Degrees   │  └──────────────┘       │ zoom: float             │    │
│  │ rotation_matrix │                         └─────────────────────────┘    │
│  └─────────────────┘                          ⚡ From hardware API          │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   PixelPoint    │  │    MapPoint     │  │       CameraSnapshot        │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────────────────┤  │
│  │ x, y: float     │  │ map_id: str     │  │ config: CameraConfig        │  │
│  └─────────────────┘  │ pixel: PixelPt  │  │ calibration: CameraCalib.   │  │
│                       └─────────────────┘  │ ptz_state: PTZState         │  │
│  ┌─────────────────┐  ┌─────────────────┐  │ (point-in-time combination) │  │
│  │     Photo       │  │    GeoTiff      │  └─────────────────────────────┘  │
│  ├─────────────────┤  ├─────────────────┤                                   │
│  │ path, w, h      │  │ geotransform    │                                   │
│  └─────────────────┘  │ crs             │                                   │
│                       └─────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENUMS                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │       CameraSpec            │  │        TiltConvention               │   │
│  ├─────────────────────────────┤  ├─────────────────────────────────────┤   │
│  │ HIKVISION_DS_2DF8425IX      │  │ POSITIVE_UP                         │   │
│  │   - sensor_width            │  │ POSITIVE_DOWN                       │   │
│  │   - base_focal_length       │  └─────────────────────────────────────┘   │
│  │   - image_width/height      │                                            │
│  │   - tilt_convention         │  (No distortion - that's per-camera        │
│  │   - max_zoom                │   calibration, not per-model)              │
│  └─────────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    DOMAIN SERVICES (Stateless)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Services receive all three pieces as arguments. They do NOT fetch data.    │
│                                                                              │
│  NAMING CONVENTION:                                                          │
│    - Service class: Service<Domain> (e.g., ServiceOrientation)              │
│    - Service file: service_<domain>.py                                       │
│    - Strategy protocol: <Domain>Strategy (e.g., OrientationStrategy)        │
│    - Strategy file: <domain>/strategy.py                                     │
│    - Strategy impl: <Name>Strategy (e.g., AdditiveOrientationStrategy)      │
│    - Strategy impl file: <domain>/<name>_strategy.py                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ServiceOrientation                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ compute_orientation(base, ptz_state, tilt_convention) -> Orientation│    │
│  │                                                                      │    │
│  │ Strategies (in orientation/ folder):                                 │    │
│  │   - AdditiveOrientationStrategy (simple angle addition)             │    │
│  │   - RotationMatrixStrategy (proper SO(3) composition)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ServiceHomography (pending)                     │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ compute(config, calibration, ptz_state) -> Homography               │    │
│  │ project_to_map(pixel, homography) -> MapPoint                       │    │
│  │ project_to_camera(map_point, homography) -> PixelPoint              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  CoordinateTransformService: NOT NEEDED - GeoTiff VO has methods            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ServiceCamera (optional)                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ get_snapshot(camera_id, ptz_state) -> CameraSnapshot                │    │
│  │   Combines Config + Calibration from repos + provided PTZState      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPOSITORIES                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Two repositories for the two persisted entities. PTZState has none.        │
│                                                                              │
│  ┌───────────────────────────────────┐  ┌────────────────────────────────┐  │
│  │      CameraConfigRepository       │  │  CameraCalibrationRepository   │  │
│  ├───────────────────────────────────┤  ├────────────────────────────────┤  │
│  │ get(camera_id) -> Config | None   │  │ get(camera_id) -> Calib | None │  │
│  │ get_by_map(map_id) -> dict        │  │ save(calibration) -> None      │  │
│  │ save(config) -> None              │  │ delete(camera_id) -> bool      │  │
│  │ delete(camera_id) -> bool         │  │ exists(camera_id) -> bool      │  │
│  │ exists(camera_id) -> bool         │  └────────────────────────────────┘  │
│  └───────────────────────────────────┘                                      │
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
- [x] `Orientation` - done (consolidated from BaseOrientation + FinalOrientation, with rotation_matrix)
- [x] `TiltConvention` - done (enum with sign property)
- [x] `LensDistortion` - done (k1, k2, p1, p2 coefficients)

### Phase 2: Composite Value Objects

**Status**: ✅ Complete

- [x] ~~`CameraInstallation`~~ - DELETED (replaced by `CameraCalibration` entity)
- [x] `CameraSpec` - done (enum with intrinsics, tilt_convention per camera model; distortion removed)

### Phase 3: Entities

**Status**: ✅ Complete

- [x] `Annotation` - moved to entities/ directory
- [x] `GroundControlPoint` - done (computed id: map_id/name)
- [x] `Map` - done (id, photo, geotiff)
- [x] ~~`Camera`~~ - DELETED and split into:
  - [x] `CameraConfig` - done (map_id, name, spec, ip_address; computed id)
  - [x] `CameraCalibration` - done (camera_id, position, height, base_orientation, distortion)
- [x] `CameraSnapshot` VO - done (combines config + calibration + ptz_state)

### Phase 4: Services

**Status**: ✅ Complete

- [x] `ServiceOrientation` with strategy pattern - done (AdditiveOrientationStrategy + RotationMatrixStrategy)
- [x] Services README.md - done (documents naming conventions)
- [x] `CoordinateTransformService` - NOT NEEDED (GeoTiff VO has pixel_to_geo/geo_to_pixel methods)
- [x] `ServiceHomography` with strategy pattern - done (IntrinsicExtrinsicStrategy wraps CameraGeometry)

### Phase 5: Migration of Existing Code

**Status**: Partially complete (updated Mar 2026 after PR #225 deep cleanup and PR #226 project hygiene)

PR #225 removed 1,289 lines of dead code and vestigial VOs. The `services/` directory
was deleted entirely. PR #226 migrated `camera_config.py` to return DDD entities
(`CameraConfig`, `CameraCalibration`) via YAML repositories, and removed the hardcoded
`CAMERAS` list (including legacy `lat`/`lon` fields). The domain model continues to
evolve with new VOs and entities added as features require them.

**CRITICAL**: The new domain model (Phases 1-4, 6) is FROZEN. Do NOT modify any files in:
- `poc_homography/domain/` (entities, VOs, enums, repositories)
- `poc_homography/infrastructure/repositories/`

Only update legacy code to USE the new domain model.

#### 5.1 Delete Legacy GPS/UTM Code
- [x] ~~Remove `lat`, `lon` fields from camera configs (use PixelPoint position instead)~~ -- `CAMERAS` list deleted in PR #226; remaining `lat`/`lon` in `__main__` block is tenant location data, not camera config
- [x] ~~Delete `data/gcps/valte.yaml` UTM coordinates and recreate with pixel coordinates~~ -- `valte.yaml` removed (UTM entries were invalid); GCPs stored as per-entity YAML files in `data/gcps/<map_id>__<name>.yaml`
- [x] ~~Remove GPS-to-UTM conversion utilities that are no longer needed~~ -- `geotiff_utils.py` deleted in PR #225
- [ ] Delete any legacy coordinate transformation code -- `camera_config.py` still contains inline coordinate helpers

#### 5.2 Update Legacy Modules
- [x] ~~Update `poc_homography/camera_config.py` to use CameraConfig + CameraCalibration entities~~ -- migrated in PR #226; functions now return domain entities via YAML repositories
- [ ] Update `poc_homography/camera_geometry.py` to use new domain VOs -- still exists, uses raw dicts
- [ ] Update `poc_homography/homography/` modules to use new domain model -- still exists (`config.py`, `interface.py`, `intrinsic_extrinsic.py`, `map_points.py`, `parameters.py`)
- [ ] Update `poc_homography/map_points/` to use new domain MapPoint VO -- still exists (`gcp_registry.py`, `map_point.py`)

#### 5.3 Fix Skipped Tests
- [x] ~~Fix `tests/calibration/test_annotation.py` - update to new Annotation API~~ -- test file deleted (annotation tests live elsewhere now)
- [ ] Fix `tests/homography/test_map_points.py` - update data paths and imports -- still skipif-guarded
- [ ] Fix `tests/homography/test_map_points_integration.py` - update data paths and imports -- still skipif-guarded
- [x] ~~Fix `tests/map_points/test_ground_control_point_collection_serialization.py` - update MapPoint API~~ -- test file deleted (serialization covered by DDD repo tests)

#### 5.4 Validation
- [ ] All pyright checks pass
- [ ] All vulture checks pass
- [ ] All tests pass (no skipped tests except intentional ones)
- [ ] Pre-commit hooks pass

### Phase 6: Configuration Migration

**Status**: ✅ Complete

- [x] Create `data/maps/` for Map YAML files
- [x] Create `data/cameras/` for Camera YAML files - done (valte__Valte.yaml)
- [x] Create `data/calibrations/` for Calibration YAML files - done (valte__Valte.yaml)
- [x] Create `data/gcps/` for GCP YAML files
- [x] `MapRepository` interface - done (get, get_all, exists, save, delete)
- [x] `GroundControlPointRepository` interface - done (get_by_map returns dict, save, delete, exists)
- [x] `CameraConfigRepository` interface - done (get, get_by_map, save, delete, exists)
- [x] `CameraCalibrationRepository` interface - done (get, save, delete, exists)
- [x] `YamlMapRepository` - done (with save/delete)
- [x] `YamlGroundControlPointRepository` - done (returns dict keyed by id)
- [x] `YamlCameraConfigRepository` - done (file naming: camera_id with "/" replaced by "__")
- [x] `YamlCameraCalibrationRepository` - done (file naming: camera_id with "/" replaced by "__")
- [x] ~~**DATA ISSUE (Phase 5.1)**: GCPs in `data/gcps/valte.yaml` have UTM coordinates~~ -- `valte.yaml` removed (invalid UTM entries); GCPs stored as per-entity YAML files `<map_id>__<name>.yaml` with pixel coordinates.
- [ ] **DATA ISSUE (Phase 5.1)**: Camera calibration positions in `data/calibrations/` may use estimated pixel values. Needs proper calibration.

---

## 6. File Structure

> Updated Feb 2026 to reflect actual codebase after PR #225 deep cleanup.
> The `services/` directory was deleted (domain services removed).
> Several VOs and entities were added as new features matured.

```
poc_homography/
├── domain/
│   ├── __init__.py             ✅
│   ├── vo/
│   │   ├── __init__.py         ✅
│   │   ├── credential.py       ✅
│   │   ├── geotiff.py          ✅
│   │   ├── height_uncertainty.py ✅
│   │   ├── image_dimensions.py ✅
│   │   ├── lens_distortion.py  ✅
│   │   ├── line_trace.py       ✅
│   │   ├── map_point.py        ✅
│   │   ├── matrix3x3.py        ✅
│   │   ├── orientation.py      ✅
│   │   ├── photo.py            ✅
│   │   ├── pixel_point.py      ✅
│   │   ├── ptz_state.py        ✅
│   │   ├── rotation.py         ✅
│   │   ├── vector3.py          ✅
│   │   └── zoom_calibration_entry.py ✅
│   │   # REMOVED: camera_intrinsics.py (vestigial, deleted in PR #225)
│   │   # REMOVED: camera_snapshot.py   (vestigial, deleted in PR #225)
│   │   # REMOVED: homography.py        (vestigial, deleted in PR #225)
│   │   # REMOVED: mask.py              (vestigial, deleted in PR #225)
│   ├── entities/
│   │   ├── __init__.py         ✅
│   │   ├── annotation.py       ✅
│   │   ├── calibration_line_trace_set.py ✅
│   │   ├── camera_calibration.py ✅
│   │   ├── camera_config.py    ✅ (replaced camera.py)
│   │   ├── captured_frame.py   ✅
│   │   ├── entity.py           ✅ (base entity class)
│   │   ├── ground_control_point.py ✅
│   │   ├── lens_calibration_table.py ✅
│   │   ├── line.py             ✅
│   │   ├── line_annotation.py  ✅
│   │   ├── map.py              ✅
│   │   └── tenant.py           ✅
│   ├── enums/
│   │   ├── __init__.py         ✅
│   │   ├── tilt_convention.py  ✅
│   │   └── camera_spec.py      ✅ (distortion removed)
│   ├── protocols/
│   │   ├── __init__.py         ✅
│   │   └── camera_controller.py ✅
│   └── repositories/
│       ├── __init__.py         ✅
│       └── repo.py             ✅ (unified repository protocol)
│       # REMOVED: individual *_repository.py files consolidated into repo.py
├── infrastructure/
│   ├── __init__.py             ✅
│   └── repositories/
│       ├── __init__.py         ✅
│       ├── base/
│       │   ├── __init__.py             ✅
│       │   ├── mixin_repo_map_filter.py    ✅
│       │   ├── mixin_repo_tenant_filter.py ✅
│       │   └── repo_yaml.py            ✅ (base YAML repo class)
│       ├── repo_yaml_annotation.py             ✅
│       ├── repo_yaml_calibration_line_trace_set.py ✅
│       ├── repo_yaml_camera_calibration.py     ✅
│       ├── repo_yaml_camera_config.py          ✅
│       ├── repo_yaml_captured_frame.py         ✅
│       ├── repo_yaml_diagnostic_session.py     ✅
│       ├── repo_yaml_ground_control_point.py   ✅
│       ├── repo_yaml_lens_calibration_table.py ✅
│       ├── repo_yaml_line.py                   ✅
│       ├── repo_yaml_line_annotation.py        ✅
│       ├── repo_yaml_map.py                    ✅
│       ├── repo_yaml_stress_test_session.py    ✅
│       ├── repo_yaml_survey_session.py         ✅
│       └── repo_yaml_tenant.py                 ✅
├── data/
│   ├── maps/
│   │   └── *.yaml              ✅
│   ├── gcps/
│   │   └── <map_id>__<name>.yaml ✅ (per-entity YAML files; old valte.yaml removed)
│   ├── cameras/
│   │   └── <tenant>__<name>.yaml ✅
│   └── calibrations/
│       └── <tenant>__<name>.yaml ✅
└── ...
# REMOVED: services/ directory (deleted in PR #225 — domain services were vestigial)
```

---

## 7. Open Questions

1. **Calibration table**: How to model zoom-dependent intrinsics? Separate VO or part of `Camera` entity? (Currently not implemented — using linear focal length approximation)

2. **GPS coordinates**: Do we still need GPS lat/lon for cameras, or is MapPoint sufficient?

---

## 8. ~~Pending~~ Completed Code Changes

### 8.1 Split Camera into CameraConfig + CameraCalibration ✅ DONE

**Current state** (conflated):
```python
@dataclass
class Camera:
    map_id: str
    name: str
    installation: CameraInstallation  # ❌ Mixed with config
    spec: CameraSpec
    ptz_state: PTZState  # ❌ Different lifecycle
    ip_address: str | None = None

    def update_ptz_state(self, new_state: PTZState) -> None:  # ❌ Mutability
        ...
```

**Target state** (separated):
```python
@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration. Set once during registration."""
    map_id: str
    name: str
    spec: CameraSpec
    ip_address: str | None = None

    @property
    def id(self) -> str:
        return f"{self.map_id}/{self.name}"


@dataclass(frozen=True)
class CameraCalibration:
    """Camera calibration data. Refined during calibration process."""
    camera_id: str  # References CameraConfig.id
    position: PixelPoint
    height: Meters
    base_orientation: Orientation
    distortion: LensDistortion
```

### 8.2 Remove CameraInstallation VO ✅ DONE

`CameraInstallation` is replaced by `CameraCalibration` entity. The fields move directly:

| CameraInstallation (remove) | CameraCalibration (new) |
|-----------------------------|-------------------------|
| `position: PixelPoint` | `position: PixelPoint` |
| `height: Meters` | `height: Meters` |
| `base_orientation: Orientation` | `base_orientation: Orientation` |
| (was in CameraSpec) | `distortion: LensDistortion` |

### 8.3 CameraSpec Enum Refactoring ✅ DONE

**Current state** (has distortion):
```python
class CameraSpec(Enum):
    HIKVISION_DS_2DF8425IX = (
        ...,
        LensDistortion(k1=..., k2=...),  # ❌ REMOVE - not model-specific
    )
```

**Target state** (only hardware-fixed values):
```python
class CameraSpec(Enum):
    HIKVISION_DS_2DF8425IX = (
        model_name,
        sensor_width,
        base_focal_length,
        image_width,
        image_height,
        tilt_convention,
        max_zoom,
        # NO distortion - it's calibrated per-camera
    )
```

### 8.4 New VO: CameraSnapshot ✅ DONE

```python
@dataclass(frozen=True)
class CameraSnapshot:
    """Immutable point-in-time combination of all camera data."""
    config: CameraConfig
    calibration: CameraCalibration
    ptz_state: PTZState
```

### 8.5 New Repositories ✅ DONE

```python
class CameraConfigRepository(Protocol):
    def get(self, camera_id: str) -> CameraConfig | None: ...
    def get_by_map(self, map_id: str) -> dict[str, CameraConfig]: ...
    def save(self, config: CameraConfig) -> None: ...
    def delete(self, camera_id: str) -> bool: ...
    def exists(self, camera_id: str) -> bool: ...


class CameraCalibrationRepository(Protocol):
    def get(self, camera_id: str) -> CameraCalibration | None: ...
    def save(self, calibration: CameraCalibration) -> None: ...
    def delete(self, camera_id: str) -> bool: ...
    def exists(self, camera_id: str) -> bool: ...
```

### 8.6 Summary of Changes ✅ ALL DONE

| Component | Change |
|-----------|--------|
| `Camera` entity | Split into `CameraConfig` + `CameraCalibration` |
| `CameraInstallation` VO | Remove (replaced by `CameraCalibration` entity) |
| `CameraSpec` enum | Remove `distortion` parameter |
| `CameraSnapshot` VO | New - combines Config + Calibration + PTZState |
| `CameraConfigRepository` | New repository interface |
| `CameraCalibrationRepository` | New repository interface |
| Domain Services | Take `config`, `calibration`, `ptz_state` as separate arguments |

---

## 9. References

- GDAL GeoTransform: https://gdal.org/tutorials/geotransforms_tut.html
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Rotation Matrix Conventions: https://en.wikipedia.org/wiki/Rotation_matrix
