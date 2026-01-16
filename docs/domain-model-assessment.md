# Domain Model Assessment

**Date:** 2025-01-16
**Context:** Issue #172 - DDD Refactoring
**Scope:** Entities, Value Objects, and Legacy Types in `poc_homography`

---

## Executive Summary

This assessment identifies duplication, missing abstractions, and architectural issues in the current domain model. The primary findings are:

1. **DRY violations** between legacy `camera_parameters.py` and new domain VOs
2. **Missing VOs** for core concepts (Homography, ImageDimensions)
3. **God Object** pattern in `CameraParameters` that should be decomposed
4. **Incomplete VOs** missing fields or factory methods

---

## Current Inventory

### Entities (7)

| Entity | Location | Purpose |
|--------|----------|---------|
| `Entity` | `domain/entities/entity.py` | Protocol for serializable entities |
| `CameraConfig` | `domain/entities/camera_config.py` | Static camera registration data |
| `CameraCalibration` | `domain/entities/camera_calibration.py` | Calibration data (position, orientation, distortion) |
| `Map` | `domain/entities/map.py` | Georeferenced map image |
| `GroundControlPoint` | `domain/entities/ground_control_point.py` | Reference point on map |
| `Annotation` | `domain/entities/annotation.py` | GCP to pixel observation link |
| `CapturedFrame` | `domain/entities/captured_frame.py` | Photo with PTZ state |

### Value Objects (15)

| VO | Location | Purpose |
|----|----------|---------|
| `CameraIntrinsics` | `domain/vo/camera_intrinsics.py` | Sensor/lens parameters, computes K matrix |
| `Orientation` | `domain/vo/orientation.py` | Yaw/pitch/roll angles, computes rotation matrix |
| `PTZState` | `domain/vo/ptz_state.py` | Pan/tilt/zoom values |
| `LensDistortion` | `domain/vo/lens_distortion.py` | Distortion coefficients (k1, k2, p1, p2, k3) |
| `HeightUncertainty` | `domain/vo/height_uncertainty.py` | Height confidence interval for error propagation |
| `Matrix3x3` | `domain/vo/matrix3x3.py` | Immutable 3x3 matrix with math operations |
| `Homography` | `domain/vo/homography.py` | Projective transformation using Matrix3x3 |
| `PixelPoint` | `domain/vo/pixel_point.py` | Pixel coordinates (x, y) |
| `MapPoint` | `domain/vo/map_point.py` | Point on map (map_id + pixel_point) |
| `GeoTransform` | `domain/vo/geotiff.py` | Affine transformation (6 parameters) |
| `GeoTiff` | `domain/vo/geotiff.py` | GeoTIFF metadata with coordinate transforms |
| `Photo` | `domain/vo/photo.py` | Image file with dimensions |
| `Mask` | `domain/vo/mask.py` | Binary segmentation mask |
| `Credential` | `domain/vo/credential.py` | Username/password |
| `CameraSnapshot` | `domain/vo/camera_snapshot.py` | Combines config + calibration + ptz_state |

### Enums (2)

| Enum | Location | Purpose |
|------|----------|---------|
| `CameraSpec` | `domain/enums/camera_spec.py` | Known camera models with hardware specs |
| `TiltConvention` | `domain/enums/tilt_convention.py` | POSITIVE_UP / POSITIVE_DOWN |

### Legacy Types (2)

| Type | Location | Purpose |
|------|----------|---------|
| `CameraParameters` | `camera_parameters.py` | All params for homography computation |
| `CameraGeometryResult` | `camera_parameters.py` | Homography computation result |

### Type Aliases

| Type | Location | Purpose |
|------|----------|---------|
| `Degrees` | `types.py` | Angular measurement |
| `Radians` | `types.py` | Angular measurement |
| `Meters` | `types.py` | Distance/position |
| `Easting` | `types.py` | UTM X-coordinate |
| `Northing` | `types.py` | UTM Y-coordinate |
| `Pixels` | `types.py` | Integer pixel coordinates |
| `PixelsFloat` | `types.py` | Subpixel coordinates |
| `Millimeters` | `types.py` | Physical dimensions |
| `Unitless` | `types.py` | Dimensionless scalars |

---

## Issues

### DRY Violations (Duplications)

#### 1. ~~DistortionCoefficients vs LensDistortion~~ ✅ RESOLVED

**Status:** Unified into single `LensDistortion` VO with all 5 coefficients (k1, k2, p1, p2, k3).

`DistortionCoefficients` has been removed. All code now uses `LensDistortion` from `domain/vo/lens_distortion.py`.

#### 2. Orientation vs pan/tilt/roll

**Problem:** Angular orientation represented in three different ways.

```python
# domain/vo/orientation.py
Orientation:      yaw: Degrees, pitch: Degrees, roll: Degrees

# camera_parameters.py
CameraParameters: pan_deg: Degrees, tilt_deg: Degrees, roll_deg: Degrees

# domain/vo/ptz_state.py
PTZState:         pan_raw: float, tilt_deg: float, zoom: float
```

**Impact:**
- No clear conversion semantics between representations
- `pan_raw` has unclear units/meaning
- `PTZState` doesn't use typed `Degrees`

**Resolution:** Clarify relationship; PTZState is raw hardware values, Orientation is computed world-referenced angles.

#### 3. CameraIntrinsics vs intrinsic_matrix

**Problem:** Intrinsic matrix concept exists in three places.

```python
# domain/vo/camera_intrinsics.py
CameraIntrinsics.K  # Computed property returning 3x3 matrix

# camera_parameters.py
CameraParameters._intrinsic_matrix_data  # Stored as bytes

# camera_geometry.py
CameraGeometry.get_intrinsics()  # Static factory method
```

**Impact:**
- No single source of truth for creating intrinsics
- `CameraGeometry.get_intrinsics()` duplicates what `CameraIntrinsics` should do

**Resolution:** Add `CameraIntrinsics.from_spec_and_zoom()` factory method.

---

### Missing Value Objects

#### 1. ~~Homography VO~~ ✅ RESOLVED

**Status:** Created `domain/vo/homography.py` with `Matrix3x3` foundation.

#### 2. ~~WorldPoint VO~~ ❌ NOT APPLICABLE

**Status:** After analysis, this VO is not needed. The "world coordinate" concept (`np.ndarray([easting, northing, height])`) is:
- **Internal to legacy code** (`CameraParameters`, `CameraGeometry`)
- **Not a domain concept** users interact with
- **Superseded by** `MapPoint` + `GeoTiff.pixel_to_geo()` for coordinate transformation

The domain model uses:
- `CameraCalibration.position` → `PixelPoint` (position on map)
- `CameraCalibration.height` → `Meters` (height above ground)
- `GeoTiff` → handles pixel ↔ UTM transformation

The raw `np.ndarray` in `CameraParameters.camera_position` is an implementation detail of the homography computation, not a domain concept.

#### 3. ImageDimensions VO

**Problem:** Width/height pairs scattered across many locations.

Currently duplicated in:
- `CameraIntrinsics.image_width/height`
- `CameraParameters.image_width/height`
- `Photo.width/height`
- `CameraSpec.image_width/height`

**Should be:**
```python
@dataclass(frozen=True)
class ImageDimensions:
    """Image width and height in pixels."""
    width: Pixels
    height: Pixels

    @property
    def aspect_ratio(self) -> float: ...

    @property
    def center(self) -> PixelPoint: ...

    def contains(self, point: PixelPoint) -> bool: ...
```

#### 4. ~~CameraPosition VO~~ ❌ NOT APPLICABLE

**Status:** Not needed. This was proposed to wrap `MapPoint` + `height` with a `to_world()` method returning `WorldPoint`. Since `WorldPoint` is not applicable, neither is this VO.

The current design with separate `position: PixelPoint` and `height: Meters` in `CameraCalibration` is sufficient. The coordinate transformation to UTM is handled by `GeoTiff.pixel_to_geo()` at computation time.

#### 5. ~~HeightUncertainty (Move to domain)~~ ✅ RESOLVED

**Status:** Moved to `domain/vo/height_uncertainty.py` with additional semantic methods (`range`, `midpoint`, `contains`) and factory method (`symmetric`).

---

### Incomplete VOs

| VO | Issue | Resolution |
|----|-------|------------|
| ~~`LensDistortion`~~ | ~~Missing `k3` coefficient~~ | ✅ Added `k3: Unitless = Unitless(0.0)` |
| `PTZState` | `pan_raw: float` has unclear semantics | Document or rename; add validation |
| `PTZState` | No type safety on fields | Use `Degrees` for angles, `Unitless` for zoom |
| `CameraIntrinsics` | No factory from CameraSpec + zoom | Add `from_spec_and_zoom(spec, zoom)` classmethod |
| `GeoTiff` | No EPSG code validation | Add validation in `__post_init__` |

---

### Architectural Issues

#### 1. CameraParameters is a God Object

`CameraParameters` bundles too many concerns:

```python
CameraParameters:
    # Image properties
    image_width, image_height

    # Camera intrinsics
    _intrinsic_matrix_data

    # Camera position
    _camera_position_data

    # Orientation
    pan_deg, tilt_deg, roll_deg

    # Map properties
    map_width, map_height, pixels_per_meter

    # Optional
    distortion, height_uncertainty, _affine_matrix_data
```

**Resolution:** Replace with composition of domain VOs:
- `CameraIntrinsics` for intrinsics
- `Orientation` for angles
- `LensDistortion` for distortion
- `GeoTiff` for affine/georeferencing

#### 2. CameraGeometryResult Mixes Concerns

Contains:
- **Homography matrices** → should be `Homography` VO
- **Validation state** (`is_valid`, `validation_messages`) → separate concern
- **Numerical metrics** (`condition_number`, `determinant`) → computation metadata

**Resolution:** Split into:
- `Homography` VO (matrices + projection methods)
- `HomographyValidation` or return tuple with metadata

#### 3. Coordinate Systems

The codebase handles coordinate systems with clear VO representation:

| Coordinate System | Current Representation | Status |
|-------------------|----------------------|--------|
| Image pixels | `PixelPoint` | ✅ OK |
| Map pixels | `MapPoint` | ✅ OK |
| Geographic (UTM) | `Easting`/`Northing` types + `GeoTiff` | ✅ OK |
| Internal computation | `np.ndarray` in legacy code | Internal detail |

---

## Recommended Actions

### Phase 1: Eliminate Duplications ✅ COMPLETED

1. ~~**Unify distortion coefficients**~~ ✅ COMPLETED
   - ~~Add `k3` to `LensDistortion`~~ ✅
   - ~~Remove `DistortionCoefficients`~~ ✅
   - ~~Update all usages~~ ✅

2. ~~**Move HeightUncertainty to domain**~~ ✅ COMPLETED
   - ~~Create `domain/vo/height_uncertainty.py`~~ ✅
   - ~~Update imports~~ ✅

### Phase 2: Add Missing Core VOs

3. ~~**Create Homography VO**~~ ✅ COMPLETED
   - ~~`domain/vo/homography.py`~~ ✅
   - ~~Move projection methods from `CameraGeometry`~~ ✅
   - ~~Include matrix validation~~ ✅

4. **Create ImageDimensions VO**
   - `domain/vo/image_dimensions.py`
   - Update `Photo`, `CameraIntrinsics` to use it

### Phase 3: Refactor Legacy

5. **Add CameraIntrinsics factory**
   - `CameraIntrinsics.from_spec_and_zoom(spec, zoom)`
   - Deprecate `CameraGeometry.get_intrinsics()`

6. **Deprecate CameraParameters**
   - Create adapter that builds from domain VOs
   - Gradually migrate callers

7. **Split CameraGeometryResult**
    - Extract `Homography` VO
    - Keep validation/metrics separate

### Phase 4: Type Safety

8. **Fix PTZState typing**
    - Document `pan_raw` semantics
    - Use typed fields (`Degrees`, `Unitless`)

9. **Add PTZState validation**
    - Validate zoom bounds (1.0 to max_zoom)
    - Validate angle ranges

---

## Design Pattern Standards

### Reference Pattern: Private Constructor with Sentinel

The `Matrix3x3` and `Homography` VOs establish the standard pattern for complex value objects:

```python
_PRIVATE_SENTINEL = object()

@dataclass(frozen=True)
class ExampleVO:
    """Immutable value object with controlled construction."""

    _data: SomeType = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError("Use ExampleVO.create() instead")

    @classmethod
    def create(cls, ...) -> ExampleVO:
        # ALL validation here
        return cls(_data=..., _sentinel=_PRIVATE_SENTINEL)
```

**Benefits:**
- Validation centralized in single location (`create()`)
- Constructor cannot be called directly (enforced at runtime)
- Immutable (frozen dataclass)
- Hashable (can be used in sets/dicts)

### When to Apply This Pattern

| Apply Pattern | Don't Apply |
|---------------|-------------|
| Stores numpy arrays | Simple scalar fields only |
| Requires validation | No invariants to enforce |
| Complex construction logic | Direct field assignment sufficient |
| Used in computations | Pure data transfer |

---

## Pattern Alignment Analysis

### Summary Table

| Class | Type | Frozen | Factory | Sentinel | Numpy | Validation | Status |
|-------|------|--------|---------|----------|-------|------------|--------|
| `Matrix3x3` | VO | ✓ | ✓ | ✓ | ✓ bytes | ✓ | ✅ Reference |
| `Homography` | VO | ✓ | ✓ | ✓ | via Matrix3x3 | ✓ | ✅ Reference |
| `CameraIntrinsics` | VO | ✓ | ✗ | ✗ | ✓ property | ✗ | ⚠️ Refactor |
| `Orientation` | VO | ✓ | ✗ | ✗ | ✓ property | ✗ | ⚠️ Refactor |
| `HeightUncertainty` | VO | ✓ | helper | ✗ | ✗ | `__post_init__` | ⚠️ Refactor |
| `LensDistortion` | VO | ✓ | helpers | ✗ | ✓ methods | partial | ⚠️ Refactor |
| `GeoTransform` | VO | ✓ | ✗ | ✗ | ✗ | in method | ⚠️ Refactor |
| `GroundControlPoint` | Entity | ✗ | ✗ | ✗ | ✗ | `__post_init__` | ⚠️ Refactor |
| `CapturedFrame` | Entity | ✓ | ✓ | ✗ | ✗ | ✗ | ⚠️ Add sentinel |
| `PixelPoint` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `MapPoint` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `GeoTiff` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `Credential` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `PTZState` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `Photo` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `Mask` | VO | ✗ | ✓ | ✗ | ✓ array | ✗ | ✅ Mutable OK |
| `CameraSnapshot` | VO | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Composition OK |
| `CameraConfig` | Entity | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `CameraCalibration` | Entity | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Composition OK |
| `Annotation` | Entity | ✓ | ✗ | ✗ | ✗ | ✗ | ✅ Simple OK |
| `Map` | Entity | ✗ | ✗ | ✗ | ✗ | ✗ | ⚠️ Consider freezing |

---

## Refactoring Opportunities

### Priority 1: Numpy Arrays Without Immutability

These VOs return mutable numpy arrays from properties, breaking immutability guarantees.

#### CameraIntrinsics

**Current:** `K` property computes and returns new numpy array each call.

```python
@property
def K(self) -> np.ndarray:
    # Returns mutable 3x3 intrinsic matrix
```

**Recommendation:**
- Store `K` as `Matrix3x3` internally
- Add `create()` factory with sentinel pattern
- Add `to_matrix() -> Matrix3x3` method
- Validate focal length and principal point ranges

#### Orientation

**Current:** `rotation_matrix` property computes and returns new numpy array.

```python
@property
def rotation_matrix(self) -> np.ndarray:
    # Returns mutable 3x3 rotation matrix
```

**Recommendation:**
- Store computed rotation matrix as `Matrix3x3` internally
- Add `create()` factory with sentinel pattern
- Add `to_rotation_matrix() -> Matrix3x3` method
- Validate angle ranges

### Priority 2: Hybrid Validation Patterns

These classes have validation split between `__post_init__` and factory methods.

#### HeightUncertainty

**Current:** Validation in `__post_init__`, helper factory `symmetric()`.

**Recommendation:**
- Add sentinel pattern to block direct construction
- Move validation from `__post_init__` to `create()`
- Keep `symmetric()` as convenience factory calling `create()`

#### GroundControlPoint

**Current:** Unfrozen, ID generation in `__post_init__`.

**Recommendation:**
- Make frozen dataclass
- Add `create()` factory with sentinel
- Generate ID in factory, not `__post_init__`

#### CapturedFrame

**Current:** Has `create()` factory but no sentinel guard.

**Recommendation:**
- Add sentinel check to `__post_init__`
- Ensures all construction goes through `create()`

### Priority 3: Missing Validation

#### LensDistortion

**Current:** `from_numpy()` accepts any array without validation.

**Recommendation:**
- Validate coefficient array length (5 elements)
- Consider adding range validation for coefficients

#### GeoTransform

**Current:** Singularity checked in `geo_to_pixel()` method at runtime.

**Recommendation:**
- Add `create()` factory that validates non-singularity
- Fail fast at construction, not at use

### Priority 4: Minor Improvements

#### Map Entity

**Current:** Unfrozen dataclass.

**Recommendation:**
- Consider making frozen for consistency
- Or document why mutability is required

---

## Dependency Graph (Current)

```
CameraConfig ──────────────┐
                           │
CameraCalibration ─────────┼──► CameraSnapshot
                           │
PTZState ──────────────────┘

Map ──► Photo
    └──► GeoTiff ──► GeoTransform

GroundControlPoint ──► MapPoint ──► PixelPoint

Annotation ──► PTZState
           └──► PixelPoint

CapturedFrame ──► PTZState

CameraSpec ──► TiltConvention
```

## Dependency Graph (Target)

```
CameraConfig ──────────────┐
                           │
CameraCalibration ─────────┼──► CameraSnapshot
                           │
PTZState ──────────────────┘

Map ──► Photo ──► ImageDimensions
    └──► GeoTiff ──► GeoTransform

CameraIntrinsics ──► ImageDimensions

Homography ──► PixelPoint

GroundControlPoint ──► MapPoint ──► PixelPoint

Annotation ──► PTZState
           └──► PixelPoint

CapturedFrame ──► PTZState
```

---

## Appendix: File Locations

```
poc_homography/
├── camera_geometry.py          # Legacy - to be refactored
├── camera_parameters.py        # Legacy - to be deprecated
├── types.py                    # Type aliases - OK
└── domain/
    ├── entities/
    │   ├── annotation.py
    │   ├── camera_calibration.py
    │   ├── camera_config.py
    │   ├── captured_frame.py
    │   ├── entity.py
    │   ├── ground_control_point.py
    │   └── map.py
    ├── enums/
    │   ├── camera_spec.py
    │   └── tilt_convention.py
    └── vo/
        ├── camera_intrinsics.py
        ├── camera_snapshot.py
        ├── credential.py
        ├── geotiff.py
        ├── height_uncertainty.py
        ├── homography.py
        ├── lens_distortion.py
        ├── map_point.py
        ├── mask.py
        ├── matrix3x3.py
        ├── orientation.py
        ├── photo.py
        ├── pixel_point.py
        └── ptz_state.py
```
