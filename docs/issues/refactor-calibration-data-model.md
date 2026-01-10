**Issue Type**: Task

## Summary
Refactor the calibration data model to properly separate Ground Control Points (GCPs), camera Annotations, and CaptureContext. The current model conflates GCPs (known world/map locations) with Annotations (where GCPs appear in specific camera images), causing conceptual confusion and incorrect file naming.

## Context
The codebase currently uses the term "GCP" to refer to both map reference points and their observed locations in camera images. This conflation creates confusion in the data model:

1. `MapPoint` incorrectly includes an `id` field, duplicating the registry's external dictionary keys
2. Files contain arrays named `gcps` that actually store camera annotations (GCP appearances in images with PTZ state)
3. No explicit `Annotation` dataclass exists to represent the camera observation of a GCP
4. `CaptureContext` (camera state during capture) is embedded in annotation data rather than being a first-class parent wrapper

This refactoring clarifies the conceptual model:
- **GCP**: A known location in map/world coordinates (just an ID referencing a MapPoint)
- **Annotation**: An observation linking a GCP to its pixel location in a specific camera image
- **CaptureContext**: Camera state (camera name, pan/tilt/zoom) when the frame was captured
- **Observation**: Runtime computed projection (not persisted)

## Scope

### In Scope
- Create `Annotation` dataclass with `gcp_id`, `image_u`, `image_v` fields
- Create/formalize `CaptureContext` dataclass with `camera`, `pan_raw`, `tilt_deg`, `zoom` fields
- Remove `id` field from `MapPoint` dataclass
- Update all YAML/JSON calibration files to new format with `capture.context` and `capture.annotations` structure
- Update all calibration tools to use new data model
- Update CLI wrappers in `tools/cli/` to use new data model
- Migrate existing test data files
- Update relevant documentation

### Out of Scope
- Changes to the homography computation algorithms
- Changes to the `MapPointRegistry` structure or serialization format
- Migration of non-calibration YAML files (e.g., `homography_config.yaml` sections unrelated to GCPs)
- Changes to PTZ control or camera configuration systems
- Modifications to coordinate transformation logic
- GPU or performance optimizations

## Definition of Done
- [ ] `Annotation` dataclass exists in `poc_homography/` with `gcp_id: str`, `image_u: float`, `image_v: float` fields
- [ ] `CaptureContext` dataclass exists in `poc_homography/` with `camera: str`, `pan_raw: float`, `tilt_deg: float`, `zoom: float` fields
- [ ] `MapPoint` dataclass has `id` field removed, retaining only `pixel_x`, `pixel_y`, `map_id`
- [ ] All YAML files in `config/` directory use new format with `capture.context` and `capture.annotations` sections
- [ ] All `test_data_*.json` files migrated to new annotation structure
- [ ] `tools/calibrate_projection.py` updated to use `Annotation` and `CaptureContext`
- [ ] `tools/validate_camera_model.py` updated to use new data model
- [ ] `tools/interactive_calibration.py` updated to use new data model
- [ ] `tools/comprehensive_calibration.py` updated to use new data model
- [ ] All CLI wrappers in `tools/cli/` updated (minimum: `calibrate_projection.py`, `validate_camera_model.py`, `interactive_calibration.py`, `comprehensive_calibration.py`, `test_data_generator.py`)
- [ ] `tools/test_data_generator.py` exports data in new annotation format
- [ ] All existing tests pass with updated data structures
- [ ] Documentation updated to reflect new terminology (GCP vs Annotation vs Observation)

## Implementation Notes

### MapPoint Refactoring

**Context**: The `MapPoint` dataclass currently includes an `id` field that duplicates the registry's dictionary keys. The registry already provides ID-based lookup through its external dictionary structure (`{id: {pixel_x, pixel_y}}`). This duplication violates single source of truth and creates serialization complexity.

**Requirements**:
- Remove `id` field from `MapPoint` dataclass in `poc_homography/map_points/map_point.py`
- Remove `map_id` field as it's also redundant (registry already knows which map it represents)
- Keep only: `pixel_x: float`, `pixel_y: float`
- Update `to_dict()` and `from_dict()` methods to match new structure
- Update `pixel` property and any cached computations

**Reference**: See `poc_homography/map_points/map_point_registry.py` for how the registry uses external dictionary keys as IDs.

### Annotation Dataclass Creation

**Context**: Currently, camera observations of GCPs are stored in YAML files under arrays named `gcps`, but these are actually annotations (linking a GCP to its pixel appearance in a specific camera image). This conceptual confusion needs resolution through an explicit dataclass.

**Requirements**:
- Create new `Annotation` dataclass in `poc_homography/` (suggest `poc_homography/calibration/annotation.py` or similar location)
- Fields: `gcp_id: str`, `image_u: float`, `image_v: float`
- Make dataclass frozen for immutability
- Implement `to_dict()` and `from_dict()` for serialization
- Add docstring clarifying: "An annotation links a Ground Control Point (GCP) to its observed pixel location in a camera image"

**Example**:
```python
@dataclass(frozen=True)
class Annotation:
    """Camera observation linking a GCP to its pixel location in an image.

    Attributes:
        gcp_id: ID of the GCP in the map point registry
        image_u: Pixel x-coordinate in camera image
        image_v: Pixel y-coordinate in camera image
    """
    gcp_id: str
    image_u: float
    image_v: float
```

### CaptureContext Dataclass Formalization

**Context**: Camera state during frame capture (camera name, pan/tilt/zoom) is currently embedded within GCP arrays or passed as separate parameters. This needs formalization as a first-class concept that groups annotations together.

**Requirements**:
- Create/formalize `CaptureContext` dataclass in `poc_homography/` (same module as `Annotation`)
- Fields: `camera: str`, `pan_raw: float`, `tilt_deg: float`, `zoom: float`
- Make dataclass frozen for immutability
- Implement `to_dict()` and `from_dict()` for serialization
- Add docstring clarifying: "Camera state when a calibration frame was captured"
- Consider adding optional fields if needed by tools: `image_width: int`, `image_height: int`, `timestamp: str`

**Example**:
```python
@dataclass(frozen=True)
class CaptureContext:
    """Camera state when a calibration frame was captured.

    Attributes:
        camera: Camera name (e.g., "Valte", "Setram")
        pan_raw: Raw pan position from PTZ API
        tilt_deg: Tilt angle in degrees
        zoom: Zoom level (1.0 = no zoom)
    """
    camera: str
    pan_raw: float
    tilt_deg: float
    zoom: float
```

### File Format Migration

**Context**: Existing YAML files use incorrect structure where arrays named `gcps` contain annotation data mixed with PTZ state. The new format separates concerns: a single `capture.context` describes camera state, and `capture.annotations` lists GCP observations.

**Current Format** (incorrect):
```yaml
gcps:
  - map_point_id: Z1
    pixel_u: 960
    pixel_v: 540
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
  - map_point_id: Z2
    pixel_u: 1200
    pixel_v: 600
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
```

**New Format** (correct):
```yaml
capture:
  context:
    camera: Valte
    pan_raw: 0.0
    tilt_deg: 30.0
    zoom: 1.0
  annotations:
    - gcp_id: Z1
      image_u: 960
      image_v: 540
    - gcp_id: Z2
      image_u: 1200
      image_v: 600
```

**Migration Requirements**:
- Update all files in `config/` directory: `config/gcps_valte_test.yaml`, `config/gcps_valte_test2.yaml`, `config/valte_gcps.yaml`
- Rename `gcps` arrays to `annotations` everywhere
- Extract shared PTZ state into single `context` object per capture
- Update field names: `map_point_id` → `gcp_id`, `pixel_u` → `image_u`, `pixel_v` → `image_v`
- Handle legacy formats in loading functions (detect old format, convert on read, warn user)

**Note**: Some YAML files like `config/gcps_valte_test.yaml` use a different structure with `ground_control_points` containing GPS coordinates and image pixels. Determine if these need migration or are a separate format (homography feature_match approach). If they are GCP-based, map the `image.u`/`image.v` fields to the new annotation structure.

### Calibration Tools Update

**Context**: Multiple tools in `tools/` and `tools/cli/` load and process GCP data. All must be updated to use the new `Annotation` and `CaptureContext` dataclasses and expect the new file format.

**Tools Requiring Updates**:
- `tools/calibrate_projection.py` - Uses GCP data for projection error analysis
- `tools/validate_camera_model.py` - Loads GCPs from YAML via `load_gcps_from_yaml()` function
- `tools/interactive_calibration.py` - Interactive GCP marking and calibration
- `tools/comprehensive_calibration.py` - Comprehensive calibration workflow
- `tools/test_data_generator.py` - Exports test data (must export in new annotation format)
- All CLI wrappers in `tools/cli/` that import or wrap the above tools

**Requirements**:
- Replace direct YAML parsing with functions that return `Annotation` and `CaptureContext` objects
- Update function signatures to accept/return new dataclasses instead of dictionaries
- Update variable names: rename `gcp` variables to `annotation` where appropriate
- Ensure backward compatibility in loading functions (detect old format, convert, warn)
- Update export functions to write new format

**Reference**: See `tools/validate_camera_model.py` function `load_gcps_from_yaml()` as an example of a loading function that needs updating.

### Test Data Migration

**Context**: Test data files (`test_data_*.json`) contain GCP annotations for testing. These must be migrated to the new annotation structure.

**Requirements**:
- Identify all `test_data_*.json` files in the repository
- Convert structure to match new `capture.context` and `capture.annotations` format
- Update any test code that loads these files to expect new structure
- Verify all tests pass after migration

**Files Known to Exist**:
- Referenced in: `tools/test_data_generator.py`, `tests/test_homography_map_points_integration.py`, `tests/test_homography_map_points.py`, `examples/demo_map_point_homography.py`

### Documentation Updates

**Context**: The terminology shift from conflating GCPs with annotations requires documentation updates to clarify the conceptual model.

**Requirements**:
- Update any documentation that explains the GCP/annotation data model
- Clarify distinction between:
  - **GCP**: Known map/world location (just an ID in the registry)
  - **Annotation**: Observation of a GCP in a camera image (persisted)
  - **Observation**: Runtime computed projection (not persisted)
  - **CaptureContext**: Camera state during capture
- Update file format examples in documentation
- Update any developer guides or README files that reference the old structure

**Files Likely Needing Updates**:
- `docs/MAP_POINT_HOMOGRAPHY.md`
- `tools/README_test_data_generator.md`
- `HOMOGRAPHY_MAP_POINTS_SUMMARY.md`

### Error Handling and Backward Compatibility

**Context**: Existing YAML/JSON files in production or user environments may use the old format. Loading functions should detect old format, convert it, and warn users.

**Requirements**:
- Implement format detection in loading functions (check for `gcps` array vs `capture.annotations`)
- If old format detected: convert to new structure in-memory, emit warning to stderr
- Warning message should guide users to migrate files using a migration script or manual conversion
- Consider creating a standalone migration script in `tools/` to batch-convert old files
- Ensure error messages are clear when required fields are missing
