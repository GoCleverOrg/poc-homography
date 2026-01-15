# Plan: Camera Frames & Annotations DDD Refactoring

## Goal

Complete the DDD refactoring for camera frame capture and GCP annotation workflows by:
1. Ensuring all data flows through proper repositories (no direct YAML loading in CLI/webapp)
2. Establishing proper domain entities for frame capture and annotations
3. Deleting legacy code that bypasses the architecture

## Current State (What's Done)

### Completed in Previous Session

1. **CameraController Protocol** - `domain/protocols/camera_controller.py`
   - Protocol for PTZ camera control (read status + move)
   - `CameraControllerError` exception

2. **HikvisionCameraController** - `infrastructure/clients/hikvision_camera_controller.py`
   - Implements ISAPI communication
   - Maintains in-memory `last_ptz_state`

3. **CapturedFrame Entity** - `domain/entities/captured_frame.py`
   - Represents photo + PTZ state
   - ID format: `{map_id}/{camera_name}/{timestamp}`

4. **RepoYamlCapturedFrame** - `infrastructure/repositories/repo_yaml_captured_frame.py`
   - Hierarchical storage: `data/frames/{map_id}/{camera_name}/`
   - Methods: `get()`, `save()`, `delete()`, `get_by_camera()`, `get_image_path()`

5. **FrameCaptureService** - `application/services/frame_capture_service.py`
   - Orchestrates: get PTZ → capture RTSP → save image → save entity

6. **Frame CLI Commands** - `cli/frame.py`
   - `hom frame capture <camera>` - captures and stores frame
   - `hom frame list` - lists frames with filters
   - `hom frame show <frame_id>` - shows frame details
   - `hom frame delete <frame_id>` - deletes frame

7. **Updated Interactive Command** - `cli/interactive.py`
   - Now supports `--frame-id` to load from repository
   - Loads GCPs from `repo_gcp` based on map_id (no direct YAML)

## Remaining Work

### Phase 1: Annotation Support

**Problem**: `calibrate.py` and `camera.py` load "GCP observations" directly from YAML files. These observations are actually `Annotation` entities that should be stored per-frame.

**Annotation Entity** already exists at `domain/entities/annotation.py`:
```python
@dataclass(frozen=True)
class Annotation:
    gcp_id: str           # References GCP in repository
    camera_pose: PTZState # PTZ state when observed
    pixel: PixelPoint     # Where GCP appears in image
```

**Tasks**:

1. **Update Annotation entity** (`domain/entities/annotation.py`)
   - Add `frame_id: str` field to link annotation to a CapturedFrame
   - Add `to_dict()` and `from_dict()` methods for serialization
   - Update `__init__.py` exports

2. **Extend RepoYamlCapturedFrame for annotations**
   - Store annotations in `{timestamp}_annotations.yaml` alongside frame
   - Add methods:
     - `get_annotations(frame_id) -> list[Annotation]`
     - `save_annotations(frame_id, annotations: list[Annotation])`
   - Load annotations lazily when requested

3. **Add CLI command for annotations** (`cli/frame.py`)
   - `hom frame annotate <frame_id>` - interactive annotation mode
   - `hom frame annotations <frame_id>` - list annotations for a frame

### Phase 2: Update CLI Commands to Use Repositories

**Files with direct YAML loading (RED FLAGS)**:

| File | Line | Issue |
|------|------|-------|
| `cli/calibrate.py` | 122-130 | `yaml.safe_load()` for GCP observations |
| `cli/camera.py` | 274 | `load_gcps_from_yaml()` for validation |
| `cli/gcp.py` | 66 | `load_gcps_from_yaml()` for GPS verification |

**Tasks**:

4. **Update `cli/calibrate.py`**
   - Change `--gcps-file` to `--frame-id`
   - Load annotations from `repo_captured_frame.get_annotations(frame_id)`
   - Look up GCP coordinates from `repo_gcp`

5. **Update `cli/camera.py` validate command**
   - Same approach: use `--frame-id` instead of `--gcps-file`
   - Load annotations and GCPs from repositories

### Phase 3: Delete Legacy Code

6. **Delete `cli/gcp.py` verify command** (GPS verification deprecated)
   - Remove `gcp_app.command("verify")`
   - Can keep the file if other GCP commands exist, or delete entirely

7. **Delete `poc_homography/gcp/verify.py`** (GPS-based verification)
   - This uses GPS lat/lon which we no longer support

8. **Delete `poc_homography/testing/data_generator.py`**
   - Legacy tool replaced by `hom frame capture` + `hom interactive`

9. **Update `cli/test_cmds.py`**
   - Remove `data-generator` command import and registration

10. **Delete `poc_homography/validation/camera_model.py` `load_gcps_from_yaml`**
    - Or refactor to use Annotation entities from repository

### Phase 4: Cleanup & Verification

11. **Update CLI main.py**
    - Remove `gcp_app` if empty after removing verify command
    - Or keep if other GCP commands remain

12. **Run full test suite**
    - Fix any broken tests
    - Update test fixtures to use repositories

13. **Run linting and type checking**
    - `uv run ruff check .`
    - `uv run mypy poc_homography/`

## Data Flow After Refactoring

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / Webapp                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ApplicationContext                            │
│  - repo_camera_config      - repo_gcp                           │
│  - repo_camera_calibration - repo_captured_frame                │
│  - frame_capture_service   - camera_controller()                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    data/ (YAML + images)                         │
│  cameras/          - CameraConfig entities                       │
│  calibrations/     - CameraCalibration entities                  │
│  ground_control_points/ - GroundControlPoint entities            │
│  maps/             - Map entities                                │
│  frames/           - CapturedFrame + Annotations                 │
│    └── {map_id}/{camera}/                                        │
│        ├── {timestamp}.yaml      (frame metadata)                │
│        ├── {timestamp}.jpg       (image)                         │
│        └── {timestamp}_annotations.yaml (GCP observations)       │
└─────────────────────────────────────────────────────────────────┘
```

## File Changes Summary

### New/Modified Files
- `domain/entities/annotation.py` - Add serialization + frame_id
- `infrastructure/repositories/repo_yaml_captured_frame.py` - Add annotation methods
- `cli/frame.py` - Add annotation commands
- `cli/calibrate.py` - Use repositories instead of YAML
- `cli/camera.py` - Use repositories instead of YAML

### Files to Delete
- `poc_homography/testing/data_generator.py`
- `poc_homography/gcp/verify.py`
- `cli/gcp.py` (or just remove verify command)

### Files to Update
- `cli/test_cmds.py` - Remove data-generator command
- `cli/main.py` - Update command registration
- `validation/camera_model.py` - Refactor or delete `load_gcps_from_yaml`

## Key Principles

1. **No direct YAML loading in CLI/webapp** - Always go through repositories
2. **Repositories are the single source of truth** - All entity access via ApplicationContext
3. **Annotations belong to frames** - Store per-frame, not in separate global files
4. **GCP definitions vs observations** - GCPs are in `repo_gcp`, observations are Annotations on frames

## Commit Strategy

1. First commit: Annotation entity updates + repository changes
2. Second commit: CLI command updates (calibrate, camera)
3. Third commit: Delete legacy files
4. Fourth commit: Test fixes and cleanup
