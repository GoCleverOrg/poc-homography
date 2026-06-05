# Survey Dataset Schema (C1)

This is the field-by-field reference for the multi-phase, multi-camera survey
dataset. The C1 domain entities are the **single source of truth** for the
schema: every capture, planning, and phase component emits into and queries
from these entities. The schema version stamped onto each record is the
constant `SURVEY_SCHEMA_VERSION = "1.0"`
(`poc_homography/domain/entities/survey/__init__.py`).

Each entity is a frozen dataclass with symmetric `to_dict()` / `from_dict()`
methods. On load, `from_dict()` calls `check_schema_version()`, which raises
`ValueError` if the serialized `schema_version` does not equal the current
`SURVEY_SCHEMA_VERSION`. That mismatch is the explicit signal that a migration
hook is required before the record can be read.

> Note on versions: the dataset schema version (`SURVEY_SCHEMA_VERSION`,
> `"1.0"`) is deliberately distinct from the plan-config sidecar version
> (`PLAN_CONFIG_SCHEMA_VERSION`, `"1"`). The sidecar
> (`SurveyPlanConfig`, see [survey_phase_catalog.md](./survey_phase_catalog.md))
> evolves independently of the run/frame dataset.

---

## `SurveyRun` — the run aggregate

Source: `poc_homography/domain/entities/survey/survey_run.py`

A single survey run for **one camera** across a set of phases. The `id`
property returns `run_id` (satisfies the `Entity` protocol). A multi-camera
launch produces one `SurveyRun` per camera, all sharing the same logical
run id reported by the operator surface.

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | `str` | Unique run identifier; also the entity `id`. |
| `camera_id` | `str` | The camera this run targets. |
| `phases` | `frozenset[SurveyPhase]` | Phases executed; serialized as a sorted list of stable string `.value`s. |
| `started_at` | `datetime` | Run start; serialized as ISO-8601. |
| `finished_at` | `datetime \| None` | Run end, or `None` while in flight. |
| `status` | `SurveyRunStatus` | Lifecycle status; defaults to `PENDING`. Enables a planner to resume interrupted runs. |
| `schema_version` | `str` | Defaults to `SURVEY_SCHEMA_VERSION` (`"1.0"`). |

The `status` field is what makes a run **resumable**: a planner can inspect a
persisted run and continue from where it stopped. Run manifests are persisted
one file per run (YAML repository at `data/survey_runs/{run_id}.yaml`).

---

## Session

In the multi-camera operator surface, a **session** is the per-camera
execution context of a run. When a run is started for N cameras, the operator
surface returns a `session_ids` map of `{camera_id: session_id}` (see
[survey_run_guide.md](./survey_run_guide.md)). Per-frame records reference the
camera through `CameraIdentity` and the run through `CaptureIdentity.run_id`;
session-scoped image and manifest assets are addressable through the survey
session endpoints (`/camera-evaluation/api/survey/sessions/{session_id}/...`).

---

## `FrameRecord` — the per-frame capture record

Source: `poc_homography/domain/entities/survey/frame_record.py`

`FrameRecord` captures the full optical and mechanical state at a single survey
frame. Its `id` property returns the per-frame `capture_id`. Fields are grouped
into named nested value objects that mirror how they are populated; each nested
VO has its own `to_dict()` / `from_dict()`.

| Field | Type |
|-------|------|
| `camera` | `CameraIdentity` |
| `capture` | `CaptureIdentity` |
| `commanded` | `CommandedState` |
| `reported` | `ReportedState` |
| `movement` | `MovementContext` |
| `pipeline` | `ImagePipelineState` |
| `image_data` | `ImageData` |
| `survey_context` | `SurveyContext` (defaults to an empty `SurveyContext`) |
| `schema_version` | `str` (defaults to `SURVEY_SCHEMA_VERSION`) |

### `CameraIdentity`

Stable identity of the camera that produced the frame.

| Field | Type |
|-------|------|
| `camera_id` | `str` |
| `brand` | `str` |
| `model` | `str` |
| `serial` | `str` |
| `firmware` | `str` |
| `channel_id` | `str` |
| `stream_id` | `str` |

### `CaptureIdentity`

Identity and timing of an individual capture within a run.

| Field | Type | Notes |
|-------|------|-------|
| `capture_id` | `str` | The frame's unique id. |
| `run_id` | `str` | Owning run. |
| `phase` | `SurveyPhase` | Serialized as the phase `.value`. |
| `burst_id` | `str \| None` | Set for frames extracted from a Phase 8 video burst; otherwise `None`. |
| `frame_index` | `int` | Index within the capture/burst. |
| `timestamp_before_move` | `datetime` | Before commanding PTZ movement. |
| `timestamp_after_move` | `datetime` | After movement completed. |
| `timestamp_at_capture` | `datetime` | At image capture. |

### `CommandedState`

The PTZ + focus state commanded to the camera for this frame. Also reused by
`VideoBurstRecord`.

| Field | Type |
|-------|------|
| `commanded_pan` | `Degrees` |
| `commanded_tilt` | `Degrees` |
| `commanded_zoom` | `Unitless` |
| `commanded_focus` | `int \| None` |

### `ReportedState`

The PTZ + optics state reported back by the camera for this frame. The
commanded-vs-reported pairing is what lets offline analysis measure mechanical
error.

| Field | Type |
|-------|------|
| `reported_pan` | `Degrees` |
| `reported_azimuth` | `Degrees \| None` |
| `reported_tilt` | `Degrees` |
| `reported_elevation` | `Degrees \| None` |
| `reported_zoom` | `Unitless` |
| `reported_focal_length_mm` | `Millimeters \| None` |
| `reported_focus` | `int \| None` |
| `ptz_settled` | `bool` |

### `MovementContext`

Movement that preceded this frame and its settling context.

| Field | Type | Notes |
|-------|------|-------|
| `prev_pan` | `Degrees` | Pose before the move. |
| `prev_tilt` | `Degrees` | |
| `prev_zoom` | `Unitless` | |
| `direction_pan` | `"cw" \| "ccw" \| "none"` | Approach direction in pan. |
| `direction_tilt` | `"up" \| "down" \| "none"` | Approach direction in tilt. |
| `direction_zoom` | `"tele" \| "wide" \| "none"` | Approach direction in zoom. |
| `settling_delay_s` | `Seconds` | Settle wait applied before capture. |
| `is_repeatability_sequence` | `bool` | True for Phase 7 repeatability frames. |

### `ImagePipelineState`

Encoder / image-pipeline state, from `StreamProfile` + `ImageOptics`.

| Field | Type |
|-------|------|
| `resolution_width` | `Pixels` |
| `resolution_height` | `Pixels` |
| `codec` | `str` |
| `profile` | `str` |
| `fps` | `FPS` |
| `eis_enabled` | `bool` |
| `eptz_enabled` | `bool` |
| `digital_zoom` | `Unitless` |
| `digital_zoom_limit` | `Unitless` |
| `mirror` | `bool` |
| `flip` | `bool` |
| `corridor_mode` | `bool` |
| `day_night_mode` | `str` |
| `crop_enabled` | `bool` |
| `stabilization_enabled` | `bool` |
| `exposure_mode` | `str` |
| `focus_mode` | `str` |

### `ImageData`

The persisted image file and its integrity / dimension metadata.

| Field | Type |
|-------|------|
| `image_path` | `Path` |
| `checksum` | `str` |
| `width` | `Pixels` |
| `height` | `Pixels` |
| `capture_format` | `str` |

### `SurveyContext`

Planner-derived survey context attached to a frame by the phase layer. These
fields originate in the C3 planner pose (and the C4 phase that drives it), not
in the C2 capture engine. All are optional and default to `None`, so frames
captured outside a phase context round-trip unchanged.

| Field | Type | Notes |
|-------|------|-------|
| `region_id` | `str \| None` | Groups cross-zoom observations of the same ground region (Phase 6). |
| `approach_direction` | `str \| None` | The direction a pose was approached from (Phases 3 / 7). |
| `sequence_index` | `int \| None` | Visit index within a repeat group (Phase 7). |

---

## `VideoBurstRecord` — Phase 8 RTSP segments

Source: `poc_homography/domain/entities/survey/video_burst_record.py`

Preserves the original encoded RTSP segment while making each contained frame
addressable for offline processing. The `id` property returns `burst_id`. The
`phase` field is typed generally but is always `SurveyPhase.STATIC_JITTER`
(Phase 8) in practice.

| Field | Type | Notes |
|-------|------|-------|
| `burst_id` | `str` | Burst id; also the entity `id`. |
| `run_id` | `str` | Owning run. |
| `camera_id` | `str` | Source camera. |
| `phase` | `SurveyPhase` | Always `STATIC_JITTER` in practice. |
| `segment_path` | `Path` | The encoded RTSP segment on disk. |
| `duration_s` | `Seconds` | Burst duration. |
| `fps` | `FPS` | Effective frame rate of the segment. |
| `codec` | `str` | Segment codec. |
| `commanded_state` | `CommandedState` | The pose held during the burst. |
| `frame_refs` | `tuple[FrameRef, ...]` | Per-frame pointers (see below). |

Helper: `frame_by_index(frame_index)` returns the `FrameRef` with that
`frame_index`, or `None`.

### `FrameRef`

A lightweight pointer to one frame within a video burst. The full
`FrameRecord` for each frame is stored separately under the frame layout.

| Field | Type |
|-------|------|
| `capture_id` | `str` |
| `frame_index` | `int` |
| `timestamp_at_capture` | `datetime` |
| `image_path` | `Path` |

---

## Grouping index fields

For browsing, frames are grouped by the tuple **`(phase, camera, zoom)`** and
counted. The grouping projection (the operator-surface `browse` view, see
[survey_offline_reprocessing.md](./survey_offline_reprocessing.md)) exposes:

| Group field | Source |
|-------------|--------|
| `phase` | `FrameRecord.capture.phase.value` (the stable phase string). |
| `camera` | `FrameRecord.camera.camera_id`. |
| `zoom` | `FrameRecord.reported.reported_zoom`, rounded to one decimal place. |
| `frame_count` | Count of frames in the group. |

The `zoom` filter matches a reported zoom factor to one decimal place, and the
`phase` filter accepts a 1..9 phase number that is mapped to the corresponding
`SurveyPhase` member before matching.
