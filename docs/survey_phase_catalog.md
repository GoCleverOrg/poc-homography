# Survey Phase Catalog

A survey run executes up to nine phases. Phases are identified by a 1..9 number
and by a stable string `.value` from the `SurveyPhase` enum
(`poc_homography/domain/enums/survey_phase.py`). The numeric order matches the
enum declaration order, so phase number N maps to the Nth enum member.

| # | Enum member | `.value` |
|---|-------------|----------|
| 1 | `CAMERA_INVENTORY` | `camera_inventory` |
| 2 | `PTZ_CHARACTERIZATION` | `ptz_characterization` |
| 3 | `ZOOM_CHARACTERIZATION` | `zoom_characterization` |
| 4 | `DENSE_NADIR` | `dense_nadir` |
| 5 | `MAIN_SURVEY` | `main_survey` |
| 6 | `CROSS_ZOOM` | `cross_zoom` |
| 7 | `REPEATABILITY` | `repeatability` |
| 8 | `STATIC_JITTER` | `static_jitter` |
| 9 | `VALIDATION` | `validation` |

Which phases run is controlled by `SurveyPlanConfig.enabled_phases` (a subset of
`{1..9}`; defaults to all nine). Every phase below names the
`SurveyPlanConfig` fields it consumes (source:
`poc_homography/domain/vo/survey_plan_config.py`). Where a phase reads a
`phase_*_range` dict for a phase that is absent from the dict, the planner falls
back to the live camera capabilities at plan time.

The frame-count formulas are estimates of the dominant term; actual counts also
depend on the geometric pose generators and camera capabilities. Unless noted,
phases emit `FrameRecord`s.

---

## Phase 1 — Camera inventory (`camera_inventory`)

**Purpose.** Enumerate the cameras in the run and their capabilities. This is
the entry phase that establishes camera identity and the operating envelope
used by later phases.

**Config consumed.** `enabled_phases` (whether the phase runs). No geometric
range fields are consumed; identity and capabilities are read from the live
camera (or the persisted capabilities sidecar).

**Output records.** Inventory metadata per camera (camera identity and
capabilities); minimal or no `FrameRecord` capture.

**Estimated frame count.** `~0` capture frames per camera (inventory is
metadata-only).

---

## Phase 2 — PTZ characterization (`ptz_characterization`)

**Purpose.** Characterize pan/tilt mechanical behaviour, including settling and
repeatable positioning.

**Config consumed.** `phase_pan_range[2]`, `phase_tilt_range[2]` for the swept
bounds; `burst_frame_count[2]` for snapshot-burst depth; `repeat_count[2]`
(default `3`) for repetitions.

**Output records.** `FrameRecord`s, with `MovementContext.direction_*` and
settling fields populated.

**Estimated frame count.** `poses x burst_frame_count[2] x repeat_count[2]`,
where `poses` is the number of pan/tilt sample positions across the configured
ranges.

---

## Phase 3 — Zoom characterization (`zoom_characterization`)

**Purpose.** Characterize zoom/optics behaviour across the lens range
(reported zoom, focal length, focus).

**Config consumed.** `zoom_levels` (default `[1.0, 5.0, 12.0, 25.0]`),
`phase_zoom_range[3]` for bounding, and `burst_frame_count[3]` for burst depth.
`SurveyContext.approach_direction` is populated here.

**Output records.** `FrameRecord`s spanning the configured zoom levels.

**Estimated frame count.** `len(zoom_levels) x burst_frame_count[3]` per camera.

---

## Phase 4 — Dense nadir (`dense_nadir`)

**Purpose.** Dense, near-nadir-pointing capture used for high-overlap ground
coverage.

**Config consumed.** `phase_pan_range[4]`, `phase_tilt_range[4]`,
`phase_zoom_range[4]`, and `grid_overlap_pct[4]` (default `80.0`). The overlap
percentage drives the tile spacing of the grid.

**Output records.** `FrameRecord`s on a dense overlapping grid.

**Estimated frame count.** `tiles(grid_overlap_pct[4], ranges)` — the tile
count grows as overlap increases (higher overlap = more, more tightly spaced
tiles).

---

## Phase 5 — Main survey (`main_survey`)

**Purpose.** The primary survey sweep that provides the bulk of coverage.

**Config consumed.** `phase_pan_range[5]`, `phase_tilt_range[5]`,
`phase_zoom_range[5]`, and `grid_overlap_pct[5]` (default `80.0`).

**Output records.** `FrameRecord`s on the main coverage grid.

**Estimated frame count.** `tiles(grid_overlap_pct[5], ranges)` — typically the
largest contributor to a run's total frame count.

---

## Phase 6 — Cross zoom (`cross_zoom`)

**Purpose.** Capture the same ground region at multiple zoom levels for
cross-zoom consistency checks.

**Config consumed.** `zoom_levels` (default `[1.0, 5.0, 12.0, 25.0]`) and
`phase_zoom_range[6]`. Frames carry `SurveyContext.region_id` to group
observations of the same region.

**Output records.** `FrameRecord`s grouped by `region_id`, one per zoom level
per region.

**Estimated frame count.** `regions x len(zoom_levels)`.

---

## Phase 7 — Repeatability (`repeatability`)

**Purpose.** Re-visit the same poses repeatedly to quantify positioning
repeatability.

**Config consumed.** `phase_pan_range[7]`, `phase_tilt_range[7]`,
`burst_frame_count[7]`, and `repeat_count[7]` (default `3`). Frames carry
`SurveyContext.sequence_index` (visit index within a repeat group) and
`SurveyContext.approach_direction`; `MovementContext.is_repeatability_sequence`
is `True`.

**Output records.** `FrameRecord`s tagged as repeatability frames.

**Estimated frame count.** `poses x repeat_count[7] x burst_frame_count[7]`.

---

## Phase 8 — Static jitter (`static_jitter`)

**Purpose.** Measure static (no-commanded-movement) jitter by recording video
while the camera holds a fixed pose.

**Jitter spec (verbatim).** Video bursts of 30 s duration at the native FPS
target, sweeping wide/mid/high/max zoom levels, capturing representative poses
per camera.

**Config consumed.**

| Field | Default | Meaning |
|-------|---------|---------|
| `jitter_burst_duration_s` | `30.0` | Burst duration in seconds. |
| `jitter_burst_fps` | `0.0` | Burst frame rate; `0.0` is the sentinel for "use the native FPS target". |
| `jitter_zoom_levels` | `[1.0, 5.0, 12.0, 25.0]` | The wide / mid / high / max zoom levels swept. |
| `jitter_pose_count` | `1` | Representative poses per camera. |

**Output records.** `VideoBurstRecord`s (one encoded RTSP segment per
zoom level per pose), each carrying `FrameRef`s into the per-frame
`FrameRecord`s extracted from the segment.

**Estimated frame count.** Bursts:
`jitter_pose_count x len(jitter_zoom_levels)` segments. Extracted frames per
burst: `effective_fps x jitter_burst_duration_s`, where `effective_fps` is the
camera's native FPS target when `jitter_burst_fps == 0.0`, otherwise
`jitter_burst_fps`. Total extracted frames:
`jitter_pose_count x len(jitter_zoom_levels) x effective_fps x jitter_burst_duration_s`.

---

## Phase 9 — Validation (`validation`)

**Purpose.** Validation / repeatability re-check, holding out a fraction of
poses to verify the dataset and the planner's geometric assumptions.

**Config consumed.** `holdout_fraction` (default `0.15`) — the fraction of
poses reserved as a validation holdout. Pan/tilt/zoom ranges fall back to the
same envelope used by the survey phases.

**Output records.** `FrameRecord`s captured at the held-out validation poses.

**Estimated frame count.** `round(holdout_fraction x survey_poses)`, where
`survey_poses` is the number of poses the main survey phases produced.
