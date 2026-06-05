# Survey Run Guide

This guide walks an operator through launching, monitoring, and aborting a
multi-camera survey run, both from the `hom survey` CLI and over REST. It
assumes familiarity with PTZ cameras; for the phase semantics see
[survey_phase_catalog.md](./survey_phase_catalog.md) and for the resulting
dataset see [survey_dataset_schema.md](./survey_dataset_schema.md).

A run is driven by a **plan config** — a camera-free YAML sidecar that captures
every knob the survey planner needs, so a run can be replayed deterministically.
The YAML is loaded into a `SurveyPlanConfig`
(`poc_homography/domain/vo/survey_plan_config.py`).

---

## 1. Author a `plan_config.yaml`

Every field below is annotated with its purpose and its source default. The
values shown are exactly the `SurveyPlanConfig` defaults; `from_dict()`
tolerates missing keys (they fall back to these defaults), so you only need to
include the fields you want to override.

```yaml
# plan_config.yaml — reproducible nine-phase survey plan.

# Sidecar schema version. Must be the literal "1"; from_dict() rejects anything
# else with a ValueError. Distinct from the dataset SURVEY_SCHEMA_VERSION.
schema_version: "1"

# Subset of phases (1..9) to execute. Default: all nine.
enabled_phases: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Per-phase pan bounds (lo, hi) in degrees. Keys are phase numbers. Phases
# absent here fall back to the live camera capabilities at plan time.
# Default: {} (empty -> all phases use camera capabilities).
phase_pan_range: {}

# Per-phase tilt bounds (lo, hi) in degrees. Default: {}.
phase_tilt_range: {}

# Per-phase zoom-factor bounds (lo, hi). Default: {}.
phase_zoom_range: {}

# Per-phase tile overlap percentage (used by the grid phases 4 and 5).
# Default: {4: 80.0, 5: 80.0}.
grid_overlap_pct:
  "4": 80.0
  "5": 80.0

# Per-phase snapshot-burst frame count (phases 2, 3, 7). Default: {} (the
# planner uses its built-in per-phase burst depth where unset).
burst_frame_count: {}

# Phase 8 (static jitter) settings.
# Burst duration in seconds. Default: 30.0.
jitter_burst_duration_s: 30.0
# Burst frame rate. 0.0 is the sentinel meaning "use the native FPS target".
# Default: 0.0.
jitter_burst_fps: 0.0
# Phase 8 zoom factors: wide / mid / high / max. Default: [1.0, 5.0, 12.0, 25.0].
jitter_zoom_levels: [1.0, 5.0, 12.0, 25.0]
# Phase 8 representative poses per camera. Default: 1.
jitter_pose_count: 1

# Zoom factors swept by phases 3, 6 and 8. Default: [1.0, 5.0, 12.0, 25.0].
zoom_levels: [1.0, 5.0, 12.0, 25.0]

# Per-phase repetition count (phases 2 and 7). Default: {2: 3, 7: 3}.
repeat_count:
  "2": 3
  "7": 3

# Phase 9 validation holdout fraction. Default: 0.15.
holdout_fraction: 0.15
```

Notes on serialization:

- Int-keyed maps (`grid_overlap_pct`, `burst_frame_count`, `repeat_count`, the
  `phase_*_range` dicts) use **string keys** in YAML; `from_dict()` coerces them
  back to ints.
- `phase_*_range` values are 2-element lists `[lo, hi]` that become tuples.
- `enabled_phases` is a list on disk and a `frozenset` in memory.

---

## 2. Launch a run

```bash
hom survey run --plan plan_config.yaml --cameras cam-04,cam-07
```

`--plan` is the path to the YAML plan above; `--cameras` is a comma-separated
list of camera ids. The command reads the YAML, builds a `SurveyPlanConfig`,
starts a run for the given cameras, and prints the run id and per-camera session
ids, then streams progress:

```
run_id: <run_id>
  cam-04: <session_id>
  cam-07: <session_id>
[cam-04] phase=main_survey frames=128 status=running
[cam-07] phase=main_survey frames=131 status=running
...
```

If the plan fails to load (bad path, invalid YAML, or an unsupported
`schema_version`) the command exits non-zero with an error on stderr. Providing
no valid camera ids also exits non-zero.

---

## 3. Monitor a run

```bash
hom survey status --run-id <run_id>
```

Prints per-camera status for the run:

```
run_id: <run_id>
  cam-04: phase=repeatability frames=842 status=running
  cam-07: phase=repeatability frames=851 status=running
```

An unknown `--run-id` exits non-zero with `Error: Unknown run_id: ...`.

To list recent runs as a table:

```bash
hom survey list --limit 20
```

---

## 4. Abort a run

```bash
hom survey abort --run-id <run_id>
```

Requests a **graceful** abort and prints the confirmation message
(`Run abort requested`). An unknown run id exits non-zero.

---

## 5. Equivalent REST calls

The same operator surface is exposed under the `/camera-evaluation` router. All
endpoints require **HTTP Basic** authentication (the API uses
`HTTPBasic`; on a failed login it returns `401` with a
`WWW-Authenticate: Basic` header). Pass credentials with `curl -u user:pass` or
an explicit `Authorization: Basic <base64>` header. Examples below use
`http://localhost:8000` as the base URL.

### Start a run

`POST /camera-evaluation/survey/run/start/`

Body matches `SurveyRunStartRequest`: a `plan_config` object (the same fields as
`plan_config.yaml`) plus a `camera_ids` list.

```bash
curl -u "$USER:$PASS" \
  -X POST http://localhost:8000/camera-evaluation/survey/run/start/ \
  -H 'Content-Type: application/json' \
  -d '{
    "plan_config": {
      "schema_version": "1",
      "enabled_phases": [1, 2, 3, 4, 5, 6, 7, 8, 9],
      "jitter_burst_duration_s": 30.0,
      "jitter_zoom_levels": [1.0, 5.0, 12.0, 25.0],
      "holdout_fraction": 0.15
    },
    "camera_ids": ["cam-04", "cam-07"]
  }'
```

Returns the `run_id` and a `session_ids` map of `{camera_id: session_id}`.

### Run status

`GET /camera-evaluation/survey/run/{run_id}/status/`

```bash
curl -u "$USER:$PASS" \
  http://localhost:8000/camera-evaluation/survey/run/<run_id>/status/
```

Returns `{"run_id", "cameras": {camera_id: {session_id, phase, frame_count, status}}}`.

### Abort a run

`POST /camera-evaluation/survey/run/{run_id}/abort/`

```bash
curl -u "$USER:$PASS" \
  -X POST http://localhost:8000/camera-evaluation/survey/run/<run_id>/abort/
```

Returns `{"run_id", "message": "Run abort requested"}`.

### List runs

`GET /camera-evaluation/survey/runs/`

```bash
curl -u "$USER:$PASS" \
  http://localhost:8000/camera-evaluation/survey/runs/
```

Returns run summaries (newest first): `run_id`, `start_time`, `camera_count`,
`total_frame_count`, `status`.

### Browse dataset groupings

`GET /camera-evaluation/survey/runs/{run_id}/groups/`

```bash
curl -u "$USER:$PASS" \
  http://localhost:8000/camera-evaluation/survey/runs/<run_id>/groups/
```

Returns the `(phase, camera, zoom, frame_count)` groupings for the run. See
[survey_offline_reprocessing.md](./survey_offline_reprocessing.md) for filtering
and offline use.
