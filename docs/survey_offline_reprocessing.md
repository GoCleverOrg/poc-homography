# Survey Offline Reprocessing

A survey dataset is designed to be reprocessed **without a camera**. The plan
config is a camera-free sidecar, and every per-frame record carries the full
commanded-vs-reported optical and mechanical state (see
[survey_dataset_schema.md](./survey_dataset_schema.md)). This page describes how
to pull a dataset, reload its plan, browse its groupings, and fetch its image
and video-burst assets.

---

## 1. Pull the dataset

Datasets are versioned with DVC. Pull the tracked data into your working tree:

```bash
dvc pull
```

This materializes the run manifests, per-frame records, captured images, and
Phase 8 video-burst segments referenced by the DVC-tracked pointers.

---

## 2. Reload the plan config

The plan config is reproducible from its YAML sidecar. Load it back into a
`SurveyPlanConfig` value object; `from_dict()` tolerates partial payloads
(missing keys fall back to defaults) and rejects an unsupported
`schema_version` with a `ValueError`.

```python
import yaml

from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig

with open("plan_config.yaml") as f:
    cfg = SurveyPlanConfig.from_dict(yaml.safe_load(f))

print(sorted(cfg.enabled_phases))      # e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(cfg.jitter_zoom_levels)          # e.g. [1.0, 5.0, 12.0, 25.0]
print(cfg.holdout_fraction)            # e.g. 0.15
```

A full `to_dict()` payload round-trips back to an equal object, so the reloaded
config exactly reproduces the planning inputs of the original run.

---

## 3. Browse groupings

Frames are grouped by the tuple `(phase, camera, zoom)` and counted. Browse from
the CLI:

```bash
hom survey browse --run-id <run_id>
hom survey browse --run-id <run_id> --phase 5
hom survey browse --run-id <run_id> --camera cam-04
hom survey browse --run-id <run_id> --zoom 12.0
```

- `--phase` is a 1..9 phase number (mapped to the corresponding `SurveyPhase`).
- `--camera` is a camera id.
- `--zoom` is a reported zoom factor, matched to one decimal place.

The output is a table of `phase`, `camera`, `zoom`, `frame_count`. When the run
has no backing repository wired, no groupings are returned.

The REST equivalent is:

`GET /camera-evaluation/survey/runs/<run_id>/groups/`

```bash
curl -u "$USER:$PASS" \
  "http://localhost:8000/camera-evaluation/survey/runs/<run_id>/groups/?phase=5&camera=cam-04&zoom=12.0"
```

All three query parameters are optional and combine as filters.

---

## 4. Access images

Images persisted for a session are served by the existing survey image
endpoint:

`GET /camera-evaluation/api/survey/sessions/{session_id}/images/{filename}`

```bash
curl -u "$USER:$PASS" \
  http://localhost:8000/camera-evaluation/api/survey/sessions/<session_id>/images/<filename> \
  -o frame.jpg
```

The `session_id` is the per-camera session id returned when the run was started
(see [survey_run_guide.md](./survey_run_guide.md)); the `filename` corresponds to
the image file name under the session's image layout. Requires HTTP Basic auth.

---

## 5. Access video bursts

Phase 8 (static jitter) produces `VideoBurstRecord`s, each pointing at an
encoded RTSP segment (`segment_path`) with `FrameRef`s into the extracted
per-frame records. The burst-serve endpoint is the streaming equivalent of the
image endpoint above:

`GET /camera-evaluation/api/survey/sessions/{session_id}/bursts/{filename}`

> Status: the burst-serve endpoint is **to be added by C2**. Until then, access
> the segment directly from `VideoBurstRecord.segment_path` after `dvc pull`,
> for example:

```python
from poc_homography.domain.entities.survey.video_burst_record import VideoBurstRecord

burst = VideoBurstRecord.from_dict(burst_payload)
print(burst.segment_path)              # path to the encoded RTSP segment
ref = burst.frame_by_index(0)          # FrameRef for frame 0, or None
print(ref.image_path)                  # extracted frame image on disk
```

Each `FrameRef` exposes `capture_id`, `frame_index`, `timestamp_at_capture`, and
`image_path`, so individual frames within a burst are directly addressable for
offline jitter analysis.
