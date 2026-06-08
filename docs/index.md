# Documentation Index

Reference documentation for the PTZ homography project. The pages below are
grouped by topic: the multi-phase **Survey** system, the **Hikvision / PTZ**
ISAPI reference, **Camera intrinsics and pose**, and the **Domain model**.

## Survey

- [survey_run_guide.md](./survey_run_guide.md) — step-by-step operator guide:
  author a `plan_config.yaml`, launch with `hom survey run`, monitor with
  `hom survey status`, abort with `hom survey abort`, and the equivalent
  `curl` REST calls.
- [survey_phase_catalog.md](./survey_phase_catalog.md) — one section per phase
  (1–9): purpose, the `SurveyPlanConfig` fields it consumes, expected output
  record types, and a frame-count estimate. Includes the Phase 8 jitter spec.
- [survey_dataset_schema.md](./survey_dataset_schema.md) — field-by-field
  reference for the C1 dataset schema: `SurveyRun`, `FrameRecord` and its nested
  value objects, `VideoBurstRecord`, and the grouping index fields.
- [survey_offline_reprocessing.md](./survey_offline_reprocessing.md) — using a
  dataset without a camera: `dvc pull`, reload the plan config, browse
  groupings, and access images and video bursts.
- [cleanplate_reconstruction_methods.md](./cleanplate_reconstruction_methods.md)
  — method comparison for offline clean-plate (empty-floor orthophoto)
  reconstruction: mask-aware temporal median, RPCA, inpainting, and multi-band
  blending, with a stated recommendation and the `hom cleanplate reconstruct`
  pipeline.

## Hikvision / PTZ

- [HIKVISION_PTZ_API_SUMMARY.md](./HIKVISION_PTZ_API_SUMMARY.md) — ISAPI
  endpoint catalog grounded in a live probe of cam-04: paths, methods, returns,
  sample values, unit conventions, and HTTP status semantics.
- [hikvision_isapi_capability_matrix.md](./hikvision_isapi_capability_matrix.md)
  — per-endpoint live-hardware report: readable, PUT-only setter, or absent,
  keyed to the committed fixtures.
- [hikvision_improvement_assessment.md](./hikvision_improvement_assessment.md)
  — review of prior duplication, the tilt default, and applied-vs-deferred
  improvements.
- [ptz_commands.md](./ptz_commands.md) — notes on PTZ command helpers.

## Camera intrinsics and pose

- [ptz_intrinsics_and_pose.md](./ptz_intrinsics_and_pose.md) — computing the
  intrinsic matrix from zoom and recovering camera position and orientation.
- [lens_calibration_requirements.md](./lens_calibration_requirements.md) —
  minimum data requirements for lens-distortion calibration.

## Domain model

- [domain-model-assessment.md](./domain-model-assessment.md) — assessment of
  entities, value objects, and legacy types (issue #172 DDD refactoring).
- [domain-model-refactoring.md](./domain-model-refactoring.md) — domain model
  and refactoring plan for the homography system.

## Other

- [README.md](./README.md) — the prior documentation index page.
