# Documentation Index

Reference documentation for the PTZ homography project.

## Hikvision ISAPI

- [HIKVISION_PTZ_API_SUMMARY.md](./HIKVISION_PTZ_API_SUMMARY.md) — ISAPI
  endpoint catalog grounded in a live probe of cam-04 (DS-2DF8425IX-AELW
  V5.8.0): paths, methods, returns, sample values, unit conventions, and HTTP
  status semantics.
- [hikvision_isapi_capability_matrix.md](./hikvision_isapi_capability_matrix.md)
  — per-endpoint live-hardware report: readable, PUT-only setter (403
  methodNotAllowed), or absent (404), keyed to the committed fixtures.
- [hikvision_improvement_assessment.md](./hikvision_improvement_assessment.md)
  — review of the prior four-implementation duplication, the wrong tilt
  default, and the applied-vs-deferred improvements.

## Camera intrinsics and pose

- [ptz_intrinsics_and_pose.md](./ptz_intrinsics_and_pose.md) — computing the
  intrinsic matrix from zoom and recovering camera position and orientation;
  notes which values are hardware-reported vs. computed.
- [lens_calibration_requirements.md](./lens_calibration_requirements.md) —
  minimum data requirements for lens-distortion calibration.
- [ptz_commands.md](./ptz_commands.md) — notes on PTZ command helpers.

## Domain model

- [domain-model-assessment.md](./domain-model-assessment.md) — assessment of
  entities, value objects, and legacy types (issue #172 DDD refactoring).
- [domain-model-refactoring.md](./domain-model-refactoring.md) — domain model
  and refactoring plan for the homography system.

## Probe fixtures

The raw cam-04 probe responses referenced throughout these docs live under
[`../tests/fixtures/hikvision/`](../tests/fixtures/hikvision/).
</content>
