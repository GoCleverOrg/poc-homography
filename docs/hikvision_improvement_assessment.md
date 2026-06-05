# Hikvision Integration — Improvement Assessment

This review documents the problems in the pre-existing Hikvision integration
and records which were fixed in this issue (CF#256) versus deferred as
follow-ups. Findings are grounded in the code as it stood before this work and
in the live probe of cam-04 (DS-2DF8425IX-AELW V5.8.0); see the
[capability matrix](./hikvision_isapi_capability_matrix.md).

## 1. Four parallel implementations

The codebase carried four independent Hikvision client implementations, each
re-deriving ISAPI paths, the XML namespace, and the ÷10 / ×10 unit math:

| # | Implementation | Role |
|---|----------------|------|
| 1 | `ptz_discovery_and_control/hikvision/hikvision_ptz_discovery.py::HikvisionPTZ` | Standalone discovery/control script. |
| 2 | `webapp/camera_survey/ptz.py::HikvisionPTZCamera` | Django survey PTZ control. |
| 3 | `poc_homography/infrastructure/clients/hikvision_camera_controller.py::HikvisionCameraController` | DDD adapter for the controller protocol. |
| 4 | `poc_homography/camera/intrinsics.py::get_ptz_status` | Inline status read for intrinsics. |

Each copy embedded its own `/ISAPI/...` literals, its own
`http://www.hikvision.com/ver20/XMLSchema` namespace string, and its own
`/ 10` decode. Divergence between copies is exactly how the wrong tilt range
(below) persisted in one place while the correct convention lived in another.

**Consolidation.** A single adapter,
`poc_homography/infrastructure/clients/hikvision/isapi_client.py::HikvisionISAPIClient`,
implements the `CameraDevice` protocol. All paths come from
`isapi_endpoints.py`, all scaling from `isapi_units.py`, and the namespace is
defined exactly once. The four call sites are re-pointed at this adapter.

## 2. Wrong tilt default (−90 / +90)

The survey code assumed a tilt range of **−90° to +90°**. The live hardware
range is **−50° to +60°** (from
[`absoluteEx/capabilities`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt)).

The wrong default lived in exactly these places:

- `webapp/camera_survey/models.py`, on the `CameraCapabilities` dataclass
  fields: `tilt_min: float = -90.0` and `tilt_max: float = 90.0`.
- `webapp/camera_survey/ptz.py`, in `HikvisionPTZCamera.__init__` /
  `get_capabilities`, where the `-90.0 / 90.0` literals were used as the
  fallback when a queried value was `None`.

A ±90 range silently accepts tilt commands the dome cannot reach (a requested
−80° or +75°), producing out-of-range moves and invalid pose assumptions. The
real range is asymmetric and narrower; only an authoritative,
hardware-sourced `CameraCapabilities` (from `absoluteEx`) avoids this.

**Fix.** The DDD `CameraCapabilities` VO is sourced from
`from_absolute_ex_element` and carries no ±90 defaults. The webapp copy and
its fallbacks are removed.

## 3. Capabilities discarded at survey time

`webapp/camera_survey/services.py` queried `ptz_camera.get_capabilities()`
solely to call `capabilities.validate_range(...)` before a sweep, then
**dropped** the result. The real, hardware-sourced ranges were never persisted
into the survey manifest, so downstream consumers had no record of the actual
tilt/zoom/speed envelope the data was collected under.

**Fix direction.** Persist the queried `CameraCapabilities` into the survey
manifest so each session records the envelope it was captured with.

## 4. Missing per-frame and per-session metadata

Two classes of available data were never captured:

- **Per-frame optics.** Focus, iris, and exposure are all readable
  (`Image/channels/1/{focusConfiguration,iris,exposure}`) but the survey
  capture loop recorded none of them alongside each frame. Without focus/iris/
  exposure per frame, captured imagery cannot be correlated with lens state.
- **Per-session health and odometry.** `System/status` exposes device health
  (uptime, CPU, memory, fan/heat state, reboot count) and **lens odometry**
  (zoom/focus/iris step counters, pan/tilt total rounds). None of this was
  recorded, so there was no session-level record of device condition or lens
  wear.

**Fix direction.** Extend the capture record with optional focus/iris/exposure
fields and capture `DeviceHealth` + `LensOdometry` + `StreamProfile` once per
session.

## 5. RTSP-decode vs. ISAPI `/picture` snapshot

The survey captured frames by opening an RTSP stream
(`webapp/camera_survey/services.py::_capture_frame_from_rtsp`) and decoding
with OpenCV. The trade-off:

| Path | Cost | Notes |
|------|------|-------|
| RTSP decode | Opens a session, waits for a keyframe, decodes a GOP | Higher latency and failure surface; needs FFmpeg and connection/read timeouts. |
| `GET /ISAPI/Streaming/channels/101/picture` | Single HTTP GET → JPEG bytes | On cam-04: 2560×1440 JPEG, ~222 KB; no session setup, no decode. |

The ISAPI `/picture` endpoint yields the same full-resolution frame at lower
latency and with a smaller failure surface.

**Fix direction.** Make `capture_snapshot()` (ISAPI `/picture`) the primary
single-frame path and keep RTSP decode as a fallback for cameras or scenarios
where `/picture` is unavailable.

## 6. Credential handling

Credential plumbing was duplicated across call sites. The project already has
the building blocks — `get_tenant_credentials` and the `Credential` value
object (`poc_homography/domain/vo/credential.py`) — but not every consumer
routed through them.

**Fix direction.** Unify credential acquisition through
`get_tenant_credentials` and the `Credential` VO, and construct the adapter
via `HikvisionISAPIClient.from_config(camera_config)` so username/password flow
from one source.

## Applied in this issue (CF#256)

- Single `HikvisionISAPIClient` adapter implementing the `CameraDevice`
  protocol; one home each for ISAPI paths (`isapi_endpoints.py`), unit math
  (`isapi_units.py`), and the XML namespace.
- DDD `CameraCapabilities` VO sourced from `absoluteEx/capabilities`, with the
  correct **−50 / +60** tilt range and **no** ±90 default.
- New VOs for the full readable surface: `DeviceInfo`, `DeviceHealth` +
  `LensOdometry`, `ImageOptics` (focus/iris/exposure/white balance),
  `StreamProfile`, `CameraPreset`, and `PTZState.focus`.
- `403 methodNotAllowed` modeled as `HikvisionUnsupportedError` (a setter
  signal, not access-denied), so optics/PTZ writes degrade predictably.
- `capture_snapshot()` via ISAPI `/picture`.
- This documentation set, grounded in the committed cam-04 fixtures.

## Deferred as follow-ups

- Persisting `CameraCapabilities` into the survey manifest (item 3).
- Per-frame focus/iris/exposure capture and per-session
  `DeviceHealth`/`LensOdometry`/`StreamProfile` capture (item 4).
- Switching the survey primary capture path to `capture_snapshot()` with RTSP
  fallback (item 5).
- Completing credential routing through `get_tenant_credentials` /
  `Credential` at every webapp and API call site (item 6).
- Retiring the legacy `HikvisionPTZ` discovery module and migrating the
  script-style tests that import it.
</content>
