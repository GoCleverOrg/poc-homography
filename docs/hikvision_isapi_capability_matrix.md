# Hikvision ISAPI Capability Matrix — DS-2DF8425IX-AELW V5.8.0

This is a live-hardware report for camera `cam-04` (`icozee-camptz-04`), a
**DS-2DF8425IX-AELW** on firmware **V5.8.0**. It records, per endpoint, what
the device actually does when probed. The full endpoint semantics are
documented in the [endpoint catalog](./HIKVISION_PTZ_API_SUMMARY.md).

## Legend

| Status | Meaning |
|--------|---------|
| **readable** | `GET` returns `200` with a parseable body. A committed fixture exists under [`tests/fixtures/hikvision/`](../tests/fixtures/hikvision/). |
| **PUT-only setter (403 methodNotAllowed)** | Endpoint exists; `GET` (or an unsupported method) returns `403` with `statusString="Invalid Operation"`, `subStatusCode="methodNotAllowed"`. **Not** access-denied. The adapter raises `HikvisionUnsupportedError`. |
| **absent (404)** | Endpoint not implemented on this model/firmware. |

The "readable" determination is grounded in the presence of a captured
fixture; the 403 determination is grounded in
[`ISAPI__403__methodNotAllowed.txt`](../tests/fixtures/hikvision/ISAPI__403__methodNotAllowed.txt).

## Readable endpoints (200, fixture present)

| Endpoint | Method | Status | Fixture |
|----------|--------|--------|---------|
| `/ISAPI/System/deviceInfo` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__System__deviceInfo.txt) |
| `/ISAPI/System/status` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__System__status.txt) |
| `/ISAPI/PTZCtrl/channels/1/status` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__status.txt) |
| `/ISAPI/PTZCtrl/channels/1/absoluteEx/capabilities` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt) |
| `/ISAPI/PTZCtrl/channels/1/capabilities` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__capabilities.txt) |
| `/ISAPI/PTZCtrl/channels/1/presets` (36 presets) | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__presets.txt) |
| `/ISAPI/Image/channels/1/focusConfiguration` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Image__channels__1__focusConfiguration.txt) |
| `/ISAPI/Image/channels/1/iris` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Image__channels__1__iris.txt) |
| `/ISAPI/Image/channels/1/exposure` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Image__channels__1__exposure.txt) |
| `/ISAPI/Image/channels/1/whiteBalance` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Image__channels__1__whiteBalance.txt) |
| `/ISAPI/Image/channels/1/capabilities` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Image__channels__1__capabilities.txt) |
| `/ISAPI/Streaming/channels/101` (H.264 2560×1440@25) | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Streaming__channels__101.txt) |
| `/ISAPI/Streaming/channels/102` | GET | readable | [link](../tests/fixtures/hikvision/ISAPI__Streaming__channels__102.txt) |
| `/ISAPI/Streaming/channels/101/picture` (JPEG 2560×1440, ~222 KB) | GET | readable | [snapshot.jpg](../tests/fixtures/hikvision/snapshot.jpg) |

## PUT-only setters and write endpoints

These endpoints accept `PUT`. A `GET` (or otherwise unsupported method)
returns `403 methodNotAllowed`.

| Endpoint | Method | Status |
|----------|--------|--------|
| `/ISAPI/System/Video/inputs/channels/1/focus` | PUT | PUT-only setter (403 methodNotAllowed) — confirmed by the 403 fixture (`requestURL` is exactly this path) |
| `/ISAPI/PTZCtrl/channels/1/absolute` | PUT | setter |
| `/ISAPI/PTZCtrl/channels/1/relative` | PUT | setter |
| `/ISAPI/PTZCtrl/channels/1/continuous` | PUT | setter |
| `/ISAPI/PTZCtrl/channels/1/momentary` | PUT | setter |
| `/ISAPI/PTZCtrl/channels/1/position3D` | PUT | setter (`isSupportPosition3D=true`) |
| `/ISAPI/PTZCtrl/channels/1/presets/{id}/goto` | PUT | setter |
| `/ISAPI/PTZCtrl/channels/1/homeposition/goto` | PUT | setter |

> The single live 403 capture
> ([`ISAPI__403__methodNotAllowed.txt`](../tests/fixtures/hikvision/ISAPI__403__methodNotAllowed.txt))
> targets `/ISAPI/System/Video/inputs/channels/1/focus`. The remaining
> setters are PUT-shaped by ISAPI contract; the adapter maps any
> `403 methodNotAllowed` they emit to `HikvisionUnsupportedError`.

## Absent fields (no hardware source anywhere)

| Field | Status | Consequence |
|-------|--------|-------------|
| Focal length (mm) | **absent (404)** — no ISAPI endpoint or tag reports it | Must be **computed**, never read. |
| Field of view (degrees) | **absent (404)** — no ISAPI endpoint or tag reports it | Must be **computed**, never read. |

Neither focal-length-mm nor FOV appears in any probed document: not in
`deviceInfo`, not in `PTZCtrl` capabilities, not in `Image` capabilities, not
in the streaming configuration. They are **derived** from the live zoom factor
and the datasheet-sourced sensor model in
`poc_homography/camera/intrinsics.py::compute_intrinsics` and
`poc_homography/domain/enums/camera_spec.py::focal_length_at_zoom`. See
[`ptz_intrinsics_and_pose.md`](./ptz_intrinsics_and_pose.md).

## Confirmed live ranges (cam-04)

From [`ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt):

| Axis | Min | Max | Unit |
|------|-----|-----|------|
| Pan (azimuth) | 0 | 360 | degrees |
| Tilt (elevation) | **−50** | **+60** | degrees |
| Zoom (absoluteZoom) | 1 | 25 | × |
| Focus | 4096 | 2576990208 | steps |
| Horizontal speed | 0.2 | 210.8 | °/s |
| Vertical speed | 0.2 | 151.8 | °/s |

The real tilt range is **−50 to +60 degrees**, not the ±90 default that the
legacy webapp code assumed (see
[`hikvision_improvement_assessment.md`](./hikvision_improvement_assessment.md)).
</content>
