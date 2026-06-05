# Hikvision ISAPI Endpoint Catalog

This document catalogs the Hikvision ISAPI surface used by the adapter
(`poc_homography/infrastructure/clients/hikvision/`). Every endpoint, sample
value, and range below is grounded in a live probe of camera `cam-04`
(`icozee-camptz-04`), a **DS-2DF8425IX-AELW** running firmware **V5.8.0**.

The raw probe responses are committed as fixtures under
[`tests/fixtures/hikvision/`](../tests/fixtures/hikvision/) and are referenced
per-endpoint below. The companion
[capability matrix](./hikvision_isapi_capability_matrix.md) records, per
endpoint, whether the live hardware returns data, rejects writes, or is absent.

## Probed Device

| Field | Value | Source tag |
|-------|-------|------------|
| Device name | `icozee-camptz-04` | `deviceName` |
| Model | `DS-2DF8425IX-AELW` | `model` |
| Device type | `IPDome` | `deviceType` |
| Serial number | `DS-2DF8425IX-AELW20240712CCWRFH3535597` | `serialNumber` |
| MAC address | `74:3f:c2:fc:a1:9b` | `macAddress` |
| Firmware | `V5.8.0` (build 230208) | `firmwareVersion` |
| Encoder | `V7.3` | `encoderVersion` |
| Boot | `V1.3.4` | `bootVersion` |
| Hardware | `0x0` | `hardwareVersion` |
| Platform | `H8` | `platformName` |
| Manufacturer | `hikvision` | `manufacturer` |

Source: [`ISAPI__System__deviceInfo.txt`](../tests/fixtures/hikvision/ISAPI__System__deviceInfo.txt).

## Transport and Conventions

- **Base URL**: `http(s)://<host>/ISAPI/...`. All paths are produced by
  `isapi_endpoints.py`; no other module hardcodes `/ISAPI/` literals.
- **Authentication**: HTTP Digest (`HTTPDigestAuth`).
- **Content type**: XML, namespace
  `http://www.hikvision.com/ver20/XMLSchema`. The namespace string is defined
  once, in `isapi_endpoints.py::HIKVISION_XML_NS`. Parsing uses `defusedxml`.
- **Channels**: PTZ and Image controls default to channel `1`; streaming
  defaults to `101` (main stream).

### Unit conventions

Hikvision encodes pan, tilt, and zoom as integers **scaled by ×10**. The
adapter centralizes this in `isapi_units.py`.

| Quantity | Encoding | Example (from `status`) |
|----------|----------|-------------------------|
| Pan (azimuth) | raw = degrees × 10 | `3202` → `320.2°` |
| Tilt (elevation) | raw = degrees × 10 | `182` → `18.2°` |
| Zoom (absoluteZoom) | raw = factor × 10 | `172` → `17.2×` |

- The decode is `value = raw / 10`; the encode is `raw = round(value * 10)`.
  `round()` keeps the round-trip symmetric for multiples of `0.1`
  (`raw_to_degrees(degrees_to_raw(18.2)) == 18.2`).
- **Raw tilt is positive = down.** A positive elevation in a raw ISAPI
  response means the lens points below the horizon. Downstream pose code
  accounts for this via `TiltConvention.POSITIVE_DOWN` on
  `CameraSpec.HIKVISION_DS_2DF8425IX`.
- The `absoluteEx/capabilities` response is the exception: it reports values
  **already in degrees** (e.g. `<elevation>18.2</elevation>`) and carries the
  authoritative min/max as XML attributes.

### Position-space ranges (cam-04, in engineering units)

| Axis | Min | Max | Unit | Source |
|------|-----|-----|------|--------|
| Pan (azimuth) | 0 | 360 | degrees | `absoluteEx/capabilities` |
| Tilt (elevation) | −50 | +60 | degrees | `absoluteEx/capabilities` |
| Zoom (absoluteZoom) | 1 | 25 | × | `absoluteEx/capabilities` |
| Focus | 4096 | 2576990208 | steps | `absoluteEx/capabilities` |
| Horizontal speed | 0.2 | 210.8 | °/s | `absoluteEx/capabilities` |
| Vertical speed | 0.2 | 151.8 | °/s | `absoluteEx/capabilities` |

> The legacy `PTZCtrl/channels/1/capabilities` response reports the same axes
> in **raw ×10 space** (pan `0..3600`, tilt `-500..600`, zoom `10..250`). The
> degree-based `absoluteEx/capabilities` is the preferred source because it
> needs no scaling and is unambiguous.

### HTTP status semantics

| Status | Meaning for this surface |
|--------|--------------------------|
| `200 OK` | Endpoint is readable; body parsed into a value object. |
| `403 Forbidden` + `subStatusCode == methodNotAllowed` | The endpoint exists but the **method is not allowed** (typically a PUT-only setter probed with GET, or a setter the firmware does not expose). This is **not** an authentication or access-denied failure. The adapter maps it to `HikvisionUnsupportedError`. |
| `404 Not Found` | The endpoint is **absent** on this firmware/model. |

The `403 methodNotAllowed` body is exactly:

```xml
<ResponseStatus version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<requestURL>/ISAPI/System/Video/inputs/channels/1/focus</requestURL>
<statusCode>6</statusCode>
<statusString>Invalid Operation</statusString>
<subStatusCode>methodNotAllowed</subStatusCode>
</ResponseStatus>
```

Source: [`ISAPI__403__methodNotAllowed.txt`](../tests/fixtures/hikvision/ISAPI__403__methodNotAllowed.txt).
Classification keys on `statusString == "Invalid Operation"` and
`subStatusCode == "methodNotAllowed"`.

## Endpoint Catalog

### Identity and health

#### `GET /ISAPI/System/deviceInfo`

Returns device identity (`DeviceInfo` value object). Sample values are in the
[Probed Device](#probed-device) table above.
Source: [`ISAPI__System__deviceInfo.txt`](../tests/fixtures/hikvision/ISAPI__System__deviceInfo.txt).

#### `GET /ISAPI/System/status`

Returns a `DeviceStatus` document combining health and **lens odometry**
(`DeviceHealth` + `LensOdometry`). Sample values from cam-04:

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| Uptime (s) | `deviceUpTime` | 126519 |
| CPU utilization (%) | `CPUList/CPU/cpuUtilization` | 56 |
| Memory usage (%) | `MemoryList/Memory/memoryUsage` | 43 |
| Total reboot count | `totalRebootCount` | 12 |
| Fan state | `DomeInfoList/DomeInfo/fanState` | 1 |
| Heat state | `DomeInfoList/DomeInfo/heatState` | 0 |
| Zoom reverse times | `CameraList/Camera/zoomReverseTimes` | 724 |
| Zoom total steps | `zoomTotalSteps` | 621 |
| Focus reverse times | `focusReverseTimes` | 4811 |
| Focus total steps | `focusTotalSteps` | 310 |
| Iris shift times | `irisShiftTimes` | 3951 |
| Iris total steps | `irisTotalSteps` | 27 |
| ICR shift times | `icrShiftTimes` | 0 |
| Lens interior times | `lensIntirTimes` | 11 |
| Camera run total time | `cameraRunTotalTime` | 182826 |
| Pan total rounds | `DomeInfo/panTotalRounds` | 206 |
| Tilt total rounds | `DomeInfo/tiltTotalRounds` | 134 |

These counters are the basis for **lens odometry** and session-health capture.
Source: [`ISAPI__System__status.txt`](../tests/fixtures/hikvision/ISAPI__System__status.txt).

### PTZ read

#### `GET /ISAPI/PTZCtrl/channels/1/status`

Returns the current absolute position as a `PTZStatus` document with an
`AbsoluteHigh` block of **raw ×10** integers.

```xml
<PTZStatus version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <AbsoluteHigh>
    <elevation>182</elevation>     <!-- 18.2°, positive = down -->
    <azimuth>3202</azimuth>        <!-- 320.2° pan -->
    <absoluteZoom>172</absoluteZoom> <!-- 17.2x -->
  </AbsoluteHigh>
</PTZStatus>
```

`status` does **not** carry a focus value; live focus is read from
`absoluteEx/capabilities` (see below).
Source: [`ISAPI__PTZCtrl__channels__1__status.txt`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__status.txt).

#### `GET /ISAPI/PTZCtrl/channels/1/absoluteEx/capabilities`

The authoritative, **degree-based** capabilities source. Values are in
engineering units; min/max are XML attributes. The element text is the current
live reading, including the current **focus position in steps**.

```xml
<PTZAbsoluteEx version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <elevation min="-50" max="60">18.2</elevation>
  <azimuth min="0" max="360">320.2</azimuth>
  <absoluteZoom min="1" max="25">17.2</absoluteZoom>
  <focus min="4096" max="2576990208"/>
  <horizontalSpeed min="0.2" max="210.8"/>
  <verticalSpeed min="0.2" max="151.8"/>
</PTZAbsoluteEx>
```

This response is the source for the `CameraCapabilities` value object.
Source: [`ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities.txt).

#### `GET /ISAPI/PTZCtrl/channels/1/capabilities`

Legacy capabilities document (`PTZChanelCap`) reporting position spaces in
**raw ×10** units, plus feature flags. cam-04 ranges:

| Space | Axis | Min | Max |
|-------|------|-----|-----|
| `AbsolutePanTiltPositionSpace` | XRange (pan) | 0 | 3600 |
| `AbsolutePanTiltPositionSpace` | YRange (tilt) | −500 | 600 |
| `AbsoluteZoomPositionSpace` | ZRange (zoom) | 10 | 250 |
| `ContinuousPanTiltSpace` | XRange / YRange | −100 | 100 |
| `ContinuousZoomSpace` | ZRange | −100 | 100 |

Feature flags include `isSupportPosition3D=true`, `maxPresetNum=300`,
`controlProtocol=PELCO-D`. Prefer `absoluteEx/capabilities` for ranges.
Source: [`ISAPI__PTZCtrl__channels__1__capabilities.txt`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__capabilities.txt).

### PTZ write

The following endpoints accept `PUT` of a `PTZData` (or position) body. They
are setters: a `GET` probe returns `403 methodNotAllowed`.

| Endpoint | Method | Body | Notes |
|----------|--------|------|-------|
| `/ISAPI/PTZCtrl/channels/1/absolute` | PUT | `PTZData/AbsoluteHigh` (raw ×10) | Move to absolute pan/tilt/zoom. |
| `/ISAPI/PTZCtrl/channels/1/relative` | PUT | `PTZData/Relative` | Adapter implements relative as a delta on the current absolute position. |
| `/ISAPI/PTZCtrl/channels/1/continuous` | PUT | `PTZData/pan,tilt,zoom` (−100..100) | Primary jog control. Send `0,0,0` to stop. |
| `/ISAPI/PTZCtrl/channels/1/momentary` | PUT | `PTZData` + duration | Timed jog. |
| `/ISAPI/PTZCtrl/channels/1/position3D` | PUT | `position3D` (0..255 box) | Drag-zoom. `isSupportPosition3D=true`. |
| `/ISAPI/PTZCtrl/channels/1/presets/{id}/goto` | PUT | empty | Recall preset `{id}`. |
| `/ISAPI/PTZCtrl/channels/1/homeposition/goto` | PUT | empty | Recall home position. |

The continuous-jog body shape:

```xml
<PTZData>
  <pan>30</pan>   <!-- -100..100; negative = left -->
  <tilt>0</tilt>  <!-- -100..100; negative = down -->
  <zoom>0</zoom>  <!-- -100..100; negative = zoom out -->
</PTZData>
```

### Presets

#### `GET /ISAPI/PTZCtrl/channels/1/presets`

Returns a `PTZPresetList` of `PTZPreset` elements. cam-04 reports **36
presets**. Each preset carries an `AbsoluteHigh` block in **raw ×10** units:

```xml
<PTZPreset>
  <enabled>true</enabled>
  <id>1</id>
  <presetName>Preset 1</presetName>
  <AbsoluteHigh>
    <elevation>511</elevation>      <!-- 51.1° -->
    <azimuth>2099</azimuth>         <!-- 209.9° -->
    <absoluteZoom>10</absoluteZoom> <!-- 1.0x -->
  </AbsoluteHigh>
</PTZPreset>
```

Source: [`ISAPI__PTZCtrl__channels__1__presets.txt`](../tests/fixtures/hikvision/ISAPI__PTZCtrl__channels__1__presets.txt).

### Optics

Each optics endpoint is `GET`-readable and parsed into a sub-VO of
`ImageOptics`.

#### `GET /ISAPI/Image/channels/1/focusConfiguration`

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| Focus style | `focusStyle` | `SEMIAUTOMATIC` |
| Focus limited | `focusLimited` | 600 |

Source: [`ISAPI__Image__channels__1__focusConfiguration.txt`](../tests/fixtures/hikvision/ISAPI__Image__channels__1__focusConfiguration.txt).

#### `GET /ISAPI/Image/channels/1/iris`

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| Iris level | `IrisLevel` | 160 |
| Min level limit | `minIrisLevelLimit` | 0 |
| Max level limit | `maxIrisLevelLimit` | 100 |

Source: [`ISAPI__Image__channels__1__iris.txt`](../tests/fixtures/hikvision/ISAPI__Image__channels__1__iris.txt).

#### `GET /ISAPI/Image/channels/1/exposure`

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| Exposure type | `ExposureType` | `auto` |
| Overexpose suppress | `OverexposeSuppress/enabled` | `false` |

Source: [`ISAPI__Image__channels__1__exposure.txt`](../tests/fixtures/hikvision/ISAPI__Image__channels__1__exposure.txt).

#### `GET /ISAPI/Image/channels/1/whiteBalance`

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| White-balance style | `WhiteBalanceStyle` | `auto` |
| Red gain | `WhiteBalanceRed` | 50 |
| Blue gain | `WhiteBalanceBlue` | 50 |

Source: [`ISAPI__Image__channels__1__whiteBalance.txt`](../tests/fixtures/hikvision/ISAPI__Image__channels__1__whiteBalance.txt).

#### `GET /ISAPI/Image/channels/1/capabilities`

Returns the image-capabilities document enumerating supported optics ranges.
Source: [`ISAPI__Image__channels__1__capabilities.txt`](../tests/fixtures/hikvision/ISAPI__Image__channels__1__capabilities.txt).

> **Optics setters.** Writing focus through
> `PUT /ISAPI/System/Video/inputs/channels/1/focus` returns
> `403 methodNotAllowed` on cam-04 (see the 403 fixture). Treat optics writes
> as best-effort; `HikvisionUnsupportedError` is the expected result when a
> setter is not exposed.

### Streaming

#### `GET /ISAPI/Streaming/channels/101`

Returns the main-stream `StreamingChannel` configuration (`StreamProfile` VO).
cam-04 main stream:

| Field | Tag | cam-04 value |
|-------|-----|--------------|
| Channel ID | `id` | 101 |
| Codec | `videoCodecType` | `H.264` |
| Resolution | `videoResolutionWidth` × `videoResolutionHeight` | 2560 × 1440 |
| Frame rate | `maxFrameRate` ÷ 100 | 2500 → **25 fps** |
| Quality control | `videoQualityControlType` | `VBR` |
| Bitrate cap (kbps) | `vbrUpperCap` | 6144 |
| Transports | `Transport/ControlProtocolList/.../streamingTransport` | `RTSP`, `HTTP`, `SHTTP`, `SRTP` |

Note `maxFrameRate` is centi-fps: `2500 / 100 = 25.0 fps`.
Source: [`ISAPI__Streaming__channels__101.txt`](../tests/fixtures/hikvision/ISAPI__Streaming__channels__101.txt).

### Snapshot

#### `GET /ISAPI/Streaming/channels/101/picture`

Returns a single JPEG frame (not XML). On cam-04 this is a **2560 × 1440
JPEG, ~222 KB** (`snapshot.jpg`, 222,559 bytes). This is the lowest-latency
single-frame capture path and avoids the cost of opening an RTSP session and
decoding a GOP.
Sample artifact: [`snapshot.jpg`](../tests/fixtures/hikvision/snapshot.jpg).

## Relationship to the Adapter

| ISAPI document | Value object | Adapter method |
|----------------|--------------|----------------|
| `System/deviceInfo` | `DeviceInfo` | `get_device_info()` |
| `System/status` | `DeviceHealth` + `LensOdometry` | `get_health()` |
| `PTZCtrl/.../status` | `PTZState` | `get_ptz_status()` |
| `PTZCtrl/.../absoluteEx/capabilities` | `CameraCapabilities` | `get_capabilities()` |
| `Image/channels/1/{focusConfiguration,iris,exposure,whiteBalance}` | `ImageOptics` | `get_optics()` |
| `Streaming/channels/101` | `StreamProfile` | `get_stream_profiles()` |
| `PTZCtrl/.../presets` | `CameraPreset[]` | `list_presets()` |
| `Streaming/channels/101/picture` | `bytes` (JPEG) | `capture_snapshot()` |
</content>
</invoke>
