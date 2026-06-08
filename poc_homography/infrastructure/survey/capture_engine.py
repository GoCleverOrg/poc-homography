"""The per-pose survey capture step (C2 of the multi-camera survey epic).

:class:`SurveyCaptureEngine` performs a single, self-contained capture at one
commanded pose and emits fully-populated C1 records. It is the reusable
building block that the run-lifecycle / pose-generation layers (C3/C4) call
once per pose; it owns neither pose generation, phase sequencing, nor
persistence.

For each pose the engine:

1. records ``timestamp_before_move``, issues the absolute move (and focus, when
   commanded), then records ``timestamp_after_move``;
2. polls the camera for the settled (reported) PTZ state;
3. snapshots :class:`ImageOptics` + stream profile + device identity *once* per
   burst (these are stable across a burst), assembling the shared
   :class:`CameraIdentity`, :class:`ReportedState`, :class:`MovementContext`,
   and :class:`ImagePipelineState`;
4. captures the burst via one of two mutually exclusive paths —
   :meth:`capture_snapshot_burst` (ISAPI ``/picture`` JPEG stills) or
   :meth:`capture_video_burst` (an RTSP video segment plus extracted frames);
5. computes a SHA-256 checksum and pixel dimensions per frame and emits one
   :class:`FrameRecord` per frame, all sharing a single ``burst_id`` with
   0-based ``frame_index``.

The engine issues **no** PTZ command between the start and end of a burst: the
move/settle/snapshot work all happens before the capture trigger, and the
capture loops touch only the snapshot endpoint or the RTSP stream.

Pipeline-state coverage note: the #256 ``CameraDevice`` adapter surfaces only a
subset of :class:`ImagePipelineState` (resolution, codec, fps from
``StreamProfile``; exposure/focus mode from ``ImageOptics``). Fields the
adapter does not expose (EIS, EPTZ, digital zoom, mirror/flip, corridor,
day/night, crop, stabilisation, encoder profile) are populated with the
conservative defaults in :data:`_UNREPORTED_PIPELINE_DEFAULTS`; surfacing them
requires extending the #256 adapter and is out of scope for this issue.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import cv2
from PIL import Image

from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.camera_config import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
)
from poc_homography.domain.entities.survey.frame_record import (
    CameraIdentity,
    CaptureIdentity,
    CommandedState,
    FrameRecord,
    FullOptics,
    ImageData,
    ImagePipelineState,
    Intrinsics,
    MovementContext,
    PanDirection,
    ReportedState,
    TiltDirection,
    ZoomDirection,
)
from poc_homography.domain.entities.survey.video_burst_record import (
    FrameRef,
    VideoBurstRecord,
)
from poc_homography.types import FPS, Degrees, Millimeters, Pixels, Seconds, Unitless

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from poc_homography.domain.enums.survey_phase import SurveyPhase
    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.domain.vo.device_info import DeviceInfo
    from poc_homography.domain.vo.image_optics import ImageOptics
    from poc_homography.domain.vo.ptz_state import PTZState
    from poc_homography.domain.vo.stream_profile import StreamProfile

# RTSP connection tuning (mirrors webapp/camera_survey/services.py).
RTSP_OPEN_TIMEOUT_MS = 10_000
RTSP_READ_TIMEOUT_MS = 10_000

# Re-encode segment container/codec for the RTSP video-burst path.
SEGMENT_FOURCC = "mp4v"
SEGMENT_SUFFIX = ".mp4"

JPEG_FORMAT = "jpeg"

# Defaults for :class:`ImagePipelineState` fields the #256 adapter does not
# surface. See the module docstring for the rationale.
_UNREPORTED_PIPELINE_DEFAULTS: dict[str, Any] = {
    "profile": "",
    "eis_enabled": False,
    "eptz_enabled": False,
    "digital_zoom": Unitless(1.0),
    "digital_zoom_limit": Unitless(1.0),
    "mirror": False,
    "flip": False,
    "corridor_mode": False,
    "day_night_mode": "",
    "crop_enabled": False,
    "stabilization_enabled": False,
}


class CaptureEngineError(Exception):
    """Raised when a capture cannot be completed (no data, stream failure)."""


@dataclass(frozen=True)
class CaptureContext:
    """Caller-supplied (C3/C4) context framing a single pose capture.

    Attributes:
        run_id: The owning survey run id.
        camera_id: Stable camera identifier (e.g. ``icozee-camptz-04``).
        phase: The survey phase this capture belongs to.
        previous_commanded: The commanded state of the immediately preceding
            pose in the run, or ``None`` for the first pose (yields
            ``"none"`` per-axis directions).
        is_repeatability_sequence: Whether this pose belongs to a repeatability
            or hysteresis sequence. Recorded verbatim; never computed here.
    """

    run_id: str
    camera_id: str
    phase: SurveyPhase
    previous_commanded: CommandedState | None = None
    is_repeatability_sequence: bool = False


@dataclass(frozen=True)
class _MoveResult:
    """The timing and reported state produced by moving to a commanded pose."""

    timestamp_before_move: datetime
    timestamp_after_move: datetime
    reported: PTZState


@dataclass(frozen=True)
class _BurstScaffold:
    """Per-burst values shared by every frame of one burst."""

    burst_id: str
    camera_identity: CameraIdentity
    reported_state: ReportedState
    movement: MovementContext
    pipeline: ImagePipelineState
    primary_profile: StreamProfile
    optics: ImageOptics
    timestamp_before_move: datetime
    timestamp_after_move: datetime


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


def _open_capture(rtsp_url: str) -> Any:
    """Open an RTSP :class:`cv2.VideoCapture` (patchable I/O seam for tests)."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _open_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> Any:
    """Open a :class:`cv2.VideoWriter` (patchable I/O seam for tests)."""
    fourcc = cv2.VideoWriter_fourcc(*SEGMENT_FOURCC)  # type: ignore[attr-defined]
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def _direction(
    delta: float,
    positive: PanDirection | TiltDirection | ZoomDirection,
    negative: PanDirection | TiltDirection | ZoomDirection,
) -> Any:
    """Map the sign of an axis delta to its direction label.

    Args:
        delta: ``commanded - previous`` for one axis.
        positive: Label for an increasing axis.
        negative: Label for a decreasing axis.

    Returns:
        ``positive`` when ``delta > 0``, ``negative`` when ``delta < 0``,
        otherwise ``"none"``.
    """
    if delta > 0.0:
        return positive
    if delta < 0.0:
        return negative
    return "none"


def _sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Decode a JPEG header and return ``(width, height)`` in pixels."""
    with Image.open(io.BytesIO(data)) as img:
        return int(img.width), int(img.height)


class SurveyCaptureEngine:
    """Capture one burst at one commanded pose, emitting C1 records.

    The engine is stateless across calls beyond the injected ``camera`` and
    factories; callers (C3/C4) thread the previous pose through
    :class:`CaptureContext`.
    """

    def __init__(
        self,
        camera: CameraDevice,
        *,
        settle_timeout_s: float = 5.0,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
        base_focal_length_mm: float = DEFAULT_BASE_FOCAL_LENGTH_MM,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], str] | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            camera: The device to drive, satisfying :class:`CameraDevice`.
            settle_timeout_s: Max seconds to wait for PTZ stabilisation.
            sensor_width_mm: Sensor width used to compute per-frame intrinsics;
                defaults to the DS-2DF8425IX constant.
            base_focal_length_mm: Base (1x-zoom) focal length used to compute
                per-frame intrinsics; defaults to the DS-2DF8425IX constant.
            clock: Returns timezone-aware UTC ``datetime``; injectable for
                deterministic tests. Defaults to :func:`_utcnow`.
            uuid_factory: Returns a fresh id string for ``burst_id``;
                injectable for deterministic tests. Defaults to
                :func:`_new_uuid`.
        """
        self._camera = camera
        self._settle_timeout_s = settle_timeout_s
        self._sensor_width_mm = Millimeters(float(sensor_width_mm))
        self._base_focal_length_mm = Millimeters(float(base_focal_length_mm))
        self._clock = clock or _utcnow
        self._new_id = uuid_factory or _new_uuid

    # -- Public capture paths ------------------------------------------------

    def capture_snapshot_burst(
        self,
        commanded: CommandedState,
        context: CaptureContext,
        *,
        burst_count: int,
        output_dir: Path,
    ) -> list[FrameRecord]:
        """Capture an ``N``-frame snapshot burst via ISAPI ``/picture``.

        Issues ``burst_count`` sequential ``capture_snapshot()`` calls at a
        single, held pose; each response becomes one JPEG file and one
        :class:`FrameRecord`. Suitable for static poses where ≈1–2 s of
        inter-frame latency is acceptable (not jitter analysis).

        Args:
            commanded: The PTZ + focus state to command and record.
            context: Run/phase/previous-pose framing for the burst.
            burst_count: Number of frames ``N >= 1``; all share one
                ``burst_id`` with ``frame_index`` ``0..N-1``.
            output_dir: Directory to write JPEG frame files into.

        Returns:
            One :class:`FrameRecord` per frame, in ``frame_index`` order.

        Raises:
            ValueError: If ``burst_count < 1``.
            CaptureEngineError: If a snapshot returns no bytes.
        """
        if burst_count < 1:
            raise ValueError(f"burst_count must be >= 1, got {burst_count}")

        scaffold = self._prepare_burst(commanded, context)
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[FrameRecord] = []
        for frame_index in range(burst_count):
            timestamp_at_capture = self._clock()
            jpeg_bytes = self._camera.capture_snapshot()
            if not jpeg_bytes:
                raise CaptureEngineError(
                    f"capture_snapshot returned no data at frame {frame_index}"
                )
            width, height = _jpeg_dimensions(jpeg_bytes)
            image_path = output_dir / f"{scaffold.burst_id}_{frame_index:04d}.jpg"
            image_path.write_bytes(jpeg_bytes)
            image_data = ImageData(
                image_path=image_path,
                checksum=_sha256_hex(jpeg_bytes),
                width=Pixels(width),
                height=Pixels(height),
                capture_format=JPEG_FORMAT,
            )
            records.append(
                self._assemble_frame(
                    scaffold=scaffold,
                    context=context,
                    commanded=commanded,
                    frame_index=frame_index,
                    timestamp_at_capture=timestamp_at_capture,
                    image_data=image_data,
                )
            )
        return records

    def capture_video_burst(
        self,
        commanded: CommandedState,
        context: CaptureContext,
        rtsp_url: str,
        *,
        burst_duration_s: float,
        output_dir: Path,
        target_fps: float | None = None,
    ) -> tuple[VideoBurstRecord, list[FrameRecord]]:
        """Capture an RTSP video segment and extract its frames.

        Records a continuous segment from the RTSP stream into a single file,
        then extracts every decoded frame as an addressable JPEG. Used for
        Phase 8 (static jitter) where frame-to-frame motion at native FPS must
        be observed while the camera is held commanded-still. No PTZ command is
        issued between the burst trigger and burst end.

        Args:
            commanded: The PTZ + focus state to command and record.
            context: Run/phase/previous-pose framing for the burst.
            rtsp_url: The RTSP stream URL to record.
            burst_duration_s: Target segment length in seconds; recording also
                stops early if the stream ends.
            output_dir: Directory to write the segment file and JPEG frames.
            target_fps: Frame rate for the written segment; falls back to the
                primary stream profile's FPS.

        Returns:
            The :class:`VideoBurstRecord` for the segment and one
            :class:`FrameRecord` per extracted frame, in ``frame_index`` order.

        Raises:
            CaptureEngineError: If the stream cannot be opened or yields no
                frames.
        """
        scaffold = self._prepare_burst(commanded, context)
        output_dir.mkdir(parents=True, exist_ok=True)

        burst_trigger = self._clock()
        frames, frame_times = self._record_segment_frames(rtsp_url, burst_duration_s, burst_trigger)
        if not frames:
            raise CaptureEngineError("RTSP video burst captured no frames")

        height, width = frames[0].shape[:2]
        fps = float(target_fps) if target_fps is not None else float(scaffold.primary_profile.fps)
        # Record the actual recorded span, not the requested target: the read
        # loop stops early when the stream ends, so the real segment can be
        # shorter than ``burst_duration_s``.
        recorded_duration_s = (frame_times[-1] - burst_trigger).total_seconds()
        segment_path = output_dir / f"{scaffold.burst_id}{SEGMENT_SUFFIX}"
        self._write_segment(segment_path, frames, fps, (int(width), int(height)))

        records: list[FrameRecord] = []
        frame_refs: list[FrameRef] = []
        for frame_index, (frame, timestamp_at_capture) in enumerate(zip(frames, frame_times)):
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise CaptureEngineError(f"Failed to JPEG-encode frame {frame_index}")
            jpeg_bytes = bytes(buffer.tobytes())
            frame_h, frame_w = frame.shape[:2]
            image_path = output_dir / f"{scaffold.burst_id}_{frame_index:04d}.jpg"
            image_path.write_bytes(jpeg_bytes)
            image_data = ImageData(
                image_path=image_path,
                checksum=_sha256_hex(jpeg_bytes),
                width=Pixels(int(frame_w)),
                height=Pixels(int(frame_h)),
                capture_format=JPEG_FORMAT,
            )
            record = self._assemble_frame(
                scaffold=scaffold,
                context=context,
                commanded=commanded,
                frame_index=frame_index,
                timestamp_at_capture=timestamp_at_capture,
                image_data=image_data,
            )
            records.append(record)
            frame_refs.append(
                FrameRef(
                    capture_id=record.capture.capture_id,
                    frame_index=frame_index,
                    timestamp_at_capture=timestamp_at_capture,
                    image_path=image_path,
                )
            )

        burst_record = VideoBurstRecord(
            burst_id=scaffold.burst_id,
            run_id=context.run_id,
            camera_id=context.camera_id,
            phase=context.phase,
            segment_path=segment_path,
            duration_s=Seconds(recorded_duration_s),
            fps=FPS(fps),
            codec=scaffold.primary_profile.codec,
            commanded_state=commanded,
            frame_refs=tuple(frame_refs),
        )
        return burst_record, records

    # -- Internal helpers ----------------------------------------------------

    def _record_segment_frames(
        self,
        rtsp_url: str,
        burst_duration_s: float,
        burst_trigger: datetime,
    ) -> tuple[list[Any], list[datetime]]:
        """Read frames from the RTSP stream until the duration elapses or it ends.

        Issues no PTZ command — only stream reads — so the camera stays
        commanded-still for the whole burst.

        Args:
            rtsp_url: The RTSP stream URL.
            burst_duration_s: Target wall-clock duration in seconds.
            burst_trigger: The burst-start timestamp the duration is measured
                from.

        Returns:
            A ``(frames, frame_times)`` pair of equal length.

        Raises:
            CaptureEngineError: If the stream cannot be opened.
        """
        cap = _open_capture(rtsp_url)
        frames: list[Any] = []
        frame_times: list[datetime] = []
        try:
            if not cap.isOpened():
                raise CaptureEngineError(f"Failed to open RTSP stream: {rtsp_url}")
            while (self._clock() - burst_trigger).total_seconds() < burst_duration_s:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
                frame_times.append(self._clock())
        finally:
            cap.release()
        return frames, frame_times

    def _write_segment(
        self,
        segment_path: Path,
        frames: list[Any],
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        """Write ``frames`` to ``segment_path`` as an encoded video segment."""
        writer = _open_writer(segment_path, fps, frame_size)
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _prepare_burst(
        self,
        commanded: CommandedState,
        context: CaptureContext,
    ) -> _BurstScaffold:
        """Move to the pose, settle, and snapshot the per-burst shared state."""
        move = self._move_and_settle(commanded)
        device_info = self._camera.get_device_info()
        optics = self._camera.get_optics()
        profiles = self._camera.get_stream_profiles()
        if not profiles:
            raise CaptureEngineError("Camera reported no stream profiles")
        primary = profiles[0]

        burst_trigger = self._clock()
        movement = self._build_movement_context(
            commanded=commanded,
            context=context,
            timestamp_after_move=move.timestamp_after_move,
            burst_trigger=burst_trigger,
        )
        return _BurstScaffold(
            burst_id=self._new_id(),
            camera_identity=_build_camera_identity(context.camera_id, device_info, primary),
            reported_state=_build_reported_state(move.reported),
            movement=movement,
            pipeline=_build_pipeline_state(primary, optics),
            primary_profile=primary,
            optics=optics,
            timestamp_before_move=move.timestamp_before_move,
            timestamp_after_move=move.timestamp_after_move,
        )

    def _move_and_settle(self, commanded: CommandedState) -> _MoveResult:
        """Command the pose (and focus) and poll for the settled state."""
        timestamp_before_move = self._clock()
        self._camera.move_absolute(
            pan=float(commanded.commanded_pan),
            tilt=float(commanded.commanded_tilt),
            zoom=float(commanded.commanded_zoom),
        )
        if commanded.commanded_focus is not None:
            self._camera.set_focus(commanded.commanded_focus)
        timestamp_after_move = self._clock()
        reported = self._camera.wait_for_stabilization(timeout_s=self._settle_timeout_s)
        return _MoveResult(
            timestamp_before_move=timestamp_before_move,
            timestamp_after_move=timestamp_after_move,
            reported=reported,
        )

    def _build_movement_context(
        self,
        *,
        commanded: CommandedState,
        context: CaptureContext,
        timestamp_after_move: datetime,
        burst_trigger: datetime,
    ) -> MovementContext:
        """Assemble the per-pose movement context (previous pose + directions)."""
        previous = context.previous_commanded
        prev_pan = previous.commanded_pan if previous is not None else commanded.commanded_pan
        prev_tilt = previous.commanded_tilt if previous is not None else commanded.commanded_tilt
        prev_zoom = previous.commanded_zoom if previous is not None else commanded.commanded_zoom

        settling_delay_s = (burst_trigger - timestamp_after_move).total_seconds()
        return MovementContext(
            prev_pan=Degrees(float(prev_pan)),
            prev_tilt=Degrees(float(prev_tilt)),
            prev_zoom=Unitless(float(prev_zoom)),
            direction_pan=_direction(commanded.commanded_pan - prev_pan, "cw", "ccw"),
            direction_tilt=_direction(commanded.commanded_tilt - prev_tilt, "up", "down"),
            direction_zoom=_direction(commanded.commanded_zoom - prev_zoom, "tele", "wide"),
            settling_delay_s=Seconds(settling_delay_s),
            is_repeatability_sequence=context.is_repeatability_sequence,
        )

    def _assemble_frame(
        self,
        *,
        scaffold: _BurstScaffold,
        context: CaptureContext,
        commanded: CommandedState,
        frame_index: int,
        timestamp_at_capture: datetime,
        image_data: ImageData,
    ) -> FrameRecord:
        """Build one :class:`FrameRecord` from the shared burst scaffold."""
        capture = CaptureIdentity(
            capture_id=f"{scaffold.burst_id}_{frame_index:04d}",
            run_id=context.run_id,
            phase=context.phase,
            burst_id=scaffold.burst_id,
            frame_index=frame_index,
            timestamp_before_move=scaffold.timestamp_before_move,
            timestamp_after_move=scaffold.timestamp_after_move,
            timestamp_at_capture=timestamp_at_capture,
        )
        intrinsics = self._build_intrinsics(commanded, image_data)
        # ``ground_homography`` and ``floor_mask_reference`` are deliberately
        # left ``None`` at survey-capture time: the ground homography needs
        # deployment-specific extrinsics (camera height, pixels-per-meter, map
        # origin) a bare survey does not possess, and the floor mask is filled
        # by an external masker. Both are populated downstream, not here.
        return FrameRecord(
            camera=scaffold.camera_identity,
            capture=capture,
            commanded=commanded,
            reported=scaffold.reported_state,
            movement=scaffold.movement,
            pipeline=scaffold.pipeline,
            image_data=image_data,
            intrinsics=intrinsics,
            full_optics=_build_full_optics(scaffold.optics),
        )

    def _build_intrinsics(
        self,
        commanded: CommandedState,
        image_data: ImageData,
    ) -> Intrinsics:
        """Compute per-frame intrinsics from commanded zoom + frame dimensions.

        Stores both the recompute inputs and the cached ``k_matrix`` (the matrix
        is never load-bearing — it can be recomputed from the inputs).
        """
        computed = compute_intrinsics(
            commanded.commanded_zoom,
            image_data.width,
            image_data.height,
            self._sensor_width_mm,
            self._base_focal_length_mm,
        )
        return Intrinsics(
            zoom=commanded.commanded_zoom,
            image_width=image_data.width,
            image_height=image_data.height,
            sensor_width_mm=self._sensor_width_mm,
            base_focal_length_mm=self._base_focal_length_mm,
            k_matrix=[[float(value) for value in row] for row in computed.K.tolist()],
        )


def _build_camera_identity(
    camera_id: str,
    device_info: DeviceInfo,
    primary_profile: StreamProfile,
) -> CameraIdentity:
    """Assemble :class:`CameraIdentity` from device info + stream profile."""
    return CameraIdentity(
        camera_id=camera_id,
        brand=device_info.manufacturer or "",
        model=device_info.model or "",
        serial=device_info.serial_number or "",
        firmware=device_info.firmware_version or "",
        channel_id="1",
        stream_id=str(primary_profile.channel_id),
    )


def _build_reported_state(reported: PTZState) -> ReportedState:
    """Assemble :class:`ReportedState` from the settled PTZ state.

    ``reported_azimuth``, ``reported_elevation``, and
    ``reported_focal_length_mm`` are left ``None`` — the #256 adapter's
    :class:`PTZState` does not surface them.
    """
    return ReportedState(
        reported_pan=Degrees(float(reported.pan_raw)),
        reported_azimuth=None,
        reported_tilt=Degrees(float(reported.tilt_deg)),
        reported_elevation=None,
        reported_zoom=Unitless(float(reported.zoom)),
        reported_focal_length_mm=None,
        reported_focus=int(reported.focus) if reported.focus is not None else None,
        ptz_settled=True,
    )


def _build_full_optics(optics: ImageOptics) -> FullOptics:
    """Map the per-burst :class:`ImageOptics` onto a per-frame :class:`FullOptics`.

    Only the fields the #256 adapter's ``ImageOptics`` surfaces are mapped
    (exposure type, iris level, white-balance style, focus limit); the optics
    fields with no adapter source (``shutter``, ``gain``) are left ``None``.
    """
    return FullOptics(
        exposure_type=optics.exposure.exposure_type or None,
        shutter=None,
        gain=None,
        iris=optics.iris.level,
        white_balance=optics.white_balance.style or None,
        focus=optics.focus.focus_limited,
    )


def _build_pipeline_state(
    primary_profile: StreamProfile,
    optics: ImageOptics,
) -> ImagePipelineState:
    """Assemble :class:`ImagePipelineState` from stream profile + optics.

    Only resolution/codec/fps (from ``StreamProfile``) and exposure/focus mode
    (from ``ImageOptics``) come from real adapter data; the remaining encoder /
    image-pipeline flags use :data:`_UNREPORTED_PIPELINE_DEFAULTS`.
    """
    return ImagePipelineState(
        resolution_width=Pixels(int(primary_profile.width)),
        resolution_height=Pixels(int(primary_profile.height)),
        codec=primary_profile.codec,
        fps=FPS(float(primary_profile.fps)),
        exposure_mode=optics.exposure.exposure_type,
        focus_mode=optics.focus.style,
        **_UNREPORTED_PIPELINE_DEFAULTS,
    )
