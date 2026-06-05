"""Rich per-frame survey record and its nested sub-value-objects.

``FrameRecord`` captures the full optical and mechanical state at every
survey frame. Fields are grouped into named sub-VOs that mirror how they are
naturally populated (camera identity from ``DeviceInfo``, pipeline state from
``StreamProfile`` + ``ImageOptics``, capture identity from the engine, etc.),
each with its own ``to_dict`` / ``from_dict`` following the project pattern in
``poc_homography/domain/vo/ptz_state.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from poc_homography.domain.entities.survey import (
    SURVEY_SCHEMA_VERSION,
    check_schema_version,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.types import (
    FPS,
    Degrees,
    Millimeters,
    Pixels,
    Seconds,
    Unitless,
)

PanDirection = Literal["cw", "ccw", "none"]
TiltDirection = Literal["up", "down", "none"]
ZoomDirection = Literal["tele", "wide", "none"]


@dataclass(frozen=True)
class CameraIdentity:
    """Stable identity of the camera that produced the frame.

    Sourced from ``DeviceInfo`` (#256) and ``StreamProfile`` (#256).
    """

    camera_id: str
    brand: str
    model: str
    serial: str
    firmware: str
    channel_id: str
    stream_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "camera_id": self.camera_id,
            "brand": self.brand,
            "model": self.model,
            "serial": self.serial,
            "firmware": self.firmware,
            "channel_id": self.channel_id,
            "stream_id": self.stream_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraIdentity:
        """Create :class:`CameraIdentity` from a dictionary."""
        return cls(
            camera_id=str(data["camera_id"]),
            brand=str(data["brand"]),
            model=str(data["model"]),
            serial=str(data["serial"]),
            firmware=str(data["firmware"]),
            channel_id=str(data["channel_id"]),
            stream_id=str(data["stream_id"]),
        )


@dataclass(frozen=True)
class CaptureIdentity:
    """Identity and timing of an individual capture within a run."""

    capture_id: str
    run_id: str
    phase: SurveyPhase
    burst_id: str | None
    frame_index: int
    timestamp_before_move: datetime
    timestamp_after_move: datetime
    timestamp_at_capture: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capture_id": self.capture_id,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "burst_id": self.burst_id,
            "frame_index": self.frame_index,
            "timestamp_before_move": self.timestamp_before_move.isoformat(),
            "timestamp_after_move": self.timestamp_after_move.isoformat(),
            "timestamp_at_capture": self.timestamp_at_capture.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureIdentity:
        """Create :class:`CaptureIdentity` from a dictionary."""
        burst_id = data.get("burst_id")
        return cls(
            capture_id=str(data["capture_id"]),
            run_id=str(data["run_id"]),
            phase=SurveyPhase(data["phase"]),
            burst_id=str(burst_id) if burst_id is not None else None,
            frame_index=int(data["frame_index"]),
            timestamp_before_move=datetime.fromisoformat(data["timestamp_before_move"]),
            timestamp_after_move=datetime.fromisoformat(data["timestamp_after_move"]),
            timestamp_at_capture=datetime.fromisoformat(data["timestamp_at_capture"]),
        )


@dataclass(frozen=True)
class CommandedState:
    """The PTZ + focus state commanded to the camera for this frame.

    Reused by :class:`~poc_homography.domain.entities.survey.video_burst_record.VideoBurstRecord`.
    """

    commanded_pan: Degrees
    commanded_tilt: Degrees
    commanded_zoom: Unitless
    commanded_focus: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "commanded_pan": float(self.commanded_pan),
            "commanded_tilt": float(self.commanded_tilt),
            "commanded_zoom": float(self.commanded_zoom),
            "commanded_focus": self.commanded_focus,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandedState:
        """Create :class:`CommandedState` from a dictionary."""
        focus = data.get("commanded_focus")
        return cls(
            commanded_pan=Degrees(float(data["commanded_pan"])),
            commanded_tilt=Degrees(float(data["commanded_tilt"])),
            commanded_zoom=Unitless(float(data["commanded_zoom"])),
            commanded_focus=int(focus) if focus is not None else None,
        )


@dataclass(frozen=True)
class ReportedState:
    """The PTZ + optics state reported back by the camera for this frame."""

    reported_pan: Degrees
    reported_azimuth: Degrees | None
    reported_tilt: Degrees
    reported_elevation: Degrees | None
    reported_zoom: Unitless
    reported_focal_length_mm: Millimeters | None
    reported_focus: int | None
    ptz_settled: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reported_pan": float(self.reported_pan),
            "reported_azimuth": (
                float(self.reported_azimuth) if self.reported_azimuth is not None else None
            ),
            "reported_tilt": float(self.reported_tilt),
            "reported_elevation": (
                float(self.reported_elevation) if self.reported_elevation is not None else None
            ),
            "reported_zoom": float(self.reported_zoom),
            "reported_focal_length_mm": (
                float(self.reported_focal_length_mm)
                if self.reported_focal_length_mm is not None
                else None
            ),
            "reported_focus": self.reported_focus,
            "ptz_settled": self.ptz_settled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportedState:
        """Create :class:`ReportedState` from a dictionary."""
        azimuth = data.get("reported_azimuth")
        elevation = data.get("reported_elevation")
        focal = data.get("reported_focal_length_mm")
        focus = data.get("reported_focus")
        return cls(
            reported_pan=Degrees(float(data["reported_pan"])),
            reported_azimuth=Degrees(float(azimuth)) if azimuth is not None else None,
            reported_tilt=Degrees(float(data["reported_tilt"])),
            reported_elevation=Degrees(float(elevation)) if elevation is not None else None,
            reported_zoom=Unitless(float(data["reported_zoom"])),
            reported_focal_length_mm=(Millimeters(float(focal)) if focal is not None else None),
            reported_focus=int(focus) if focus is not None else None,
            ptz_settled=bool(data["ptz_settled"]),
        )


@dataclass(frozen=True)
class MovementContext:
    """Movement that preceded this frame and its settling context."""

    prev_pan: Degrees
    prev_tilt: Degrees
    prev_zoom: Unitless
    direction_pan: PanDirection
    direction_tilt: TiltDirection
    direction_zoom: ZoomDirection
    settling_delay_s: Seconds
    is_repeatability_sequence: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prev_pan": float(self.prev_pan),
            "prev_tilt": float(self.prev_tilt),
            "prev_zoom": float(self.prev_zoom),
            "direction_pan": self.direction_pan,
            "direction_tilt": self.direction_tilt,
            "direction_zoom": self.direction_zoom,
            "settling_delay_s": float(self.settling_delay_s),
            "is_repeatability_sequence": self.is_repeatability_sequence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MovementContext:
        """Create :class:`MovementContext` from a dictionary."""
        return cls(
            prev_pan=Degrees(float(data["prev_pan"])),
            prev_tilt=Degrees(float(data["prev_tilt"])),
            prev_zoom=Unitless(float(data["prev_zoom"])),
            direction_pan=data["direction_pan"],
            direction_tilt=data["direction_tilt"],
            direction_zoom=data["direction_zoom"],
            settling_delay_s=Seconds(float(data["settling_delay_s"])),
            is_repeatability_sequence=bool(data["is_repeatability_sequence"]),
        )


@dataclass(frozen=True)
class ImagePipelineState:
    """Encoder / image-pipeline state, from ``StreamProfile`` + ``ImageOptics``."""

    resolution_width: Pixels
    resolution_height: Pixels
    codec: str
    profile: str
    fps: FPS
    eis_enabled: bool
    eptz_enabled: bool
    digital_zoom: Unitless
    digital_zoom_limit: Unitless
    mirror: bool
    flip: bool
    corridor_mode: bool
    day_night_mode: str
    crop_enabled: bool
    stabilization_enabled: bool
    exposure_mode: str
    focus_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "resolution_width": int(self.resolution_width),
            "resolution_height": int(self.resolution_height),
            "codec": self.codec,
            "profile": self.profile,
            "fps": float(self.fps),
            "eis_enabled": self.eis_enabled,
            "eptz_enabled": self.eptz_enabled,
            "digital_zoom": float(self.digital_zoom),
            "digital_zoom_limit": float(self.digital_zoom_limit),
            "mirror": self.mirror,
            "flip": self.flip,
            "corridor_mode": self.corridor_mode,
            "day_night_mode": self.day_night_mode,
            "crop_enabled": self.crop_enabled,
            "stabilization_enabled": self.stabilization_enabled,
            "exposure_mode": self.exposure_mode,
            "focus_mode": self.focus_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImagePipelineState:
        """Create :class:`ImagePipelineState` from a dictionary."""
        return cls(
            resolution_width=Pixels(int(data["resolution_width"])),
            resolution_height=Pixels(int(data["resolution_height"])),
            codec=str(data["codec"]),
            profile=str(data["profile"]),
            fps=FPS(float(data["fps"])),
            eis_enabled=bool(data["eis_enabled"]),
            eptz_enabled=bool(data["eptz_enabled"]),
            digital_zoom=Unitless(float(data["digital_zoom"])),
            digital_zoom_limit=Unitless(float(data["digital_zoom_limit"])),
            mirror=bool(data["mirror"]),
            flip=bool(data["flip"]),
            corridor_mode=bool(data["corridor_mode"]),
            day_night_mode=str(data["day_night_mode"]),
            crop_enabled=bool(data["crop_enabled"]),
            stabilization_enabled=bool(data["stabilization_enabled"]),
            exposure_mode=str(data["exposure_mode"]),
            focus_mode=str(data["focus_mode"]),
        )


@dataclass(frozen=True)
class ImageData:
    """The persisted image file and its integrity / dimension metadata."""

    image_path: Path
    checksum: str
    width: Pixels
    height: Pixels
    capture_format: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": str(self.image_path),
            "checksum": self.checksum,
            "width": int(self.width),
            "height": int(self.height),
            "capture_format": self.capture_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageData:
        """Create :class:`ImageData` from a dictionary."""
        return cls(
            image_path=Path(data["image_path"]),
            checksum=str(data["checksum"]),
            width=Pixels(int(data["width"])),
            height=Pixels(int(data["height"])),
            capture_format=str(data["capture_format"]),
        )


@dataclass(frozen=True)
class SurveyContext:
    """Planner-derived survey context attached to a frame by the phase layer.

    These fields originate in the C3 planner pose (and the C4 phase that drives
    it), not in the C2 capture engine: ``region_id`` groups cross-zoom
    observations of the same ground region (Phase 6), ``approach_direction``
    records the direction a pose was approached from (Phases 3/7), and
    ``sequence_index`` is the visit index within a repeat group (Phase 7). All
    are optional and default to ``None`` so frames captured outside a phase
    context (and pre-existing serialised frames) round-trip unchanged.
    """

    region_id: str | None = None
    approach_direction: str | None = None
    sequence_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region_id": self.region_id,
            "approach_direction": self.approach_direction,
            "sequence_index": self.sequence_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyContext:
        """Create :class:`SurveyContext` from a dictionary."""
        region_id = data.get("region_id")
        approach_direction = data.get("approach_direction")
        sequence_index = data.get("sequence_index")
        return cls(
            region_id=str(region_id) if region_id is not None else None,
            approach_direction=(
                str(approach_direction) if approach_direction is not None else None
            ),
            sequence_index=int(sequence_index) if sequence_index is not None else None,
        )


@dataclass(frozen=True, eq=False)
class FrameRecord:
    """The full optical + mechanical state captured at a single survey frame.

    The ``id`` property is the per-frame ``capture_id``, satisfying the
    ``Entity`` protocol. ``schema_version`` is stamped from
    :data:`SURVEY_SCHEMA_VERSION` and validated on load.
    """

    camera: CameraIdentity
    capture: CaptureIdentity
    commanded: CommandedState
    reported: ReportedState
    movement: MovementContext
    pipeline: ImagePipelineState
    image_data: ImageData
    survey_context: SurveyContext = field(default_factory=SurveyContext)
    schema_version: str = field(default=SURVEY_SCHEMA_VERSION)

    @property
    def id(self) -> str:
        """Unique identifier — the per-frame capture id."""
        return self.capture.capture_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "camera": self.camera.to_dict(),
            "capture": self.capture.to_dict(),
            "commanded": self.commanded.to_dict(),
            "reported": self.reported.to_dict(),
            "movement": self.movement.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "image_data": self.image_data.to_dict(),
            "survey_context": self.survey_context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameRecord:
        """Create :class:`FrameRecord` from a dictionary.

        Raises:
            ValueError: If ``schema_version`` is unrecognised.
        """
        version = check_schema_version(str(data["schema_version"]))
        return cls(
            camera=CameraIdentity.from_dict(data["camera"]),
            capture=CaptureIdentity.from_dict(data["capture"]),
            commanded=CommandedState.from_dict(data["commanded"]),
            reported=ReportedState.from_dict(data["reported"]),
            movement=MovementContext.from_dict(data["movement"]),
            pipeline=ImagePipelineState.from_dict(data["pipeline"]),
            image_data=ImageData.from_dict(data["image_data"]),
            survey_context=SurveyContext.from_dict(data.get("survey_context") or {}),
            schema_version=version,
        )
