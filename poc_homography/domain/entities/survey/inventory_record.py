"""Camera-inventory C1 record for Phase 1 of the multi-camera survey.

Unlike the image-bearing :class:`FrameRecord`, the inventory record captures the
camera's full self-report in a single pass at session start: device identity and
firmware, PTZ capabilities, the image-optics pipeline state, runtime health and
lens odometry, the available stream profiles, and the stored PTZ presets. Every
field is sourced verbatim from the #256 ``CameraDevice`` value objects, each of
which already provides ``to_dict`` / ``from_dict``; this record only aggregates
them and stamps the survey phase, run/camera identity, and schema version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from poc_homography.domain.entities.survey import (
    SURVEY_SCHEMA_VERSION,
    check_schema_version,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.vo.camera_capabilities import CameraCapabilities
from poc_homography.domain.vo.camera_preset import CameraPreset
from poc_homography.domain.vo.device_health import DeviceHealth
from poc_homography.domain.vo.device_info import DeviceInfo
from poc_homography.domain.vo.image_optics import ImageOptics
from poc_homography.domain.vo.stream_profile import StreamProfile


@dataclass(frozen=True, eq=False)
class CameraInventoryRecord:
    """The full camera self-report captured once per run by Phase 1.

    The ``id`` property is the ``record_id``, satisfying the ``Entity``
    protocol. ``phase`` is always :attr:`SurveyPhase.CAMERA_INVENTORY`; it is
    stored explicitly so the record is phase-tagged identically to every other
    C1 record. ``schema_version`` is stamped from :data:`SURVEY_SCHEMA_VERSION`
    and validated on load.

    Attributes:
        record_id: Unique per-run inventory record id.
        run_id: The owning survey run id.
        camera_id: Stable camera identifier (e.g. ``icozee-camptz-04``).
        captured_at: When the inventory pass ran.
        device_info: Device identity and firmware (#256 ``DeviceInfo``).
        capabilities: PTZ position/speed and zoom/focus limits
            (#256 ``CameraCapabilities``).
        optics: Focus/iris/exposure/white-balance image-pipeline state
            (#256 ``ImageOptics``).
        health: Runtime health and lens odometry (#256 ``DeviceHealth``).
        stream_profiles: The available streaming profiles (#256
            ``StreamProfile``); the first is the primary channel.
        presets: The stored PTZ presets (#256 ``CameraPreset``).
        phase: Always :attr:`SurveyPhase.CAMERA_INVENTORY`.
        schema_version: Survey schema version, stamped and validated on load.
    """

    record_id: str
    run_id: str
    camera_id: str
    captured_at: datetime
    device_info: DeviceInfo
    capabilities: CameraCapabilities
    optics: ImageOptics
    health: DeviceHealth
    stream_profiles: tuple[StreamProfile, ...]
    presets: tuple[CameraPreset, ...]
    phase: SurveyPhase = SurveyPhase.CAMERA_INVENTORY
    schema_version: str = field(default=SURVEY_SCHEMA_VERSION)

    @property
    def id(self) -> str:
        """Unique identifier — the inventory record id."""
        return self.record_id

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
            "record_id": self.record_id,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "phase": self.phase.value,
            "captured_at": self.captured_at.isoformat(),
            "device_info": self.device_info.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "optics": self.optics.to_dict(),
            "health": self.health.to_dict(),
            "stream_profiles": [profile.to_dict() for profile in self.stream_profiles],
            "presets": [preset.to_dict() for preset in self.presets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraInventoryRecord:
        """Create :class:`CameraInventoryRecord` from a dictionary.

        Raises:
            ValueError: If ``schema_version`` is unrecognised.
        """
        version = check_schema_version(str(data["schema_version"]))
        return cls(
            record_id=str(data["record_id"]),
            run_id=str(data["run_id"]),
            camera_id=str(data["camera_id"]),
            captured_at=datetime.fromisoformat(data["captured_at"]),
            device_info=DeviceInfo.from_dict(data["device_info"]),
            capabilities=CameraCapabilities.from_dict(data["capabilities"]),
            optics=ImageOptics.from_dict(data["optics"]),
            health=DeviceHealth.from_dict(data["health"]),
            stream_profiles=tuple(StreamProfile.from_dict(p) for p in data["stream_profiles"]),
            presets=tuple(CameraPreset.from_dict(p) for p in data["presets"]),
            phase=SurveyPhase(data["phase"]),
            schema_version=version,
        )
