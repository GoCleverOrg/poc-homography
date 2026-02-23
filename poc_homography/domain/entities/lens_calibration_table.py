"""Lens calibration table entity for zoom-indexed distortion persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry


@dataclass(frozen=True, eq=False)
class LensCalibrationTable:
    """Zoom-indexed lens calibration data for a single camera.

    This entity is purely for persistence — business logic (interpolation,
    nearest-entry lookup) stays on the legacy ``CameraCalibrationTable``.

    Attributes:
        id: Camera identifier (e.g. ``"cam01"``).
        entries: Calibration entries sorted by zoom_factor.
        created_date: ISO-format creation timestamp.
        last_modified: ISO-format last modification timestamp.
    """

    id: str
    entries: tuple[ZoomCalibrationEntry, ...]
    created_date: str
    last_modified: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "entries": [e.to_dict() for e in self.entries],
            "created_date": self.created_date,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LensCalibrationTable:
        """Create LensCalibrationTable from dictionary."""
        entries = tuple(ZoomCalibrationEntry.from_dict(e) for e in data.get("entries", []))
        return cls(
            id=data["id"],
            entries=entries,
            created_date=data.get("created_date", ""),
            last_modified=data.get("last_modified", ""),
        )
