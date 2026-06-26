"""Per-model lens-distortion table: shareable default + full run history.

Tier above the per-camera ``LensCalibrationTable``. Holds, for one camera
*model*, the robust aggregated coefficient per zoom bin (``entries``) and the
complete provenance of contributing measurements (``run_history``) so the
aggregate can be recomputed and its precision audited as runs accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.calibration_run_record import CalibrationRunRecord
from poc_homography.domain.vo.model_zoom_coefficient import ModelZoomCoefficient


@dataclass(frozen=True, eq=False)
class ModelCalibrationTable:
    """Aggregated lens-distortion calibration for a camera model.

    Attributes:
        model_name: Camera model identifier (e.g. ``"DS-2DF8425IX-AELW"``).
        entries: Aggregated coefficients per zoom bin, sorted by ``zoom_bin``.
        run_history: All contributing per-camera/per-zoom measurements.
        created_date: ISO-format creation timestamp.
        last_modified: ISO-format last modification timestamp.
    """

    model_name: str
    entries: tuple[ModelZoomCoefficient, ...]
    run_history: tuple[CalibrationRunRecord, ...]
    created_date: str
    last_modified: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.model_name == other.model_name

    def __hash__(self) -> int:
        return hash(self.model_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "entries": [e.to_dict() for e in self.entries],
            "run_history": [r.to_dict() for r in self.run_history],
            "created_date": self.created_date,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCalibrationTable:
        """Create ModelCalibrationTable from dictionary."""
        entries = tuple(ModelZoomCoefficient.from_dict(e) for e in data.get("entries", []))
        run_history = tuple(CalibrationRunRecord.from_dict(r) for r in data.get("run_history", []))
        return cls(
            model_name=data["model_name"],
            entries=entries,
            run_history=run_history,
            created_date=data.get("created_date", ""),
            last_modified=data.get("last_modified", ""),
        )
