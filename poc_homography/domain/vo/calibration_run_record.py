"""Provenance record for a single per-camera, per-zoom distortion measurement.

Each record is one observation contributing to a per-model aggregate. Keeping
the full run history is what lets the model coefficients gain precision as more
runs accumulate (the reported per-bin std shrinks with N).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.types import Unitless


@dataclass(frozen=True)
class CalibrationRunRecord:
    """One per-camera, per-zoom distortion measurement (model-aggregate input).

    Attributes:
        camera_id: Camera unit that produced the measurement.
        zoom_factor: Zoom level the measurement was taken at.
        distortion: Measured distortion coefficients.
        calibration_date: ISO-format timestamp of the measurement.
        validation_rmse: Held-out line-straightness RMSE (lower is better).
        num_lines_used: Number of lines the solve used (higher is better).
    """

    camera_id: str
    zoom_factor: Unitless
    distortion: LensDistortion
    calibration_date: str
    validation_rmse: float = 0.0
    num_lines_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "camera_id": self.camera_id,
            "zoom_factor": float(self.zoom_factor),
            "distortion": self.distortion.to_dict(),
            "calibration_date": self.calibration_date,
            "validation_rmse": self.validation_rmse,
            "num_lines_used": self.num_lines_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRunRecord:
        """Create CalibrationRunRecord from dictionary."""
        return cls(
            camera_id=str(data["camera_id"]),
            zoom_factor=Unitless(data["zoom_factor"]),
            distortion=LensDistortion.from_dict(data["distortion"]),
            calibration_date=data.get("calibration_date", ""),
            validation_rmse=float(data.get("validation_rmse", 0.0)),
            num_lines_used=int(data.get("num_lines_used", 0)),
        )
