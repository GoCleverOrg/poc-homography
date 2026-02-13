"""Zoom-level calibration entry value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.types import PixelsFloat, Unitless


@dataclass(frozen=True)
class ZoomCalibrationEntry:
    """Calibration data for a specific zoom level.

    Immutable value object that stores distortion coefficients and metadata
    for a single zoom level within a lens calibration table.
    """

    zoom_factor: Unitless
    distortion: LensDistortion
    calibration_date: str
    source_images: tuple[str, ...]
    validation_rmse: float = 0.0
    num_lines_used: int = 0
    fx: PixelsFloat = PixelsFloat(0.0)  # noqa: RUF009
    fy: PixelsFloat = PixelsFloat(0.0)  # noqa: RUF009
    cx: PixelsFloat = PixelsFloat(0.0)  # noqa: RUF009
    cy: PixelsFloat = PixelsFloat(0.0)  # noqa: RUF009
    reprojection_error_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "zoom_factor": float(self.zoom_factor),
            "distortion": self.distortion.to_dict(),
            "calibration_date": self.calibration_date,
            "source_images": list(self.source_images),
            "validation_rmse": self.validation_rmse,
            "num_lines_used": self.num_lines_used,
        }
        if self.fx != 0.0 or self.fy != 0.0:
            d["fx"] = float(self.fx)
            d["fy"] = float(self.fy)
            d["cx"] = float(self.cx)
            d["cy"] = float(self.cy)
        if self.reprojection_error_px != 0.0:
            d["reprojection_error_px"] = self.reprojection_error_px
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZoomCalibrationEntry:
        """Create ZoomCalibrationEntry from dictionary."""
        source_images = data.get("source_images", [])
        if isinstance(source_images, list):
            source_images = tuple(source_images)
        elif isinstance(source_images, str):
            source_images = (source_images,)

        return cls(
            zoom_factor=Unitless(data["zoom_factor"]),
            distortion=LensDistortion.from_dict(data["distortion"]),
            calibration_date=data.get("calibration_date", ""),
            source_images=source_images,
            validation_rmse=float(data.get("validation_rmse", 0.0)),
            num_lines_used=int(data.get("num_lines_used", 0)),
            fx=PixelsFloat(data.get("fx", 0.0)),
            fy=PixelsFloat(data.get("fy", 0.0)),
            cx=PixelsFloat(data.get("cx", 0.0)),
            cy=PixelsFloat(data.get("cy", 0.0)),
            reprojection_error_px=float(data.get("reprojection_error_px", 0.0)),
        )
