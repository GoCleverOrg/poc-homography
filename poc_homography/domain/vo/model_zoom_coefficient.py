"""Aggregated distortion coefficient for a camera *model* at one zoom bin.

This is the shareable per-model default (robust mean across all per-camera runs
in the bin) plus the precision it was determined with (per-coefficient std and
the contributing run/camera counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.types import Unitless


@dataclass(frozen=True)
class ModelZoomCoefficient:
    """Robust per-model distortion aggregate at one zoom bin.

    Attributes:
        zoom_bin: Bin centre (zoom factor) this aggregate applies to.
        distortion: Robust (weighted-mean) distortion coefficients.
        k1_std: Std of contributing k1 values (precision; shrinks with runs).
        k2_std: Std of contributing k2 values.
        num_runs: Number of measurements that contributed (after outlier reject).
        num_cameras: Number of distinct camera units that contributed.
        mean_validation_rmse: Mean held-out RMSE of contributing measurements.
    """

    zoom_bin: Unitless
    distortion: LensDistortion
    k1_std: float
    k2_std: float
    num_runs: int
    num_cameras: int
    mean_validation_rmse: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "zoom_bin": float(self.zoom_bin),
            "distortion": self.distortion.to_dict(),
            "k1_std": self.k1_std,
            "k2_std": self.k2_std,
            "num_runs": self.num_runs,
            "num_cameras": self.num_cameras,
            "mean_validation_rmse": self.mean_validation_rmse,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelZoomCoefficient:
        """Create ModelZoomCoefficient from dictionary."""
        return cls(
            zoom_bin=Unitless(data["zoom_bin"]),
            distortion=LensDistortion.from_dict(data["distortion"]),
            k1_std=float(data.get("k1_std", 0.0)),
            k2_std=float(data.get("k2_std", 0.0)),
            num_runs=int(data.get("num_runs", 0)),
            num_cameras=int(data.get("num_cameras", 0)),
            mean_validation_rmse=float(data.get("mean_validation_rmse", 0.0)),
        )
