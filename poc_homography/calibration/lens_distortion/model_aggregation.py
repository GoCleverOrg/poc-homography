"""Aggregate per-camera distortion runs into a per-model calibration table.

Pure, offline-testable. Bins measurements by zoom, robustly combines the
coefficients of each bin (RMSE/line-count-weighted mean with MAD outlier
rejection), and reports the per-coefficient std so precision is auditable and
visibly improves as more runs accumulate.

The model table is the *shareable default* (tier above per-camera tables): used
when a specific camera unit has not been individually calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.domain.entities.model_calibration_table import ModelCalibrationTable
from poc_homography.domain.vo.calibration_run_record import CalibrationRunRecord
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.model_zoom_coefficient import ModelZoomCoefficient
from poc_homography.types import Unitless

if TYPE_CHECKING:
    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable

_MAD_TO_STD = 1.4826  # MAD -> std for a normal distribution


@dataclass(frozen=True)
class AggregationConfig:
    """Knobs for model aggregation.

    Attributes:
        zoom_step: Bin width (zoom factors are grouped by rounding to this step).
        outlier_mad_k: Reject a run if |k1 - median| > k * MAD-derived std.
        min_runs_for_outlier: Only attempt outlier rejection at/above this count.
        eps_rmse: Added to validation_rmse in the weight denominator (avoids /0).
    """

    zoom_step: float = 0.5
    outlier_mad_k: float = 3.0
    min_runs_for_outlier: int = 4
    eps_rmse: float = 0.1


def zoom_bin(zoom: float, step: float) -> float:
    """Round a zoom factor to its bin centre."""
    if step <= 0:
        raise ValueError(f"zoom_step must be positive, got {step}")
    return round(round(zoom / step) * step, 10)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted mean and weighted population std."""
    wsum = float(weights.sum())
    if wsum <= 0:
        return float(values.mean()), float(values.std())
    mean = float((values * weights).sum() / wsum)
    if len(values) < 2:
        return mean, 0.0
    var = float((weights * (values - mean) ** 2).sum() / wsum)
    return mean, float(np.sqrt(max(var, 0.0)))


def _keep_inliers(k1: np.ndarray, cfg: AggregationConfig) -> np.ndarray:
    """Boolean mask of inliers by MAD on k1 (the dominant coefficient)."""
    if len(k1) < cfg.min_runs_for_outlier:
        return np.ones(len(k1), dtype=bool)
    med = float(np.median(k1))
    mad = float(np.median(np.abs(k1 - med)))
    if mad <= 0:
        return np.ones(len(k1), dtype=bool)
    return np.abs(k1 - med) <= cfg.outlier_mad_k * _MAD_TO_STD * mad


def aggregate_records(
    records: tuple[CalibrationRunRecord, ...],
    config: AggregationConfig | None = None,
) -> tuple[ModelZoomCoefficient, ...]:
    """Bin records by zoom and produce one robust aggregate per bin."""
    cfg = config or AggregationConfig()
    bins: dict[float, list[CalibrationRunRecord]] = {}
    for rec in records:
        bins.setdefault(zoom_bin(float(rec.zoom_factor), cfg.zoom_step), []).append(rec)

    out: list[ModelZoomCoefficient] = []
    for zb in sorted(bins):
        recs = bins[zb]
        k1 = np.array([float(r.distortion.k1) for r in recs])
        keep = _keep_inliers(k1, cfg)
        kept = [r for r, ok in zip(recs, keep) if ok]
        if not kept:
            kept = recs  # never drop everything

        w = np.array([max(r.num_lines_used, 1) / (r.validation_rmse + cfg.eps_rmse) for r in kept])
        coeffs = {
            name: np.array([float(getattr(r.distortion, name)) for r in kept])
            for name in ("k1", "k2", "k3", "p1", "p2")
        }
        means: dict[str, float] = {}
        k1_std = k2_std = 0.0
        for name, vals in coeffs.items():
            mean, std = _weighted_mean_std(vals, w)
            means[name] = mean
            if name == "k1":
                k1_std = std
            elif name == "k2":
                k2_std = std

        out.append(
            ModelZoomCoefficient(
                zoom_bin=Unitless(zb),
                distortion=LensDistortion(
                    k1=Unitless(means["k1"]),
                    k2=Unitless(means["k2"]),
                    p1=Unitless(means["p1"]),
                    p2=Unitless(means["p2"]),
                    k3=Unitless(means["k3"]),
                ),
                k1_std=k1_std,
                k2_std=k2_std,
                num_runs=len(kept),
                num_cameras=len({r.camera_id for r in kept}),
                mean_validation_rmse=float(np.mean([r.validation_rmse for r in kept])),
            )
        )
    return tuple(out)


def records_from_camera_table(
    table: LensCalibrationTable,
    camera_id: str | None = None,
) -> tuple[CalibrationRunRecord, ...]:
    """Convert a per-camera calibration table into provenance run records."""
    cam = camera_id or table.id
    return tuple(
        CalibrationRunRecord(
            camera_id=cam,
            zoom_factor=e.zoom_factor,
            distortion=e.distortion,
            calibration_date=e.calibration_date,
            validation_rmse=e.validation_rmse,
            num_lines_used=e.num_lines_used,
        )
        for e in table.entries
    )


def build_model_table(
    model_name: str,
    records: tuple[CalibrationRunRecord, ...],
    *,
    now: str | None = None,
    config: AggregationConfig | None = None,
) -> ModelCalibrationTable:
    """Build a model table from scratch out of run records."""
    ts = now or datetime.now().isoformat()
    return ModelCalibrationTable(
        model_name=model_name,
        entries=aggregate_records(records, config),
        run_history=tuple(records),
        created_date=ts,
        last_modified=ts,
    )


def fold_camera_table(
    model_table: ModelCalibrationTable | None,
    camera_table: LensCalibrationTable,
    *,
    model_name: str,
    camera_id: str | None = None,
    now: str | None = None,
    config: AggregationConfig | None = None,
) -> ModelCalibrationTable:
    """Append a camera run to the model history and re-aggregate (more = precise).

    Passing ``model_table=None`` starts a fresh model table. The returned table's
    per-bin std reflects all runs seen so far, so it shrinks as consistent runs
    accumulate.
    """
    ts = now or datetime.now().isoformat()
    new_records = records_from_camera_table(camera_table, camera_id)
    history = (model_table.run_history if model_table else ()) + new_records
    created = model_table.created_date if model_table else ts
    return ModelCalibrationTable(
        model_name=model_name,
        entries=aggregate_records(history, config),
        run_history=history,
        created_date=created,
        last_modified=ts,
    )
