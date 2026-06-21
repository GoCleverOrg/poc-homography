"""Per-camera calibration-confidence aggregation (#366).

This module is the canonical, freshly-authored aggregation that JOINS a camera's
already-persisted intrinsic calibration (its
:class:`~poc_homography.domain.entities.lens_calibration_table.LensCalibrationTable`)
with its already-persisted extrinsic registration records (its
:class:`~poc_homography.domain.entities.ptz_registration.PtzRegistration` list)
into ONE per-camera confidence record carrying a derived Good/Fair/Poor label
and a [0, 1] score.

It is **read-only**: it recomputes NO calibration, re-solves NO homography, and
reads NO camera, MinIO, or database directly. It consumes whatever the
repositories already returned and summarizes it. Cameras missing either half of
the calibration are reported as ``"Uncalibrated"`` with a zero score.

The Good/Fair/Poor vocabulary is reused verbatim from
:mod:`poc_homography.validation.gcp_distribution`; ``"Uncalibrated"`` is used for
cameras with no persisted calibration to summarize.

Scoring
-------
Three independent quality axes are derived from the persisted summaries:

  * Intrinsic axis, from the worst per-zoom intrinsic error in pixels. The
    per-zoom error of an entry is ``max(validation_rmse, reprojection_error_px)``.
    A persisted entry can carry a ``0.0`` error simply because no metric was
    recorded (both fields default to ``0.0``); a physically-implausible perfect
    ``0.0`` is therefore treated as *unmeasured* and excluded from the summary,
    so a camera whose entries carry no positive error is reported
    ``Uncalibrated`` rather than misreported as perfect.
  * Extrinsic axis, from the worst hold-out reprojection RMS in ground meters.
  * Inlier axis, from the minimum RANSAC/ICP inlier ratio across the validated
    registrations (those carrying hold-out reprojection stats).

The overall ``label`` is the WORST band across the three axes (order
Poor < Fair < Good). The overall ``score`` in [0, 1] is the mean of three
clamped sub-scores:

    intr_sub   = clamp01(1 - worst_rmse_px / FAIR_INTRINSIC_PX)
    extr_sub   = clamp01(1 - worst_rms_m / FAIR_EXTRINSIC_M)
    inlier_sub = clamp01(min_inlier_ratio)

Thresholds (freshly authored)
-----------------------------
    Intrinsic px  : worst_rmse_px <= GOOD_INTRINSIC_PX (2.0)  -> Good
                    worst_rmse_px <= FAIR_INTRINSIC_PX (5.0)  -> Fair
                    otherwise                                 -> Poor
    Extrinsic m   : worst_rms_m   <= GOOD_EXTRINSIC_M (0.25)  -> Good
                    worst_rms_m   <= FAIR_EXTRINSIC_M (1.0)   -> Fair
                    otherwise                                 -> Poor
    Inlier ratio  : min_inlier_ratio >= GOOD_INLIER_RATIO (0.8) -> Good
                    min_inlier_ratio >= FAIR_INLIER_RATIO (0.5) -> Fair
                    otherwise                                    -> Poor

The intrinsic Fair band is anchored on the project-wide
:data:`~poc_homography.calibration.TARGET_ERROR_THRESHOLD_PX` (5.0 px).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from poc_homography.calibration import TARGET_ERROR_THRESHOLD_PX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
    from poc_homography.domain.entities.ptz_registration import PtzRegistration

# --- Quality labels (reused verbatim from gcp_distribution) -------------------

#: Best intrinsic+extrinsic confidence band.
GOOD = "Good"
#: Acceptable but not best confidence band.
FAIR = "Fair"
#: Below-acceptable confidence band.
POOR = "Poor"
#: Label for a camera missing persisted intrinsic and/or extrinsic calibration.
UNCALIBRATED = "Uncalibrated"

#: Ordered worst-to-best so the overall label can pick the worst axis.
_LABEL_RANK: dict[str, int] = {POOR: 0, FAIR: 1, GOOD: 2}

# --- Intrinsic thresholds (pixels) --------------------------------------------

#: worst per-zoom intrinsic error at/below this (px) -> Good intrinsic axis.
GOOD_INTRINSIC_PX = 2.0
#: worst per-zoom intrinsic error at/below this (px) -> Fair; anchored on the
#: project-wide target reprojection-error threshold (5.0 px).
FAIR_INTRINSIC_PX = TARGET_ERROR_THRESHOLD_PX

# --- Extrinsic thresholds (ground meters) -------------------------------------

#: worst hold-out reprojection RMS at/below this (m) -> Good extrinsic axis.
GOOD_EXTRINSIC_M = 0.25
#: worst hold-out reprojection RMS at/below this (m) -> Fair extrinsic axis.
FAIR_EXTRINSIC_M = 1.0

# --- Inlier-ratio thresholds (unitless) ---------------------------------------

#: minimum inlier ratio at/above this -> Good inlier axis.
GOOD_INLIER_RATIO = 0.8
#: minimum inlier ratio at/above this -> Fair inlier axis.
FAIR_INLIER_RATIO = 0.5


@dataclass(frozen=True)
class ZoomIntrinsicPoint:
    """One persisted per-zoom intrinsic-calibration data point.

    Attributes:
        zoom_factor: Zoom level the entry was calibrated at.
        validation_rmse: Persisted intrinsic validation RMSE in pixels.
        reprojection_error_px: Persisted reprojection error in pixels.
    """

    zoom_factor: float
    validation_rmse: float
    reprojection_error_px: float


@dataclass(frozen=True)
class IntrinsicSummary:
    """Aggregated view of a camera's persisted intrinsic calibration.

    Attributes:
        num_zoom_entries: Number of per-zoom calibration entries.
        worst_rmse_px: Worst per-entry intrinsic error in pixels.
        mean_rmse_px: Mean per-entry intrinsic error in pixels.
        per_zoom: The per-zoom data points the summary was computed over.
    """

    num_zoom_entries: int
    worst_rmse_px: float
    mean_rmse_px: float
    per_zoom: tuple[ZoomIntrinsicPoint, ...]


@dataclass(frozen=True)
class ExtrinsicSummary:
    """Aggregated view of a camera's persisted extrinsic registrations.

    Computed over the *validated* registrations only (those carrying
    meters-unit reprojection stats).

    Attributes:
        num_registrations: Total persisted registrations for the camera.
        num_validated: Registrations carrying hold-out reprojection stats.
        worst_rms_m: Worst hold-out reprojection RMS in meters.
        worst_p90_m: Worst hold-out 90th-percentile error in meters.
        worst_max_m: Worst hold-out maximum error in meters.
        mean_rms_m: Mean hold-out reprojection RMS in meters.
        min_inlier_ratio: Minimum RANSAC/ICP inlier ratio across validated regs.
    """

    num_registrations: int
    num_validated: int
    worst_rms_m: float
    worst_p90_m: float
    worst_max_m: float
    mean_rms_m: float
    min_inlier_ratio: float


@dataclass(frozen=True)
class CameraConfidenceReport:
    """A per-camera calibration-confidence record (intrinsic ⋈ extrinsic).

    Attributes:
        camera_id: Camera identifier (the intrinsic/extrinsic join key).
        camera_name: Human-readable camera name.
        calibrated: True iff BOTH an intrinsic summary and an extrinsic summary
            are present.
        label: Overall confidence label (Good/Fair/Poor, or Uncalibrated).
        score: Overall confidence score in [0, 1] (0.0 when uncalibrated).
        intrinsic: Intrinsic summary, or None when no persisted intrinsics.
        extrinsic: Extrinsic summary, or None when no validated registrations.
        notes: Human-readable notes (e.g. what is missing).
    """

    camera_id: str
    camera_name: str
    calibrated: bool
    label: str
    score: float
    intrinsic: IntrinsicSummary | None
    extrinsic: ExtrinsicSummary | None
    notes: tuple[str, ...]


def _clamp01(value: float) -> float:
    """Clamp a float to the [0, 1] range."""
    return max(0.0, min(1.0, value))


def _worst_label(labels: Sequence[str]) -> str:
    """Return the worst label across an axis set (order Poor < Fair < Good)."""
    return min(labels, key=lambda label: _LABEL_RANK[label])


def _entry_error_px(validation_rmse: float, reprojection_error_px: float) -> float:
    """Per-zoom intrinsic error of an entry = ``max`` of its two error fields."""
    return max(float(validation_rmse), float(reprojection_error_px))


def _intrinsic_label(worst_rmse_px: float) -> str:
    """Map a worst intrinsic error (px) to a Good/Fair/Poor band."""
    if worst_rmse_px <= GOOD_INTRINSIC_PX:
        return GOOD
    if worst_rmse_px <= FAIR_INTRINSIC_PX:
        return FAIR
    return POOR


def _extrinsic_label(worst_rms_m: float) -> str:
    """Map a worst extrinsic RMS (m) to a Good/Fair/Poor band."""
    if worst_rms_m <= GOOD_EXTRINSIC_M:
        return GOOD
    if worst_rms_m <= FAIR_EXTRINSIC_M:
        return FAIR
    return POOR


def _inlier_label(min_inlier_ratio: float) -> str:
    """Map a minimum inlier ratio to a Good/Fair/Poor band."""
    if min_inlier_ratio >= GOOD_INLIER_RATIO:
        return GOOD
    if min_inlier_ratio >= FAIR_INLIER_RATIO:
        return FAIR
    return POOR


def _build_intrinsic_summary(lens_table: LensCalibrationTable) -> IntrinsicSummary | None:
    """Aggregate a non-empty lens table into an :class:`IntrinsicSummary`.

    Worst/mean are computed over entries with a *positive* per-zoom error only:
    a ``0.0`` error means the metric was not recorded (both source fields default
    to ``0.0``), not a perfect calibration. Returns None when no entry carries a
    positive error, so the camera is reported as uncalibrated rather than scored
    as perfect. ``per_zoom`` still carries every entry for display.
    """
    per_zoom = tuple(
        ZoomIntrinsicPoint(
            zoom_factor=float(entry.zoom_factor),
            validation_rmse=float(entry.validation_rmse),
            reprojection_error_px=float(entry.reprojection_error_px),
        )
        for entry in lens_table.entries
    )
    measured = [
        error
        for point in per_zoom
        if (error := _entry_error_px(point.validation_rmse, point.reprojection_error_px)) > 0.0
    ]
    if not measured:
        return None
    return IntrinsicSummary(
        num_zoom_entries=len(per_zoom),
        worst_rmse_px=max(measured),
        mean_rmse_px=sum(measured) / len(measured),
        per_zoom=per_zoom,
    )


def _build_extrinsic_summary(
    registrations: Sequence[PtzRegistration],
    validated: Sequence[PtzRegistration],
) -> ExtrinsicSummary:
    """Aggregate validated registrations into an :class:`ExtrinsicSummary`."""
    # Validated regs are guaranteed to carry reprojection_stats by the caller.
    rms = [float(reg.reprojection_stats.rms_m) for reg in validated]  # type: ignore[union-attr]
    p90 = [float(reg.reprojection_stats.p90_m) for reg in validated]  # type: ignore[union-attr]
    max_m = [float(reg.reprojection_stats.max_m) for reg in validated]  # type: ignore[union-attr]
    inliers = [float(reg.registration.inlier_ratio) for reg in validated]
    return ExtrinsicSummary(
        num_registrations=len(registrations),
        num_validated=len(validated),
        worst_rms_m=max(rms),
        worst_p90_m=max(p90),
        worst_max_m=max(max_m),
        mean_rms_m=sum(rms) / len(rms),
        min_inlier_ratio=min(inliers),
    )


def build_camera_confidence(
    *,
    camera_id: str,
    camera_name: str,
    lens_table: LensCalibrationTable | None,
    registrations: Sequence[PtzRegistration],
) -> CameraConfidenceReport:
    """Join persisted intrinsics with persisted extrinsic stats for one camera.

    This is a pure read-only aggregation: it summarizes whatever the
    repositories already returned and recomputes no calibration. A camera is
    ``calibrated`` only when it has BOTH a non-empty intrinsic calibration and at
    least one validated extrinsic registration; otherwise it is reported as
    ``"Uncalibrated"`` with a zero score (the partial summary, if any, is still
    attached and the missing half is named in ``notes``).

    Args:
        camera_id: Camera identifier (the join key).
        camera_name: Human-readable camera name.
        lens_table: The camera's persisted lens calibration table, or None.
        registrations: The camera's persisted extrinsic registration records.

    Returns:
        A populated :class:`CameraConfidenceReport`.
    """
    intrinsic = (
        _build_intrinsic_summary(lens_table)
        if lens_table is not None and lens_table.entries
        else None
    )

    validated = [reg for reg in registrations if reg.reprojection_stats is not None]
    extrinsic = _build_extrinsic_summary(registrations, validated) if len(validated) >= 1 else None

    calibrated = intrinsic is not None and extrinsic is not None
    if not calibrated:
        notes: list[str] = []
        if intrinsic is None:
            notes.append("no persisted intrinsic calibration metric")
        if extrinsic is None:
            notes.append("no persisted extrinsic validation records")
        return CameraConfidenceReport(
            camera_id=camera_id,
            camera_name=camera_name,
            calibrated=False,
            label=UNCALIBRATED,
            score=0.0,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            notes=tuple(notes),
        )

    # Both summaries present (narrows the Optionals for type-checkers).
    assert intrinsic is not None
    assert extrinsic is not None

    axis_labels = (
        _intrinsic_label(intrinsic.worst_rmse_px),
        _extrinsic_label(extrinsic.worst_rms_m),
        _inlier_label(extrinsic.min_inlier_ratio),
    )
    label = _worst_label(axis_labels)

    intr_sub = _clamp01(1.0 - intrinsic.worst_rmse_px / FAIR_INTRINSIC_PX)
    extr_sub = _clamp01(1.0 - extrinsic.worst_rms_m / FAIR_EXTRINSIC_M)
    inlier_sub = _clamp01(extrinsic.min_inlier_ratio)
    score = (intr_sub + extr_sub + inlier_sub) / 3.0

    return CameraConfidenceReport(
        camera_id=camera_id,
        camera_name=camera_name,
        calibrated=True,
        label=label,
        score=score,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        notes=(),
    )


__all__ = [
    "FAIR",
    "FAIR_EXTRINSIC_M",
    "FAIR_INLIER_RATIO",
    "FAIR_INTRINSIC_PX",
    "GOOD",
    "GOOD_EXTRINSIC_M",
    "GOOD_INLIER_RATIO",
    "GOOD_INTRINSIC_PX",
    "POOR",
    "UNCALIBRATED",
    "CameraConfidenceReport",
    "ExtrinsicSummary",
    "IntrinsicSummary",
    "ZoomIntrinsicPoint",
    "build_camera_confidence",
]
