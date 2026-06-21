"""Unit tests for the per-camera calibration-confidence aggregation (#366).

These tests exercise the pure join + threshold logic of
:mod:`poc_homography.calibration.confidence` with hand-built persisted-record
fixtures. No DB, no CLI, no recomputation: only aggregation and labeling.
"""

from __future__ import annotations

import numpy as np

from poc_homography.calibration.confidence import build_camera_confidence
from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
from poc_homography.calibration.extrinsic.validation import ReprojectionStats
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry


def _entry(
    zoom: float = 1.0,
    *,
    validation_rmse: float = 0.0,
    reprojection_error_px: float = 0.0,
) -> ZoomCalibrationEntry:
    return ZoomCalibrationEntry(
        zoom_factor=zoom,
        distortion=LensDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        calibration_date="2024-01-01T00:00:00",
        source_images=(),
        validation_rmse=validation_rmse,
        reprojection_error_px=reprojection_error_px,
    )


def _lens_table(*entries: ZoomCalibrationEntry, camera_id: str = "cam1") -> LensCalibrationTable:
    return LensCalibrationTable(
        id=camera_id,
        entries=entries,
        created_date="2024-01-01T00:00:00",
        last_modified="2024-01-01T00:00:00",
    )


def _registration(
    *,
    inlier_ratio: float = 1.0,
    rms_m: float | None = 0.1,
    camera_id: str = "cam1",
    zoom: float = 1.0,
) -> PtzRegistration:
    result = PtzRegistrationResult(
        homography_matrix=np.eye(3, dtype=np.float64),
        inverse_matrix=np.eye(3, dtype=np.float64),
        num_lines=4,
        num_inliers=4,
        inlier_ratio=inlier_ratio,
        mean_perp_error=0.0,
        max_perp_error=0.0,
        rmse=0.0,
        run_id="run1",
        camera_id=camera_id,
        commanded_zoom=zoom,
        commanded_tilt=-15.0,
    )
    stats = (
        ReprojectionStats(rms_m=rms_m, p90_m=rms_m, max_m=rms_m, n_points=4)
        if rms_m is not None
        else None
    )
    return PtzRegistration.from_result(result, reprojection_stats=stats)


def test_good_case() -> None:
    """Low px error, low meters error, high inlier ratio -> Good, high score."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(validation_rmse=1.0, reprojection_error_px=1.5)),
        registrations=[_registration(inlier_ratio=0.95, rms_m=0.1)],
    )
    assert report.calibrated is True
    assert report.label == "Good"
    assert 0.0 <= report.score <= 1.0
    assert report.score > 0.7


def test_fair_boundary() -> None:
    """Worst intrinsic at the Fair threshold downgrades the overall label."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(reprojection_error_px=5.0)),  # == FAIR_INTRINSIC_PX
        registrations=[_registration(inlier_ratio=0.95, rms_m=0.1)],
    )
    assert report.calibrated is True
    assert report.label == "Fair"
    assert 0.0 <= report.score <= 1.0


def test_poor_case_worst_axis_dominates() -> None:
    """A single Poor axis (low inlier ratio) drives the overall label to Poor."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(validation_rmse=1.0)),  # Good intrinsic
        registrations=[_registration(inlier_ratio=0.2, rms_m=0.1)],  # Poor inlier
    )
    assert report.label == "Poor"
    assert 0.0 <= report.score <= 1.0


def test_uncalibrated_when_no_lens_table() -> None:
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=None,
        registrations=[_registration()],
    )
    assert report.calibrated is False
    assert report.label == "Uncalibrated"
    assert report.score == 0.0
    assert report.intrinsic is None
    assert report.extrinsic is not None  # partial summary still attached
    assert any("intrinsic" in note for note in report.notes)


def test_uncalibrated_when_no_validated_registrations() -> None:
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(validation_rmse=1.0)),
        registrations=[_registration(rms_m=None)],  # no reprojection stats
    )
    assert report.calibrated is False
    assert report.label == "Uncalibrated"
    assert report.score == 0.0
    assert report.intrinsic is not None  # partial summary still attached
    assert report.extrinsic is None
    assert any("extrinsic" in note for note in report.notes)


def test_uncalibrated_when_intrinsic_metrics_are_all_zero() -> None:
    """All-zero (unrecorded) intrinsic errors are unmeasured, not perfect."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(validation_rmse=0.0, reprojection_error_px=0.0)),
        registrations=[_registration(inlier_ratio=0.95, rms_m=0.1)],
    )
    assert report.calibrated is False
    assert report.label == "Uncalibrated"
    assert report.score == 0.0
    assert report.intrinsic is None  # zero error is treated as unmeasured
    assert report.extrinsic is not None  # partial summary still attached
    assert any("intrinsic" in note for note in report.notes)


def test_intrinsic_summary_ignores_zero_metric_entries() -> None:
    """Worst/mean intrinsic error skip zero-error (unrecorded) entries."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(
            _entry(zoom=1.0, validation_rmse=0.0),  # unrecorded -> excluded
            _entry(zoom=2.0, validation_rmse=3.0),  # measured
        ),
        registrations=[_registration(inlier_ratio=0.95, rms_m=0.1)],
    )
    assert report.intrinsic is not None
    assert report.intrinsic.num_zoom_entries == 2  # both kept for display
    assert report.intrinsic.worst_rmse_px == 3.0
    assert report.intrinsic.mean_rmse_px == 3.0


def test_uncalibrated_when_no_records_at_all() -> None:
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=None,
        registrations=[],
    )
    assert report.label == "Uncalibrated"
    assert report.score == 0.0
    assert report.intrinsic is None
    assert report.extrinsic is None
    assert len(report.notes) == 2


def test_aggregation_picks_worst_and_mean_across_entries() -> None:
    """Intrinsic summary uses max(error) for worst and the mean across entries."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(
            _entry(zoom=1.0, validation_rmse=1.0, reprojection_error_px=0.5),  # error 1.0
            _entry(zoom=2.0, validation_rmse=2.0, reprojection_error_px=4.0),  # error 4.0
        ),
        registrations=[_registration(inlier_ratio=0.9, rms_m=0.1)],
    )
    assert report.intrinsic is not None
    assert report.intrinsic.num_zoom_entries == 2
    assert report.intrinsic.worst_rmse_px == 4.0
    assert report.intrinsic.mean_rmse_px == (1.0 + 4.0) / 2.0


def test_aggregation_picks_worst_and_min_across_registrations() -> None:
    """Extrinsic summary takes worst meters fields and the minimum inlier ratio."""
    report = build_camera_confidence(
        camera_id="cam1",
        camera_name="Cam 1",
        lens_table=_lens_table(_entry(validation_rmse=1.0)),
        registrations=[
            _registration(inlier_ratio=0.9, rms_m=0.2),
            _registration(inlier_ratio=0.6, rms_m=0.8, zoom=2.0),
            _registration(rms_m=None, zoom=3.0),  # unvalidated -> counted but excluded from stats
        ],
    )
    assert report.extrinsic is not None
    assert report.extrinsic.num_registrations == 3
    assert report.extrinsic.num_validated == 2
    assert report.extrinsic.worst_rms_m == 0.8
    assert report.extrinsic.mean_rms_m == (0.2 + 0.8) / 2.0
    assert report.extrinsic.min_inlier_ratio == 0.6
