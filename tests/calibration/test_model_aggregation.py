"""Tests for per-model lens-distortion aggregation (offline)."""

from __future__ import annotations

from poc_homography.calibration.lens_distortion.model_aggregation import (
    AggregationConfig,
    aggregate_records,
    build_model_table,
    fold_camera_table,
    zoom_bin,
)
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.entities.model_calibration_table import ModelCalibrationTable
from poc_homography.domain.vo.calibration_run_record import CalibrationRunRecord
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.types import Unitless


def _rec(camera_id: str, zoom: float, k1: float, *, rmse: float = 1.0, lines: int = 20):
    return CalibrationRunRecord(
        camera_id=camera_id,
        zoom_factor=Unitless(zoom),
        distortion=LensDistortion(k1=Unitless(k1), k2=Unitless(0.0)),
        calibration_date="2026-01-01T00:00:00",
        validation_rmse=rmse,
        num_lines_used=lines,
    )


def test_zoom_bin_groups_near_zoom() -> None:
    assert zoom_bin(1.04, 0.5) == 1.0
    assert zoom_bin(1.26, 0.5) == 1.5
    assert zoom_bin(2.0, 0.5) == 2.0


def test_aggregate_bins_and_counts_cameras() -> None:
    records = (
        _rec("camA", 1.0, -0.18),
        _rec("camB", 1.04, -0.20),
        _rec("camC", 1.0, -0.19),
        _rec("camA", 2.0, -0.05),
    )
    entries = aggregate_records(records, AggregationConfig(zoom_step=0.5))
    assert [float(e.zoom_bin) for e in entries] == [1.0, 2.0]
    wide = entries[0]
    assert wide.num_runs == 3
    assert wide.num_cameras == 3
    assert -0.20 < float(wide.distortion.k1) < -0.18
    assert wide.k1_std > 0.0


def test_more_consistent_runs_reduce_spread_relative_to_outlier() -> None:
    # An outlier run is rejected once enough consistent runs exist.
    consistent = [_rec(f"cam{i}", 1.0, -0.18 + 0.001 * i) for i in range(5)]
    with_outlier = (*consistent, _rec("camX", 1.0, 0.9))  # wildly off
    entries = aggregate_records(tuple(with_outlier), AggregationConfig(zoom_step=0.5))
    e = entries[0]
    assert e.num_runs == 5  # outlier dropped
    assert float(e.distortion.k1) < -0.17  # pulled to the consistent cluster, not toward 0.9


def test_weighting_favours_low_rmse_high_lines() -> None:
    records = (
        _rec("camA", 1.0, -0.30, rmse=10.0, lines=10),  # noisy -> low weight
        _rec("camB", 1.0, -0.10, rmse=0.2, lines=40),  # clean -> high weight
    )
    e = aggregate_records(records, AggregationConfig(zoom_step=0.5))[0]
    assert float(e.distortion.k1) > -0.20  # closer to the clean -0.10


def _camera_table(camera_id: str, k1: float) -> LensCalibrationTable:
    entry = ZoomCalibrationEntry(
        zoom_factor=Unitless(1.0),
        distortion=LensDistortion(k1=Unitless(k1), k2=Unitless(0.0)),
        calibration_date="2026-01-01T00:00:00",
        source_images=(),
        validation_rmse=1.0,
        num_lines_used=20,
    )
    return LensCalibrationTable(id=camera_id, entries=(entry,), created_date="t", last_modified="t")


def test_fold_grows_history_and_reaggregates() -> None:
    model = fold_camera_table(None, _camera_table("cam01", -0.18), model_name="MODEL-X", now="t0")
    assert len(model.run_history) == 1
    assert model.created_date == "t0"

    model2 = fold_camera_table(model, _camera_table("cam02", -0.20), model_name="MODEL-X", now="t1")
    assert len(model2.run_history) == 2
    assert model2.created_date == "t0"  # preserved
    assert model2.last_modified == "t1"
    assert model2.entries[0].num_cameras == 2


def test_round_trip_serialization() -> None:
    model = build_model_table(
        "MODEL-X",
        (_rec("camA", 1.0, -0.18), _rec("camB", 1.0, -0.19)),
        now="t",
    )
    restored = ModelCalibrationTable.from_dict(model.to_dict())
    assert restored == model
    assert len(restored.run_history) == 2
    assert float(restored.entries[0].distortion.k1) == float(model.entries[0].distortion.k1)
