"""Offline tests for the per-camera calibration-confidence report CLI (#366).

These tests exercise the read-only intrinsic⋈extrinsic join (``_collect_reports``)
and the Typer command (human + JSON, single-camera + tenant) with fake repos and
a fake session factory -- NO live Neon access and no recomputation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pytest  # noqa: TC002  (used only in test-param annotations under future-annotations)
from typer.testing import CliRunner

from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
from poc_homography.calibration.extrinsic.validation import ReprojectionStats
from poc_homography.cli import confidence_report as cmd
from poc_homography.cli.main import app
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry

if TYPE_CHECKING:
    from collections.abc import Iterator


def _lens_table(camera_id: str) -> LensCalibrationTable:
    entry = ZoomCalibrationEntry(
        zoom_factor=1.0,
        distortion=LensDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        calibration_date="2024-01-01T00:00:00",
        source_images=(),
        validation_rmse=1.0,
        reprojection_error_px=1.0,
    )
    return LensCalibrationTable(
        id=camera_id,
        entries=(entry,),
        created_date="2024-01-01T00:00:00",
        last_modified="2024-01-01T00:00:00",
    )


def _registration(camera_id: str) -> PtzRegistration:
    result = PtzRegistrationResult(
        homography_matrix=np.eye(3, dtype=np.float64),
        inverse_matrix=np.eye(3, dtype=np.float64),
        num_lines=4,
        num_inliers=4,
        inlier_ratio=0.95,
        mean_perp_error=0.0,
        max_perp_error=0.0,
        rmse=0.0,
        run_id="run1",
        camera_id=camera_id,
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
    )
    stats = ReprojectionStats(rms_m=0.1, p90_m=0.1, max_m=0.1, n_points=4)
    return PtzRegistration.from_result(result, reprojection_stats=stats)


class _FakeLensRepo:
    """RepoPostgresLensCalibrationTable double keyed by camera id."""

    def __init__(self, tables: dict[str, LensCalibrationTable]) -> None:
        self._tables = tables

    def get(self, camera_id: str) -> LensCalibrationTable | None:
        return self._tables.get(camera_id)


class _FakePtzRepo:
    """RepoPostgresPtzRegistration double keyed by camera id."""

    def __init__(self, regs: dict[str, list[PtzRegistration]]) -> None:
        self._regs = regs

    def get_by_camera(self, camera_id: str) -> list[PtzRegistration]:
        return self._regs.get(camera_id, [])


class _FakeSession:
    """Minimal session object handed to the repository doubles."""


@contextmanager
def _fake_session_factory() -> Iterator[_FakeSession]:
    yield _FakeSession()


# ---------------------------------------------------------------------------
# _collect_reports: the read-only join
# ---------------------------------------------------------------------------


def test_collect_reports_joins_intrinsic_and_extrinsic() -> None:
    lens_repo = _FakeLensRepo({"cam1": _lens_table("cam1")})
    ptz_repo = _FakePtzRepo({"cam1": [_registration("cam1")]})

    reports = cmd._collect_reports([("cam1", "Cam 1")], lens_repo=lens_repo, ptz_repo=ptz_repo)

    assert len(reports) == 1
    assert reports[0].camera_id == "cam1"
    assert reports[0].calibrated is True
    assert reports[0].label == "Good"


def test_collect_reports_marks_uncalibrated_when_missing_half() -> None:
    lens_repo = _FakeLensRepo({})  # no intrinsics
    ptz_repo = _FakePtzRepo({"cam1": [_registration("cam1")]})

    reports = cmd._collect_reports([("cam1", "Cam 1")], lens_repo=lens_repo, ptz_repo=ptz_repo)

    assert reports[0].calibrated is False
    assert reports[0].label == "Uncalibrated"


# ---------------------------------------------------------------------------
# CLI command: --camera human + json
# ---------------------------------------------------------------------------


def _patch_repos(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tables: dict[str, LensCalibrationTable],
    regs: dict[str, list[PtzRegistration]],
) -> None:
    """Wire the module-level repo classes + session factory to offline doubles."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    monkeypatch.setattr(cmd, "RepoPostgresLensCalibrationTable", lambda _s: _FakeLensRepo(tables))
    monkeypatch.setattr(cmd, "RepoPostgresPtzRegistration", lambda _s: _FakePtzRepo(regs))
    monkeypatch.setattr(cmd, "get_session", _fake_session_factory)


def test_command_camera_human(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_repos(
        monkeypatch,
        tables={"cam1": _lens_table("cam1")},
        regs={"cam1": [_registration("cam1")]},
    )
    monkeypatch.setattr(cmd, "get_camera_by_name", lambda _n: {"id": "cam1", "name": "Cam 1"})

    result = CliRunner().invoke(app, ["calibrate", "confidence-report", "--camera", "cam1"])

    assert result.exit_code == 0, result.output
    assert "CALIBRATED" in result.output
    assert "Good" in result.output
    assert "cam1" in result.output


def test_command_camera_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_repos(
        monkeypatch,
        tables={"cam1": _lens_table("cam1")},
        regs={"cam1": [_registration("cam1")]},
    )
    monkeypatch.setattr(cmd, "get_camera_by_name", lambda _n: {"id": "cam1", "name": "Cam 1"})

    result = CliRunner().invoke(
        app, ["calibrate", "confidence-report", "--camera", "cam1", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["camera_id"] == "cam1"
    assert payload["calibrated"] is True
    assert payload["label"] == "Good"
    assert "tenant" not in payload  # single-camera JSON is a bare dict


# ---------------------------------------------------------------------------
# CLI command: --tenant reports all, marking uncalibrated ones
# ---------------------------------------------------------------------------


def test_command_tenant_marks_uncalibrated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_repos(
        monkeypatch,
        tables={"cam1": _lens_table("cam1")},  # cam2 has no intrinsics
        regs={"cam1": [_registration("cam1")], "cam2": [_registration("cam2")]},
    )
    monkeypatch.setattr(
        cmd,
        "get_cameras_for_tenant",
        lambda _t: [{"id": "cam1", "name": "Cam 1"}, {"id": "cam2", "name": "Cam 2"}],
    )

    result = CliRunner().invoke(
        app, ["calibrate", "confidence-report", "--tenant", "icozee", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tenant"] == "icozee"
    labels = {c["camera_id"]: c["label"] for c in payload["cameras"]}
    assert labels["cam1"] == "Good"
    assert labels["cam2"] == "Uncalibrated"


def test_command_tenant_human_marks_uncalibrated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_repos(
        monkeypatch,
        tables={"cam1": _lens_table("cam1")},
        regs={"cam1": [_registration("cam1")], "cam2": [_registration("cam2")]},
    )
    monkeypatch.setattr(
        cmd,
        "get_cameras_for_tenant",
        lambda _t: [{"id": "cam1", "name": "Cam 1"}, {"id": "cam2", "name": "Cam 2"}],
    )

    result = CliRunner().invoke(app, ["calibrate", "confidence-report", "--tenant", "icozee"])

    assert result.exit_code == 0, result.output
    assert "UNCALIBRATED" in result.output
    assert "CALIBRATED" in result.output


# ---------------------------------------------------------------------------
# CLI command: argument validation
# ---------------------------------------------------------------------------


def test_command_requires_exactly_one_target_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    result = CliRunner().invoke(app, ["calibrate", "confidence-report"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_command_requires_exactly_one_target_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    result = CliRunner().invoke(
        app, ["calibrate", "confidence-report", "--camera", "cam1", "--tenant", "icozee"]
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output
