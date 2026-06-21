"""Offline tests for the multi-camera rollout driver CLI command (#367).

The rollout is a thin loop over already-merged leaves: it resolves the
``calib_cameras.json`` names to real camera ids, skips the excluded set, runs
leaf-2's per-camera runner, collects leaf-3's confidence per camera, and prints a
rollout summary. These tests exercise the pure name->id mapping/exclusion helper
and the Typer command with stubbed runner + confidence collector -- NO live
MinIO, Neon, camera, or network. Continue-on-error and the failure-driven exit
code are asserted directly.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from poc_homography.calibration.confidence import CameraConfidenceReport
from poc_homography.calibration.extrinsic import HoldoutValidationError
from poc_homography.calibration.extrinsic.validation import ReprojectionStats
from poc_homography.cli import rollout as cmd
from poc_homography.cli.main import app

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_TENANT_CAMERAS = [
    {"id": "icozee-camptz-01", "name": "Cam01", "map_id": "icozee-map"},
    {"id": "icozee-camptz-02", "name": "Cam02", "map_id": "icozee-map"},
    {"id": "icozee-camptz-03", "name": "Cam03", "map_id": "icozee-map"},
    {"id": "icozee-camptz-05", "name": "Cam05", "map_id": "icozee-map"},
]


def _report(
    camera_id: str, camera_name: str, label: str = "Good", score: float = 0.91
) -> CameraConfidenceReport:
    return CameraConfidenceReport(
        camera_id=camera_id,
        camera_name=camera_name,
        calibrated=True,
        label=label,
        score=score,
        intrinsic=None,
        extrinsic=None,
        notes=(),
    )


def _write_cameras_file(tmp_path: Path, *, available: list[dict], excluded: list[dict]) -> Path:
    path = tmp_path / "calib_cameras.json"
    path.write_text(
        json.dumps(
            {
                "available": available,
                "excluded": excluded,
                "model": "Hikvision DS-2DF8425IX-AELW",
                "probed_at": "2026-06-21",
            }
        )
    )
    return path


class _FakeSession:
    """Minimal session object handed to the doubles."""


@contextmanager
def _fake_session_factory() -> Iterator[_FakeSession]:
    yield _FakeSession()


# ---------------------------------------------------------------------------
# DoD 1 + 2: pure name->id mapping + exclusion
# ---------------------------------------------------------------------------


def test_resolve_targets_maps_names_to_ids() -> None:
    spec = {
        "available": [{"name": "cam02"}, {"name": "cam03"}],
        "excluded": [],
    }

    targets, annotations = cmd._resolve_targets(spec, _TENANT_CAMERAS)

    assert annotations == []
    assert [(t.json_name, t.camera_id, t.camera_name) for t in targets] == [
        ("cam02", "icozee-camptz-02", "Cam02"),
        ("cam03", "icozee-camptz-03", "Cam03"),
    ]


def test_resolve_targets_skips_excluded_by_name() -> None:
    spec = {
        "available": [{"name": "cam02"}, {"name": "cam05"}],
        "excluded": [{"name": "cam05", "reason": "taken"}],
    }

    targets, annotations = cmd._resolve_targets(spec, _TENANT_CAMERAS)

    assert [t.json_name for t in targets] == ["cam02"]
    skipped = [a for a in annotations if a.skipped]
    assert [a.json_name for a in skipped] == ["cam05"]
    assert "taken" in skipped[0].reason


def test_resolve_targets_records_unresolved_name() -> None:
    spec = {"available": [{"name": "cam99"}], "excluded": []}

    targets, annotations = cmd._resolve_targets(spec, _TENANT_CAMERAS)

    assert targets == []
    assert len(annotations) == 1
    assert annotations[0].json_name == "cam99"
    assert annotations[0].skipped is False  # unresolved is a failure, not a skip
    assert "no matching tenant camera" in annotations[0].reason


def test_resolve_targets_unparseable_name_distinct_reason() -> None:
    spec = {"available": [{"name": "cam2b"}], "excluded": []}

    targets, annotations = cmd._resolve_targets(spec, _TENANT_CAMERAS)

    assert targets == []
    assert len(annotations) == 1
    assert annotations[0].json_name == "cam2b"
    assert annotations[0].skipped is False  # still a failure, not a skip
    assert "cannot parse camera index from 'cam2b'" in annotations[0].reason


def test_resolve_targets_dedups_duplicate_available() -> None:
    spec = {
        "available": [{"name": "cam02"}, {"name": "cam02"}],
        "excluded": [],
    }

    targets, annotations = cmd._resolve_targets(spec, _TENANT_CAMERAS)

    assert annotations == []
    assert [t.json_name for t in targets] == ["cam02"]  # deduped: runs once


# ---------------------------------------------------------------------------
# CLI wiring helpers
# ---------------------------------------------------------------------------


def _wire_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    monkeypatch.setattr(cmd.MinioFrameStore, "from_env", classmethod(lambda _cls: object()))
    monkeypatch.setattr(cmd, "_load_ortho_context", lambda _map_id: ([], object()))
    monkeypatch.setattr(cmd, "get_cameras_for_tenant", lambda _t: list(_TENANT_CAMERAS))
    monkeypatch.setattr(cmd, "get_session", _fake_session_factory)


# ---------------------------------------------------------------------------
# DoD 4 + 5: all-pass, summary lists PASS + confidence label
# ---------------------------------------------------------------------------


def test_command_all_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(cmd, "_run_camera_registration", lambda **_kwargs: ["rec"])
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(
        tmp_path,
        available=[{"name": "cam02"}, {"name": "cam03"}],
        excluded=[{"name": "cam05", "reason": "taken"}],
    )

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code == 0, result.output
    assert "cam02" in result.output
    assert "icozee-camptz-02" in result.output
    assert "PASS" in result.output
    assert "Good" in result.output
    assert "Rollout: 2 passed, 0 failed" in result.output


# ---------------------------------------------------------------------------
# DoD 3: continue-on-error, one camera fails, exit code non-zero
# ---------------------------------------------------------------------------


def test_command_continue_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _wire_cli(monkeypatch)

    def runner(**kwargs: object) -> list[str]:
        if kwargs["camera_id"] == "icozee-camptz-02":
            stats = ReprojectionStats(rms_m=2.0, p90_m=2.0, max_m=2.0, n_points=4)
            raise HoldoutValidationError(2.0, 1.0, stats)
        return ["rec"]

    monkeypatch.setattr(cmd, "_run_camera_registration", runner)
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(
        tmp_path,
        available=[{"name": "cam02"}, {"name": "cam03"}],
        excluded=[],
    )

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code != 0
    # The loop continued: the healthy camera still ran and passed.
    assert "icozee-camptz-03" in result.output
    assert "PASS" in result.output
    assert "FAIL" in result.output
    assert "Rollout: 1 passed, 1 failed" in result.output


def test_command_excluded_skip_in_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(cmd, "_run_camera_registration", lambda **_kwargs: ["rec"])
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(
        tmp_path,
        available=[{"name": "cam02"}, {"name": "cam05"}],
        excluded=[{"name": "cam05", "reason": "taken (user-reserved)"}],
    )

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output
    assert "taken" in result.output
    assert "Rollout: 1 passed, 0 failed, 1 skipped" in result.output


def test_command_missing_cameras_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _wire_cli(monkeypatch)
    missing = tmp_path / "nope.json"

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(missing)],
    )

    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "failed to load" in result.output.lower()


# ---------------------------------------------------------------------------
# Fix 1: nothing-attempted (empty / fully-excluded available) is not success
# ---------------------------------------------------------------------------


def test_command_empty_available_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire_cli(monkeypatch)
    cameras_file = _write_cameras_file(tmp_path, available=[], excluded=[])

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code != 0
    assert "no cameras to run" in result.output.lower()


# ---------------------------------------------------------------------------
# Fix 2: a leaf raising typer.Exit(1) FAILs the camera with the code, loop goes on
# ---------------------------------------------------------------------------


def test_command_leaf_typer_exit_fails_camera(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire_cli(monkeypatch)

    def runner(**kwargs: object) -> list[str]:
        if kwargs["camera_id"] == "icozee-camptz-02":
            raise cmd.typer.Exit(1)
        return ["rec"]

    monkeypatch.setattr(cmd, "_run_camera_registration", runner)
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(
        tmp_path,
        available=[{"name": "cam02"}, {"name": "cam03"}],
        excluded=[],
    )

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code != 0
    assert "leaf exited with code 1" in result.output
    # The loop continued: the healthy camera still ran and passed.
    assert "icozee-camptz-03" in result.output
    assert "Rollout: 1 passed, 1 failed" in result.output


# ---------------------------------------------------------------------------
# Fix 4: coverage for diff-introduced branches
# ---------------------------------------------------------------------------


def test_command_camera_without_map_id_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(
        cmd, "get_cameras_for_tenant", lambda _t: [{"id": "icozee-camptz-02", "name": "Cam02"}]
    )
    monkeypatch.setattr(cmd, "_run_camera_registration", lambda **_kwargs: ["rec"])
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(tmp_path, available=[{"name": "cam02"}], excluded=[])

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code != 0
    assert "no map_id configured" in result.output
    assert "Rollout: 0 passed, 1 failed" in result.output


def test_load_cameras_file_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "calib_cameras.json"
    path.write_text("{not valid json")

    with pytest.raises(cmd.typer.Exit) as excinfo:
        cmd._load_cameras_file(path)

    assert excinfo.value.exit_code == 1


def test_load_cameras_file_top_level_list(tmp_path: Path) -> None:
    path = tmp_path / "calib_cameras.json"
    path.write_text(json.dumps([{"name": "cam02"}]))

    with pytest.raises(cmd.typer.Exit) as excinfo:
        cmd._load_cameras_file(path)

    assert excinfo.value.exit_code == 1


def test_command_unresolved_name_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(cmd, "_run_camera_registration", lambda **_kwargs: ["rec"])
    monkeypatch.setattr(
        cmd,
        "_collect_confidence",
        lambda camera_id, camera_name, **_kwargs: _report(camera_id, camera_name),
    )
    cameras_file = _write_cameras_file(
        tmp_path,
        available=[{"name": "cam02"}, {"name": "cam99"}],
        excluded=[],
    )

    result = CliRunner().invoke(
        app,
        ["calibrate", "rollout", "--tenant", "icozee", "--cameras-file", str(cameras_file)],
    )

    assert result.exit_code != 0
    assert "cam99" in result.output
    assert "FAIL" in result.output
    assert "Rollout: 1 passed, 1 failed" in result.output
