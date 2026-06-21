"""Offline round-trip tests for the PtzRegistration repositories (#364).

These tests need no live database: the Postgres repo is exercised against an
in-memory SQLite engine (the model's JSONB columns carry a SQLite ``JSON``
variant), and the YAML repo against a ``tmp_path`` directory. They verify that a
``PtzRegistrationResult`` plus an optional ``ReprojectionStats`` survive the
save/get cycle, that camera-scoped lookups return the right records, and that a
None reprojection summary round-trips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from sqlalchemy import Table, create_engine
from sqlalchemy.orm import Session

from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
from poc_homography.calibration.extrinsic.validation import ReprojectionStats
from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.infrastructure.models.ptz_registration import PtzRegistrationModel
from poc_homography.infrastructure.repositories import (
    RepoPostgresPtzRegistration,
    RepoYamlPtzRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make_result(
    camera_id: str = "map01/cam01",
    run_id: str | None = "run_a",
    zoom: float = 2.0,
    tilt: float = 15.0,
) -> PtzRegistrationResult:
    homography = np.array([[1.2, 0.1, 5.0], [0.0, 1.1, 7.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inverse = np.linalg.inv(homography)
    return PtzRegistrationResult(
        homography_matrix=homography,
        inverse_matrix=inverse,
        num_lines=4,
        num_inliers=3,
        inlier_ratio=0.75,
        mean_perp_error=1.2,
        max_perp_error=3.4,
        rmse=1.5,
        run_id=run_id,
        camera_id=camera_id,
        commanded_zoom=zoom,
        commanded_tilt=tilt,
    )


def _make_stats() -> ReprojectionStats:
    return ReprojectionStats(rms_m=0.12, p90_m=0.2, max_m=0.31, n_points=8)


def _assert_round_trip(loaded: PtzRegistration, original: PtzRegistration) -> None:
    assert loaded is not None
    assert loaded.id == original.id
    assert loaded.camera_id == original.camera_id
    assert loaded.run_id == original.run_id
    assert loaded.commanded_zoom == pytest.approx(original.commanded_zoom)
    assert loaded.commanded_tilt == pytest.approx(original.commanded_tilt)

    reg, orig_reg = loaded.registration, original.registration
    assert np.allclose(reg.homography_matrix, orig_reg.homography_matrix)
    assert np.allclose(reg.inverse_matrix, orig_reg.inverse_matrix)
    assert reg.num_lines == orig_reg.num_lines
    assert reg.num_inliers == orig_reg.num_inliers
    assert reg.inlier_ratio == pytest.approx(orig_reg.inlier_ratio)
    assert reg.mean_perp_error == pytest.approx(orig_reg.mean_perp_error)
    assert reg.max_perp_error == pytest.approx(orig_reg.max_perp_error)
    assert reg.rmse == pytest.approx(orig_reg.rmse)


# ---------------------------------------------------------------------------
# Postgres repo against in-memory SQLite (no live DB)
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_session() -> Iterator[Session]:
    """Yield a Session over an in-memory SQLite with only the PTZ table created."""
    engine = create_engine("sqlite://")
    cast("Table", PtzRegistrationModel.__table__).create(engine)
    with Session(engine) as session:
        yield session


class TestRepoPostgresPtzRegistration:
    def test_save_and_get_round_trips_result_and_stats(self, sqlite_session: Session) -> None:
        repo = RepoPostgresPtzRegistration(sqlite_session)
        entity = PtzRegistration.from_result(_make_result(), reprojection_stats=_make_stats())
        repo.save(entity)

        loaded = repo.get(entity.id)
        _assert_round_trip(loaded, entity)
        assert loaded.reprojection_stats is not None
        assert loaded.reprojection_stats.rms_m == pytest.approx(0.12)
        assert loaded.reprojection_stats.p90_m == pytest.approx(0.2)
        assert loaded.reprojection_stats.max_m == pytest.approx(0.31)
        assert loaded.reprojection_stats.n_points == 8

    def test_reprojection_stats_none_round_trips(self, sqlite_session: Session) -> None:
        repo = RepoPostgresPtzRegistration(sqlite_session)
        entity = PtzRegistration.from_result(_make_result(), reprojection_stats=None)
        repo.save(entity)

        loaded = repo.get(entity.id)
        assert loaded is not None
        assert loaded.reprojection_stats is None

    def test_get_by_camera_returns_records(self, sqlite_session: Session) -> None:
        repo = RepoPostgresPtzRegistration(sqlite_session)
        cam = "map01/cam01"
        e1 = PtzRegistration.from_result(_make_result(camera_id=cam, run_id="run_a", zoom=2.0))
        e2 = PtzRegistration.from_result(_make_result(camera_id=cam, run_id="run_b", zoom=4.0))
        other = PtzRegistration.from_result(_make_result(camera_id="map01/cam02", run_id="run_a"))
        repo.save(e1)
        repo.save(e2)
        repo.save(other)

        records = repo.get_by_camera(cam)
        ids = {r.id for r in records}
        assert ids == {e1.id, e2.id}
        assert all(r.camera_id == cam for r in records)

    def test_get_all(self, sqlite_session: Session) -> None:
        repo = RepoPostgresPtzRegistration(sqlite_session)
        e1 = PtzRegistration.from_result(_make_result(run_id="run_a"))
        e2 = PtzRegistration.from_result(_make_result(run_id="run_b"))
        repo.save(e1)
        repo.save(e2)

        ids = {r.id for r in repo.get_all()}
        assert e1.id in ids
        assert e2.id in ids

    def test_save_is_upsert(self, sqlite_session: Session) -> None:
        repo = RepoPostgresPtzRegistration(sqlite_session)
        entity = PtzRegistration.from_result(_make_result(), reprojection_stats=None)
        repo.save(entity)
        repo.save(PtzRegistration.from_result(_make_result(), reprojection_stats=_make_stats()))

        assert len(repo.get_all()) == 1
        loaded = repo.get(entity.id)
        assert loaded is not None
        assert loaded.reprojection_stats is not None


# ---------------------------------------------------------------------------
# YAML repo
# ---------------------------------------------------------------------------


class TestRepoYamlPtzRegistration:
    def test_save_and_get_round_trip(self, tmp_path: Path) -> None:
        repo = RepoYamlPtzRegistration(tmp_path)
        entity = PtzRegistration.from_result(_make_result(), reprojection_stats=_make_stats())
        repo.save(entity)
        repo.clear_cache()

        loaded = repo.get(entity.id)
        _assert_round_trip(loaded, entity)
        assert loaded.reprojection_stats is not None
        assert loaded.reprojection_stats.rms_m == pytest.approx(0.12)

    def test_reprojection_stats_none_round_trips(self, tmp_path: Path) -> None:
        repo = RepoYamlPtzRegistration(tmp_path)
        entity = PtzRegistration.from_result(_make_result(), reprojection_stats=None)
        repo.save(entity)
        repo.clear_cache()

        loaded = repo.get(entity.id)
        assert loaded is not None
        assert loaded.reprojection_stats is None

    def test_get_by_camera_returns_records(self, tmp_path: Path) -> None:
        repo = RepoYamlPtzRegistration(tmp_path)
        cam = "map01/cam01"
        e1 = PtzRegistration.from_result(_make_result(camera_id=cam, run_id="run_a", zoom=2.0))
        e2 = PtzRegistration.from_result(_make_result(camera_id=cam, run_id="run_b", zoom=4.0))
        other = PtzRegistration.from_result(_make_result(camera_id="map01/cam02", run_id="run_a"))
        repo.save(e1)
        repo.save(e2)
        repo.save(other)
        repo.clear_cache()

        records = repo.get_by_camera(cam)
        ids = {r.id for r in records}
        assert ids == {e1.id, e2.id}


# ---------------------------------------------------------------------------
# Entity factory invariant
# ---------------------------------------------------------------------------


def test_from_result_requires_camera_id() -> None:
    with pytest.raises(ValueError, match="camera_id is required"):
        PtzRegistration.from_result(_make_result(camera_id=""))


def test_from_dict_requires_camera_id() -> None:
    data = PtzRegistration.from_result(_make_result()).to_dict()
    data["camera_id"] = ""
    with pytest.raises(ValueError, match="camera_id is required"):
        PtzRegistration.from_dict(data)


def test_make_id_distinguishes_close_floats() -> None:
    """Distinct float64 PTZ states must not collapse to the same key."""
    id_a = PtzRegistration.make_id("cam", "run", 2.0000001, 15.0)
    id_b = PtzRegistration.make_id("cam", "run", 2.0000002, 15.0)
    assert id_a != id_b


def test_make_id_round_trips_via_recompute() -> None:
    """The recomputed id matches the id stored in to_dict (no %g divergence)."""
    entity = PtzRegistration.from_result(_make_result(zoom=2.0000001, tilt=15.0000003))
    assert entity.to_dict()["id"] == entity.id
