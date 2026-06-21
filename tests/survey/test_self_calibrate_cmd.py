"""Tests for the ``hom calibrate self-calibrate`` command (#351).

Exercise the extracted ``_run_self_calibration`` helper offline: the survey-frame
loader and the scene self-calibrator are monkeypatched to canned doubles, the
Postgres lens-table repo is replaced with an in-memory capture, and the session
factory is a no-op context manager -- so nothing touches MinIO, Neon, or the
scene-calibration solver. A DB round-trip integration test (gated on
``DATABASE_URL``) uses the real Postgres repo to prove the table persists.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.frame_source import SurveyFrame
from poc_homography.calibration.lens_distortion.scene_self_calibration import (
    SceneCalibrationResult,
    SkippedZoom,
)
from poc_homography.cli import self_calibrate as cmd
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.types import PixelsFloat, Unitless

if TYPE_CHECKING:
    from collections.abc import Generator


def _canned_frames() -> list[SurveyFrame]:
    """Three canned survey frames spanning two distinct zooms."""
    return [
        SurveyFrame(
            image=np.zeros((4, 6, 3), dtype=np.uint8), commanded_zoom=1.0, commanded_tilt=-10.0
        ),
        SurveyFrame(
            image=np.ones((4, 6, 3), dtype=np.uint8), commanded_zoom=1.0, commanded_tilt=-12.0
        ),
        SurveyFrame(
            image=np.full((4, 6, 3), 2, dtype=np.uint8), commanded_zoom=5.0, commanded_tilt=-20.0
        ),
    ]


def _canned_result(camera_id: str) -> SceneCalibrationResult:
    """A canned result: two calibrated zooms + one skipped zoom."""
    table = LensCalibrationTable(
        id=camera_id,
        entries=(
            ZoomCalibrationEntry(
                zoom_factor=Unitless(1.0),
                distortion=LensDistortion(k1=Unitless(-0.1), k2=Unitless(0.01)),
                calibration_date="2026-06-21T10:00:00",
                source_images=("img_a.jpg",),
                validation_rmse=0.42,
                num_lines_used=7,
                fx=PixelsFloat(1000.0),
                fy=PixelsFloat(1001.0),
                cx=PixelsFloat(640.0),
                cy=PixelsFloat(360.0),
                reprojection_error_px=0.31,
            ),
            ZoomCalibrationEntry(
                zoom_factor=Unitless(5.0),
                distortion=LensDistortion(k1=Unitless(-0.2), k2=Unitless(0.02)),
                calibration_date="2026-06-21T10:01:00",
                source_images=("img_b.jpg",),
                validation_rmse=0.55,
                num_lines_used=9,
                fx=PixelsFloat(5000.0),
                fy=PixelsFloat(5002.0),
                cx=PixelsFloat(641.0),
                cy=PixelsFloat(361.0),
                reprojection_error_px=0.48,
            ),
        ),
        created_date="2026-06-21T10:00:00",
        last_modified="2026-06-21T10:01:00",
    )
    skipped = (SkippedZoom(zoom_factor=10.0, num_lines=2, reason="only 2 usable line(s)"),)
    return SceneCalibrationResult(table=table, skipped_zooms=skipped)


class _CapturingLensRepo:
    """In-memory ``RepoPostgresLensCalibrationTable`` double capturing saves."""

    saved: list[LensCalibrationTable] = []

    def __init__(self, session: object) -> None:
        self._session = session

    def save(self, table: LensCalibrationTable) -> bool:
        type(self).saved.append(table)
        return True


@contextmanager
def _fake_session_factory() -> Generator[object, None, None]:
    """A no-op session context manager (nothing touches Neon)."""
    yield object()


def _install_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: list[SurveyFrame],
    result: SceneCalibrationResult,
) -> list[list[tuple[np.ndarray, float]]]:
    """Patch the loader, calibrator, and repo; return a calibrator-arg sink."""
    monkeypatch.setattr(cmd, "iter_survey_frames", lambda **_kwargs: iter(frames))

    captured_pairs: list[list[tuple[np.ndarray, float]]] = []

    def fake_calibrate(
        image_zoom_pairs: list[tuple[np.ndarray, float]], *, camera_id: str
    ) -> SceneCalibrationResult:
        captured_pairs.append(image_zoom_pairs)
        return result

    monkeypatch.setattr(cmd, "calibrate_scene_self", fake_calibrate)

    _CapturingLensRepo.saved = []
    monkeypatch.setattr(cmd, "RepoPostgresLensCalibrationTable", _CapturingLensRepo)

    return captured_pairs


def test_run_self_calibration_saves_table_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frames = _canned_frames()
    result = _canned_result("cam01")
    captured_pairs = _install_doubles(monkeypatch, frames=frames, result=result)

    returned = cmd._run_self_calibration(
        run_id="run-1",
        camera_id="cam01",
        frame_store=object(),
        session_factory=_fake_session_factory,
    )

    # The canned result is returned and the table was persisted verbatim.
    assert returned is result
    assert _CapturingLensRepo.saved == [result.table]

    # The calibrator received one (image, zoom) pair per canned frame.
    assert len(captured_pairs) == 1
    pairs = captured_pairs[0]
    assert [z for _img, z in pairs] == [1.0, 1.0, 5.0]

    # The per-zoom report carries each entry's diagnostics and the skip reason.
    cmd._print_report(camera_id="cam01", result=returned)
    out = capsys.readouterr().out
    assert "7" in out  # num_lines_used (zoom 1.0)
    assert "9" in out  # num_lines_used (zoom 5.0)
    assert "0.42" in out  # validation_rmse (zoom 1.0)
    assert "1000.0" in out  # fx (zoom 1.0)
    assert "5000.0" in out  # fx (zoom 5.0)
    assert "only 2 usable line(s)" in out  # skipped-zoom reason
    assert "zooms_calibrated=2" in out
    assert "skipped=1" in out


def test_run_self_calibration_applies_zoom_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _canned_frames()
    result = _canned_result("cam01")
    captured_pairs = _install_doubles(monkeypatch, frames=frames, result=result)

    cmd._run_self_calibration(
        run_id="run-1",
        camera_id="cam01",
        frame_store=object(),
        session_factory=_fake_session_factory,
        zoom_filter=[1.0],
    )

    # Only the two zoom==1.0 frames reach the calibrator.
    assert len(captured_pairs) == 1
    pairs = captured_pairs[0]
    assert [z for _img, z in pairs] == [1.0, 1.0]


def test_run_self_calibration_empty_frames_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typer

    result = _canned_result("cam01")
    _install_doubles(monkeypatch, frames=[], result=result)

    with pytest.raises(typer.Exit):
        cmd._run_self_calibration(
            run_id="run-1",
            camera_id="cam01",
            frame_store=object(),
            session_factory=_fake_session_factory,
        )


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_run_self_calibration_persists_to_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB round-trip: the real Postgres repo persists and re-reads the table."""
    from pathlib import Path

    from sqlalchemy.orm import Session as SASession

    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.entities.map import Map
    from poc_homography.domain.entities.tenant import Tenant
    from poc_homography.domain.enums import CameraSpec
    from poc_homography.domain.vo.credential import Credential
    from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
    from poc_homography.domain.vo.photo import Photo
    from poc_homography.infrastructure.database import get_engine
    from poc_homography.infrastructure.repositories import (
        RepoPostgresCameraConfig,
        RepoPostgresMap,
        RepoPostgresTenant,
    )
    from poc_homography.infrastructure.repositories.repo_postgres_lens_calibration_table import (
        RepoPostgresLensCalibrationTable,
    )
    from poc_homography.types import Easting, Meters, Northing, Pixels

    engine = get_engine()
    with SASession(engine) as db_session:
        db_session.begin()
        try:
            # Persist the FK parents (tenant -> map -> camera_config).
            RepoPostgresTenant(db_session).save(
                Tenant(
                    id="sc_tenant",
                    name="SC Tenant",
                    description="self-calibrate test tenant",
                    location_lat="39d38'25.72\"N",
                    location_lon="0d13'48.63\"W",
                )
            )
            RepoPostgresMap(db_session).save(
                Map(
                    id="sc_map",
                    tenant_id="sc_tenant",
                    photo=Photo(path=Path("m.png"), width=Pixels(1000), height=Pixels(800)),
                    geotiff=GeoTiff(
                        geotransform=GeoTransform(
                            origin_easting=Easting(500000.0),
                            pixel_width=Meters(0.5),
                            row_rotation=Unitless(0.0),
                            origin_northing=Northing(4400000.0),
                            col_rotation=Unitless(0.0),
                            pixel_height=Meters(-0.5),
                        ),
                        crs="EPSG:25830",
                    ),
                )
            )
            camera_id = "sc_map/cam01"
            RepoPostgresCameraConfig(db_session).save(
                CameraConfig(
                    id=camera_id,
                    tenant_id="sc_tenant",
                    map_id="sc_map",
                    name="cam01",
                    spec=CameraSpec.HIKVISION_DS_2DF8425IX,
                    credential=Credential(username="admin", password="secret"),
                    ip_address="192.168.1.100",
                )
            )

            result = _canned_result(camera_id)
            monkeypatch.setattr(cmd, "iter_survey_frames", lambda **_kwargs: iter(_canned_frames()))
            monkeypatch.setattr(cmd, "calibrate_scene_self", lambda _pairs, *, camera_id: result)

            @contextmanager
            def real_session_factory() -> Generator[object, None, None]:
                yield db_session

            cmd._run_self_calibration(
                run_id="run-1",
                camera_id=camera_id,
                frame_store=object(),
                session_factory=real_session_factory,
            )

            loaded = RepoPostgresLensCalibrationTable(db_session).get(camera_id)
            assert loaded is not None
            assert loaded.id == camera_id
            assert len(loaded.entries) == 2
            assert float(loaded.entries[0].validation_rmse) == pytest.approx(0.42)
        finally:
            db_session.rollback()
