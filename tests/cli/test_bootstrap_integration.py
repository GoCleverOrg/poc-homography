"""DB-gated integration test for the idempotent bootstrap command (#374).

Proves that after running ``_run_bootstrap`` against the real database for the
``icozee`` tenant, the downstream pipeline rows (``lens_calibration_tables``,
``ptz_registrations``) — which sit at the bottom of the tenant -> map ->
camera_config FK chain — persist with NO ForeignKeyViolation, and that
re-running bootstrap is a no-op. Everything rolls back at the end.

This file lives under ``tests/cli/`` (no shared ``db_session`` fixture), so it
inlines a rollback session fixture and carries its own DATABASE_URL skip guard;
the ``tests/infrastructure`` collection gate does not reach this directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
from poc_homography.calibration.extrinsic.validation import ReprojectionStats
from poc_homography.camera_config import get_cameras_for_tenant, get_tenant_by_id
from poc_homography.cli import bootstrap as cmd
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.infrastructure.repositories import (
    RepoPostgresCameraConfig,
    RepoPostgresLensCalibrationTable,
    RepoPostgresPtzRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration

_CAMERA_ID = "icozee-camptz-03"


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a DB session that rolls back after the test."""
    from sqlalchemy.orm import Session as SASession

    from poc_homography.infrastructure.database import get_engine

    engine = get_engine()
    with SASession(engine) as session:
        session.begin()
        yield session
        session.rollback()


def _lens_table(camera_id: str) -> LensCalibrationTable:
    entry = ZoomCalibrationEntry(
        zoom_factor=1.0,
        distortion=LensDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        calibration_date="2024-01-01T00:00:00",
        source_images=(),
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
    )
    return LensCalibrationTable(
        id=camera_id,
        entries=(entry,),
        created_date="2024-01-01T00:00:00",
        last_modified="2024-01-01T00:00:00",
    )


def _ptz_registration(camera_id: str) -> PtzRegistration:
    homography = np.array([[1.2, 0.1, 5.0], [0.0, 1.1, 7.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    result = PtzRegistrationResult(
        homography_matrix=homography,
        inverse_matrix=np.linalg.inv(homography),
        num_lines=4,
        num_inliers=3,
        inlier_ratio=0.75,
        mean_perp_error=1.2,
        max_perp_error=3.4,
        rmse=1.5,
        run_id="run_a",
        camera_id=camera_id,
        commanded_zoom=2.0,
        commanded_tilt=15.0,
    )
    return PtzRegistration.from_result(
        result,
        reprojection_stats=ReprojectionStats(rms_m=0.12, p90_m=0.2, max_m=0.31, n_points=8),
        created_date="2024-01-01T00:00:00",
        last_modified="2024-01-01T00:00:00",
    )


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_bootstrap_enables_pipeline_persistence(db_session: Session) -> None:
    from contextlib import contextmanager

    @contextmanager
    def session_factory() -> Iterator[Session]:
        # Reuse the rolled-back test session so the whole test is one transaction.
        yield db_session

    tenant_dict = get_tenant_by_id("icozee")
    assert tenant_dict is not None
    cameras = get_cameras_for_tenant("icozee")
    assert cameras

    tenant_map = cmd._load_tenant_map("icozee", Path("data/maps"))
    tenant = Tenant.from_dict(tenant_dict)
    credential = Credential("", "")

    summary = cmd._run_bootstrap(
        tenant_id="icozee",
        tenant=tenant,
        tenant_map=tenant_map,
        cameras=cameras,
        credential=credential,
        session_factory=session_factory,
    )
    assert _CAMERA_ID in summary.camera_ids

    # The FK-required rows now exist: persisting the bottom-of-chain pipeline
    # records keyed by the camera id succeeds with no ForeignKeyViolation.
    RepoPostgresLensCalibrationTable(db_session).save(_lens_table(_CAMERA_ID))
    RepoPostgresPtzRegistration(db_session).save(_ptz_registration(_CAMERA_ID))
    db_session.flush()

    # Re-running bootstrap is a no-op (upsert by id; row counts unchanged).
    before = len(RepoPostgresCameraConfig(db_session).get_all())
    cmd._run_bootstrap(
        tenant_id="icozee",
        tenant=tenant,
        tenant_map=tenant_map,
        cameras=cameras,
        credential=credential,
        session_factory=session_factory,
    )
    after = len(RepoPostgresCameraConfig(db_session).get_all())
    assert after == before
