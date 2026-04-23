"""Shared fixtures for PostgreSQL integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.entities.map import Map
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import Easting, Meters, Northing, Pixels, Unitless

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip all postgres integration tests if DATABASE_URL is not set."""
    if not os.environ.get("DATABASE_URL"):
        skip = pytest.mark.skip(reason="DATABASE_URL not set")
        for item in items:
            if "test_repo_postgres" in str(item.fspath):
                item.add_marker(skip)


@pytest.fixture
def db_session() -> Session:  # type: ignore[misc]
    """Yield a DB session that rolls back after each test."""
    from sqlalchemy.orm import Session as SASession

    from poc_homography.infrastructure.database import get_engine

    engine = get_engine()
    with SASession(engine) as session:
        session.begin()
        yield session  # type: ignore[misc]
        session.rollback()


@pytest.fixture
def test_tenant(db_session: Session) -> Tenant:
    """Create and persist a Tenant entity for FK deps."""
    from poc_homography.infrastructure.repositories import RepoPostgresTenant

    tenant = Tenant(
        id="test_tenant",
        name="Test Tenant",
        description="Integration test tenant",
        location_lat="39d38'25.72\"N",
        location_lon="0d13'48.63\"W",
    )
    repo = RepoPostgresTenant(db_session)
    repo.save(tenant)
    return tenant


@pytest.fixture
def test_map(db_session: Session, test_tenant: Tenant) -> Map:
    """Create and persist a Map entity for FK deps."""
    from poc_homography.infrastructure.repositories import RepoPostgresMap

    m = Map(
        id="test_map",
        tenant_id=test_tenant.id,
        photo=Photo(path=Path("test_map.png"), width=Pixels(1000), height=Pixels(800)),
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
    repo = RepoPostgresMap(db_session)
    repo.save(m)
    return m


@pytest.fixture
def test_camera_config(db_session: Session, test_tenant: Tenant, test_map: Map) -> CameraConfig:
    """Create and persist a CameraConfig entity for FK deps."""
    from poc_homography.infrastructure.repositories import RepoPostgresCameraConfig

    cc = CameraConfig(
        id="test_map/cam01",
        tenant_id=test_tenant.id,
        map_id=test_map.id,
        name="cam01",
        spec=CameraSpec.HIKVISION_DS_2DF8425IX,
        credential=Credential(username="admin", password="secret"),
        ip_address="192.168.1.100",
    )
    repo = RepoPostgresCameraConfig(db_session)
    repo.save(cc)
    return cc
