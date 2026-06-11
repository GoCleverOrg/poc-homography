"""Integration tests for the camera registry + ``camera_config`` wiring.

These exercise the full path that consumers (the Django ``webapp`` tools
camera_diagnostic / camera_survey / camera_evaluation, and the CLI/API layers)
travel: ``camera_config`` public functions -> :class:`CameraRegistry` ->
database loader. The PostgreSQL session and repository are mocked so the tests
run without a live database, but the registry's *real* ``_load_from_database``
plumbing (``get_session`` + ``RepoPostgresCameraConfig.get_all``) is executed.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import poc_homography.camera_config as cc
from poc_homography.camera_registry import CameraRegistry


@pytest.fixture(autouse=True)
def _reset_singleton():
    CameraRegistry._instance = None
    yield
    CameraRegistry._instance = None


def _make_entity(cam_id, tenant_id, ip, username, password, name="Cam"):
    return SimpleNamespace(
        id=cam_id,
        tenant_id=tenant_id,
        map_id="map-1",
        name=name,
        ip_address=ip,
        spec=SimpleNamespace(model_name="DS-2DF8425IX-AELW"),
        credential=SimpleNamespace(username=username, password=password),
    )


@pytest.fixture
def mock_db(monkeypatch):
    """Patch the registry's DB dependencies to return controlled entities.

    Returns a mutable list of entities the fake repository will yield.
    """
    entities: list = []

    @contextmanager
    def fake_get_session():
        yield object()  # session is unused by the fake repo

    class FakeRepo:
        def __init__(self, session):
            self._session = session

        def get_all(self):
            return list(entities)

    import poc_homography.infrastructure.database as db
    import poc_homography.infrastructure.repositories as repos

    monkeypatch.setattr(db, "get_session", fake_get_session)
    monkeypatch.setattr(repos, "RepoPostgresCameraConfig", FakeRepo)
    return entities


@pytest.fixture
def no_db(monkeypatch):
    """Make the registry's DB load fail, forcing the hardcoded fallback."""

    def boom():
        raise RuntimeError("DATABASE_URL not set")

    import poc_homography.infrastructure.database as db

    monkeypatch.setattr(db, "get_session", boom)


@pytest.fixture
def clear_creds(monkeypatch):
    """Remove all tenant/global credential sources from the environment."""
    for var in (
        "VALTE_CAMERA_USERNAME",
        "VALTE_CAMERA_PASSWORD",
        "CAMERA_USERNAME",
        "CAMERA_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cc, "USERNAME", None)
    monkeypatch.setattr(cc, "PASSWORD", None)


@pytest.mark.integration
def test_camera_config_functions_return_db_backed_data(mock_db):
    mock_db.append(_make_entity("valte_cam01", "valte", "9.8.7.6", "dbuser", "dbpass"))

    cam = cc.get_camera_by_id("valte_cam01")
    assert cam["ip"] == "9.8.7.6"  # from the database
    assert "k1" in cam  # calibration still merged from hardcoded list

    all_cams = cc.get_camera_configs()
    assert any(c["id"] == "valte_cam01" and c["ip"] == "9.8.7.6" for c in all_cams)

    valte_cams = cc.get_cameras_for_tenant("valte")
    assert any(c["id"] == "valte_cam01" for c in valte_cams)


@pytest.mark.integration
def test_get_rtsp_url_uses_per_camera_credentials(mock_db, clear_creds):
    # clear_creds proves the per-camera DB credentials are what get used.
    mock_db.append(_make_entity("valte_cam01", "valte", "9.8.7.6", "dbuser", "dbpass"))

    url = cc.get_rtsp_url("valte_cam01", stream_type="main")
    assert url == "rtsp://dbuser:dbpass@9.8.7.6:554/Streaming/Channels/101"


@pytest.mark.integration
def test_fallback_mode_uses_tenant_credentials_when_db_unavailable(no_db, monkeypatch):
    # No DB -> registry falls back to hardcoded cameras, which carry no
    # per-camera credentials, so tenant env vars are used.
    monkeypatch.setenv("VALTE_CAMERA_USERNAME", "tenantuser")
    monkeypatch.setenv("VALTE_CAMERA_PASSWORD", "tenantpass")

    cam = cc.get_camera_by_id("valte_cam01")
    assert cam["ip"] == "10.207.99.178"  # hardcoded fallback

    url = cc.get_rtsp_url("valte_cam01", stream_type="main")
    assert url == "rtsp://tenantuser:tenantpass@10.207.99.178:554/Streaming/Channels/101"


@pytest.mark.integration
def test_missing_credentials_raise_value_error(no_db, clear_creds):
    with pytest.raises(ValueError, match="credentials not set"):
        cc.get_rtsp_url("valte_cam01")
