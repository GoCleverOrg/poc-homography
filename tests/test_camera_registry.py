"""Unit tests for :mod:`poc_homography.camera_registry`.

Covers TTL cache behaviour (hit / miss / expiry), database fallback, per-camera
fallback, calibration merge, cache invalidation, credential mapping and thread
safety. The database layer is mocked throughout — these tests never touch a real
PostgreSQL connection.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from poc_homography import camera_registry
from poc_homography.camera_registry import (
    CameraRegistry,
    _entity_to_camera_dict,
    _resolve_ttl,
    get_registry,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the process-wide singleton around every test."""
    CameraRegistry._instance = None
    yield
    CameraRegistry._instance = None


@pytest.fixture
def clock(monkeypatch):
    """Controllable monotonic clock for TTL tests."""
    state = {"t": 1000.0}
    monkeypatch.setattr(camera_registry, "_now", lambda: state["t"])
    return state


def _fresh_registry(monkeypatch, *, db_cameras, ttl="300"):
    """Build a registry whose DB load returns ``db_cameras`` and count loads.

    ``db_cameras`` may be a list (rows), ``None`` (DB unavailable / empty), or a
    callable returning one of those.
    """
    monkeypatch.setenv("CAMERA_CACHE_TTL", ttl)
    reg = get_registry()
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return db_cameras() if callable(db_cameras) else db_cameras

    monkeypatch.setattr(reg, "_load_from_database", fake_load)
    return reg, calls


# --------------------------------------------------------------------------- #
# TTL resolution
# --------------------------------------------------------------------------- #


def test_resolve_ttl_default(monkeypatch):
    monkeypatch.delenv("CAMERA_CACHE_TTL", raising=False)
    assert _resolve_ttl() == 300


def test_resolve_ttl_from_env(monkeypatch):
    monkeypatch.setenv("CAMERA_CACHE_TTL", "42")
    assert _resolve_ttl() == 42


def test_resolve_ttl_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CAMERA_CACHE_TTL", "not-a-number")
    assert _resolve_ttl() == 300


# --------------------------------------------------------------------------- #
# Cache hit / miss / TTL expiry
# --------------------------------------------------------------------------- #


def test_cache_miss_triggers_load(monkeypatch, clock):
    reg, calls = _fresh_registry(monkeypatch, db_cameras=None)
    reg.get_all_cameras()
    assert calls["n"] == 1


def test_cache_hit_does_not_reload(monkeypatch, clock):
    reg, calls = _fresh_registry(monkeypatch, db_cameras=None)
    reg.get_all_cameras()
    reg.get_all_cameras()
    reg.get_camera_by_id("valte_cam01")
    assert calls["n"] == 1


def test_ttl_expiry_triggers_reload(monkeypatch, clock):
    reg, calls = _fresh_registry(monkeypatch, db_cameras=None, ttl="300")
    reg.get_all_cameras()
    assert calls["n"] == 1
    clock["t"] += 299  # still fresh
    reg.get_all_cameras()
    assert calls["n"] == 1
    clock["t"] += 2  # now 301s elapsed, expired
    reg.get_all_cameras()
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Fallback behaviour
# --------------------------------------------------------------------------- #


def test_db_unavailable_falls_back_to_hardcoded(monkeypatch, clock):
    reg, _ = _fresh_registry(monkeypatch, db_cameras=None)
    cams = reg.get_all_cameras()
    ids = {c["id"] for c in cams}
    assert "valte_cam01" in ids
    # Calibration fields preserved from the hardcoded list.
    assert "k1" in reg.get_camera_by_id("valte_cam01")


def test_empty_db_result_falls_back_to_hardcoded(monkeypatch, clock):
    reg, _ = _fresh_registry(monkeypatch, db_cameras=[])
    # An explicit empty list from _load_from_database means "no overlay";
    # hardcoded cameras remain.
    assert reg.get_camera_by_id("valte_cam01") is not None


def test_load_from_database_returns_none_on_exception(monkeypatch):
    """The real loader swallows DB errors and signals fallback via None."""

    def boom():
        raise RuntimeError("DATABASE_URL not set")

    import poc_homography.infrastructure.database as db

    monkeypatch.setattr(db, "get_session", boom)
    reg = get_registry()
    assert reg._load_from_database() is None


# --------------------------------------------------------------------------- #
# DB overlay / calibration merge
# --------------------------------------------------------------------------- #


def test_db_overlay_merges_over_hardcoded_calibration(monkeypatch, clock):
    db_rows = [
        {
            "id": "valte_cam01",
            "tenant_id": "valte",
            "name": "Cam01",
            "ip": "9.9.9.9",
            "username": "dbuser",
            "password": "dbpass",
            "model": "HIKVISION_DS_2DF8425IX",
        }
    ]
    reg, _ = _fresh_registry(monkeypatch, db_cameras=db_rows)
    cam = reg.get_camera_by_id("valte_cam01")
    assert cam["ip"] == "9.9.9.9"  # DB overrides
    assert cam["username"] == "dbuser"
    assert cam["k1"] == pytest.approx(-0.341052)  # calibration merged from hardcoded


def test_db_null_field_does_not_clobber_hardcoded(monkeypatch, clock):
    db_rows = [
        {
            "id": "valte_cam01",
            "tenant_id": "valte",
            "name": "Cam01",
            "ip": None,  # NULL ip_address must not wipe the hardcoded ip
            "username": "dbuser",
            "password": "dbpass",
        }
    ]
    reg, _ = _fresh_registry(monkeypatch, db_cameras=db_rows)
    cam = reg.get_camera_by_id("valte_cam01")
    assert cam["ip"] == "10.207.99.178"


def test_camera_only_in_db_is_added(monkeypatch, clock):
    db_rows = [
        {
            "id": "brandnew_cam99",
            "tenant_id": "brandnew",
            "name": "Cam99",
            "ip": "1.2.3.4",
            "username": "u",
            "password": "p",
        }
    ]
    reg, _ = _fresh_registry(monkeypatch, db_cameras=db_rows)
    cam = reg.get_camera_by_id("brandnew_cam99")
    assert cam is not None
    assert cam["ip"] == "1.2.3.4"


def test_per_camera_fallback_when_absent_from_db(monkeypatch, clock):
    # DB only knows about an unrelated camera; valte_cam01 still resolves from
    # the hardcoded list.
    db_rows = [{"id": "other_cam", "tenant_id": "x", "ip": "1.1.1.1"}]
    reg, _ = _fresh_registry(monkeypatch, db_cameras=db_rows)
    cam = reg.get_camera_by_id("valte_cam01")
    assert cam is not None
    assert cam["ip"] == "10.207.99.178"


def test_get_cameras_for_tenant_filters(monkeypatch, clock):
    reg, _ = _fresh_registry(monkeypatch, db_cameras=None)
    icozee = reg.get_cameras_for_tenant("icozee")
    assert len(icozee) >= 1
    assert all(c["tenant_id"] == "icozee" for c in icozee)


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #


def test_invalidate_forces_reload(monkeypatch, clock):
    reg, calls = _fresh_registry(monkeypatch, db_cameras=None)
    reg.get_all_cameras()
    assert calls["n"] == 1
    invalidate_cache()
    reg.get_all_cameras()
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Entity → dict transform
# --------------------------------------------------------------------------- #


def test_entity_to_camera_dict_maps_fields():
    entity = SimpleNamespace(
        id="valte_cam01",
        tenant_id="valte",
        map_id="map-1",
        name="Cam01",
        ip_address="10.0.0.1",
        spec=SimpleNamespace(model_name="DS-2DF8425IX-AELW"),
        credential=SimpleNamespace(username="admin", password="secret"),
    )
    d = _entity_to_camera_dict(entity)
    assert d == {
        "id": "valte_cam01",
        "tenant_id": "valte",
        "map_id": "map-1",
        "name": "Cam01",
        "ip": "10.0.0.1",
        "username": "admin",
        "password": "secret",
        "model": "DS-2DF8425IX-AELW",
    }


# --------------------------------------------------------------------------- #
# Singleton + thread safety
# --------------------------------------------------------------------------- #


def test_singleton_identity():
    assert get_registry() is get_registry()
    assert CameraRegistry() is get_registry()


def test_concurrent_access_does_not_corrupt(monkeypatch, clock):
    reg, _ = _fresh_registry(monkeypatch, db_cameras=None)
    results: list[int] = []
    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(20):
                results.append(len(reg.get_all_cameras()))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Every read saw a consistent, non-empty camera count.
    assert results
    assert len(set(results)) == 1
    assert results[0] > 0
