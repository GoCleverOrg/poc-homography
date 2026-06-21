"""Offline tests for the idempotent DB bootstrap CLI command (#374).

The bootstrap command ensures a tenant's calibration pipeline FK reference rows
(tenant -> map -> camera_configs) exist, sourced from the registry + map
sidecar. These tests exercise the upsert seam and the Typer command with
in-memory repo doubles + a fake session — NO live Neon or network. Idempotency
is asserted directly: running the seam twice yields no duplicate rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from poc_homography.cli import bootstrap as cmd
from poc_homography.cli.main import app
from poc_homography.domain.entities.map import Map
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# In-memory doubles
# ---------------------------------------------------------------------------


class _FakeRepo:
    """In-memory repo double: ``save`` upserts by id into a shared dict.

    Each repo class gets its OWN class-level ``rows`` dict so the three repo
    doubles do not collide. ``save`` mirrors RepoPostgres.save (upsert by id).
    """

    rows: dict[str, object] = {}

    def __init__(self, session: object) -> None:
        self._session = session

    def save(self, entity: object) -> None:
        type(self).rows[entity.id] = entity  # type: ignore[attr-defined]

    def get(self, entity_id: str) -> object | None:
        return type(self).rows.get(entity_id)


class _FakeTenantRepo(_FakeRepo):
    rows: dict[str, object] = {}


class _FakeMapRepo(_FakeRepo):
    rows: dict[str, object] = {}


class _FakeCameraRepo(_FakeRepo):
    rows: dict[str, object] = {}


class _FakeSession:
    """Minimal session object handed to the repo doubles."""


@contextmanager
def _fake_session_factory() -> Iterator[_FakeSession]:
    yield _FakeSession()


def _install_repo_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the bootstrap module's repo seams to fresh in-memory doubles."""
    _FakeTenantRepo.rows = {}
    _FakeMapRepo.rows = {}
    _FakeCameraRepo.rows = {}
    monkeypatch.setattr(cmd, "RepoPostgresTenant", _FakeTenantRepo)
    monkeypatch.setattr(cmd, "RepoPostgresMap", _FakeMapRepo)
    monkeypatch.setattr(cmd, "RepoPostgresCameraConfig", _FakeCameraRepo)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _tenant() -> Tenant:
    return Tenant(id="icozee", name="Icozee")


def _map() -> Map:
    return Map(
        id="icozee_cropped",
        tenant_id="icozee",
        photo=Photo(path=Path("icozee-cropped.tif"), width=3327, height=2731),
        geotiff=GeoTiff(
            geotransform=GeoTransform(
                origin_easting=69315.30,
                pixel_width=0.15,
                row_rotation=0.0,
                origin_northing=222999.00,
                col_rotation=0.0,
                pixel_height=-0.15,
            ),
            crs="EPSG:31370",
        ),
        asset_key="icozee/icozee-cropped.tif",
    )


def _cameras() -> list[dict]:
    return [
        {"id": "icozee-camptz-02", "tenant_id": "icozee", "name": "Cam02", "ip": "10.0.0.2"},
        {"id": "icozee-camptz-03", "tenant_id": "icozee", "name": "Cam03", "ip": "10.0.0.3"},
    ]


# ---------------------------------------------------------------------------
# _run_bootstrap: persists tenant + map + one camera_config per camera
# ---------------------------------------------------------------------------


def test_run_bootstrap_persists_reference_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_repo_doubles(monkeypatch)
    tenant_map = _map()

    summary = cmd._run_bootstrap(
        tenant_id="icozee",
        tenant=_tenant(),
        tenant_map=tenant_map,
        cameras=_cameras(),
        credential=Credential("admin", "pw"),
        session_factory=_fake_session_factory,
    )

    # Exactly one tenant row and one map row.
    assert list(_FakeTenantRepo.rows) == ["icozee"]
    assert list(_FakeMapRepo.rows) == ["icozee_cropped"]
    saved_map = _FakeMapRepo.rows["icozee_cropped"]
    assert saved_map.asset_key == "icozee/icozee-cropped.tif"  # type: ignore[attr-defined]

    # One camera_config per camera, each anchored to the map id + tenant.
    assert set(_FakeCameraRepo.rows) == {"icozee-camptz-02", "icozee-camptz-03"}
    cam = _FakeCameraRepo.rows["icozee-camptz-02"]
    assert cam.tenant_id == "icozee"  # type: ignore[attr-defined]
    assert cam.map_id == "icozee_cropped"  # type: ignore[attr-defined]
    assert cam.name == "Cam02"  # type: ignore[attr-defined]
    assert cam.spec == CameraSpec.HIKVISION_DS_2DF8425IX  # type: ignore[attr-defined]
    assert cam.ip_address == "10.0.0.2"  # type: ignore[attr-defined]

    assert summary.tenant_id == "icozee"
    assert summary.map_id == "icozee_cropped"
    assert summary.camera_ids == ["icozee-camptz-02", "icozee-camptz-03"]


def test_run_bootstrap_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_repo_doubles(monkeypatch)

    def run() -> None:
        cmd._run_bootstrap(
            tenant_id="icozee",
            tenant=_tenant(),
            tenant_map=_map(),
            cameras=_cameras(),
            credential=Credential("admin", "pw"),
            session_factory=_fake_session_factory,
        )

    run()
    sizes = (len(_FakeTenantRepo.rows), len(_FakeMapRepo.rows), len(_FakeCameraRepo.rows))
    run()  # second run upserts the SAME ids -> no duplicates
    assert (len(_FakeTenantRepo.rows), len(_FakeMapRepo.rows), len(_FakeCameraRepo.rows)) == sizes
    assert sizes == (1, 1, 2)


# ---------------------------------------------------------------------------
# _load_tenant_map: missing sidecar -> clear typer error
# ---------------------------------------------------------------------------


def test_load_tenant_map_missing_sidecar_raises(tmp_path: Path) -> None:
    with pytest.raises(cmd.typer.Exit) as excinfo:
        cmd._load_tenant_map("nope", tmp_path)
    assert excinfo.value.exit_code == 1


def test_load_tenant_map_derives_asset_key(tmp_path: Path) -> None:
    (tmp_path / "icozee.yaml").write_text(
        "id: icozee_cropped\n"
        "tenant_id: icozee\n"
        "photo:\n"
        "  path: icozee-cropped.tif\n"
        "  width: 3327\n"
        "  height: 2731\n"
        "geotiff:\n"
        "  geotransform:\n"
        "    origin_easting: 69315.30\n"
        "    pixel_width: 0.15\n"
        "    row_rotation: 0.0\n"
        "    origin_northing: 222999.00\n"
        "    col_rotation: 0.0\n"
        "    pixel_height: -0.15\n"
        "  crs: EPSG:31370\n"
    )

    tenant_map = cmd._load_tenant_map("icozee", tmp_path)

    assert tenant_map.id == "icozee_cropped"
    assert tenant_map.asset_key == "icozee/icozee-cropped.tif"


# ---------------------------------------------------------------------------
# Typer command wiring
# ---------------------------------------------------------------------------


def _wire_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    _install_repo_doubles(monkeypatch)
    monkeypatch.setattr(cmd, "get_tenant_by_id", lambda _t: {"id": "icozee", "name": "Icozee"})
    monkeypatch.setattr(cmd, "get_cameras_for_tenant", lambda _t: _cameras())
    monkeypatch.setattr(cmd, "get_tenant_credentials", lambda _t: ("admin", "pw"))
    monkeypatch.setattr(cmd, "get_session", _fake_session_factory)
    monkeypatch.setattr(cmd, "_load_tenant_map", lambda _t, _d: _map())


def test_command_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)

    result = CliRunner().invoke(app, ["calibrate", "bootstrap", "--tenant", "icozee"])

    assert result.exit_code == 0, result.output
    assert "Bootstrap: tenant=icozee map=icozee_cropped cameras=2" in result.output


def test_command_missing_database_url_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = CliRunner().invoke(app, ["calibrate", "bootstrap", "--tenant", "icozee"])

    assert result.exit_code != 0
    assert "DATABASE_URL" in result.output


def test_command_unknown_tenant_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(cmd, "get_tenant_by_id", lambda _t: None)

    result = CliRunner().invoke(app, ["calibrate", "bootstrap", "--tenant", "ghost"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()
