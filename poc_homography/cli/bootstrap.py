"""Idempotent DB bootstrap CLI command for the calibration pipeline (#374).

The automatic calibration pipeline persists ``lens_calibration_tables`` and
``ptz_registrations`` keyed by camera id, but those tables sit at the bottom of
an FK chain: ``tenants(id)`` <- ``maps.tenant_id``; ``maps(id)`` <-
``camera_configs.map_id``; ``camera_configs(id)`` <-
``lens_calibration_tables.id``. When the reference tables are empty, the
pipeline cannot persist anything (FK violation).

This command ensures the three reference rows a tenant's pipeline requires —
the tenant, its map, and one ``camera_config`` per registry camera — exist in
the database, sourced entirely from the EXISTING registry + map sidecars. It
authors **no** new data. Every row is keyed by its stable id and persisted via
:meth:`RepoPostgres.save` (an upsert), so the command is idempotent: re-running
it updates rows in place and never duplicates.

The offline-testable seam :func:`_run_bootstrap` references the three repo
classes as module-level names so tests can swap them for in-memory doubles and
exercise tenant -> map -> camera_configs upsert ordering with NO live Neon
access. The Typer command resolves the registry + sidecar inputs and delegates
to that seam.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer

from poc_homography.camera_config import (
    get_cameras_for_tenant,
    get_tenant_by_id,
    get_tenant_credentials,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories import (
    RepoPostgresCameraConfig,
    RepoPostgresMap,
    RepoPostgresTenant,
    RepoYamlMap,
)
from poc_homography.maps import DEFAULT_MAPS_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path

    from sqlalchemy.orm import Session

    from poc_homography.domain.entities.map import Map


@dataclass(frozen=True)
class BootstrapSummary:
    """The reference rows ensured by one bootstrap run.

    Attributes:
        tenant_id: The bootstrapped tenant id.
        map_id: The stable map id (the sidecar's internal ``id``).
        camera_ids: The camera ids whose ``camera_config`` rows were upserted.
    """

    tenant_id: str
    map_id: str
    camera_ids: list[str]


def _build_camera_configs(
    *,
    tenant_id: str,
    map_id: str,
    cameras: Sequence[dict],
    credential: Credential,
) -> list[CameraConfig]:
    """Build one :class:`CameraConfig` per registry camera dict.

    Each config is anchored to the resolved ``map_id`` (the cameras' FK target)
    and carries the tenant credential plaintext so the live pipeline can reach
    the camera. Authors no new data — every field is sourced from the registry.

    Args:
        tenant_id: The owning tenant id.
        map_id: The stable map id the cameras reference.
        cameras: Registry camera dicts (``id``, ``name``, ``ip``).
        credential: The tenant camera credential.

    Returns:
        The list of camera configs, one per input camera, in input order.
    """
    configs: list[CameraConfig] = []
    for cam in cameras:
        configs.append(
            CameraConfig(
                id=str(cam["id"]),
                tenant_id=tenant_id,
                map_id=map_id,
                name=str(cam.get("name") or cam["id"]),
                spec=CameraSpec.HIKVISION_DS_2DF8425IX,
                credential=credential,
                ip_address=cam.get("ip"),
            )
        )
    return configs


def _run_bootstrap(
    *,
    tenant_id: str,
    tenant: Tenant,
    tenant_map: Map,
    cameras: Sequence[dict],
    credential: Credential,
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> BootstrapSummary:
    """Upsert the tenant, its map, and each camera config in FK order.

    Opens one session and upserts in the order the FK chain requires: tenant ->
    map -> camera_configs. Every save is an upsert keyed by id, so re-running is
    idempotent (rows are updated in place, never duplicated). The three repo
    classes are module-level names so tests can swap them for in-memory doubles.

    Args:
        tenant_id: The tenant id (also the summary key).
        tenant: The tenant entity to upsert.
        tenant_map: The map entity to upsert (FK child of the tenant).
        cameras: Registry camera dicts to turn into camera configs.
        credential: The tenant camera credential for each camera config.
        session_factory: Database session context-manager factory.

    Returns:
        A :class:`BootstrapSummary` of the rows ensured.
    """
    configs = _build_camera_configs(
        tenant_id=tenant_id,
        map_id=tenant_map.id,
        cameras=cameras,
        credential=credential,
    )
    with session_factory() as session:
        RepoPostgresTenant(session).save(tenant)
        RepoPostgresMap(session).save(tenant_map)
        camera_repo = RepoPostgresCameraConfig(session)
        for config in configs:
            camera_repo.save(config)

    return BootstrapSummary(
        tenant_id=tenant_id,
        map_id=tenant_map.id,
        camera_ids=[config.id for config in configs],
    )


def _load_tenant_map(tenant_id: str, maps_dir: Path) -> Map:
    """Load a tenant's map entity from its ``data/maps/<tenant>.yaml`` sidecar.

    ``RepoYamlMap`` reads by filename stem (the tenant), and the loaded
    ``Map.id`` is the sidecar's internal ``id`` (the stable map id). Derives the
    ``{tenant_id}/{photo.path}`` asset key when the sidecar omits it.

    Args:
        tenant_id: The tenant whose sidecar to load.
        maps_dir: The directory holding the ``<tenant>.yaml`` sidecars.

    Returns:
        The map entity with a guaranteed ``asset_key``.

    Raises:
        typer.Exit: If the sidecar is missing.
    """
    tenant_map = RepoYamlMap(maps_dir).get(tenant_id)
    if tenant_map is None:
        typer.echo(
            f"Error: Map sidecar for tenant '{tenant_id}' not found at "
            f"data/maps/{tenant_id}.yaml — run bootstrap requires the sidecar.",
            err=True,
        )
        raise typer.Exit(1)
    if not tenant_map.asset_key:
        tenant_map = dataclasses.replace(
            tenant_map, asset_key=f"{tenant_map.tenant_id}/{tenant_map.photo.path}"
        )
    return tenant_map


@calibrate_app.command("bootstrap")
def bootstrap_command(
    tenant: str = typer.Option(..., "--tenant", help="Tenant id (e.g., 'icozee')"),
) -> None:
    """Ensure the reference rows the tenant's calibration pipeline FKs require.

    Idempotently upserts the tenant, its map, and one ``camera_config`` per
    registry camera (in FK order), sourced from the existing registry + map
    sidecar. Re-running updates rows in place and never duplicates. Run this
    once per tenant before ``run-camera`` / ``rollout``.
    """
    if not os.environ.get("DATABASE_URL"):
        typer.echo("Error: DATABASE_URL environment variable is not set.", err=True)
        raise typer.Exit(1)

    tenant_dict = get_tenant_by_id(tenant)
    if not tenant_dict:
        typer.echo(f"Error: Tenant '{tenant}' not found.", err=True)
        raise typer.Exit(1)

    cameras = get_cameras_for_tenant(tenant)
    if not cameras:
        typer.echo(f"Error: No cameras found for tenant '{tenant}'.", err=True)
        raise typer.Exit(1)

    username, password = get_tenant_credentials(tenant)
    credential = Credential(username or "", password or "")

    tenant_map = _load_tenant_map(tenant, DEFAULT_MAPS_DIR)

    summary = _run_bootstrap(
        tenant_id=tenant,
        tenant=Tenant.from_dict(tenant_dict),
        tenant_map=tenant_map,
        cameras=cameras,
        credential=credential,
        session_factory=get_session,
    )

    typer.echo(
        f"Bootstrap: tenant={summary.tenant_id} map={summary.map_id} "
        f"cameras={len(summary.camera_ids)}"
    )


__all__ = ["bootstrap_command"]
