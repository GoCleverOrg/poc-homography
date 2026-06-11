"""Map-asset upload CLI commands (issue #290)."""

from __future__ import annotations

from pathlib import Path

import typer

from poc_homography.infrastructure.clients.minio_map_store import MinioMapStore
from poc_homography.maps import DEFAULT_MAPS_DIR, upload_map_assets

maps_app = typer.Typer(help="Map asset commands")


@maps_app.command("upload")
def upload_command(
    maps_dir: Path = typer.Option(
        DEFAULT_MAPS_DIR, help="Directory holding map YAML sidecars and GeoTIFFs"
    ),
) -> None:
    """Upload ``data/maps/*.tif`` GeoTIFFs to the MinIO map-assets bucket.

    Each map sidecar yields a tenant-scoped key ``{tenant_id}/{photo.path}``.
    Tifs absent on disk are skipped (run ``dvc pull`` to materialize them).
    """
    try:
        store = MinioMapStore.from_env()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    outcomes = upload_map_assets(Path(maps_dir), store)

    uploaded = 0
    missing = 0
    for outcome in outcomes:
        if outcome.status == "uploaded" and outcome.result is not None:
            uploaded += 1
            typer.echo(f"uploaded {outcome.key} (sha256={outcome.result.sha256})")
        else:
            missing += 1
            typer.echo(f"{outcome.key}: SKIP (missing on disk — run 'dvc pull')")

    typer.echo(f"Summary: {uploaded} uploaded, {missing} missing ({len(outcomes)} total)")
