"""Framework-free pipeline that uploads map GeoTIFFs to object storage.

Each map is described by a YAML sidecar in ``data/maps/`` (e.g. ``icozee.yaml``)
that carries a ``tenant_id`` and a ``photo.path`` (the GeoTIFF filename). The
object key is ``{tenant_id}/{photo.path}`` so map assets are tenant-scoped, e.g.
``icozee/icozee-cropped.tif``.

The real ``.tif`` files are not normally on disk (map assets are served from
S3/MinIO at runtime); place the ``.tif`` next to its YAML sidecar to upload it.
When a tif is absent the pipeline records a ``missing`` outcome and continues
(graceful skip) rather than raising, so a run reports clearly which assets are
still absent.

This module is deliberately framework-free: it takes any object exposing
``ensure_bucket()`` and ``put_map(data, key)`` (see
:class:`poc_homography.infrastructure.clients.minio_map_store.MinioMapStore`),
which keeps it unit-testable with a fake store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import yaml

if TYPE_CHECKING:
    from poc_homography.infrastructure.clients.minio_frame_store import PutResult

# <project_root>/data/maps, computed relative to this file.
DEFAULT_MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "maps"


class MapStore(Protocol):
    """Minimal store interface the uploader depends on (structural typing)."""

    def ensure_bucket(self) -> None:
        """Create the target bucket if it does not exist (idempotent)."""
        ...

    def put_map(self, data: bytes, key: str, content_type: str = ...) -> PutResult:
        """Upload ``data`` under ``key`` and return its location + sha256."""
        ...


@dataclass(frozen=True)
class UploadOutcome:
    """Result of processing a single map sidecar."""

    map_id: str
    key: str
    status: Literal["uploaded", "missing"]
    result: PutResult | None


def upload_map_assets(
    maps_dir: Path,
    store: MapStore,
    *,
    ensure_bucket: bool = True,
) -> list[UploadOutcome]:
    """Upload every map GeoTIFF under ``maps_dir`` to ``store``.

    Globs ``maps_dir/*.yaml``; for each sidecar with both ``tenant_id`` and
    ``photo.path``, computes the key ``{tenant_id}/{photo.path}`` and resolves
    the tif at ``maps_dir/{photo.path}``. Existing tifs are uploaded via
    ``store.put_map(bytes, key)``; absent tifs yield a ``missing`` outcome. The
    operation is idempotent: re-running overwrites the same keys.

    Args:
        maps_dir: Directory holding the YAML sidecars and (locally-present) tifs.
        store: Object exposing ``ensure_bucket()`` and ``put_map(data, key)``.
        ensure_bucket: When True, call ``store.ensure_bucket()`` once up front.

    Returns:
        One :class:`UploadOutcome` per usable sidecar, in sorted filename order.
    """
    if ensure_bucket:
        store.ensure_bucket()

    outcomes: list[UploadOutcome] = []
    for sidecar in sorted(maps_dir.glob("*.yaml")):
        with sidecar.open() as f:
            data = yaml.safe_load(f) or {}

        tenant_id = data.get("tenant_id")
        photo = data.get("photo") or {}
        path = photo.get("path") if isinstance(photo, dict) else None
        if not tenant_id or not path:
            # Sidecar lacks the fields needed to derive a key; skip it.
            continue

        map_id = str(data.get("id", sidecar.stem))
        key = f"{tenant_id}/{path}"
        tif_path = maps_dir / path
        if not tif_path.is_file():
            outcomes.append(UploadOutcome(map_id=map_id, key=key, status="missing", result=None))
            continue

        result = store.put_map(tif_path.read_bytes(), key)
        outcomes.append(UploadOutcome(map_id=map_id, key=key, status="uploaded", result=result))

    return outcomes


__all__ = ["DEFAULT_MAPS_DIR", "UploadOutcome", "upload_map_assets"]
