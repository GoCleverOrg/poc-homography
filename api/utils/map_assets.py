"""Resolve a map's GeoTIFF to a local path, materialising it from object storage.

Maps may be backed by an object-storage asset (``Map.asset_key``, populated by
the upload pipeline — see #290/#291) rather than a file under ``data/maps``. The
tile/info endpoints need a *local* path to tile with ``tifffile``/PIL, so when a
map has an ``asset_key`` we download the GeoTIFF bytes once into a process-wide
``/tmp`` cache and reuse that path across the many per-tile requests. Maps
without an ``asset_key`` fall back to the legacy filesystem path so existing
local-only setups keep working.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from homography_web.frame_utils import DATA_MAPS_DIR

from poc_homography.infrastructure.clients.minio_map_store import MinioMapStore

if TYPE_CHECKING:
    from poc_homography.domain.entities.map import Map

# Process-wide cache for downloaded GeoTIFF assets. Asset keys are immutable (a
# fresh upload gets a fresh key), so caching key -> bytes on disk is safe to
# reuse across requests and across worker restarts.
_CACHE_DIR = Path(tempfile.gettempdir()) / "poc_homography_map_assets"


def _cache_path(asset_key: str) -> Path:
    """Local cache path mirroring ``asset_key`` under the cache directory."""
    # Strip any leading slash so the key stays *inside* the cache dir.
    return _CACHE_DIR / asset_key.lstrip("/")


def _materialise(asset_key: str, store: MinioMapStore) -> Path:
    """Download ``asset_key`` into the cache (idempotent) and return its path."""
    target = _cache_path(asset_key)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    data = store.get_map(asset_key)
    # Write to a unique temp file then atomically replace, so concurrent tile
    # requests never observe a half-written GeoTIFF.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.partial")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def resolve_map_geotiff(map_entity: Map, *, store: MinioMapStore | None = None) -> Path | None:
    """Resolve ``map_entity`` to a local GeoTIFF path.

    When the map carries an ``asset_key`` the GeoTIFF is materialised from object
    storage into a ``/tmp`` cache (the store is built from the environment when
    not injected). Otherwise the legacy ``data/maps`` path is used. Returns
    ``None`` only when no asset key is set *and* the local file is absent.
    """
    if map_entity.asset_key:
        store = store or MinioMapStore.from_env()
        return _materialise(map_entity.asset_key, store)

    local = DATA_MAPS_DIR / map_entity.photo.path
    return local if local.is_file() else None


__all__ = ["resolve_map_geotiff"]
