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

from botocore.exceptions import ClientError
from homography_web.frame_utils import DATA_MAPS_DIR

from poc_homography.infrastructure.clients.minio_map_store import MinioMapStore

if TYPE_CHECKING:
    from poc_homography.domain.entities.map import Map

# Process-wide cache for downloaded GeoTIFF assets. Asset keys are immutable (a
# fresh upload gets a fresh key), so caching key -> bytes on disk is safe to
# reuse across requests and across worker restarts.
_CACHE_DIR = Path(tempfile.gettempdir()) / "poc_homography_map_assets"

# S3/MinIO error codes that mean "the object simply is not there" — treated as a
# missing asset (``None`` -> 404) rather than a server error.
_MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "404"})


def _cache_path(asset_key: str) -> Path:
    """Local cache path for ``asset_key``, kept strictly inside the cache dir.

    Raises:
        ValueError: If ``asset_key`` would escape the cache directory (e.g. via
            ``..`` segments) — defence-in-depth against a malformed/poisoned key.
    """
    base = _CACHE_DIR.resolve()
    target = (base / asset_key.lstrip("/")).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        msg = f"asset_key escapes the cache directory: {asset_key!r}"
        raise ValueError(msg) from exc
    return target


def _materialise(asset_key: str, store: MinioMapStore) -> Path:
    """Download ``asset_key`` into the cache (idempotent) and return its path."""
    target = _cache_path(asset_key)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    data = store.get_map(asset_key)
    # Write to a per-writer temp file then atomically replace, so concurrent tile
    # requests never observe a half-written GeoTIFF. ``mkstemp`` gives a unique
    # name across threads and processes; the temp is unlinked if anything fails
    # before the atomic replace (and is a no-op afterwards, since it was renamed).
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".partial")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def resolve_map_geotiff(map_entity: Map, *, store: MinioMapStore | None = None) -> Path | None:
    """Resolve ``map_entity`` to a local GeoTIFF path.

    When the map carries an ``asset_key`` the GeoTIFF is materialised from object
    storage into a ``/tmp`` cache (the store is built from the environment when
    not injected). Otherwise the legacy ``data/maps`` path is used. Returns
    ``None`` when the asset is absent in object storage *or* (no asset key) the
    local file is absent — preserving the legacy "missing map -> 404" behaviour.
    """
    if map_entity.asset_key:
        store = store or MinioMapStore.from_env()
        try:
            return _materialise(map_entity.asset_key, store)
        except ClientError as exc:
            # A map row that references an object missing from storage is a
            # "no map asset" condition (404), not a 500. Other S3 errors
            # (auth, connectivity) are genuine server faults and propagate.
            code = exc.response.get("Error", {}).get("Code")
            if code in _MISSING_OBJECT_CODES:
                return None
            raise

    local = DATA_MAPS_DIR / map_entity.photo.path
    return local if local.is_file() else None


__all__ = ["resolve_map_geotiff"]
