"""TTL-cached camera registry.

Bridges the PostgreSQL ``CameraConfig`` store (Neon) and the hardcoded
``camera_config.CAMERAS`` fallback so that existing callers transparently get
database-backed camera data without code changes.

Design notes
------------
* The PostgreSQL ``camera_configs`` table (see
  :class:`poc_homography.infrastructure.models.camera_config.CameraConfigModel`)
  stores only *registration* data: ``id``, ``tenant_id``, ``map_id``, ``name``,
  ``spec``, ``credential`` (``{username, password}``) and ``ip_address``.  It does
  **not** store the rich calibration parameters (``k1``, ``k2``,
  ``pan_offset_deg``, ``sensor_width_mm`` …) that homography
  and calibration consumers rely on.  Those still live in the hardcoded
  ``CAMERAS`` list.

  To satisfy both "read from the database" and "no changes in the 19 importers",
  the registry **merges** the database base fields (ip, credentials, name) onto
  the hardcoded calibration template for the matching camera ``id``.  Cameras
  absent from the database fall back to their hardcoded entry verbatim.

* If the database is unavailable (``DATABASE_URL`` unset, connection error, empty
  result set) the registry falls back to the hardcoded ``CAMERAS`` / tenant data.

* Access is thread-safe and the cache honours a TTL configured via the
  ``CAMERA_CACHE_TTL`` environment variable (seconds, default 300).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_config import CameraConfig

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = 300


def _now() -> float:
    """Monotonic clock used for TTL accounting.

    Indirected through a module-level function so tests can monkeypatch it.
    """
    return time.monotonic()


def _resolve_ttl() -> int:
    """Resolve the cache TTL from ``CAMERA_CACHE_TTL`` (default 300 seconds)."""
    raw = os.getenv("CAMERA_CACHE_TTL")
    if raw is None or raw == "":
        return DEFAULT_CACHE_TTL
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid CAMERA_CACHE_TTL=%r; falling back to default %d seconds",
            raw,
            DEFAULT_CACHE_TTL,
        )
        return DEFAULT_CACHE_TTL


def _spec_model_name(spec: Any) -> str:
    """Best-effort human-readable model name for a ``CameraSpec`` enum member."""
    return getattr(spec, "model_name", None) or getattr(spec, "name", str(spec))


def _entity_to_camera_dict(entity: CameraConfig) -> dict[str, Any]:
    """Transform a ``CameraConfig`` entity into a legacy camera dict.

    Maps database field names to the keys used by the hardcoded ``CAMERAS``
    list (``ip_address`` -> ``ip``) and flattens the credential value object
    into per-camera ``username`` / ``password`` fields.
    """
    credential = entity.credential
    return {
        "id": entity.id,
        "tenant_id": entity.tenant_id,
        "map_id": entity.map_id,
        "name": entity.name,
        "ip": entity.ip_address,
        "username": credential.username,
        "password": credential.password,
        "model": _spec_model_name(entity.spec),
    }


def _hardcoded_cameras() -> list[dict[str, Any]]:
    """Hardcoded fallback cameras (lazy import to avoid an import cycle)."""
    from poc_homography.camera_config import CAMERAS

    return CAMERAS


class CameraRegistry:
    """Singleton TTL cache over camera registration data.

    The cache stores::

        {
            "cameras": {camera_id: camera_dict, ...},
            "last_loaded": <monotonic timestamp or None>,
            "ttl": <seconds>,
        }

    Tenant data is intentionally out of scope: ``camera_config`` already loads
    tenants from a dynamic source (the DDD YAML repository) with env-injected
    credentials, so it does not need this cache.
    """

    _instance: CameraRegistry | None = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> CameraRegistry:
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init_state()
                    cls._instance = instance
        return cls._instance

    def _init_state(self) -> None:
        # ``_lock`` guards the in-memory cache (held only for microseconds).
        # ``_refresh_lock`` serialises database refreshes so the (slow) DB load
        # never runs while ``_lock`` is held — readers are not blocked behind a
        # network round-trip, and at most one thread queries the DB per refresh.
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._cameras: dict[str, dict[str, Any]] = {}
        self._last_loaded: float | None = None
        # Placeholder until the first refresh resolves the real TTL; never used
        # for an expiry decision (the initial cache is always expired because
        # ``_last_loaded`` is None).
        self._ttl: int = DEFAULT_CACHE_TTL

    # -- internal helpers -------------------------------------------------

    def _is_expired(self) -> bool:
        if self._last_loaded is None:
            return True
        return (_now() - self._last_loaded) >= self._ttl

    def _load_from_database(self) -> list[dict[str, Any]] | None:
        """Load cameras from PostgreSQL.

        Returns a list of camera dicts on success, or ``None`` when the database
        is unavailable / returns no rows (signalling the caller to fall back).
        """
        try:
            from poc_homography.infrastructure.database import get_session
            from poc_homography.infrastructure.repositories import (
                RepoPostgresCameraConfig,
            )

            with get_session() as session:
                entities = RepoPostgresCameraConfig(session).get_all()
        except Exception as exc:
            # Log only the exception *type*, never the traceback or message:
            # SQLAlchemy/psycopg connection errors can embed DATABASE_URL
            # (user:password@host) in their text.
            logger.warning(
                "Camera database query failed (%s); falling back to hardcoded CAMERAS",
                type(exc).__name__,
            )
            return None

        if not entities:
            logger.info("Camera database returned no rows; falling back to hardcoded CAMERAS")
            return None

        return [_entity_to_camera_dict(e) for e in entities]

    def _build_cameras(self) -> dict[str, dict[str, Any]]:
        """Build the camera cache, merging DB data over hardcoded calibration.

        Starts from the hardcoded ``CAMERAS`` (preserving calibration fields),
        then overlays the non-empty database fields for each matching camera id.
        Cameras present only in the database are added; cameras present only in
        the hardcoded list are kept as-is.
        """
        cameras: dict[str, dict[str, Any]] = {
            cam["id"]: dict(cam) for cam in _hardcoded_cameras() if cam.get("id")
        }

        db_cameras = self._load_from_database()
        if db_cameras is None:
            return cameras

        for db_cam in db_cameras:
            cam_id = db_cam.get("id")
            if not cam_id:
                continue
            # Only overlay non-None DB values so a NULL ip_address does not wipe
            # the hardcoded calibration template's ip.
            overlay = {k: v for k, v in db_cam.items() if v is not None}
            base = cameras.get(cam_id, {})
            cameras[cam_id] = {**base, **overlay}
        return cameras

    def _ensure_fresh(self) -> None:
        """Refresh the cache if the TTL has lapsed, without blocking readers.

        The database load runs *outside* ``self._lock`` so concurrent reads are
        never stalled behind a network round-trip. ``self._refresh_lock``
        ensures only one thread performs the load per refresh; the result is
        swapped in under ``self._lock`` (a microsecond-scale critical section).
        """
        with self._lock:
            if not self._is_expired():
                return

        with self._refresh_lock:
            # Re-check: another thread may have refreshed while we waited.
            with self._lock:
                if not self._is_expired():
                    return

            ttl = _resolve_ttl()
            cameras = self._build_cameras()  # DB I/O — no cache lock held
            now = _now()

            with self._lock:
                self._ttl = ttl
                self._cameras = cameras
                self._last_loaded = now

    # -- public API -------------------------------------------------------

    def get_all_cameras(self) -> list[dict[str, Any]]:
        """Return all cameras (database-backed where available)."""
        self._ensure_fresh()
        with self._lock:
            return [dict(cam) for cam in self._cameras.values()]

    def get_camera_by_id(self, camera_id: str) -> dict[str, Any] | None:
        """Return a single camera by id, or ``None`` if unknown.

        The cache is always seeded from the hardcoded ``CAMERAS`` list (then
        overlaid with database data), so a camera absent from the database is
        still served from its hardcoded entry — no separate fallback is needed.
        """
        self._ensure_fresh()
        with self._lock:
            cam = self._cameras.get(camera_id)
            return dict(cam) if cam is not None else None

    def get_cameras_for_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all cameras belonging to ``tenant_id``."""
        self._ensure_fresh()
        with self._lock:
            return [
                dict(cam) for cam in self._cameras.values() if cam.get("tenant_id") == tenant_id
            ]

    def invalidate(self) -> None:
        """Clear the cache and reset the load timestamp (thread-safe)."""
        with self._lock:
            self._cameras = {}
            self._last_loaded = None


def get_registry() -> CameraRegistry:
    """Return the process-wide :class:`CameraRegistry` singleton."""
    return CameraRegistry()


def invalidate_cache() -> None:
    """Invalidate the camera registry cache.

    The next access after invalidation triggers a fresh database query.
    Intended for administrative actions (e.g. after rotating credentials or
    updating a camera's IP address in the database).
    """
    get_registry().invalidate()


__all__ = ["CameraRegistry", "get_registry", "invalidate_cache"]
