"""Session-aware frame, annotation, and line-annotation helpers for the API layer.

Replaces the repo-backed functions in ``homography_web.frame_utils`` so that
API routers never instantiate ``RepoYaml*`` classes.  Pure utilities and path
constants are re-exported from ``frame_utils`` for convenience.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — kept at runtime; re-exported and used by API route modules
from typing import TYPE_CHECKING

from homography_web.frame_utils import (
    CALIBRATIONS_DIR,
    DATA_MAPS_DIR,
    FRAMES_DIR,
    WEBAPP_DIR,
    normalize_array,
    validate_image_filename,
)

from api.utils.map_assets import resolve_map_geotiff
from poc_homography.infrastructure.repositories import (
    RepoPostgresCapturedFrame,
    RepoPostgresLineAnnotation,
    RepoPostgresMap,
)
from poc_homography.maps.geotiff_io import extract_geotiff

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.domain.entities.annotation import Annotation
    from poc_homography.domain.entities.captured_frame import CapturedFrame
    from poc_homography.domain.entities.map import Map

__all__ = [
    "CALIBRATIONS_DIR",
    "DATA_MAPS_DIR",
    "FRAMES_DIR",
    "WEBAPP_DIR",
    "extract_geotiff",
    "get_frame_image_path",
    "get_map_for_tenant",
    "image_filename_to_frame",
    "list_frames",
    "list_image_filenames",
    "load_annotations_for_frame",
    "load_line_annotations_for_frame",
    "map_has_image",
    "normalize_array",
    "resolve_map",
    "resolve_map_for_tenant",
    "save_annotations_for_frame",
    "validate_image_filename",
]


def get_map_for_tenant(tenant_id: str, session: Session) -> Map | None:
    """Return the ``Map`` for *tenant_id*, or ``None``."""
    repo = RepoPostgresMap(session)
    maps = repo.get_by_tenant(tenant_id)
    if maps:
        return next(iter(maps.values()))
    return None


def resolve_map_for_tenant(tenant_id: str, session: Session) -> tuple[Map, Path]:
    """Resolve the ``Map`` and its image file for *tenant_id*.

    Raises:
        RuntimeError: If no map is configured or the file is missing.
    """
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        raise RuntimeError(f"No map configured for tenant: {tenant_id}")

    map_file = resolve_map_geotiff(map_entity)
    if map_file is None:
        raise RuntimeError(f"Map asset not found for tenant: {tenant_id}")

    return map_entity, map_file


def resolve_map(map_id: str, session: Session) -> tuple[Map, Path]:
    """Resolve a specific ``Map`` and its image file by *map_id*.

    The image is resolved via :func:`resolve_map_geotiff`, so object-storage
    assets (``Map.asset_key``, see #292) are materialised the same way the
    tile/info endpoints do; local-only maps fall back to ``data/maps``.

    Raises:
        LookupError: If no map with *map_id* exists.
        RuntimeError: If the map's image asset is missing.
    """
    map_entity = RepoPostgresMap(session).get(map_id)
    if map_entity is None:
        raise LookupError(f"Map not found: {map_id}")

    map_file = resolve_map_geotiff(map_entity)
    if map_file is None:
        raise RuntimeError(f"Map asset not found: {map_id}")

    return map_entity, map_file


def map_has_image(map_entity: Map) -> bool:
    """Return ``True`` iff *map_entity* has a usable reference image.

    A map is "configured" when it carries an object-storage asset key (#292)
    or a local file under ``data/maps``. The asset-key branch is intentionally
    cheap — it does not download the GeoTIFF — so listing many maps for the
    selector never triggers per-map object-storage fetches.
    """
    if map_entity.asset_key:
        return True
    path = map_entity.photo.path
    # An empty path normalises to ``Path(".")``; treat that as unconfigured.
    if not str(path) or str(path) == ".":
        return False
    return (DATA_MAPS_DIR / path).is_file()


def _frame_repo(session: Session) -> RepoPostgresCapturedFrame:
    return RepoPostgresCapturedFrame(session, frames_dir=FRAMES_DIR)


def list_frames(session: Session, map_id: str | None = None) -> list[CapturedFrame]:
    """Return captured frames, optionally filtered by *map_id*."""
    repo = _frame_repo(session)
    if map_id is not None:
        return repo.get_by_map(map_id)
    return repo.get_all()


def image_filename_to_frame(
    filename: str, session: Session, *, map_id: str | None = None
) -> CapturedFrame | None:
    """Look up a ``CapturedFrame`` by its image filename."""
    for frame in list_frames(session, map_id):
        if frame.image_path.name == filename:
            return frame
    return None


def get_frame_image_path(frame: CapturedFrame) -> Path:
    """Return the absolute path to a frame's image file."""
    return FRAMES_DIR / frame.map_id / frame.camera_name / str(frame.image_path)


def list_image_filenames(session: Session, map_id: str | None = None) -> list[str]:
    """Return sorted image filenames from the captured-frame repo."""
    return sorted(f.image_path.name for f in list_frames(session, map_id))


def load_annotations_for_frame(frame_id: str, session: Session) -> list[dict]:
    """Load point annotations for a frame in legacy dict format.

    Returns list of ``{gcp_id, pixel_x, pixel_y}`` dicts.
    """
    repo = _frame_repo(session)
    annotations = repo.get_annotations(frame_id)
    return [
        {
            "gcp_id": ann.gcp_id,
            "pixel_x": round(float(ann.pixel.x), 1),
            "pixel_y": round(float(ann.pixel.y), 1),
        }
        for ann in annotations
    ]


def save_annotations_for_frame(
    frame_id: str, annotations: list[Annotation], session: Session
) -> None:
    """Persist point annotations for *frame_id*."""
    _frame_repo(session).save_annotations(frame_id, annotations)


def load_line_annotations_for_frame(frame_id: str, session: Session) -> list[dict]:
    """Load line annotations for a frame in legacy dict format."""
    repo = RepoPostgresLineAnnotation(session)
    results: list[dict] = []
    for ann in repo.get_by_frame_id(frame_id):
        entry: dict = {
            "line_id": ann.line_id,
            "start_pixel_x": float(ann.start_pixel.x),
            "start_pixel_y": float(ann.start_pixel.y),
            "end_pixel_x": float(ann.end_pixel.x),
            "end_pixel_y": float(ann.end_pixel.y),
        }
        if ann.points is not None:
            entry["points"] = [[float(p.x), float(p.y)] for p in ann.points]
        results.append(entry)
    return results
