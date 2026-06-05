"""Shared utilities for accessing captured frames, images and annotations via DDD repos.

All four webapp apps (homography_precision, camera_annotator, camera_line_annotator,
distortion_validator) use this module instead of reading test YAML files directly.

Pattern follows existing ``calibration_utils.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from django.http import FileResponse, HttpResponse

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.infrastructure.repositories import (
    RepoYamlAnnotation,
    RepoYamlCapturedFrame,
    RepoYamlLineAnnotation,
    RepoYamlMap,
    RepoYamlTenant,
)
from poc_homography.types import Easting, Meters, Northing, Unitless

if TYPE_CHECKING:
    import tifffile
    from django.http import HttpRequest
    from numpy.typing import NDArray

    from poc_homography.domain.entities.captured_frame import CapturedFrame
    from poc_homography.domain.entities.map import Map

# Paths relative to project root
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
DATA_MAPS_DIR = PROJECT_ROOT / "data" / "maps"
FRAMES_DIR = PROJECT_ROOT / "data" / "captured_frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
GCPS_DIR = PROJECT_ROOT / "data" / "gcps"
LINE_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "line_annotations"
LINES_DIR = PROJECT_ROOT / "data" / "lines"
CALIBRATIONS_DIR = PROJECT_ROOT / "data" / "lens_calibrations"
CALIBRATION_LINE_TRACES_DIR = PROJECT_ROOT / "data" / "calibration_line_traces"

# ---------------------------------------------------------------------------
# Filename validation helpers
# ---------------------------------------------------------------------------


def validate_image_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks."""
    if not filename:
        return False
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def normalize_array(arr: NDArray) -> NDArray[np.uint8]:
    """Normalize array values to 0-255 range for display.

    Args:
        arr: Input array.

    Returns:
        Normalized uint8 array.
    """
    if arr.dtype == np.uint8:
        return arr

    # Handle floating point and other types
    arr = arr.astype(np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val - min_val > 0:
        arr = (arr - min_val) / (max_val - min_val) * 255
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def extract_geotiff(tif: tifffile.TiffFile) -> GeoTiff | None:
    """Extract GeoTiff VO from TIFF tags.

    Args:
        tif: Open tifffile TiffFile object.

    Returns:
        GeoTiff value object, or None if metadata not available.
    """
    page = tif.pages[0]
    # type: ignore needed because tifffile types are incomplete
    tags = {tag.name: tag for tag in page.tags.values()}  # type: ignore[union-attr]

    gt_params: list[float] | None = None
    crs: str | None = None

    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        try:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value

            origin_x = tiepoint[3] - tiepoint[0] * scale[0]
            origin_y = tiepoint[4] + tiepoint[1] * scale[1]

            gt_params = [
                float(origin_x),
                float(scale[0]),
                0.0,
                float(origin_y),
                0.0,
                -float(scale[1]),
            ]
        except (IndexError, TypeError, ValueError):
            pass

    elif "ModelTransformationTag" in tags:
        try:
            matrix = tags["ModelTransformationTag"].value
            gt_params = [
                float(matrix[3]),
                float(matrix[0]),
                float(matrix[1]),
                float(matrix[7]),
                float(matrix[4]),
                float(matrix[5]),
            ]
        except (IndexError, TypeError, ValueError):
            pass

    if "GeoKeyDirectoryTag" in tags:
        try:
            geo_keys = tags["GeoKeyDirectoryTag"].value
            for i in range(4, len(geo_keys), 4):
                key_id = geo_keys[i]
                if key_id in (3072, 2048):
                    crs = f"EPSG:{geo_keys[i + 3]}"
                    break
        except (IndexError, TypeError, ValueError):
            pass

    if gt_params is None or crs is None:
        return None

    return GeoTiff(
        geotransform=GeoTransform(
            origin_easting=Easting(gt_params[0]),
            pixel_width=Meters(gt_params[1]),
            row_rotation=Unitless(gt_params[2]),
            origin_northing=Northing(gt_params[3]),
            col_rotation=Unitless(gt_params[4]),
            pixel_height=Meters(gt_params[5]),
        ),
        crs=crs,
    )


# ---------------------------------------------------------------------------
# Tenant-aware helpers
# ---------------------------------------------------------------------------


class TenantIdError(ValueError):
    """Raised when tenant_id is missing or invalid."""


def get_tenant_id(request: HttpRequest) -> str:
    """Extract tenant_id from request query string.

    Args:
        request: Django HTTP request.

    Returns:
        Tenant ID string.

    Raises:
        TenantIdError: If tenant_id is missing or unknown.
    """
    tenant_id: str | None = request.GET.get("tenant_id")
    if not tenant_id:
        raise TenantIdError("Missing required query parameter: tenant_id")
    if get_tenant_repo().get(tenant_id) is None:
        raise TenantIdError(f"Unknown tenant_id: {tenant_id!r}")
    return tenant_id


_tenant_repo: RepoYamlTenant | None = None
_map_repo: RepoYamlMap | None = None


def get_tenant_repo() -> RepoYamlTenant:
    """Return a cached Tenant repository instance."""
    global _tenant_repo
    if _tenant_repo is None:
        _tenant_repo = RepoYamlTenant(PROJECT_ROOT / "data" / "tenants")
    return _tenant_repo


def get_map_repo() -> RepoYamlMap:
    """Return a cached Map repository instance."""
    global _map_repo
    if _map_repo is None:
        _map_repo = RepoYamlMap(PROJECT_ROOT / "data" / "maps")
    return _map_repo


def get_map_from_tenant_id(tenant_id: str) -> Map | None:
    """Look up the ``Map`` entity that belongs to *tenant_id*.

    Searches the YAML map repository for an exact tenant match and returns
    the map entity, or ``None`` when no map is configured for the tenant.

    Args:
        tenant_id: Tenant identifier (must be provided explicitly).
    """
    maps = get_map_repo().get_by_tenant(tenant_id)
    if maps:
        return cast("Map", next(iter(maps.values())))
    return None


def resolve_map_for_tenant(tenant_id: str) -> tuple[Map, Path]:
    """Look up the Map entity for *tenant_id* and verify its image file.

    Args:
        tenant_id: Tenant identifier.

    Returns:
        Tuple of (Map entity, absolute Path to the map image file).

    Raises:
        RuntimeError: If no map is configured or the file is missing.
    """
    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        raise RuntimeError(f"No map configured for tenant: {tenant_id}")

    map_file = DATA_MAPS_DIR / map_entity.photo.path
    if not map_file.exists():
        raise RuntimeError(f"Map file not found: {map_file}")

    return map_entity, map_file


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_annotation_repo: RepoYamlAnnotation | None = None
_frame_repo: RepoYamlCapturedFrame | None = None
_line_ann_repo: RepoYamlLineAnnotation | None = None
_line_anns_by_frame: dict[str, list] | None = None
_frames: list[CapturedFrame] | None = None
_image_to_frame: dict[str, CapturedFrame] | None = None


def get_annotation_repo() -> RepoYamlAnnotation:
    """Return a cached Annotation repository instance."""
    global _annotation_repo
    if _annotation_repo is None:
        _annotation_repo = RepoYamlAnnotation(ANNOTATIONS_DIR)
    return _annotation_repo


def get_line_annotation_repo() -> RepoYamlLineAnnotation:
    """Return a cached LineAnnotation repository instance."""
    global _line_ann_repo
    if _line_ann_repo is None:
        _line_ann_repo = RepoYamlLineAnnotation(LINE_ANNOTATIONS_DIR)
    return _line_ann_repo


def get_frame_repo() -> RepoYamlCapturedFrame:
    """Return a cached CapturedFrame repository instance."""
    global _frame_repo
    if _frame_repo is None:
        _frame_repo = RepoYamlCapturedFrame(FRAMES_DIR)
    return _frame_repo


def list_frames(map_id: str | None = None) -> list[CapturedFrame]:
    """Return captured frames (cached).

    Args:
        map_id: If provided, only return frames belonging to this map.
    """
    global _frames
    if _frames is None:
        _frames = get_frame_repo().get_all()
    if map_id is None:
        return _frames
    return [f for f in _frames if f.map_id == map_id]


def _get_image_to_frame() -> dict[str, CapturedFrame]:
    """Build and cache mapping from image filename -> CapturedFrame."""
    global _image_to_frame
    if _image_to_frame is None:
        _image_to_frame = {f.image_path.name: f for f in list_frames()}
    return _image_to_frame


def get_frame_image_path(frame: CapturedFrame) -> Path:
    """Return the absolute path to a frame's image file."""
    return get_frame_repo().get_image_path(frame)


def list_image_filenames(map_id: str | None = None) -> list[str]:
    """Return sorted list of image filenames from the captured-frame repo.

    Args:
        map_id: If provided, only return filenames for frames belonging to this map.
    """
    mapping = _get_image_to_frame()
    if map_id is None:
        return sorted(mapping)
    return sorted(fn for fn, frame in mapping.items() if frame.map_id == map_id)


def image_filename_to_frame(filename: str) -> CapturedFrame | None:
    """Look up a CapturedFrame by its image filename (e.g. ``valte_30.8_13.1_1_20260112.png``)."""
    return _get_image_to_frame().get(filename)


def load_annotations_for_frame(frame_id: str) -> list[dict]:
    """Load point annotations for a frame in legacy dict format.

    Returns list of ``{gcp_id, pixel_x, pixel_y}`` dicts.
    """
    annotations = get_frame_repo().get_annotations(frame_id)
    return [
        {
            "gcp_id": ann.gcp_id,
            "pixel_x": round(float(ann.pixel.x), 1),
            "pixel_y": round(float(ann.pixel.y), 1),
        }
        for ann in annotations
    ]


def _get_line_anns_by_frame() -> dict[str, list]:
    """Build and cache mapping from frame_id -> list of line annotations."""
    global _line_anns_by_frame
    if _line_anns_by_frame is None:
        by_frame: dict[str, list] = {}
        for ann in get_line_annotation_repo().get_all():
            by_frame.setdefault(ann.frame_id, []).append(ann)
        _line_anns_by_frame = by_frame
    return _line_anns_by_frame


def load_line_annotations_for_frame(frame_id: str) -> list[dict]:
    """Load line annotations for a frame in legacy dict format.

    Returns list of dicts with ``line_id``, pixel endpoints, and optional
    ``points`` array for n-point polylines.
    """
    results: list[dict] = []
    for ann in _get_line_anns_by_frame().get(frame_id, []):
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


_invalidation_callbacks: list = []


def register_invalidation_callback(cb) -> None:
    """Register a callback to be called when caches are invalidated."""
    _invalidation_callbacks.append(cb)


def invalidate_cache() -> None:
    """Clear all module-level caches. Useful after saves."""
    global _annotation_repo, _frame_repo, _frames, _image_to_frame
    global _line_ann_repo, _line_anns_by_frame, _tenant_repo, _map_repo
    _annotation_repo = None
    _frame_repo = None
    _frames = None
    _image_to_frame = None
    _line_ann_repo = None
    _line_anns_by_frame = None
    _tenant_repo = None
    _map_repo = None
    for cb in _invalidation_callbacks:
        cb()


# ---------------------------------------------------------------------------
# Shared image utilities (used by camera_annotator & camera_line_annotator)
# ---------------------------------------------------------------------------


def get_available_images(map_id: str | None = None) -> list[str]:
    """Return available image filenames from the CapturedFrame repo."""
    return list_image_filenames(map_id)


def get_available_images_fresh(map_id: str | None = None) -> list[str]:
    """Return available image filenames by reading directly from disk.

    Creates a fresh ``RepoYamlCapturedFrame`` to bypass the module-level
    cache, ensuring newly written YAML files are always reflected.  The
    module-level globals are **not** mutated.
    """
    repo = RepoYamlCapturedFrame(FRAMES_DIR, create_dir=False)
    frames = repo.get_all()
    if map_id is not None:
        frames = [f for f in frames if f.map_id == map_id]
    return sorted(f.image_path.name for f in frames)


def _fresh_frame_by_filename(filename: str) -> CapturedFrame | None:
    """Look up a CapturedFrame by filename using a fresh disk read.

    Used as a fallback when the module-level cache does not contain a
    recently written frame.
    """
    repo = RepoYamlCapturedFrame(FRAMES_DIR, create_dir=False)
    for frame in repo.get_all():
        if frame.image_path.name == filename:
            return frame
    return None


def get_current_image(
    request: HttpRequest,
    session_key: str,
    map_id: str | None = None,
) -> str | None:
    """Get the current image filename from session, or first available image.

    Args:
        request: Django HTTP request.
        session_key: Session key used to store current image filename.
        map_id: If provided, only consider images for this map.
    """
    session_image = request.session.get(session_key)  # type: ignore[reportAttributeAccessIssue]  # session injected by Django SessionMiddleware
    if session_image:
        frame = image_filename_to_frame(session_image)
        if frame is not None and (map_id is None or frame.map_id == map_id):
            return session_image
        # Cache miss — check disk in case the frame was recently added
        if session_image in get_available_images_fresh(map_id):
            return session_image

    images = get_available_images(map_id)
    if images:
        return images[0]

    return None


def serve_current_image(
    request: HttpRequest,
    session_key: str,
    map_id: str | None = None,
) -> HttpResponse | FileResponse:
    """Serve the current image file for an annotator app.

    Args:
        request: Django HTTP request.
        session_key: Session key used to store current image filename.
        map_id: If provided, only consider images for this map.
    """
    import mimetypes

    from django.http import FileResponse

    current_image = get_current_image(request, session_key, map_id)
    if not current_image:
        return HttpResponse("No image available", status=404)

    frame = image_filename_to_frame(current_image)
    if frame is None:
        frame = _fresh_frame_by_filename(current_image)
    if frame is None:
        return HttpResponse("Image not found", status=404)

    resolved_path = get_frame_image_path(frame)
    if not resolved_path.exists():
        return HttpResponse("Image not found", status=404)

    mime_type, _ = mimetypes.guess_type(str(resolved_path))
    if not mime_type:
        mime_type = "image/jpeg"

    response = FileResponse(
        open(resolved_path, "rb"),
        content_type=mime_type,
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response
