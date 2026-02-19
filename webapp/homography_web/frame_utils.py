"""Shared utilities for accessing captured frames, images and annotations via DDD repos.

All four webapp apps (homography_precision, camera_annotator, camera_line_annotator,
distortion_validator) use this module instead of reading test YAML files directly.

Pattern follows existing ``calibration_utils.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from poc_homography.infrastructure.repositories import (
    RepoYamlCapturedFrame,
    RepoYamlLineAnnotation,
)

if TYPE_CHECKING:
    from poc_homography.domain.entities.captured_frame import CapturedFrame

# Paths relative to project root
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "captured_frames"
LINE_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "line_annotations"
LINES_DIR = PROJECT_ROOT / "data" / "lines"

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_frame_repo: RepoYamlCapturedFrame | None = None
_frames: list[CapturedFrame] | None = None
_image_to_frame: dict[str, CapturedFrame] | None = None


def get_frame_repo() -> RepoYamlCapturedFrame:
    """Return a cached CapturedFrame repository instance."""
    global _frame_repo
    if _frame_repo is None:
        _frame_repo = RepoYamlCapturedFrame(FRAMES_DIR)
    return _frame_repo


def list_frames() -> list[CapturedFrame]:
    """Return all captured frames (cached)."""
    global _frames
    if _frames is None:
        _frames = get_frame_repo().get_all()
    return _frames


def _get_image_to_frame() -> dict[str, CapturedFrame]:
    """Build and cache mapping from image filename -> CapturedFrame."""
    global _image_to_frame
    if _image_to_frame is None:
        _image_to_frame = {f.image_path.name: f for f in list_frames()}
    return _image_to_frame


def get_frame_image_path(frame: CapturedFrame) -> Path:
    """Return the absolute path to a frame's image file."""
    return get_frame_repo().get_image_path(frame)


def list_image_filenames() -> list[str]:
    """Return sorted list of image filenames from the captured-frame repo.

    Backward-compatible with the old ``get_available_images()`` pattern.
    """
    return sorted(_get_image_to_frame())


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


def load_line_annotations_for_frame(frame_id: str) -> list[dict]:
    """Load line annotations for a frame in legacy dict format.

    Returns list of dicts with ``line_id``, pixel endpoints, and optional
    ``points`` array for n-point polylines.
    """
    repo = RepoYamlLineAnnotation(LINE_ANNOTATIONS_DIR)
    all_annotations = repo.get_all()
    results: list[dict] = []
    for ann in all_annotations:
        if ann.frame_id != frame_id:
            continue
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


def invalidate_cache() -> None:
    """Clear all module-level caches. Useful after saves."""
    global _frame_repo, _frames, _image_to_frame
    _frame_repo = None
    _frames = None
    _image_to_frame = None
