"""Shared utilities for calibration-related Django apps.

Provides filename validation, safe path resolution, and mtime-based
calibration table caching used by both ``lens_calibration`` and
``distortion_validator``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Filename validation helpers
# ---------------------------------------------------------------------------

def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks."""
    if not filename:
        return False
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def resolve_safe_path(filename: str, base_dir: Path) -> Path | None:
    """Resolve *filename* under *base_dir*, returning ``None`` on traversal."""
    if not validate_filename(filename):
        return None
    try:
        resolved = (base_dir / filename).resolve()
        if not resolved.is_relative_to(base_dir.resolve()):
            return None
        return resolved
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Calibration file cache
# ---------------------------------------------------------------------------

_calibration_cache: dict[tuple[str, float], Any] = {}


def get_cached_calibration_table(filepath: Path):
    """Return a cached CameraCalibrationTable, invalidated by mtime."""
    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    key_path = str(filepath)
    mtime = filepath.stat().st_mtime
    cache_key = (key_path, mtime)
    if cache_key not in _calibration_cache:
        # Evict stale entries for this path
        _calibration_cache.pop(
            next((k for k in _calibration_cache if k[0] == key_path), None),  # type: ignore[arg-type]
            None,
        )
        _calibration_cache[cache_key] = CameraCalibrationTable.load(filepath)
    return _calibration_cache[cache_key]
