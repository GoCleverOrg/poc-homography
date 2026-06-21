"""Map-asset pipeline: GeoTIFF upload, raster loading, and ortho line detection."""

from __future__ import annotations

from poc_homography.maps.asset_uploader import (
    DEFAULT_MAPS_DIR,
    UploadOutcome,
    upload_map_assets,
)
from poc_homography.maps.geotiff_io import (
    extract_geotiff,
    load_ortho_raster,
    load_ortho_raster_from_bytes,
)
from poc_homography.maps.ortho_line_detection import OrthoLineDetector

__all__ = [
    "DEFAULT_MAPS_DIR",
    "OrthoLineDetector",
    "UploadOutcome",
    "extract_geotiff",
    "load_ortho_raster",
    "load_ortho_raster_from_bytes",
    "upload_map_assets",
]
