"""Offline GeoTIFF raster + georeference loading via ``tifffile``.

This module standardizes orthophoto raster loading on :mod:`tifffile` (NOT
rasterio) so the same code path works fully offline against a saved GeoTIFF or
against bytes fetched from MinIO (:meth:`MinioMapStore.get_map`).

:func:`extract_geotiff` parses the GeoTIFF affine + CRS from the TIFF tags. It
is the single source of truth for that tag parsing; the webapp and API layers
(``webapp/homography_precision``, ``webapp/point_picker``, ``webapp/line_picker``,
``api/utils/frame_helpers``) import it from here.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.types import Easting, Meters, Northing, Unitless

if TYPE_CHECKING:
    from numpy.typing import NDArray

# GeoTIFF tag keys (GeoKeyDirectoryTag) that carry the projected/geographic CRS.
_PROJECTED_CRS_KEY = 3072
_GEOGRAPHIC_CRS_KEY = 2048


def extract_geotiff(tif: tifffile.TiffFile) -> GeoTiff | None:
    """Extract a :class:`GeoTiff` value object from TIFF tags.

    Reads the GDAL-style georeference encoded in the GeoTIFF tags: either
    ``ModelPixelScaleTag`` + ``ModelTiepointTag`` (north-up rasters) or a full
    ``ModelTransformationTag`` (rotated rasters), plus the CRS EPSG code from
    ``GeoKeyDirectoryTag``.

    Args:
        tif: Open ``tifffile.TiffFile`` object.

    Returns:
        ``GeoTiff`` value object, or ``None`` if georeference tags are absent.
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
                if key_id in (_PROJECTED_CRS_KEY, _GEOGRAPHIC_CRS_KEY):
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


def load_ortho_raster_from_bytes(data: bytes) -> tuple[NDArray[np.uint8], GeoTiff]:
    """Load a georeferenced orthophoto raster from in-memory GeoTIFF bytes.

    Mirrors the ``MinioMapStore.get_map`` -> ``extract_geotiff`` flow so the same
    detector runs against object-store bytes or a local file.

    Args:
        data: Raw GeoTIFF file bytes.

    Returns:
        Tuple of ``(image, geotiff)`` where ``image`` is the raster pixel array
        and ``geotiff`` is the parsed georeference.

    Raises:
        ValueError: If the GeoTIFF carries no georeference tags.
    """
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        geotiff = extract_geotiff(tif)
        image = tif.asarray()
    if geotiff is None:
        raise ValueError("GeoTIFF has no georeference tags (ModelPixelScale/Tiepoint or CRS)")
    return np.asarray(image), geotiff


def load_ortho_raster(path: str | Path) -> tuple[NDArray[np.uint8], GeoTiff]:
    """Load a georeferenced orthophoto raster from a local GeoTIFF file.

    Args:
        path: Path to a ``.tif`` GeoTIFF on disk.

    Returns:
        Tuple of ``(image, geotiff)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the GeoTIFF carries no georeference tags.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {p}")
    return load_ortho_raster_from_bytes(p.read_bytes())


__all__ = ["extract_geotiff", "load_ortho_raster", "load_ortho_raster_from_bytes"]
