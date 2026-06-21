#!/usr/bin/env python3
"""Offline tests for orthophoto-side line detection (issue #356).

A synthetic georeferenced GeoTIFF is written with ``tifffile`` (geo tags via
``extratags``) so the whole detector runs fully offline — no MinIO, no DVC
assets. The fixture paints vertical "parking marking" stripes on a dark
background; the detector must recover them with both map-pixel and UTM endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import tifffile

from poc_homography.domain.vo.geotiff import GeoTiff
from poc_homography.maps.geotiff_io import load_ortho_raster
from poc_homography.maps.ortho_line_detection import OrthoLineDetector

if TYPE_CHECKING:
    from pathlib import Path

# North-up EPSG:25830 georeference matching the Valte ortho convention.
_ORIGIN_EASTING = 737575.05
_ORIGIN_NORTHING = 4391595.45
_PIXEL_SIZE = 0.15  # meters / pixel
_EPSG = 25830

# GeoTIFF tag codes.
_MODEL_PIXEL_SCALE = 33550
_MODEL_TIEPOINT = 33922
_GEO_KEY_DIRECTORY = 34735
_TIFF_DOUBLE = 12
_TIFF_SHORT = 3


def _geo_extratags() -> list[tuple]:
    """Build north-up GeoTIFF geo-reference extratags for ``tifffile.imwrite``."""
    pixel_scale = (_PIXEL_SIZE, _PIXEL_SIZE, 0.0)
    tiepoint = (0.0, 0.0, 0.0, _ORIGIN_EASTING, _ORIGIN_NORTHING, 0.0)
    # GeoKeyDirectory: header (version, rev, minor, numkeys) + ProjectedCSType key.
    geo_keys = (1, 1, 0, 1, 3072, 0, 1, _EPSG)
    return [
        (_MODEL_PIXEL_SCALE, _TIFF_DOUBLE, len(pixel_scale), pixel_scale, True),
        (_MODEL_TIEPOINT, _TIFF_DOUBLE, len(tiepoint), tiepoint, True),
        (_GEO_KEY_DIRECTORY, _TIFF_SHORT, len(geo_keys), geo_keys, True),
    ]


def _write_fixture_geotiff(path: Path, *, width: int = 256, height: int = 256) -> list[int]:
    """Write a synthetic ortho GeoTIFF with vertical stripes; return their columns."""
    image = np.zeros((height, width), dtype=np.uint8)
    stripe_cols = [60, 128, 196]
    for col in stripe_cols:
        cv2.line(image, (col, 20), (col, height - 20), color=255, thickness=3)

    tifffile.imwrite(str(path), image, extratags=_geo_extratags())
    return stripe_cols


def test_fixture_geotiff_loads_with_georeference(tmp_path: Path) -> None:
    """The synthetic fixture round-trips through the tifffile loader."""
    tif_path = tmp_path / "ortho.tif"
    _write_fixture_geotiff(tif_path)

    image, geotiff = load_ortho_raster(tif_path)

    assert image.shape == (256, 256)
    assert isinstance(geotiff, GeoTiff)
    assert geotiff.crs == f"EPSG:{_EPSG}"
    assert geotiff.geotransform.pixel_width == pytest.approx(_PIXEL_SIZE)
    assert geotiff.geotransform.origin_easting == pytest.approx(_ORIGIN_EASTING)


def test_detects_dominant_vertical_lines(tmp_path: Path) -> None:
    """The detector recovers the painted vertical stripes as dominant lines."""
    tif_path = tmp_path / "ortho.tif"
    _write_fixture_geotiff(tif_path)

    lines = OrthoLineDetector().detect_from_path(tif_path)

    assert lines, "expected at least one detected ortho line"
    # Painted stripes are vertical -> pixel angle near +/-90 degrees.
    vertical = [line for line in lines if abs(abs(line.angle_deg) - 90.0) <= 5.0]
    assert vertical, "expected dominant lines to be vertical (parking markings)"


def test_endpoints_consistent_in_pixel_and_utm(tmp_path: Path) -> None:
    """Every line's UTM endpoints match the GeoTIFF affine applied to its pixels."""
    tif_path = tmp_path / "ortho.tif"
    _write_fixture_geotiff(tif_path)

    _, geotiff = load_ortho_raster(tif_path)
    lines = OrthoLineDetector().detect_from_path(tif_path)
    assert lines

    for line in lines:
        exp_start = geotiff.pixel_to_geo(line.start_pixel[0], line.start_pixel[1])
        exp_end = geotiff.pixel_to_geo(line.end_pixel[0], line.end_pixel[1])

        assert line.start_utm[0] == pytest.approx(float(exp_start[0]), abs=1e-6)
        assert line.start_utm[1] == pytest.approx(float(exp_start[1]), abs=1e-6)
        assert line.end_utm[0] == pytest.approx(float(exp_end[0]), abs=1e-6)
        assert line.end_utm[1] == pytest.approx(float(exp_end[1]), abs=1e-6)

        # UTM endpoints invert back to the originating pixels (<1e-3 px).
        rx, ry = geotiff.geo_to_pixel(line.start_utm[0], line.start_utm[1])
        assert rx == pytest.approx(float(line.start_pixel[0]), abs=1e-3)
        assert ry == pytest.approx(float(line.start_pixel[1]), abs=1e-3)

        # Ground length equals pixel length times the (square) pixel size.
        assert line.length_m == pytest.approx(float(line.length_px) * _PIXEL_SIZE, rel=1e-6)


def test_detect_from_bytes_matches_path(tmp_path: Path) -> None:
    """detect_from_bytes (MinIO-style) yields the same count as detect_from_path."""
    tif_path = tmp_path / "ortho.tif"
    _write_fixture_geotiff(tif_path)

    detector = OrthoLineDetector()
    from_path = detector.detect_from_path(tif_path)
    from_bytes = detector.detect_from_bytes(tif_path.read_bytes())

    assert len(from_bytes) == len(from_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
