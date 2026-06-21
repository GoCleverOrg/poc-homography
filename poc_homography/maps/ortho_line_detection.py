"""Orthophoto-side line / parking-marking detection.

Detects straight-line segments (parking markings, lane lines, kerbs) directly on
a georeferenced orthophoto raster and returns each segment in BOTH map-pixel and
UTM world coordinates.

The detection pipeline reuses the OpenCV shape from
:class:`poc_homography.calibration.lens_distortion.line_detection.LineDetector`
(Canny + probabilistic Hough + parallel clustering). This module is the
ortho-side counterpart to that camera-side detector: it composes ``LineDetector``
and projects each detected pixel endpoint to UTM via
:meth:`poc_homography.domain.vo.geotiff.GeoTiff.pixel_to_geo`.

It runs fully offline against a saved GeoTIFF (local path) and, equivalently,
against GeoTIFF bytes fetched from MinIO (:meth:`MinioMapStore.get_map`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.calibration.lens_distortion.line_detection import (
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.domain.vo.ortho_line import OrthoLine
from poc_homography.maps.geotiff_io import load_ortho_raster, load_ortho_raster_from_bytes
from poc_homography.types import PixelsFloat

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from poc_homography.domain.vo.geotiff import GeoTiff


class OrthoLineDetector:
    """Detects line segments on a georeferenced orthophoto, in pixel + UTM frames.

    Wraps :class:`LineDetector` and, for every detected pixel-space segment,
    projects both endpoints to UTM using the GeoTIFF affine, yielding typed
    :class:`OrthoLine` results.
    """

    def __init__(self, config: LineDetectionConfig | None = None) -> None:
        """Initialize the detector.

        Args:
            config: Line-detection configuration passed straight through to the
                underlying :class:`LineDetector`. Uses its defaults if ``None``.
        """
        self._detector = LineDetector(config)

    def detect(self, image: NDArray[np.uint8], geotiff: GeoTiff) -> list[OrthoLine]:
        """Detect ortho lines on an in-memory raster.

        Args:
            image: Orthophoto raster pixel array (grayscale or BGR/RGB).
            geotiff: Georeference for ``image`` (pixel <-> UTM affine + CRS).

        Returns:
            List of :class:`OrthoLine`, ordered by detector confidence (highest
            first), each with map-pixel and UTM endpoints.
        """
        candidates = self._detector.detect(image)
        ortho_lines: list[OrthoLine] = []
        for c in candidates:
            start_utm = geotiff.pixel_to_geo(PixelsFloat(c.start[0]), PixelsFloat(c.start[1]))
            end_utm = geotiff.pixel_to_geo(PixelsFloat(c.end[0]), PixelsFloat(c.end[1]))
            ortho_lines.append(
                OrthoLine.from_endpoints(
                    start_pixel=c.start,
                    end_pixel=c.end,
                    start_utm=start_utm,
                    end_utm=end_utm,
                    confidence=c.confidence,
                    cluster_id=c.cluster_id,
                )
            )
        return ortho_lines

    def detect_from_path(self, path: str | Path) -> list[OrthoLine]:
        """Detect ortho lines from a local GeoTIFF file (fully offline).

        Args:
            path: Path to a ``.tif`` GeoTIFF on disk.

        Returns:
            List of detected :class:`OrthoLine`.
        """
        image, geotiff = load_ortho_raster(path)
        return self.detect(image, geotiff)

    def detect_from_bytes(self, data: bytes) -> list[OrthoLine]:
        """Detect ortho lines from in-memory GeoTIFF bytes.

        Mirrors the ``MinioMapStore.get_map`` flow: hand the returned bytes
        straight to this method.

        Args:
            data: Raw GeoTIFF file bytes.

        Returns:
            List of detected :class:`OrthoLine`.
        """
        image, geotiff = load_ortho_raster_from_bytes(data)
        return self.detect(image, geotiff)


__all__ = ["OrthoLineDetector"]
