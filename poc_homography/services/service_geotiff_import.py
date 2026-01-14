"""Service for importing GeoTIFF files and creating Map entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import rasterio
from PIL import Image

from poc_homography.domain.entities.map import Map
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import Meters, Pixels

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GeoTiffImportResult:
    """Result of importing a GeoTIFF file.

    Attributes:
        map_entity: The created Map entity.
        png_path: Path to the generated PNG file.
    """

    map_entity: Map
    png_path: Path


class ServiceGeoTiffImport:
    """Service for importing GeoTIFF files and creating Map entities.

    This service handles:
    - Reading GeoTIFF metadata (geotransform, CRS)
    - Converting TIFF to PNG format
    - Creating the Map entity with all required data

    Example usage:
        service = GeoTiffImportService()
        result = service.import_geotiff(
            tiff_path=Path("input.tif"),
            output_dir=Path("maps/"),
            map_id="my_map",
        )
        # result.map_entity is ready to be saved via repository
        # result.png_path is the path to the generated PNG
    """

    def import_geotiff(
        self,
        tiff_path: Path,
        output_dir: Path,
        map_id: str,
    ) -> GeoTiffImportResult:
        """Import a GeoTIFF file and create a Map entity.

        Args:
            tiff_path: Path to the source GeoTIFF file.
            output_dir: Directory where the PNG will be created.
            map_id: Unique identifier for the map.

        Returns:
            GeoTiffImportResult containing the Map entity and PNG path.

        Raises:
            FileNotFoundError: If the TIFF file doesn't exist.
            ValueError: If the TIFF lacks required geospatial metadata.
        """
        if not tiff_path.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {tiff_path}")

        # Extract geospatial metadata using rasterio
        geotiff = self._extract_geotiff_metadata(tiff_path)

        # Convert to PNG and get dimensions
        png_path = output_dir / f"{map_id}.png"
        width, height = self._convert_to_png(tiff_path, png_path)

        # Create Photo VO
        photo = Photo(
            path=png_path,
            width=Pixels(width),
            height=Pixels(height),
        )

        # Create Map entity
        map_entity = Map(
            id=map_id,
            photo=photo,
            geotiff=geotiff,
        )

        return GeoTiffImportResult(
            map_entity=map_entity,
            png_path=png_path,
        )

    def _extract_geotiff_metadata(self, tiff_path: Path) -> GeoTiff:
        """Extract geospatial metadata from a GeoTIFF file.

        Args:
            tiff_path: Path to the GeoTIFF file.

        Returns:
            GeoTiff VO with geotransform and CRS.

        Raises:
            ValueError: If the TIFF lacks geotransform or CRS.
        """
        with rasterio.open(tiff_path) as src:
            if src.transform is None:
                raise ValueError(f"GeoTIFF lacks geotransform: {tiff_path}")
            if src.crs is None:
                raise ValueError(f"GeoTIFF lacks CRS: {tiff_path}")

            # Convert rasterio Affine to our GeoTransform
            # Affine: (a, b, c, d, e, f) where:
            #   a = pixel_width, b = row_rotation, c = origin_x
            #   d = col_rotation, e = pixel_height, f = origin_y
            transform = src.transform
            geotransform = GeoTransform(
                origin_easting=transform.c,
                pixel_width=Meters(transform.a),
                row_rotation=transform.b,
                origin_northing=transform.f,
                col_rotation=transform.d,
                pixel_height=Meters(transform.e),
            )

            crs_str = src.crs.to_string()

            return GeoTiff(geotransform=geotransform, crs=crs_str)

    def _convert_to_png(self, tiff_path: Path, png_path: Path) -> tuple[int, int]:
        """Convert a TIFF file to PNG format.

        Args:
            tiff_path: Path to the source TIFF file.
            png_path: Path where the PNG will be saved.

        Returns:
            Tuple of (width, height) in pixels.
        """
        png_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(tiff_path) as img:
            # Convert to RGB if necessary (handles various TIFF modes)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            img.save(png_path, "PNG")
            return img.width, img.height
