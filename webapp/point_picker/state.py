"""State management for the point picker Django app."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime

import numpy as np
import tifffile
from PIL import Image

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.map_points.gcp_registry import GCPRegistry
from poc_homography.map_points.map_point import MapPoint
from poc_homography.types import Easting, Meters, Northing, Unitless

# Tag abbreviation mapping
TAG_ABBREVIATIONS = {
    "parking_spot": "PS",
    "arrows": "AR",
    "crosswalk": "CW",
    "extra": "EX",
}

ABBREVIATION_TO_TAG = {v: k for k, v in TAG_ABBREVIATIONS.items()}


def _extract_geotiff(tif: tifffile.TiffFile) -> GeoTiff | None:
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

    # Try to extract GeoTIFF metadata
    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        # Common GeoTIFF format: pixel scale + tiepoint
        try:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value

            origin_x = tiepoint[3] - tiepoint[0] * scale[0]
            origin_y = tiepoint[4] + tiepoint[1] * scale[1]

            gt_params = [
                float(origin_x), float(scale[0]), 0.0,
                float(origin_y), 0.0, -float(scale[1]),
            ]
        except (IndexError, TypeError, ValueError):
            pass

    elif "ModelTransformationTag" in tags:
        # Alternative: 4x4 transformation matrix
        try:
            matrix = tags["ModelTransformationTag"].value
            gt_params = [
                float(matrix[3]), float(matrix[0]), float(matrix[1]),
                float(matrix[7]), float(matrix[4]), float(matrix[5]),
            ]
        except (IndexError, TypeError, ValueError):
            pass

    # Try to get CRS info from GeoKeyDirectoryTag
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


class PointPickerState:
    """Mutable state for the point picker application."""

    def __init__(
        self,
        image_path: Path,
        geotiff: GeoTiff | None = None,
    ) -> None:
        """Initialize state with image file.

        Args:
            image_path: Path to the image file (PNG, TIFF, etc.).
            geotiff: Optional GeoTiff VO with geotransform and CRS.
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = image_path.stem
        self.registry = GCPRegistry(map_id=self.map_id, points={})

        # Detect file type and load accordingly
        suffix = image_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            # Load TIFF metadata using tifffile
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                # type: ignore needed because tifffile types are incomplete
                self.width: int = page.imagewidth  # type: ignore[union-attr]
                self.height: int = page.imagelength  # type: ignore[union-attr]

                # Extract GeoTiff VO from TIFF if not provided
                if geotiff is None:
                    geotiff = _extract_geotiff(tif)
        else:
            # Load other image formats (PNG, JPG, etc.) using PIL
            with Image.open(image_path) as img:
                self.width = img.width
                self.height = img.height

        self.geotiff = geotiff

    def get_next_id(self, tag: str) -> str:
        """Get the next auto-incremented ID for a tag.

        Args:
            tag: Tag name (parking_spot, arrows, crosswalk, extra).

        Returns:
            Next ID string (e.g., "PS5" if PS1-PS4 exist).
        """
        abbrev = TAG_ABBREVIATIONS.get(tag)
        if not abbrev:
            raise ValueError(f"Unknown tag: {tag}")

        # Find max number for this prefix
        max_num = 0
        for point_id in self.registry.points:
            if point_id.startswith(abbrev):
                try:
                    num = int(point_id[len(abbrev) :])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"{abbrev}{max_num + 1}"

    def add_point(
        self, tag: str, pixel_x: float, pixel_y: float, point_id: str | None = None
    ) -> str:
        """Add a new point with auto-generated or custom ID.

        Args:
            tag: Tag category.
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.
            point_id: Optional custom ID. If None, auto-generates based on tag.

        Returns:
            Point ID (generated or provided).
        """
        if point_id is None:
            point_id = self.get_next_id(tag)
        point = MapPoint(pixel_x=pixel_x, pixel_y=pixel_y)

        # Create new registry with added point (immutable pattern)
        new_points = dict(self.registry.points)
        new_points[point_id] = point
        self.registry = GCPRegistry(map_id=self.map_id, points=new_points)

        return point_id

    def update_point(self, point_id: str, pixel_x: float, pixel_y: float) -> None:
        """Update point coordinates (for drag operations).

        Args:
            point_id: Point ID to update.
            pixel_x: New X pixel coordinate.
            pixel_y: New Y pixel coordinate.
        """
        if point_id not in self.registry.points:
            raise KeyError(f"Point not found: {point_id}")

        new_points = dict(self.registry.points)
        new_points[point_id] = MapPoint(pixel_x=pixel_x, pixel_y=pixel_y)
        self.registry = GCPRegistry(map_id=self.map_id, points=new_points)

    def delete_point(self, point_id: str) -> None:
        """Delete a point.

        Args:
            point_id: Point ID to delete.
        """
        if point_id not in self.registry.points:
            raise KeyError(f"Point not found: {point_id}")

        new_points = dict(self.registry.points)
        del new_points[point_id]
        self.registry = GCPRegistry(map_id=self.map_id, points=new_points)

    def load_from_repo(self, data_dir: Path, map_id: str) -> None:
        """Load points from the DDD GCP repository.

        Args:
            data_dir: Directory containing per-GCP YAML files.
            map_id: Map identifier to load.
        """
        from poc_homography.map_points.gcp_registry import from_gcp_repo

        self.registry = from_gcp_repo(data_dir, map_id)
        self.map_id = self.registry.map_id

    def save_to_repo(self, data_dir: Path) -> None:
        """Save points to the DDD GCP repository.

        Args:
            data_dir: Directory for per-GCP YAML files.
        """
        from poc_homography.map_points.gcp_registry import save_to_gcp_repo

        save_to_gcp_repo(self.registry, data_dir)

    def get_geo_coords(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.

        Returns:
            Tuple of (easting, northing) or None if no geotransform.
        """
        if self.geotiff:
            return self.geotiff.pixel_to_geo(pixel_x, pixel_y)
        return None


# Module-level state
_state: PointPickerState | None = None


def initialize_state(
    image_path: Path,
    geotiff: GeoTiff | None = None,
) -> None:
    """Initialize the module-level state.

    Args:
        image_path: Path to the image file (PNG, TIFF, etc.).
        geotiff: Optional GeoTiff VO with geotransform and CRS.
    """
    global _state
    _state = PointPickerState(image_path, geotiff=geotiff)


def get_state() -> PointPickerState:
    """Get the current application state."""
    if _state is None:
        raise RuntimeError("Application not initialized. Call initialize_state() first.")
    return _state


def get_tag_from_id(point_id: str) -> str:
    """Extract tag from point ID prefix.

    Args:
        point_id: Point ID (e.g., "PS1", "AR2").

    Returns:
        Tag name (e.g., "parking_spot", "arrows").
    """
    for abbrev, tag in ABBREVIATION_TO_TAG.items():
        if point_id.startswith(abbrev):
            return tag
    return "extra"
