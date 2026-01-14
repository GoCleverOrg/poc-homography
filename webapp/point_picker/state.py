"""State management for the point picker Django app."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from PIL import Image

from poc_homography.geotiff_utils import apply_geotransform
from poc_homography.map_points.map_point import MapPoint
from poc_homography.map_points.map_point_registry import MapPointRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Tag abbreviation mapping
TAG_ABBREVIATIONS = {
    "parking_spot": "PS",
    "arrows": "AR",
    "crosswalk": "CW",
    "extra": "EX",
}

ABBREVIATION_TO_TAG = {v: k for k, v in TAG_ABBREVIATIONS.items()}


def _extract_geotransform(tif: tifffile.TiffFile) -> tuple[list[float] | None, str | None]:
    """Extract GeoTIFF geotransform and CRS info from TIFF tags.

    Args:
        tif: Open tifffile TiffFile object.

    Returns:
        Tuple of (geotransform, crs_string).
        geotransform is 6-element list [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
        or None if not available.
    """
    page = tif.pages[0]
    # type: ignore needed because tifffile types are incomplete
    tags = {tag.name: tag for tag in page.tags.values()}  # type: ignore[union-attr]

    geotransform = None
    crs = None

    # Try to extract GeoTIFF metadata
    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        # Common GeoTIFF format: pixel scale + tiepoint
        try:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value

            # GDAL-style geotransform: [originX, pixelWidth, rotationX, originY, rotationY, pixelHeight]
            # tiepoint = [I, J, K, X, Y, Z] where (I,J,K) is pixel coord and (X,Y,Z) is map coord
            origin_x = tiepoint[3] - tiepoint[0] * scale[0]
            origin_y = tiepoint[4] + tiepoint[1] * scale[1]

            geotransform = [
                float(origin_x),  # GT[0]: origin X
                float(scale[0]),  # GT[1]: pixel width
                0.0,  # GT[2]: rotation (typically 0)
                float(origin_y),  # GT[3]: origin Y
                0.0,  # GT[4]: rotation (typically 0)
                -float(scale[1]),  # GT[5]: pixel height (negative for north-up)
            ]
        except (IndexError, TypeError, ValueError):
            pass

    elif "ModelTransformationTag" in tags:
        # Alternative: 4x4 transformation matrix
        try:
            matrix = tags["ModelTransformationTag"].value
            geotransform = [
                float(matrix[3]),  # origin X
                float(matrix[0]),  # pixel width
                float(matrix[1]),  # rotation
                float(matrix[7]),  # origin Y
                float(matrix[4]),  # rotation
                float(matrix[5]),  # pixel height
            ]
        except (IndexError, TypeError, ValueError):
            pass

    # Try to get CRS info from GeoKeyDirectoryTag
    if "GeoKeyDirectoryTag" in tags:
        try:
            geo_keys = tags["GeoKeyDirectoryTag"].value
            # Look for ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
            for i in range(4, len(geo_keys), 4):
                key_id = geo_keys[i]
                if key_id == 3072:  # ProjectedCSTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
                    break
                elif key_id == 2048:  # GeographicTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
        except (IndexError, TypeError, ValueError):
            pass

    return geotransform, crs


class PointPickerState:
    """Mutable state for the point picker application."""

    def __init__(
        self,
        image_path: Path,
        geotransform: list[float] | None = None,
        crs: str | None = None,
    ) -> None:
        """Initialize state with image file.

        Args:
            image_path: Path to the image file (PNG, TIFF, etc.).
            geotransform: Optional 6-parameter geotransform [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height].
            crs: Optional CRS string (e.g., "EPSG:25830").
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = image_path.stem
        self.registry = MapPointRegistry(map_id=self.map_id, points={})

        # Detect file type and load accordingly
        suffix = image_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            # Load TIFF metadata using tifffile
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                # type: ignore needed because tifffile types are incomplete
                self.width: int = page.imagewidth  # type: ignore[union-attr]
                self.height: int = page.imagelength  # type: ignore[union-attr]

                # Extract geotransform and CRS from TIFF if not provided
                if geotransform is None or crs is None:
                    tiff_gt, tiff_crs = _extract_geotransform(tif)
                    if geotransform is None:
                        geotransform = tiff_gt
                    if crs is None:
                        crs = tiff_crs
        else:
            # Load other image formats (PNG, JPG, etc.) using PIL
            with Image.open(image_path) as img:
                self.width = img.width
                self.height = img.height

        self.geotransform = geotransform
        self.crs = crs

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
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

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
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

    def delete_point(self, point_id: str) -> None:
        """Delete a point.

        Args:
            point_id: Point ID to delete.
        """
        if point_id not in self.registry.points:
            raise KeyError(f"Point not found: {point_id}")

        new_points = dict(self.registry.points)
        del new_points[point_id]
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

    def load_registry(self, path: Path) -> None:
        """Load points from YAML/JSON file.

        Args:
            path: Path to the file.
        """
        self.registry = MapPointRegistry.load(path)
        self.map_id = self.registry.map_id

    def save_registry(self, path: Path) -> None:
        """Save points to YAML/JSON file.

        Args:
            path: Path to the file.
        """
        self.registry.save(path)

    def get_geo_coords(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.

        Returns:
            Tuple of (easting, northing) or None if no geotransform.
        """
        if self.geotransform:
            return apply_geotransform(pixel_x, pixel_y, self.geotransform)
        return None


# Module-level state
_state: PointPickerState | None = None


def initialize_state(
    image_path: Path,
    geotransform: list[float] | None = None,
    crs: str | None = None,
) -> None:
    """Initialize the module-level state.

    Args:
        image_path: Path to the image file (PNG, TIFF, etc.).
        geotransform: Optional 6-parameter geotransform.
        crs: Optional CRS string (e.g., "EPSG:25830").
    """
    global _state
    _state = PointPickerState(image_path, geotransform=geotransform, crs=crs)


def get_state() -> PointPickerState:
    """Get the current application state."""
    if _state is None:
        raise RuntimeError("Application not initialized. Call initialize_state() first.")
    return _state


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
