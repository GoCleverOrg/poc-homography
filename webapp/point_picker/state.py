"""State management for the point picker Django app."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime

import tifffile
from homography_web.frame_utils import extract_geotiff
from PIL import Image

from poc_homography.domain.vo.geotiff import GeoTiff
from poc_homography.map_points.gcp_registry import GCPRegistry
from poc_homography.map_points.map_point import MapPoint

# Tag abbreviation mapping
TAG_ABBREVIATIONS = {
    "parking_spot": "PS",
    "arrows": "AR",
    "crosswalk": "CW",
    "extra": "EX",
}

ABBREVIATION_TO_TAG = {v: k for k, v in TAG_ABBREVIATIONS.items()}


class PointPickerState:
    """Mutable state for the point picker application."""

    def __init__(
        self,
        image_path: Path,
        geotiff: GeoTiff | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Initialize state with image file.

        Args:
            image_path: Path to the image file (PNG, TIFF, etc.).
            geotiff: Optional GeoTiff VO with geotransform and CRS.
            width: Image width in pixels (avoids file I/O when provided).
            height: Image height in pixels (avoids file I/O when provided).
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = image_path.stem
        self.registry = GCPRegistry(map_id=self.map_id, points={})

        # Detect file type and load accordingly
        suffix = image_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            # Use provided dimensions or read from TIFF
            if width is not None and height is not None:
                self.width: int = width
                self.height: int = height
                # Still need to extract GeoTiff VO from TIFF if not provided
                if geotiff is None:
                    with tifffile.TiffFile(image_path) as tif:
                        geotiff = extract_geotiff(tif)
            else:
                # Load TIFF metadata using tifffile
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    # type: ignore needed because tifffile types are incomplete
                    self.width = page.imagewidth  # type: ignore[union-attr]
                    self.height = page.imagelength  # type: ignore[union-attr]

                    # Extract GeoTiff VO from TIFF if not provided
                    if geotiff is None:
                        geotiff = extract_geotiff(tif)
        else:
            if width is not None and height is not None:
                self.width = width
                self.height = height
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
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Initialize the module-level state.

    Args:
        image_path: Path to the image file (PNG, TIFF, etc.).
        geotiff: Optional GeoTiff VO with geotransform and CRS.
        width: Image width in pixels (avoids file I/O when provided).
        height: Image height in pixels (avoids file I/O when provided).
    """
    global _state
    _state = PointPickerState(image_path, geotiff=geotiff, width=width, height=height)


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
