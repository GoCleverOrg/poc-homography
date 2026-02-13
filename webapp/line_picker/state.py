"""State management for the line picker Django app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime

import tifffile
import yaml
from PIL import Image


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


@dataclass
class Line:
    """Represents a line defined by two pixel coordinate endpoints.

    Lines are independent entities with their own coordinates, not tied to GCPs.
    This allows lines to be defined anywhere on the map, and camera annotations
    can reference any portion of the line even when endpoints are not visible.

    Attributes:
        line_id: Unique identifier for the line (e.g., "L1", "L2").
        start_x: X coordinate of the start point (map pixels).
        start_y: Y coordinate of the start point (map pixels).
        end_x: X coordinate of the end point (map pixels).
        end_y: Y coordinate of the end point (map pixels).
    """

    line_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    def to_dict(self) -> dict[str, str | float]:
        """Convert line to dictionary format.

        Returns:
            Dictionary with line_id and coordinates.
        """
        return {
            "line_id": self.line_id,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
        }


class LinePickerState:
    """Mutable state for the line picker application."""

    def __init__(
        self,
        image_path: Path,
        map_id: str,
        geotransform: list[float] | None = None,
        crs: str | None = None,
    ) -> None:
        """Initialize state with image file and map identifier.

        Args:
            image_path: Path to the map image file (PNG, TIFF, etc.).
            map_id: Map identifier for tagging saved line files.
            geotransform: Optional 6-parameter geotransform [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height].
            crs: Optional CRS string (e.g., "EPSG:25830").
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = map_id
        self.lines: list[Line] = []

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

    def get_next_id(self) -> str:
        """Get the next auto-incremented line ID.

        Returns:
            Next ID string (e.g., "L1" if no lines exist, "L5" if L1-L4 exist).
        """
        # Find max number for L prefix
        max_num = 0
        for line in self.lines:
            if line.line_id.startswith("L"):
                try:
                    num = int(line.line_id[1:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"L{max_num + 1}"

    def add_line(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        line_id: str | None = None,
    ) -> str:
        """Add a new line defined by two pixel coordinate endpoints.

        Args:
            start_x: X coordinate of the start point (map pixels).
            start_y: Y coordinate of the start point (map pixels).
            end_x: X coordinate of the end point (map pixels).
            end_y: Y coordinate of the end point (map pixels).
            line_id: Optional custom line ID. If None, auto-generates (L1, L2, etc.).

        Returns:
            Line ID (generated or provided).

        Raises:
            ValueError: If start and end points are the same.
        """
        # Validate different endpoints
        if start_x == end_x and start_y == end_y:
            raise ValueError("Start and end points must be different")

        # Generate ID if not provided
        if line_id is None:
            line_id = self.get_next_id()

        # Create and add line
        line = Line(
            line_id=line_id,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )
        self.lines.append(line)

        return line_id

    def delete_line(self, line_id: str) -> None:
        """Delete a line by ID.

        Args:
            line_id: Line ID to delete.

        Raises:
            KeyError: If line ID is not found.
        """
        for i, line in enumerate(self.lines):
            if line.line_id == line_id:
                self.lines.pop(i)
                return
        raise KeyError(f"Line not found: {line_id}")

    def get_line(self, line_id: str) -> Line | None:
        """Get a line by ID.

        Args:
            line_id: Line ID to retrieve.

        Returns:
            Line object if found, None otherwise.
        """
        for line in self.lines:
            if line.line_id == line_id:
                return line
        return None

    def save_lines(self, path: Path) -> None:
        """Save lines to YAML file.

        Args:
            path: Path to the output YAML file.
        """
        data = {
            "map_id": self.map_id,
            "lines": [line.to_dict() for line in self.lines],
        }
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8"
        )

    def load_lines(self, path: Path) -> None:
        """Load lines from YAML file.

        Args:
            path: Path to the input YAML file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
            KeyError: If required keys are missing.
            ValueError: If YAML content is empty or map_id doesn't match.
        """
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if data is None:
            raise ValueError("YAML content is empty")

        # Validate map_id matches
        file_map_id = data.get("map_id")
        if file_map_id != self.map_id:
            raise ValueError(f"Map ID mismatch: expected {self.map_id}, got {file_map_id}")

        # Load lines with pixel coordinates
        self.lines = []
        for line_data in data.get("lines", []):
            line = Line(
                line_id=line_data["line_id"],
                start_x=float(line_data["start_x"]),
                start_y=float(line_data["start_y"]),
                end_x=float(line_data["end_x"]),
                end_y=float(line_data["end_y"]),
            )
            self.lines.append(line)


# Module-level state
_state: LinePickerState | None = None


def initialize_state(
    image_path: Path,
    map_id: str,
    geotransform: list[float] | None = None,
    crs: str | None = None,
) -> None:
    """Initialize the module-level state.

    Args:
        image_path: Path to the map image file (PNG, TIFF, etc.).
        map_id: Map identifier for tagging saved line files.
        geotransform: Optional 6-parameter geotransform.
        crs: Optional CRS string (e.g., "EPSG:25830").
    """
    global _state

    _state = LinePickerState(
        image_path,
        map_id,
        geotransform=geotransform,
        crs=crs,
    )


def get_state() -> LinePickerState:
    """Get the current application state.

    Returns:
        Current LinePickerState instance.

    Raises:
        RuntimeError: If state has not been initialized.
    """
    if _state is None:
        raise RuntimeError("Application not initialized. Call initialize_state() first.")
    return _state


# ---------------------------------------------------------------------------
# Repository adapter functions (bridge legacy Line <-> DDD repos)
# ---------------------------------------------------------------------------


def from_line_repo(data_dir: Path, map_id: str) -> list[Line]:
    """Load lines from the DDD ``RepoYamlLine`` repository.

    Args:
        data_dir: Directory containing per-Line YAML files.
        map_id: Map identifier to filter lines by.

    Returns:
        List of legacy Line objects for the given map.
    """
    from poc_homography.infrastructure.repositories import RepoYamlLine

    repo = RepoYamlLine(data_dir)
    all_lines = repo.get_all()

    return [
        Line(
            line_id=dl.name,
            start_x=float(dl.start.x),
            start_y=float(dl.start.y),
            end_x=float(dl.end.x),
            end_y=float(dl.end.y),
        )
        for dl in all_lines
        if dl.map_id == map_id
    ]


def save_to_line_repo(lines: list[Line], map_id: str, data_dir: Path) -> None:
    """Save legacy Line objects to the DDD ``RepoYamlLine`` repository.

    Each line is converted to a domain ``Line`` entity and persisted
    as an individual YAML file.

    Args:
        lines: Legacy Line objects to persist.
        map_id: Map identifier for the lines.
        data_dir: Directory for per-Line YAML files.
    """
    from poc_homography.domain.entities.line import Line as DomainLine
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.infrastructure.repositories import RepoYamlLine

    repo = RepoYamlLine(data_dir)
    for line in lines:
        domain_line = DomainLine(
            name=line.line_id,
            map_id=map_id,
            start=PixelPoint.create(line.start_x, line.start_y),
            end=PixelPoint.create(line.end_x, line.end_y),
        )
        repo.save(domain_line)


def list_line_map_ids(data_dir: Path) -> list[str]:
    """Return sorted unique map IDs found in the line repository.

    Args:
        data_dir: Directory containing per-Line YAML files.

    Returns:
        Sorted list of unique map_id strings.
    """
    from poc_homography.infrastructure.repositories import RepoYamlLine

    repo = RepoYamlLine(data_dir)
    all_lines = repo.get_all()
    return sorted({line.map_id for line in all_lines})
