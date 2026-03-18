"""State management for the line picker Django app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

import tifffile
from homography_web.frame_utils import (
    LINES_DIR,
    register_invalidation_callback,
    resolve_map_for_tenant,
)
from PIL import Image

from poc_homography.infrastructure.repositories import RepoYamlLine

if TYPE_CHECKING:
    from poc_homography.domain.vo.geotiff import GeoTiff


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
        geotiff: GeoTiff | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Initialize state with image file and map identifier.

        Args:
            image_path: Path to the map image file (PNG, TIFF, etc.).
            map_id: Map identifier for tagging saved line files.
            geotiff: Optional GeoTiff VO with geotransform and CRS.
            width: Image width in pixels (avoids file I/O when provided).
            height: Image height in pixels (avoids file I/O when provided).
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = map_id
        self.lines: list[Line] = []

        # Detect file type and load accordingly
        suffix = image_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            if width is not None and height is not None:
                self.width: int = width
                self.height: int = height
                if geotiff is None:
                    from homography_web.frame_utils import extract_geotiff

                    with tifffile.TiffFile(image_path) as tif:
                        geotiff = extract_geotiff(tif)
            else:
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    # type: ignore needed because tifffile types are incomplete
                    self.width = page.imagewidth  # type: ignore[union-attr]
                    self.height = page.imagelength  # type: ignore[union-attr]

                    if geotiff is None:
                        from homography_web.frame_utils import extract_geotiff

                        geotiff = extract_geotiff(tif)
        else:
            if width is not None and height is not None:
                self.width = width
                self.height = height
            else:
                with Image.open(image_path) as img:
                    self.width = img.width
                    self.height = img.height

        self.geotiff = geotiff

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


# Per-tenant state (lazily initialized)
_states: dict[str, LinePickerState] = {}


def get_state(tenant_id: str) -> LinePickerState:
    """Get or lazily initialize state for a tenant.

    Args:
        tenant_id: Tenant identifier.

    Returns:
        LinePickerState for the given tenant.

    Raises:
        RuntimeError: If no map is configured for the tenant or map file not found.
    """
    if tenant_id in _states:
        return _states[tenant_id]

    map_entity, map_file = resolve_map_for_tenant(tenant_id)

    state = LinePickerState(
        map_file,
        map_entity.id,
        width=int(map_entity.photo.width),
        height=int(map_entity.photo.height),
    )
    state.lines = from_line_repo(LINES_DIR, map_entity.id)
    _states[tenant_id] = state
    return state


# ---------------------------------------------------------------------------
# Repository adapter functions (bridge legacy Line <-> DDD repos)
# ---------------------------------------------------------------------------

_line_repo_cache: dict[str, RepoYamlLine] = {}


def _get_line_repo(data_dir: Path) -> RepoYamlLine:
    """Return a cached RepoYamlLine for *data_dir*."""
    key = str(data_dir)
    if key not in _line_repo_cache:
        _line_repo_cache[key] = RepoYamlLine(data_dir)
    return _line_repo_cache[key]


def from_line_repo(data_dir: Path, map_id: str) -> list[Line]:
    """Load lines from the DDD ``RepoYamlLine`` repository.

    Args:
        data_dir: Directory containing per-Line YAML files.
        map_id: Map identifier to filter lines by.

    Returns:
        List of legacy Line objects for the given map.
    """
    repo = _get_line_repo(data_dir)

    return [
        Line(
            line_id=dl.name,
            start_x=float(dl.start.x),
            start_y=float(dl.start.y),
            end_x=float(dl.end.x),
            end_y=float(dl.end.y),
        )
        for dl in repo.get_by_map_id(map_id)
    ]


def save_single_line_to_repo(line: Line, map_id: str, data_dir: Path) -> None:
    """Persist a single legacy Line to the DDD ``RepoYamlLine`` repository.

    Args:
        line: Legacy Line object to persist.
        map_id: Map identifier for the line.
        data_dir: Directory for per-Line YAML files.
    """
    from poc_homography.domain.entities.line import Line as DomainLine
    from poc_homography.domain.vo.pixel_point import PixelPoint

    repo = _get_line_repo(data_dir)
    domain_line = DomainLine(
        name=line.line_id,
        map_id=map_id,
        start=PixelPoint.create(line.start_x, line.start_y),
        end=PixelPoint.create(line.end_x, line.end_y),
    )
    repo.save(domain_line)


def save_to_line_repo(lines: list[Line], map_id: str, data_dir: Path) -> None:
    """Save legacy Line objects to the DDD ``RepoYamlLine`` repository.

    Each line is converted to a domain ``Line`` entity and persisted
    as an individual YAML file.

    Args:
        lines: Legacy Line objects to persist.
        map_id: Map identifier for the lines.
        data_dir: Directory for per-Line YAML files.
    """
    for line in lines:
        save_single_line_to_repo(line, map_id, data_dir)


def delete_line_from_repo(line_id: str, map_id: str, data_dir: Path) -> None:
    """Delete a line from the DDD ``RepoYamlLine`` repository.

    Args:
        line_id: Line identifier (e.g. "L1").
        map_id: Map identifier for constructing the entity ID.
        data_dir: Directory for per-Line YAML files.
    """
    repo = _get_line_repo(data_dir)
    entity_id = f"{map_id}/{line_id}"
    repo.delete(entity_id)


def list_line_map_ids(data_dir: Path) -> list[str]:
    """Return sorted unique map IDs found in the line repository.

    Args:
        data_dir: Directory containing per-Line YAML files.

    Returns:
        Sorted list of unique map_id strings.
    """
    repo = _get_line_repo(data_dir)
    all_lines = repo.get_all()
    return sorted({line.map_id for line in all_lines})


register_invalidation_callback(_states.clear)
register_invalidation_callback(_line_repo_cache.clear)
