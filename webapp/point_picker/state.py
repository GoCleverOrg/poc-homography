"""State management for the point picker Django app."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

import tifffile
from homography_web.frame_utils import (
    GCPS_DIR,
    extract_geotiff,
    register_invalidation_callback,
    resolve_map_for_tenant,
)
from PIL import Image

from poc_homography.domain.vo.geotiff import GeoTiff  # noqa: TC001 - used at runtime
from poc_homography.infrastructure.repositories import RepoYamlGroundControlPoint
from poc_homography.map_points.gcp_registry import GCPRegistry
from poc_homography.map_points.map_point import MapPoint

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.domain.entities.map import Map

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


# Per-(tenant, map) state (lazily initialized)
_states: dict[tuple[str, str], PointPickerState] = {}


def get_state(
    tenant_id: str,
    session: Session | None = None,
    map_id: str | None = None,
) -> PointPickerState:
    """Get or lazily initialize state for a tenant's map.

    Args:
        tenant_id: Tenant identifier.
        session: Optional SQLAlchemy session. When provided, GCPs are loaded
            from PostgreSQL instead of YAML files.
        map_id: Optional specific map identifier. When ``None`` the tenant's
            first configured map is resolved (backward-compatible behavior).

    Returns:
        PointPickerState for the given (tenant, map).

    Raises:
        RuntimeError: If no map is configured for the tenant or map file not found.
        LookupError: If *map_id* is provided but no such map exists.
    """
    # Fast path: an explicit, previously resolved (tenant, map) entry.
    if map_id is not None and (tenant_id, map_id) in _states:
        return _states[(tenant_id, map_id)]

    # Resolve the concrete map (a ``map_id=None`` call picks the tenant's
    # default map), then key the cache by its real id — so the default path
    # reuses the same cached state as an explicit call for that map.
    map_entity, map_file = _resolve_map(tenant_id, session, map_id)
    key = (tenant_id, map_entity.id)
    if key in _states:
        return _states[key]

    state = PointPickerState(
        map_file,
        width=int(map_entity.photo.width),
        height=int(map_entity.photo.height),
    )
    # Anchor the state to the resolved map id rather than the image filename
    # stem, so persistence (add/update/delete) always targets the right map.
    state.map_id = map_entity.id

    if session is not None:
        from poc_homography.map_points.gcp_registry import from_gcp_repo_pg

        state.registry = from_gcp_repo_pg(session, map_entity.id)
    elif GCPS_DIR.exists():
        from poc_homography.map_points.gcp_registry import from_gcp_repo

        state.registry = from_gcp_repo(GCPS_DIR, map_entity.id)

    _states[key] = state
    return state


def _resolve_map(
    tenant_id: str,
    session: Session | None,
    map_id: str | None,
) -> tuple[Map, Path]:
    """Resolve the (Map, image-file) pair for *tenant_id*/*map_id*."""
    if map_id is not None and session is not None:
        from api.utils.frame_helpers import resolve_map as resolve_pg_by_id

        return resolve_pg_by_id(map_id, session)
    if map_id is not None:
        from homography_web.frame_utils import DATA_MAPS_DIR, get_map_repo

        map_entity = get_map_repo().get(map_id)
        if map_entity is None:
            raise LookupError(f"Map not found: {map_id}")
        return map_entity, DATA_MAPS_DIR / map_entity.photo.path
    if session is not None:
        from api.utils.frame_helpers import resolve_map_for_tenant as resolve_pg

        return resolve_pg(tenant_id, session)
    return resolve_map_for_tenant(tenant_id)


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


# ---------------------------------------------------------------------------
# Repository adapter functions (bridge legacy MapPoint <-> DDD repos)
# ---------------------------------------------------------------------------

_gcp_repo_cache: dict[str, RepoYamlGroundControlPoint] = {}


def _get_gcp_repo(data_dir: Path) -> RepoYamlGroundControlPoint:
    """Return a cached RepoYamlGroundControlPoint for *data_dir*."""
    key = str(data_dir)
    if key not in _gcp_repo_cache:
        _gcp_repo_cache[key] = RepoYamlGroundControlPoint(data_dir)
    return _gcp_repo_cache[key]


def save_gcp_to_repo(
    point_id: str,
    pixel_x: float,
    pixel_y: float,
    map_id: str,
    data_dir: Path,
) -> None:
    """Persist a single GCP to the DDD ``RepoYamlGroundControlPoint`` repository.

    Accepts raw coordinates so callers can write to disk *before* mutating
    in-memory state, keeping the two in sync even when the write fails.

    Args:
        point_id: Point identifier (e.g. "PS1").
        pixel_x: X pixel coordinate.
        pixel_y: Y pixel coordinate.
        map_id: Map identifier for the GCP.
        data_dir: Directory for per-GCP YAML files.
    """
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.map_point import MapPoint as DomainMapPoint
    from poc_homography.domain.vo.pixel_point import PixelPoint

    repo = _get_gcp_repo(data_dir)
    gcp = GroundControlPoint(
        name=point_id,
        map_point=DomainMapPoint(
            map_id=map_id,
            pixel_point=PixelPoint.create(pixel_x, pixel_y),
        ),
    )
    repo.save(gcp)


def delete_gcp_from_repo(point_id: str, map_id: str, data_dir: Path) -> None:
    """Delete a GCP from the DDD ``RepoYamlGroundControlPoint`` repository.

    Args:
        point_id: Point identifier (e.g. "PS1").
        map_id: Map identifier for constructing the entity ID.
        data_dir: Directory for per-GCP YAML files.
    """
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.map_point import MapPoint as DomainMapPoint
    from poc_homography.domain.vo.pixel_point import PixelPoint

    repo = _get_gcp_repo(data_dir)
    # Use GroundControlPoint.id to derive the entity ID rather than duplicating the format.
    _zero = PixelPoint.create(0, 0)
    entity_id = GroundControlPoint(
        name=point_id, map_point=DomainMapPoint(map_id=map_id, pixel_point=_zero)
    ).id
    repo.delete(entity_id)


def save_gcp_to_repo_pg(
    point_id: str,
    pixel_x: float,
    pixel_y: float,
    map_id: str,
    session: Session,
) -> None:
    """Persist a single GCP via ``RepoPostgresGroundControlPoint``."""
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.map_point import MapPoint as DomainMapPoint
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.infrastructure.repositories import RepoPostgresGroundControlPoint

    repo = RepoPostgresGroundControlPoint(session)
    gcp = GroundControlPoint(
        name=point_id,
        map_point=DomainMapPoint(
            map_id=map_id,
            pixel_point=PixelPoint.create(pixel_x, pixel_y),
        ),
    )
    repo.save(gcp)


def delete_gcp_from_repo_pg(point_id: str, map_id: str, session: Session) -> None:
    """Delete a GCP via ``RepoPostgresGroundControlPoint``."""
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.map_point import MapPoint as DomainMapPoint
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.infrastructure.repositories import RepoPostgresGroundControlPoint

    repo = RepoPostgresGroundControlPoint(session)
    _zero = PixelPoint.create(0, 0)
    entity_id = GroundControlPoint(
        name=point_id, map_point=DomainMapPoint(map_id=map_id, pixel_point=_zero)
    ).id
    repo.delete(entity_id)


register_invalidation_callback(_states.clear)
register_invalidation_callback(_gcp_repo_cache.clear)
