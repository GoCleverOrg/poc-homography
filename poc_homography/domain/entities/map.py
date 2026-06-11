"""Map entity representing a georeferenced map image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.photo import Photo


@dataclass(frozen=True, eq=False)
class Map:
    """A georeferenced map image.

    Represents a map image file (PNG) with associated GeoTiff metadata for
    converting between pixel coordinates and geographic coordinates.

    Attributes:
        id: Unique identifier for the map (e.g., "valte").
        tenant_id: ID of the tenant this map belongs to (e.g., "valte").
        photo: The map image with its dimensions.
        geotiff: GeoTiff metadata for coordinate transformation.
        asset_key: Object-storage key (e.g. ``maps/valte.tif``) for the map
            GeoTIFF asset. Endpoint-agnostic; the API derives URLs at serve
            time. ``None`` until the asset is uploaded (see #290).
        asset_url: Optional fully-qualified URL to the map asset. ``None``
            when the asset is resolved from ``asset_key`` at serve time.
    """

    id: str
    tenant_id: str
    photo: Photo
    geotiff: GeoTiff
    asset_key: str | None = None
    asset_url: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "photo": self.photo.to_dict(),
            "geotiff": self.geotiff.to_dict(),
            "asset_key": self.asset_key,
            "asset_url": self.asset_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Map:
        """Create Map from dictionary."""
        from poc_homography.domain.vo.geotiff import GeoTiff
        from poc_homography.domain.vo.photo import Photo

        return cls(
            id=data["id"],
            tenant_id=data.get("tenant_id", ""),
            photo=Photo.from_dict(data["photo"]),
            geotiff=GeoTiff.from_dict(data["geotiff"]),
            asset_key=data.get("asset_key"),
            asset_url=data.get("asset_url"),
        )
