"""Tenant entity representing a deployment site."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, eq=False)
class Tenant:
    """A deployment site (e.g., a port terminal or logistics yard).

    A Tenant is the top-level organizational unit that owns Maps and Cameras.
    Each Tenant represents a physical geographic location where the system
    is deployed.

    Attributes:
        id: Unique identifier for the tenant (e.g., "valte").
        name: Human-readable name (e.g., "Valte").
        description: Brief description of the deployment site.
        location_lat: Latitude in DMS format (e.g., "39\u00b038'25.72\"N").
        location_lon: Longitude in DMS format (e.g., "0\u00b013'48.63\"W").
    """

    id: str
    name: str
    description: str = ""
    location_lat: str = ""
    location_lon: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
        }
        if self.description:
            data["description"] = self.description
        if self.location_lat or self.location_lon:
            data["location"] = {
                "lat": self.location_lat,
                "lon": self.location_lon,
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tenant:
        """Create Tenant from dictionary."""
        location = data.get("location", {})
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            location_lat=location.get("lat", ""),
            location_lon=location.get("lon", ""),
        )
