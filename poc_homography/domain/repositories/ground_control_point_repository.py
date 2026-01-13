"""Ground control point repository interface."""

from typing import Protocol

from poc_homography.domain.entities.ground_control_point import GroundControlPoint


class GroundControlPointRepository(Protocol):
    """Repository interface for GroundControlPoint entities.

    Provides access to ground control points, supporting queries by map
    and individual GCP operations.

    Naming convention: Implementations should be suffixed with "Repository"
    (e.g., YamlGroundControlPointRepository, FileGroundControlPointRepository).
    """

    def get_by_map(self, map_id: str) -> dict[str, GroundControlPoint]:
        """Retrieve all GCPs for a specific map.

        Args:
            map_id: Identifier for the map.

        Returns:
            Dictionary mapping GCP ID to GroundControlPoint entity.
            Empty dict if no GCPs exist for the map.
        """
        ...

    def save(self, gcp: GroundControlPoint) -> None:
        """Save a GCP (create or update).

        Args:
            gcp: The GroundControlPoint to save.
        """
        ...

    def delete(self, gcp_id: str) -> bool:
        """Delete a GCP by its ID.

        Args:
            gcp_id: Unique identifier for the GCP (format: "map_id/name").

        Returns:
            True if the GCP was deleted, False if it didn't exist.
        """
        ...

    def exists(self, gcp_id: str) -> bool:
        """Check if a GCP exists.

        Args:
            gcp_id: Unique identifier for the GCP (format: "map_id/name").

        Returns:
            True if the GCP exists, False otherwise.
        """
        ...
