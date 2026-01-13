"""Map repository interface."""

from typing import Protocol

from poc_homography.domain.entities.map import Map


class MapRepository(Protocol):
    """Repository interface for Map entities.

    Implementations handle the infrastructure concerns of loading map data,
    including reading image files for dimensions and parsing GeoTiff metadata.

    Naming convention: Implementations should be suffixed with "Repository"
    (e.g., FileMapRepository, YamlMapRepository).
    """

    def get(self, map_id: str) -> Map | None:
        """Retrieve a map by its ID.

        Args:
            map_id: Unique identifier for the map.

        Returns:
            The Map entity if found, None otherwise.
        """
        ...

    def get_all(self) -> list[Map]:
        """Retrieve all available maps.

        Returns:
            List of all Map entities.
        """
        ...

    def exists(self, map_id: str) -> bool:
        """Check if a map exists.

        Args:
            map_id: Unique identifier for the map.

        Returns:
            True if the map exists, False otherwise.
        """
        ...
