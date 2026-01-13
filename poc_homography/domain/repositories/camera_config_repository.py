"""Camera configuration repository interface."""

from typing import Protocol

from poc_homography.domain.entities.camera_config import CameraConfig


class CameraConfigRepository(Protocol):
    """Repository interface for CameraConfig entities.

    Implementations handle the infrastructure concerns of loading and saving
    camera configuration data (registration data that rarely changes).

    Naming convention: Implementations should be suffixed with "Repository"
    (e.g., YamlCameraConfigRepository).
    """

    def get(self, camera_id: str) -> CameraConfig | None:
        """Retrieve a camera configuration by its ID.

        Args:
            camera_id: Unique identifier for the camera (format: "map_id/name").

        Returns:
            The CameraConfig entity if found, None otherwise.
        """
        ...

    def get_by_map(self, map_id: str) -> dict[str, CameraConfig]:
        """Retrieve all camera configurations for a specific map.

        Args:
            map_id: ID of the map to get camera configs for.

        Returns:
            Dictionary mapping camera_id to CameraConfig for all cameras on the map.
        """
        ...

    def save(self, config: CameraConfig) -> None:
        """Save a camera configuration (create or update).

        Args:
            config: The CameraConfig entity to save.
        """
        ...

    def delete(self, camera_id: str) -> bool:
        """Delete a camera configuration by its ID.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the configuration was deleted, False if it didn't exist.
        """
        ...

    def exists(self, camera_id: str) -> bool:
        """Check if a camera configuration exists.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the configuration exists, False otherwise.
        """
        ...
