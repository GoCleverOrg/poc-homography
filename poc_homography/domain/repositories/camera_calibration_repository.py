"""Camera calibration repository interface."""

from typing import Protocol

from poc_homography.domain.entities.camera_calibration import CameraCalibration


class CameraCalibrationRepository(Protocol):
    """Repository interface for CameraCalibration entities.

    Implementations handle the infrastructure concerns of loading and saving
    camera calibration data (data that is refined during calibration workflows).

    Naming convention: Implementations should be suffixed with "Repository"
    (e.g., YamlCameraCalibrationRepository).
    """

    def get(self, camera_id: str) -> CameraCalibration | None:
        """Retrieve a camera calibration by camera ID.

        Args:
            camera_id: Unique identifier for the camera (format: "map_id/name").

        Returns:
            The CameraCalibration entity if found, None otherwise.
        """
        ...

    def save(self, calibration: CameraCalibration) -> None:
        """Save a camera calibration (create or update).

        Args:
            calibration: The CameraCalibration entity to save.
        """
        ...

    def delete(self, camera_id: str) -> bool:
        """Delete a camera calibration by camera ID.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the calibration was deleted, False if it didn't exist.
        """
        ...

    def exists(self, camera_id: str) -> bool:
        """Check if a camera calibration exists.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the calibration exists, False otherwise.
        """
        ...
