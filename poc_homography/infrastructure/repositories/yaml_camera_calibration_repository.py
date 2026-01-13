"""YAML-based CameraCalibration repository implementation."""

from pathlib import Path

import yaml

from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.orientation import Orientation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.types import Degrees, Meters, PixelsFloat, Unitless


class YamlCameraCalibrationRepository:
    """Repository that loads CameraCalibration entities from YAML files.

    Expected YAML format:
        camera_id: valte/Valte
        position:
          x: 1234.5
          y: 5678.9
        height: 4.71
        base_orientation:
          yaw: 51.7
          pitch: -0.25
          roll: 0.0
        distortion:
          k1: -0.341052
          k2: 0.787571
          p1: 0.0
          p2: 0.0

    Files are stored with names like "valte__Valte.yaml" where
    the camera_id "valte/Valte" has "/" replaced with "__".
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing calibration YAML files.
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CameraCalibration] = {}

    def get(self, camera_id: str) -> CameraCalibration | None:
        """Retrieve a camera calibration by camera ID.

        Args:
            camera_id: Unique identifier for the camera (format: "map_id/name").

        Returns:
            The CameraCalibration entity if found, None otherwise.
        """
        if camera_id in self._cache:
            return self._cache[camera_id]

        yaml_path = self._id_to_path(camera_id)
        if not yaml_path.exists():
            return None

        calibration = self._load_calibration(yaml_path)
        if calibration:
            self._cache[camera_id] = calibration
        return calibration

    def save(self, calibration: CameraCalibration) -> None:
        """Save a camera calibration (create or update).

        Args:
            calibration: The CameraCalibration entity to save.
        """
        yaml_path = self._id_to_path(calibration.camera_id)

        data = {
            "camera_id": calibration.camera_id,
            "position": {
                "x": float(calibration.position.x),
                "y": float(calibration.position.y),
            },
            "height": float(calibration.height),
            "base_orientation": {
                "yaw": float(calibration.base_orientation.yaw),
                "pitch": float(calibration.base_orientation.pitch),
                "roll": float(calibration.base_orientation.roll),
            },
            "distortion": {
                "k1": float(calibration.distortion.k1),
                "k2": float(calibration.distortion.k2),
                "p1": float(calibration.distortion.p1),
                "p2": float(calibration.distortion.p2),
            },
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Update cache
        self._cache[calibration.camera_id] = calibration

    def delete(self, camera_id: str) -> bool:
        """Delete a camera calibration by camera ID.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the calibration was deleted, False if it didn't exist.
        """
        yaml_path = self._id_to_path(camera_id)

        if not yaml_path.exists():
            return False

        yaml_path.unlink()

        # Remove from cache
        if camera_id in self._cache:
            del self._cache[camera_id]

        return True

    def exists(self, camera_id: str) -> bool:
        """Check if a camera calibration exists.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the calibration exists, False otherwise.
        """
        if camera_id in self._cache:
            return True
        yaml_path = self._id_to_path(camera_id)
        return yaml_path.exists()

    def _id_to_path(self, camera_id: str) -> Path:
        """Convert a camera ID to a file path.

        Args:
            camera_id: Camera ID in format "map_id/name".

        Returns:
            Path to the YAML file.
        """
        # Replace "/" with "__" for file-safe naming
        filename = camera_id.replace("/", "__") + ".yaml"
        return self._data_dir / filename

    def _load_calibration(self, yaml_path: Path) -> CameraCalibration | None:
        """Load a CameraCalibration entity from a YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            The CameraCalibration entity, or None if loading fails.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        camera_id = data["camera_id"]

        # Parse position
        pos_data = data["position"]
        position = PixelPoint(
            _x=PixelsFloat(pos_data["x"]),
            _y=PixelsFloat(pos_data["y"]),
        )

        height = Meters(data["height"])

        # Parse base orientation
        orient_data = data["base_orientation"]
        base_orientation = Orientation(
            yaw=Degrees(orient_data["yaw"]),
            pitch=Degrees(orient_data["pitch"]),
            roll=Degrees(orient_data.get("roll", 0.0)),
        )

        # Parse distortion
        dist_data = data["distortion"]
        distortion = LensDistortion(
            k1=Unitless(dist_data.get("k1", 0.0)),
            k2=Unitless(dist_data.get("k2", 0.0)),
            p1=Unitless(dist_data.get("p1", 0.0)),
            p2=Unitless(dist_data.get("p2", 0.0)),
        )

        return CameraCalibration(
            camera_id=camera_id,
            position=position,
            height=height,
            base_orientation=base_orientation,
            distortion=distortion,
        )
