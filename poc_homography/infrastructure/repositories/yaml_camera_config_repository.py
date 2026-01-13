"""YAML-based CameraConfig repository implementation."""

from pathlib import Path

import yaml

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential


class YamlCameraConfigRepository:
    """Repository that loads CameraConfig entities from YAML files.

    Expected YAML format:
        map_id: valte
        name: Valte
        spec: HIKVISION_DS_2DF8425IX
        credential:
          username: admin
          password: password123
        ip_address: 10.207.99.178  # optional

    Files are stored with names like "valte__Valte.yaml" where
    the camera_id "valte/Valte" has "/" replaced with "__".
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing camera YAML files.
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CameraConfig] = {}

    def get(self, camera_id: str) -> CameraConfig | None:
        """Retrieve a camera configuration by its ID.

        Args:
            camera_id: Unique identifier for the camera (format: "map_id/name").

        Returns:
            The CameraConfig entity if found, None otherwise.
        """
        if camera_id in self._cache:
            return self._cache[camera_id]

        yaml_path = self._id_to_path(camera_id)
        if not yaml_path.exists():
            return None

        config = self._load_config(yaml_path)
        if config:
            self._cache[camera_id] = config
        return config

    def get_by_map(self, map_id: str) -> dict[str, CameraConfig]:
        """Retrieve all camera configurations for a specific map.

        Args:
            map_id: ID of the map to get camera configs for.

        Returns:
            Dictionary mapping camera_id to CameraConfig for all cameras on the map.
        """
        result: dict[str, CameraConfig] = {}
        prefix = f"{map_id}__"

        for yaml_path in self._data_dir.glob("*.yaml"):
            if yaml_path.stem.startswith(prefix):
                config = self._load_config(yaml_path)
                if config and config.map_id == map_id:
                    result[config.id] = config
                    self._cache[config.id] = config

        return result

    def save(self, config: CameraConfig) -> None:
        """Save a camera configuration (create or update).

        Args:
            config: The CameraConfig entity to save.
        """
        yaml_path = self._id_to_path(config.id)

        data = {
            "map_id": config.map_id,
            "name": config.name,
            "spec": config.spec.name,  # Store enum name
            "credential": {
                "username": config.credential.username,
                "password": config.credential.password,
            },
        }

        if config.ip_address:
            data["ip_address"] = config.ip_address

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Update cache
        self._cache[config.id] = config

    def delete(self, camera_id: str) -> bool:
        """Delete a camera configuration by its ID.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the configuration was deleted, False if it didn't exist.
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
        """Check if a camera configuration exists.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            True if the configuration exists, False otherwise.
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

    def _load_config(self, yaml_path: Path) -> CameraConfig | None:
        """Load a CameraConfig entity from a YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            The CameraConfig entity, or None if loading fails.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        map_id = data["map_id"]
        name = data["name"]

        # Look up CameraSpec enum by name
        spec_name = data["spec"]
        spec = CameraSpec[spec_name]

        # Parse credential
        cred_data = data["credential"]
        credential = Credential(
            username=cred_data["username"],
            password=cred_data["password"],
        )

        ip_address = data.get("ip_address")

        return CameraConfig(
            map_id=map_id,
            name=name,
            spec=spec,
            credential=credential,
            ip_address=ip_address,
        )
