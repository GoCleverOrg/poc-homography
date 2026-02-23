"""YAML-based CapturedFrame repository with hierarchical folder structure."""

from __future__ import annotations

from pathlib import Path

import yaml

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.entities.captured_frame import CapturedFrame


class RepoYamlCapturedFrame:
    """Repository for CapturedFrame entities with hierarchical folder storage.

    Unlike other YAML repositories, this one uses a nested folder structure:
    - data/frames/{map_id}/{camera_name}/{timestamp}.yaml
    - Images stored alongside YAML files

    This structure allows efficient querying by map and camera.
    """

    def __init__(self, data_dir: Path, *, create_dir: bool = True) -> None:
        """Initialize the repository.

        Args:
            data_dir: Root directory for frames (typically data/frames/).
            create_dir: Whether to create the directory if it doesn't exist.
        """
        self._data_dir = Path(data_dir)
        if create_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CapturedFrame] = {}

    def image_dir_for(self, map_id: str, camera_name: str) -> Path:
        """Return the directory where images are stored for a given map/camera.

        Args:
            map_id: Map identifier.
            camera_name: Camera name.

        Returns:
            Path to the image storage directory.
        """
        return self._data_dir / map_id / camera_name

    def _id_to_path(self, entity_id: str) -> Path:
        """Convert entity ID to YAML file path.

        ID format: map_id/camera_name/timestamp
        Path: data_dir/map_id/camera_name/timestamp.yaml
        """
        parts = entity_id.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid CapturedFrame ID: {entity_id}")

        map_id, camera_name, timestamp = parts
        return self._data_dir / map_id / camera_name / f"{timestamp}.yaml"

    def _path_to_id(self, path: Path) -> str:
        """Convert YAML file path to entity ID."""
        # Path: data_dir/map_id/camera_name/timestamp.yaml
        timestamp = path.stem
        camera_name = path.parent.name
        map_id = path.parent.parent.name
        return f"{map_id}/{camera_name}/{timestamp}"

    def get(self, entity_id: str) -> CapturedFrame | None:
        """Get a captured frame by ID."""
        if entity_id in self._cache:
            return self._cache[entity_id]

        path = self._id_to_path(entity_id)
        if not path.exists():
            return None

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return None

        entity = CapturedFrame.from_dict(data)
        self._cache[entity_id] = entity
        return entity

    def save(self, entity: CapturedFrame) -> None:
        """Save a captured frame."""
        path = self._id_to_path(entity.id)

        # Create parent directories (map_id/camera_name/)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(entity.to_dict(), f, default_flow_style=False, sort_keys=False)

        self._cache[entity.id] = entity

    def delete(self, entity_id: str) -> bool:
        """Delete a captured frame, its associated image, and annotations."""
        path = self._id_to_path(entity_id)
        if not path.exists():
            return False

        # Load to get image path before deleting
        entity = self.get(entity_id)

        path.unlink()
        self._cache.pop(entity_id, None)

        # Delete associated image (relative to YAML location)
        if entity and entity.image_path:
            image_full_path = (path.parent / entity.image_path).resolve()
            if (
                image_full_path.is_relative_to(self._data_dir.resolve())
                and image_full_path.exists()
            ):
                image_full_path.unlink()

        # Delete associated annotations
        self.delete_annotations(entity_id)

        return True

    def exists(self, entity_id: str) -> bool:
        """Check if a captured frame exists."""
        if entity_id in self._cache:
            return True
        return self._id_to_path(entity_id).exists()

    def get_all(self) -> list[CapturedFrame]:
        """Get all captured frames."""
        entities = []
        # Walk through map_id/camera_name/ subdirectories
        for yaml_path in self._data_dir.glob("*/*/*.yaml"):
            if yaml_path.stem.endswith("_annotations"):
                continue
            entity_id = self._path_to_id(yaml_path)
            entity = self.get(entity_id)
            if entity:
                entities.append(entity)
        return entities

    def get_by_map(self, map_id: str) -> list[CapturedFrame]:
        """Get all frames for a specific map.

        Args:
            map_id: Map identifier.

        Returns:
            List of CapturedFrame entities for the map.
        """
        map_dir = self._data_dir / map_id
        if not map_dir.exists():
            return []

        entities = []
        for yaml_path in map_dir.glob("*/*.yaml"):
            if yaml_path.stem.endswith("_annotations"):
                continue
            entity_id = self._path_to_id(yaml_path)
            entity = self.get(entity_id)
            if entity:
                entities.append(entity)
        return entities

    def get_by_camera(
        self,
        map_id: str,
        camera_name: str,
    ) -> list[CapturedFrame]:
        """Get all frames for a specific camera.

        Efficiently queries only the camera's subdirectory.

        Args:
            map_id: Map identifier.
            camera_name: Camera name.

        Returns:
            List of CapturedFrame entities for the camera.
        """
        camera_dir = self._data_dir / map_id / camera_name
        if not camera_dir.exists():
            return []

        entities = []
        for yaml_path in camera_dir.glob("*.yaml"):
            if yaml_path.stem.endswith("_annotations"):
                continue
            entity_id = self._path_to_id(yaml_path)
            entity = self.get(entity_id)
            if entity:
                entities.append(entity)
        return entities

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    def get_image_path(self, entity: CapturedFrame) -> Path:
        """Get the absolute path to a frame's image.

        Args:
            entity: The captured frame entity.

        Returns:
            Absolute path to the image file.
        """
        yaml_path = self._id_to_path(entity.id)
        return yaml_path.parent / entity.image_path

    def _id_to_annotations_path(self, entity_id: str) -> Path:
        """Convert entity ID to annotations YAML file path.

        ID format: map_id/camera_name/timestamp
        Path: data_dir/map_id/camera_name/timestamp_annotations.yaml
        """
        parts = entity_id.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid CapturedFrame ID: {entity_id}")

        map_id, camera_name, timestamp = parts
        return self._data_dir / map_id / camera_name / f"{timestamp}_annotations.yaml"

    def get_annotations(self, frame_id: str) -> list[Annotation]:
        """Get all annotations for a captured frame.

        Args:
            frame_id: ID of the captured frame.

        Returns:
            List of Annotation entities for the frame.
            Empty list if frame doesn't exist or has no annotations.
        """
        annotations_path = self._id_to_annotations_path(frame_id)
        if not annotations_path.exists():
            return []

        with open(annotations_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "annotations" not in data:
            return []

        return [Annotation.from_dict(ann) for ann in data["annotations"]]

    def save_annotations(self, frame_id: str, annotations: list[Annotation]) -> None:
        """Save annotations for a captured frame.

        Args:
            frame_id: ID of the captured frame.
            annotations: List of Annotation entities to save.

        Raises:
            ValueError: If the frame doesn't exist.
        """
        # Verify frame exists
        if not self.exists(frame_id):
            raise ValueError(f"Cannot save annotations: frame '{frame_id}' not found")

        annotations_path = self._id_to_annotations_path(frame_id)

        # Create parent directories if needed
        annotations_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "frame_id": frame_id,
            "annotations": [ann.to_dict() for ann in annotations],
        }

        with open(annotations_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def delete_annotations(self, frame_id: str) -> bool:
        """Delete annotations for a captured frame.

        Args:
            frame_id: ID of the captured frame.

        Returns:
            True if annotations were deleted, False if they didn't exist.
        """
        annotations_path = self._id_to_annotations_path(frame_id)
        if not annotations_path.exists():
            return False

        annotations_path.unlink()
        return True
