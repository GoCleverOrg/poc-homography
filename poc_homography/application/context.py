"""Application context - the Composition Root for dependency injection.

This module provides the central location where all dependencies are wired together.
Instead of creating repositories scattered throughout the codebase, all repository
instantiation happens here, providing a single source of truth for the DI graph.

Usage:
    # Default configuration (uses standard data directory)
    ctx = ApplicationContext.default()

    # Custom data directory (useful for testing)
    ctx = ApplicationContext(data_dir=Path("/custom/data"))

    # Access repositories
    config = ctx.camera_config_repo.get("camera-1")
    calibration = ctx.camera_calibration_repo.get("camera-1")
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from poc_homography.infrastructure.repositories import (
    YamlCameraCalibrationRepository,
    YamlCameraConfigRepository,
    YamlGroundControlPointRepository,
    YamlMapRepository,
)


@dataclass
class ApplicationContext:
    """Composition Root - single place where the dependency graph is wired.

    This class serves as the central registry for all application dependencies.
    Repositories are lazily initialized on first access and cached for reuse.

    Attributes:
        data_dir: Root directory containing all data subdirectories.

    Example:
        ctx = ApplicationContext.default()
        cameras = ctx.camera_config_repo.get_all()
        for camera in cameras:
            calibration = ctx.camera_calibration_repo.get(camera.id)
    """

    data_dir: Path

    @cached_property
    def camera_config_repo(self) -> YamlCameraConfigRepository:
        """Repository for camera configuration entities."""
        return YamlCameraConfigRepository(self.data_dir / "cameras")

    @cached_property
    def camera_calibration_repo(self) -> YamlCameraCalibrationRepository:
        """Repository for camera calibration entities."""
        return YamlCameraCalibrationRepository(self.data_dir / "calibrations")

    @cached_property
    def map_repo(self) -> YamlMapRepository:
        """Repository for map entities."""
        return YamlMapRepository(self.data_dir / "maps")

    @cached_property
    def gcp_repo(self) -> YamlGroundControlPointRepository:
        """Repository for ground control point entities."""
        return YamlGroundControlPointRepository(self.data_dir / "ground_control_points")

    @classmethod
    def default(cls) -> ApplicationContext:
        """Create an ApplicationContext with the default data directory.

        The default data directory is located at the project root under 'data/'.

        Returns:
            ApplicationContext configured with standard paths.
        """
        # Navigate from this file to project root: application/context.py -> application -> poc_homography -> project_root
        project_root = Path(__file__).parent.parent.parent
        return cls(data_dir=project_root / "data")
