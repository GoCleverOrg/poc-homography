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
    config = ctx.repo_camera_config.get("camera-1")
    calibration = ctx.repo_camera_calibration.get("camera-1")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from poc_homography.application.services import Sam3PromptTestingService
from poc_homography.infrastructure.clients import Sam3ApiClient
from poc_homography.infrastructure.repositories import (
    RepoYamlCameraCalibration,
    RepoYamlCameraConfig,
    RepoYamlGroundControlPoint,
    RepoYamlMap,
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
        cameras = ctx.repo_camera_config.get_all()
        for camera in cameras:
            calibration = ctx.repo_camera_calibration.get(camera.id)
    """

    data_dir: Path

    @cached_property
    def repo_camera_config(self) -> RepoYamlCameraConfig:
        """Repository for camera configuration entities."""
        return RepoYamlCameraConfig(self.data_dir / "cameras")

    @cached_property
    def repo_camera_calibration(self) -> RepoYamlCameraCalibration:
        """Repository for camera calibration entities."""
        return RepoYamlCameraCalibration(self.data_dir / "calibrations")

    @cached_property
    def repo_map(self) -> RepoYamlMap:
        """Repository for map entities."""
        return RepoYamlMap(self.data_dir / "maps")

    @cached_property
    def repo_gcp(self) -> RepoYamlGroundControlPoint:
        """Repository for ground control point entities."""
        return RepoYamlGroundControlPoint(self.data_dir / "ground_control_points")

    @cached_property
    def sam3_api_key(self) -> str | None:
        """SAM3 API key from ROBOFLOW_API_KEY environment variable.

        Returns:
            API key string if set, None otherwise.
        """
        return os.environ.get("ROBOFLOW_API_KEY")

    @cached_property
    def sam3_client(self) -> Sam3ApiClient | None:
        """SAM3 API client for image segmentation.

        Returns:
            Sam3ApiClient if API key is available, None otherwise.
        """
        if self.sam3_api_key is None:
            return None
        return Sam3ApiClient(self.sam3_api_key)

    @cached_property
    def sam3_testing_service(self) -> Sam3PromptTestingService | None:
        """Service for testing SAM3 prompts.

        Returns:
            Sam3PromptTestingService if SAM3 client is available, None otherwise.
        """
        if self.sam3_client is None:
            return None
        return Sam3PromptTestingService(self.sam3_client)

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
