#!/usr/bin/env python3
"""
Homography provider using map points for coordinate transformation.

This module implements homography-based transformation between camera image pixels
and map coordinates using map point references instead of GPS coordinates.

Key Concepts:
    - Map Points: Reference points on a map with pixel coordinates
    - GCPs (Ground Control Points): Correspondences between camera pixels and map point IDs
    - Homography: 3x3 transformation matrix mapping between coordinate systems
    - Round-trip validation: Camera -> Map -> Camera consistency checks

Coordinate Systems:
    - Camera pixels: (u, v) in image space, origin at top-left
    - Map coordinates: (pixel_x, pixel_y) in map image space

Example usage:
    >>> from poc_homography.map_points import MapPointRegistry
    >>> from poc_homography.homography_map_points import MapPointHomography
    >>>
    >>> # Load map points
    >>> registry = MapPointRegistry.load("map_points.json")
    >>>
    >>> # Create homography provider
    >>> homography = MapPointHomography(map_id="map_valte")
    >>>
    >>> # Compute homography from GCPs
    >>> gcps = [
    ...     {"pixel_x": 800, "pixel_y": 580, "map_point_id": "A7"},
    ...     {"pixel_x": 1082, "pixel_y": 390, "map_point_id": "A6"},
    ...     # ... more GCPs
    ... ]
    >>> result = homography.compute_from_gcps(gcps, registry)
    >>>
    >>> # Project camera pixel to map - returns MapPoint
    >>> map_point = homography.camera_to_map((960, 540))
    >>> print(f"Point {map_point.id} on map '{map_point.map_id}'")
    >>> print(f"Pixel: ({map_point.pixel_x}, {map_point.pixel_y})")
    >>>
    >>> # Use cached PixelPoint property
    >>> pixel = map_point.pixel
    >>> print(f"As PixelPoint: ({pixel.x}, {pixel.y})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from poc_homography.map_points import MapPoint, MapPointRegistry


@dataclass
class HomographyResult:
    """Result of homography computation including matrix and quality metrics.

    Attributes:
        homography_matrix: 3x3 transformation matrix (camera pixels -> map coords)
        inverse_matrix: 3x3 inverse transformation matrix (map coords -> camera pixels)
        num_gcps: Total number of ground control points used
        num_inliers: Number of inlier points after RANSAC
        inlier_ratio: Ratio of inliers to total points [0.0, 1.0]
        mean_reproj_error: Mean reprojection error in pixels
        max_reproj_error: Maximum reprojection error in pixels
        rmse: Root mean square error in pixels
    """

    homography_matrix: npt.NDArray[np.float64]
    inverse_matrix: npt.NDArray[np.float64]
    num_gcps: int
    num_inliers: int
    inlier_ratio: float
    mean_reproj_error: float
    max_reproj_error: float
    rmse: float


class MapPointHomography:
    """Homography provider for transforming between camera pixels and map coordinates.

    This class computes and applies homography transformations using map point
    references. It maintains internal state of the current homography matrix
    and provides methods for bidirectional coordinate transformation.

    Attributes:
        _map_id: Identifier of the map for generated MapPoints
        _H: Forward homography matrix (camera -> map), shape (3, 3)
        _H_inv: Inverse homography matrix (map -> camera), shape (3, 3)
        _is_valid: Whether a valid homography has been computed
        _result: Last computation result with quality metrics
        _point_counter: Counter for generating unique point IDs
    """

    def __init__(self, map_id: str) -> None:
        """Initialize homography provider with map identifier.

        Args:
            map_id: Identifier of the map for generated MapPoints (e.g., "map_valte")
        """
        self._map_id = map_id
        self._H: npt.NDArray[np.float64] | None = None
        self._H_inv: npt.NDArray[np.float64] | None = None
        self._is_valid: bool = False
        self._result: HomographyResult | None = None
        self._point_counter: int = 0

    @property
    def map_id(self) -> str:
        """Get the map identifier."""
        return self._map_id

    def compute_from_gcps(
        self,
        gcps: list[dict[str, Any]],
        map_registry: MapPointRegistry,
        ransac_threshold: float = 50.0,
        min_inlier_ratio: float = 0.5,
    ) -> HomographyResult:
        """Compute homography from ground control points.

        This method extracts camera pixel coordinates and corresponding map
        coordinates from GCPs, then computes a homography using RANSAC to
        handle outliers.

        Args:
            gcps: List of GCP dictionaries with keys:
                - "pixel_x": Camera pixel x coordinate
                - "pixel_y": Camera pixel y coordinate
                - "map_point_id": ID of map point in registry
            map_registry: Registry containing map points with pixel coordinates
            ransac_threshold: RANSAC reprojection error threshold in pixels (default: 50.0)
            min_inlier_ratio: Minimum ratio of inliers to consider valid (default: 0.5)

        Returns:
            HomographyResult with computed matrices and quality metrics

        Raises:
            ValueError: If insufficient GCPs, missing map points, or poor fit quality
            RuntimeError: If homography computation fails
        """
        if len(gcps) < 4:
            raise ValueError(f"Need at least 4 GCPs, got {len(gcps)}")

        # Extract correspondences
        camera_pixels = []
        map_coords = []

        for gcp in gcps:
            # Extract camera pixel
            camera_pixels.append([gcp["pixel_x"], gcp["pixel_y"]])

            # Extract map coordinate from registry
            map_point_id = gcp["map_point_id"]
            if map_point_id not in map_registry.points:
                raise ValueError(f"Map point not found in registry: {map_point_id}")

            map_point = map_registry.points[map_point_id]
            map_coords.append([map_point.pixel_x, map_point.pixel_y])

        # Convert to numpy arrays
        camera_pixels_array = np.array(camera_pixels, dtype=np.float32)
        map_coords_array = np.array(map_coords, dtype=np.float32)

        # Compute homography using RANSAC
        H, mask = cv2.findHomography(
            camera_pixels_array, map_coords_array, cv2.RANSAC, ransac_threshold
        )

        if H is None:
            raise RuntimeError("Homography computation failed")

        # Convert to numpy array for type safety
        H_array: npt.NDArray[np.float64] = np.asarray(H, dtype=np.float64)

        # Check validity
        if np.linalg.det(H_array) == 0:
            raise RuntimeError("Computed homography is singular")

        # Count inliers
        num_inliers = int(np.sum(mask))
        inlier_ratio = num_inliers / len(gcps)

        if inlier_ratio < min_inlier_ratio:
            raise ValueError(f"Inlier ratio too low: {inlier_ratio:.2%} < {min_inlier_ratio:.2%}")

        # Compute inverse
        H_inv = np.asarray(np.linalg.inv(H_array), dtype=np.float64)

        # Compute reprojection errors
        errors = []
        for i, gcp in enumerate(gcps):
            # Only consider inliers
            if mask[i] == 0:
                continue

            # Project camera pixel to map
            camera_pt = np.array([[[gcp["pixel_x"], gcp["pixel_y"]]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(camera_pt, H)[0, 0]

            # Expected map coordinate
            map_point = map_registry.points[gcp["map_point_id"]]
            expected = np.array([map_point.pixel_x, map_point.pixel_y])

            # Calculate error in pixels
            error = float(np.linalg.norm(projected - expected))
            errors.append(error)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

        # Store state
        self._H = H_array
        self._H_inv = H_inv
        self._is_valid = True

        # Create result
        result = HomographyResult(
            homography_matrix=H_array,
            inverse_matrix=H_inv,
            num_gcps=len(gcps),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            mean_reproj_error=mean_error,
            max_reproj_error=max_error,
            rmse=rmse,
        )

        self._result = result
        return result

    def camera_to_map(
        self,
        camera_pixel: tuple[float, float],
        point_id: str = "",
    ) -> MapPoint:
        """Transform camera pixel to map coordinate.

        Args:
            camera_pixel: (x, y) pixel coordinates in camera image
            point_id: Optional ID for the generated MapPoint (auto-generated if empty)

        Returns:
            MapPoint with pixel coordinates on the map

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        # Convert to required shape for cv2.perspectiveTransform
        point = np.array([[[camera_pixel[0], camera_pixel[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self._H)

        # Generate point ID if not provided
        if not point_id:
            self._point_counter += 1
            point_id = f"proj_{self._point_counter}"

        return MapPoint(
            id=point_id,
            pixel_x=float(transformed[0, 0, 0]),
            pixel_y=float(transformed[0, 0, 1]),
            map_id=self._map_id,
        )

    def map_to_camera(self, map_coord: tuple[float, float]) -> tuple[float, float]:
        """Transform map coordinate to camera pixel.

        Args:
            map_coord: (pixel_x, pixel_y) coordinates on the map

        Returns:
            (x, y) pixel coordinates in camera image

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H_inv is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        # Convert to required shape for cv2.perspectiveTransform
        point = np.array([[[map_coord[0], map_coord[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self._H_inv)

        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def camera_to_map_batch(
        self,
        camera_pixels: list[tuple[float, float]],
        point_id_prefix: str = "proj",
    ) -> list[MapPoint]:
        """Transform multiple camera pixels to map coordinates.

        Args:
            camera_pixels: List of (x, y) pixel coordinates
            point_id_prefix: Prefix for generated MapPoint IDs (default: "proj")

        Returns:
            List of MapPoints with pixel coordinates on the map

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        # Convert to required shape
        points = np.array([[[p[0], p[1]]] for p in camera_pixels], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, self._H)

        # Extract results and create MapPoints
        results = []
        for i in range(len(transformed)):
            self._point_counter += 1
            t = transformed[i]
            results.append(
                MapPoint(
                    id=f"{point_id_prefix}_{self._point_counter}",
                    pixel_x=float(t[0][0]),
                    pixel_y=float(t[0][1]),
                    map_id=self._map_id,
                )
            )
        return results

    def map_to_camera_batch(
        self, map_coords: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Transform multiple map coordinates to camera pixels.

        Args:
            map_coords: List of (pixel_x, pixel_y) coordinates on the map

        Returns:
            List of (x, y) pixel coordinates

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H_inv is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        # Convert to required shape
        points = np.array([[[c[0], c[1]]] for c in map_coords], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, self._H_inv)

        # Extract results
        return [
            (float(transformed[i][0][0]), float(transformed[i][0][1]))
            for i in range(len(transformed))
        ]

    def is_valid(self) -> bool:
        """Check if a valid homography has been computed.

        Returns:
            True if homography is valid and ready for projections
        """
        return self._is_valid

    def get_result(self) -> HomographyResult | None:
        """Get the last computation result.

        Returns:
            HomographyResult if compute_from_gcps was called, None otherwise
        """
        return self._result

    def get_homography_matrix(self) -> npt.NDArray[np.float64]:
        """Get the forward homography matrix (camera -> map).

        Returns:
            3x3 homography matrix

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        return self._H.copy()

    def get_inverse_matrix(self) -> npt.NDArray[np.float64]:
        """Get the inverse homography matrix (map -> camera).

        Returns:
            3x3 inverse homography matrix

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        if not self._is_valid or self._H_inv is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")

        return self._H_inv.copy()
