#!/usr/bin/env python3
"""
Homography provider using map points for coordinate transformation.

Implements homography-based transformation between camera image pixels
and map coordinates using map point references.

Coordinate Systems:
    - Camera pixels: (u, v) in image space, origin at top-left
    - Map coordinates: (pixel_x, pixel_y) in map image space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from poc_homography.domain.vo import PixelPoint
from poc_homography.map_points import GCPRegistry, MapPoint


@dataclass(frozen=True)
class MapPointComputationResult:
    """Result of MapPoint homography computation including matrix and quality metrics.

    This class is distinct from HomographyResult in homography_interface.py, which
    provides a generic result structure. This class contains detailed metrics
    specific to MapPoint-based GCP homography computation.

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


@dataclass(frozen=True)
class LineComputationResult:
    """Result of line-based homography computation including matrix and quality metrics.

    Attributes:
        homography_matrix: 3x3 transformation matrix (camera pixels -> map coords)
        inverse_matrix: 3x3 inverse transformation matrix (map coords -> camera pixels)
        num_lines: Total number of lines used for computation
        num_inliers: Number of inlier point-to-line correspondences after RANSAC
        inlier_ratio: Ratio of inliers to total point-to-line correspondences [0.0, 1.0]
        mean_perp_error: Mean perpendicular distance error in map pixels
        max_perp_error: Maximum perpendicular distance error in map pixels
        rmse: Root mean square of perpendicular errors in map pixels
    """

    homography_matrix: npt.NDArray[np.float64]
    inverse_matrix: npt.NDArray[np.float64]
    num_lines: int
    num_inliers: int
    inlier_ratio: float
    mean_perp_error: float
    max_perp_error: float
    rmse: float


class MapPointHomography:
    """Homography provider for transforming between camera pixels and map coordinates.

    Computes and applies homography transformations using map point references.
    Maintains internal state of the current homography matrix for bidirectional
    coordinate transformation.
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
        self._result: MapPointComputationResult | None = None

    def _require_forward_homography(self) -> npt.NDArray[np.float64]:
        """Return forward homography matrix or raise RuntimeError if not computed."""
        if not self._is_valid or self._H is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")
        return self._H

    def _require_inverse_homography(self) -> npt.NDArray[np.float64]:
        """Return inverse homography matrix or raise RuntimeError if not computed."""
        if not self._is_valid or self._H_inv is None:
            raise RuntimeError("No valid homography. Call compute_from_gcps() first.")
        return self._H_inv

    @property
    def map_id(self) -> str:
        """Get the map identifier."""
        return self._map_id

    def compute_from_gcps(
        self,
        gcps: list[dict[str, Any]],
        map_registry: GCPRegistry,
        ransac_threshold: float = 50.0,
        min_inlier_ratio: float = 0.5,
    ) -> MapPointComputationResult:
        """Compute homography from ground control points.

        This method extracts camera pixel coordinates and corresponding map
        coordinates from GCPs, then computes a homography using RANSAC to
        handle outliers.

        Args:
            gcps: List of annotation-GCP correspondence dictionaries with keys:
                - "pixel_x": Camera pixel x coordinate (annotation)
                - "pixel_y": Camera pixel y coordinate (annotation)
                - "gcp_id": ID of GCP (MapPoint) in registry
            map_registry: Registry containing GCPs (MapPoints) with pixel coordinates
            ransac_threshold: RANSAC reprojection error threshold in pixels (default: 50.0)
            min_inlier_ratio: Minimum ratio of inliers to consider valid (default: 0.5)

        Returns:
            MapPointComputationResult with computed matrices and quality metrics

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

            # Extract GCP coordinate from registry
            gcp_id = gcp["gcp_id"]
            if gcp_id not in map_registry.points:
                raise ValueError(f"GCP not found in registry: {gcp_id}")

            map_point = map_registry.points[gcp_id]
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
        if abs(np.linalg.det(H_array)) < 1e-15:
            raise RuntimeError("Computed homography is singular or near-singular")

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

            # Expected GCP coordinate
            map_point = map_registry.points[gcp["gcp_id"]]
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
        result = MapPointComputationResult(
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

    def compute_from_lines(
        self,
        line_annotations: list[dict[str, Any]],
        line_registry: dict[str, dict[str, float]],
        ransac_threshold: float = 50.0,
        min_inlier_ratio: float = 0.5,
        max_iterations: int = 20,
        convergence_threshold: float = 0.1,
    ) -> LineComputationResult:
        """Compute homography from line correspondences using iterative closest point.

        This method uses an Iterative Closest Point (ICP) approach for lines:
        1. Start with initial homography estimate from endpoint matching
        2. Project camera points to map using current H
        3. Find closest point on corresponding map line (perpendicular projection)
        4. Use (camera_point, closest_point_on_line) as correspondences
        5. Recompute H with cv2.findHomography
        6. Iterate until convergence

        This handles the case where annotated line segments don't align with
        registry line endpoints - we find the true correspondence by projecting
        onto the infinite line.

        Args:
            line_annotations: List of line annotation dictionaries with keys:
                - "line_id": ID of line in registry
                - "start_pixel_x", "start_pixel_y": Camera start point
                - "end_pixel_x", "end_pixel_y": Camera end point
            line_registry: Dict mapping line_id to line definition with keys:
                - "start_x", "start_y": Map start point
                - "end_x", "end_y": Map end point
            ransac_threshold: Max reprojection error (in map pixels) for inliers
            min_inlier_ratio: Minimum ratio of inliers to consider valid
            max_iterations: Maximum ICP iterations
            convergence_threshold: Stop when mean error change is below this

        Returns:
            LineComputationResult with computed matrices and quality metrics

        Raises:
            ValueError: If insufficient lines, missing lines, or poor fit quality
            RuntimeError: If homography computation fails
        """
        if len(line_annotations) < 2:
            raise ValueError(
                f"Need at least 2 line annotations (4 points), got {len(line_annotations)}"
            )

        # Parse line annotations and build data structures
        camera_points = []  # All camera points
        line_indices = []  # Which line each point belongs to
        line_info = []  # Line metadata

        for idx, ann in enumerate(line_annotations):
            line_id = ann["line_id"]
            if line_id not in line_registry:
                raise ValueError(f"Line not found in registry: {line_id}")

            line_def = line_registry[line_id]

            # Get map line endpoints
            map_start = np.array([line_def["start_x"], line_def["start_y"]])
            map_end = np.array([line_def["end_x"], line_def["end_y"]])
            line_len = np.linalg.norm(map_end - map_start)

            if line_len < 1e-10:
                raise ValueError(f"Line {line_id} has zero length")

            # Get camera annotation points
            cam_start = np.array([ann["start_pixel_x"], ann["start_pixel_y"]])
            cam_end = np.array([ann["end_pixel_x"], ann["end_pixel_y"]])

            camera_points.extend([cam_start, cam_end])
            line_indices.extend([idx, idx])

            line_info.append(
                {
                    "line_id": line_id,
                    "map_start": map_start,
                    "map_end": map_end,
                    "cam_start": cam_start,
                    "cam_end": cam_end,
                }
            )

        camera_points = np.array(camera_points, dtype=np.float64)
        num_points = len(camera_points)

        # Step 1: Get initial homography estimate using endpoint matching
        # Try both orientations for each line and pick best combination
        H = self._get_initial_line_homography(line_info, ransac_threshold)

        if H is None:
            raise RuntimeError("Failed to compute initial homography estimate")

        # Step 2: Iterative Closest Point refinement
        prev_mean_error = float("inf")

        for iteration in range(max_iterations):
            # Project all camera points to map
            cam_pts_cv = camera_points.reshape(-1, 1, 2).astype(np.float32)
            projected = cv2.perspectiveTransform(cam_pts_cv, H).reshape(-1, 2)

            # Find closest point on corresponding line for each projected point
            map_correspondences = np.zeros_like(projected)

            for i, (proj_pt, line_idx) in enumerate(zip(projected, line_indices)):
                info = line_info[line_idx]
                # Project onto infinite line (not clamped to segment)
                proj_pt_typed: npt.NDArray[np.float64] = np.asarray(proj_pt, dtype=np.float64)
                closest = self._project_point_onto_line(
                    proj_pt_typed, info["map_start"], info["map_end"]
                )
                map_correspondences[i] = closest

            # Compute new homography from (camera_points, map_correspondences)
            H_new, mask = cv2.findHomography(
                camera_points.astype(np.float32),
                map_correspondences.astype(np.float32),
                cv2.RANSAC,
                ransac_threshold,
            )

            if H_new is None:
                break

            H = H_new

            # Compute current mean error (perpendicular distance to lines)
            errors = []
            for i, line_idx in enumerate(line_indices):
                info = line_info[line_idx]
                cam_pt = camera_points[i]
                pt_cv = np.array([[[cam_pt[0], cam_pt[1]]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt_cv, H)[0, 0]
                dist = self._point_to_line_distance(proj, info["map_start"], info["map_end"])
                errors.append(dist)

            mean_error = np.mean(errors)

            # Check convergence
            if abs(prev_mean_error - mean_error) < convergence_threshold:
                break

            prev_mean_error = mean_error

        H_array = np.asarray(H, dtype=np.float64)

        # Check validity
        if abs(np.linalg.det(H_array)) < 1e-15:
            raise RuntimeError("Computed homography is singular or near-singular")

        # Count inliers based on perpendicular distance to lines
        inlier_count = 0
        errors = []
        for i, line_idx in enumerate(line_indices):
            info = line_info[line_idx]
            cam_pt = camera_points[i]
            pt_cv = np.array([[[cam_pt[0], cam_pt[1]]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt_cv, H)[0, 0]
            dist = self._point_to_line_distance(proj, info["map_start"], info["map_end"])
            errors.append(dist)
            if dist < ransac_threshold:
                inlier_count += 1

        inlier_ratio = inlier_count / num_points

        if inlier_ratio < min_inlier_ratio:
            raise ValueError(f"Inlier ratio too low: {inlier_ratio:.2%} < {min_inlier_ratio:.2%}")

        # Compute inverse
        H_inv = np.asarray(np.linalg.inv(H_array), dtype=np.float64)

        errors = np.array(errors)
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))

        # Store state
        self._H = H_array
        self._H_inv = H_inv
        self._is_valid = True

        # Create result
        result = LineComputationResult(
            homography_matrix=H_array,
            inverse_matrix=H_inv,
            num_lines=len(line_annotations),
            num_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_perp_error=mean_error,
            max_perp_error=max_error,
            rmse=rmse,
        )

        return result

    def _get_initial_line_homography(
        self,
        line_info: list[dict[str, Any]],
        ransac_threshold: float,
    ) -> npt.NDArray[np.float64] | None:
        """Get initial homography estimate by trying different endpoint orientations.

        Tries both forward and reverse orientations for each line and picks
        the combination that gives the best initial fit.
        """
        n_lines = len(line_info)

        # Try all 2^n combinations of orientations (limited to 16 lines max)
        if n_lines > 16:
            # For many lines, use greedy approach
            return self._get_initial_homography_greedy(line_info, ransac_threshold)

        best_H = None
        best_error = float("inf")

        for orientation_bits in range(2**n_lines):
            camera_points = []
            map_points = []

            for i, info in enumerate(line_info):
                cam_start = info["cam_start"]
                cam_end = info["cam_end"]
                map_start = info["map_start"]
                map_end = info["map_end"]

                # Check if this line should be reversed
                if (orientation_bits >> i) & 1:
                    # Reversed: cam_start -> map_end, cam_end -> map_start
                    camera_points.extend([cam_start, cam_end])
                    map_points.extend([map_end, map_start])
                else:
                    # Forward: cam_start -> map_start, cam_end -> map_end
                    camera_points.extend([cam_start, cam_end])
                    map_points.extend([map_start, map_end])

            camera_pts = np.array(camera_points, dtype=np.float32)
            map_pts = np.array(map_points, dtype=np.float32)

            H, mask = cv2.findHomography(camera_pts, map_pts, cv2.RANSAC, ransac_threshold)

            if H is None:
                continue

            # Compute total reprojection error
            projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            error = np.sum(np.linalg.norm(projected - map_pts, axis=1))

            if error < best_error:
                best_error = error
                best_H = H

        if best_H is None:
            return None
        return np.asarray(best_H, dtype=np.float64)

    def _get_initial_homography_greedy(
        self,
        line_info: list[dict[str, Any]],
        ransac_threshold: float,
    ) -> npt.NDArray[np.float64] | None:
        """Greedy approach for initial homography with many lines."""
        camera_points = []
        map_points = []

        # Start with forward orientation for all
        for info in line_info:
            camera_points.extend([info["cam_start"], info["cam_end"]])
            map_points.extend([info["map_start"], info["map_end"]])

        camera_pts = np.array(camera_points, dtype=np.float32)
        map_pts = np.array(map_points, dtype=np.float32)

        H, _ = cv2.findHomography(camera_pts, map_pts, cv2.RANSAC, ransac_threshold)
        if H is None:
            return None
        return np.asarray(H, dtype=np.float64)

    def _project_point_onto_line(
        self,
        point: npt.NDArray[np.float64],
        line_start: npt.NDArray[np.float64],
        line_end: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Project a point onto an infinite line (not clamped to segment).

        Args:
            point: Point to project [x, y]
            line_start: Line start point [x, y]
            line_end: Line end point [x, y]

        Returns:
            Closest point on the infinite line
        """
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-10:
            return line_start.copy()

        # Parameter t for projection (not clamped)
        t = np.dot(point_vec, line_vec) / line_len_sq

        # Return projected point on infinite line
        return line_start + t * line_vec

    def _point_to_line_distance(
        self,
        point: npt.NDArray[np.float64],
        line_start: npt.NDArray[np.float64],
        line_end: npt.NDArray[np.float64],
    ) -> float:
        """Compute perpendicular distance from point to line segment.

        Args:
            point: Point coordinates [x, y]
            line_start: Line start point [x, y]
            line_end: Line end point [x, y]

        Returns:
            Perpendicular distance
        """
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-10:
            return float(np.linalg.norm(point_vec))

        # Project point onto line (unclamped to allow extension)
        t = np.dot(point_vec, line_vec) / line_len_sq
        projection = line_start + t * line_vec

        return float(np.linalg.norm(point - projection))

    def camera_to_map(
        self,
        camera_pixel: PixelPoint,
        point_id: str = "",
    ) -> MapPoint:
        """Transform camera pixel to map coordinate.

        Args:
            camera_pixel: Pixel coordinates in camera image
            point_id: Optional ID for the generated MapPoint (auto-generated if empty)

        Returns:
            MapPoint with pixel coordinates on the map

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        H = self._require_forward_homography()
        point = np.array([[[camera_pixel.x, camera_pixel.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)

        return MapPoint(
            pixel_x=float(transformed[0, 0, 0]),
            pixel_y=float(transformed[0, 0, 1]),
        )

    def map_to_camera(self, map_coord: PixelPoint) -> PixelPoint:
        """Transform map coordinate to camera pixel.

        Args:
            map_coord: Pixel coordinates on the map

        Returns:
            Pixel coordinates in camera image

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        H_inv = self._require_inverse_homography()
        point = np.array([[[map_coord.x, map_coord.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H_inv)
        return PixelPoint.create(float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

    def camera_to_map_batch(
        self,
        camera_pixels: list[PixelPoint],
        point_id_prefix: str = "proj",
    ) -> list[MapPoint]:
        """Transform multiple camera pixels to map coordinates.

        Args:
            camera_pixels: List of pixel coordinates
            point_id_prefix: Prefix for generated MapPoint IDs (default: "proj")

        Returns:
            List of MapPoints with pixel coordinates on the map

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        H = self._require_forward_homography()
        points = np.array([[[p.x, p.y]] for p in camera_pixels], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, H)
        transformed_2d: npt.NDArray[np.float64] = np.asarray(transformed, dtype=np.float64).reshape(
            -1, 2
        )

        results = []
        for pt in transformed_2d:
            results.append(
                MapPoint(
                    pixel_x=float(pt[0]),
                    pixel_y=float(pt[1]),
                )
            )
        return results

    def map_to_camera_batch(self, map_coords: list[PixelPoint]) -> list[PixelPoint]:
        """Transform multiple map coordinates to camera pixels.

        Args:
            map_coords: List of pixel coordinates on the map

        Returns:
            List of pixel coordinates in camera image

        Raises:
            RuntimeError: If no valid homography has been computed
        """
        H_inv = self._require_inverse_homography()
        points = np.array([[[c.x, c.y]] for c in map_coords], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, H_inv)
        transformed_2d: npt.NDArray[np.float64] = np.asarray(transformed, dtype=np.float64).reshape(
            -1, 2
        )
        return [PixelPoint.create(float(pt[0]), float(pt[1])) for pt in transformed_2d]

    def is_valid(self) -> bool:
        """Check if a valid homography has been computed."""
        return self._is_valid

    def get_result(self) -> MapPointComputationResult | None:
        """Get the last computation result."""
        return self._result

    def get_homography_matrix(self) -> npt.NDArray[np.float64]:
        """Get the forward homography matrix (camera -> map)."""
        return self._require_forward_homography().copy()

    def get_inverse_matrix(self) -> npt.NDArray[np.float64]:
        """Get the inverse homography matrix (map -> camera)."""
        return self._require_inverse_homography().copy()
