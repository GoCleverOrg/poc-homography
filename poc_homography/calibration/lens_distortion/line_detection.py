"""Automatic parking line detection using computer vision.

This module provides Hough transform-based detection of parking spot lines
in camera images. Detected lines can be used as calibration targets for
lens distortion estimation.

The detection pipeline:
1. Convert to grayscale
2. Apply Gaussian blur to reduce noise
3. Apply Canny edge detection
4. Use Probabilistic Hough Line Transform
5. Filter by length, orientation, and edge strength
6. Cluster parallel lines
7. Return candidate lines with confidence scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition

logger = logging.getLogger(__name__)


@dataclass
class LineDetectionConfig:
    """Configuration for parking line detection.

    Attributes:
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        gaussian_kernel_size: Size of Gaussian blur kernel (must be odd).
        hough_rho: Distance resolution for Hough transform (pixels).
        hough_theta: Angle resolution for Hough transform (radians).
        hough_threshold: Accumulator threshold for Hough transform.
        min_line_length: Minimum line length in pixels.
        max_line_gap: Maximum gap between line segments to connect.
        min_confidence: Minimum confidence score to keep a line.
        parallel_angle_tolerance: Max angle difference for parallel lines (degrees).
        merge_distance_threshold: Max distance to merge nearby parallel lines.
    """

    canny_low: int = 50
    canny_high: int = 150
    gaussian_kernel_size: int = 5
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50
    min_line_length: int = 100
    max_line_gap: int = 10
    min_confidence: float = 0.3
    parallel_angle_tolerance: float = 5.0  # degrees
    merge_distance_threshold: float = 20.0  # pixels

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError(f"gaussian_kernel_size must be odd, got {self.gaussian_kernel_size}")
        if self.canny_low >= self.canny_high:
            raise ValueError(
                f"canny_low ({self.canny_low}) must be less than canny_high ({self.canny_high})"
            )
        if self.min_line_length <= 0:
            raise ValueError(f"min_line_length must be positive, got {self.min_line_length}")


@dataclass
class CandidateLine:
    """A candidate parking line detected by the algorithm.

    Attributes:
        start: (x, y) start point in pixels.
        end: (x, y) end point in pixels.
        confidence: Detection confidence (0.0 to 1.0).
        edge_strength: Average edge magnitude along the line.
        angle_deg: Line angle in degrees (0 = horizontal).
        length: Line length in pixels.
        cluster_id: ID of parallel line cluster (-1 if unclustered).
    """

    start: tuple[float, float]
    end: tuple[float, float]
    confidence: float = 1.0
    edge_strength: float = 0.0
    angle_deg: float = 0.0
    length: float = 0.0
    cluster_id: int = -1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived properties if not set."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        if self.length == 0.0:
            object.__setattr__(self, "length", float(np.sqrt(dx * dx + dy * dy)))
        if self.angle_deg == 0.0 and (dx != 0 or dy != 0):
            object.__setattr__(self, "angle_deg", float(np.degrees(np.arctan2(dy, dx))))

    def to_camera_line(
        self,
        line_id: str,
        image_path: str,
        ptz_position: PTZPosition,
        line_type: str = "parking",
    ) -> CameraLine:
        """Convert to CameraLine with additional metadata.

        Args:
            line_id: Unique identifier for the line.
            image_path: Path to the source image.
            ptz_position: Camera PTZ position.
            line_type: Type of line detected.

        Returns:
            CameraLine instance.
        """
        from poc_homography.calibration.lens_distortion.models import CameraLine

        return CameraLine(
            line_id=line_id,
            image_path=image_path,
            start_pixel=self.start,
            end_pixel=self.end,
            ptz_position=ptz_position,
            line_type=line_type,
            confidence=self.confidence,
        )


class LineDetector:
    """Detects parking spot lines in camera images.

    Uses a combination of edge detection, Hough transforms, and
    filtering heuristics to find straight lines that likely
    correspond to parking spot markings.
    """

    def __init__(self, config: LineDetectionConfig | None = None) -> None:
        """Initialize detector with configuration.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or LineDetectionConfig()

    def detect(self, image: np.ndarray) -> list[CandidateLine]:
        """Detect parking lines in an image.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            List of detected candidate lines, sorted by confidence.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size),
            0,
        )

        # Edge detection
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

        # Compute edge magnitude for confidence scoring
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=self.config.hough_rho,
            theta=self.config.hough_theta,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        if lines is None:
            logger.debug("No lines detected by Hough transform")
            return []

        # Convert to candidate lines
        candidates = []
        for line in lines:
            # HoughLinesP returns shape (N, 1, 4), flatten to get [x1, y1, x2, y2]
            flat_line = line.flatten()
            x1 = int(flat_line[0])
            y1 = int(flat_line[1])
            x2 = int(flat_line[2])
            y2 = int(flat_line[3])

            # Calculate edge strength along the line
            edge_strength = self._calculate_edge_strength(edge_magnitude, (x1, y1), (x2, y2))

            candidate = CandidateLine(
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
                edge_strength=edge_strength,
            )
            candidates.append(candidate)

        logger.debug(f"Detected {len(candidates)} raw candidate lines")

        # Filter by minimum length
        candidates = [c for c in candidates if c.length >= self.config.min_line_length]
        logger.debug(f"After length filter: {len(candidates)} candidates")

        # Cluster parallel lines
        candidates = self._cluster_parallel_lines(candidates)

        # Calculate confidence scores
        candidates = self._calculate_confidence(candidates)

        # Filter by minimum confidence
        candidates = [c for c in candidates if c.confidence >= self.config.min_confidence]
        logger.debug(f"After confidence filter: {len(candidates)} candidates")

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    def detect_from_file(self, image_path: str | Path) -> list[CandidateLine]:
        """Detect parking lines from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detected candidate lines.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return self.detect(image)

    def _calculate_edge_strength(
        self,
        edge_magnitude: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        num_samples: int = 20,
    ) -> float:
        """Calculate average edge magnitude along a line.

        Args:
            edge_magnitude: Edge magnitude image from Sobel operator.
            start: (x, y) start point.
            end: (x, y) end point.
            num_samples: Number of points to sample along the line.

        Returns:
            Average edge magnitude along the line.
        """
        height, width = edge_magnitude.shape[:2]

        # Sample points along the line
        t = np.linspace(0, 1, num_samples)
        xs = (start[0] + t * (end[0] - start[0])).astype(int)
        ys = (start[1] + t * (end[1] - start[1])).astype(int)

        # Clip to image bounds
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)

        # Get edge magnitudes at sampled points
        magnitudes = edge_magnitude[ys, xs]

        return float(np.mean(magnitudes))

    def _cluster_parallel_lines(self, candidates: list[CandidateLine]) -> list[CandidateLine]:
        """Group parallel lines into clusters.

        Lines with similar angles are grouped together. This helps
        identify parking spot patterns where lines are parallel.

        Args:
            candidates: List of candidate lines.

        Returns:
            Candidates with cluster_id assigned.
        """
        if not candidates:
            return candidates

        # Normalize angles to [0, 180) range (direction-independent)
        angles = []
        for c in candidates:
            angle = c.angle_deg % 180
            angles.append(angle)

        # Simple clustering by angle proximity
        cluster_id = 0
        assigned = [False] * len(candidates)
        result = []

        for i, c in enumerate(candidates):
            if assigned[i]:
                continue

            # Start new cluster
            cluster_members = [i]
            assigned[i] = True

            for j in range(i + 1, len(candidates)):
                if assigned[j]:
                    continue

                # Check if angles are close enough
                angle_diff = abs(angles[i] - angles[j])
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff <= self.config.parallel_angle_tolerance:
                    cluster_members.append(j)
                    assigned[j] = True

            # Assign cluster ID to all members
            for idx in cluster_members:
                old = candidates[idx]
                result.append(
                    CandidateLine(
                        start=old.start,
                        end=old.end,
                        confidence=old.confidence,
                        edge_strength=old.edge_strength,
                        angle_deg=old.angle_deg,
                        length=old.length,
                        cluster_id=cluster_id,
                        metadata=old.metadata,
                    )
                )

            cluster_id += 1

        return result

    def _calculate_confidence(self, candidates: list[CandidateLine]) -> list[CandidateLine]:
        """Calculate confidence scores for candidate lines.

        Confidence is based on:
        - Edge strength (higher = more confident)
        - Line length (longer = more confident)
        - Cluster membership (part of parallel group = more confident)

        Args:
            candidates: List of candidate lines.

        Returns:
            Candidates with confidence scores updated.
        """
        if not candidates:
            return candidates

        # Get max values for normalization
        max_edge = max(c.edge_strength for c in candidates) or 1.0
        max_length = max(c.length for c in candidates) or 1.0

        # Count cluster sizes
        cluster_sizes: dict[int, int] = {}
        for c in candidates:
            if c.cluster_id >= 0:
                cluster_sizes[c.cluster_id] = cluster_sizes.get(c.cluster_id, 0) + 1

        max_cluster = max(cluster_sizes.values()) if cluster_sizes else 1

        result = []
        for c in candidates:
            # Normalize factors to [0, 1]
            edge_score = c.edge_strength / max_edge
            length_score = c.length / max_length

            cluster_score = 0.5  # Default for unclustered
            if c.cluster_id >= 0 and c.cluster_id in cluster_sizes:
                cluster_score = cluster_sizes[c.cluster_id] / max_cluster

            # Weighted combination
            confidence = 0.4 * edge_score + 0.3 * length_score + 0.3 * cluster_score

            result.append(
                CandidateLine(
                    start=c.start,
                    end=c.end,
                    confidence=float(confidence),
                    edge_strength=c.edge_strength,
                    angle_deg=c.angle_deg,
                    length=c.length,
                    cluster_id=c.cluster_id,
                    metadata=c.metadata,
                )
            )

        return result

    def visualize(
        self,
        image: np.ndarray,
        candidates: list[CandidateLine],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw detected lines on an image for visualization.

        Args:
            image: Input image (BGR).
            candidates: List of detected candidate lines.
            show_confidence: Whether to show confidence scores.

        Returns:
            Image with lines drawn.
        """
        output = image.copy()

        # Color map for clusters
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for c in candidates:
            # Select color based on cluster
            if c.cluster_id >= 0:
                color = colors[c.cluster_id % len(colors)]
            else:
                color = (128, 128, 128)  # Gray for unclustered

            # Draw line
            pt1 = (int(c.start[0]), int(c.start[1]))
            pt2 = (int(c.end[0]), int(c.end[1]))
            thickness = max(1, int(c.confidence * 3))
            cv2.line(output, pt1, pt2, color, thickness)

            # Draw confidence label
            if show_confidence:
                mid_x = int((c.start[0] + c.end[0]) / 2)
                mid_y = int((c.start[1] + c.end[1]) / 2)
                label = f"{c.confidence:.2f}"
                cv2.putText(
                    output,
                    label,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return output
