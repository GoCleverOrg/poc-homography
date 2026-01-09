"""
MaskMatcher interface and implementations for mask-based alignment scoring.

This module provides an abstract base class for mask matching algorithms
and concrete implementations including distance transform-based edge alignment.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class MaskMatcher(ABC):
    """
    Abstract base class for mask matching algorithms.

    Mask matchers compute a correlation score between two binary masks,
    typically used to assess alignment quality in image registration tasks.
    """

    @abstractmethod
    def compute_correlation(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute correlation score between two binary masks.

        Args:
            mask1: Reference mask (binary, uint8, 0/255 values)
            mask2: Camera mask to compare (binary, uint8, 0/255 values)

        Returns:
            Correlation score in range [0.0, 1.0] where:
            - 1.0 indicates perfect alignment
            - 0.0 indicates poor alignment
        """
        pass


class DistanceTransformMatcher(MaskMatcher):
    """
    Distance transform-based edge alignment matcher.

    Uses distance transform on reference mask edges to create smooth gradients
    for evaluating alignment with camera mask edges. This approach is robust
    for sparse mask alignment where edges may not overlap exactly.

    Algorithm:
    1. Extract edges from reference mask using Canny edge detection
    2. Compute distance transform from reference edges
    3. Extract edges from camera mask using Canny edge detection
    4. Compute mean distance of camera edges in reference distance field
    5. Normalize distance error to correlation-like score [0.0, 1.0]

    Mathematical formulation:
        E_edge = (1/N) × Σᵢ DT_ref(edge_pixel_i)
        score = max(0.0, 1.0 - E_edge / max_distance)

    where max_distance is the image diagonal.
    """

    def __init__(
        self,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        distance_type: int = cv2.DIST_L2,
        mask_size: int = 5,
        target_size: tuple[int, int] = (512, 512),
    ):
        """
        Initialize DistanceTransformMatcher.

        Args:
            canny_threshold1: Lower threshold for Canny edge detector (default: 50)
            canny_threshold2: Upper threshold for Canny edge detector (default: 150)
            distance_type: Distance transform type (default: cv2.DIST_L2)
            mask_size: Mask size for distance transform (default: 5)
            target_size: Target size (width, height) for mask resizing (default: (512, 512))
        """
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.distance_type = distance_type
        self.mask_size = mask_size
        self.target_size = target_size

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask: convert to grayscale if needed and resize to target size.

        Args:
            mask: Input mask (may be 2D or 3D, any size)

        Returns:
            Preprocessed 2D grayscale mask at target size
        """
        # Convert 3-channel to grayscale if needed
        if len(mask.shape) == 3:
            if mask.shape[2] == 3:
                # Convert RGB to grayscale
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            elif mask.shape[2] == 1:
                # Remove unnecessary channel dimension
                mask = mask[:, :, 0]

        # Resize to target size if needed
        if mask.shape[:2] != (self.target_size[1], self.target_size[0]):
            # OpenCV resize expects (width, height)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _extract_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract edges from mask using Canny edge detection.

        Args:
            mask: Binary mask (uint8, 0/255 values)

        Returns:
            Binary edge map (uint8, 0/255 values)
        """
        edges = cv2.Canny(mask, self.canny_threshold1, self.canny_threshold2)
        return edges

    def _compute_distance_transform(self, edges: np.ndarray) -> np.ndarray:
        """
        Compute distance transform from edge map.

        Args:
            edges: Binary edge map (uint8, 0/255 values)

        Returns:
            Distance transform (float32, distance values)
        """
        # Invert edges: distance transform computes distance to nearest zero pixel
        # We want distance to nearest edge (255 pixel), so invert
        inverted_edges = cv2.bitwise_not(edges)

        # Compute distance transform
        dist_transform = cv2.distanceTransform(inverted_edges, self.distance_type, self.mask_size)

        return dist_transform

    def compute_correlation(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute correlation score using distance transform edge alignment.

        Args:
            mask1: Reference mask (binary, uint8, 0/255 values)
            mask2: Camera mask (binary, uint8, 0/255 values)

        Returns:
            Correlation score in [0.0, 1.0] where 1.0 = perfect alignment
        """
        # Preprocess masks
        mask1_processed = self._preprocess_mask(mask1)
        mask2_processed = self._preprocess_mask(mask2)

        # Extract edges from both masks
        edges_ref = self._extract_edges(mask1_processed)
        edges_cam = self._extract_edges(mask2_processed)

        # Check if either mask has no edges
        if np.sum(edges_ref) == 0 or np.sum(edges_cam) == 0:
            return 0.0

        # Compute distance transform on reference edges
        dist_transform = self._compute_distance_transform(edges_ref)

        # Find camera edge pixel locations
        edge_pixels = np.where(edges_cam > 0)

        # If no edge pixels found, return 0.0
        if len(edge_pixels[0]) == 0:
            return 0.0

        # Compute mean distance of camera edges in reference distance field
        distances = dist_transform[edge_pixels]
        mean_distance = np.mean(distances)

        # Compute max distance for normalization (image diagonal)
        height, width = mask1_processed.shape
        max_distance = np.sqrt(height**2 + width**2)

        # Normalize to correlation-like score [0.0, 1.0]
        # Lower distance = better alignment = higher score
        score = max(0.0, 1.0 - mean_distance / max_distance)

        # Clamp to valid range
        score = np.clip(score, 0.0, 1.0)

        return float(score)
