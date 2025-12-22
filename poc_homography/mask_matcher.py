"""
Mask matching strategy pattern for computing correlation between binary masks.

This module provides an abstract interface for mask matching algorithms used in
auto-calibration workflows. The primary use case is comparing a cartography mask
(e.g., satellite/map imagery) with a camera frame mask to find the best alignment.

Strategy Pattern:
    The MaskMatcher interface allows different matching algorithms to be swapped
    without changing client code. Currently provides:
    - ECCMaskMatcher: Uses OpenCV's Enhanced Correlation Coefficient (ECC)

Mask Format:
    Input masks should be binary uint8 numpy arrays where:
    - 0 = background pixels
    - 255 = feature pixels (e.g., water, roads, buildings)

Correlation Score:
    All implementations return correlation scores normalized to [0.0, 1.0] where:
    - 1.0 = perfect match
    - 0.5 = moderate correlation
    - 0.0 = no correlation or matching failure

Usage Example:
    matcher = ECCMaskMatcher()
    score = matcher.compute_correlation(camera_mask, cartography_mask)
    if score > 0.7:
        print(f"Good match found with correlation: {score:.3f}")
"""

from abc import ABC, abstractmethod
from typing import Tuple
import cv2
import numpy as np


class MaskMatcher(ABC):
    """Abstract base class for mask matching strategies.

    This interface defines the contract for mask matching algorithms.
    Implementations handle mask preprocessing (resizing, type conversion)
    internally and return normalized correlation scores.

    All methods should be stateless - they can be called multiple times
    with different mask pairs without side effects.
    """

    @abstractmethod
    def compute_correlation(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute correlation score between two binary masks.

        This method preprocesses the masks as needed (resize, type conversion)
        and computes a correlation score indicating how well they match.

        Args:
            mask1: First binary mask as uint8 numpy array (0=background, 255=features)
                Shape can be (height, width) or (height, width, 1)
            mask2: Second binary mask as uint8 numpy array (0=background, 255=features)
                Shape can be (height, width) or (height, width, 1)

        Returns:
            float: Correlation score in range [0.0, 1.0] where:
                - 1.0 = perfect match
                - 0.0 = no correlation or matching failure

        Raises:
            ValueError: If masks are not valid binary uint8 arrays
            TypeError: If mask data types are incorrect

        Note:
            Masks can have different dimensions - implementations should handle
            resizing internally to a common size for comparison.
        """
        pass


class ECCMaskMatcher(MaskMatcher):
    """Enhanced Correlation Coefficient (ECC) based mask matcher.

    This implementation uses OpenCV's findTransformECC() algorithm to compute
    correlation between masks. ECC iteratively refines a geometric transformation
    (rotation, translation, scaling) to maximize correlation.

    Algorithm Details:
        - Motion Model: MOTION_EUCLIDEAN (rotation + translation + scale)
        - Common Size: Masks resized to 512x512 before matching
        - Termination: Max 50 iterations or epsilon 1e-3
        - Error Handling: Returns 0.0 on cv2.error exceptions

    The ECC algorithm returns correlation values that can range from approximately
    -1 to 1, where:
        - 1.0 = perfect positive correlation
        - 0.0 = no correlation
        - -1.0 = perfect negative correlation (inverted)

    For mask matching, negative correlations are meaningless, so we clamp the
    result to [0.0, 1.0] range.

    Performance:
        - Typical runtime: 50-200ms for 512x512 masks
        - GPU acceleration: Not available (OpenCV CPU implementation)

    References:
        Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment
        using enhanced correlation coefficient maximization. IEEE TPAMI, 30(10).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        max_iterations: int = 50,
        epsilon: float = 1e-3
    ):
        """Initialize ECC mask matcher with algorithm parameters.

        Args:
            target_size: Common size to resize masks before matching.
                Default (512, 512) balances accuracy and performance.
            max_iterations: Maximum ECC iterations before termination.
                Default 50 is sufficient for most mask alignments.
            epsilon: Convergence threshold for ECC algorithm.
                Default 1e-3 provides good accuracy/speed tradeoff.
        """
        self.target_size = target_size
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        # Define motion model and termination criteria
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.max_iterations,
            self.epsilon
        )

    def compute_correlation(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute ECC correlation between two binary masks.

        Preprocesses masks (resize, normalize) and uses OpenCV's findTransformECC
        to compute correlation. Handles exceptions gracefully by returning 0.0.

        Args:
            mask1: First binary mask as uint8 array (0=background, 255=features)
            mask2: Second binary mask as uint8 array (0=background, 255=features)

        Returns:
            float: Normalized correlation score in [0.0, 1.0]
                Returns 0.0 if masks are invalid or ECC fails

        Raises:
            ValueError: If masks are empty or have invalid dimensions
            TypeError: If mask data types are not numpy arrays

        Implementation Notes:
            1. Validates input masks
            2. Resizes to common size (default 512x512)
            3. Converts to float32 and normalizes to [0.0, 1.0]
            4. Initializes identity warp matrix
            5. Runs ECC algorithm to find optimal alignment
            6. Clamps correlation to [0.0, 1.0] (removes negative correlations)
        """
        # Validate inputs
        if not isinstance(mask1, np.ndarray) or not isinstance(mask2, np.ndarray):
            raise TypeError("Masks must be numpy arrays")

        if mask1.size == 0 or mask2.size == 0:
            raise ValueError("Masks cannot be empty")

        if len(mask1.shape) not in [2, 3] or len(mask2.shape) not in [2, 3]:
            raise ValueError(
                f"Masks must be 2D or 3D arrays, got shapes {mask1.shape} and {mask2.shape}"
            )

        # Convert to grayscale if needed (take first channel if 3D)
        if len(mask1.shape) == 3:
            mask1 = mask1[:, :, 0]
        if len(mask2.shape) == 3:
            mask2 = mask2[:, :, 0]

        # Resize to common size
        m1_resized = cv2.resize(mask1, self.target_size)
        m2_resized = cv2.resize(mask2, self.target_size)

        # Convert to float32 and normalize to [0.0, 1.0]
        m1_float = m1_resized.astype(np.float32) / 255.0
        m2_float = m2_resized.astype(np.float32) / 255.0

        # Initialize warp matrix (2x3 for Euclidean transformation)
        # Start with identity transformation
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # Compute ECC alignment and correlation
            # Returns: (correlation_coefficient, warp_matrix)
            correlation, _ = cv2.findTransformECC(
                m1_float,
                m2_float,
                warp_matrix,
                self.warp_mode,
                self.criteria
            )

            # Clamp to [0.0, 1.0] range
            # Negative correlations are meaningless for mask matching
            # (would indicate inverted masks)
            correlation = max(0.0, min(1.0, correlation))

            return float(correlation)

        except cv2.error as e:
            # ECC can fail if:
            # - Masks are too dissimilar (no convergence)
            # - Masks are uniform (no features to align)
            # - Numerical instability in optimization
            # In all cases, return 0.0 to indicate no correlation
            return 0.0
