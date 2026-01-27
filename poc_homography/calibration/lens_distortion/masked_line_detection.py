"""Masked line detection using SAM3 segmentation.

This module combines SAM3 ground marking segmentation with Hough line
detection to dramatically reduce false positive line detections.
Instead of detecting lines across the entire image, lines are only
detected within regions identified as ground markings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc_homography.calibration.lens_distortion.line_detection import (
    CandidateLine,
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.calibration.lens_distortion.sam3_masking import (
    SAM3Config,
    SAM3Result,
    SAM3Segmenter,
    apply_mask,
    dilate_mask,
)

logger = logging.getLogger(__name__)


@dataclass
class MaskedLineDetectionConfig:
    """Configuration for masked line detection.

    Attributes:
        sam3_config: Configuration for SAM3 segmentation.
        line_config: Configuration for line detection.
        dilate_mask: Whether to dilate the mask before line detection.
        dilation_kernel_size: Size of dilation kernel if dilating.
        dilation_iterations: Number of dilation iterations.
    """

    sam3_config: SAM3Config | None = None
    line_config: LineDetectionConfig | None = None
    dilate_mask: bool = True
    dilation_kernel_size: int = 5
    dilation_iterations: int = 2


@dataclass
class MaskedLineDetectionResult:
    """Result of masked line detection.

    Attributes:
        lines: Detected candidate lines.
        sam3_result: SAM3 segmentation result.
        masked_image: Image with SAM3 mask applied.
        original_line_count: Number of lines detected without masking (for comparison).
    """

    lines: list[CandidateLine]
    sam3_result: SAM3Result
    masked_image: np.ndarray
    original_line_count: int = 0


class MaskedLineDetector:
    """Detects lines using SAM3 masking to filter to ground markings only.

    This detector first uses SAM3 to segment ground markings (parking lines,
    road markings) from the image, then runs Hough line detection only on
    the masked regions. This typically reduces detected lines by 90%+ while
    keeping the relevant parking/road markings.

    Example:
        >>> detector = MaskedLineDetector()
        >>> result = detector.detect_from_file("parking_lot.jpg")
        >>> print(f"Detected {len(result.lines)} lines (vs {result.original_line_count} without mask)")
    """

    def __init__(self, config: MaskedLineDetectionConfig | None = None) -> None:
        """Initialize detector with configuration.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or MaskedLineDetectionConfig()
        self.sam3_segmenter = SAM3Segmenter(self.config.sam3_config)
        self.line_detector = LineDetector(self.config.line_config)

    def detect(
        self, image: np.ndarray, include_original_count: bool = False
    ) -> MaskedLineDetectionResult:
        """Detect lines using SAM3 masking.

        Args:
            image: Input image as numpy array (BGR).
            include_original_count: If True, also detect lines on unmasked
                image for comparison (slower).

        Returns:
            MaskedLineDetectionResult with detected lines and metadata.
        """
        # Step 1: SAM3 segmentation
        logger.info("Running SAM3 segmentation...")
        sam3_result = self.sam3_segmenter.segment(image)

        if sam3_result.error:
            logger.error(f"SAM3 segmentation failed: {sam3_result.error}")
            # Fall back to unmasked detection
            lines = self.line_detector.detect(image)
            return MaskedLineDetectionResult(
                lines=lines,
                sam3_result=sam3_result,
                masked_image=image.copy(),
                original_line_count=len(lines),
            )

        logger.info(
            f"SAM3 found {len(sam3_result.polygons)} polygons, "
            f"coverage: {sam3_result.coverage_percent:.2f}%"
        )

        # Step 2: Optionally dilate mask
        mask = sam3_result.mask
        if self.config.dilate_mask and sam3_result.coverage_percent > 0:
            mask = dilate_mask(
                mask,
                kernel_size=self.config.dilation_kernel_size,
                iterations=self.config.dilation_iterations,
            )

        # Step 3: Apply mask to image
        masked_image = apply_mask(image, mask)

        # Step 4: Detect lines on masked image
        logger.info("Running line detection on masked image...")
        lines = self.line_detector.detect(masked_image)
        logger.info(f"Detected {len(lines)} lines on masked image")

        # Step 5: Optionally count lines on original for comparison
        original_count = 0
        if include_original_count:
            logger.info("Running line detection on original image for comparison...")
            original_lines = self.line_detector.detect(image)
            original_count = len(original_lines)
            logger.info(
                f"Original: {original_count} lines, Masked: {len(lines)} lines "
                f"({(1 - len(lines) / max(original_count, 1)) * 100:.1f}% reduction)"
            )

        return MaskedLineDetectionResult(
            lines=lines,
            sam3_result=sam3_result,
            masked_image=masked_image,
            original_line_count=original_count,
        )

    def detect_from_file(
        self, image_path: str | Path, include_original_count: bool = False
    ) -> MaskedLineDetectionResult:
        """Detect lines from an image file using SAM3 masking.

        Args:
            image_path: Path to the image file.
            include_original_count: If True, also detect lines on unmasked
                image for comparison.

        Returns:
            MaskedLineDetectionResult with detected lines and metadata.

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

        return self.detect(image, include_original_count)

    def visualize(
        self,
        image: np.ndarray,
        result: MaskedLineDetectionResult,
        show_mask_overlay: bool = True,
    ) -> np.ndarray:
        """Visualize detection results.

        Args:
            image: Original input image.
            result: Detection result to visualize.
            show_mask_overlay: Whether to show SAM3 mask as overlay.

        Returns:
            Visualization image with lines and optional mask overlay.
        """
        output = image.copy()

        # Optionally overlay mask
        if show_mask_overlay and result.sam3_result.mask is not None:
            mask = result.sam3_result.mask
            mask_colored = np.zeros_like(output)
            mask_colored[mask > 0] = [0, 255, 0]  # Green overlay
            output = cv2.addWeighted(output, 0.7, mask_colored, 0.3, 0)

        # Draw lines
        output = self.line_detector.visualize(output, result.lines, show_confidence=True)

        # Add info text
        info_text = (
            f"Lines: {len(result.lines)} | "
            f"Coverage: {result.sam3_result.coverage_percent:.1f}%"
        )
        if result.original_line_count > 0:
            reduction = (1 - len(result.lines) / result.original_line_count) * 100
            info_text += f" | Reduction: {reduction:.1f}%"

        cv2.putText(
            output,
            info_text,
            (10, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return output
