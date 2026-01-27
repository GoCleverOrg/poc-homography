"""SAM3-based image masking for ground marking detection.

This module provides functionality to use Roboflow's SAM3 API to segment
ground markings (parking lines, road markings) from camera images. The
resulting masks can be used to filter line detection to only consider
actual ground markings rather than spurious edges.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class SAM3Config:
    """Configuration for SAM3 segmentation.

    Attributes:
        api_key: Roboflow API key. If None, reads from ROBOFLOW_API_KEY env var.
        prompt: Text prompt for segmentation.
        timeout: API request timeout in seconds.
        min_confidence: Minimum confidence threshold for predictions.
    """

    api_key: str | None = None
    prompt: str = "white lines on ground"
    timeout: int = 120
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration and resolve API key."""
        if self.api_key is None:
            self.api_key = os.environ.get("ROBOFLOW_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "SAM3 API key required. Set ROBOFLOW_API_KEY environment variable "
                "or pass api_key to SAM3Config."
            )


@dataclass
class SAM3Result:
    """Result of SAM3 segmentation.

    Attributes:
        mask: Binary mask (255 for detected regions, 0 otherwise).
        polygons: List of polygon coordinates for each detection.
        confidences: Confidence scores for each detection.
        coverage_percent: Percentage of image covered by mask.
        prompt: The prompt used for segmentation.
        error: Error message if segmentation failed, None otherwise.
    """

    mask: np.ndarray
    polygons: list[list[tuple[int, int]]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    coverage_percent: float = 0.0
    prompt: str = ""
    error: str | None = None


class SAM3Segmenter:
    """Segments ground markings using Roboflow SAM3 API."""

    API_URL = "https://serverless.roboflow.com/sam3/concept_segment"

    def __init__(self, config: SAM3Config | None = None) -> None:
        """Initialize segmenter with configuration.

        Args:
            config: SAM3 configuration. Uses defaults if None.
        """
        self.config = config or SAM3Config()

    def segment(self, image: np.ndarray) -> SAM3Result:
        """Segment ground markings in an image.

        Args:
            image: Input image as numpy array (BGR).

        Returns:
            SAM3Result containing the mask and detection details.
        """
        height, width = image.shape[:2]

        # Encode image as base64
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            return SAM3Result(
                mask=np.zeros((height, width), dtype=np.uint8),
                error="Failed to encode image",
            )

        image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # Call API
        api_url = f"{self.API_URL}?api_key={self.config.api_key}"
        request_body = {
            "format": "polygon",
            "image": {"type": "base64", "value": image_base64},
            "prompts": [{"type": "text", "text": self.config.prompt}],
        }

        try:
            logger.debug(f"Calling SAM3 API with prompt: '{self.config.prompt}'")
            response = requests.post(
                api_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                error_msg = f"SAM3 API error: {response.status_code}"
                logger.error(error_msg)
                return SAM3Result(
                    mask=np.zeros((height, width), dtype=np.uint8),
                    error=error_msg,
                )

            api_response = response.json()

        except requests.exceptions.Timeout:
            return SAM3Result(
                mask=np.zeros((height, width), dtype=np.uint8),
                error="SAM3 API timeout",
            )
        except requests.exceptions.RequestException as e:
            return SAM3Result(
                mask=np.zeros((height, width), dtype=np.uint8),
                error=f"SAM3 API request failed: {e}",
            )

        # Parse response and create mask
        return self._parse_response(api_response, height, width)

    def segment_from_file(self, image_path: str | Path) -> SAM3Result:
        """Segment ground markings from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            SAM3Result containing the mask and detection details.

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

        return self.segment(image)

    def _parse_response(
        self, api_response: dict[str, Any], height: int, width: int
    ) -> SAM3Result:
        """Parse SAM3 API response into SAM3Result.

        Args:
            api_response: Raw API response.
            height: Image height.
            width: Image width.

        Returns:
            Parsed SAM3Result.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons: list[list[tuple[int, int]]] = []
        confidences: list[float] = []

        prompt_results = api_response.get("prompt_results", [])

        for prompt_result in prompt_results:
            predictions = prompt_result.get("predictions", [])

            for prediction in predictions:
                confidence = prediction.get("confidence", 0.0)

                # Skip low-confidence predictions
                if confidence < self.config.min_confidence:
                    continue

                confidences.append(confidence)
                masks = prediction.get("masks", [])

                for polygon_data in masks:
                    if isinstance(polygon_data, list) and len(polygon_data) >= 3:
                        # Convert to integer points
                        pts = [
                            (int(pt[0]), int(pt[1]))
                            for pt in polygon_data
                            if len(pt) >= 2
                        ]

                        if len(pts) >= 3:
                            polygons.append(pts)
                            pts_array = np.array(pts, dtype=np.int32)
                            cv2.fillPoly(mask, [pts_array], (255,))

        # Calculate coverage
        total_pixels = height * width
        white_pixels = cv2.countNonZero(mask)
        coverage = (white_pixels / total_pixels) * 100

        logger.debug(
            f"SAM3 segmentation: {len(polygons)} polygons, "
            f"{len(confidences)} detections, {coverage:.2f}% coverage"
        )

        return SAM3Result(
            mask=mask,
            polygons=polygons,
            confidences=confidences,
            coverage_percent=coverage,
            prompt=self.config.prompt,
        )


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to image, keeping only masked regions.

    Args:
        image: Input image (BGR or grayscale).
        mask: Binary mask (255 for regions to keep).

    Returns:
        Masked image with non-masked regions set to black.
    """
    if len(image.shape) == 3:
        # Color image - expand mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(image, mask_3ch)
    else:
        # Grayscale image
        return cv2.bitwise_and(image, mask)


def dilate_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Dilate mask to include nearby edges.

    This can help include line edges that may be slightly outside
    the detected polygon boundaries.

    Args:
        mask: Binary mask.
        kernel_size: Size of dilation kernel.
        iterations: Number of dilation iterations.

    Returns:
        Dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)
