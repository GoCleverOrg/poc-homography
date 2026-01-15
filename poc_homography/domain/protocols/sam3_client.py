"""Protocol for SAM3 segmentation API client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poc_homography.types import Unitless  # noqa: TC001 - used at runtime in dataclass


@dataclass(frozen=True)
class Sam3Detection:
    """A single detection from the SAM3 API response.

    This DTO represents the structured data returned by the SAM3 API
    for a single detected object/region.

    Attributes:
        confidence: Confidence score of the detection (0.0-1.0).
        polygons: Tuple of polygon coordinates. Each polygon is a tuple
            of (x, y) coordinate tuples.
    """

    confidence: Unitless
    polygons: tuple[tuple[tuple[int, int], ...], ...]


class Sam3Client(Protocol):
    """Protocol for SAM3 segmentation API clients.

    This protocol defines the interface for clients that call the SAM3
    concept segmentation API. Infrastructure implementations provide
    the actual HTTP communication.
    """

    def segment(self, image_base64: str, prompt: str) -> list[Sam3Detection]:
        """Segment an image using the given text prompt.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt describing what to segment (e.g., "road markings").

        Returns:
            List of detections, each containing confidence and polygon coordinates.

        Raises:
            Sam3ApiError: If the API call fails.
        """
        ...


class Sam3ApiError(Exception):
    """Error from the SAM3 API."""

    pass
