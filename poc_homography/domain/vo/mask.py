"""Binary mask value object for image segmentation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Mask:
    """Binary mask representing segmented regions in an image.

    This value object encapsulates a binary mask and provides domain behavior
    for coverage calculation and mask creation from polygon coordinates.

    Note: This is a non-frozen dataclass because numpy arrays are not hashable.
    While not strictly immutable, the mask data should be treated as read-only
    after creation.

    Attributes:
        data: Binary mask as numpy array (dtype=uint8, values 0 or 255).
        height: Height of the mask in pixels.
        width: Width of the mask in pixels.
    """

    data: NDArray[np.uint8]
    height: int
    width: int

    @property
    def coverage(self) -> float:
        """Calculate percentage of image covered by the mask.

        Returns:
            Coverage percentage (0.0-100.0).
        """
        total_pixels = self.height * self.width
        if total_pixels == 0:
            return 0.0
        white_pixels = cv2.countNonZero(self.data)
        return (white_pixels / total_pixels) * 100.0

    @classmethod
    def from_polygons(
        cls,
        polygons: list[list[tuple[int, int]]],
        shape: tuple[int, int],
    ) -> Mask:
        """Create a mask from polygon coordinates.

        Args:
            polygons: List of polygons, where each polygon is a list of (x, y) points.
            shape: Tuple of (height, width) for the mask dimensions.

        Returns:
            Mask with the polygons filled in.
        """
        height, width = shape
        mask_data = np.zeros((height, width), dtype=np.uint8)

        for polygon in polygons:
            if len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask_data, [pts], (255,))

        return cls(data=mask_data, height=height, width=width)

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> Mask:
        """Create an empty mask (all zeros).

        Args:
            shape: Tuple of (height, width) for the mask dimensions.

        Returns:
            Empty mask with no regions filled.
        """
        height, width = shape
        return cls(
            data=np.zeros((height, width), dtype=np.uint8),
            height=height,
            width=width,
        )
