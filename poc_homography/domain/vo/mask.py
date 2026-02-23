"""Binary mask value object for image segmentation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.domain.vo.image_dimensions import ImageDimensions

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
        dimensions: Dimensions of the mask (width and height in pixels).
    """

    data: NDArray[np.uint8]
    dimensions: ImageDimensions

    @property
    def width(self) -> int:
        """Width of the mask in pixels (backward-compatible property)."""
        return self.dimensions.width

    @property
    def height(self) -> int:
        """Height of the mask in pixels (backward-compatible property)."""
        return self.dimensions.height

    @property
    def coverage(self) -> float:
        """Calculate percentage of image covered by the mask.

        Returns:
            Coverage percentage (0.0-100.0).
        """
        total_pixels = self.dimensions.area
        if total_pixels == 0:
            return 0.0
        white_pixels = int(np.count_nonzero(self.data))
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
        dimensions = ImageDimensions.create(width=width, height=height)
        mask_data = np.zeros((height, width), dtype=np.uint8)

        for polygon in polygons:
            if len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                _fill_polygon(mask_data, pts)

        return cls(data=mask_data, dimensions=dimensions)


def _fill_polygon(mask: np.ndarray, pts: np.ndarray) -> None:
    """Fill a convex polygon on a uint8 mask using scanline (pure numpy, no cv2).

    Uses matplotlib-free scanline rasterisation.  ``pts`` is an Nx2 int32
    array of (x, y) vertices.  The filled region is set to 255.
    """
    if len(pts) < 3:
        return
    ys = pts[:, 1]
    y_min, y_max = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    y_min = max(y_min, 0)
    y_max = min(y_max, h - 1)

    for y in range(y_min, y_max + 1):
        intersections: list[float] = []
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if y0 == y1:
                continue
            if (y < min(y0, y1)) or (y >= max(y0, y1)):
                continue
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            intersections.append(x_cross)
        intersections.sort()
        for j in range(0, len(intersections) - 1, 2):
            x_start = max(int(intersections[j]), 0)
            x_end = min(int(intersections[j + 1]), w - 1)
            if x_start <= x_end:
                mask[y, x_start : x_end + 1] = 255
