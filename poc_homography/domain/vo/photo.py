"""Photo value object representing an image file with dimensions."""

from dataclasses import dataclass
from pathlib import Path

from poc_homography.types import Pixels


@dataclass(frozen=True)
class Photo:
    """An image file with its dimensions.

    This VO represents an image that has been loaded/inspected, with its
    dimensions cached. The actual image loading is done by infrastructure
    (repositories), keeping the domain pure.

    Attributes:
        path: Path to the image file.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    path: Path
    width: Pixels
    height: Pixels
