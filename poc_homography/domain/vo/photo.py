"""Photo value object representing an image file with dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poc_homography.types import Pixels


@dataclass(frozen=True)
class Photo:
    """An image file with its dimensions.

    This VO represents an image that has been loaded/inspected, with its
    dimensions stored. The actual image loading and dimension extraction
    is done by infrastructure services when creating Map entities.

    Attributes:
        path: Path to the image file (PNG format).
        width: Image width in pixels.
        height: Image height in pixels.
    """

    path: Path
    width: Pixels
    height: Pixels

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Photo:
        """Create Photo from dictionary."""
        return cls(
            path=Path(data["path"]),
            width=Pixels(data["width"]),
            height=Pixels(data["height"]),
        )
