"""Photo value object representing an image file with dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            relative_to: If provided, store path relative to this directory.

        Returns:
            Dictionary with path (and optionally relative path).
        """
        if relative_to:
            try:
                path_str = str(self.path.relative_to(relative_to))
            except ValueError:
                # Path is not relative to base, use absolute
                path_str = str(self.path)
        else:
            path_str = str(self.path)

        return {"path": path_str}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, width: Pixels, height: Pixels, base_path: Path | None = None
    ) -> Photo:
        """Create Photo from dictionary.

        Args:
            data: Dictionary with 'path' key.
            width: Image width (must be provided by caller after loading image).
            height: Image height (must be provided by caller after loading image).
            base_path: Base directory for resolving relative paths.

        Returns:
            Photo instance with resolved path and dimensions.
        """
        path = Path(data["path"])
        if not path.is_absolute() and base_path:
            path = base_path / data["path"]

        return cls(path=path, width=width, height=height)
