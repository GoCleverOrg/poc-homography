"""Protocol for CapturedFrame repositories.

Extends the base Repo protocol with frame-specific operations needed
by application services.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime in Protocol
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.domain.entities.captured_frame import CapturedFrame


class CapturedFrameRepository(Protocol):
    """Repository protocol for captured frame persistence.

    Application services depend on this protocol rather than on
    concrete infrastructure implementations.
    """

    def save(self, entity: CapturedFrame) -> None:
        """Save a captured frame entity."""
        ...

    def get(self, entity_id: str) -> CapturedFrame | None:
        """Retrieve a captured frame by ID."""
        ...

    def image_dir_for(self, map_id: str, camera_name: str) -> Path:
        """Return the directory where images are stored for a camera."""
        ...
