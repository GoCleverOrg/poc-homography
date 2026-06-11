"""Presentation DTOs (view models) for frame annotations.

These typed frozen dataclasses replace the raw-dict passthrough returns of the
legacy loaders in ``frame_utils.py``. Each DTO exposes a ``to_dict()`` that
reproduces the exact legacy JSON shape at the serialization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameAnnotation:
    """A point annotation (GCP) in legacy presentation form.

    ``pixel_x``/``pixel_y`` are expected to already be rounded to 1 decimal by
    the loader, matching the prior dict behavior.
    """

    gcp_id: str
    pixel_x: float
    pixel_y: float

    def to_dict(self) -> dict:
        """Return the exact legacy dict shape ``{gcp_id, pixel_x, pixel_y}``."""
        return {
            "gcp_id": self.gcp_id,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
        }


@dataclass(frozen=True)
class LineAnnotationView:
    """A line annotation in legacy presentation form.

    The ``points`` key is included in :meth:`to_dict` only when ``points`` is
    not ``None`` (matching the prior conditional dict behavior for n-point
    polylines).
    """

    line_id: str
    start_pixel_x: float
    start_pixel_y: float
    end_pixel_x: float
    end_pixel_y: float
    points: list[list[float]] | None = None

    def to_dict(self) -> dict:
        """Return the legacy dict; include ``points`` only when not ``None``."""
        entry: dict = {
            "line_id": self.line_id,
            "start_pixel_x": self.start_pixel_x,
            "start_pixel_y": self.start_pixel_y,
            "end_pixel_x": self.end_pixel_x,
            "end_pixel_y": self.end_pixel_y,
        }
        if self.points is not None:
            entry["points"] = self.points
        return entry
