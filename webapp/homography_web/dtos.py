"""Pure data transfer objects for annotation data.

No Django dependencies — safe to import from services and tests without
a configured Django environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PointAnnotationDTO:
    """Typed representation of a point annotation (GCP-to-pixel mapping)."""

    gcp_id: str
    pixel_x: float
    pixel_y: float

    def to_dict(self) -> dict[str, str | float]:
        """Serialize to a plain dict (for JSON responses and legacy APIs)."""
        return {"gcp_id": self.gcp_id, "pixel_x": self.pixel_x, "pixel_y": self.pixel_y}


@dataclass(frozen=True, slots=True)
class LineAnnotationDTO:
    """Typed representation of a line annotation (line-to-pixel mapping).

    Attributes:
        line_id: Identifier of the line in the line registry.
        start_pixel_x: Camera pixel x of the line start.
        start_pixel_y: Camera pixel y of the line start.
        end_pixel_x: Camera pixel x of the line end.
        end_pixel_y: Camera pixel y of the line end.
        points: Optional polyline vertices as ``[[x, y], ...]`` for n-point lines.
    """

    line_id: str
    start_pixel_x: float
    start_pixel_y: float
    end_pixel_x: float
    end_pixel_y: float
    points: list[list[float]] | None = field(default=None)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON responses and legacy APIs)."""
        d: dict = {
            "line_id": self.line_id,
            "start_pixel_x": self.start_pixel_x,
            "start_pixel_y": self.start_pixel_y,
            "end_pixel_x": self.end_pixel_x,
            "end_pixel_y": self.end_pixel_y,
        }
        if self.points is not None:
            d["points"] = self.points
        return d
