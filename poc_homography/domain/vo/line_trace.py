"""LineTrace value object — a single N-point pixel trace of a line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LineTrace:
    """Single line observation with N-point pixel trace."""

    line_id: str
    points: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"line_id": self.line_id, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineTrace:
        """Create LineTrace from dictionary."""
        return cls(
            line_id=data["line_id"],
            points=tuple(tuple(p) for p in data["points"]),
        )
