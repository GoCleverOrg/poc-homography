"""Lens distortion value object for optical distortion coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.types import Unitless


@dataclass(frozen=True)
class LensDistortion:
    """Lens distortion coefficients (OpenCV model).

    Describes radial and tangential distortion of a camera lens.
    Uses the standard OpenCV distortion model.

    Attributes:
        k1: Radial distortion coefficient (1st order). Negative = barrel distortion.
        k2: Radial distortion coefficient (2nd order).
        p1: Tangential distortion coefficient.
        p2: Tangential distortion coefficient.
    """

    k1: Unitless = Unitless(0.0)  # noqa: RUF009
    k2: Unitless = Unitless(0.0)  # noqa: RUF009
    p1: Unitless = Unitless(0.0)  # noqa: RUF009
    p2: Unitless = Unitless(0.0)  # noqa: RUF009

    @property
    def has_distortion(self) -> bool:
        """True if any distortion coefficient is non-zero."""
        return self.k1 != 0.0 or self.k2 != 0.0 or self.p1 != 0.0 or self.p2 != 0.0

    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        """Return distortion coefficients as tuple (k1, k2, p1, p2)."""
        return (float(self.k1), float(self.k2), float(self.p1), float(self.p2))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "p1": float(self.p1),
            "p2": float(self.p2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LensDistortion:
        """Create LensDistortion from dictionary."""
        return cls(
            k1=Unitless(data.get("k1", 0.0)),
            k2=Unitless(data.get("k2", 0.0)),
            p1=Unitless(data.get("p1", 0.0)),
            p2=Unitless(data.get("p2", 0.0)),
        )
