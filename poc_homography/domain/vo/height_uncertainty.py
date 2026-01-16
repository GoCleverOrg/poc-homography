"""Height uncertainty value object for camera height confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.types import Meters


@dataclass(frozen=True)
class HeightUncertainty:
    """Height uncertainty bounds for error propagation.

    Represents a confidence interval for camera height, typically from
    height calibration. Used for propagating uncertainty to world
    coordinate projections.

    Attributes:
        lower: Lower bound of height confidence interval in meters.
        upper: Upper bound of height confidence interval in meters.
    """

    lower: Meters
    upper: Meters

    def __post_init__(self) -> None:
        """Validate height uncertainty bounds."""
        if self.lower <= 0:
            raise ValueError(f"Lower bound must be positive, got {self.lower}")
        if self.upper <= 0:
            raise ValueError(f"Upper bound must be positive, got {self.upper}")
        if self.lower > self.upper:
            raise ValueError(f"Lower bound ({self.lower}) cannot exceed upper bound ({self.upper})")

    # --- Semantic Methods ---

    @property
    def range(self) -> Meters:
        """The uncertainty range (upper - lower)."""
        return Meters(float(self.upper) - float(self.lower))

    @property
    def midpoint(self) -> Meters:
        """The midpoint of the confidence interval."""
        return Meters((float(self.lower) + float(self.upper)) / 2)

    def contains(self, height: Meters) -> bool:
        """Check if a height value falls within the uncertainty bounds."""
        return self.lower <= height <= self.upper

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeightUncertainty:
        """Create HeightUncertainty from dictionary."""
        return cls(
            lower=Meters(data["lower"]),
            upper=Meters(data["upper"]),
        )

    # --- Factory Methods ---

    @classmethod
    def symmetric(cls, center: Meters, margin: Meters) -> HeightUncertainty:
        """Create symmetric uncertainty around a center value.

        Args:
            center: The center height value.
            margin: The uncertainty margin (+/-).

        Returns:
            HeightUncertainty with bounds [center - margin, center + margin].

        Raises:
            ValueError: If resulting lower bound is not positive.
        """
        lower = Meters(float(center) - float(margin))
        upper = Meters(float(center) + float(margin))
        return cls(lower=lower, upper=upper)
