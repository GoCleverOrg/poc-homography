"""Lens distortion value object for optical distortion coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from poc_homography.types import Unitless


@dataclass(frozen=True)
class LensDistortion:
    """Lens distortion coefficients (OpenCV 5-parameter model).

    Describes radial and tangential distortion:
    - Radial (barrel/pincushion): k1, k2, k3
    - Tangential (decentering): p1, p2

    Positive k1 = barrel distortion (edges curve outward)
    Negative k1 = pincushion distortion (edges curve inward)

    Attributes:
        k1: 1st radial distortion coefficient (most significant).
        k2: 2nd radial distortion coefficient.
        p1: 1st tangential distortion coefficient.
        p2: 2nd tangential distortion coefficient.
        k3: 3rd radial distortion coefficient (usually 0).
    """

    k1: Unitless = Unitless(0.0)  # noqa: RUF009
    k2: Unitless = Unitless(0.0)  # noqa: RUF009
    p1: Unitless = Unitless(0.0)  # noqa: RUF009
    p2: Unitless = Unitless(0.0)  # noqa: RUF009
    k3: Unitless = Unitless(0.0)  # noqa: RUF009

    # --- Semantic Methods ---

    @property
    def has_distortion(self) -> bool:
        """True if any distortion coefficient is non-zero."""
        return not self.is_zero()

    def is_zero(self, atol: float = 1e-12) -> bool:
        """True if all coefficients are effectively zero (no distortion)."""
        return all(abs(c) < atol for c in (self.k1, self.k2, self.p1, self.p2, self.k3))

    # --- Numpy Interop ---

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in OpenCV format [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @classmethod
    def from_array(cls, coeffs: np.ndarray) -> LensDistortion:
        """Create from numpy array [k1, k2, p1, p2, k3].

        Args:
            coeffs: Array of 5 distortion coefficients.

        Raises:
            ValueError: If array does not have exactly 5 elements.
        """
        if len(coeffs) != 5:
            raise ValueError(f"Expected 5 distortion coefficients, got {len(coeffs)}")
        return cls(
            k1=Unitless(float(coeffs[0])),
            k2=Unitless(float(coeffs[1])),
            p1=Unitless(float(coeffs[2])),
            p2=Unitless(float(coeffs[3])),
            k3=Unitless(float(coeffs[4])),
        )

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "k3": float(self.k3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LensDistortion:
        """Create LensDistortion from dictionary (k3 defaults to 0.0 for backward compat)."""
        return cls(
            k1=Unitless(data.get("k1", 0.0)),
            k2=Unitless(data.get("k2", 0.0)),
            p1=Unitless(data.get("p1", 0.0)),
            p2=Unitless(data.get("p2", 0.0)),
            k3=Unitless(data.get("k3", 0.0)),
        )

    # --- Factory Methods ---

    @classmethod
    def radial_only(
        cls,
        k1: Unitless,
        k2: Unitless = Unitless(0.0),
    ) -> LensDistortion:
        """Create with only radial distortion (common case)."""
        return cls(k1=k1, k2=k2)
