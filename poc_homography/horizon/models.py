"""Value objects for horizon detection.

Frozen, JSON-round-trippable dataclasses describing the placement of the
horizon line within a PTZ frame (:class:`HorizonEstimate`) and the result of a
one-time tilt→elevation calibration (:class:`CalibrationResult`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

HORIZON_SCHEMA_VERSION = "1"
"""Schema version for the horizon value objects.

Deliberately the literal ``"1"`` and independent of every other
``*_SCHEMA_VERSION`` in the codebase; the horizon VOs evolve on their own.
"""


class FramePlacement(Enum):
    """Where the horizon line falls relative to the image frame.

    The image ``y`` axis points down, so a *larger* row is *lower* in the frame.
    Ground (below the horizon) occupies the lower rows; sky occupies the upper
    rows.
    """

    IN_FRAME = "in_frame"
    """The horizon line crosses the frame; a concrete row is available."""

    ABOVE_FRAME = "above_frame"
    """The horizon is above the top edge — the whole frame is ground."""

    BELOW_FRAME = "below_frame"
    """The horizon is below the bottom edge — the whole frame is sky."""


@dataclass(frozen=True)
class HorizonEstimate:
    """Estimated horizon placement within a single frame.

    Attributes:
        schema_version: VO schema version (always ``"1"``).
        placement: Whether the horizon is in/above/below the frame.
        image_height: Frame height in pixels (interprets ``row``/fractions).
        row: Horizon image row in pixels, or ``None`` when not ``IN_FRAME``.
        ground_fraction: Fraction of the frame below the horizon (ground), in
            ``[0.0, 1.0]``. ``1.0`` for ``ABOVE_FRAME``, ``0.0`` for
            ``BELOW_FRAME``.
        method: Source of the estimate (e.g. ``"geometric"``, ``"cv"``,
            ``"geometric+cv"``).
        confidence: Heuristic confidence in ``[0.0, 1.0]``.
    """

    placement: FramePlacement
    image_height: int
    row: float | None = None
    ground_fraction: float = 0.0
    method: str = "geometric"
    confidence: float = 1.0
    schema_version: str = HORIZON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "schema_version": self.schema_version,
            "placement": self.placement.value,
            "image_height": self.image_height,
            "row": self.row,
            "ground_fraction": self.ground_fraction,
            "method": self.method,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HorizonEstimate:
        """Reconstruct a :class:`HorizonEstimate` from :meth:`to_dict` output.

        Raises:
            ValueError: If ``schema_version`` is unsupported.
        """
        version = data.get("schema_version", HORIZON_SCHEMA_VERSION)
        if version != HORIZON_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r}; expected {HORIZON_SCHEMA_VERSION!r}"
            )
        row = data.get("row")
        return cls(
            placement=FramePlacement(data["placement"]),
            image_height=int(data["image_height"]),
            row=None if row is None else float(row),
            ground_fraction=float(data.get("ground_fraction", 0.0)),
            method=str(data.get("method", "geometric")),
            confidence=float(data.get("confidence", 1.0)),
            schema_version=HORIZON_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class CalibrationResult:
    """Result of fitting the tilt→true-elevation mount offset.

    Attributes:
        schema_version: VO schema version (always ``"1"``).
        tilt_offset_deg: Reported tilt at which the optical axis is horizontal
            (true elevation 0). Expected ≈ ``-31`` for ``icozee-camptz-04``.
        vfov_deg: Vertical field of view recovered from the fit, in degrees.
        zoom: Zoom factor the samples were captured at.
        rms_fraction_residual: RMS residual between fitted and observed horizon
            fractions (0 = perfect fit).
        n_samples: Number of (tilt, row) samples used.
    """

    tilt_offset_deg: float
    vfov_deg: float
    zoom: float
    rms_fraction_residual: float
    n_samples: int
    schema_version: str = HORIZON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "schema_version": self.schema_version,
            "tilt_offset_deg": self.tilt_offset_deg,
            "vfov_deg": self.vfov_deg,
            "zoom": self.zoom,
            "rms_fraction_residual": self.rms_fraction_residual,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationResult:
        """Reconstruct a :class:`CalibrationResult` from :meth:`to_dict` output.

        Raises:
            ValueError: If ``schema_version`` is unsupported.
        """
        version = data.get("schema_version", HORIZON_SCHEMA_VERSION)
        if version != HORIZON_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r}; expected {HORIZON_SCHEMA_VERSION!r}"
            )
        return cls(
            tilt_offset_deg=float(data["tilt_offset_deg"]),
            vfov_deg=float(data["vfov_deg"]),
            zoom=float(data["zoom"]),
            rms_fraction_residual=float(data["rms_fraction_residual"]),
            n_samples=int(data["n_samples"]),
            schema_version=HORIZON_SCHEMA_VERSION,
        )
