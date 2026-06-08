"""Per-azimuth tilt envelope value object (horizon-aware survey constraint).

A :class:`TiltEnvelope` records, for a coarse set of calibrated azimuths, the
*maximum-up useful tilt* — the most-up the camera can point at that azimuth
while the frame still contains ground rather than pure sky. Survey pose
generators consume the envelope to skip sky tiles.

Convention (matching :mod:`poc_homography.horizon.geometry` and
:class:`poc_homography.camera.intrinsics.PTZStatus`): **positive tilt = down**.
Pointing further up means a *smaller* (more negative) tilt. The per-azimuth
``tilt_upper_bound`` is therefore a *lower numeric clamp*: a useful pose has
``tilt >= upper_bound(pan)``; any pose with ``tilt < upper_bound(pan)`` points
above the useful ground band and is sky. The name "upper bound" refers to the
upper limit *in the scene* (toward the sky), not a numeric maximum.

The envelope is a frozen, JSON-round-trippable VO following the project pattern
(``schema_version`` discipline, ``to_dict``/``from_dict``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TILT_ENVELOPE_SCHEMA_VERSION = "1"
"""Schema version for :class:`TiltEnvelope`.

Deliberately the literal ``"1"`` and independent of every other
``*_SCHEMA_VERSION`` in the codebase; the envelope VO evolves on its own.
"""

# Tolerance for matching an exact calibrated azimuth (in degrees).
_AZIMUTH_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TiltEnvelope:
    """A per-azimuth maximum-up-useful-tilt envelope.

    Attributes:
        bounds: ``azimuth_deg -> tilt_upper_bound_deg`` for each calibrated
            azimuth. Azimuths are pan angles in degrees; bounds are reported
            tilt in degrees (positive = down).
        tilt_offset_deg: Confirmed reported tilt at which the optical axis is
            horizontal (true elevation 0).
        vfov_deg: Vertical field of view at ``zoom``, in degrees.
        zoom: Zoom factor the calibration frames were captured at.
        interpolation: Interpolation rule for pans between calibrated azimuths;
            always ``"linear"`` (circular linear interpolation, see
            :meth:`upper_bound`).
        schema_version: VO schema version (always ``"1"``).
    """

    bounds: dict[float, float] = field(default_factory=dict)
    tilt_offset_deg: float = 0.0
    vfov_deg: float = 0.0
    zoom: float = 1.0
    interpolation: str = "linear"
    schema_version: str = TILT_ENVELOPE_SCHEMA_VERSION

    def upper_bound(self, pan_deg: float) -> float:
        """Return the max-up useful tilt at ``pan_deg`` (positive = down).

        The bound is **linearly interpolated** between the two nearest
        calibrated azimuths, treating pan as circular (wrapping across the
        360°/0° seam). Linear (rather than nearest-neighbour) interpolation is
        chosen so the constraint varies smoothly around the pan circle and a
        coarse calibration set does not produce abrupt per-tile jumps.

        Args:
            pan_deg: Pan angle in degrees (any value; reduced modulo 360).

        Returns:
            The interpolated ``tilt_upper_bound`` in degrees.

        Raises:
            ValueError: If the envelope has no calibrated azimuths.
        """
        if not self.bounds:
            raise ValueError("empty TiltEnvelope has no bound to interpolate")

        azimuths = sorted(self.bounds)
        if len(azimuths) == 1:
            return self.bounds[azimuths[0]]

        pan = pan_deg % 360.0
        for az in azimuths:
            if abs(az - pan) <= _AZIMUTH_TOLERANCE:
                return self.bounds[az]

        lo_az: float | None = None
        hi_az: float | None = None
        for az in azimuths:
            if az <= pan:
                lo_az = az
            if az >= pan and hi_az is None:
                hi_az = az

        if lo_az is not None and hi_az is not None:
            return self._lerp(lo_az, hi_az, pan)

        # Wrap-around seam: pan sits below the smallest or above the largest
        # calibrated azimuth, so interpolate across the 360°/0° boundary.
        lo_az = azimuths[-1]
        hi_az = azimuths[0]
        pan_eff = pan + 360.0 if pan < hi_az else pan
        span = (hi_az + 360.0) - lo_az
        frac = (pan_eff - lo_az) / span
        return _interpolate(self.bounds[lo_az], self.bounds[hi_az], frac)

    def _lerp(self, lo_az: float, hi_az: float, pan: float) -> float:
        """Linearly interpolate the bound between two bracketing azimuths."""
        span = hi_az - lo_az
        if span <= _AZIMUTH_TOLERANCE:
            return self.bounds[lo_az]
        frac = (pan - lo_az) / span
        return _interpolate(self.bounds[lo_az], self.bounds[hi_az], frac)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Bound keys (azimuth floats) are stringified so the result survives
        ``json.dumps`` / ``yaml.safe_dump`` round-trips.
        """
        return {
            "schema_version": self.schema_version,
            "bounds": {str(k): v for k, v in self.bounds.items()},
            "tilt_offset_deg": self.tilt_offset_deg,
            "vfov_deg": self.vfov_deg,
            "zoom": self.zoom,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TiltEnvelope:
        """Reconstruct a :class:`TiltEnvelope` from :meth:`to_dict` output.

        Raises:
            ValueError: If ``schema_version`` is unsupported.
        """
        version = data.get("schema_version", TILT_ENVELOPE_SCHEMA_VERSION)
        if version != TILT_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r}; expected {TILT_ENVELOPE_SCHEMA_VERSION!r}"
            )
        raw_bounds = data.get("bounds") or {}
        return cls(
            bounds={float(k): float(v) for k, v in raw_bounds.items()},
            tilt_offset_deg=float(data.get("tilt_offset_deg", 0.0)),
            vfov_deg=float(data.get("vfov_deg", 0.0)),
            zoom=float(data.get("zoom", 1.0)),
            interpolation=str(data.get("interpolation", "linear")),
            schema_version=TILT_ENVELOPE_SCHEMA_VERSION,
        )


def _interpolate(lo_value: float, hi_value: float, frac: float) -> float:
    """Linear blend ``lo_value -> hi_value`` by ``frac`` in ``[0, 1]``."""
    return lo_value + (hi_value - lo_value) * frac
