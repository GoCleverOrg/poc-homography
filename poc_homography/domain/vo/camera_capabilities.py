"""Camera capabilities value object (degrees-based, real hardware ranges)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.domain.vo._xml import find_child
from poc_homography.types import Degrees, FocusSteps, Unitless

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


@dataclass(frozen=True)
class CameraCapabilities:
    """PTZ camera position and speed limits, expressed in degrees.

    Sourced from the ISAPI ``absoluteEx/capabilities`` document, whose
    min/max bounds are already in degrees (pan/tilt) or dimensionless (zoom).
    Tilt bounds reflect the real hardware envelope (-50 down .. 60 up), not the
    legacy -90/90 placeholder.

    Attributes:
        pan_min: Minimum pan (azimuth) in degrees.
        pan_max: Maximum pan (azimuth) in degrees.
        tilt_min: Minimum tilt (elevation) in degrees.
        tilt_max: Maximum tilt (elevation) in degrees.
        zoom_min: Minimum zoom factor (dimensionless).
        zoom_max: Maximum zoom factor (dimensionless).
        focus_min: Minimum focus position in raw lens steps.
        focus_max: Maximum focus position in raw lens steps.
        pan_speed_min: Minimum horizontal pan speed (degrees/second).
        pan_speed_max: Maximum horizontal pan speed (degrees/second).
        tilt_speed_min: Minimum vertical tilt speed (degrees/second).
        tilt_speed_max: Maximum vertical tilt speed (degrees/second).
    """

    pan_min: Degrees
    pan_max: Degrees
    tilt_min: Degrees
    tilt_max: Degrees
    zoom_min: Unitless
    zoom_max: Unitless
    focus_min: FocusSteps
    focus_max: FocusSteps
    pan_speed_min: float
    pan_speed_max: float
    tilt_speed_min: float
    tilt_speed_max: float

    @classmethod
    def from_absolute_ex_element(cls, elem: Element) -> CameraCapabilities:
        """Build capabilities from a ``PTZAbsoluteEx`` element.

        Reads the ``min``/``max`` XML attributes of ``<elevation>``,
        ``<azimuth>``, ``<absoluteZoom>``, ``<focus>``, ``<horizontalSpeed>``
        and ``<verticalSpeed>``. These bounds are already in their target units
        (degrees, zoom factor, focus steps), so no scaling is applied.

        Args:
            elem: The ``PTZAbsoluteEx`` root element.

        Returns:
            A populated :class:`CameraCapabilities`.
        """

        def bounds(tag: str) -> tuple[str, str]:
            child = find_child(elem, tag)
            if child is None:
                raise ValueError(f"missing <{tag}> in absoluteEx capabilities")
            return child.attrib["min"], child.attrib["max"]

        elevation_min, elevation_max = bounds("elevation")
        azimuth_min, azimuth_max = bounds("azimuth")
        zoom_lo, zoom_hi = bounds("absoluteZoom")
        focus_lo, focus_hi = bounds("focus")
        h_speed_min, h_speed_max = bounds("horizontalSpeed")
        v_speed_min, v_speed_max = bounds("verticalSpeed")

        return cls(
            pan_min=Degrees(float(azimuth_min)),
            pan_max=Degrees(float(azimuth_max)),
            tilt_min=Degrees(float(elevation_min)),
            tilt_max=Degrees(float(elevation_max)),
            zoom_min=Unitless(float(zoom_lo)),
            zoom_max=Unitless(float(zoom_hi)),
            focus_min=FocusSteps(int(focus_lo)),
            focus_max=FocusSteps(int(focus_hi)),
            pan_speed_min=float(h_speed_min),
            pan_speed_max=float(h_speed_max),
            tilt_speed_min=float(v_speed_min),
            tilt_speed_max=float(v_speed_max),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pan_min": float(self.pan_min),
            "pan_max": float(self.pan_max),
            "tilt_min": float(self.tilt_min),
            "tilt_max": float(self.tilt_max),
            "zoom_min": float(self.zoom_min),
            "zoom_max": float(self.zoom_max),
            "focus_min": int(self.focus_min),
            "focus_max": int(self.focus_max),
            "pan_speed_min": self.pan_speed_min,
            "pan_speed_max": self.pan_speed_max,
            "tilt_speed_min": self.tilt_speed_min,
            "tilt_speed_max": self.tilt_speed_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraCapabilities:
        """Create :class:`CameraCapabilities` from a dictionary."""
        return cls(
            pan_min=Degrees(float(data["pan_min"])),
            pan_max=Degrees(float(data["pan_max"])),
            tilt_min=Degrees(float(data["tilt_min"])),
            tilt_max=Degrees(float(data["tilt_max"])),
            zoom_min=Unitless(float(data["zoom_min"])),
            zoom_max=Unitless(float(data["zoom_max"])),
            focus_min=FocusSteps(int(data["focus_min"])),
            focus_max=FocusSteps(int(data["focus_max"])),
            pan_speed_min=float(data["pan_speed_min"]),
            pan_speed_max=float(data["pan_speed_max"]),
            tilt_speed_min=float(data["tilt_speed_min"]),
            tilt_speed_max=float(data["tilt_speed_max"]),
        )
