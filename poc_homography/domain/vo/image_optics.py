"""Image optics value objects (focus, iris, exposure, white balance)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.domain.vo._xml import find_text

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


@dataclass(frozen=True)
class FocusState:
    """Focus configuration.

    Attributes:
        style: Focus style (e.g., ``SEMIAUTOMATIC``).
        focus_limited: Near-focus limit in centimeters, or ``None`` if absent.
    """

    style: str
    focus_limited: int | None

    @classmethod
    def from_element(cls, elem: Element) -> FocusState:
        """Build :class:`FocusState` from a ``FocusConfiguration`` element."""
        limited = find_text(elem, "focusLimited")
        return cls(
            style=find_text(elem, "focusStyle") or "",
            focus_limited=int(limited) if limited is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"style": self.style, "focus_limited": self.focus_limited}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FocusState:
        """Create :class:`FocusState` from a dictionary."""
        limited = data.get("focus_limited")
        return cls(
            style=data["style"],
            focus_limited=int(limited) if limited is not None else None,
        )


@dataclass(frozen=True)
class IrisState:
    """Iris level configuration.

    Attributes:
        level: Current iris level.
        min_level: Minimum iris level limit.
        max_level: Maximum iris level limit.
    """

    level: int
    min_level: int
    max_level: int

    @classmethod
    def from_element(cls, elem: Element) -> IrisState:
        """Build :class:`IrisState` from an ``Iris`` element."""
        return cls(
            level=int(find_text(elem, "IrisLevel") or 0),
            min_level=int(find_text(elem, "minIrisLevelLimit") or 0),
            max_level=int(find_text(elem, "maxIrisLevelLimit") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "min_level": self.min_level,
            "max_level": self.max_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IrisState:
        """Create :class:`IrisState` from a dictionary."""
        return cls(
            level=int(data["level"]),
            min_level=int(data["min_level"]),
            max_level=int(data["max_level"]),
        )


@dataclass(frozen=True)
class ExposureState:
    """Exposure configuration.

    Attributes:
        exposure_type: Exposure mode (e.g., ``auto``).
        overexpose_suppress: Whether overexposure suppression is enabled.
    """

    exposure_type: str
    overexpose_suppress: bool

    @classmethod
    def from_element(cls, elem: Element) -> ExposureState:
        """Build :class:`ExposureState` from an ``Exposure`` element."""
        enabled = find_text(elem, "OverexposeSuppress", "enabled")
        return cls(
            exposure_type=find_text(elem, "ExposureType") or "",
            overexpose_suppress=(enabled or "").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exposure_type": self.exposure_type,
            "overexpose_suppress": self.overexpose_suppress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExposureState:
        """Create :class:`ExposureState` from a dictionary."""
        return cls(
            exposure_type=data["exposure_type"],
            overexpose_suppress=bool(data["overexpose_suppress"]),
        )


@dataclass(frozen=True)
class WhiteBalanceState:
    """White balance configuration.

    Attributes:
        style: White balance style (e.g., ``auto``).
        red: Red channel gain.
        blue: Blue channel gain.
    """

    style: str
    red: int
    blue: int

    @classmethod
    def from_element(cls, elem: Element) -> WhiteBalanceState:
        """Build :class:`WhiteBalanceState` from a ``WhiteBalance`` element."""
        return cls(
            style=find_text(elem, "WhiteBalanceStyle") or "",
            red=int(find_text(elem, "WhiteBalanceRed") or 0),
            blue=int(find_text(elem, "WhiteBalanceBlue") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"style": self.style, "red": self.red, "blue": self.blue}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WhiteBalanceState:
        """Create :class:`WhiteBalanceState` from a dictionary."""
        return cls(
            style=data["style"],
            red=int(data["red"]),
            blue=int(data["blue"]),
        )


@dataclass(frozen=True)
class ImageOptics:
    """Aggregate of the four optics value objects.

    Attributes:
        focus: Focus configuration.
        iris: Iris configuration.
        exposure: Exposure configuration.
        white_balance: White balance configuration.
    """

    focus: FocusState
    iris: IrisState
    exposure: ExposureState
    white_balance: WhiteBalanceState

    def to_dict(self) -> dict[str, Any]:
        """Convert to a nested dictionary for serialization."""
        return {
            "focus": self.focus.to_dict(),
            "iris": self.iris.to_dict(),
            "exposure": self.exposure.to_dict(),
            "white_balance": self.white_balance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageOptics:
        """Create :class:`ImageOptics` from a nested dictionary."""
        return cls(
            focus=FocusState.from_dict(data["focus"]),
            iris=IrisState.from_dict(data["iris"]),
            exposure=ExposureState.from_dict(data["exposure"]),
            white_balance=WhiteBalanceState.from_dict(data["white_balance"]),
        )
