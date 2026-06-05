"""ISAPI endpoint paths and XML namespace for Hikvision cameras.

This module is the single source of truth for every ISAPI request path used by
the Hikvision adapter. Paths are parameterized by channel (default 1 for PTZ /
Image controls, default 101 for streaming). No other module in the codebase may
hardcode ``/ISAPI/`` path literals or the Hikvision XML namespace string.
"""

from __future__ import annotations

HIKVISION_XML_NS = "http://www.hikvision.com/ver20/XMLSchema"
"""Hikvision ISAPI XML namespace. The ONLY definition of this string in the repo."""

NS = {"h": HIKVISION_XML_NS}
"""Namespace map for ElementTree ``find``/``findtext`` calls."""


def ptz_status(ch: int = 1) -> str:
    """Return the PTZ status path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/status"


def ptz_capabilities(ch: int = 1) -> str:
    """Return the PTZ capabilities path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/capabilities"


def ptz_absolute_ex_capabilities(ch: int = 1) -> str:
    """Return the absoluteEx capabilities path (degree-based ranges) for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/absoluteEx/capabilities"


def ptz_absolute(ch: int = 1) -> str:
    """Return the absolute positioning path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/absolute"


def ptz_relative(ch: int = 1) -> str:
    """Return the relative positioning path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/relative"


def ptz_continuous(ch: int = 1) -> str:
    """Return the continuous movement path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/continuous"


def ptz_momentary(ch: int = 1) -> str:
    """Return the momentary movement path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/momentary"


def ptz_position3d(ch: int = 1) -> str:
    """Return the 3D positioning (drag-zoom) path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/position3D"


def ptz_presets(ch: int = 1) -> str:
    """Return the presets list path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/presets"


def ptz_preset_goto(preset_id: int, ch: int = 1) -> str:
    """Return the goto-preset path for ``preset_id`` on ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/presets/{preset_id}/goto"


def ptz_home_goto(ch: int = 1) -> str:
    """Return the goto-home-position path for ``ch``."""
    return f"/ISAPI/PTZCtrl/channels/{ch}/homeposition/goto"


def device_info() -> str:
    """Return the device-info path."""
    return "/ISAPI/System/deviceInfo"


def system_status() -> str:
    """Return the system-status path (device health)."""
    return "/ISAPI/System/status"


def streaming_channel(ch: int = 101) -> str:
    """Return the streaming-channel configuration path for ``ch``."""
    return f"/ISAPI/Streaming/channels/{ch}"


def streaming_picture(ch: int = 101) -> str:
    """Return the snapshot (JPEG) path for ``ch``."""
    return f"/ISAPI/Streaming/channels/{ch}/picture"


def image_focus_configuration(ch: int = 1) -> str:
    """Return the focus-configuration path for ``ch``."""
    return f"/ISAPI/Image/channels/{ch}/focusConfiguration"


def image_iris(ch: int = 1) -> str:
    """Return the iris path for ``ch``."""
    return f"/ISAPI/Image/channels/{ch}/iris"


def image_exposure(ch: int = 1) -> str:
    """Return the exposure path for ``ch``."""
    return f"/ISAPI/Image/channels/{ch}/exposure"


def image_white_balance(ch: int = 1) -> str:
    """Return the white-balance path for ``ch``."""
    return f"/ISAPI/Image/channels/{ch}/whiteBalance"


def image_capabilities(ch: int = 1) -> str:
    """Return the image capabilities path for ``ch``."""
    return f"/ISAPI/Image/channels/{ch}/capabilities"
