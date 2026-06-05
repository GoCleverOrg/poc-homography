"""Streaming channel profile value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.domain.vo._xml import find_all, find_child, find_text
from poc_homography.types import Pixels

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

_FPS_SCALE = 100


@dataclass(frozen=True)
class StreamProfile:
    """Video streaming profile for a single channel.

    Attributes:
        channel_id: Streaming channel id (e.g., 101).
        codec: Video codec type (e.g., ``H.264``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Maximum frame rate in frames per second (``maxFrameRate`` / 100).
        bitrate_kbps: Upper bitrate cap in kbps.
        quality_control: Quality control type (e.g., ``VBR``).
        transports: Supported streaming transports (e.g., RTSP, HTTP).
    """

    channel_id: int
    codec: str
    width: Pixels
    height: Pixels
    fps: float
    bitrate_kbps: int
    quality_control: str
    transports: list[str]

    @classmethod
    def from_element(cls, elem: Element) -> StreamProfile:
        """Build :class:`StreamProfile` from a ``StreamingChannel`` element."""
        video = find_child(elem, "Video")
        video = video if video is not None else elem

        raw_fps = find_text(video, "maxFrameRate")
        fps = (int(raw_fps) / _FPS_SCALE) if raw_fps is not None else 0.0

        bitrate = find_text(video, "vbrUpperCap") or find_text(video, "constantBitRate")

        transports: list[str] = []
        for control in find_all(elem, "Transport", "ControlProtocolList", "ControlProtocol"):
            value = find_text(control, "streamingTransport")
            if value is not None:
                transports.append(value)

        return cls(
            channel_id=int(find_text(elem, "id") or 0),
            codec=find_text(video, "videoCodecType") or "",
            width=Pixels(int(find_text(video, "videoResolutionWidth") or 0)),
            height=Pixels(int(find_text(video, "videoResolutionHeight") or 0)),
            fps=fps,
            bitrate_kbps=int(bitrate) if bitrate is not None else 0,
            quality_control=find_text(video, "videoQualityControlType") or "",
            transports=transports,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "channel_id": self.channel_id,
            "codec": self.codec,
            "width": int(self.width),
            "height": int(self.height),
            "fps": self.fps,
            "bitrate_kbps": self.bitrate_kbps,
            "quality_control": self.quality_control,
            "transports": list(self.transports),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamProfile:
        """Create :class:`StreamProfile` from a dictionary."""
        return cls(
            channel_id=int(data["channel_id"]),
            codec=data["codec"],
            width=Pixels(int(data["width"])),
            height=Pixels(int(data["height"])),
            fps=float(data["fps"]),
            bitrate_kbps=int(data["bitrate_kbps"]),
            quality_control=data["quality_control"],
            transports=list(data["transports"]),
        )
