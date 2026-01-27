"""Data models for lens distortion calibration.

This module defines the core data structures used throughout the lens
distortion calibration pipeline.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from poc_homography.types import Degrees, Meters, Pixels


@dataclass(frozen=True)
class PTZPosition:
    """Camera PTZ (Pan-Tilt-Zoom) position when an image was captured.

    Attributes:
        pan_deg: Pan angle in degrees (positive = right/clockwise from above).
        tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention).
        zoom_factor: Zoom multiplier (1.0 = no zoom).
    """

    pan_deg: Degrees
    tilt_deg: Degrees
    zoom_factor: float

    def __post_init__(self) -> None:
        """Validate PTZ position values."""
        if self.zoom_factor <= 0:
            raise ValueError(f"zoom_factor must be positive, got {self.zoom_factor}")


@dataclass(frozen=True)
class CameraLine:
    """A line detected in a camera image.

    Represents a parking spot line or other straight feature detected in
    a camera image, stored in pixel coordinates.

    Attributes:
        line_id: Unique identifier for this line.
        image_path: Path to the source image.
        start_pixel: (u, v) start point in image pixels.
        end_pixel: (u, v) end point in image pixels.
        ptz_position: Camera PTZ position when image was captured.
        line_type: Category of line ("boundary", "divider", "crosswalk", etc.).
        confidence: Detection confidence score (0.0 to 1.0).
        edge_pixels: Optional array of actual edge pixel coordinates along
            the line, shape (N, 2). If provided, sample_points() returns
            these instead of interpolated points.
    """

    line_id: str
    image_path: str
    start_pixel: tuple[float, float]
    end_pixel: tuple[float, float]
    ptz_position: PTZPosition
    line_type: str = "parking"
    confidence: float = 1.0
    edge_pixels: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Validate line coordinates."""
        if len(self.start_pixel) != 2:
            raise ValueError(f"start_pixel must have 2 elements, got {len(self.start_pixel)}")
        if len(self.end_pixel) != 2:
            raise ValueError(f"end_pixel must have 2 elements, got {len(self.end_pixel)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @property
    def length_pixels(self) -> float:
        """Calculate line length in pixels."""
        dx = self.end_pixel[0] - self.start_pixel[0]
        dy = self.end_pixel[1] - self.start_pixel[1]
        return float(np.sqrt(dx * dx + dy * dy))

    @property
    def angle_degrees(self) -> float:
        """Calculate line angle in degrees (0 = horizontal, 90 = vertical)."""
        dx = self.end_pixel[0] - self.start_pixel[0]
        dy = self.end_pixel[1] - self.start_pixel[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    @property
    def has_edge_pixels(self) -> bool:
        """Check if this line has valid edge pixel data.

        Returns:
            True if edge_pixels is not None and has at least 2 points,
            False otherwise.
        """
        return self.edge_pixels is not None and len(self.edge_pixels) >= 2

    def to_points_array(self) -> np.ndarray:
        """Convert to numpy array of shape (2, 2) with start and end points."""
        return np.array([self.start_pixel, self.end_pixel], dtype=np.float64)

    def sample_points(
        self,
        num_samples: int = 20,
        mode: Literal["edge_pixels", "linear", "auto"] = "edge_pixels",
    ) -> np.ndarray:
        """Sample points along the line.

        Args:
            num_samples: Number of points to sample.
            mode: Sampling mode, one of:
                - "edge_pixels" (default): Returns actual edge pixel coordinates
                  which contain the distortion signal needed for calibration.
                  Raises ValueError if edge_pixels is None or has fewer than 2
                  points. Use this mode for calibration.
                - "linear": Returns linearly interpolated points between start
                  and end pixels. Use this mode for visualization only, as it
                  does not contain distortion information.
                - "auto": DEPRECATED. Uses edge_pixels if available, otherwise
                  falls back to linear interpolation. Provided for backwards
                  compatibility only; new code should explicitly choose a mode.

        Returns:
            Array of shape (N, 2) with (u, v) coordinates.

        Raises:
            ValueError: If mode="edge_pixels" and edge_pixels is None or has
                fewer than 2 points.
            ValueError: If mode is not one of "edge_pixels", "linear", or "auto".
        """
        valid_modes = ("edge_pixels", "linear", "auto")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        if mode == "edge_pixels":
            if not self.has_edge_pixels:
                raise ValueError(
                    f"Line {self.line_id} has no edge_pixels data. "
                    "Cannot use mode='edge_pixels' for calibration."
                )
            return self._sample_edge_pixels(num_samples)

        if mode == "linear":
            return self._sample_linear(num_samples)

        # mode == "auto" (deprecated)
        warnings.warn(
            "mode='auto' is deprecated. Explicitly use 'edge_pixels' for calibration "
            "or 'linear' for visualization.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.has_edge_pixels:
            return self._sample_edge_pixels(num_samples)

        logger.debug(
            "Line %s: edge_pixels not available, falling back to linear interpolation",
            self.line_id,
        )
        return self._sample_linear(num_samples)

    def _sample_edge_pixels(self, num_samples: int) -> np.ndarray:
        """Sample from actual edge pixels (internal helper).

        Assumes has_edge_pixels is True.
        """
        edge_arr = np.array(self.edge_pixels, dtype=np.float64)
        if len(edge_arr) <= num_samples:
            return edge_arr
        # Subsample evenly
        indices = np.linspace(0, len(edge_arr) - 1, num_samples, dtype=int)
        return edge_arr[indices]

    def _sample_linear(self, num_samples: int) -> np.ndarray:
        """Sample via linear interpolation between endpoints (internal helper)."""
        t = np.linspace(0, 1, num_samples)
        start = np.array(self.start_pixel)
        end = np.array(self.end_pixel)
        points = start + np.outer(t, end - start)
        return points.astype(np.float64)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "line_id": self.line_id,
            "image_path": self.image_path,
            "start_pixel_x": self.start_pixel[0],
            "start_pixel_y": self.start_pixel[1],
            "end_pixel_x": self.end_pixel[0],
            "end_pixel_y": self.end_pixel[1],
            "pan_deg": float(self.ptz_position.pan_deg),
            "tilt_deg": float(self.ptz_position.tilt_deg),
            "zoom_factor": self.ptz_position.zoom_factor,
            "line_type": self.line_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraLine:
        """Create from dictionary."""
        return cls(
            line_id=data["line_id"],
            image_path=data["image_path"],
            start_pixel=(data["start_pixel_x"], data["start_pixel_y"]),
            end_pixel=(data["end_pixel_x"], data["end_pixel_y"]),
            ptz_position=PTZPosition(
                pan_deg=data["pan_deg"],
                tilt_deg=data["tilt_deg"],
                zoom_factor=data["zoom_factor"],
            ),
            line_type=data.get("line_type", "parking"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass(frozen=True)
class GroundTruthLine:
    """A line from ground truth (aerial map) in world coordinates.

    Represents a parking spot line as defined on an aerial/satellite map,
    stored in world coordinates (meters).

    Attributes:
        line_id: Unique identifier for this line (matches camera line IDs).
        start_world: (X, Y) start point in meters (East, North).
        end_world: (X, Y) end point in meters (East, North).
        map_id: Identifier of the source map.
    """

    line_id: str
    start_world: tuple[Meters, Meters]
    end_world: tuple[Meters, Meters]
    map_id: str = ""

    def __post_init__(self) -> None:
        """Validate line coordinates."""
        if len(self.start_world) != 2:
            raise ValueError(f"start_world must have 2 elements, got {len(self.start_world)}")
        if len(self.end_world) != 2:
            raise ValueError(f"end_world must have 2 elements, got {len(self.end_world)}")

    @property
    def length_meters(self) -> Meters:
        """Calculate line length in meters."""
        dx = float(self.end_world[0]) - float(self.start_world[0])
        dy = float(self.end_world[1]) - float(self.start_world[1])
        return float(np.sqrt(dx * dx + dy * dy))  # type: ignore[return-value]

    def to_points_array(self) -> np.ndarray:
        """Convert to numpy array of shape (2, 2) with start and end points."""
        return np.array(
            [
                [float(self.start_world[0]), float(self.start_world[1])],
                [float(self.end_world[0]), float(self.end_world[1])],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "line_id": self.line_id,
            "start_world_x": float(self.start_world[0]),
            "start_world_y": float(self.start_world[1]),
            "end_world_x": float(self.end_world[0]),
            "end_world_y": float(self.end_world[1]),
            "map_id": self.map_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GroundTruthLine:
        """Create from dictionary."""
        return cls(
            line_id=data["line_id"],
            start_world=(data["start_world_x"], data["start_world_y"]),
            end_world=(data["end_world_x"], data["end_world_y"]),
            map_id=data.get("map_id", ""),
        )


@dataclass(frozen=True)
class LineCorrespondence:
    """Mapping between a camera line and its ground truth equivalent.

    Used to establish which detected line in the camera image corresponds
    to which known line in the ground truth map. Any pairing is allowed
    for flexibility in matching strategies.

    Attributes:
        camera_line: The detected line in camera image coordinates.
        ground_truth_line: The corresponding line in world coordinates.
    """

    camera_line: CameraLine
    ground_truth_line: GroundTruthLine


@dataclass(frozen=True)
class MapPixelLine:
    """A line defined in map pixel coordinates.

    Used for lines drawn on aerial/satellite map images before
    conversion to world coordinates.

    Attributes:
        line_id: Unique identifier for this line.
        start_pixel: (x, y) start point in map pixels.
        end_pixel: (x, y) end point in map pixels.
        map_id: Identifier of the source map.
    """

    line_id: str
    start_pixel: tuple[Pixels, Pixels]
    end_pixel: tuple[Pixels, Pixels]
    map_id: str = ""

    def to_world_coordinates(self, geotransform: list[float]) -> GroundTruthLine:
        """Convert map pixel coordinates to world coordinates using geotransform.

        Args:
            geotransform: 6-element GDAL-style geotransform
                [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]

        Returns:
            GroundTruthLine with world coordinates.
        """
        origin_x = geotransform[0]
        pixel_width = geotransform[1]
        rotation_x = geotransform[2]
        origin_y = geotransform[3]
        rotation_y = geotransform[4]
        pixel_height = geotransform[5]

        # Convert start point
        start_x = (
            origin_x
            + float(self.start_pixel[0]) * pixel_width
            + float(self.start_pixel[1]) * rotation_x
        )
        start_y = (
            origin_y
            + float(self.start_pixel[0]) * rotation_y
            + float(self.start_pixel[1]) * pixel_height
        )

        # Convert end point
        end_x = (
            origin_x
            + float(self.end_pixel[0]) * pixel_width
            + float(self.end_pixel[1]) * rotation_x
        )
        end_y = (
            origin_y
            + float(self.end_pixel[0]) * rotation_y
            + float(self.end_pixel[1]) * pixel_height
        )

        return GroundTruthLine(
            line_id=self.line_id,
            start_world=(start_x, start_y),  # type: ignore[arg-type]
            end_world=(end_x, end_y),  # type: ignore[arg-type]
            map_id=self.map_id,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "line_id": self.line_id,
            "start_x": float(self.start_pixel[0]),
            "start_y": float(self.start_pixel[1]),
            "end_x": float(self.end_pixel[0]),
            "end_y": float(self.end_pixel[1]),
            "map_id": self.map_id,
        }

    @classmethod
    def from_dict(cls, data: dict, map_id: str = "") -> MapPixelLine:
        """Create from dictionary (compatible with line_picker format)."""
        return cls(
            line_id=data["line_id"],
            start_pixel=(data["start_x"], data["start_y"]),
            end_pixel=(data["end_x"], data["end_y"]),
            map_id=data.get("map_id", map_id),
        )
