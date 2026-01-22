"""Data models for lens distortion calibration.

This module defines the core data structures used throughout the lens
distortion calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
    """

    line_id: str
    image_path: str
    start_pixel: tuple[float, float]
    end_pixel: tuple[float, float]
    ptz_position: PTZPosition
    line_type: str = "parking"
    confidence: float = 1.0

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

    def to_points_array(self) -> np.ndarray:
        """Convert to numpy array of shape (2, 2) with start and end points."""
        return np.array([self.start_pixel, self.end_pixel], dtype=np.float64)

    def sample_points(self, num_samples: int = 20) -> np.ndarray:
        """Sample evenly-spaced points along the line.

        Args:
            num_samples: Number of points to sample.

        Returns:
            Array of shape (num_samples, 2) with (u, v) coordinates.
        """
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
    to which known line in the ground truth map.

    Attributes:
        camera_line: The detected line in camera image coordinates.
        ground_truth_line: The corresponding line in world coordinates.
    """

    camera_line: CameraLine
    ground_truth_line: GroundTruthLine

    def __post_init__(self) -> None:
        """Validate correspondence."""
        # Line IDs should match or be explicitly paired
        pass  # Allow any pairing for flexibility


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
