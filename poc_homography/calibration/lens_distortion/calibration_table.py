"""Zoom-dependent calibration storage with interpolation.

This module provides storage and retrieval of calibration results
that vary with zoom level. When calibration is only performed at
certain zoom levels, this module interpolates coefficients for
intermediate zoom values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZoomCalibrationEntry:
    """Calibration data for a specific zoom level.

    Attributes:
        zoom_factor: The zoom level this calibration applies to.
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
        k3: Third radial distortion coefficient.
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.
        calibration_date: Date/time when calibration was performed.
        source_images: List of image paths or session IDs used.
        validation_rmse: Line straightness RMSE from validation.
        num_lines_used: Number of lines used for calibration.
        reprojection_error_px: Reprojection error in pixels (OpenCV method only).
    """

    zoom_factor: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    calibration_date: str
    source_images: tuple[str, ...] = field(default_factory=tuple)
    validation_rmse: float = 0.0
    num_lines_used: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    reprojection_error_px: float = 0.0

    def __post_init__(self) -> None:
        """Validate entry data."""
        if self.zoom_factor <= 0:
            raise ValueError(f"zoom_factor must be positive, got {self.zoom_factor}")

    def to_distortion_coefficients(self) -> DistortionCoefficients:
        """Convert to DistortionCoefficients instance."""
        return DistortionCoefficients(
            k1=Unitless(self.k1),
            k2=Unitless(self.k2),
            p1=Unitless(self.p1),
            p2=Unitless(self.p2),
            k3=Unitless(self.k3),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "zoom_factor": self.zoom_factor,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "p1": self.p1,
            "p2": self.p2,
            "calibration_date": self.calibration_date,
            "source_images": list(self.source_images),
            "validation_rmse": self.validation_rmse,
            "num_lines_used": self.num_lines_used,
        }
        if self.fx != 0.0 or self.fy != 0.0:
            d["fx"] = self.fx
            d["fy"] = self.fy
            d["cx"] = self.cx
            d["cy"] = self.cy
        if self.reprojection_error_px != 0.0:
            d["reprojection_error_px"] = self.reprojection_error_px
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ZoomCalibrationEntry:
        """Create from dictionary."""
        source_images = data.get("source_images", [])
        if isinstance(source_images, list):
            source_images = tuple(source_images)
        elif isinstance(source_images, str):
            source_images = (source_images,)

        return cls(
            zoom_factor=float(data["zoom_factor"]),
            k1=float(data["k1"]),
            k2=float(data["k2"]),
            k3=float(data.get("k3", 0.0)),
            p1=float(data.get("p1", 0.0)),
            p2=float(data.get("p2", 0.0)),
            calibration_date=data.get("calibration_date", ""),
            source_images=source_images,
            validation_rmse=float(data.get("validation_rmse", 0.0)),
            num_lines_used=int(data.get("num_lines_used", 0)),
            fx=float(data.get("fx", 0.0)),
            fy=float(data.get("fy", 0.0)),
            cx=float(data.get("cx", 0.0)),
            cy=float(data.get("cy", 0.0)),
            reprojection_error_px=float(data.get("reprojection_error_px", 0.0)),
        )

    @classmethod
    def from_solver_result(
        cls,
        zoom_factor: float,
        distortion: DistortionCoefficients,
        validation_rmse: float = 0.0,
        source_images: list[str] | None = None,
        num_lines_used: int = 0,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
        reprojection_error_px: float = 0.0,
    ) -> ZoomCalibrationEntry:
        """Create from distortion solver result.

        Args:
            zoom_factor: Zoom level for this calibration.
            distortion: Optimized distortion coefficients.
            validation_rmse: RMSE from validation.
            source_images: Images used for calibration.
            num_lines_used: Number of lines used.
            fx: Focal length X used during calibration.
            fy: Focal length Y used during calibration.
            cx: Principal point X used during calibration.
            cy: Principal point Y used during calibration.
            reprojection_error_px: Reprojection error in pixels (OpenCV method).

        Returns:
            New ZoomCalibrationEntry.
        """
        return cls(
            zoom_factor=zoom_factor,
            k1=float(distortion.k1),
            k2=float(distortion.k2),
            k3=float(distortion.k3),
            p1=float(distortion.p1),
            p2=float(distortion.p2),
            calibration_date=datetime.now().isoformat(),
            source_images=tuple(source_images or []),
            validation_rmse=validation_rmse,
            num_lines_used=num_lines_used,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            reprojection_error_px=reprojection_error_px,
        )


@dataclass
class CameraCalibrationTable:
    """Stores calibration entries for a camera across zoom levels.

    Supports interpolation between calibrated zoom levels for smooth
    transitions when the exact zoom level wasn't calibrated.

    Attributes:
        camera_id: Unique identifier for the camera.
        entries: Dictionary mapping zoom_factor to calibration entry.
        created_date: When this table was created.
        last_modified: When this table was last modified.
    """

    camera_id: str
    entries: dict[float, ZoomCalibrationEntry] = field(default_factory=dict)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_entry(self, entry: ZoomCalibrationEntry) -> None:
        """Add or update a calibration entry.

        Args:
            entry: The calibration entry to add.
        """
        self.entries[entry.zoom_factor] = entry
        self.last_modified = datetime.now().isoformat()
        logger.info(f"Added calibration for zoom {entry.zoom_factor}")

    def remove_entry(self, zoom_factor: float) -> bool:
        """Remove a calibration entry.

        Args:
            zoom_factor: The zoom level to remove.

        Returns:
            True if entry was removed, False if not found.
        """
        if zoom_factor in self.entries:
            del self.entries[zoom_factor]
            self.last_modified = datetime.now().isoformat()
            return True
        return False

    def _find_matching_zoom(self, zoom_factor: float, tolerance: float = 1e-6) -> float | None:
        """Find a matching zoom level within tolerance.

        Args:
            zoom_factor: The zoom level to find.
            tolerance: Maximum difference for a match.

        Returns:
            The matching zoom level key, or None if no match.
        """
        for stored_zoom in self.entries:
            if abs(stored_zoom - zoom_factor) < tolerance:
                return stored_zoom
        return None

    def get_coefficients(self, zoom_factor: float) -> DistortionCoefficients:
        """Get distortion coefficients for a zoom level.

        If the exact zoom level is calibrated, returns those coefficients.
        Otherwise, interpolates between the nearest calibrated levels.

        Args:
            zoom_factor: The desired zoom level.

        Returns:
            DistortionCoefficients for the zoom level.

        Raises:
            ValueError: If no calibration entries exist.
        """
        if not self.entries:
            raise ValueError("No calibration entries available")

        # Check for exact match (with floating-point tolerance)
        matching_zoom = self._find_matching_zoom(zoom_factor)
        if matching_zoom is not None:
            return self.entries[matching_zoom].to_distortion_coefficients()

        # Sort available zoom levels
        zoom_levels = sorted(self.entries.keys())

        # Handle edge cases
        if zoom_factor <= zoom_levels[0]:
            logger.debug(f"Zoom {zoom_factor} below minimum {zoom_levels[0]}, using minimum")
            return self.entries[zoom_levels[0]].to_distortion_coefficients()

        if zoom_factor >= zoom_levels[-1]:
            logger.debug(f"Zoom {zoom_factor} above maximum {zoom_levels[-1]}, using maximum")
            return self.entries[zoom_levels[-1]].to_distortion_coefficients()

        # Find bracketing zoom levels
        lower_zoom = zoom_levels[0]
        upper_zoom = zoom_levels[-1]

        for i in range(len(zoom_levels) - 1):
            if zoom_levels[i] <= zoom_factor <= zoom_levels[i + 1]:
                lower_zoom = zoom_levels[i]
                upper_zoom = zoom_levels[i + 1]
                break

        # Linear interpolation
        return self._interpolate(lower_zoom, upper_zoom, zoom_factor)

    def _interpolate(
        self,
        lower_zoom: float,
        upper_zoom: float,
        target_zoom: float,
    ) -> DistortionCoefficients:
        """Interpolate coefficients between two zoom levels.

        Uses linear interpolation for each coefficient.

        Args:
            lower_zoom: Lower bracket zoom level.
            upper_zoom: Upper bracket zoom level.
            target_zoom: Desired zoom level.

        Returns:
            Interpolated DistortionCoefficients.
        """
        lower = self.entries[lower_zoom]
        upper = self.entries[upper_zoom]

        # Calculate interpolation factor
        t = (target_zoom - lower_zoom) / (upper_zoom - lower_zoom)

        # Interpolate each coefficient
        k1 = lower.k1 + t * (upper.k1 - lower.k1)
        k2 = lower.k2 + t * (upper.k2 - lower.k2)
        k3 = lower.k3 + t * (upper.k3 - lower.k3)
        p1 = lower.p1 + t * (upper.p1 - lower.p1)
        p2 = lower.p2 + t * (upper.p2 - lower.p2)

        logger.debug(
            f"Interpolated zoom {target_zoom} between {lower_zoom} and {upper_zoom} (t={t:.3f})"
        )

        return DistortionCoefficients(
            k1=Unitless(k1),
            k2=Unitless(k2),
            p1=Unitless(p1),
            p2=Unitless(p2),
            k3=Unitless(k3),
        )

    def get_intrinsics(self, zoom_factor: float) -> dict[str, float] | None:
        """Get interpolated intrinsics for a zoom level.

        Returns dict with fx, fy, cx, cy if any entry stores intrinsics,
        otherwise None.

        Args:
            zoom_factor: The desired zoom level.

        Returns:
            Dict with intrinsic values, or None if no intrinsics stored.
        """
        if not self.entries:
            return None

        # Check if any entry has intrinsics
        if not any(e.fx != 0.0 or e.fy != 0.0 for e in self.entries.values()):
            return None

        # Check for exact match
        matching_zoom = self._find_matching_zoom(zoom_factor)
        if matching_zoom is not None:
            e = self.entries[matching_zoom]
            return {"fx": e.fx, "fy": e.fy, "cx": e.cx, "cy": e.cy}

        zoom_levels = sorted(self.entries.keys())

        # Edge cases: clamp to bounds
        if zoom_factor <= zoom_levels[0]:
            e = self.entries[zoom_levels[0]]
            return {"fx": e.fx, "fy": e.fy, "cx": e.cx, "cy": e.cy}
        if zoom_factor >= zoom_levels[-1]:
            e = self.entries[zoom_levels[-1]]
            return {"fx": e.fx, "fy": e.fy, "cx": e.cx, "cy": e.cy}

        # Find bracketing zoom levels and interpolate
        for i in range(len(zoom_levels) - 1):
            if zoom_levels[i] <= zoom_factor <= zoom_levels[i + 1]:
                lower = self.entries[zoom_levels[i]]
                upper = self.entries[zoom_levels[i + 1]]
                t = (zoom_factor - zoom_levels[i]) / (zoom_levels[i + 1] - zoom_levels[i])
                return {
                    "fx": lower.fx + t * (upper.fx - lower.fx),
                    "fy": lower.fy + t * (upper.fy - lower.fy),
                    "cx": lower.cx + t * (upper.cx - lower.cx),
                    "cy": lower.cy + t * (upper.cy - lower.cy),
                }

        return None

    def get_zoom_levels(self) -> list[float]:
        """Get list of calibrated zoom levels."""
        return sorted(self.entries.keys())

    def get_entry(self, zoom_factor: float) -> ZoomCalibrationEntry | None:
        """Get the exact calibration entry for a zoom level.

        Unlike get_coefficients, this does not interpolate.

        Args:
            zoom_factor: The zoom level to look up.

        Returns:
            ZoomCalibrationEntry if found, None otherwise.
        """
        matching_zoom = self._find_matching_zoom(zoom_factor)
        if matching_zoom is not None:
            return self.entries[matching_zoom]
        return None

    def get_nearest_entry(self, zoom_factor: float) -> ZoomCalibrationEntry | None:
        """Get the calibration entry nearest to a zoom level.

        Args:
            zoom_factor: The target zoom level.

        Returns:
            Nearest ZoomCalibrationEntry, or None if no entries.
        """
        if not self.entries:
            return None

        nearest_zoom = min(self.entries.keys(), key=lambda z: abs(z - zoom_factor))
        return self.entries[nearest_zoom]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "camera_id": self.camera_id,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraCalibrationTable:
        """Create from dictionary.

        Raises:
            ValueError: If data is None or not a dict.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for calibration data, got {type(data).__name__}")

        table = cls(
            camera_id=data["camera_id"],
            created_date=data.get("created_date", ""),
            last_modified=data.get("last_modified", ""),
        )

        for entry_data in data.get("entries", []):
            entry = ZoomCalibrationEntry.from_dict(entry_data)
            table.entries[entry.zoom_factor] = entry

        return table

    def save(self, path: str | Path) -> None:
        """Save calibration table to YAML file.

        Args:
            path: Path to output file.
        """
        path = Path(path)
        data = self.to_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved calibration table to {path}")

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibrationTable:
        """Load calibration table from YAML file.

        Args:
            path: Path to input file.

        Returns:
            Loaded CameraCalibrationTable.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Calibration file is empty: {path}")

        table = cls.from_dict(data)
        logger.info(f"Loaded calibration table from {path} with {len(table.entries)} entries")

        return table

    def summary(self) -> str:
        """Generate a summary string of the calibration table."""
        lines = [
            f"Camera Calibration Table: {self.camera_id}",
            f"Created: {self.created_date}",
            f"Modified: {self.last_modified}",
            f"Entries: {len(self.entries)}",
            "",
        ]

        if self.entries:
            # Check if any entry has intrinsics stored
            has_intrinsics = any(e.fx != 0.0 for e in self.entries.values())
            if has_intrinsics:
                lines.append("Zoom Level | k1        | k2        | RMSE   | fx       | fy")
                lines.append("-" * 70)
            else:
                lines.append("Zoom Level | k1        | k2        | RMSE")
                lines.append("-" * 50)

            for zoom in sorted(self.entries.keys()):
                entry = self.entries[zoom]
                if has_intrinsics:
                    lines.append(
                        f"{zoom:10.1f} | {entry.k1:9.6f} | {entry.k2:9.6f} | {entry.validation_rmse:.4f} | {entry.fx:8.1f} | {entry.fy:.1f}"
                    )
                else:
                    lines.append(
                        f"{zoom:10.1f} | {entry.k1:9.6f} | {entry.k2:9.6f} | {entry.validation_rmse:.4f}"
                    )

        return "\n".join(lines)


def load_calibration_for_camera(
    camera_id: str,
    calibration_dir: str | Path,
) -> CameraCalibrationTable | None:
    """Load calibration table for a camera from standard location.

    Looks for file named '{camera_id}_calibration.yaml' in calibration_dir.

    Args:
        camera_id: Camera identifier.
        calibration_dir: Directory containing calibration files.

    Returns:
        CameraCalibrationTable if found, None otherwise.
    """
    calibration_dir = Path(calibration_dir)
    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    if calibration_file.exists():
        return CameraCalibrationTable.load(calibration_file)

    logger.debug(f"No calibration file found for camera {camera_id}")
    return None


def load_lens_distortion_as_table(
    camera_id: str,
    calibration_dir: str | Path,
) -> dict[float, dict[str, float]] | None:
    """Load lens distortion calibration as a plain dict keyed by zoom factor.

    Converts the YAML-based ``CameraCalibrationTable`` into the lightweight
    ``dict[float, dict[str, float]]`` format expected by
    ``IntrinsicExtrinsicHomography.calibration_table``.

    Args:
        camera_id: Camera identifier.
        calibration_dir: Directory containing calibration YAML files.

    Returns:
        Dictionary mapping zoom_factor to ``{k1, k2, k3, p1, p2}`` dicts,
        or ``None`` if no calibration file exists.
    """
    calibration_dir = Path(calibration_dir)
    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    if not calibration_file.exists():
        return None

    try:
        table = CameraCalibrationTable.load(calibration_file)
        if not table.entries:
            return None

        result: dict[float, dict[str, float]] = {}
        for zoom, entry in table.entries.items():
            result[zoom] = {
                "k1": entry.k1,
                "k2": entry.k2,
                "k3": entry.k3,
                "p1": entry.p1,
                "p2": entry.p2,
            }
        return result
    except Exception:
        logger.warning(f"Failed to load lens distortion calibration for {camera_id}", exc_info=True)
        return None
