#!/usr/bin/env python3
"""
Calibration history persistence using YAML storage.

This module provides classes for storing and retrieving calibration results
in a persistent YAML file. This allows tracking of calibration performance
over time and retrieval of previously calibrated camera heights.

The history is stored in a YAML file with one entry per calibration session.
Each entry includes the estimated height, confidence bounds, point counts,
method used, and camera location information.

Usage Example:
    >>> from poc_homography.calibration_history import CalibrationHistory, CalibrationHistoryEntry
    >>> from datetime import datetime
    >>>
    >>> # Create a history manager (defaults to ~/.poc-homography/calibration_history.yaml)
    >>> history = CalibrationHistory()
    >>>
    >>> # Add a calibration result
    >>> entry = CalibrationHistoryEntry(
    ...     camera_name="Valte",
    ...     timestamp=datetime.now(),
    ...     estimated_height=5.23,
    ...     confidence_interval=(5.10, 5.36),
    ...     inlier_count=7,
    ...     outlier_count=2,
    ...     method="mad",
    ...     camera_gps_lat=39.640472,
    ...     camera_gps_lon=-0.230194
    ... )
    >>> history.add_entry(entry)
    >>> history.save()
    >>>
    >>> # Retrieve latest calibration for a camera
    >>> latest = history.get_latest("Valte")
    >>> if latest:
    ...     print(f"Last calibration: {latest.estimated_height}m")
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
import yaml


@dataclass
class CalibrationHistoryEntry:
    """
    A single calibration history entry with metadata and results.

    This dataclass represents one calibration session result that can be
    persisted to YAML storage. It includes all essential information needed
    to track calibration performance and retrieve past results.

    Attributes:
        camera_name: Identifier for the camera (e.g., "Valte", "Camera1")
        timestamp: When the calibration was performed
        estimated_height: The calibrated camera height in meters
        confidence_interval: Tuple of (lower_bound, upper_bound) in meters
        inlier_count: Number of inlier points used in calibration
        outlier_count: Number of outlier points rejected
        method: Calibration method used ('mad', 'ransac', 'simple')
        camera_gps_lat: Camera GPS latitude in decimal degrees
        camera_gps_lon: Camera GPS longitude in decimal degrees
    """
    camera_name: str
    timestamp: datetime
    estimated_height: float
    confidence_interval: Tuple[float, float]
    inlier_count: int
    outlier_count: int
    method: str
    camera_gps_lat: float
    camera_gps_lon: float

    def to_dict(self) -> dict:
        """
        Convert entry to dictionary for YAML serialization.

        Returns:
            Dictionary representation with timestamp as ISO format string
        """
        data = asdict(self)
        # Convert datetime to ISO format string
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationHistoryEntry':
        """
        Create entry from dictionary loaded from YAML.

        Args:
            data: Dictionary with entry data

        Returns:
            CalibrationHistoryEntry instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Convert timestamp string to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # Convert confidence_interval list to tuple
        if isinstance(data['confidence_interval'], list):
            data['confidence_interval'] = tuple(data['confidence_interval'])

        return cls(**data)


class CalibrationHistory:
    """
    Manages calibration history with YAML file persistence.

    This class provides a simple interface for storing and retrieving
    calibration results. Entries are stored in a YAML file and can be
    queried by camera name or retrieved in bulk.

    The default storage location is ~/.poc-homography/calibration_history.yaml
    but can be customized via the storage_path parameter.

    Attributes:
        storage_path: Path to the YAML storage file
        entries: List of CalibrationHistoryEntry objects currently loaded
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize calibration history manager.

        Args:
            storage_path: Optional custom path to YAML storage file.
                         If None, defaults to ~/.poc-homography/calibration_history.yaml

        Note:
            The storage directory will be created if it doesn't exist.
            If the storage file doesn't exist, it will be created on first save.
        """
        if storage_path is None:
            # Default to user's home directory
            home = Path.home()
            storage_dir = home / '.poc-homography'
            self.storage_path = storage_dir / 'calibration_history.yaml'
        else:
            self.storage_path = Path(storage_path)

        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize entries list
        self.entries: List[CalibrationHistoryEntry] = []

        # Load existing history if file exists
        if self.storage_path.exists():
            self.load()

    def add_entry(self, entry: CalibrationHistoryEntry) -> None:
        """
        Add a new calibration entry to the history.

        The entry is added to the in-memory list. Call save() to persist
        the changes to disk.

        Args:
            entry: CalibrationHistoryEntry to add

        Note:
            Entries are automatically sorted by timestamp (newest first)
            after adding.
        """
        self.entries.append(entry)
        # Sort entries by timestamp, newest first
        self.entries.sort(key=lambda e: e.timestamp, reverse=True)

    def get_entries(
        self,
        camera_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[CalibrationHistoryEntry]:
        """
        Get calibration entries, optionally filtered and limited.

        Args:
            camera_name: If provided, only return entries for this camera.
                        If None, return entries for all cameras.
            limit: Maximum number of entries to return. If None, return all.

        Returns:
            List of CalibrationHistoryEntry objects, sorted by timestamp
            (newest first)

        Example:
            >>> # Get all entries
            >>> all_entries = history.get_entries()
            >>>
            >>> # Get last 10 entries for "Valte" camera
            >>> valte_recent = history.get_entries(camera_name="Valte", limit=10)
        """
        # Filter by camera name if specified
        if camera_name is not None:
            filtered_entries = [
                e for e in self.entries
                if e.camera_name == camera_name
            ]
        else:
            filtered_entries = self.entries.copy()

        # Apply limit if specified
        if limit is not None:
            filtered_entries = filtered_entries[:limit]

        return filtered_entries

    def get_latest(self, camera_name: str) -> Optional[CalibrationHistoryEntry]:
        """
        Get the most recent calibration entry for a specific camera.

        Args:
            camera_name: Name of the camera to query

        Returns:
            Most recent CalibrationHistoryEntry for the camera, or None if
            no entries exist for this camera

        Example:
            >>> latest = history.get_latest("Valte")
            >>> if latest:
            ...     print(f"Height: {latest.estimated_height}m")
            ...     print(f"Calibrated: {latest.timestamp}")
        """
        camera_entries = self.get_entries(camera_name=camera_name, limit=1)
        return camera_entries[0] if camera_entries else None

    def save(self) -> None:
        """
        Persist the current calibration history to YAML file.

        This method serializes all entries to YAML format and writes them
        to the storage file. The file will be created if it doesn't exist.

        Raises:
            IOError: If the file cannot be written
            yaml.YAMLError: If serialization fails

        Note:
            The YAML file uses safe_dump for security. Only standard Python
            types are serialized.
        """
        # Convert entries to dictionaries
        data = {
            'calibrations': [entry.to_dict() for entry in self.entries]
        }

        # Write to YAML file
        with open(self.storage_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self) -> None:
        """
        Load calibration history from YAML file.

        This method reads the YAML file and deserializes entries into
        CalibrationHistoryEntry objects. If the file doesn't exist,
        the entries list is initialized as empty.

        Raises:
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If entry data is invalid or missing required fields

        Note:
            This method is automatically called during __init__ if the
            storage file exists. You typically don't need to call it manually
            unless you want to reload from disk.
        """
        # If file doesn't exist, start with empty list
        if not self.storage_path.exists():
            self.entries = []
            return

        # Load from YAML file
        with open(self.storage_path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle empty file or missing calibrations key
        if data is None or 'calibrations' not in data:
            self.entries = []
            return

        # Convert dictionaries to CalibrationHistoryEntry objects
        self.entries = [
            CalibrationHistoryEntry.from_dict(entry_data)
            for entry_data in data['calibrations']
        ]

        # Sort entries by timestamp, newest first
        self.entries.sort(key=lambda e: e.timestamp, reverse=True)

    def clear(self, camera_name: Optional[str] = None) -> None:
        """
        Clear calibration history entries.

        Args:
            camera_name: If provided, only clear entries for this camera.
                        If None, clear all entries.

        Note:
            Changes are made to the in-memory list only. Call save() to
            persist the deletion to disk.

        Example:
            >>> # Clear all entries
            >>> history.clear()
            >>> history.save()
            >>>
            >>> # Clear only entries for "Valte" camera
            >>> history.clear(camera_name="Valte")
            >>> history.save()
        """
        if camera_name is None:
            # Clear all entries
            self.entries = []
        else:
            # Clear only entries for specified camera
            self.entries = [
                e for e in self.entries
                if e.camera_name != camera_name
            ]
