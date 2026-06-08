"""Per-camera catalog of stable pose ids across survey runs.

A :class:`PoseCatalog` maps a deterministic ``pose_id`` (from
:func:`~poc_homography.survey.planner.poses.canonical_pose_key`) to the
physical ``(pan, tilt, zoom)`` geometry. Because the id is derived purely from
quantised geometry, the same physical pose maps to the same ``pose_id`` across
runs and catalog instances — enabling multi-visit grouping by
``(camera_id, pose_id)``.

The entity is frozen; mutation helpers (:meth:`PoseCatalog.with_pose`) return a
new catalog. ``to_dict`` emits entries with deterministically-sorted keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.survey import (
    SURVEY_SCHEMA_VERSION,
    check_schema_version,
)
from poc_homography.survey.planner.poses import canonical_pose_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    from poc_homography.survey.planner.poses import Pose

CatalogEntry = tuple[float, float, float]
"""A cataloged pose as ``(pan, tilt, zoom)``."""


@dataclass(frozen=True, eq=False)
class PoseCatalog:
    """Catalog of stable pose ids for one camera.

    The ``id`` property is the ``catalog_id``, satisfying the ``Entity``
    protocol. ``entries`` maps ``pose_id -> (pan, tilt, zoom)``.
    """

    catalog_id: str
    camera_id: str
    entries: dict[str, CatalogEntry] = field(default_factory=dict)
    schema_version: str = field(default=SURVEY_SCHEMA_VERSION)

    @property
    def id(self) -> str:
        """Unique identifier — the catalog id."""
        return self.catalog_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @staticmethod
    def assign(pan: float, tilt: float, zoom: float) -> str:
        """Return the deterministic ``pose_id`` for a physical pose.

        Purely derived from geometry, so the same physical pose maps to the
        same id across catalog instances and runs. Does not mutate the catalog;
        use :meth:`with_pose` to record the entry.
        """
        return canonical_pose_key(pan, tilt, zoom)

    def with_pose(self, pan: float, tilt: float, zoom: float) -> PoseCatalog:
        """Return a new catalog that additionally contains this pose."""
        pose_id = self.assign(pan, tilt, zoom)
        new_entries = dict(self.entries)
        new_entries[pose_id] = (float(pan), float(tilt), float(zoom))
        return PoseCatalog(
            catalog_id=self.catalog_id,
            camera_id=self.camera_id,
            entries=new_entries,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_poses(
        cls,
        catalog_id: str,
        camera_id: str,
        poses: Iterable[Pose],
    ) -> PoseCatalog:
        """Build a catalog deterministically from a sequence of poses.

        Insertion order does not affect the result: a given physical pose
        always maps to the same ``pose_id``, and ``to_dict`` sorts keys.
        """
        entries: dict[str, CatalogEntry] = {}
        for pose in poses:
            pose_id = canonical_pose_key(float(pose.pan), float(pose.tilt), float(pose.zoom))
            entries[pose_id] = (float(pose.pan), float(pose.tilt), float(pose.zoom))
        return cls(catalog_id=catalog_id, camera_id=camera_id, entries=entries)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization with sorted entry keys."""
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "camera_id": self.camera_id,
            "entries": {
                pose_id: [
                    float(self.entries[pose_id][0]),
                    float(self.entries[pose_id][1]),
                    float(self.entries[pose_id][2]),
                ]
                for pose_id in sorted(self.entries)
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseCatalog:
        """Create :class:`PoseCatalog` from a dictionary.

        Raises:
            ValueError: If ``schema_version`` is unrecognised.
        """
        version = check_schema_version(str(data["schema_version"]))
        raw_entries = data.get("entries") or {}
        entries: dict[str, CatalogEntry] = {
            str(pose_id): (float(values[0]), float(values[1]), float(values[2]))
            for pose_id, values in raw_entries.items()
        }
        return cls(
            catalog_id=str(data["catalog_id"]),
            camera_id=str(data["camera_id"]),
            entries=entries,
            schema_version=version,
        )
