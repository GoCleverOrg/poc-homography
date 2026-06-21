"""Persistence entity for a per-PTZ-state extrinsic registration record.

This entity is purely for persistence (#364). It is the missing storage half of
the extrinsic solvers shipped in #340: it carries the
:class:`~poc_homography.calibration.extrinsic.ptz_registration.PtzRegistrationResult`
produced by ``register_frame_lines`` / ``validate_with_holdout`` /
``consolidate_ptz_frames`` together with the optional meters-unit
:class:`~poc_homography.calibration.extrinsic.validation.ReprojectionStats`
those validation paths report.

The ``calibration/extrinsic`` dataclasses remain the single source of truth —
this module only (de)serializes them. It deliberately adds **no** new metric
fields and re-solves nothing.

Records are keyed by ``(camera_id, run_id, PTZ-state)`` via :meth:`make_id`, so a
single camera can hold many registrations (one per survey run and commanded
zoom/tilt). The ``camera_id`` is preserved as a first-class attribute so later
intrinsic+extrinsic joins can group by camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
from poc_homography.calibration.extrinsic.validation import ReprojectionStats

# Scalar (non-matrix) metric fields carried verbatim from PtzRegistrationResult.
_METRIC_FIELDS = (
    "num_lines",
    "num_inliers",
    "inlier_ratio",
    "mean_perp_error",
    "max_perp_error",
    "rmse",
)


def result_to_dict(result: PtzRegistrationResult) -> dict[str, Any]:
    """Serialize a :class:`PtzRegistrationResult` to a JSON-friendly dict.

    The 3x3 homography/inverse matrices become nested lists; every other field
    is copied verbatim.

    Args:
        result: The solver result to serialize.

    Returns:
        A dict with ``homography_matrix``/``inverse_matrix`` as nested lists and
        all scalar metrics plus PTZ provenance.
    """
    return {
        "homography_matrix": np.asarray(result.homography_matrix, dtype=np.float64).tolist(),
        "inverse_matrix": np.asarray(result.inverse_matrix, dtype=np.float64).tolist(),
        "num_lines": int(result.num_lines),
        "num_inliers": int(result.num_inliers),
        "inlier_ratio": float(result.inlier_ratio),
        "mean_perp_error": float(result.mean_perp_error),
        "max_perp_error": float(result.max_perp_error),
        "rmse": float(result.rmse),
        "run_id": result.run_id,
        "camera_id": result.camera_id,
        "commanded_zoom": float(result.commanded_zoom),
        "commanded_tilt": float(result.commanded_tilt),
    }


def result_from_dict(data: dict[str, Any]) -> PtzRegistrationResult:
    """Reconstruct a :class:`PtzRegistrationResult` from :func:`result_to_dict`.

    Args:
        data: A dict produced by :func:`result_to_dict` (matrices as nested
            lists).

    Returns:
        The rebuilt result with ``float64`` numpy matrices.
    """
    return PtzRegistrationResult(
        homography_matrix=np.asarray(data["homography_matrix"], dtype=np.float64),
        inverse_matrix=np.asarray(data["inverse_matrix"], dtype=np.float64),
        num_lines=int(data["num_lines"]),
        num_inliers=int(data["num_inliers"]),
        inlier_ratio=float(data["inlier_ratio"]),
        mean_perp_error=float(data["mean_perp_error"]),
        max_perp_error=float(data["max_perp_error"]),
        rmse=float(data["rmse"]),
        run_id=data.get("run_id"),
        camera_id=data.get("camera_id"),
        commanded_zoom=float(data["commanded_zoom"]),
        commanded_tilt=float(data["commanded_tilt"]),
    )


def stats_to_dict(stats: ReprojectionStats) -> dict[str, Any]:
    """Serialize a :class:`ReprojectionStats` to a JSON-friendly dict."""
    return {
        "rms_m": float(stats.rms_m),
        "p90_m": float(stats.p90_m),
        "max_m": float(stats.max_m),
        "n_points": int(stats.n_points),
    }


def stats_from_dict(data: dict[str, Any]) -> ReprojectionStats:
    """Reconstruct a :class:`ReprojectionStats` from :func:`stats_to_dict`."""
    return ReprojectionStats(
        rms_m=float(data["rms_m"]),
        p90_m=float(data["p90_m"]),
        max_m=float(data["max_m"]),
        n_points=int(data["n_points"]),
    )


def metrics_of(result: PtzRegistrationResult) -> dict[str, Any]:
    """Return just the scalar solver metrics of ``result`` (no matrices/PTZ)."""
    full = result_to_dict(result)
    return {key: full[key] for key in _METRIC_FIELDS}


@dataclass(frozen=True, eq=False)
class PtzRegistration:
    """A persisted per-PTZ-state extrinsic registration for one camera.

    Attributes:
        camera_id: Camera this registration belongs to (the join key). Required.
        run_id: Survey run this PTZ state came from (None if not applicable).
        commanded_zoom: Zoom the camera was commanded to for this frame.
        commanded_tilt: Tilt the camera was commanded to for this frame.
        registration: The solver result (source of truth for matrices+metrics).
        reprojection_stats: Optional meters-unit reprojection summary from
            ``validate_with_holdout`` / ``consolidate_ptz_frames``.
        created_date: ISO-format creation timestamp.
        last_modified: ISO-format last modification timestamp.
    """

    camera_id: str
    run_id: str | None
    commanded_zoom: float
    commanded_tilt: float
    registration: PtzRegistrationResult
    reprojection_stats: ReprojectionStats | None = None
    created_date: str = ""
    last_modified: str = ""

    @staticmethod
    def make_id(
        camera_id: str,
        run_id: str | None,
        commanded_zoom: float,
        commanded_tilt: float,
    ) -> str:
        """Build the opaque composite id for a ``(camera, run, PTZ-state)`` key.

        Args:
            camera_id: Camera identifier.
            run_id: Survey run id, or None when not run-scoped.
            commanded_zoom: Commanded zoom of the PTZ state.
            commanded_tilt: Commanded tilt of the PTZ state.

        Returns:
            A stable string id of the form ``"{camera}/{run}/z{zoom}/t{tilt}"``.
            ``repr`` is used for the floats so two distinct ``float64`` PTZ states
            never collapse to the same key (and the recomputed id always matches
            the stored one round-trip).
        """
        run = run_id if run_id is not None else "_norun"
        return f"{camera_id}/{run}/z{float(commanded_zoom)!r}/t{float(commanded_tilt)!r}"

    @property
    def id(self) -> str:
        """Composite ``(camera_id, run_id, PTZ-state)`` identifier."""
        return self.make_id(self.camera_id, self.run_id, self.commanded_zoom, self.commanded_tilt)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_result(
        cls,
        registration: PtzRegistrationResult,
        *,
        reprojection_stats: ReprojectionStats | None = None,
        created_date: str = "",
        last_modified: str = "",
    ) -> PtzRegistration:
        """Build a persistable record from a solver result.

        The camera/run/zoom/tilt key components are taken from ``registration``;
        a ``camera_id`` is mandatory since it is the join key.

        Args:
            registration: The solver result to persist.
            reprojection_stats: Optional meters-unit reprojection summary.
            created_date: ISO-format creation timestamp.
            last_modified: ISO-format last modification timestamp.

        Returns:
            The wrapped :class:`PtzRegistration`.

        Raises:
            ValueError: If ``registration.camera_id`` is None or empty.
        """
        if not registration.camera_id:
            raise ValueError("registration.camera_id is required to persist a PtzRegistration")
        return cls(
            camera_id=registration.camera_id,
            run_id=registration.run_id,
            commanded_zoom=registration.commanded_zoom,
            commanded_tilt=registration.commanded_tilt,
            registration=registration,
            reprojection_stats=reprojection_stats,
            created_date=created_date,
            last_modified=last_modified,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "run_id": self.run_id,
            "commanded_zoom": self.commanded_zoom,
            "commanded_tilt": self.commanded_tilt,
            "registration": result_to_dict(self.registration),
            "reprojection_stats": (
                stats_to_dict(self.reprojection_stats)
                if self.reprojection_stats is not None
                else None
            ),
            "created_date": self.created_date,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PtzRegistration:
        """Create a :class:`PtzRegistration` from a dictionary.

        Raises:
            ValueError: If ``camera_id`` is missing or empty — the join key is
                mandatory, mirroring :meth:`from_result`.
        """
        camera_id = data.get("camera_id")
        if not camera_id:
            raise ValueError("camera_id is required to load a PtzRegistration")
        stats = data.get("reprojection_stats")
        return cls(
            camera_id=camera_id,
            run_id=data.get("run_id"),
            commanded_zoom=float(data["commanded_zoom"]),
            commanded_tilt=float(data["commanded_tilt"]),
            registration=result_from_dict(data["registration"]),
            reprojection_stats=stats_from_dict(stats) if stats is not None else None,
            created_date=data.get("created_date", ""),
            last_modified=data.get("last_modified", ""),
        )
