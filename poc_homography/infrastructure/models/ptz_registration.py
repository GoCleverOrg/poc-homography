"""SQLAlchemy ORM model for the :class:`PtzRegistration` entity."""

from __future__ import annotations

from sqlalchemy import JSON, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from poc_homography.infrastructure.database import Base

# Postgres uses JSONB; the SQLite variant lets the offline unit tests round-trip
# the JSON columns against an in-memory database without a live Postgres.
_JSON = JSONB().with_variant(JSON(), "sqlite")


class PtzRegistrationModel(Base):
    """Per-(camera, run, PTZ-state) extrinsic registration records.

    Domain entity: ``poc_homography.domain.entities.ptz_registration.PtzRegistration``

    Added additively (#364) as the persistence half of the extrinsic solvers
    from #340. Keyed by a synthetic ``id`` (``PtzRegistration.make_id``) so one
    camera holds many registrations; ``camera_id`` and ``run_id`` are indexed for
    later intrinsic+extrinsic joins and run-scoped lookups.

    Value objects stored as JSONB:
      - homography_matrix / inverse_matrix: 3x3 nested lists.
      - metrics: scalar ``PtzRegistrationResult`` metrics
        ``{num_lines, num_inliers, inlier_ratio, mean_perp_error, max_perp_error, rmse}``.
      - reprojection_stats: optional ``ReprojectionStats``
        ``{rms_m, p90_m, max_m, n_points}`` (None when not validated).
    """

    __tablename__ = "ptz_registrations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    commanded_zoom: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    commanded_tilt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    homography_matrix: Mapped[list] = mapped_column(_JSON, nullable=False)
    inverse_matrix: Mapped[list] = mapped_column(_JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(_JSON, nullable=False)
    reprojection_stats: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    created_date: Mapped[str] = mapped_column(String, nullable=False, default="")
    last_modified: Mapped[str] = mapped_column(String, nullable=False, default="")


__all__ = ["PtzRegistrationModel"]
