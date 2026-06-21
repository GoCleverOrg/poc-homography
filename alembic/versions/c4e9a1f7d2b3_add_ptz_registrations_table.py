"""add ptz_registrations table

Additive migration: creates the ``ptz_registrations`` table — the persistence
half (#364) of the extrinsic PTZ registration solvers from #340. One row per
``(camera_id, run_id, PTZ-state)`` keyed by a synthetic ``id``; the homography
matrices, scalar solver metrics, and optional meters-unit reprojection stats are
stored as JSONB. No existing table is touched.

Revision ID: c4e9a1f7d2b3
Revises: 3a9f1c4b2e7d
Create Date: 2026-06-21
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "c4e9a1f7d2b3"
down_revision: str | None = "3a9f1c4b2e7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ptz_registrations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False, server_default=""),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("commanded_zoom", sa.Float(), nullable=False, server_default="0"),
        sa.Column("commanded_tilt", sa.Float(), nullable=False, server_default="0"),
        sa.Column("homography_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("inverse_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reprojection_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_date", sa.String(), nullable=False, server_default=""),
        sa.Column("last_modified", sa.String(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_ptz_registrations_camera_id"), "ptz_registrations", ["camera_id"], unique=False
    )
    op.create_index(
        op.f("ix_ptz_registrations_run_id"), "ptz_registrations", ["run_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_ptz_registrations_run_id"), table_name="ptz_registrations")
    op.drop_index(op.f("ix_ptz_registrations_camera_id"), table_name="ptz_registrations")
    op.drop_table("ptz_registrations")
