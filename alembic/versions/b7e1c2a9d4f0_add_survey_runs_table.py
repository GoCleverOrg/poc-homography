"""add survey_runs table

Additive migration: creates the ``survey_runs`` table for the rich survey
dataset (#258). The existing ``survey_sessions`` table is untouched.

Revision ID: b7e1c2a9d4f0
Revises: f2b4cf9a31ad
Create Date: 2026-06-05
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "b7e1c2a9d4f0"
down_revision: str | None = "f2b4cf9a31ad"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "survey_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default=""),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_survey_runs_camera_id"), "survey_runs", ["camera_id"], unique=False)
    op.create_index(op.f("ix_survey_runs_status"), "survey_runs", ["status"], unique=False)
    op.create_index(
        op.f("ix_survey_runs_created_date"), "survey_runs", ["created_date"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_survey_runs_created_date"), table_name="survey_runs")
    op.drop_index(op.f("ix_survey_runs_status"), table_name="survey_runs")
    op.drop_index(op.f("ix_survey_runs_camera_id"), table_name="survey_runs")
    op.drop_table("survey_runs")
