"""initial schema: all entity tables and users

Revision ID: f2b4cf9a31ad
Revises:
Create Date: 2026-04-22
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "f2b4cf9a31ad"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- 1. tenants (root entity, no FK dependencies) ---
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("location_lat", sa.String(), nullable=False, server_default=""),
        sa.Column("location_lon", sa.String(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- 2. maps (depends on tenants) ---
    op.create_table(
        "maps",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("photo", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("geotiff", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_maps_tenant_id"), "maps", ["tenant_id"], unique=False)

    # --- 3. users (depends on tenants) ---
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)
    op.create_index(op.f("ix_users_tenant_id"), "users", ["tenant_id"], unique=False)

    # --- 4. camera_configs (depends on tenants, maps) ---
    op.create_table(
        "camera_configs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("map_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("spec", sa.String(), nullable=False),
        sa.Column("credential", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["map_id"], ["maps.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_camera_configs_tenant_id"), "camera_configs", ["tenant_id"], unique=False
    )
    op.create_index(
        op.f("ix_camera_configs_map_id"), "camera_configs", ["map_id"], unique=False
    )

    # --- 5. camera_calibrations (depends on camera_configs) ---
    op.create_table(
        "camera_calibrations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("position", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("base_orientation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("distortion", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["id"], ["camera_configs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- 6. lens_calibration_tables (depends on camera_configs) ---
    op.create_table(
        "lens_calibration_tables",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("entries", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_date", sa.String(), nullable=False, server_default=""),
        sa.Column("last_modified", sa.String(), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(["id"], ["camera_configs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- 7. captured_frames (depends on maps) ---
    op.create_table(
        "captured_frames",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("map_id", sa.String(), nullable=False),
        sa.Column("camera_name", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ptz_state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["map_id"], ["maps.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("map_id", "camera_name", "timestamp", name="uq_captured_frame_composite"),
    )
    op.create_index(
        op.f("ix_captured_frames_map_id"), "captured_frames", ["map_id"], unique=False
    )
    op.create_index(
        op.f("ix_captured_frames_camera_name"), "captured_frames", ["camera_name"], unique=False
    )

    # --- 8. ground_control_points (depends on maps) ---
    op.create_table(
        "ground_control_points",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("map_id", sa.String(), nullable=False),
        sa.Column("map_point", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["map_id"], ["maps.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("map_id", "name", name="uq_gcp_map_name"),
    )
    op.create_index(
        op.f("ix_ground_control_points_name"), "ground_control_points", ["name"], unique=False
    )
    op.create_index(
        op.f("ix_ground_control_points_map_id"), "ground_control_points", ["map_id"], unique=False
    )

    # --- 9. lines (depends on maps) ---
    op.create_table(
        "lines",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("map_id", sa.String(), nullable=False),
        sa.Column("start", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("end", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["map_id"], ["maps.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("map_id", "name", name="uq_line_map_name"),
    )
    op.create_index(op.f("ix_lines_name"), "lines", ["name"], unique=False)
    op.create_index(op.f("ix_lines_map_id"), "lines", ["map_id"], unique=False)

    # --- 10. annotations (depends on captured_frames) ---
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("gcp_id", sa.String(), nullable=False),
        sa.Column("frame_id", sa.String(), nullable=False),
        sa.Column("camera_pose", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("pixel", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["frame_id"], ["captured_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("frame_id", "gcp_id", name="uq_annotation_frame_gcp"),
    )
    op.create_index(op.f("ix_annotations_gcp_id"), "annotations", ["gcp_id"], unique=False)
    op.create_index(op.f("ix_annotations_frame_id"), "annotations", ["frame_id"], unique=False)

    # --- 11. line_annotations (depends on captured_frames) ---
    op.create_table(
        "line_annotations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("line_id", sa.String(), nullable=False),
        sa.Column("frame_id", sa.String(), nullable=False),
        sa.Column("camera_pose", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("start_pixel", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("end_pixel", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("points", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["frame_id"], ["captured_frames.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("frame_id", "line_id", name="uq_line_annotation_frame_line"),
    )
    op.create_index(
        op.f("ix_line_annotations_line_id"), "line_annotations", ["line_id"], unique=False
    )
    op.create_index(
        op.f("ix_line_annotations_frame_id"), "line_annotations", ["frame_id"], unique=False
    )

    # --- 12. calibration_line_trace_sets (standalone, no FK) ---
    op.create_table(
        "calibration_line_trace_sets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("image", sa.String(), nullable=False),
        sa.Column("camera_pose", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("line_traces", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- 13. diagnostic_sessions (standalone, no FK) ---
    op.create_table(
        "diagnostic_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False, server_default=""),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_diagnostic_sessions_tenant_id"), "diagnostic_sessions", ["tenant_id"], unique=False
    )
    op.create_index(
        op.f("ix_diagnostic_sessions_camera_id"), "diagnostic_sessions", ["camera_id"], unique=False
    )
    op.create_index(
        op.f("ix_diagnostic_sessions_created_date"),
        "diagnostic_sessions",
        ["created_date"],
        unique=False,
    )

    # --- 14. stress_test_sessions (standalone, no FK) ---
    op.create_table(
        "stress_test_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False, server_default=""),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_stress_test_sessions_tenant_id"),
        "stress_test_sessions",
        ["tenant_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_stress_test_sessions_camera_id"),
        "stress_test_sessions",
        ["camera_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_stress_test_sessions_created_date"),
        "stress_test_sessions",
        ["created_date"],
        unique=False,
    )

    # --- 15. survey_sessions (standalone, no FK) ---
    op.create_table(
        "survey_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False, server_default=""),
        sa.Column("created_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_survey_sessions_tenant_id"), "survey_sessions", ["tenant_id"], unique=False
    )
    op.create_index(
        op.f("ix_survey_sessions_camera_id"), "survey_sessions", ["camera_id"], unique=False
    )
    op.create_index(
        op.f("ix_survey_sessions_created_date"), "survey_sessions", ["created_date"], unique=False
    )


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.drop_table("survey_sessions")
    op.drop_table("stress_test_sessions")
    op.drop_table("diagnostic_sessions")
    op.drop_table("calibration_line_trace_sets")
    op.drop_table("line_annotations")
    op.drop_table("annotations")
    op.drop_table("lines")
    op.drop_table("ground_control_points")
    op.drop_table("captured_frames")
    op.drop_table("lens_calibration_tables")
    op.drop_table("camera_calibrations")
    op.drop_table("camera_configs")
    op.drop_table("users")
    op.drop_table("maps")
    op.drop_table("tenants")
