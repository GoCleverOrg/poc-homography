"""add asset_key/asset_url to maps

Revision ID: 3a9f1c4b2e7d
Revises: 121dcd273bee
Create Date: 2026-06-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3a9f1c4b2e7d"
down_revision: str | Sequence[str] | None = "121dcd273bee"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Object-storage reference for the GeoTIFF map asset. Nullable so existing
    # rows need no back-fill; populated by the upload pipeline (#290).
    op.add_column("maps", sa.Column("asset_key", sa.String(), nullable=True))
    op.add_column("maps", sa.Column("asset_url", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("maps", "asset_url")
    op.drop_column("maps", "asset_key")
