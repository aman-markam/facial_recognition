"""Add location_name to attendance

Revision ID: c5f891a24d7b
Revises: b3fab0549139
Create Date: 2026-09-22 16:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5f891a24d7b'
down_revision: Union[str, Sequence[str], None] = 'b3fab0549139'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('attendance', sa.Column('location_name', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('attendance', 'location_name')
