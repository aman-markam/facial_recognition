"""Add latitude and longitude to attendance

Revision ID: e74a12948f10
Revises: ac6624e2e97d
Create Date: 2026-09-22 14:22:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e74a12948f10'
down_revision: Union[str, Sequence[str], None] = 'ac6624e2e97d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('attendance', sa.Column('latitude', sa.Float(), nullable=True))
    op.add_column('attendance', sa.Column('longitude', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('attendance', 'longitude')
    op.drop_column('attendance', 'latitude')
