"""create face_embeddings table

Revision ID: f8a21c90b3d4
Revises: cf79476ef280
Create Date: 2026-09-29 14:46:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f8a21c90b3d4"
down_revision: Union[str, Sequence[str], None] = "cf79476ef280"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "face_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employee_id", sa.Integer(), nullable=False),
        sa.Column("employee_code", sa.String(length=50), nullable=False),
        sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["employee_id"],
            ["employees.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", name="uq_face_embeddings_employee_id"),
        sa.UniqueConstraint("employee_code", name="uq_face_embeddings_employee_code"),
    )
    op.create_index(
        op.f("ix_face_embeddings_employee_id"),
        "face_embeddings",
        ["employee_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_face_embeddings_employee_code"),
        "face_embeddings",
        ["employee_code"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_face_embeddings_employee_code"), table_name="face_embeddings")
    op.drop_index(op.f("ix_face_embeddings_employee_id"), table_name="face_embeddings")
    op.drop_table("face_embeddings")
