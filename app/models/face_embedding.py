from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    __table_args__ = (
        UniqueConstraint(
            "employee_id",
            name="uq_face_embeddings_employee_id",
        ),
        UniqueConstraint(
            "employee_code",
            name="uq_face_embeddings_employee_code",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    employee_code: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    embedding: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    employee = relationship(
        "Employee",
        back_populates="face_embedding",
    )
