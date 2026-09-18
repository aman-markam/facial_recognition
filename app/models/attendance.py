from datetime import date, datetime, time

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database.base import Base


class Attendance(Base):

    __tablename__ = "attendance"

    __table_args__ = (

        # One attendance record per employee per day
        UniqueConstraint(
            "employee_id",
            "attendance_date",
            name="uq_employee_attendance_date",
        ),

        # Useful for employee attendance history
        Index(
            "ix_attendance_employee_date",
            "employee_id",
            "attendance_date",
        ),

        # Useful for dashboard/report date filtering
        Index(
            "ix_attendance_date",
            "attendance_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    employee_id: Mapped[int] = mapped_column(
        ForeignKey(
            "employees.id"
        ),
        nullable=False,
        index=True,
    )

    attendance_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )

    check_in: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    check_out: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    working_minutes: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="PRESENT",
        nullable=False,
    )

    confidence: Mapped[float | None] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    employee = relationship(
        "Employee",
        backref="attendance_records",
    )