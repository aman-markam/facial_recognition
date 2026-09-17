from datetime import date, datetime, time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


# ==================================================
# Basic attendance response
# ==================================================

class AttendanceResponse(BaseModel):

    id: int

    employee_id: int

    attendance_date: date

    check_in: time | None

    check_out: time | None

    working_minutes: int | None

    status: str

    confidence: Decimal | None

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


# ==================================================
# Attendance response with employee information
# ==================================================

class AttendanceEmployeeResponse(BaseModel):

    id: int

    employee_id: int

    employee_code: str

    employee_name: str

    attendance_date: date

    check_in: time | None

    check_out: time | None

    working_minutes: int | None

    status: str

    confidence: Decimal | None

    model_config = ConfigDict(
        from_attributes=True
    )


# ==================================================
# Attendance summary
# ==================================================

class AttendanceSummary(BaseModel):

    date: date

    total_employees: int

    present: int

    late: int

    absent: int