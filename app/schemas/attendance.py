from datetime import date, datetime, time
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


# ==================================================
# Attendance summary
# ==================================================

class AttendanceSummary(BaseModel):
    date: date
    total_employees: int
    present: int
    late: int
    absent: int


# ==================================================
# Multi-Person Attendance Response (Phase 4.5)
# ==================================================

class AttendanceResultItem(BaseModel):
    employee_code: Optional[str] = None
    name: Optional[str] = None
    live: bool = False
    recognition_confidence: float = 0.0
    live_score: float = 0.0
    spoof_score: float = 0.0
    attendance_action: str = "REJECTED"  # CHECK_IN, CHECK_OUT, ALREADY_RECORDED, REJECTED
    message: Optional[str] = None


class MultiPersonAttendanceResponse(BaseModel):
    success: bool = True
    processed_faces: int = 0
    results: List[AttendanceResultItem] = Field(default_factory=list)
    message: Optional[str] = None