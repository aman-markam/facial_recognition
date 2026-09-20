from datetime import date
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.employee import Employee
from app.models.attendance import Attendance
from app.schemas.attendance import (
    AttendanceResponse,
    AttendanceEmployeeResponse,
    AttendanceSummary,
)
from app.dependencies.auth import get_current_admin


router = APIRouter(
    prefix="/attendance",
    tags=["Dashboard Attendance"],
    dependencies=[Depends(get_current_admin)]
)


# ==================================================
# 1. TODAY'S ATTENDANCE
# ==================================================

@router.get(
    "/today",
    response_model=list[AttendanceEmployeeResponse]
)
def get_today_attendance(
    db: Session = Depends(get_db)
):

    today = date.today()

    records = (
        db.query(
            Attendance.id,
            Attendance.employee_id,
            Employee.employee_code,
            Employee.name.label("employee_name"),
            Attendance.attendance_date,
            Attendance.check_in,
            Attendance.check_out,
            Attendance.working_minutes,
            Attendance.status,
            Attendance.confidence,
        )
        .join(
            Employee,
            Employee.id == Attendance.employee_id
        )
        .filter(
            Attendance.attendance_date == today
        )
        .order_by(
            Attendance.check_in
        )
        .all()
    )

    return records


# ==================================================
# 2. ATTENDANCE BY DATE
# ==================================================

@router.get(
    "/",
    response_model=list[AttendanceEmployeeResponse]
)
def get_attendance(
    attendance_date: date | None = Query(
        default=None
    ),
    db: Session = Depends(get_db)
):

    if attendance_date is None:
        attendance_date = date.today()

    records = (
        db.query(
            Attendance.id,
            Attendance.employee_id,
            Employee.employee_code,
            Employee.name.label("employee_name"),
            Attendance.attendance_date,
            Attendance.check_in,
            Attendance.check_out,
            Attendance.working_minutes,
            Attendance.status,
            Attendance.confidence,
        )
        .join(
            Employee,
            Employee.id == Attendance.employee_id
        )
        .filter(
            Attendance.attendance_date == attendance_date
        )
        .order_by(
            Attendance.check_in
        )
        .all()
    )

    return records


# ==================================================
# 3. EMPLOYEE ATTENDANCE HISTORY
# ==================================================

@router.get(
    "/employee/{employee_id}",
    response_model=list[AttendanceResponse]
)
def get_employee_attendance(
    employee_id: int,
    db: Session = Depends(get_db)
):

    records = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id
        )
        .order_by(
            Attendance.attendance_date.desc()
        )
        .all()
    )

    return records


# ==================================================
# 4. ATTENDANCE SUMMARY
# ==================================================

@router.get(
    "/summary",
    response_model=AttendanceSummary
)
def get_attendance_summary(
    attendance_date: date | None = Query(
        default=None
    ),
    db: Session = Depends(get_db)
):

    if attendance_date is None:
        attendance_date = date.today()

    total_employees = (
        db.query(Employee)
        .filter(
            Employee.is_active == True
        )
        .count()
    )

    present = (
        db.query(Attendance)
        .filter(
            Attendance.attendance_date == attendance_date,
            Attendance.status.in_(
                ["PRESENT", "LATE"]
            )
        )
        .count()
    )

    late = (
        db.query(Attendance)
        .filter(
            Attendance.attendance_date == attendance_date,
            Attendance.status == "LATE"
        )
        .count()
    )

    absent = total_employees - present

    return {
        "date": attendance_date,
        "total_employees": total_employees,
        "present": present,
        "late": late,
        "absent": absent
    }

@router.get(
    "/report",
    response_model=list[AttendanceEmployeeResponse]
)
def get_attendance_report(
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: Session = Depends(get_db)
):

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be greater than end_date"
        )

    records = (
        db.query(
            Attendance.id,
            Attendance.employee_id,
            Employee.employee_code,
            Employee.name.label("employee_name"),
            Attendance.attendance_date,
            Attendance.check_in,
            Attendance.check_out,
            Attendance.working_minutes,
            Attendance.status,
            Attendance.confidence,
        )
        .join(
            Employee,
            Employee.id == Attendance.employee_id
        )
        .filter(
            Attendance.attendance_date >= start_date,
            Attendance.attendance_date <= end_date
        )
        .order_by(
            Attendance.attendance_date.desc(),
            Attendance.check_in
        )
        .all()
    )

    return records