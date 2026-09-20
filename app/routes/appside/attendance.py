from datetime import datetime, time, timedelta

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.employee import Employee
from app.models.attendance import Attendance
from face_engine import FaceEngine


OFFICE_START_TIME = time(9, 30)


router = APIRouter(
    prefix="/api/v1/appside/attendance",
    tags=["Appside Attendance"]
)


face_engine = FaceEngine()


@router.post("/mark")
async def mark_attendance(
    image: UploadFile = File(...),
    employee_code: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    action: str | None = Form(default=None),
    db: Session = Depends(get_db)
):

    # ==========================================
    # READ IMAGE
    # ==========================================

    image_bytes = await image.read()

    if not image_bytes:

        return {
            "success": False,
            "message": "Empty image"
        }


    # ==========================================
    # PRE-CHECK EMPLOYEE IF CODE PROVIDED BY KIOSK
    # ==========================================

    provided_code = employee_code.strip() if (employee_code and employee_code.strip()) else None
    if provided_code:
        emp_check = (
            db.query(Employee)
            .filter(Employee.employee_code == provided_code)
            .first()
        )
        if not emp_check and provided_code.isdigit():
            emp_check = (
                db.query(Employee)
                .filter(Employee.id == int(provided_code))
                .first()
            )
        if not emp_check:
            return {
                "success": False,
                "message": f"Employee ID / Code '{provided_code}' is not registered in database"
            }
        if not emp_check.is_active:
            return {
                "success": False,
                "message": f"Employee '{provided_code}' is inactive"
            }

    # ==========================================
    # FACE RECOGNITION (INCLUDES OCCLUSION & LIVENESS GATES)
    # ==========================================

    result = face_engine.recognize(
        image_bytes
    )

    if not result["success"]:
        return result


    recognized_code = (
        result["employee_id"]
    )

    confidence = (
        result["confidence"]
    )

    target_code = provided_code if provided_code else recognized_code

    if provided_code and recognized_code != target_code:
        return {
            "success": False,
            "message": f"Captured face does not match employee code '{target_code}'"
        }


    # ==========================================
    # FIND EMPLOYEE IN DATABASE
    # ==========================================

    employee = (
        db.query(Employee)
        .filter(
            Employee.employee_code == target_code
        )
        .first()
    )

    if not employee and target_code.isdigit():
        employee = (
            db.query(Employee)
            .filter(
                Employee.id == int(target_code)
            )
            .first()
        )


    if not employee:

        return {
            "success": False,
            "message": f"Employee ID '{target_code}' is not registered in database"
        }


    # ==========================================
    # ACTIVE EMPLOYEE CHECK
    # ==========================================

    if not employee.is_active:

        return {
            "success": False,
            "message": "Employee is inactive"
        }


    # ==========================================
    # CURRENT DATE / TIME
    # ==========================================

    now = datetime.now()

    today = now.date()

    yesterday = today - timedelta(days=1)

    current_time = now.time()

    requested_action = (mode or action or "").lower().strip()


    # ==========================================
    # FIND OPEN ATTENDANCE (TODAY OR YESTERDAY)
    # ==========================================

    open_attendance = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee.id,
            Attendance.check_out.is_(None),
            Attendance.attendance_date.in_([today, yesterday])
        )
        .order_by(
            Attendance.attendance_date.desc()
        )
        .first()
    )


    # ==========================================
    # EXPLICIT MODE VALIDATIONS
    # ==========================================

    if requested_action in ["check_out", "checkout", "out"] and not open_attendance:
        return {
            "success": False,
            "message": "No active check-in record found to check out."
        }

    if requested_action in ["check_in", "checkin", "in"] and open_attendance:
        return {
            "success": True,
            "message": "Employee is already checked in",
            "employee_id": employee.employee_code,
            "employee_name": employee.name,
            "attendance_id": open_attendance.id,
            "check_in": open_attendance.check_in,
            "check_out": open_attendance.check_out,
            "working_minutes": open_attendance.working_minutes,
            "status": open_attendance.status,
            "confidence": float(open_attendance.confidence) if open_attendance.confidence else confidence
        }


    # ==========================================
    # CASE 1: CHECK-OUT OPEN ATTENDANCE
    # ==========================================

    if open_attendance:

        open_attendance.check_out = current_time

        check_in_datetime = datetime.combine(
            open_attendance.attendance_date,
            open_attendance.check_in
        )

        check_out_datetime = datetime.combine(
            today,
            current_time
        )

        working_seconds = (
            check_out_datetime - check_in_datetime
        ).total_seconds()

        if working_seconds < 0:
            working_seconds = 0

        open_attendance.working_minutes = int(
            working_seconds // 60
        )

        db.commit()

        db.refresh(open_attendance)

        return {
            "success": True,
            "message": "Check-out successful",

            "employee_id":
                employee.employee_code,

            "employee_name":
                employee.name,

            "attendance_id":
                open_attendance.id,

            "check_in":
                open_attendance.check_in,

            "check_out":
                open_attendance.check_out,

            "working_minutes":
                open_attendance.working_minutes,

            "status":
                open_attendance.status,

            "confidence":
                float(
                    open_attendance.confidence
                )
                if open_attendance.confidence
                else confidence
        }


    # ==========================================
    # CASE 2: CHECK IF ALREADY COMPLETED TODAY
    # ==========================================

    today_completed = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee.id,
            Attendance.attendance_date == today,
            Attendance.check_out.is_not(None)
        )
        .first()
    )

    if today_completed:

        return {
            "success": True,
            "message": "Attendance already completed",

            "employee_id":
                employee.employee_code,

            "employee_name":
                employee.name,

            "attendance_id":
                today_completed.id,

            "check_in":
                today_completed.check_in,

            "check_out":
                today_completed.check_out,

            "working_minutes":
                today_completed.working_minutes,

            "status":
                today_completed.status,

            "confidence":
                float(
                    today_completed.confidence
                )
                if today_completed.confidence
                else confidence
        }


    # ==========================================
    # CASE 3: NEW CHECK-IN FOR TODAY
    # ==========================================

    if current_time > OFFICE_START_TIME:
        status = "LATE"
    else:
        status = "PRESENT"

    attendance = Attendance(
        employee_id=employee.id,
        attendance_date=today,
        check_in=current_time,
        check_out=None,
        working_minutes=None,
        status=status,
        confidence=confidence
    )

    db.add(attendance)

    try:

        db.commit()

        db.refresh(attendance)

    except IntegrityError:

        db.rollback()

        attendance = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id == employee.id,
                Attendance.attendance_date == today
            )
            .first()
        )

        if not attendance:

            return {
                "success": False,
                "message": "Unable to create attendance record"
            }

    return {
        "success": True,
        "message": "Check-in successful",

        "employee_id":
            employee.employee_code,

        "employee_name":
            employee.name,

        "attendance_id":
            attendance.id,

        "check_in":
            attendance.check_in,

        "check_out":
            attendance.check_out,

        "working_minutes":
            attendance.working_minutes,

        "status":
            attendance.status,

        "confidence":
            float(
                attendance.confidence
            )
            if attendance.confidence
            else confidence
    }