from datetime import datetime, time

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.employee import Employee
from app.models.attendance import Attendance
from face_engine import FaceEngine


# --------------------------------------------------
# Attendance settings
# --------------------------------------------------

OFFICE_START_TIME = time(9, 30)


# --------------------------------------------------
# Router
# --------------------------------------------------

router = APIRouter(
    prefix="/api/v1/appside/attendance",
    tags=["Appside Attendance"]
)


# --------------------------------------------------
# Face recognition engine
# --------------------------------------------------

face_engine = FaceEngine()


# --------------------------------------------------
# Mark attendance
# --------------------------------------------------

@router.post("/mark")
async def mark_attendance(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # ==================================================
    # 1. Read uploaded image
    # ==================================================

    image_bytes = await image.read()

    # ==================================================
    # 2. Recognize face
    # ==================================================

    result = face_engine.recognize(
        image_bytes
    )

    # Face recognition failed
    if not result["success"]:
        return result

    employee_code = result["employee_id"]
    confidence = result["confidence"]

    # ==================================================
    # 3. Find employee in database
    # ==================================================

    employee = (
        db.query(Employee)
        .filter(
            Employee.employee_code == employee_code
        )
        .first()
    )

    if not employee:
        return {
            "success": False,
            "message": "Employee not found"
        }

    # ==================================================
    # 4. Check employee status
    # ==================================================

    if not employee.is_active:
        return {
            "success": False,
            "message": "Employee is inactive"
        }

    # ==================================================
    # 5. Get current date and time
    # ==================================================

    now = datetime.now()

    today = now.date()
    current_time = now.time()

    # ==================================================
    # 6. Find today's attendance record
    # ==================================================

    attendance = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee.id,
            Attendance.attendance_date == today
        )
        .first()
    )

    # ==================================================
    # CASE 1: No attendance record
    # → CHECK-IN
    # ==================================================

    if not attendance:

        # Determine attendance status
        if current_time > OFFICE_START_TIME:
            status = "LATE"
        else:
            status = "PRESENT"

        # Create attendance record
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
        db.commit()
        db.refresh(attendance)

        return {
            "success": True,
            "message": "Check-in successful",
            "employee_id": employee.employee_code,
            "employee_name": employee.name,
            "attendance_id": attendance.id,
            "check_in": attendance.check_in,
            "check_out": attendance.check_out,
            "working_minutes": attendance.working_minutes,
            "status": attendance.status,
            "confidence": confidence
        }

    # ==================================================
    # CASE 2: Checked in but NOT checked out
    # → CHECK-OUT
    # ==================================================

    if attendance.check_in and not attendance.check_out:

        # Set checkout time
        attendance.check_out = current_time

        # --------------------------------------------------
        # Calculate working time
        # --------------------------------------------------

        check_in_datetime = datetime.combine(
            today,
            attendance.check_in
        )

        check_out_datetime = datetime.combine(
            today,
            current_time
        )

        working_seconds = (
            check_out_datetime - check_in_datetime
        ).total_seconds()

        # Convert seconds → minutes
        attendance.working_minutes = int(
            working_seconds // 60
        )

        # Save changes
        db.commit()
        db.refresh(attendance)

        return {
            "success": True,
            "message": "Check-out successful",
            "employee_id": employee.employee_code,
            "employee_name": employee.name,
            "attendance_id": attendance.id,
            "check_in": attendance.check_in,
            "check_out": attendance.check_out,
            "working_minutes": attendance.working_minutes,
            "status": attendance.status,
            "confidence": float(
                attendance.confidence
            ) if attendance.confidence else confidence
        }

    # ==================================================
    # CASE 3: Already checked in AND checked out
    # → ATTENDANCE COMPLETED
    # ==================================================

    return {
        "success": True,
        "message": "Attendance already completed",
        "employee_id": employee.employee_code,
        "employee_name": employee.name,
        "attendance_id": attendance.id,
        "check_in": attendance.check_in,
        "check_out": attendance.check_out,
        "working_minutes": attendance.working_minutes,
        "status": attendance.status,
        "confidence": float(
            attendance.confidence
        ) if attendance.confidence else confidence
    }