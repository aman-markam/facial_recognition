from datetime import datetime, time

from fastapi import (
    APIRouter,
    Depends,
    File,
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
    # FACE RECOGNITION
    # ==========================================

    result = face_engine.recognize(
        image_bytes
    )


    if not result["success"]:

        return result


    employee_code = (
        result["employee_id"]
    )

    confidence = (
        result["confidence"]
    )


    # ==========================================
    # FIND EMPLOYEE
    # ==========================================

    employee = (
        db.query(Employee)
        .filter(
            Employee.employee_code
            == employee_code
        )
        .first()
    )


    if not employee:

        return {
            "success": False,
            "message": "Employee not found"
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

    current_time = now.time()


    # ==========================================
    # FIND TODAY'S ATTENDANCE
    # ==========================================

    attendance = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id
            == employee.id,

            Attendance.attendance_date
            == today
        )
        .first()
    )


    # ==========================================
    # CASE 1
    # FIRST SCAN → CHECK-IN
    # ==========================================

    if not attendance:

        if (
            current_time
            > OFFICE_START_TIME
        ):

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

            db.refresh(
                attendance
            )

        except IntegrityError:

            # Another request may have
            # inserted today's attendance
            # at exactly the same time.

            db.rollback()


            attendance = (
                db.query(Attendance)
                .filter(
                    Attendance.employee_id
                    == employee.id,

                    Attendance.attendance_date
                    == today
                )
                .first()
            )


            if not attendance:

                return {
                    "success": False,
                    "message": (
                        "Unable to create "
                        "attendance record"
                    )
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


    # ==========================================
    # CASE 2
    # CHECKED-IN → CHECK-OUT
    # ==========================================

    if (
        attendance.check_in
        and not attendance.check_out
    ):

        attendance.check_out = (
            current_time
        )


        check_in_datetime = (
            datetime.combine(
                today,
                attendance.check_in
            )
        )


        check_out_datetime = (
            datetime.combine(
                today,
                current_time
            )
        )


        working_seconds = (
            check_out_datetime
            - check_in_datetime
        ).total_seconds()


        # Prevent negative working time

        if working_seconds < 0:

            working_seconds = 0


        attendance.working_minutes = int(
            working_seconds // 60
        )


        db.commit()

        db.refresh(
            attendance
        )


        return {
            "success": True,
            "message": "Check-out successful",

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


    # ==========================================
    # CASE 3
    # ALREADY COMPLETED
    # ==========================================

    return {
        "success": True,
        "message": "Attendance already completed",

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