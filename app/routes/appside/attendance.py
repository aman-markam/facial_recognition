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
from app.services.multiface_tracker import MultiFaceTracker
from face_engine import FaceEngine


OFFICE_START_TIME = time(9, 30)


router = APIRouter(
    prefix="/api/v1/appside/attendance",
    tags=["Appside Attendance"]
)


face_engine = FaceEngine()
multi_face_tracker = MultiFaceTracker()


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
# ============================================================
# MULTI-FACE ATTENDANCE
# ============================================================

@router.post("/mark-multiple")
async def mark_multiple_attendance(
    images: list[UploadFile] = File(...),
    mode: str | None = Form(default=None),
    action: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    """
    Multi-face attendance endpoint.

    Flow:
        1. Receive exactly 5 camera frames
        2. Detect multiple faces
        3. Track faces across all 5 frames
        4. Verify liveness
        5. Recognize enrolled employees
        6. Check-in / check-out each recognized employee

    Existing /mark endpoint is not affected.
    """

    # ========================================================
    # 1. VALIDATE FRAMES
    # ========================================================

    if len(images) != 5:
        return {
            "success": False,
            "message": "Exactly 5 images are required.",
            "received_frames": len(images),
            "required_frames": 5,
        }

    # ========================================================
    # 2. READ ALL FRAMES
    # ========================================================

    image_frames: list[bytes] = []

    for index, image in enumerate(images, start=1):

        image_bytes = await image.read()

        if not image_bytes:
            return {
                "success": False,
                "message": f"Frame {index} is empty.",
            }

        image_frames.append(image_bytes)

    # ========================================================
    # 3. MULTI-FACE TRACKING + LIVENESS + RECOGNITION
    # ========================================================

    try:

        tracking_result = multi_face_tracker.process_frames(
            image_frames
        )

    except Exception as exc:

        return {
            "success": False,
            "message": "Multi-face processing failed.",
            "error": str(exc),
        }

    # ========================================================
    # 4. GET ELIGIBLE EMPLOYEES
    # ========================================================

    employees_to_attendance = tracking_result.get(
        "employees_to_attendance",
        []
    )

    if not employees_to_attendance:

        return {
            "success": False,
            "message": "No eligible recognized employees found.",
            "total_faces": tracking_result.get(
                "total_faces",
                0
            ),
            "total_tracks": tracking_result.get(
                "total_tracks",
                0
            ),
            "eligible_employees": 0,
            "processed_employees": [],
        }

    # ========================================================
    # 5. NORMALIZE ACTION
    # ========================================================

    requested_action = (
        mode or action or ""
    ).strip().lower()

    if requested_action in ["checkin", "in"]:
        requested_action = "check_in"

    elif requested_action in ["checkout", "out"]:
        requested_action = "check_out"

    elif requested_action not in [
        "",
        "check_in",
        "check_out",
    ]:

        return {
            "success": False,
            "message": (
                "Invalid action. "
                "Use check_in or check_out."
            ),
        }

    # ========================================================
    # 6. CURRENT DATE / TIME
    # ========================================================

    now = datetime.now()

    today = now.date()

    yesterday = today - timedelta(days=1)

    current_time = now.time()

    processed_employees = []

    # ========================================================
    # 7. PROCESS EACH RECOGNIZED EMPLOYEE
    # ========================================================

    for tracked_employee in employees_to_attendance:

        # ----------------------------------------------------
        # Employee code returned by tracker
        # ----------------------------------------------------

        employee_code = tracked_employee.get(
            "employee_id"
        )

        # ----------------------------------------------------
        # Recognition confidence
        # ----------------------------------------------------

        confidence = float(
            tracked_employee.get(
                "confidence",
                tracked_employee.get(
                    "recognition_confidence",
                    0.0
                )
            ) or 0.0
        )

        # ----------------------------------------------------
        # Liveness information
        # ----------------------------------------------------

        liveness_average = float(
            tracked_employee.get(
                "liveness_average",
                0.0
            ) or 0.0
        )

        live_frames = int(
            tracked_employee.get(
                "live_frames",
                0
            ) or 0
        )

        total_frames = int(
            tracked_employee.get(
                "total_frames",
                0
            ) or 0
        )

        # ====================================================
        # FIND EMPLOYEE
        # ====================================================

        employee = None

        if employee_code is not None:

            employee = (
                db.query(Employee)
                .filter(
                    Employee.employee_code
                    == str(employee_code)
                )
                .first()
            )

            # ------------------------------------------------
            # Fallback: employee_code may actually be DB ID
            # ------------------------------------------------

            if employee is None:

                try:

                    employee = (
                        db.query(Employee)
                        .filter(
                            Employee.id
                            == int(employee_code)
                        )
                        .first()
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    employee = None

        # ====================================================
        # EMPLOYEE NOT FOUND
        # ====================================================

        if employee is None:

            processed_employees.append(
                {
                    "employee_id": employee_code,
                    "success": False,
                    "action": None,
                    "message": "Employee not found.",
                    "attendance_id": None,
                    "check_in": None,
                    "check_out": None,
                    "working_minutes": None,
                    "status": None,
                    "confidence": confidence,
                    "recognition_confidence": confidence,
                    "liveness_average": liveness_average,
                    "live_frames": live_frames,
                    "total_frames": total_frames,
                }
            )

            continue

        # ====================================================
        # ACTIVE EMPLOYEE CHECK
        # ====================================================

        if not employee.is_active:

            processed_employees.append(
                {
                    "employee_id":
                        employee.employee_code,
                    "employee_name":
                        employee.name,
                    "success": False,
                    "action": None,
                    "message":
                        "Employee is inactive.",
                    "attendance_id": None,
                    "check_in": None,
                    "check_out": None,
                    "working_minutes": None,
                    "status": None,
                    "confidence": confidence,
                    "recognition_confidence": confidence,
                    "liveness_average": liveness_average,
                    "live_frames": live_frames,
                    "total_frames": total_frames,
                }
            )

            continue

        # ====================================================
        # FIND OPEN ATTENDANCE
        # ====================================================

        open_attendance = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id
                == employee.id,

                Attendance.check_out.is_(None),

                Attendance.attendance_date.in_(
                    [today, yesterday]
                ),
            )
            .order_by(
                Attendance.attendance_date.desc()
            )
            .first()
        )

        # ====================================================
        # EXPLICIT CHECK-IN
        # ====================================================

        if requested_action == "check_in":

            if open_attendance is not None:

                processed_employees.append(
                    {
                        "employee_id":
                            employee.employee_code,
                        "employee_name":
                            employee.name,
                        "success": True,
                        "action": "check_in",
                        "message":
                            "Employee is already checked in.",
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
                            confidence,
                        "recognition_confidence":
                            confidence,
                        "liveness_average":
                            liveness_average,
                        "live_frames":
                            live_frames,
                        "total_frames":
                            total_frames,
                    }
                )

                continue

        # ====================================================
        # EXPLICIT CHECK-OUT
        # ====================================================

        if requested_action == "check_out":

            if open_attendance is None:

                processed_employees.append(
                    {
                        "employee_id":
                            employee.employee_code,
                        "employee_name":
                            employee.name,
                        "success": False,
                        "action": "check_out",
                        "message":
                            "No active check-in record found.",
                        "attendance_id": None,
                        "check_in": None,
                        "check_out": None,
                        "working_minutes": None,
                        "status": None,
                        "confidence": confidence,
                        "recognition_confidence":
                            confidence,
                        "liveness_average":
                            liveness_average,
                        "live_frames":
                            live_frames,
                        "total_frames":
                            total_frames,
                    }
                )

                continue

            # ------------------------------------------------
            # CHECK OUT
            # ------------------------------------------------

            check_in_datetime = datetime.combine(
                open_attendance.attendance_date,
                open_attendance.check_in,
            )

            check_out_datetime = datetime.combine(
                today,
                current_time,
            )

            working_seconds = (
                check_out_datetime
                - check_in_datetime
            ).total_seconds()

            if working_seconds < 0:
                working_seconds = 0

            working_minutes = int(
                working_seconds // 60
            )

            open_attendance.check_out = current_time

            open_attendance.working_minutes = (
                working_minutes
            )

            db.commit()

            db.refresh(open_attendance)

            processed_employees.append(
                {
                    "employee_id":
                        employee.employee_code,
                    "employee_name":
                        employee.name,
                    "success": True,
                    "action": "check_out",
                    "message":
                        "Check-out successful.",
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
                        confidence,
                    "recognition_confidence":
                        confidence,
                    "liveness_average":
                        liveness_average,
                    "live_frames":
                        live_frames,
                    "total_frames":
                        total_frames,
                }
            )

            continue

        # ====================================================
        # AUTOMATIC MODE
        #
        # No action supplied:
        #
        # open attendance -> CHECK OUT
        # no attendance   -> CHECK IN
        # ====================================================

        if not requested_action:

            if open_attendance is not None:

                check_in_datetime = datetime.combine(
                    open_attendance.attendance_date,
                    open_attendance.check_in,
                )

                check_out_datetime = datetime.combine(
                    today,
                    current_time,
                )

                working_seconds = (
                    check_out_datetime
                    - check_in_datetime
                ).total_seconds()

                if working_seconds < 0:
                    working_seconds = 0

                working_minutes = int(
                    working_seconds // 60
                )

                open_attendance.check_out = current_time

                open_attendance.working_minutes = (
                    working_minutes
                )

                db.commit()

                db.refresh(open_attendance)

                processed_employees.append(
                    {
                        "employee_id":
                            employee.employee_code,
                        "employee_name":
                            employee.name,
                        "success": True,
                        "action": "check_out",
                        "message":
                            "Check-out successful.",
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
                            confidence,
                        "recognition_confidence":
                            confidence,
                        "liveness_average":
                            liveness_average,
                        "live_frames":
                            live_frames,
                        "total_frames":
                            total_frames,
                    }
                )

                continue

        # ====================================================
        # CHECK IF TODAY ALREADY COMPLETED
        # ====================================================

        today_completed = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id
                == employee.id,

                Attendance.attendance_date
                == today,

                Attendance.check_out.is_not(None),
            )
            .first()
        )

        if today_completed:

            processed_employees.append(
                {
                    "employee_id":
                        employee.employee_code,
                    "employee_name":
                        employee.name,
                    "success": True,
                    "action": None,
                    "message":
                        "Attendance already completed.",
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
                        confidence,
                    "recognition_confidence":
                        confidence,
                    "liveness_average":
                        liveness_average,
                    "live_frames":
                        live_frames,
                    "total_frames":
                        total_frames,
                }
            )

            continue

        # ====================================================
        # CREATE NEW CHECK-IN
        # ====================================================

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
            confidence=confidence,
        )

        db.add(attendance)

        try:

            db.commit()

            db.refresh(attendance)

        except IntegrityError:

            db.rollback()

            existing_attendance = (
                db.query(Attendance)
                .filter(
                    Attendance.employee_id
                    == employee.id,

                    Attendance.attendance_date
                    == today,
                )
                .first()
            )

            if existing_attendance:

                processed_employees.append(
                    {
                        "employee_id":
                            employee.employee_code,
                        "employee_name":
                            employee.name,
                        "success": True,
                        "action": None,
                        "message":
                            "Attendance already exists.",
                        "attendance_id":
                            existing_attendance.id,
                        "check_in":
                            existing_attendance.check_in,
                        "check_out":
                            existing_attendance.check_out,
                        "working_minutes":
                            existing_attendance.working_minutes,
                        "status":
                            existing_attendance.status,
                        "confidence":
                            confidence,
                        "recognition_confidence":
                            confidence,
                        "liveness_average":
                            liveness_average,
                        "live_frames":
                            live_frames,
                        "total_frames":
                            total_frames,
                    }
                )

                continue

            processed_employees.append(
                {
                    "employee_id":
                        employee.employee_code,
                    "employee_name":
                        employee.name,
                    "success": False,
                    "action": "check_in",
                    "message":
                        "Unable to create attendance record.",
                    "attendance_id": None,
                    "check_in": None,
                    "check_out": None,
                    "working_minutes": None,
                    "status": None,
                    "confidence": confidence,
                    "recognition_confidence":
                        confidence,
                    "liveness_average":
                        liveness_average,
                    "live_frames":
                        live_frames,
                    "total_frames":
                        total_frames,
                }
            )

            continue

        # ====================================================
        # SUCCESSFUL CHECK-IN
        # ====================================================

        processed_employees.append(
            {
                "employee_id":
                    employee.employee_code,
                "employee_name":
                    employee.name,
                "success": True,
                "action": "check_in",
                "message":
                    "Check-in successful.",
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
                    confidence,
                "recognition_confidence":
                    confidence,
                "liveness_average":
                    liveness_average,
                "live_frames":
                    live_frames,
                "total_frames":
                    total_frames,
            }
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    successful_count = sum(
        1
        for item in processed_employees
        if item.get("success") is True
    )

    return {
        "success": successful_count > 0,

        "message": (
            f"Attendance processed for "
            f"{len(processed_employees)} employee(s)."
        ),

        "successful_count":
            successful_count,

        "total_faces":
            tracking_result.get(
                "total_faces",
                0
            ),

        "total_tracks":
            tracking_result.get(
                "total_tracks",
                0
            ),

        "eligible_employees":
            tracking_result.get(
                "eligible_employees",
                len(employees_to_attendance)
            ),

        "processed_employees":
            processed_employees,
    }