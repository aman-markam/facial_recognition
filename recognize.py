"""
Live Multi-Person Facial Attendance Recognition Kiosk (recognize.py)

Captures high-resolution video stream from webcam, tracks multiple faces simultaneously,
performs 3D liveness, anti-spoofing, multi-angle pose estimation, distance upscaling,
occlusion checking (cloth/scarf/hand), and records attendance for all recognized employees.
"""

import argparse
from datetime import datetime, timedelta
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.engine import make_url

from app.core.config import settings

# The SQLAlchemy engine is created on import, so select the kiosk database first.
settings.DATABASE_URL = make_url(settings.DATABASE_URL).set(
    database="face_attendance"
).render_as_string(hide_password=False)

from app.database.database import SessionLocal
from app.models.attendance import Attendance
from app.models.employee import Employee
from app.services.multiface_tracker import MultiFaceTracker
from app.core.liveness_config import liveness_config


def _record_attendance(employee_code: str, action: str, confidence: float) -> tuple[bool, str]:
    now = datetime.now()
    today = now.date()
    is_checkout = (action or "").lower().strip() in ["check_out", "checkout", "out"]

    with SessionLocal() as db:
        employee = (
            db.query(Employee)
            .filter(Employee.employee_code == employee_code)
            .first()
        )
        if employee is None:
            return False, f"Employee not found: {employee_code}"
        if not employee.is_active:
            return False, f"Employee is inactive: {employee_code} - {employee.name}"

        open_attendance = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id == employee.id,
                Attendance.check_out.is_(None),
                Attendance.attendance_date.in_([today, today - timedelta(days=1)]),
            )
            .order_by(Attendance.attendance_date.desc())
            .first()
        )

        if is_checkout:
            if open_attendance is None:
                return False, f"No active check-in record found: {employee_code} - {employee.name}"
            if open_attendance.check_in is None:
                return False, f"Check-in time is missing: {employee_code} - {employee.name}"

            open_attendance.check_out = now.time()
            check_in_datetime = datetime.combine(
                open_attendance.attendance_date,
                open_attendance.check_in,
            )
            check_out_datetime = datetime.combine(today, now.time())
            working_seconds = max(
                0,
                (check_out_datetime - check_in_datetime).total_seconds(),
            )
            open_attendance.working_minutes = int(working_seconds // 60)
            db.commit()
            return True, f"Check-out successful: {employee_code} - {employee.name}"

        if open_attendance is not None:
            return False, f"Already checked in: {employee_code} - {employee.name}"

        completed_attendance = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id == employee.id,
                Attendance.attendance_date == today,
                Attendance.check_out.is_not(None),
            )
            .first()
        )
        if completed_attendance is not None:
            return False, f"Attendance already completed: {employee_code} - {employee.name}"

        db.add(
            Attendance(
                employee_id=employee.id,
                attendance_date=today,
                check_in=now.time(),
                status="PRESENT",
                confidence=confidence,
            )
        )
        db.commit()
        return True, f"Check-in successful: {employee_code} - {employee.name}"


def recognize_multiple_webcam(action: str = "check_in", camera_index: int = 0) -> list[dict]:
    """
    Runs live multi-person facial recognition and attendance via webcam.
    """
    print(f"\n==================================================")
    print(f" MULTI-PERSON FACIAL ATTENDANCE KIOSK ({action.upper()})")
    print(f"==================================================")

    tracker = MultiFaceTracker()
    try:
        with SessionLocal() as db:
            employees_map = {
                str(employee.employee_code): employee.name
                for employee in db.query(Employee).all()
            }
    except Exception as e:
        print(f"❌ Database unavailable: {e}")
        print("Verify DATABASE_URL points to the face_attendance PostgreSQL database.")
        return []

    # Open webcam with native high resolution
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open webcam at index {camera_index}")
        return []

    # Set camera resolution to 1280x720 ideal
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    act_title = "CHECK-OUT" if (action or "").lower() in ["check_out", "checkout", "out"] else "CHECK-IN"

    frame_buffer: list[bytes] = []
    raw_frames: list[np.ndarray] = []
    last_results: list[dict] = []
    processed_records: list[dict] = []

    REQUIRED_FRAMES = 5
    status_text = "Position faces in front of camera..."

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            # Encode frame to JPEG
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frame_buffer.append(buf.tobytes())
                raw_frames.append(frame.copy())

            # Every 5 frames, run multi-person tracking pipeline
            if len(frame_buffer) == REQUIRED_FRAMES:
                try:
                    res = tracker.process_frames(frame_buffer)
                    last_results = res.get("tracks", [])
                    employees_eligible = res.get("employees", [])

                    # Record attendance for eligible recognized employees
                    for emp_track in employees_eligible:
                        emp_id = str(emp_track.get("employee_id"))
                        emp_name = employees_map.get(emp_id, f"Employee {emp_id}")

                        confidence = float(
                            emp_track.get("recognition_confidence", 0.0) or 0.0
                        )
                        succ, msg = _record_attendance(
                            emp_id,
                            action,
                            confidence,
                        )
                        if succ:
                            print(f"✅ {msg}")
                            status_text = f"SUCCESS: {emp_name} ({act_title})"
                            processed_records.append({
                                "employee_id": emp_id,
                                "name": emp_name,
                                "status": "SUCCESS",
                                "message": msg,
                            })
                        else:
                            print(f"ℹ️ {msg}")
                            status_text = msg

                except Exception as err:
                    print(f"Processing error: {err}")

                # Reset frame buffer for next burst
                frame_buffer.clear()
                raw_frames.clear()

            # Annotate current frame with bounding boxes and tracking details
            display_frame = frame.copy()
            fh, fw, _ = display_frame.shape

            # Header overlay banner
            cv2.rectangle(display_frame, (0, 0), (fw, 50), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                f"MODE: {act_title} | Multi-Person Kiosk (Press 'Q' to Exit)",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Footer status banner
            cv2.rectangle(display_frame, (0, fh - 40), (fw, fh), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                status_text,
                (15, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Draw tracked face boxes
            for track in last_results:
                obs_list = track.get("observations", [])
                if not obs_list:
                    continue
                latest_obs = obs_list[-1]
                bbox = latest_obs.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]
                is_live = track.get("passed_liveness", False)
                recognized = track.get("recognized", False)
                emp_id = track.get("employee_id")
                confidence = float(track.get("recognition_confidence", 0.0) or 0.0)
                is_occluded = latest_obs.get("is_occluded", False)
                occlusion_reason = latest_obs.get("occlusion_reason")

                # Color coding: Green = Live, Yellow = Occluded, Red = Rejected
                if recognized and emp_id and confidence >= 0.60:
                    box_color = (0, 220, 0)
                    label = f"Employee: {emp_id}"
                elif confidence >= 0.60 and emp_id:
                    box_color = (0, 220, 0)
                    label = f"Employee: {emp_id}"
                elif is_occluded:
                    box_color = (0, 165, 255)
                    label = f"⚠ Check: {occlusion_reason or 'Face quality'}"
                else:
                    box_color = (0, 0, 230)
                    label = "Unknown"

                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                label_top = max(0, y1 - 30)
                cv2.rectangle(display_frame, (x1, label_top), (x2, y1), box_color, -1)
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 5, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                if confidence > 0:
                    score_label = f"Score: {confidence:.3f}"
                    score_top = min(fh - 25, y2)
                    score_bottom = min(fh, score_top + 28)
                    cv2.rectangle(
                        display_frame,
                        (x1, score_top),
                        (x2, score_bottom),
                        box_color,
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        score_label,
                        (x1 + 5, min(fh - 7, score_top + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

            cv2.imshow("Multi-Person Facial Attendance System", display_frame)

            # Press 'q' or 'Q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\nKiosk stopped by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return processed_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Person Facial Attendance Recognition Kiosk")
    parser.add_argument("--action", type=str, default="check_in", choices=["check_in", "check_out"], help="Attendance action")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    recognize_multiple_webcam(action=args.action, camera_index=args.camera)
