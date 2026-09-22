"""
Live Multi-Person Facial Attendance Recognition Kiosk (recognize.py)

Captures high-resolution video stream from webcam, tracks multiple faces simultaneously,
performs 3D liveness, anti-spoofing, multi-angle pose estimation, distance upscaling,
occlusion checking (cloth/scarf/hand), and records attendance for all recognized employees.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import database
from app.services.multiface_tracker import MultiFaceTracker
from app.core.liveness_config import liveness_config


def recognize_multiple_webcam(action: str = "check_in", camera_index: int = 0) -> list[dict]:
    """
    Runs live multi-person facial recognition and attendance via webcam.
    """
    print(f"\n==================================================")
    print(f" MULTI-PERSON FACIAL ATTENDANCE KIOSK ({action.upper()})")
    print(f"==================================================")

    # Initialize PostgreSQL database table if not present
    try:
        database.initialize()
    except Exception as e:
        print(f"Note: Local database initialize skipped or failed: {e}")

    tracker = MultiFaceTracker()
    employees_map = database.get_employees()

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

                        succ, msg = database.record_attendance(emp_id, emp_name, action=action)
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
                is_occluded = latest_obs.get("is_occluded", False)
                occlusion_reason = latest_obs.get("occlusion_reason")

                # Color coding: Green = Recognized Live, Yellow = Occluded, Red = Rejected
                if is_live and recognized and emp_id:
                    emp_name = employees_map.get(str(emp_id), f"Employee {emp_id}")
                    box_color = (0, 220, 0)
                    label = f"✓ {emp_name} ({emp_id})"
                elif is_occluded:
                    box_color = (0, 165, 255)
                    label = f"⚠ Covered: {occlusion_reason or 'Cloth/Hand'}"
                else:
                    box_color = (0, 0, 230)
                    label = "✕ Unrecognized / Spoof"

                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(display_frame, (x1, max(0, y1 - 25)), (x2, y1), box_color, -1)
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 5, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
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
