from pathlib import Path

import cv2
import pandas as pd

import database
from trainer import MODEL_PATH


def recognize_once(action: str = "check_in") -> str | None:
	from face_engine import face_engine
	employees = database.get_employees()
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("Could not open the webcam.")

	recognized_msg = None
	act_title = "CHECK-OUT" if (action or "").lower() in ["check_out", "checkout", "out"] else "CHECK-IN"

	try:
		while recognized_msg is None:
			ok, frame = camera.read()
			if not ok:
				break

			ok_encode, buffer = cv2.imencode(".jpg", frame)
			if not ok_encode:
				continue

			result = face_engine.recognize(buffer.tobytes())

			if result["success"]:
				employee_id = result["employee_id"]
				name = employees.get(str(employee_id), f"Employee {employee_id}")
				_, msg = database.record_attendance(employee_id, name, action=action)
				recognized_msg = msg
				cv2.putText(frame, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			else:
				feedback = result.get("message", "Looking for face...")
				cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

			cv2.putText(frame, f"MODE: {act_title} (Q to cancel)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			cv2.imshow("Kiosk Attendance", frame)
			if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()
	return recognized_msg


def export_excel(path: str | Path) -> None:
	pd.DataFrame(database.attendance_rows(), columns=["employee_id", "name", "date", "time"]).to_excel(path, index=False)
