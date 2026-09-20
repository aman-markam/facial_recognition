from pathlib import Path

import cv2
import pandas as pd

import database
from trainer import MODEL_PATH


def recognize_once(cascade: cv2.CascadeClassifier) -> str | None:
	if not MODEL_PATH.exists():
		raise RuntimeError("Train the recognizer before taking attendance.")
	employees = database.get_employees()
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read(str(MODEL_PATH))
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("Could not open the webcam.")
	recognized_name = None
	try:
		while recognized_name is None:
			ok, frame = camera.read()
			if not ok:
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			for x, y, width, height in cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100)):
				label, confidence = recognizer.predict(gray[y : y + height, x : x + width])
				employee_id = str(label)
				if confidence < 75 and employee_id in employees:
					if database.record_attendance(employee_id, employees[employee_id]):
						recognized_name = f"{employee_id} - {employees[employee_id]}"
					else:
						recognized_name = f"Already recorded: {employee_id} - {employees[employee_id]}"
				cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
			cv2.putText(frame, "Q to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			cv2.imshow("Take attendance", frame)
			if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()
	return recognized_name


def export_excel(path: str | Path) -> None:
	pd.DataFrame(database.attendance_rows(), columns=["employee_id", "name", "date", "time"]).to_excel(path, index=False)
