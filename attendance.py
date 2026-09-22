from pathlib import Path

import cv2
import pandas as pd

import database
from trainer import MODEL_PATH


def recognize_once(action: str = "check_in") -> str | None:
    from recognize import recognize_multiple_webcam
    records = recognize_multiple_webcam(action=action)
    if records:
        return f"Processed attendance for {len(records)} employee(s)."
    return "No attendance recorded."


def export_excel(path: str | Path) -> None:
	pd.DataFrame(database.attendance_rows(), columns=["employee_id", "name", "date", "time"]).to_excel(path, index=False)
