from pathlib import Path

import cv2


def capture_faces(
	employee_id: str,
	name: str,
	cascade: cv2.CascadeClassifier,
	target_count: int = 30,
) -> int:
	output_dir = Path(__file__).resolve().parent / "data" / "faces" / employee_id
	output_dir.mkdir(parents=True, exist_ok=True)
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("Could not open the webcam.")

	captured = 0
	try:
		while captured < target_count:
			ok, frame = camera.read()
			if not ok:
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
			for x, y, width, height in faces[:1]:
				captured += 1
				face = gray[y : y + height, x : x + width]
				cv2.imwrite(str(output_dir / f"{captured:03d}.jpg"), face)
				cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
			cv2.putText(
				frame,
				f"{name}: {captured}/{target_count} - Q to cancel",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
			)
			cv2.imshow("Register employee", frame)
			if cv2.waitKey(100) & 0xFF in (ord("q"), ord("Q")):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()
	return captured
