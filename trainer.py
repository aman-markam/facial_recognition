from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
FACES_DIR = ROOT / "data" / "faces"
MODEL_PATH = ROOT / "data" / "models" / "trainer.yml"


def train_model() -> int:
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	images: list = []
	labels: list[int] = []
	for employee_dir in FACES_DIR.iterdir():
		if not employee_dir.is_dir() or not employee_dir.name.isdigit():
			continue
		for image_path in employee_dir.glob("*.jpg"):
			image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
			if image is not None:
				images.append(image)
				labels.append(int(employee_dir.name))
	if not images:
		raise RuntimeError("No captured face images were found.")
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	recognizer.train(images, __import__("numpy").array(labels))
	recognizer.write(str(MODEL_PATH))
	return len(images)
