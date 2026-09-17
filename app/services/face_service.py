import json
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "face_data" / "embeddings.json"


class FaceService:

    def __init__(self):
        print("Loading InsightFace...")
        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace loaded.")

    def get_face_embedding(self, image_bytes: bytes):
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        faces = self.app.get(image)
        if len(faces) == 0:
            raise ValueError("No face detected")

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. Please keep only one face in the frame."
            )

        embedding = faces[0].embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding.tolist()

    def create_average_embedding(self, images: list[bytes]):
        embeddings = []

        for index, image_bytes in enumerate(images, start=1):
            try:
                embedding = self.get_face_embedding(image_bytes)
                embeddings.append(np.array(embedding))
            except ValueError as e:
                raise ValueError(f"Image {index}: {str(e)}")

        if not embeddings:
            raise ValueError("No valid face embeddings found")

        average_embedding = np.mean(embeddings, axis=0)
        average_embedding = average_embedding / (
            np.linalg.norm(average_embedding) + 1e-10
        )

        return average_embedding.tolist()

    def save_embedding(self, employee_code: str, embedding: list[float]):
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        data[employee_code] = embedding

        with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True


face_service = FaceService()