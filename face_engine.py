import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


ROOT = Path(__file__).resolve().parent

EMBEDDINGS_PATH = ROOT / "face_data" / "embeddings.json"

THRESHOLD = 0.50


class FaceEngine:

    def __init__(self):
        print("Loading InsightFace...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        self.embeddings = self._load_embeddings()

        print("InsightFace loaded.")
        print(
            f"Loaded {len(self.embeddings)} employee(s)."
        )

    def _load_embeddings(self):

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_PATH}"
            )

        with open(EMBEDDINGS_PATH, "r") as file:
            data = json.load(file)

        return {
            employee_id: np.array(
                embedding,
                dtype=np.float32
            )
            for employee_id, embedding in data.items()
        }

    @staticmethod
    def cosine_similarity(a, b):

        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        return float(np.dot(a, b))

    def find_best_match(self, embedding):

        best_employee = None
        best_score = -1

        for employee_id, stored_embedding in self.embeddings.items():

            score = self.cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score
                best_employee = employee_id

        return best_employee, best_score

    def recognize(self, image_bytes: bytes):

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return {
                "success": False,
                "message": "Invalid image"
            }

        faces = self.app.get(image)

        if not faces:
            return {
                "success": False,
                "message": "No face detected"
            }

        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected"
            }

        face = faces[0]

        embedding = face.embedding
        embedding = embedding / (
        np.linalg.norm(embedding) + 1e-10
        )
        employee_id, score = self.find_best_match(
            embedding
        )

        if score < THRESHOLD:

            return {
                "success": False,
                "message": "Unknown face",
                "confidence": round(score, 4)
            }

        return {
            "success": True,
            "employee_id": employee_id,
            "confidence": round(score, 4)
        }