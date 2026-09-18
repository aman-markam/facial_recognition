import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


ROOT = Path(__file__).resolve().parent

EMBEDDINGS_PATH = (
    ROOT / "face_data" / "embeddings.json"
)

# Recognition threshold
THRESHOLD = 0.55


class FaceEngine:

    def __init__(self):

        print("Loading InsightFace...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        self.embeddings = (
            self._load_embeddings()
        )

        print("InsightFace loaded.")

        print(
            f"Loaded "
            f"{len(self.embeddings)} "
            f"employee(s)."
        )


    # ==========================================
    # LOAD EMBEDDINGS
    # ==========================================

    def _load_embeddings(self):

        if not EMBEDDINGS_PATH.exists():

            print(
                "Embeddings file not found:"
            )

            print(
                EMBEDDINGS_PATH
            )

            return {}


        try:

            with open(
                EMBEDDINGS_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)


            return {
                employee_id: np.array(
                    embedding,
                    dtype=np.float32
                )
                for employee_id, embedding
                in data.items()
            }


        except Exception as e:

            print(
                "Error loading embeddings:",
                e
            )

            return {}


    # ==========================================
    # RELOAD EMBEDDINGS
    # ==========================================

    def reload_embeddings(self):

        self.embeddings = (
            self._load_embeddings()
        )

        print(
            f"Embeddings reloaded. "
            f"Loaded "
            f"{len(self.embeddings)} "
            f"employee(s)."
        )


    # ==========================================
    # COSINE SIMILARITY
    # ==========================================

    @staticmethod
    def cosine_similarity(
        a,
        b
    ):

        a = a / (
            np.linalg.norm(a)
            + 1e-10
        )

        b = b / (
            np.linalg.norm(b)
            + 1e-10
        )

        return float(
            np.dot(a, b)
        )


    # ==========================================
    # FIND BEST MATCH
    # ==========================================

    def find_best_match(
        self,
        embedding
    ):

        best_employee = None
        best_score = -1.0


        for (
            employee_id,
            stored_embedding
        ) in self.embeddings.items():

            score = (
                self.cosine_similarity(
                    embedding,
                    stored_embedding
                )
            )


            print(
                f"Match "
                f"{employee_id}: "
                f"{score:.4f}"
            )


            if score > best_score:

                best_score = score

                best_employee = (
                    employee_id
                )


        return (
            best_employee,
            best_score
        )


    # ==========================================
    # RECOGNIZE FACE
    # ==========================================

    def recognize(
        self,
        image_bytes: bytes
    ):

        # Always get latest employees

        self.reload_embeddings()


        if not self.embeddings:

            return {
                "success": False,
                "message": (
                    "No enrolled faces found"
                )
            }


        # --------------------------------------
        # IMAGE DECODE
        # --------------------------------------

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


        # --------------------------------------
        # FACE DETECTION
        # --------------------------------------

        faces = self.app.get(
            image
        )


        if not faces:

            return {
                "success": False,
                "message": "No face detected"
            }


        if len(faces) > 1:

            return {
                "success": False,
                "message": (
                    "Multiple faces detected. "
                    "Please keep only one person "
                    "in front of the camera."
                )
            }


        # --------------------------------------
        # GET EMBEDDING
        # --------------------------------------

        face = faces[0]

        embedding = face.embedding


        if embedding is None:

            return {
                "success": False,
                "message": (
                    "Unable to generate "
                    "face embedding"
                )
            }


        embedding = (
            embedding /
            (
                np.linalg.norm(
                    embedding
                )
                + 1e-10
            )
        )


        # --------------------------------------
        # FIND BEST MATCH
        # --------------------------------------

        (
            employee_id,
            score
        ) = self.find_best_match(
            embedding
        )


        # --------------------------------------
        # UNKNOWN FACE
        # --------------------------------------

        if (
            employee_id is None
            or score < THRESHOLD
        ):

            return {
                "success": False,
                "message": (
                    "Face not recognized"
                ),
                "confidence": round(
                    score,
                    4
                )
            }


        # --------------------------------------
        # SUCCESS
        # --------------------------------------

        return {
            "success": True,
            "employee_id": employee_id,
            "confidence": round(
                score,
                4
            )
        }


# ==========================================
# SINGLE INSTANCE
# ==========================================

face_engine = FaceEngine()