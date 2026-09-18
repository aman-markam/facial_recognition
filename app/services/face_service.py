import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    ROOT / "face_data" / "embeddings.json"
)


class FaceService:

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

        print("InsightFace loaded.")


    # ==========================================
    # GET FACE EMBEDDING
    # ==========================================

    def get_face_embedding(
        self,
        image_bytes: bytes
    ):

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Invalid image file"
            )


        faces = self.app.get(
            image
        )


        if len(faces) == 0:
            raise ValueError(
                "No face detected"
            )


        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. "
                "Only one person should be visible."
            )


        face = faces[0]


        embedding = face.embedding


        if embedding is None:
            raise ValueError(
                "Unable to generate face embedding"
            )


        embedding = embedding / (
            np.linalg.norm(embedding)
            + 1e-10
        )


        return embedding


    # ==========================================
    # CREATE AVERAGE EMBEDDING
    # ==========================================

    def create_average_embedding(
        self,
        images: list[bytes]
    ):

        if not images:
            raise ValueError(
                "No images provided"
            )


        if len(images) < 5:
            raise ValueError(
                "At least 5 images are required"
            )


        if len(images) > 20:
            raise ValueError(
                "Maximum 20 images are allowed"
            )


        embeddings = []


        for index, image_bytes in enumerate(
            images,
            start=1
        ):

            try:

                embedding = (
                    self.get_face_embedding(
                        image_bytes
                    )
                )

                embeddings.append(
                    embedding
                )

            except ValueError as e:

                raise ValueError(
                    f"Image {index}: {str(e)}"
                )


        if not embeddings:
            raise ValueError(
                "No valid face embeddings found"
            )


        # Convert to numpy array

        embeddings_array = np.array(
            embeddings,
            dtype=np.float32
        )


        # Average embeddings

        average_embedding = np.mean(
            embeddings_array,
            axis=0
        )


        # Normalize final embedding

        average_embedding = (
            average_embedding /
            (
                np.linalg.norm(
                    average_embedding
                )
                + 1e-10
            )
        )


        return average_embedding.tolist()


    # ==========================================
    # SAVE EMBEDDING
    # ==========================================

    def save_embedding(
        self,
        employee_code: str,
        embedding: list[float]
    ):

        EMBEDDINGS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # Load existing embeddings

        if EMBEDDINGS_PATH.exists():

            try:

                with open(
                    EMBEDDINGS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

            except (
                json.JSONDecodeError,
                OSError
            ):

                data = {}

        else:

            data = {}


        # Save / replace employee embedding

        data[
            employee_code
        ] = embedding


        # Write file

        with open(
            EMBEDDINGS_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2
            )


        return True


# ==========================================
# SINGLE SERVICE INSTANCE
# ==========================================

face_service = FaceService()