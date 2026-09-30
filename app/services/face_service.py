import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from app.services import embedding_store


logger = logging.getLogger(__name__)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


class FaceService:

    def __init__(self):

        logger.info(
            "Loading InsightFace for enrollment..."
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(960, 960)
        )

        logger.info(
            "InsightFace enrollment service loaded."
        )

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

        if getattr(face, "kps", None) is not None and len(face.kps) >= 2:
            iod = float(np.linalg.norm(face.kps[0] - face.kps[1]))
            if iod < 20:
                raise ValueError(
                    f"Insufficient resolution: Inter-ocular distance ({int(iod)}px) falls below 20px."
                )

        embedding = face.embedding

        if embedding is None:
            raise ValueError(
                "Unable to generate face embedding"
            )

        embedding = normalize_embedding(np.array(embedding, dtype=np.float32))

        return embedding

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

        embeddings_array = np.array(
            embeddings,
            dtype=np.float32
        )

        average_embedding = np.mean(
            embeddings_array,
            axis=0
        )

        average_embedding = normalize_embedding(average_embedding)

        return average_embedding.tolist()

    def _load_embeddings(self, db: Session | None = None):
        return embedding_store.load_all_embeddings(db)

    def save_embedding(
        self,
        employee_code: str,
        embedding: list[float],
        db: Session | None = None,
    ):
        return embedding_store.save_embedding(
            employee_code,
            embedding,
            db,
        )

    def delete_embedding(
        self,
        employee_code: str,
        db: Session | None = None,
    ):
        return embedding_store.delete_embedding(
            employee_code,
            db,
        )


face_service = FaceService()
