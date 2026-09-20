import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
try:
    from filelock import FileLock
except ImportError:
    import contextlib
    @contextlib.contextmanager
    def FileLock(*args, **kwargs):
        yield
from insightface.app import FaceAnalysis
from face_engine import FaceEngine


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

FACE_DATA_DIR = (
    ROOT / "face_data"
)

EMBEDDINGS_PATH = (
    FACE_DATA_DIR / "embeddings.json"
)

LOCK_PATH = (
    FACE_DATA_DIR / "embeddings.json.lock"
)

BACKUP_PATH = (
    FACE_DATA_DIR / "embeddings.json.backup"
)


logger = logging.getLogger(__name__)


class FaceService:

    def __init__(self):

        FACE_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

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
            det_size=(640, 640)
        )

        logger.info(
            "InsightFace enrollment service loaded."
        )

    # ==========================================
    # FACE EMBEDDING
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

        if getattr(face, "kps", None) is not None and len(face.kps) >= 2:
            iod = float(np.linalg.norm(face.kps[0] - face.kps[1]))
            if iod < 35:
                raise ValueError(
                    f"Insufficient resolution: Inter-ocular distance ({int(iod)}px) falls below 35px."
                )

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

        embeddings_array = np.array(
            embeddings,
            dtype=np.float32
        )

        average_embedding = np.mean(
            embeddings_array,
            axis=0
        )

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
    # LOAD EMBEDDINGS SAFELY
    # ==========================================

    def _load_embeddings(self):

        if not EMBEDDINGS_PATH.exists():

            logger.info(
                "No embeddings file exists yet."
            )

            return {}

        lock = FileLock(
            str(LOCK_PATH),
            timeout=10
        )

        with lock:

            try:

                with open(
                    EMBEDDINGS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

            except json.JSONDecodeError:

                logger.exception(
                    "CRITICAL: embeddings.json is corrupted."
                )

                raise RuntimeError(
                    "Face embedding database is corrupted. "
                    "Enrollment has been stopped to prevent "
                    "data loss."
                )

            except OSError:

                logger.exception(
                    "Unable to read embeddings file."
                )

                raise RuntimeError(
                    "Unable to read face embedding database."
                )

            if not isinstance(
                data,
                dict
            ):

                raise RuntimeError(
                    "Invalid embeddings database format."
                )

            return {
                employee_id: np.array(
                    embedding,
                    dtype=np.float32
                )
                for employee_id, embedding
                in data.items()
            }

    # ==========================================
    # SAVE EMBEDDING SAFELY
    # ==========================================

    def save_embedding(
        self,
        employee_code: str,
        embedding: list[float]
    ):

        FACE_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        lock = FileLock(
            str(LOCK_PATH),
            timeout=10
        )

        with lock:

            # ----------------------------------
            # READ EXISTING DATA
            # ----------------------------------

            if EMBEDDINGS_PATH.exists():

                try:

                    with open(
                        EMBEDDINGS_PATH,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        data = json.load(file)

                except json.JSONDecodeError:

                    logger.exception(
                        "CRITICAL: refusing to overwrite "
                        "corrupted embeddings.json"
                    )

                    raise RuntimeError(
                        "Face embedding database is corrupted. "
                        "No changes were made."
                    )

                except OSError:

                    logger.exception(
                        "Unable to read embeddings database"
                    )

                    raise RuntimeError(
                        "Unable to read face embedding database."
                    )

            else:

                data = {}

            if not isinstance(
                data,
                dict
            ):

                raise RuntimeError(
                    "Invalid embeddings database format."
                )

            # ----------------------------------
            # DUPLICATE PHYSICAL FACE ENROLLMENT CHECK
            # ----------------------------------
            cand_arr = np.array(embedding, dtype=np.float32)
            cand_norm = cand_arr / (np.linalg.norm(cand_arr) + 1e-10)

            for existing_code, existing_emb in data.items():
                if existing_code != employee_code:
                    ex_arr = np.array(existing_emb, dtype=np.float32)
                    ex_norm = ex_arr / (np.linalg.norm(ex_arr) + 1e-10)
                    similarity = float(np.dot(cand_norm, ex_norm))

                    if similarity >= 0.55:
                        logger.warning(
                            "Duplicate face enrollment blocked: employee=%s matches existing employee=%s (similarity=%.4f)",
                            employee_code,
                            existing_code,
                            similarity
                        )
                        raise ValueError(
                            f"This face is already enrolled in the system for employee '{existing_code}' (similarity: {similarity:.2f}). "
                            f"Re-enrolling an existing face for a new employee is not allowed."
                        )

            # ----------------------------------
            # BACKUP CURRENT DATA
            # ----------------------------------

            if EMBEDDINGS_PATH.exists():

                try:

                    with open(
                        EMBEDDINGS_PATH,
                        "r",
                        encoding="utf-8"
                    ) as source:

                        existing_content = (
                            source.read()
                        )

                    with open(
                        BACKUP_PATH,
                        "w",
                        encoding="utf-8"
                    ) as backup:

                        backup.write(
                            existing_content
                        )

                except OSError:

                    logger.exception(
                        "Unable to create embeddings backup"
                    )

                    raise RuntimeError(
                        "Unable to create face database backup."
                    )

            # ----------------------------------
            # UPDATE DATA
            # ----------------------------------

            data[employee_code] = embedding

            # ----------------------------------
            # ATOMIC WRITE
            # ----------------------------------

            temp_path = (
                EMBEDDINGS_PATH.with_suffix(
                    ".json.tmp"
                )
            )

            try:

                with open(
                    temp_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        data,
                        file,
                        indent=2
                    )

                    file.flush()

                    os.fsync(
                        file.fileno()
                    )

                os.replace(
                    temp_path,
                    EMBEDDINGS_PATH
                )

            except OSError:

                logger.exception(
                    "Failed to atomically save embeddings"
                )

                if temp_path.exists():

                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                raise RuntimeError(
                    "Failed to save face embedding."
                )

            logger.info(
                "Face embedding saved: employee=%s",
                employee_code
            )

        return True

    # ==========================================
    # DELETE EMBEDDING SAFELY
    # ==========================================

    def delete_embedding(
        self,
        employee_code: str
    ):

        if not EMBEDDINGS_PATH.exists():

            return False

        lock = FileLock(
            str(LOCK_PATH),
            timeout=10
        )

        with lock:

            try:

                with open(
                    EMBEDDINGS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

            except json.JSONDecodeError:

                logger.exception(
                    "CRITICAL: cannot modify corrupted embeddings.json"
                )

                raise RuntimeError(
                    "Face embedding database is corrupted. "
                    "No changes were made."
                )

            if employee_code not in data:

                return False

            # Backup before modification
            with open(
                BACKUP_PATH,
                "w",
                encoding="utf-8"
            ) as backup:

                json.dump(
                    data,
                    backup,
                    indent=2
                )

            del data[employee_code]

            temp_path = (
                EMBEDDINGS_PATH.with_suffix(
                    ".json.tmp"
                )
            )

            with open(
                temp_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    data,
                    file,
                    indent=2
                )

                file.flush()

                os.fsync(
                    file.fileno()
                )

            os.replace(
                temp_path,
                EMBEDDINGS_PATH
            )

            logger.info(
                "Face embedding deleted: employee=%s",
                employee_code
            )

            return True


face_service = FaceService()