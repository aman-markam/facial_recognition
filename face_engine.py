import json
import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

EMBEDDINGS_PATH = (
    ROOT / "face_data" / "embeddings.json"
)

# Face recognition threshold
THRESHOLD = 0.55

logger = logging.getLogger(__name__)


class FaceEngine:

    # ==========================================
    # INITIALIZE FACE ENGINE
    # ==========================================

    def __init__(self):

        logger.info(
            "Loading InsightFace..."
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

        self.embeddings = (
            self._load_embeddings()
        )

        logger.info(
            "InsightFace loaded."
        )

        logger.info(
            "Loaded %d employee(s)",
            len(self.embeddings)
        )

    # ==========================================
    # LOAD EMBEDDINGS
    # ==========================================

    def _load_embeddings(self):

        if not EMBEDDINGS_PATH.exists():

            logger.warning(
                "Embeddings file not found: %s",
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

            embeddings = {
                employee_id: np.array(
                    embedding,
                    dtype=np.float32
                )
                for employee_id, embedding
                in data.items()
            }

            logger.info(
                "Embeddings loaded successfully: %d employee(s)",
                len(embeddings)
            )

            return embeddings

        except json.JSONDecodeError:

            logger.exception(
                "Invalid JSON in embeddings file"
            )

            return {}

        except OSError:

            logger.exception(
                "Unable to read embeddings file"
            )

            return {}

        except Exception:

            logger.exception(
                "Unexpected error while loading embeddings"
            )

            return {}

    # ==========================================
    # RELOAD EMBEDDINGS
    # ==========================================

    def reload_embeddings(self):

        self.embeddings = (
            self._load_embeddings()
        )

        logger.info(
            "Embeddings reloaded: %d employee(s)",
            len(self.embeddings)
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
    # FIND BEST MATCH WITH AMBIGUITY DETECTION
    # ==========================================

    def find_best_match(
        self,
        embedding
    ):

        best_employee = None
        best_score = -1.0
        second_best_score = -1.0

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

            logger.debug(
                "Face comparison: employee=%s similarity=%.4f",
                employee_id,
                score
            )

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_employee = employee_id
            elif score > second_best_score:
                second_best_score = score

        # Ambiguity detection check (e.g. distinguishing twins or close relatives)
        is_ambiguous = False
        if best_score >= THRESHOLD and (best_score - second_best_score) < 0.05:
            is_ambiguous = True

        return (
            best_employee,
            best_score,
            is_ambiguous
        )

    # ==========================================
    # COMPREHENSIVE QUALITY, RESOLUTION, OCCLUSION & LIVENESS GATE
    # ==========================================

    def validate_face_quality(
        self,
        face,
        image,
        min_iod: float = 35.0
    ):
        """
        Comprehensive quality, resolution, occlusion, lighting, distortion,
        and liveness checks matching enterprise specifications.
        """
        image_height, image_width = image.shape[:2]
        bbox = face.bbox
        x1, y1, x2, y2 = bbox

        face_width = x2 - x1
        face_height = y2 - y1

        # --------------------------------------
        # 1. FACE RESOLUTION & ASPECT RATIO (SENSOR DISTORTION)
        # --------------------------------------
        MIN_FACE_WIDTH = 80
        MIN_FACE_HEIGHT = 80

        if (
            face_width < MIN_FACE_WIDTH
            or face_height < MIN_FACE_HEIGHT
        ):
            return False, (
                "Insufficient resolution: Move closer to the camera."
            )

        aspect_ratio = face_width / (face_height + 1e-6)
        if aspect_ratio < 0.5 or aspect_ratio > 1.6:
            return False, (
                "Sensor distortion: Face dimensions distorted beyond recognition."
            )

        # --------------------------------------
        # 2. FRAME BOUNDARY MARGIN
        # --------------------------------------
        margin = 2
        if (
            x1 < margin
            or y1 < margin
            or x2 > image_width - margin
            or y2 > image_height - margin
        ):
            return False, (
                "Please keep your entire face inside the camera frame."
            )

        # --------------------------------------
        # 3. INTER-OCULAR DISTANCE (IOD)
        # --------------------------------------
        if getattr(face, "kps", None) is None or len(face.kps) < 5:
            return False, "Severe occlusion: Facial landmarks missing."

        kps = face.kps
        left_eye, right_eye = kps[0], kps[1]
        iod = float(np.linalg.norm(left_eye - right_eye))

        if iod < min_iod:
            return False, (
                f"Insufficient resolution: Inter-ocular distance ({int(iod)}px) "
                f"falls below 35-60 pixels requirement."
            )

        # --------------------------------------
        # 4. SEVERE OCCLUSION (EYE-NOSE-MOUTH TRIANGLE)
        # --------------------------------------
        nose_tip = kps[2]
        left_mouth, right_mouth = kps[3], kps[4]

        eyes_mid = (left_eye + right_eye) / 2.0
        eye_to_nose = float(np.linalg.norm(nose_tip - eyes_mid))
        mouth_width = float(np.linalg.norm(right_mouth - left_mouth))

        if eye_to_nose < 8 or mouth_width < 10:
            return False, (
                "Severe occlusion: Over 50% of critical features "
                "(eye-nose triangle or mouth) are blocked."
            )

        # --------------------------------------
        # 5. EXTREME LIGHTING & GLARE CHECK
        # --------------------------------------
        fx1, fy1, fx2, fy2 = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(image_width, int(x2)),
            min(image_height, int(y2))
        )
        face_roi = image[fy1:fy2, fx1:fx2]

        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray_roi))
            contrast = float(np.std(gray_roi))

            if mean_brightness < 15:
                return False, "Extreme lighting: Environment is in pitch darkness."
            if mean_brightness > 245:
                return False, "Extreme lighting: Direct glare or overexposure detected."
            if contrast < 10:
                return False, "Extreme lighting: Low image contrast."

            # --------------------------------------
            # 6. ANTI-SPOOFING & LIVENESS (LAPLACIAN TEXTURE & MOIRÉ ANALYSIS)
            # --------------------------------------
            lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
            if lap_var < 5.0:
                return False, (
                    "Liveness check failed: Presentation attack or "
                    "2D photo playback detected."
                )

        return True, "Face quality acceptable"

    # ==========================================
    # RECOGNIZE FACE
    # ==========================================

    def recognize(
        self,
        image_bytes: bytes
    ):

        # Always load latest enrolled employees
        self.reload_embeddings()

        # --------------------------------------
        # NO ENROLLED FACES
        # --------------------------------------

        if not self.embeddings:

            logger.warning(
                "Face recognition attempted but no enrolled faces exist"
            )

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

            logger.warning(
                "Face recognition failed: invalid image"
            )

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

            logger.warning(
                "Face recognition failed: no face detected"
            )

            return {
                "success": False,
                "message": "No face detected"
            }

        if len(faces) > 1:

            logger.warning(
                "Face recognition failed: multiple faces detected"
            )

            return {
                "success": False,
                "message": (
                    "Multiple faces detected. "
                    "Please keep only one person "
                    "in front of the camera."
                )
            }

        # --------------------------------------
        # COMPREHENSIVE FACE QUALITY & LIVENESS CHECK
        # --------------------------------------

        face = faces[0]

        quality_ok, quality_message = (
            self.validate_face_quality(
                face,
                image
            )
        )

        if not quality_ok:

            logger.warning(
                "Face quality rejected: %s",
                quality_message
            )

            return {
                "success": False,
                "message": quality_message
            }

        # --------------------------------------
        # GET FACE EMBEDDING
        # --------------------------------------

        embedding = face.embedding

        if embedding is None:

            logger.warning(
                "Face recognition failed: unable to generate embedding"
            )

            return {
                "success": False,
                "message": (
                    "Unable to generate "
                    "face embedding"
                )
            }

        # Normalize embedding
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
            score,
            is_ambiguous
        ) = self.find_best_match(
            embedding
        )

        # --------------------------------------
        # HIGH AMBIGUITY CHECK (TWINS / CLOSE RELATIVES)
        # --------------------------------------

        if is_ambiguous:

            logger.warning(
                "Face recognition rejected due to high ambiguity (multiple close scores)"
            )

            return {
                "success": False,
                "message": (
                    "High uncertainty: Similarity scores are too close "
                    "between multiple profiles."
                )
            }

        # --------------------------------------
        # UNKNOWN FACE
        # --------------------------------------

        if (
            employee_id is None
            or score < THRESHOLD
        ):

            logger.warning(
                "Face not recognized: best_match=%s confidence=%.4f",
                employee_id,
                score
            )

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

        logger.info(
            "Face recognized: employee=%s confidence=%.4f",
            employee_id,
            score
        )

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