import logging

import cv2
import numpy as np

from insightface.app import FaceAnalysis


logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

LIVE_THRESHOLD = 0.80


class LivenessService:

    def __init__(self):

        logger.info(
            "Loading InsightFace with liveness..."
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ],
            addons=[
                "liveness"
            ],
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        logger.info(
            "InsightFace liveness service loaded."
        )

    # ==========================================
    # CHECK LIVENESS
    # ==========================================

    def check_liveness(
        self,
        image_bytes: bytes
    ):

        # --------------------------------------
        # DECODE IMAGE
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
                "Liveness check failed: invalid image"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": "Invalid image"
            }

        # --------------------------------------
        # DETECT FACE + LIVENESS
        # --------------------------------------

        try:

            faces = self.app.get(
                image
            )

        except Exception:

            logger.exception(
                "Liveness inference failed"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Unable to perform "
                    "liveness check"
                )
            }

        # --------------------------------------
        # NO FACE
        # --------------------------------------

        if not faces:

            logger.warning(
                "Liveness check failed: no face detected"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": "No face detected"
            }

        # --------------------------------------
        # MULTIPLE FACES
        # --------------------------------------

        if len(faces) > 1:

            logger.warning(
                "Liveness check rejected: multiple faces"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Multiple faces detected. "
                    "Only one person should be "
                    "in front of the kiosk."
                )
            }

        # --------------------------------------
        # GET FACE
        # --------------------------------------

        face = faces[0]

        # --------------------------------------
        # CHECK LIVENESS RESULT
        # --------------------------------------

        liveness = getattr(
            face,
            "liveness",
            None
        )

        if liveness is None:

            logger.error(
                "Liveness result is unavailable"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Liveness verification "
                    "is unavailable"
                )
            }

        live_score = getattr(
            liveness,
            "live_score",
            None
        )

        is_live = getattr(
            liveness,
            "is_live",
            None
        )

        # --------------------------------------
        # VALIDATE SCORE
        # --------------------------------------

        if live_score is None:

            logger.error(
                "Liveness model returned no score"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Unable to determine "
                    "whether the face is live"
                )
            }

        live_score = float(
            live_score
        )

        # --------------------------------------
        # APPLY OUR THRESHOLD
        # --------------------------------------

        passed = (
            is_live is True
            and live_score >= LIVE_THRESHOLD
        )

        # --------------------------------------
        # SPOOF
        # --------------------------------------

        if not passed:

            logger.warning(
                "Liveness rejected: score=%.4f",
                live_score
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": round(
                    live_score,
                    4
                ),
                "message": (
                    "Liveness verification failed. "
                    "Please look directly at the camera."
                )
            }

        # --------------------------------------
        # LIVE
        # --------------------------------------

        logger.info(
            "Liveness passed: score=%.4f",
            live_score
        )

        return {
            "success": True,
            "is_live": True,
            "live_score": round(
                live_score,
                4
            ),
            "message": "Live person detected"
        }


liveness_service = LivenessService()
