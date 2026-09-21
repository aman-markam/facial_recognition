import logging

import cv2
import numpy as np

from insightface.app import FaceAnalysis


logger = logging.getLogger(__name__)


LIVE_THRESHOLD = 0.80


class LivenessService:

    def __init__(self):

        logger.info(
            "Loading InsightFace liveness model..."
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ],
            addons=[
                "liveness"
            ]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        logger.info(
            "InsightFace liveness model loaded."
        )

    # ==========================================
    # DECODE IMAGE
    # ==========================================

    @staticmethod
    def _decode_image(
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

        return image

    # ==========================================
    # CHECK SINGLE FACE LIVENESS
    # ==========================================

    def _check_face_liveness(self, face):
        """
        Check liveness for an already detected InsightFace Face object.

        InsightFace returns liveness as:

        {
            "status": "ok",
            "is_live": True,
            "live_score": 0.99
        }
        """

        try:
            liveness_data = face.get("liveness")

            if not liveness_data:
                logger.warning(
                    "Liveness data not available for detected face"
                )

                return {
                    "is_live": False,
                    "live_score": 0.0,
                    "message": "Liveness data unavailable"
                }

            # InsightFace returns a dictionary.
            live_score = liveness_data.get("live_score")
            is_live = liveness_data.get("is_live")

            if live_score is None:
                logger.warning(
                    "Liveness score missing from InsightFace result: %s",
                    liveness_data
                )

                return {
                    "is_live": False,
                    "live_score": 0.0,
                    "message": "Liveness score unavailable"
                }

            live_score = float(live_score)

            # Use InsightFace's result AND our threshold.
            is_live = bool(is_live) and live_score >= LIVE_THRESHOLD

            logger.debug(
                "Face liveness: score=%.4f threshold=%.2f live=%s",
                live_score,
                LIVE_THRESHOLD,
                is_live
            )

            return {
                "is_live": is_live,
                "live_score": round(live_score, 4),
                "message": (
                    "Live face detected"
                    if is_live
                    else "Spoof or non-live face detected"
                )
            }

        except (TypeError, ValueError, AttributeError):
            logger.exception(
                "Invalid liveness result returned by InsightFace"
            )

            return {
                "is_live": False,
                "live_score": 0.0,
                "message": "Invalid liveness result"
            }

        except Exception:
            logger.exception(
                "Unexpected error during face liveness check"
            )

            return {
                "is_live": False,
                "live_score": 0.0,
                "message": "Liveness check failed"
            }
    # ==========================================
    # SINGLE-PERSON API
    # ==========================================

    def check_liveness(
        self,
        image_bytes: bytes
    ):

        image = self._decode_image(
            image_bytes
        )

        if image is None:

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": "Invalid image"
            }

        try:

            faces = self.app.get(
                image
            )

        except Exception:

            logger.exception(
                "Liveness detection failed"
            )

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Liveness detection failed"
                )
            }

        if not faces:

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": "No face detected"
            }

        if len(faces) > 1:

            return {
                "success": False,
                "is_live": False,
                "live_score": 0.0,
                "message": (
                    "Multiple faces detected. "
                    "Use multi-person liveness."
                )
            }

        result = self._check_face_liveness(
            faces[0]
        )

        return {
            "success": True,
            "is_live": result["is_live"],
            "live_score": result["live_score"],
            "message": result["message"]
        }

    # ==========================================
    # MULTI-FACE API
    # ==========================================

    def check_multiple_faces(
        self,
        image_bytes: bytes
    ):

        image = self._decode_image(
            image_bytes
        )

        if image is None:

            return {
                "success": False,
                "total_faces": 0,
                "faces": [],
                "message": "Invalid image"
            }

        try:

            detected_faces = self.app.get(
                image
            )

        except Exception:

            logger.exception(
                "Multi-face liveness detection failed"
            )

            return {
                "success": False,
                "total_faces": 0,
                "faces": [],
                "message": (
                    "Liveness detection failed"
                )
            }

        if not detected_faces:

            logger.info(
                "Multi-face liveness: "
                "no faces detected"
            )

            return {
                "success": True,
                "total_faces": 0,
                "faces": [],
                "message": "No faces detected"
            }

        results = []

        for index, face in enumerate(
            detected_faces,
            start=1
        ):

            liveness_result = (
                self._check_face_liveness(
                    face
                )
            )

            result = {
                "face_index": index,
                "is_live": (
                    liveness_result[
                        "is_live"
                    ]
                ),
                "live_score": (
                    liveness_result[
                        "live_score"
                    ]
                ),
                "message": (
                    liveness_result[
                        "message"
                    ]
                )
            }

            results.append(
                result
            )

            logger.info(
                "Face %d liveness: "
                "live=%s score=%.4f",
                index,
                result["is_live"],
                result["live_score"]
            )

        live_faces = sum(
            1
            for result in results
            if result["is_live"]
        )

        spoof_faces = (
            len(results) -
            live_faces
        )

        return {
            "success": True,
            "total_faces": len(
                results
            ),
            "live_faces": live_faces,
            "spoof_faces": spoof_faces,
            "faces": results,
            "message": (
                f"{live_faces}/"
                f"{len(results)} "
                "face(s) passed liveness"
            )
        }


liveness_service = (
    LivenessService()
)