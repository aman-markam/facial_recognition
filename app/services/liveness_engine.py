"""
Liveness Engine (InsightFace 2.0 Official Liveness Addon + Multi-Layer Pipeline)
Integrates:
1. InsightFace Multi-Face Detection
2. FaceAttributeEngine (Mask / Sunglasses / Eyes)
3. HandDetectionEngine (MediaPipe Hands / Face-Hand Overlap)
4. InsightFace Official Liveness Addon (addons=["liveness"])
5. AntiSpoofEngine (MiniFASNet Presentation Attack Detection)
"""

import logging
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.liveness_config import liveness_config
from app.services.face_attribute_engine import FaceAttributeEngine
from app.services.hand_detection_engine import HandDetectionEngine
from app.services.anti_spoof_engine import AntiSpoofEngine
from app.schemas.liveness import (
    FaceLivenessResult,
    MultiFaceLivenessResponse
)

logger = logging.getLogger(__name__)


class LivenessEngine:

    def __init__(self):
        logger.info("Initializing InsightFace 2.0 Liveness Engine with addons=['liveness']...")
        
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                addons=["liveness"]
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace Liveness Engine ready.")
        except Exception as e:
            logger.error("Failed to load InsightFace with addons=['liveness']: %s", e)
            logger.info("Falling back to standard InsightFace buffalo_l...")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # Auxiliary Engines
        self.attribute_engine = FaceAttributeEngine()
        self.hand_engine = HandDetectionEngine()
        self.anti_spoof_engine = AntiSpoofEngine()

    def analyze_image(self, image: np.ndarray) -> MultiFaceLivenessResponse:
        """
        Analyze an OpenCV frame for all detected faces up to MAX_FACES.
        Returns MultiFaceLivenessResponse.
        """
        if image is None or len(image.shape) != 3:
            return MultiFaceLivenessResponse(
                success=False,
                message="Invalid image input"
            )

        # 1. Multi-Face Detection
        faces = self.app.get(image)

        if not faces:
            return MultiFaceLivenessResponse(
                success=True,
                total_faces=0,
                live_faces=0,
                spoof_faces=0,
                faces=[],
                message="No face detected"
            )

        max_faces = liveness_config.MAX_FACES
        if len(faces) > max_faces:
            logger.warning("Detected %d faces, truncating to MAX_FACES=%d", len(faces), max_faces)
            faces = faces[:max_faces]

        face_results = []
        live_count = 0
        spoof_count = 0

        for idx, face in enumerate(faces):
            bbox = face.bbox.tolist()

            # Step A: Face State / Attribute Check (Mask / Sunglasses / Eyes)
            attr_res = self.attribute_engine.analyze_face(image, bbox)
            if attr_res.is_occluded:
                spoof_count += 1
                face_results.append(
                    FaceLivenessResult(
                        face_index=idx,
                        bbox=bbox,
                        is_live=False,
                        live_score=0.0,
                        anti_spoof_score=0.0,
                        is_occluded=True,
                        occlusion_reason=attr_res.reason,
                        status="occluded_rejected"
                    )
                )
                continue

            # Step B: Hand Detection (Hand-Face Overlap)
            hand_res = self.hand_engine.check_hand_face_overlap(image, bbox)
            if hand_res.has_hand_overlap:
                spoof_count += 1
                face_results.append(
                    FaceLivenessResult(
                        face_index=idx,
                        bbox=bbox,
                        is_live=False,
                        live_score=0.0,
                        anti_spoof_score=0.0,
                        is_occluded=True,
                        occlusion_reason=f"Face occluded by hand (overlap {hand_res.overlap_ratio:.2f})",
                        status="hand_occlusion_rejected"
                    )
                )
                continue

            # Step C: InsightFace Official Liveness Addon Score
            insight_live = False
            insight_score = 0.0
            insight_status = "ok"

            liveness_data = getattr(face, "liveness", None)
            if liveness_data is not None:
                if isinstance(liveness_data, dict):
                    insight_score = float(liveness_data.get("live_score", 0.0))
                    insight_live = bool(liveness_data.get("is_live", False))
                    insight_status = str(liveness_data.get("status", "ok"))
                elif hasattr(liveness_data, "live_score"):
                    insight_score = float(getattr(liveness_data, "live_score", 0.0))
                    insight_live = bool(getattr(liveness_data, "is_live", False))
                    insight_status = str(getattr(liveness_data, "status", "ok"))

            # Step D: MiniFASNet Anti-Spoofing Score
            pad_res = self.anti_spoof_engine.analyze_spoof(image, bbox)
            anti_spoof_score = pad_res["score"]

            # Combined Liveness Verdict
            is_face_live = (
                (insight_score >= liveness_config.LIVENESS_THRESHOLD or insight_live) and
                (anti_spoof_score >= liveness_config.MINIFASNET_THRESHOLD)
            )

            if is_face_live:
                live_count += 1
            else:
                spoof_count += 1

            rejection_reason = None
            if not is_face_live:
                if insight_score < liveness_config.LIVENESS_THRESHOLD:
                    rejection_reason = f"InsightFace liveness failed (score {insight_score:.2f})"
                elif anti_spoof_score < liveness_config.MINIFASNET_THRESHOLD:
                    rejection_reason = pad_res.get("reason", "Anti-spoofing check failed")

            face_results.append(
                FaceLivenessResult(
                    face_index=idx,
                    bbox=bbox,
                    is_live=is_face_live,
                    live_score=insight_score,
                    anti_spoof_score=anti_spoof_score,
                    is_occluded=False,
                    occlusion_reason=rejection_reason,
                    status=insight_status
                )
            )

        return MultiFaceLivenessResponse(
            success=True,
            total_faces=len(faces),
            live_faces=live_count,
            spoof_faces=spoof_count,
            faces=face_results,
            message=f"Processed {len(faces)} face(s): {live_count} live, {spoof_count} spoof/rejected"
        )
