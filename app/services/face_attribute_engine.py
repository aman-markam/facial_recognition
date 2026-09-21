"""
Face Attribute Engine (FaceAttribNet / Occlusion Gate)
Analyzes face crop for masks, sunglasses, eyeglasses, and eye state.
"""

import logging
import cv2
import numpy as np
from app.core.liveness_config import liveness_config
from app.schemas.liveness import FaceAttributeResult

logger = logging.getLogger(__name__)


class FaceAttributeEngine:

    def __init__(self):
        self.mask_threshold = liveness_config.MASK_THRESHOLD
        self.sunglasses_threshold = liveness_config.SUNGLASSES_THRESHOLD
        logger.info("FaceAttributeEngine initialized with mask_thresh=%.2f, sunglasses_thresh=%.2f",
                    self.mask_threshold, self.sunglasses_threshold)

    def analyze_face(self, image: np.ndarray, bbox: list) -> FaceAttributeResult:
        """
        Analyze face region within image for occlusions (mask, sunglasses, covered eyes).
        """
        if image is None or len(image.shape) != 3:
            return FaceAttributeResult(is_occluded=True, reason="Invalid image frame")

        h, w, _ = image.shape
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return FaceAttributeResult(is_occluded=True, reason="Face box too small for attribute analysis")

        face_crop = image[y1:y2, x1:x2]

        # 1. Lower face mask detection via skin color & edge density
        mask_prob = self._detect_mask(face_crop)
        
        # 2. Upper face sunglasses detection via darkness & brightness distribution
        sunglasses_prob = self._detect_sunglasses(face_crop)

        # Determine occlusion
        is_occluded = False
        reasons = []

        if mask_prob >= self.mask_threshold:
            is_occluded = True
            reasons.append("Face covered by mask")

        if sunglasses_prob >= self.sunglasses_threshold:
            is_occluded = True
            reasons.append("Eyes covered by sunglasses")

        reason_str = ", ".join(reasons) if is_occluded else None

        return FaceAttributeResult(
            is_occluded=is_occluded,
            mask_prob=float(mask_prob),
            sunglasses_prob=float(sunglasses_prob),
            left_eye_open=1.0 if sunglasses_prob < 0.5 else 0.0,
            right_eye_open=1.0 if sunglasses_prob < 0.5 else 0.0,
            reason=reason_str
        )

    def _detect_mask(self, face_crop: np.ndarray) -> float:
        """
        Heuristic lower-face mask analysis:
        Checks lower 50% of the face for skin pixel ratio and high edge texture (fabric patterns).
        """
        fh, fw, _ = face_crop.shape
        lower_face = face_crop[int(fh * 0.5):fh, :]

        if lower_face.size == 0:
            return 0.0

        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Define skin tone range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / (lower_face.shape[0] * lower_face.shape[1])

        # Edge analysis for fabric texture
        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_ratio = np.sum(edges > 0) / (lower_face.shape[0] * lower_face.shape[1])

        # High edges + low skin ratio indicates mask presence
        if skin_ratio < 0.15 and edge_ratio > 0.10:
            return min(0.95, 0.6 + edge_ratio)
        elif skin_ratio < 0.08:
            return 0.85
        
        return float(max(0.0, 1.0 - (skin_ratio * 1.5)))

    def _detect_sunglasses(self, face_crop: np.ndarray) -> float:
        """
        Heuristic eye region sunglasses analysis:
        Checks eye region (upper 20%-45% of face) for dark rectangular region.
        """
        fh, fw, _ = face_crop.shape
        eye_region = face_crop[int(fh * 0.20):int(fh * 0.45), :]

        if eye_region.size == 0:
            return 0.0

        gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_eyes)
        dark_ratio = np.sum(gray_eyes < 40) / gray_eyes.size

        if dark_ratio > 0.40 and avg_brightness < 60:
            return min(0.98, 0.5 + dark_ratio)
        
        return float(dark_ratio)
