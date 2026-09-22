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

        # 1. Lower & mid face occlusion detection (mask, scarf, cloth, bandana)
        mask_prob = self._detect_mask_or_cloth(face_crop)
        
        # 2. Upper face sunglasses & dark coverage detection
        sunglasses_prob = self._detect_sunglasses(face_crop)

        # Determine occlusion
        is_occluded = False
        reasons = []

        if mask_prob >= self.mask_threshold:
            is_occluded = True
            reasons.append("Face covered by mask, scarf, or cloth")

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

    def _detect_mask_or_cloth(self, face_crop: np.ndarray) -> float:
        """
        Lower & mid face occlusion analysis (mask, scarf, cloth, bandana, hand cover):
        Checks lower 55% of the face for skin pixel ratio, fabric texture edges, and color consistency.
        """
        fh, fw, _ = face_crop.shape
        lower_face = face_crop[int(fh * 0.45):fh, :]

        if lower_face.size == 0:
            return 0.0

        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Define HSV skin tone range
        lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / float(lower_face.shape[0] * lower_face.shape[1])

        # Edge analysis for fabric/cloth texture
        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_ratio = np.sum(edges > 0) / float(lower_face.shape[0] * lower_face.shape[1])

        # Fabric/cloth: low skin ratio (< 0.20) or high texture edge ratio (> 0.12)
        if skin_ratio < 0.15 and edge_ratio > 0.08:
            return min(0.98, 0.65 + edge_ratio)
        elif skin_ratio < 0.08:
            return 0.90
        elif skin_ratio < 0.25 and edge_ratio > 0.12:
            return 0.80

        # Calculate skin deficit
        prob = max(0.0, 1.0 - (skin_ratio * 1.6))
        return float(min(1.0, prob))

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
        dark_ratio = np.sum(gray_eyes < 40) / float(gray_eyes.size)

        if dark_ratio > 0.40 and avg_brightness < 60:
            return min(0.98, 0.5 + dark_ratio)
        
        return float(dark_ratio)
