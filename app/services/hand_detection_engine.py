"""
Hand Detection Engine (MediaPipe Hands / Hand-Face Occlusion Check)
Detects hands in frame and checks for spatial overlap with face bounding boxes.
"""

import logging
import cv2
import numpy as np
from app.core.liveness_config import liveness_config
from app.schemas.liveness import HandOverlapResult

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe package not installed. Hand detection falling back to skin contour analysis.")


class HandDetectionEngine:

    def __init__(self):
        self.overlap_threshold = liveness_config.HAND_FACE_OVERLAP_THRESHOLD
        self.mp_hands = None
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=4,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe Hands initialized for hand-face occlusion check.")
            except Exception as e:
                logger.warning("Could not initialize MediaPipe Hands: %s", e)

    def check_hand_face_overlap(self, image: np.ndarray, face_bbox: list) -> HandOverlapResult:
        """
        Check if any detected hand overlaps with the face bounding box.
        """
        if image is None or len(image.shape) != 3:
            return HandOverlapResult(has_hand_overlap=False, overlap_ratio=0.0)

        h, w, _ = image.shape
        fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox[:4]]
        face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))

        hand_bboxes = []

        if self.mp_hands is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    xs = [lm.x * w for lm.x in hand_landmarks.landmark]
                    ys = [lm.y * h for lm.y in hand_landmarks.landmark]
                    
                    hx1, hx2 = min(xs), max(xs)
                    hy1, hy2 = min(ys), max(ys)
                    hand_bboxes.append((hx1, hy1, hx2, hy2))

        # Fallback skin contour check if no mediapipe hands or to augment
        if not hand_bboxes:
            hand_bboxes = self._detect_hand_contours_fallback(image, (fx1, fy1, fx2, fy2))

        if not hand_bboxes:
            return HandOverlapResult(has_hand_overlap=False, overlap_ratio=0.0, hands_detected=0)

        max_overlap_ratio = 0.0

        for hx1, hy1, hx2, hy2 in hand_bboxes:
            # Calculate intersection box
            ix1 = max(fx1, hx1)
            iy1 = max(fy1, hy1)
            ix2 = min(fx2, hx2)
            iy2 = min(fy2, hy2)

            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection_area / face_area
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio

        has_overlap = max_overlap_ratio >= self.overlap_threshold

        return HandOverlapResult(
            has_hand_overlap=has_overlap,
            overlap_ratio=float(max_overlap_ratio),
            hands_detected=len(hand_bboxes)
        )

    def _detect_hand_contours_fallback(self, image: np.ndarray, face_box: tuple) -> list:
        """
        Fallback heuristic to find non-face skin blobs near face boundaries.
        """
        h, w, _ = image.shape
        fx1, fy1, fx2, fy2 = face_box

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Zero out the core of the face box so face itself isn't treated as hand
        margin = 10
        cx1, cy1 = int(max(0, fx1 + margin)), int(max(0, fy1 + margin))
        cx2, cy2 = int(min(w, fx2 - margin)), int(min(h, fy2 - margin))
        
        if cx2 > cx1 and cy2 > cy1:
            skin_mask[cy1:cy2, cx1:cx2] = 0

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:  # Minimum hand blob size
                x, y, bw, bh = cv2.boundingRect(c)
                hand_boxes.append((x, y, x + bw, y + bh))

        return hand_boxes
