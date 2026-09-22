import json
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

EMBEDDINGS_PATH = (
    ROOT / "face_data" / "embeddings.json"
)

# Face recognition threshold
THRESHOLD = 0.55

# InsightFace liveness threshold
LIVE_THRESHOLD = 0.80

# Face quality (lowered limits for distant face detection)
MIN_FACE_WIDTH = 40
MIN_FACE_HEIGHT = 40
MIN_IOD = 20.0

# Lighting
MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 245
MIN_CONTRAST = 10

# Blur
MIN_LAPLACIAN_VARIANCE = 5.0

# ============================================================
# OCCLUSION CONFIGURATION
# ============================================================

# Skin density threshold
SKIN_DENSITY_THRESHOLD = 0.06

# Connected skin blob threshold
SKIN_BLOB_AREA_RATIO = 0.02

# Lower-face edge density
EDGE_DENSITY_THRESHOLD = 0.12

# Final occlusion confidence
OCCLUSION_CONFIDENCE_THRESHOLD = 0.40

# Generic rejection message
FACE_NOT_RECOGNIZED_MESSAGE = (
    "Face not recognized. "
    "Please keep your face fully visible."
)

logger = logging.getLogger(__name__)


class FaceEngine:

    # ========================================================
    # INITIALIZE
    # ========================================================

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
            det_size=(960, 960)
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

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

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

    # ========================================================
    # RELOAD EMBEDDINGS
    # ========================================================

    def reload_embeddings(self):

        self.embeddings = (
            self._load_embeddings()
        )

        logger.info(
            "Embeddings reloaded: %d employee(s)",
            len(self.embeddings)
        )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

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

    # ========================================================
    # FIND BEST MATCH
    # ========================================================

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

        # ====================================================
        # AMBIGUITY DETECTION
        # ====================================================

        is_ambiguous = False

        if (
            best_score >= THRESHOLD
            and
            (best_score - second_best_score) < 0.05
        ):

            is_ambiguous = True

        return (
            best_employee,
            best_score,
            is_ambiguous
        )

    # ========================================================
    # OCCLUSION - SKIN MASK
    # ========================================================

    def _skin_mask(
        self,
        bgr: np.ndarray
    ) -> np.ndarray:

        """
        Detect likely skin regions using a combination
        of YCrCb and HSV.

        This is a heuristic signal primarily useful for
        detecting a hand entering the face region.
        """

        ycrcb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2YCrCb
        )

        hsv = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2HSV
        )

        # ----------------------------------------------------
        # YCrCb skin rule
        # ----------------------------------------------------

        mask_ycrcb = cv2.inRange(
            ycrcb,
            np.array(
                [0, 133, 77],
                dtype=np.uint8
            ),
            np.array(
                [255, 173, 127],
                dtype=np.uint8
            )
        )

        # ----------------------------------------------------
        # HSV skin rule
        # ----------------------------------------------------

        mask_hsv = cv2.inRange(
            hsv,
            np.array(
                [0, 30, 60],
                dtype=np.uint8
            ),
            np.array(
                [25, 180, 255],
                dtype=np.uint8
            )
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        mask = cv2.bitwise_and(
            mask_ycrcb,
            mask_hsv
        )

        # ----------------------------------------------------
        # Clean mask
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        return mask

    # ========================================================
    # OCCLUSION - SKIN SIGNAL
    # ========================================================

    def _skin_signal(
        self,
        bgr: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:

        h, w = bgr.shape[:2]

        total = h * w

        if total <= 0:

            return (
                0.0,
                0.0,
                np.zeros(
                    (h, w),
                    dtype=np.uint8
                )
            )

        mask = self._skin_mask(
            bgr
        )

        density = (
            float(
                np.count_nonzero(mask)
            )
            /
            total
        )

        # ----------------------------------------------------
        # Connected components
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                8
            )
        )

        largest = 0

        if num_labels > 1:

            largest = int(
                np.max(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA
                    ]
                )
            )

        blob_ratio = (
            float(largest)
            /
            float(total)
        )

        return (
            density,
            blob_ratio,
            mask
        )

    # ========================================================
    # OCCLUSION - EDGE SIGNAL
    # ========================================================

    def _edge_signal(
        self,
        bgr: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            gray,
            60,
            160
        )

        h, w = gray.shape

        total = h * w

        if total <= 0:

            return (
                0.0,
                0.0,
                edges
            )

        edge_density = (
            float(
                np.count_nonzero(edges)
            )
            /
            total
        )

        # ----------------------------------------------------
        # Lower-face band
        #
        # Relative to the detected face ROI.
        # ----------------------------------------------------

        y0 = int(
            0.45 * h
        )

        y1 = int(
            0.85 * h
        )

        x0 = int(
            0.20 * w
        )

        x1 = int(
            0.80 * w
        )

        band = edges[
            y0:y1,
            x0:x1
        ]

        if band.size:

            band_density = (
                float(
                    np.count_nonzero(
                        band
                    )
                )
                /
                float(band.size)
            )

        else:

            band_density = 0.0

        return (
            edge_density,
            band_density,
            edges
        )

    # ========================================================
    # OCCLUSION - COLOR REGION
    # ========================================================

    def _color_region_signal(
        self,
        bgr: np.ndarray
    ) -> float:

        """
        Compare upper-face and lower-face color statistics.

        Large divergence can indicate:
            - mask
            - scarf
            - cloth
            - hand
            - other foreign object
        """

        h, w = bgr.shape[:2]

        # ----------------------------------------------------
        # Upper face
        # ----------------------------------------------------

        upper = bgr[
            int(0.05 * h):
            int(0.40 * h),

            int(0.25 * w):
            int(0.75 * w)
        ]

        # ----------------------------------------------------
        # Lower face
        # ----------------------------------------------------

        lower = bgr[
            int(0.55 * h):
            int(0.90 * h),

            int(0.25 * w):
            int(0.75 * w)
        ]

        if (
            upper.size == 0
            or
            lower.size == 0
        ):

            return 0.0

        # ----------------------------------------------------
        # HSV statistics
        # ----------------------------------------------------

        def stats(
            patch
        ):

            hsv = cv2.cvtColor(
                patch,
                cv2.COLOR_BGR2HSV
            )

            pixels = hsv.reshape(
                -1,
                3
            )

            return (
                pixels.mean(axis=0),
                pixels.std(axis=0)
            )

        mu_upper, sd_upper = stats(
            upper
        )

        mu_lower, sd_lower = stats(
            lower
        )

        # ----------------------------------------------------
        # Normalize differences
        # ----------------------------------------------------

        d_hue = (
            abs(
                float(
                    mu_upper[0]
                    -
                    mu_lower[0]
                )
            )
            /
            90.0
        )

        d_sat = (
            abs(
                float(
                    mu_upper[1]
                    -
                    mu_lower[1]
                )
            )
            /
            128.0
        )

        d_val = (
            abs(
                float(
                    mu_upper[2]
                    -
                    mu_lower[2]
                )
            )
            /
            128.0
        )

        d_std = (
            float(
                np.linalg.norm(
                    sd_upper
                    -
                    sd_lower
                )
            )
            /
            128.0
        )

        divergence = (
            0.25
            *
            (
                d_hue
                +
                d_sat
                +
                d_val
            )
            +
            0.25
            *
            d_std
        )

        return float(
            min(
                1.0,
                divergence
            )
        )

    # ========================================================
    # COMPLETE OCCLUSION ANALYSIS
    # ========================================================

    def analyze_occlusion(
        self,
        face_roi: np.ndarray
    ):

        """
        Analyze a detected face ROI for possible
        hand/mask/scarf occlusion.

        Returns:

            detected
            confidence
            reason
            score_breakdown
        """

        if (
            face_roi is None
            or
            face_roi.size == 0
        ):

            return {
                "detected": True,
                "confidence": 1.0,
                "reason": "Invalid face region",
                "score_breakdown": {}
            }

        # ====================================================
        # SIGNAL 1 + 2: SKIN
        # ====================================================

        (
            skin_density,
            blob_ratio,
            skin_mask
        ) = self._skin_signal(
            face_roi
        )

        # ====================================================
        # SIGNAL 3: EDGES
        # ====================================================

        (
            edge_density,
            lower_face_edge,
            edges
        ) = self._edge_signal(
            face_roi
        )

        # ====================================================
        # SIGNAL 4: COLOR
        # ====================================================

        color_divergence = (
            self._color_region_signal(
                face_roi
            )
        )

        # ====================================================
        # SKIN SCORE
        # ====================================================

        skin_score = 0.0

        if (
            skin_density
            >=
            SKIN_DENSITY_THRESHOLD
        ):

            skin_score += 0.5

        if (
            blob_ratio
            >=
            SKIN_BLOB_AREA_RATIO
        ):

            skin_score += 0.5

        # ====================================================
        # EDGE SCORE
        # ====================================================

        edge_score = 0.0

        if (
            lower_face_edge
            >=
            EDGE_DENSITY_THRESHOLD
        ):

            edge_score = 1.0

        elif (
            lower_face_edge
            >=
            EDGE_DENSITY_THRESHOLD * 0.75
        ):

            edge_score = 0.5

        # ====================================================
        # COLOR SCORE
        # ====================================================

        if color_divergence >= 0.45:

            color_score = 1.0

        elif color_divergence >= 0.30:

            color_score = 0.5

        else:

            color_score = 0.0

        # ====================================================
        # WEIGHTED SCORE
        # ====================================================

        weights = {
            "skin": 0.45,
            "edges": 0.25,
            "color": 0.30
        }

        confidence = (
            weights["skin"]
            *
            skin_score
            +
            weights["edges"]
            *
            edge_score
            +
            weights["color"]
            *
            color_score
        )

        # ====================================================
        # IMPORTANT:
        #
        # We require multiple signals before rejecting.
        #
        # This prevents a normal face from being rejected
        # just because of one skin/color anomaly.
        # ====================================================

        positive_signals = 0

        if skin_score > 0:

            positive_signals += 1

        if edge_score > 0:

            positive_signals += 1

        if color_score > 0:

            positive_signals += 1

        # ----------------------------------------------------
        # Strong occlusion:
        #
        # Two or more independent signals + confidence.
        # ----------------------------------------------------

        detected = (
            confidence
            >=
            OCCLUSION_CONFIDENCE_THRESHOLD
            and
            positive_signals >= 2
        )

        # ====================================================
        # SPECIAL HAND SIGNAL
        #
        # A large skin blob combined with lower-face
        # disruption is a strong indication of a hand.
        # ====================================================

        hand_like = (
            skin_density
            >=
            SKIN_DENSITY_THRESHOLD
            and
            blob_ratio
            >=
            SKIN_BLOB_AREA_RATIO
            and
            (
                lower_face_edge
                >=
                EDGE_DENSITY_THRESHOLD * 0.75
            )
        )

        if hand_like:

            detected = True

        # ====================================================
        # REASON
        # ====================================================

        reasons = []

        if skin_score > 0:

            reasons.append(
                (
                    f"skin("
                    f"density={skin_density:.3f}, "
                    f"blob={blob_ratio:.3f}"
                    f")"
                )
            )

        if edge_score > 0:

            reasons.append(
                (
                    "lower_face_edge="
                    f"{lower_face_edge:.3f}"
                )
            )

        if color_score > 0:

            reasons.append(
                (
                    "color_divergence="
                    f"{color_divergence:.3f}"
                )
            )

        if hand_like:

            reasons.append(
                "hand-like skin region detected"
            )

        if not reasons:

            reasons.append(
                "no significant occlusion signals"
            )

        return {
            "detected": bool(detected),
            "confidence": round(
                float(confidence),
                4
            ),
            "reason": " | ".join(
                reasons
            ),
            "score_breakdown": {
                "skin_density":
                    float(skin_density),

                "largest_skin_blob_ratio":
                    float(blob_ratio),

                "edge_density":
                    float(edge_density),

                "lower_face_edge_density":
                    float(lower_face_edge),

                "color_divergence":
                    float(color_divergence),

                "skin_score":
                    float(skin_score),

                "edge_score":
                    float(edge_score),

                "color_score":
                    float(color_score),

                "positive_signals":
                    positive_signals,

                "hand_like":
                    bool(hand_like),

                "final_confidence":
                    float(confidence)
            }
        }

    # ========================================================
    # FACE QUALITY / OCCLUSION / LIVENESS
    # ========================================================

    def validate_face_quality(
        self,
        face,
        image,
        min_iod: float = MIN_IOD
    ):

        image_height, image_width = (
            image.shape[:2]
        )

        bbox = face.bbox

        x1, y1, x2, y2 = bbox

        face_width = float(
            x2 - x1
        )

        face_height = float(
            y2 - y1
        )

        # ====================================================
        # 1. FACE RESOLUTION
        # ====================================================

        if (
            face_width < MIN_FACE_WIDTH
            or
            face_height < MIN_FACE_HEIGHT
        ):

            return (
                False,
                "Insufficient resolution: Move closer to the camera."
            )

        # ====================================================
        # 2. ASPECT RATIO
        # ====================================================

        aspect_ratio = (
            face_width
            /
            (face_height + 1e-6)
        )

        if (
            aspect_ratio < 0.50
            or
            aspect_ratio > 1.60
        ):

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 3. FRAME BOUNDARY
        # ====================================================

        margin = 2

        if (
            x1 < margin
            or
            y1 < margin
            or
            x2 > image_width - margin
            or
            y2 > image_height - margin
        ):

            return (
                False,
                "Please keep your entire face inside the camera frame."
            )

        # ====================================================
        # 4. LANDMARKS
        # ====================================================

        kps = getattr(
            face,
            "kps",
            None
        )

        if (
            kps is None
            or
            len(kps) < 5
        ):

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        left_eye = np.asarray(
            kps[0],
            dtype=np.float32
        )

        right_eye = np.asarray(
            kps[1],
            dtype=np.float32
        )

        nose_tip = np.asarray(
            kps[2],
            dtype=np.float32
        )

        left_mouth = np.asarray(
            kps[3],
            dtype=np.float32
        )

        right_mouth = np.asarray(
            kps[4],
            dtype=np.float32
        )

        # ====================================================
        # 5. INTER-OCULAR DISTANCE
        # ====================================================

        iod = float(
            np.linalg.norm(
                left_eye
                -
                right_eye
            )
        )

        if iod < min_iod:

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 6. EYE / NOSE GEOMETRY
        # ====================================================

        eyes_mid = (
            left_eye
            +
            right_eye
        ) / 2.0

        eye_to_nose = float(
            np.linalg.norm(
                nose_tip
                -
                eyes_mid
            )
        )

        if eye_to_nose < 8.0:

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 7. MOUTH LANDMARKS
        # ====================================================

        mouth_width = float(
            np.linalg.norm(
                right_mouth
                -
                left_mouth
            )
        )

        if mouth_width < 10.0:

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 8. EXTRACT FACE ROI
        # ====================================================

        fx1 = max(
            0,
            int(x1)
        )

        fy1 = max(
            0,
            int(y1)
        )

        fx2 = min(
            image_width,
            int(x2)
        )

        fy2 = min(
            image_height,
            int(y2)
        )

        if (
            fx2 <= fx1
            or
            fy2 <= fy1
        ):

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        face_roi = image[
            fy1:fy2,
            fx1:fx2
        ]

        if face_roi.size == 0:

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 9. OCCLUSION ANALYSIS
        # ====================================================

        occlusion_result = (
            self.analyze_occlusion(
                face_roi
            )
        )

        logger.info(
            (
                "Occlusion analysis: "
                "detected=%s "
                "confidence=%.4f "
                "reason=%s"
            ),
            occlusion_result["detected"],
            occlusion_result["confidence"],
            occlusion_result["reason"]
        )

        if occlusion_result["detected"]:

            logger.warning(
                "Face rejected because of possible occlusion"
            )

            return (
                False,
                FACE_NOT_RECOGNIZED_MESSAGE
            )

        # ====================================================
        # 10. LIGHTING
        # ====================================================

        gray_roi = cv2.cvtColor(
            face_roi,
            cv2.COLOR_BGR2GRAY
        )

        mean_brightness = float(
            np.mean(gray_roi)
        )

        contrast = float(
            np.std(gray_roi)
        )

        if mean_brightness < MIN_BRIGHTNESS:

            return (
                False,
                "Lighting is too dark. Please improve lighting."
            )

        if mean_brightness > MAX_BRIGHTNESS:

            return (
                False,
                "Too much glare. Please improve lighting."
            )

        if contrast < MIN_CONTRAST:

            return (
                False,
                "Insufficient image contrast."
            )

        # ====================================================
        # 11. BLUR
        # ====================================================

        laplacian_variance = float(
            cv2.Laplacian(
                gray_roi,
                cv2.CV_64F
            ).var()
        )

        if (
            laplacian_variance
            <
            MIN_LAPLACIAN_VARIANCE
        ):

            return (
                False,
                "Image is too blurry. Please hold still."
            )

        # ====================================================
        # 12. INSIGHTFACE LIVENESS
        # ====================================================

        liveness = getattr(
            face,
            "liveness",
            None
        )

        if liveness is not None:

            try:

                is_live = bool(
                    liveness.get(
                        "is_live",
                        False
                    )
                )

                live_score = float(
                    liveness.get(
                        "live_score",
                        0.0
                    )
                )

                logger.info(
                    "Liveness score: %.4f",
                    live_score
                )

                if (
                    not is_live
                    or
                    live_score < LIVE_THRESHOLD
                ):

                    return (
                        False,
                        FACE_NOT_RECOGNIZED_MESSAGE
                    )

            except Exception:

                logger.exception(
                    "Liveness validation failed"
                )

                return (
                    False,
                    FACE_NOT_RECOGNIZED_MESSAGE
                )

        # ====================================================
        # SUCCESS
        # ====================================================

        return (
            True,
            "Face quality acceptable"
        )

    # ========================================================
    # RECOGNIZE FACE
    # ========================================================

    def recognize(
        self,
        image_bytes: bytes
    ):

        # ====================================================
        # LOAD LATEST EMBEDDINGS
        # ====================================================

        self.reload_embeddings()

        if not self.embeddings:

            return {
                "success": False,
                "message": "No enrolled faces found"
            }

        # ====================================================
        # DECODE IMAGE
        # ====================================================

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE
            }

        # ====================================================
        # DETECT FACES
        # ====================================================

        try:

            faces = self.app.get(
                image
            )

        except Exception:

            logger.exception(
                "InsightFace failed during recognition"
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE
            }

        # ====================================================
        # NO FACE
        # ====================================================

        if not faces:

            logger.warning(
                "No face detected"
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE
            }

        # ====================================================
        # MULTIPLE FACES
        # ====================================================

        if len(faces) > 1:

            logger.warning(
                "Multiple faces detected"
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE
            }

        # ====================================================
        # SINGLE FACE
        # ====================================================

        face = faces[0]

        # ====================================================
        # QUALITY + OCCLUSION + LIVENESS
        # ====================================================

        quality_ok, quality_message = (
            self.validate_face_quality(
                face,
                image
            )
        )

        if not quality_ok:

            logger.warning(
                "Face rejected: %s",
                quality_message
            )

            return {
                "success": False,
                "message": quality_message
            }

        # ====================================================
        # EMBEDDING
        # ====================================================

        embedding = getattr(
            face,
            "embedding",
            None
        )

        if embedding is None:

            logger.warning(
                "Unable to generate face embedding"
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE
            }

        # ====================================================
        # NORMALIZE
        # ====================================================

        embedding = (
            embedding
            /
            (
                np.linalg.norm(
                    embedding
                )
                + 1e-10
            )
        )

        # ====================================================
        # MATCH
        # ====================================================

        (
            employee_id,
            score,
            is_ambiguous
        ) = self.find_best_match(
            embedding
        )

        # ====================================================
        # AMBIGUOUS
        # ====================================================

        if is_ambiguous:

            logger.warning(
                "Recognition rejected due to ambiguity"
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE,
                "confidence": round(
                    score,
                    4
                )
            }

        # ====================================================
        # UNKNOWN
        # ====================================================

        if (
            employee_id is None
            or
            score < THRESHOLD
        ):

            logger.warning(
                (
                    "Face not recognized: "
                    "employee=%s confidence=%.4f"
                ),
                employee_id,
                score
            )

            return {
                "success": False,
                "message": FACE_NOT_RECOGNIZED_MESSAGE,
                "confidence": round(
                    score,
                    4
                )
            }

        # ====================================================
        # SUCCESS
        # ====================================================

        logger.info(
            (
                "Face recognized: "
                "employee=%s confidence=%.4f"
            ),
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

    # ========================================================
    # MULTI-FACE DETECTION
    # ========================================================

    def detect_faces(
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

            return {
                "success": False,
                "message": "Invalid image",
                "faces": []
            }

        try:

            faces = self.app.get(
                image
            )

        except Exception:

            logger.exception(
                "InsightFace failed during multi-face detection"
            )

            return {
                "success": False,
                "message": "Face detection failed",
                "faces": []
            }

        if not faces:

            return {
                "success": True,
                "message": "No faces detected",
                "faces": []
            }

        logger.info(
            "Multi-face detection: %d face(s) detected",
            len(faces)
        )

        return {
            "success": True,
            "message": (
                f"{len(faces)} face(s) detected"
            ),
            "faces": faces
        }


# ============================================================
# SINGLE INSTANCE
# ============================================================

face_engine = FaceEngine()