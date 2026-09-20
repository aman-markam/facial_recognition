import logging
from typing import List

from app.services.liveness_service import liveness_service


logger = logging.getLogger(__name__)


# ==============================
# Multi-frame configuration
# ==============================

REQUIRED_FRAMES = 5

# At least 4 out of 5 frames must pass liveness
MIN_LIVE_FRAMES = 4

# Same threshold used by the single-frame liveness service
LIVE_THRESHOLD = 0.80


class MultiFrameLivenessService:
    """
    Performs liveness verification across multiple frames.

    Example:

        5 frames captured
              ↓
        Liveness check
              ↓
        4 or more LIVE
              ↓
             LIVE

        3 or fewer LIVE
              ↓
            SPOOF
    """

    def __init__(
        self,
        required_frames: int = REQUIRED_FRAMES,
        min_live_frames: int = MIN_LIVE_FRAMES,
        live_threshold: float = LIVE_THRESHOLD,
    ):
        self.required_frames = required_frames
        self.min_live_frames = min_live_frames
        self.live_threshold = live_threshold

        if self.min_live_frames > self.required_frames:
            raise ValueError(
                "min_live_frames cannot be greater than required_frames"
            )

        logger.info(
            "Multi-frame liveness service initialized: "
            "required_frames=%d, min_live_frames=%d, threshold=%.2f",
            self.required_frames,
            self.min_live_frames,
            self.live_threshold,
        )

    def check_frames(self, image_frames: List[bytes]) -> dict:
        """
        Check multiple image frames for liveness.

        Args:
            image_frames:
                List of image bytes.

        Returns:
            Dictionary containing:
                success
                is_live
                total_frames
                live_frames
                spoof_frames
                scores
                average_score
                message
        """

        # --------------------------------
        # Validate number of frames
        # --------------------------------

        if not image_frames:
            logger.warning(
                "Multi-frame liveness failed: no frames received"
            )

            return {
                "success": False,
                "is_live": False,
                "total_frames": 0,
                "live_frames": 0,
                "spoof_frames": 0,
                "scores": [],
                "average_score": 0.0,
                "message": "No frames received",
            }

        if len(image_frames) != self.required_frames:
            logger.warning(
                "Multi-frame liveness failed: expected %d frames, received %d",
                self.required_frames,
                len(image_frames),
            )

            return {
                "success": False,
                "is_live": False,
                "total_frames": len(image_frames),
                "live_frames": 0,
                "spoof_frames": len(image_frames),
                "scores": [],
                "average_score": 0.0,
                "message": (
                    f"Exactly {self.required_frames} frames are required"
                ),
            }

        # --------------------------------
        # Process each frame
        # --------------------------------

        scores = []
        live_frames = 0
        spoof_frames = 0

        for index, image_bytes in enumerate(image_frames, start=1):

            if not image_bytes:
                logger.warning(
                    "Frame %d is empty",
                    index,
                )

                scores.append(0.0)
                spoof_frames += 1
                continue

            try:
                result = liveness_service.check_liveness(image_bytes)

            except Exception:
                logger.exception(
                    "Unexpected error during liveness check for frame %d",
                    index,
                )

                scores.append(0.0)
                spoof_frames += 1
                continue

            live_score = float(
                result.get("live_score", 0.0) or 0.0
            )

            is_live = bool(result.get("is_live", False))

            scores.append(round(live_score, 4))

            # --------------------------------
            # Frame decision
            # --------------------------------

            if is_live and live_score >= self.live_threshold:
                live_frames += 1

                logger.info(
                    "Frame %d: LIVE | score=%.4f",
                    index,
                    live_score,
                )

            else:
                spoof_frames += 1

                logger.info(
                    "Frame %d: SPOOF | score=%.4f",
                    index,
                    live_score,
                )

        # --------------------------------
        # Calculate average score
        # --------------------------------

        if scores:
            average_score = sum(scores) / len(scores)
        else:
            average_score = 0.0

        average_score = round(average_score, 4)

        # --------------------------------
        # Final decision
        # --------------------------------

        is_live = live_frames >= self.min_live_frames

        if is_live:
            message = "Live person detected"

            logger.info(
                "MULTI-FRAME LIVENESS PASSED | "
                "live_frames=%d/%d | average_score=%.4f",
                live_frames,
                self.required_frames,
                average_score,
            )

        else:
            message = "Liveness verification failed. Possible spoof detected."

            logger.warning(
                "MULTI-FRAME LIVENESS FAILED | "
                "live_frames=%d/%d | average_score=%.4f",
                live_frames,
                self.required_frames,
                average_score,
            )

        return {
            "success": is_live,
            "is_live": is_live,
            "total_frames": self.required_frames,
            "live_frames": live_frames,
            "spoof_frames": spoof_frames,
            "scores": scores,
            "average_score": average_score,
            "message": message,
        }


# ==========================================
# Singleton instance
# ==========================================

multiframe_liveness_service = MultiFrameLivenessService()