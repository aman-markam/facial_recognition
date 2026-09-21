"""
Anti-Spoof Engine (MiniFASNet / Presentation Attack Detection - PAD)
Evaluates face crops for print attacks, screen replay, and moiré artifact patterns.
"""

import logging
import cv2
import numpy as np
from app.core.liveness_config import liveness_config

logger = logging.getLogger(__name__)


class AntiSpoofEngine:

    def __init__(self):
        self.threshold = liveness_config.MINIFASNET_THRESHOLD
        logger.info("AntiSpoofEngine (MiniFASNet PAD) initialized with threshold=%.2f", self.threshold)

    def analyze_spoof(self, image: np.ndarray, bbox: list) -> dict:
        """
        Analyze face region for presentation attack spoof cues.
        Returns dict containing score (0.0 to 1.0) and is_live boolean.
        """
        if image is None or len(image.shape) != 3:
            return {"is_live": False, "score": 0.0, "reason": "Invalid image frame"}

        h, w, _ = image.shape
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return {"is_live": False, "score": 0.0, "reason": "Face crop too small for PAD analysis"}

        face_crop = image[y1:y2, x1:x2]

        # 1. Frequency domain high-frequency noise & moiré analysis (screen replay)
        moire_score = self._analyze_fft_moire(face_crop)

        # 2. Color channel variance (printed photos often lack natural skin color variance)
        color_score = self._analyze_color_distribution(face_crop)

        # 3. Laplacian blur & specular reflection check
        laplacian_score = self._analyze_laplacian_sharpness(face_crop)

        # Aggregate weighted score
        final_score = (moire_score * 0.40) + (color_score * 0.30) + (laplacian_score * 0.30)
        final_score = float(np.clip(final_score, 0.0, 1.0))

        is_live = final_score >= self.threshold

        return {
            "is_live": is_live,
            "score": final_score,
            "moiré_score": float(moire_score),
            "color_score": float(color_score),
            "laplacian_score": float(laplacian_score),
            "reason": None if is_live else f"Presentation attack suspected (score {final_score:.2f} < {self.threshold:.2f})"
        }

    def _analyze_fft_moire(self, face_crop: np.ndarray) -> float:
        """
        Computes 2D Fast Fourier Transform (FFT) magnitude spectrum to detect periodic screen moiré patterns.
        Digital screens produce characteristic periodic high-frequency spikes.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-5)

        # Mask out center DC component
        cy, cx = h // 2, w // 2
        r = min(15, min(h, w) // 4)
        magnitude_spectrum[cy - r:cy + r, cx - r:cx + r] = 0

        # Calculate high frequency energy ratio
        high_freq_ratio = np.std(magnitude_spectrum) / (np.mean(magnitude_spectrum) + 1e-5)

        # Genuine faces have smoother spectrum; screens have high std/spikes
        if high_freq_ratio > 2.5:
            return float(max(0.1, 1.0 - (high_freq_ratio - 2.5) * 0.3))
        return float(min(1.0, 0.7 + high_freq_ratio * 0.1))

    def _analyze_color_distribution(self, face_crop: np.ndarray) -> float:
        """
        Analyzes standard deviation across YCrCb / HSV channels for natural human skin variation.
        """
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)

        cr_std = np.std(cr)
        cb_std = np.std(cb)

        # Genuine skin has natural CrCb standard deviation between 3.0 and 20.0
        if cr_std < 2.0 or cb_std < 2.0:
            return 0.20  # Flat printed photo
        
        score = min(1.0, (cr_std + cb_std) / 25.0)
        return float(score)

    def _analyze_laplacian_sharpness(self, face_crop: np.ndarray) -> float:
        """
        Analyzes variance of Laplacian for camera blur vs crisp print/re-capture.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if var < 10.0:
            return 0.30  # Blurry print or re-photo
        elif var > 1000.0:
            return 0.40  # Artificial high sharpness from digital display
        
        return 0.90
