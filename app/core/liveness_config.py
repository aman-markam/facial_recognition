"""
Configuration settings for Phase 4.5 Multi-person Liveness & Anti-Spoofing Architecture
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LivenessConfig(BaseSettings):
    # InsightFace liveness threshold (0.0 to 1.0)
    LIVENESS_THRESHOLD: float = 0.80

    # MiniFASNet anti-spoofing threshold (0.0 to 1.0)
    MINIFASNET_THRESHOLD: float = 0.70

    # Multi-frame aggregation rules (out of TOTAL_FRAMES)
    MIN_LIVE_FRAMES: int = 3
    TOTAL_FRAMES: int = 5

    # Occlusion thresholds
    MASK_THRESHOLD: float = 0.70
    SUNGLASSES_THRESHOLD: float = 0.75
    HAND_FACE_OVERLAP_THRESHOLD: float = 0.15

    # Face recognition matching threshold
    FACE_RECOGNITION_THRESHOLD: float = 0.55

    # Maximum allowed faces in a frame
    MAX_FACES: int = 10

    # Detection & Distance settings
    DETECTOR_SIZE: tuple[int, int] = (960, 960)
    MIN_FACE_SIZE: int = 40

    # Pose angle thresholds (degrees)
    MAX_YAW: float = 60.0    # 3/4 & moderate profile supported
    MAX_PITCH: float = 45.0
    MAX_ROLL: float = 45.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


liveness_config = LivenessConfig()
