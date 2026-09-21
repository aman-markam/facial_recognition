"""
Pydantic Schemas for Phase 4.5 Liveness & Anti-Spoofing
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class FaceAttributeResult(BaseModel):
    is_occluded: bool = False
    mask_prob: float = 0.0
    sunglasses_prob: float = 0.0
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    reason: Optional[str] = None


class HandOverlapResult(BaseModel):
    has_hand_overlap: bool = False
    overlap_ratio: float = 0.0
    hands_detected: int = 0


class FaceLivenessResult(BaseModel):
    face_index: int
    bbox: List[float] = Field(default_factory=list)
    is_live: bool = False
    live_score: float = 0.0
    anti_spoof_score: float = 0.0
    is_occluded: bool = False
    occlusion_reason: Optional[str] = None
    status: str = "ok"
    employee_code: Optional[str] = None
    employee_name: Optional[str] = None
    recognition_confidence: float = 0.0


class MultiFaceLivenessResponse(BaseModel):
    success: bool = True
    total_faces: int = 0
    live_faces: int = 0
    spoof_faces: int = 0
    faces: List[FaceLivenessResult] = Field(default_factory=list)
    message: str = "Processed faces successfully"


class MultiFrameLivenessResponse(BaseModel):
    success: bool = True
    is_live: bool = False
    total_frames: int = 5
    live_frames: int = 0
    spoof_frames: int = 0
    scores: List[float] = Field(default_factory=list)
    average_score: float = 0.0
    tracks: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = "Multi-frame evaluation complete"
