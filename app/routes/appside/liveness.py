import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from app.services.liveness_engine import LivenessEngine
from app.services.multiframe_liveness_service import multiframe_liveness_service

router = APIRouter(
    prefix="/api/v1/appside/liveness",
    tags=["Appside Liveness"],
)

liveness_engine = LivenessEngine()


@router.post("/verify-single")
async def verify_single_image_liveness(
    image: UploadFile = File(...)
):
    """
    Verify multi-face liveness on a single image frame using Phase 4.5 pipeline:
    - FaceAttributeEngine (Mask / Sunglasses)
    - HandDetectionEngine (MediaPipe Hands overlap)
    - InsightFace Liveness Addon
    - AntiSpoofEngine (MiniFASNet presentation attack detection)
    """
    image_bytes = await image.read()
    if not image_bytes:
        return {
            "success": False,
            "message": "Empty image file"
        }

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "success": False,
            "message": "Could not decode image"
        }

    response = liveness_engine.analyze_image(frame)
    return response.model_dump()


@router.post("/verify")
async def verify_liveness(
    images: list[UploadFile] = File(...)
):
    """
    Verify whether the submitted 5 frames belong to a live person.
    """
    required_frames = multiframe_liveness_service.required_frames

    if len(images) != required_frames:
        return {
            "success": False,
            "is_live": False,
            "total_frames": len(images),
            "live_frames": 0,
            "spoof_frames": len(images),
            "scores": [],
            "average_score": 0.0,
            "message": f"Exactly {required_frames} image files are required",
        }

    image_frames = []
    for index, image in enumerate(images, start=1):
        if not image.content_type or not image.content_type.startswith("image/"):
            return {
                "success": False,
                "is_live": False,
                "message": f"Frame {index} must be a valid image file",
            }

        image_bytes = await image.read()
        if not image_bytes:
            return {
                "success": False,
                "is_live": False,
                "message": f"Frame {index} is empty",
            }

        image_frames.append(image_bytes)

    result = multiframe_liveness_service.check_frames(image_frames)
    return result