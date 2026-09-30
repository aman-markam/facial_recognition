import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from face_engine import FaceEngine

router = APIRouter(
    prefix="/api/v1/appside/face",
    tags=["Appside - Face Recognition"]
)

face_engine = FaceEngine()


@router.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...)
):
    contents = await image.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image"
        )

    result = face_engine.recognize(contents)

    if not result.get("success", False):
        return {
            "success": False,
            "message": result.get("message", "Recognition failed"),
            "confidence": result.get("confidence", 0.0)
        }

    return result