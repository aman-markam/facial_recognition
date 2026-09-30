"""
API Router for YOLO Till Monitoring and Device Anti-Spoofing Analysis
"""

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.yolo_detector import yolo_till_detector

router = APIRouter(
    prefix="/api/v1/appside/till",
    tags=["Till YOLO Monitoring"],
)


@router.post("/analyze-frame")
async def analyze_till_frame(
    image: UploadFile = File(...)
):
    """
    Analyzes a single camera frame for till monitoring:
    - Detects number of persons present at the till
    - Detects electronic devices (cell phones, laptops, screens, tablets)
    - Detects potential spoofing devices shown to camera
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image format")

    analysis_result = yolo_till_detector.analyze_till_frame(frame)

    return {
        "success": True,
        "till_analysis": analysis_result
    }
