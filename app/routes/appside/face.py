import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.face_service import face_service


router = APIRouter(
    prefix="/api/v1/appside/face",
    tags=["Appside - Face"]
)


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

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    result = face_service.recognize(frame)

    return result