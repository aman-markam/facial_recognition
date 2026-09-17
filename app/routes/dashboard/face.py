from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from sqlalchemy.orm import Session
from typing import Annotated

from app.database.database import get_db
from app.models.employee import Employee
from app.schemas.face import FaceEnrollmentResponse
from app.services.face_service import face_service


router = APIRouter(
    prefix="/employees",
    tags=["Dashboard - Face Enrollment"]
)


@router.post(
    "/{employee_id}/face",
    response_model=FaceEnrollmentResponse
)
async def enroll_employee_face(
    employee_id: int,
    images: Annotated[list[UploadFile], File(description="Upload 5 to 20 face images")],
    db: Session = Depends(get_db)
):

    employee = (
        db.query(Employee)
        .filter(
            Employee.id == employee_id
        )
        .first()
    )

    if not employee:
        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    if not employee.is_active:
        raise HTTPException(
            status_code=400,
            detail="Employee is inactive"
        )

    if len(images) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least 5 images"
        )

    if len(images) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 images allowed"
        )

    image_bytes_list = []

    for image in images:

        if not image.content_type:
            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="All uploaded files must be images"
            )

        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Empty image: {image.filename}"
            )

        image_bytes_list.append(
            image_bytes
        )

    try:

        embedding = (
            face_service.create_average_embedding(
                image_bytes_list
            )
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    face_service.save_embedding(
        employee.employee_code,
        embedding
    )

    return {
        "success": True,
        "employee_id": employee.id,
        "employee_code": employee.employee_code,
        "employee_name": employee.name,
        "message": (
            f"Face enrolled successfully "
            f"using {len(images)} images"
        )
    }