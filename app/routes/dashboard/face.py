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
from app.services.embedding_store import is_enrolled
from app.services.face_service import face_service


router = APIRouter(
    prefix="/employees",
    tags=["Dashboard - Face Enrollment"]
)


def _reload_recognition_caches() -> None:
    from app.routes.appside.attendance import face_engine, multi_face_tracker
    from app.routes.appside import face as appside_face

    face_engine.reload_embeddings()
    appside_face.face_engine.reload_embeddings()
    if hasattr(multi_face_tracker, "engine"):
        multi_face_tracker.engine.reload_embeddings()


@router.post(
    "/{employee_id}/face",
    response_model=FaceEnrollmentResponse
)
async def enroll_employee_face(

    employee_id: str,

    images: Annotated[
        list[UploadFile],
        File(
            description="Upload 5 to 20 face images"
        )
    ],

    db: Session = Depends(get_db)

):

    ident_str = str(employee_id)
    employee = (
        db.query(Employee)
        .filter(Employee.employee_code == ident_str)
        .first()
    )
    if not employee and ident_str.isdigit():
        employee = (
            db.query(Employee)
            .filter(Employee.id == int(ident_str))
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

    if is_enrolled(employee.employee_code, db):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Face is already enrolled "
                f"for employee "
                f"{employee.employee_code}."
            )
        )

    if len(images) < 5:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload at least 5 images"
            )
        )

    if len(images) > 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum 20 images allowed"
            )
        )

    image_bytes_list = []

    for image in images:

        if not image.content_type:

            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

        if not image.content_type.startswith(
            "image/"
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "All uploaded files "
                    "must be images"
                )
            )

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Empty image: "
                    f"{image.filename}"
                )
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

    try:

        face_service.save_embedding(
            employee.employee_code,
            embedding,
            db,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=409,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e) or (
                "Failed to save face enrollment"
            )
        )

    try:
        _reload_recognition_caches()
    except Exception:
        pass

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


face_direct_router = APIRouter(
    prefix="/face",
    tags=["Dashboard - Face Enrollment"]
)


@router.post(
    "/reload-embeddings",
    tags=["Dashboard - Face Enrollment"]
)
@face_direct_router.post(
    "/reload-embeddings",
    tags=["Dashboard - Face Enrollment"]
)
async def reload_face_embeddings():
    """
    Reload face enrollment vectors from PostgreSQL into the in-memory cache.
    """
    from app.routes.appside.attendance import face_engine

    try:
        _reload_recognition_caches()
        count = len(face_engine.embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload face embeddings: {str(e)}"
        )

    return {
        "success": True,
        "message": "Successfully reloaded face embeddings into active cache.",
        "employee_count": count
    }
