from pathlib import Path
import json

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
from app.dependencies.auth import get_current_admin


router = APIRouter(
    prefix="/employees",
    tags=["Dashboard - Face Enrollment"],
    dependencies=[Depends(get_current_admin)]
)


# ==========================================
# EMBEDDINGS FILE
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

EMBEDDINGS_PATH = (
    ROOT / "face_data" / "embeddings.json"
)


# ==========================================
# CHECK EXISTING ENROLLMENT
# ==========================================

def face_already_enrolled(
    employee_code: str
) -> bool:

    if not EMBEDDINGS_PATH.exists():
        return False

    try:

        with open(
            EMBEDDINGS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print(
            "Checking enrollment for:",
            employee_code
        )

        print(
            "Existing employees:",
            list(data.keys())
        )

        return employee_code in data

    except Exception as e:

        print(
            "Enrollment check error:",
            e
        )

        return False


# ==========================================
# ENROLL FACE
# ==========================================

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

    # --------------------------------------
    # FIND EMPLOYEE
    # --------------------------------------

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


    # --------------------------------------
    # ACTIVE CHECK
    # --------------------------------------

    if not employee.is_active:

        raise HTTPException(
            status_code=400,
            detail="Employee is inactive"
        )


    # --------------------------------------
    # DUPLICATE ENROLLMENT CHECK
    # --------------------------------------

    if face_already_enrolled(
        employee.employee_code
    ):
        print(
            "================================"
        )

        print(
            "EMPLOYEE ID:",
            employee.id
        )

        print(
            "EMPLOYEE CODE:",
            employee.employee_code
        )

        print(
            "EMBEDDINGS PATH:",
            EMBEDDINGS_PATH
        )

        print(
            "FILE EXISTS:",
            EMBEDDINGS_PATH.exists()
        )

        print(
            "ALREADY ENROLLED:",
            face_already_enrolled(
                employee.employee_code
            )
        )

        print(
            "================================"
)
        print(
            f"Duplicate enrollment blocked: "
            f"{employee.employee_code}"
        )

        raise HTTPException(
            status_code=409,
            detail=(
                f"Face is already enrolled "
                f"for employee "
                f"{employee.employee_code}."
            )
        )


    # --------------------------------------
    # IMAGE COUNT
    # --------------------------------------

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


    # --------------------------------------
    # READ IMAGES
    # --------------------------------------

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


    # --------------------------------------
    # CREATE EMBEDDING
    # --------------------------------------

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


    # --------------------------------------
    # SAVE EMBEDDING
    # --------------------------------------

    try:

        face_service.save_embedding(
            employee.employee_code,
            embedding
        )

    except ValueError as e:

        raise HTTPException(
            status_code=409,
            detail=str(e)
        )

    except Exception as e:

        print(
            "Failed to save embedding:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail=str(e) or (
                "Failed to save face enrollment"
            )
        )


        # --------------------------------------
        # RELOAD ACTIVE VECTOR MEMORY CACHE
        # --------------------------------------
        try:
            from face_engine import FaceEngine
            from app.routes.appside.attendance import multi_face_tracker, face_engine
            face_engine.reload_embeddings()
            if hasattr(multi_face_tracker, "multiface_engine"):
                multi_face_tracker.multiface_engine.reload_embeddings()
        except Exception as exc:
            print("Notice: Error auto-reloading memory cache:", exc)

    # --------------------------------------
    # RESPONSE
    # --------------------------------------

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


# ==========================================
# RELOAD EMBEDDINGS ENDPOINT
# ==========================================

face_direct_router = APIRouter(
    prefix="/face",
    tags=["Dashboard - Face Enrollment"],
    dependencies=[Depends(get_current_admin)]
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
    Reload face enrollment vectors from face_data/embeddings.json into active memory cache.
    """
    from face_engine import FaceEngine
    from app.routes.appside.attendance import multi_face_tracker, face_engine

    try:
        face_engine.reload_embeddings()
        count = len(face_engine.embeddings)
        if hasattr(multi_face_tracker, "multiface_engine"):
            multi_face_tracker.multiface_engine.reload_embeddings()
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