from fastapi import APIRouter, File, UploadFile

from app.services.multiframe_liveness_service import (
    multiframe_liveness_service,
)


router = APIRouter(
    prefix="/api/v1/appside/liveness",
    tags=["Appside Liveness"],
)


@router.post("/verify")
async def verify_liveness(
    images: list[UploadFile] = File(...)
):
    """
    Verify whether the submitted frames belong to a live person.

    Expected:
        Exactly 5 image files.

    This endpoint:
        - accepts image files
        - performs multi-frame liveness
        - does NOT perform face recognition
        - does NOT mark attendance
    """

    required_frames = (
        multiframe_liveness_service.required_frames
    )

    # ==========================================
    # Validate number of files
    # ==========================================

    if len(images) != required_frames:
        return {
            "success": False,
            "is_live": False,
            "total_frames": len(images),
            "live_frames": 0,
            "spoof_frames": len(images),
            "scores": [],
            "average_score": 0.0,
            "message": (
                f"Exactly {required_frames} image files "
                f"are required"
            ),
        }

    # ==========================================
    # Read uploaded files
    # ==========================================

    image_frames = []

    for index, image in enumerate(
        images,
        start=1,
    ):

        # --------------------------------------
        # Validate content type
        # --------------------------------------

        if not image.content_type:
            return {
                "success": False,
                "is_live": False,
                "message": (
                    f"Frame {index} has no content type"
                ),
            }

        if not image.content_type.startswith("image/"):
            return {
                "success": False,
                "is_live": False,
                "message": (
                    f"Frame {index} must be an image file"
                ),
            }

        # --------------------------------------
        # Read file bytes
        # --------------------------------------

        image_bytes = await image.read()

        if not image_bytes:
            return {
                "success": False,
                "is_live": False,
                "message": (
                    f"Frame {index} is empty"
                ),
            }

        image_frames.append(image_bytes)

    # ==========================================
    # Run multi-frame liveness
    # ==========================================

    result = (
        multiframe_liveness_service.check_frames(
            image_frames
        )
    )

    return result