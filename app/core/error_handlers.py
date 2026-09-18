from fastapi import (
    Request,
    HTTPException,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError


async def http_exception_handler(
    request: Request,
    exc: HTTPException
):
    """
    Handles FastAPI HTTPException errors.
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "error": "HTTP_ERROR",
            "details": None,
        },
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Handles request validation errors.
    """

    errors = []

    for error in exc.errors():

        location = ".".join(
            str(item)
            for item in error.get(
                "loc",
                []
            )
        )

        errors.append({
            "field": location,
            "message": error.get(
                "msg",
                "Invalid value"
            ),
            "type": error.get(
                "type",
                "validation_error"
            ),
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Request validation failed",
            "error": "VALIDATION_ERROR",
            "details": errors,
        },
    )


async def integrity_error_handler(
    request: Request,
    exc: IntegrityError
):
    """
    Handles database integrity errors.

    Raw SQL/database information is intentionally
    not returned to the client.
    """

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "success": False,
            "message": (
                "Database constraint violation"
            ),
            "error": "DATABASE_CONSTRAINT_ERROR",
            "details": None,
        },
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
):
    """
    Handles unexpected server errors.

    Do not expose internal exception details
    to the frontend.
    """

    print(
        "Unhandled server error:",
        repr(exc)
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": (
                "An unexpected server error occurred"
            ),
            "error": "INTERNAL_SERVER_ERROR",
            "details": None,
        },
    )