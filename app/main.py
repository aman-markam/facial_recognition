from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware
from app.core.error_handlers import (
    http_exception_handler,
    validation_exception_handler,
    integrity_error_handler,
    general_exception_handler,
)
from app.routes.appside.attendance import router as attendance_router
from app.routes.dashboard.auth import router as dashboard_auth_router

from app.routes.dashboard.employees import (
    router as dashboard_employee_router
)
from app.routes.dashboard.attendance import (
    router as dashboard_attendance_router
)
from app.routes.dashboard.reports import (
    router as dashboard_reports_router
)
from app.routes.dashboard.face import (
    router as dashboard_face_router
)
app = FastAPI(
    title="Face Attendance API",
    version="1.0.0"
)
app.add_exception_handler(
    HTTPException,
    http_exception_handler
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler
)

app.add_exception_handler(
    IntegrityError,
    integrity_error_handler
)

app.add_exception_handler(
    Exception,
    general_exception_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    dashboard_auth_router,
    prefix="/api/v1/dashboard"
)

app.include_router(
    attendance_router,
)

app.include_router(
    dashboard_employee_router,
    prefix="/api/v1/dashboard"
)
app.include_router(
    dashboard_attendance_router,
    prefix="/api/v1/dashboard"
)
app.include_router(
    dashboard_reports_router,
    prefix="/api/v1/dashboard"
)
app.include_router(
    dashboard_face_router,
    prefix="/api/v1/dashboard"
)
@app.get("/")
def root():

    return {
        "message": "Face Attendance API is running"
    }