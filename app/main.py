from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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