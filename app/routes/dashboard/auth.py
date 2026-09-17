from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    verify_password
)
from app.database.database import get_db
from app.models.admin import Admin
from app.schemas.auth import (
    AdminLogin,
    LoginResponse
)


router = APIRouter(
    prefix="/auth",
    tags=["Dashboard Authentication"]
)


@router.post(
    "/login",
    response_model=LoginResponse
)
def login(
    login_data: AdminLogin,
    db: Session = Depends(get_db)
):

    admin = (
        db.query(Admin)
        .filter(
            Admin.username == login_data.username
        )
        .first()
    )

    if not admin:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not verify_password(
        login_data.password,
        admin.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not admin.is_active:
        raise HTTPException(
            status_code=403,
            detail="Admin account is inactive"
        )

    token = create_access_token({
        "sub": str(admin.id),
        "username": admin.username
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "admin": admin
    }