from fastapi import (
    Depends,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.orm import Session

from app.core.security import (
    verify_access_token
)
from app.database.database import get_db
from app.models.admin import Admin


security = HTTPBearer()


# ==========================================
# GET CURRENT ADMIN
# ==========================================

def get_current_admin(
    credentials:
        HTTPAuthorizationCredentials =
        Depends(security),

    db: Session =
        Depends(get_db)
):

    token = credentials.credentials


    payload = verify_access_token(
        token
    )


    if not payload:

        raise HTTPException(
            status_code=
                status.HTTP_401_UNAUTHORIZED,

            detail="Invalid or expired token",

            headers={
                "WWW-Authenticate":
                    "Bearer"
            }
        )


    admin_id = payload.get(
        "sub"
    )


    if not admin_id:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )


    try:

        admin_id = int(
            admin_id
        )

    except (
        ValueError,
        TypeError
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )


    admin = (
        db.query(Admin)
        .filter(
            Admin.id == admin_id
        )
        .first()
    )


    if not admin:

        raise HTTPException(
            status_code=401,
            detail="Admin not found"
        )


    if not admin.is_active:

        raise HTTPException(
            status_code=403,
            detail="Admin account is inactive"
        )


    return admin