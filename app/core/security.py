from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# ==========================================
# PASSWORD HASH
# ==========================================

def hash_password(
    password: str
) -> str:

    return pwd_context.hash(
        password
    )


# ==========================================
# VERIFY PASSWORD
# ==========================================

def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# ==========================================
# CREATE ACCESS TOKEN
# ==========================================

def create_access_token(
    data: dict,
    expires_minutes: int | None = None
) -> str:

    to_encode = data.copy()


    expire = (
        datetime.now(
            timezone.utc
        )
        + timedelta(
            minutes=(
                expires_minutes
                or settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        )
    )


    to_encode.update({
        "exp": expire
    })


    token = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )


    return token


# ==========================================
# VERIFY ACCESS TOKEN
# ==========================================

def verify_access_token(
    token: str
):

    try:

        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[
                settings.ALGORITHM
            ]
        )


        user_id = payload.get(
            "sub"
        )


        if user_id is None:

            return None


        return payload


    except jwt.ExpiredSignatureError:

        return None


    except jwt.JWTError:

        return None