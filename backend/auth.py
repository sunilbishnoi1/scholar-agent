import logging
import os
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models.database import User

logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY", "your_default_secret_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class TokenData(BaseModel):
    email: str | None = None


def verify_password(plain_password, hashed_password):
    try:
        if isinstance(hashed_password, str) and hashed_password.startswith("$2"):
            if isinstance(plain_password, str):
                b = plain_password.encode("utf-8")
                if len(b) > 72:
                    plain_password = b[:72].decode("utf-8", errors="ignore")
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        return False


def get_password_hash(password):
    try:
        if isinstance(password, str):
            b = password.encode("utf-8")
            if len(b) > 72:
                password = b[:72].decode("utf-8", errors="ignore")
        return pwd_context.hash(password)
    except ValueError:
        raise ValueError("Password is too long after encoding; truncate to 72 bytes before hashing")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return _get_user_from_neon_token(token, db, credentials_exception)


def _get_user_from_neon_token(token: str, db: Session, credentials_exception: HTTPException):
    """Validate Neon Auth JWT

    Decode JWT (without signature verification for now - TODO: add JWKS validation)
    """
    try:
        payload = jwt.decode(
            token,
            key="",
            options={"verify_signature": False, "verify_aud": False},
        )

        exp = payload.get("exp")
        if exp and exp < datetime.now(UTC).timestamp():
            logger.warning("Token expired")
            raise credentials_exception

        user_id = payload.get("sub")
        email = payload.get("email")
        user_metadata = payload.get("user_metadata", {})
        name = user_metadata.get("name") or (email.split("@")[0] if email else "User")

        if not user_id:
            logger.warning("Token missing 'sub' claim")
            raise credentials_exception

        if not email:
            logger.warning("Token missing 'email' claim")
            raise credentials_exception

        user = db.query(User).filter(User.id == user_id).first()

        if user is None:
            existing_by_email = db.query(User).filter(User.email == email).first()

            if existing_by_email:
                logger.info(
                    f"User found by email {email}, updating id from "
                    f"{existing_by_email.id} to {user_id}"
                )
                existing_by_email.id = user_id
                existing_by_email.name = name
                existing_by_email.hashed_password = "neon_auth_managed"
                db.commit()
                db.refresh(existing_by_email)
                user = existing_by_email
            else:
                logger.info(f"Creating new user: {email} (id: {user_id})")
                user = User(
                    id=user_id,
                    email=email,
                    name=name,
                    hashed_password="neon_auth_managed",
                    tier="free",
                    monthly_budget_usd=1.0,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"User created successfully: {user.id}")

        return user

    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}", exc_info=True)
        raise credentials_exception
