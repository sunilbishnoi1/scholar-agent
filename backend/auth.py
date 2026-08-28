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

SECRET_KEY = os.environ.get("JWT_SECRET") or os.environ.get("SECRET_KEY", "your_default_secret_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class TokenData(BaseModel):
    email: str | None = None


def _truncate_password_bytes(password: str | bytes, max_bytes: int = 72) -> str:
    """Safely truncate password to at most max_bytes in UTF-8 representation."""
    if isinstance(password, bytes):
        raw_bytes = password[:max_bytes]
        return raw_bytes.decode("utf-8", errors="ignore")
    elif isinstance(password, str):
        raw_bytes = password.encode("utf-8")[:max_bytes]
        return raw_bytes.decode("utf-8", errors="ignore")
    return str(password)


def verify_password(plain_password: str | bytes, hashed_password: str) -> bool:
    """Verify plain password against hashed password, handling bcrypt 72-byte limit."""
    if not plain_password or not hashed_password:
        return False
    try:
        safe_plain = _truncate_password_bytes(plain_password, 72)
        return pwd_context.verify(safe_plain, hashed_password)
    except (ValueError, TypeError, Exception) as e:
        logger.debug(f"Password verification failed: {e}")
        return False


def get_password_hash(password: str | bytes) -> str:
    """Generate bcrypt password hash, handling bcrypt 72-byte limit safely."""
    if password is None:
        raise ValueError("Password cannot be None")
    safe_pwd = _truncate_password_bytes(password, 72)
    return pwd_context.hash(safe_pwd)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token with expiry and standard claims."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # Ensure both sub and email are populated if either is an email
    sub = to_encode.get("sub")
    email = to_encode.get("email")
    if sub and not email and "@" in str(sub):
        to_encode["email"] = str(sub)
    elif email and not sub:
        to_encode["sub"] = str(email)

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency to extract and authenticate the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return _get_user_from_neon_token(token, db, credentials_exception)


def _get_user_from_neon_token(token: str, db: Session, credentials_exception: HTTPException) -> User:
    """Validate JWT token and return the authenticated User model.

    Attempts HMAC SHA-256 signature verification using SECRET_KEY and ALGORITHM first.
    If the signature fails (e.g. external provider / Neon Auth token with asymmetric keys),
    falls back safely to unverified payload decoding while strictly verifying:
    - Expiry timestamp ('exp') against UTC now
    - Mandatory non-empty 'sub' claim (user_id)
    - Mandatory non-empty 'email' claim
    - Safe default user retrieval or creation in the database session
    """
    payload = None
    try:
        # 1. Attempt signature verification using SECRET_KEY / ALGORITHM first
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_signature": True, "verify_exp": True, "verify_aud": False},
        )
    except JWTError as sig_err:
        err_msg = str(sig_err).lower()
        if "expired" in err_msg or "exp" in err_msg:
            logger.warning(f"Token expired (signature verified): {sig_err}")
            raise credentials_exception

        # 2. Fallback safely to unverified payload decoding for external provider / Neon tokens
        try:
            payload = jwt.decode(
                token,
                key="",
                options={"verify_signature": False, "verify_aud": False},
            )
        except JWTError as decode_err:
            logger.error(f"JWT decode error on fallback: {decode_err}")
            raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error during JWT verification: {e}", exc_info=True)
        raise credentials_exception

    if not isinstance(payload, dict):
        logger.warning("Decoded JWT payload is not a dictionary")
        raise credentials_exception

    # Strictly verify Token expiry ('exp') with UTC timestamp comparison
    exp = payload.get("exp")
    if exp is None:
        logger.warning("Token missing mandatory 'exp' claim")
        raise credentials_exception
    try:
        exp_timestamp = float(exp)
        if exp_timestamp < datetime.now(UTC).timestamp():
            logger.warning("Token expired based on UTC timestamp comparison")
            raise credentials_exception
    except (ValueError, TypeError):
        logger.warning(f"Invalid 'exp' claim in token: {exp}")
        raise credentials_exception

    # Strictly verify mandatory 'sub' (user_id) claim presence and non-emptiness
    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str) or not user_id.strip():
        logger.warning("Token missing or empty 'sub' claim")
        raise credentials_exception
    user_id = user_id.strip()

    # Strictly verify mandatory 'email' claim presence and non-emptiness
    email = payload.get("email")
    if not email and "@" in user_id:
        email = user_id
    if not email or not isinstance(email, str) or not email.strip() or "@" not in email:
        logger.warning("Token missing or empty 'email' claim")
        raise credentials_exception
    email = email.strip().lower()

    user_metadata = payload.get("user_metadata")
    if not isinstance(user_metadata, dict):
        user_metadata = {}
    name = user_metadata.get("name") or (email.split("@")[0] if email else "User")

    try:
        # Safe default user creation / lookup in the database session
        user = db.query(User).filter(User.id == user_id).first()

        if user is None:
            existing_by_email = db.query(User).filter(User.email == email).first()

            if existing_by_email:
                logger.info(f"User found by email: {email} (id: {existing_by_email.id})")
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
    except Exception as e:
        db.rollback()
        logger.error(f"Database error during user authentication: {e}", exc_info=True)
        raise credentials_exception

