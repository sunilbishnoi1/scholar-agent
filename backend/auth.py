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

# --- Configuration ---
# Generate a secret key with: openssl rand -hex 32
SECRET_KEY = os.environ.get("SECRET_KEY", "your_default_secret_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# --- Pydantic Schemas ---
class TokenData(BaseModel):
    email: str | None = None

# --- Utility Functions ---
def verify_password(plain_password, hashed_password):
    """Verify a password against a stored hash.

    Bcrypt (the underlying C library) has a 72-byte limit on passwords. If the stored
    hash is a bcrypt hash (starts with $2), truncate the provided plaintext to 72 bytes
    (UTF-8) before verifying to avoid an exception coming from the bcrypt backend.
    """
    try:
        if isinstance(hashed_password, str) and hashed_password.startswith("$2"):
            # Truncate to bcrypt's 72-byte limit when needed
            if isinstance(plain_password, str):
                b = plain_password.encode("utf-8")
                if len(b) > 72:
                    # Truncate bytes and decode safely
                    plain_password = b[:72].decode("utf-8", errors="ignore")
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        # In case an unexpected ValueError still bubbles up from backend, return False
        return False

def get_password_hash(password):
    """Hash a password for storage.

    If bcrypt is the configured scheme, truncate the input to 72 bytes to avoid
    ValueError from the underlying bcrypt library for very long inputs.
    """
    try:
        if isinstance(password, str):
            b = password.encode("utf-8")
            if len(b) > 72:
                password = b[:72].decode("utf-8", errors="ignore")
        return pwd_context.hash(password)
    except ValueError:
        # Surface predictable behavior: raise a clear error for callers
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

# --- Dependency for getting current user ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user
