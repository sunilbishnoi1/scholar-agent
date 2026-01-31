from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# Handle sqlite-specific connection args
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False, "timeout": 15}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Exports
__all__ = ["engine", "SessionLocal", "get_db"]