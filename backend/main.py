import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import markdown  # type: ignore
from celery import Celery
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sib_api_v3_sdk.rest import ApiException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, joinedload, scoped_session, selectinload, sessionmaker

load_dotenv()

import auth
from agents.blackboard import WorkingMemoryBlackboard
from agents.llm import get_llm_client
from agents.orchestrator import ResearchOrchestrator, ScholarAgentOrchestrator
from agents.planner_agent import ResearchPlannerAgent
from agents.schemas import ResearchReport
from db import engine, get_db
from models.database import (
    AgentPlan,
    Base,
    EvidenceMatrixEntry,
    LLMInteraction,
    PaperCache,
    PaperReference,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
    User,
)
from paper_retriever import PaperRetriever
from realtime.events import (
    AgentEvent,
    AgentProgressTracker,
    EventType,
    create_pipeline_completed_event,
    create_pipeline_error_event,
    create_pipeline_stopped_event,
    sync_broadcast_agent_update,
)
from services.cancellation_manager import TaskCancelledException, cancellation_manager
from services.usage_tracker import UsageTracker

try:
    from rag.service import RAGService
except ImportError:
    RAGService = None


def create_db_and_tables():
    """Initialize database tables and run schema migrations."""
    logging.info("Initializing database tables and verifying schema...")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created/verified successfully.")

        _run_schema_migrations()

        _verify_schema()

        logging.info("Database initialization and schema verification completed successfully.")

    except Exception as e:
        logging.error(f"Error during database initialization: {e}", exc_info=True)
        raise


async def start_redis_listener():
    """Start Redis pub/sub listener for WebSocket broadcasts."""
    try:
        from realtime.manager import get_connection_manager

        manager = get_connection_manager()
        await manager.start_redis_listener()
    except Exception as e:
        logging.error(f"Failed to start Redis listener: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for FastAPI."""
    try:
        from realtime.manager import get_connection_manager

        loop = asyncio.get_running_loop()
        get_connection_manager().set_event_loop(loop)
    except Exception as e:
        logging.warning(f"Could not register main event loop with ConnectionManager: {e}")
    create_db_and_tables()
    await start_redis_listener()
    yield
    try:
        from realtime.manager import get_connection_manager

        manager = get_connection_manager()
        await manager.stop_redis_listener()
    except Exception:
        pass


def _is_postgresql() -> bool:
    """Check if we're connected to PostgreSQL."""
    return "postgresql" in str(engine.url)


def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    if _is_postgresql():
        result = conn.execute(
            text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = :table_name 
                AND (table_schema = current_schema() OR table_schema = 'public')
            )
        """),
            {"table_name": table_name},
        )
        return bool(result.scalar())
    else:
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
            {"table_name": table_name},
        )
        return result.fetchone() is not None


def _get_existing_columns(conn, table_name: str) -> set[str]:
    """Get existing columns (lowercase) for a table, handling both PostgreSQL and SQLite."""
    if _is_postgresql():
        result = conn.execute(
            text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE (table_schema = current_schema() OR table_schema = 'public')
            AND table_name = :table_name
        """),
            {"table_name": table_name},
        )
        return {str(row[0]).lower() for row in result.fetchall()}
    else:
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        return {str(row[1]).lower() for row in result.fetchall()}


def _add_column_if_not_exists(
    conn, table_name: str, column_name: str, column_def: str, existing_columns: set[str]
) -> bool:
    """Add a column if it doesn't exist. Returns True if column was added or already exists."""
    if column_name.lower() in existing_columns:
        return True

    try:
        if _is_postgresql():
            sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_def}"
        else:
            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"

        conn.execute(text(sql))
        logging.info(f"Successfully added column '{column_name}' to '{table_name}'.")
        existing_columns.add(column_name.lower())
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate column" in error_str:
            logging.info(f"Column '{column_name}' already exists in '{table_name}'.")
            existing_columns.add(column_name.lower())
            return True
        logging.error(f"Failed to add column '{column_name}' to '{table_name}': {e}")
        return False


# Complete schema column specifications: table_name -> list of (column_name, postgres_col_def, sqlite_col_def)
TABLE_SCHEMA_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "users": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("email", "VARCHAR(255)", "VARCHAR(255)"),
        ("name", "VARCHAR(255)", "VARCHAR(255)"),
        ("hashed_password", "VARCHAR(255)", "VARCHAR(255)"),
        ("institution", "VARCHAR(255) NULL", "VARCHAR(255)"),
        ("tier", "VARCHAR(50) DEFAULT 'free'", "VARCHAR(50) DEFAULT 'free'"),
        ("monthly_budget_usd", "DOUBLE PRECISION DEFAULT 1.0", "REAL DEFAULT 1.0"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "research_projects": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("user_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("title", "VARCHAR(1024)", "VARCHAR(1024)"),
        ("research_question", "TEXT", "TEXT"),
        ("keywords", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("subtopics", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("status", "VARCHAR(50) DEFAULT 'planning'", "VARCHAR(50) DEFAULT 'planning'"),
        ("total_papers_found", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("max_papers", "INTEGER DEFAULT 30", "INTEGER DEFAULT 30"),
        ("report", "JSONB NULL", "JSON NULL"),
        ("report_status", "VARCHAR(50) DEFAULT 'empty'", "VARCHAR(50) DEFAULT 'empty'"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "agent_plans": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("agent_type", "VARCHAR(100) NULL", "VARCHAR(100)"),
        ("plan_steps", "JSONB NULL", "JSON NULL"),
        ("current_step", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("plan_metadata", "JSONB NULL", "JSON NULL"),
    ],
    "paper_references": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("title", "VARCHAR(1024) NULL", "VARCHAR(1024)"),
        ("authors", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("abstract", "TEXT NULL", "TEXT"),
        ("url", "VARCHAR(2048) NULL", "VARCHAR(2048)"),
        ("embeddings", "JSONB NULL", "JSON NULL"),
        ("relevance_score", "DOUBLE PRECISION NULL", "REAL"),
    ],
    "user_usage": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("user_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("month", "DATE DEFAULT CURRENT_DATE", "DATE"),
        ("total_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("prompt_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("completion_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("total_cost_usd", "DOUBLE PRECISION DEFAULT 0.0", "REAL DEFAULT 0.0"),
        ("projects_created", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("papers_analyzed", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("llm_calls", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "llm_interactions": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("user_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64) NULL", "VARCHAR(64)"),
        ("agent_type", "VARCHAR(100) NULL", "VARCHAR(100)"),
        ("model", "VARCHAR(100) NULL", "VARCHAR(100)"),
        ("task_type", "VARCHAR(100) NULL", "VARCHAR(100)"),
        ("prompt_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("completion_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("total_tokens", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("cost_usd", "DOUBLE PRECISION DEFAULT 0.0", "REAL DEFAULT 0.0"),
        ("latency_ms", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0"),
        ("prompt_preview", "TEXT NULL", "TEXT"),
        ("response_preview", "TEXT NULL", "TEXT"),
        ("success", "BOOLEAN DEFAULT TRUE", "BOOLEAN DEFAULT 1"),
        ("error_message", "TEXT NULL", "TEXT"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "paper_cache": [
        ("doi", "VARCHAR(256)", "VARCHAR(256)"),
        ("arxiv_id", "VARCHAR(64) NULL", "VARCHAR(64)"),
        ("s2_id", "VARCHAR(64) NULL", "VARCHAR(64)"),
        ("title", "VARCHAR(1024) DEFAULT ''", "VARCHAR(1024) DEFAULT ''"),
        ("authors", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("year", "INTEGER NULL", "INTEGER"),
        ("venue", "VARCHAR(512) NULL", "VARCHAR(512)"),
        ("abstract", "TEXT NULL", "TEXT"),
        ("parsed_markdown", "TEXT NULL", "TEXT"),
        ("sections_json", "JSONB NULL", "JSON NULL"),
        ("tables_json", "JSONB NULL", "JSON NULL"),
        ("source_url", "VARCHAR(2048) NULL", "VARCHAR(2048)"),
        ("is_full_text", "BOOLEAN DEFAULT FALSE", "BOOLEAN DEFAULT 0"),
        ("fetched_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "research_reports": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("title", "VARCHAR(1024) DEFAULT ''", "VARCHAR(1024) DEFAULT ''"),
        ("executive_summary", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
        ("methodology_overview", "JSONB NULL", "JSON NULL"),
        ("quality_score", "DOUBLE PRECISION DEFAULT 0.0", "REAL DEFAULT 0.0"),
        ("thematic_sections", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("conflicts_and_debates", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("generated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "evidence_matrix_entries": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("paper_id", "VARCHAR(256) DEFAULT ''", "VARCHAR(256) DEFAULT ''"),
        ("title", "VARCHAR(1024) DEFAULT ''", "VARCHAR(1024) DEFAULT ''"),
        ("methodology_type", "TEXT NULL", "TEXT"),
        ("benchmark_dataset", "TEXT NULL", "TEXT"),
        ("primary_metric", "VARCHAR(512) NULL", "VARCHAR(512)"),
        ("primary_limitation", "TEXT NULL", "TEXT"),
        ("authors", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("year", "INTEGER NULL", "INTEGER"),
        ("doi", "VARCHAR(256) NULL", "VARCHAR(256)"),
        ("url", "VARCHAR(2048) NULL", "VARCHAR(2048)"),
        ("is_full_text", "BOOLEAN DEFAULT FALSE", "BOOLEAN DEFAULT 0"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
    "research_gaps": [
        ("id", "VARCHAR(64)", "VARCHAR(64)"),
        ("project_id", "VARCHAR(64)", "VARCHAR(64)"),
        ("gap_id", "VARCHAR(64) DEFAULT ''", "VARCHAR(64) DEFAULT ''"),
        ("description", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
        ("importance", "VARCHAR(32) DEFAULT 'high'", "VARCHAR(32) DEFAULT 'high'"),
        ("recommended_methodology", "TEXT DEFAULT ''", "TEXT DEFAULT ''"),
        ("grounding_paper_ids", "JSONB DEFAULT '[]'::jsonb", "JSON DEFAULT '[]'"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ],
}


def _run_schema_migrations():
    """Universal schema migration: automatically inspect and add missing columns to all tables."""
    is_pg = _is_postgresql()
    db_name = "PostgreSQL" if is_pg else "SQLite"
    logging.info(f"Running universal schema migrations... (Database: {db_name})")

    try:
        with engine.begin() as conn:
            for table_name, col_specs in TABLE_SCHEMA_SPECS.items():
                if not _table_exists(conn, table_name):
                    logging.info(f"Table '{table_name}' does not exist yet (skipping migration).")
                    continue

                existing_columns = _get_existing_columns(conn, table_name)
                logging.info(f"Table '{table_name}' columns in DB: {existing_columns}")

                added_count = 0
                for col_name, pg_def, sqlite_def in col_specs:
                    col_def = pg_def if is_pg else sqlite_def
                    if col_name.lower() not in existing_columns:
                        if _add_column_if_not_exists(
                            conn, table_name, col_name, col_def, existing_columns
                        ):
                            added_count += 1

                if added_count > 0:
                    logging.info(f"Migrated {added_count} missing column(s) into table '{table_name}'.")

            logging.info("Universal schema migrations completed successfully.")

    except Exception as e:
        logging.error(f"Schema migration failed: {e}", exc_info=True)
        raise


def _verify_schema():
    """Verify that all required columns exist across all defined tables after migrations."""
    try:
        with engine.connect() as conn:
            missing_by_table: dict[str, set[str]] = {}

            for table_name, col_specs in TABLE_SCHEMA_SPECS.items():
                if not _table_exists(conn, table_name):
                    continue

                existing_columns = _get_existing_columns(conn, table_name)
                required_columns = {col_name.lower() for col_name, _, _ in col_specs}
                missing = required_columns - existing_columns

                if missing:
                    missing_by_table[table_name] = missing

            if missing_by_table:
                logging.error(f"SCHEMA VERIFICATION FAILED! Missing columns detected: {missing_by_table}")
                raise RuntimeError(
                    f"Database schema verification failed. Missing columns: {missing_by_table}"
                )

            logging.info("Schema verification passed. All tables and columns are fully synchronized.")

    except Exception as e:
        logging.error(f"Schema verification error: {e}", exc_info=True)
        raise


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = FastAPI(lifespan=lifespan)
celery_app = Celery("literature_agent", broker=REDIS_URL)


def _get_cors_origins() -> list[str]:
    environment = os.environ.get("ENVIRONMENT", "development")
    allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")

    base_origins = [
        "http://localhost:8000",
        "http://localhost:5174",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5173",
    ]

    if environment == "production":
        production_origins = [
            "https://scholar-agent.vercel.app",
            "https://scholaragent.dpdns.org",
        ]
        base_origins.extend(production_origins)

    if allowed_origins_env:
        env_origins = [origin.strip() for origin in allowed_origins_env.split(",")]
        base_origins.extend(env_origins)

    return base_origins


origins = _get_cors_origins()
logging.info(f"CORS origins configured: {origins}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)

    origin = request.headers.get("origin")
    headers = {"Access-Control-Allow-Credentials": "true"}

    if origin and (origin in origins or "*" in origins):
        headers["Access-Control-Allow-Origin"] = origin
    elif origins:
        headers["Access-Control-Allow-Origin"] = origins[0]

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_message": str(exc)},
        headers=headers,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def root():
    """
    Root endpoint for API discovery and health checks.
    Render and other platforms may probe this endpoint.
    """
    return {
        "name": "Scholar Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }


class PaperReferenceSchema(BaseModel):
    id: str
    title: str
    authors: list[str] | None = []
    abstract: str | None = None
    url: str | None = None
    relevance_score: float | None = None

    class Config:
        from_attributes = True


class AgentPlanSchema(BaseModel):
    id: str
    agent_type: str
    plan_steps: list
    current_step: int
    plan_metadata: dict

    class Config:
        from_attributes = True


class ResearchProjectSchema(BaseModel):
    id: str
    user_id: str
    title: str
    research_question: str
    keywords: list[str]
    subtopics: list[str]
    status: str
    total_papers_found: int
    max_papers: int | None = 30
    report: dict | None = None
    report_status: str | None = "empty"
    created_at: datetime
    agent_plans: list[AgentPlanSchema] = []
    paper_references: list[PaperReferenceSchema] = []

    class Config:
        from_attributes = True


class ProjectCreate(BaseModel):
    title: str
    research_question: str
    max_papers: int | None = 30


# --- New Authentication Schemas ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str


class UserOut(BaseModel):
    id: str
    email: EmailStr
    name: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


# --- Authentication Router ---
auth_router = APIRouter()


@auth_router.post("/register", response_model=UserOut)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    new_user = User(email=user.email, name=user.name, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@auth_router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/users/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(auth.get_current_user)):
    return current_user


# --- Projects Router ---
projects_router = APIRouter()


@projects_router.get("/projects", response_model=list[ResearchProjectSchema])
def get_projects(
    db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)
):
    projects = (
        db.query(ResearchProject)
        .options(
            selectinload(ResearchProject.agent_plans),
            selectinload(ResearchProject.paper_references),
        )
        .filter(ResearchProject.user_id == current_user.id)
        .all()
    )
    return projects


@projects_router.get("/projects/{project_id}", response_model=ResearchProjectSchema)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    project = (
        db.query(ResearchProject)
        .options(
            selectinload(ResearchProject.agent_plans),
            selectinload(ResearchProject.paper_references),
        )
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@projects_router.post("/projects", response_model=ResearchProjectSchema)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    llm_client = get_llm_client()
    planner = ResearchPlannerAgent(llm_client)

    initial_plan = planner.generate_initial_plan(project.research_question, project.title)
    generated_keywords = initial_plan.get("keywords", [])
    generated_subtopics = initial_plan.get("subtopics", [])

    new_project = ResearchProject(
        user_id=current_user.id,
        title=project.title,
        research_question=project.research_question,
        keywords=generated_keywords,
        subtopics=generated_subtopics,
        status="created",
        max_papers=project.max_papers if project.max_papers is not None else 30,
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project


@projects_router.post("/projects/{project_id}/start")
def start_literature_review(
    project_id: str,
    max_papers: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Clear any previous cancellation state
    cancellation_manager.clear_cancellation(project_id)

    # Determine effective max_papers
    effective_max_papers = (
        max_papers if (max_papers is not None and max_papers > 0) else (project.max_papers or 30)
    )

    project.status = "searching"
    db.commit()

    use_celery = os.environ.get("ENABLE_CELERY", "true").lower() not in ("false", "0", "no", "off")
    dispatched_to_celery = False
    if use_celery:
        try:
            active_workers = celery_app.control.ping(timeout=0.3)
            if active_workers:
                job = celery_app.send_task("run_literature_review", args=[project_id, effective_max_papers])
                cancellation_manager.register_task(project_id, job.id)
                dispatched_to_celery = True
                return {
                    "job_id": job.id,
                    "status": "queued",
                    "estimated_duration": f"PT{effective_max_papers // 2}M",
                }
            else:
                logging.info(
                    "No active Celery worker responded to ping. Running pipeline in local background thread."
                )
        except Exception as e:
            logging.warning(
                f"Celery dispatch unavailable ({e}), running pipeline in local background thread."
            )

    if not dispatched_to_celery:
        import threading

        thread = threading.Thread(
            target=_run_literature_review_internal,
            args=[project_id, effective_max_papers],
            daemon=True,
        )
        thread.start()
        return {
            "job_id": f"thread-{uuid.uuid4()}",
            "status": "started",
            "estimated_duration": f"PT{effective_max_papers // 2}M",
        }


class StopProjectResponse(BaseModel):
    project_id: str
    status: str
    message: str


@projects_router.post("/projects/{project_id}/stop", response_model=StopProjectResponse)
def stop_literature_review(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """
    Immediately stop a running literature review task for a project.
    Cancels in-flight background worker/thread, revokes Celery job, and sets status to 'stopped'.
    """
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Signal cancellation across threads, Redis, and Celery
    cancellation_manager.cancel_project(project_id)
    cancellation_manager.revoke_task(project_id, celery_app=celery_app)

    # Mark database status as stopped
    project.status = "stopped"
    db.commit()

    # Broadcast WebSocket update
    try:
        stop_event = create_pipeline_stopped_event(
            project_id=project_id,
            message="Research task was stopped by user.",
        )
        sync_broadcast_agent_update(project_id, stop_event)
    except Exception as e:
        logging.warning(f"Failed to broadcast pipeline_stopped event: {e}")

    return StopProjectResponse(
        project_id=project_id,
        status="stopped",
        message="Research task stopped successfully.",
    )


class DeleteProjectResponse(BaseModel):
    id: str
    deleted: bool
    message: str


class ReportResponse(BaseModel):
    project_id: str
    report: dict | None = None
    report_status: str
    message: str | None = None

    class Config:
        from_attributes = True


@projects_router.delete("/projects/{project_id}", response_model=DeleteProjectResponse)
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """
    Delete a project and all its associated data.
    This includes: agent plans, paper references, and RAG data (if available).
    """
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_title = project.title

    if RAGService is not None:
        try:
            rag_service = RAGService()
            rag_service.delete_project_data(project_id)
            logging.info(f"Deleted RAG data for project {project_id}")
        except Exception as e:
            logging.warning(f"Failed to delete RAG data for project {project_id}: {e}")

    db.query(ResearchReportModel).filter(ResearchReportModel.project_id == project_id).delete()
    db.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == project_id).delete()
    db.query(ResearchGapModel).filter(ResearchGapModel.project_id == project_id).delete()
    db.query(LLMInteraction).filter(LLMInteraction.project_id == project_id).delete()
    db.query(AgentPlan).filter(AgentPlan.project_id == project_id).delete()
    db.query(PaperReference).filter(PaperReference.project_id == project_id).delete()
    db.delete(project)
    db.commit()

    logging.info(f"User {current_user.id} deleted project {project_id} ('{project_title}')")

    return DeleteProjectResponse(
        id=project_id,
        deleted=True,
        message=f"Project '{project_title}' and all associated data deleted successfully",
    )


class MatrixResponse(BaseModel):
    project_id: str
    count: int
    total: int
    entries: list[dict[str, Any]]
    matrix: list[dict[str, Any]]


class GapsResponse(BaseModel):
    project_id: str
    count: int
    total: int
    gaps: list[dict[str, Any]]


class PaperSectionsResponse(BaseModel):
    paper_id: str
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    is_full_text: bool = False
    sections: list[dict[str, Any]] = []
    tables: list[Any] = []
    source_url: str | None = None


@projects_router.get("/projects/{project_id}/report", response_model=ReportResponse)
def get_project_report(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Get the structured research report for a project."""
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    report_model = (
        db.query(ResearchReportModel)
        .filter(ResearchReportModel.project_id == project_id)
        .order_by(ResearchReportModel.generated_at.desc())
        .first()
    )

    report_data = None
    report_status = project.report_status or "empty"

    # Index project paper references for author and URL backfilling
    ref_by_id = {p.id: p for p in (project.paper_references or [])}
    ref_by_title = {p.title.lower().strip(): p for p in (project.paper_references or []) if p.title}

    if report_model:
        matrix_entries = (
            db.query(EvidenceMatrixEntry)
            .filter(EvidenceMatrixEntry.project_id == project_id)
            .all()
        )
        gap_entries = (
            db.query(ResearchGapModel)
            .filter(ResearchGapModel.project_id == project_id)
            .all()
        )

        matrix_rows = []
        for row in matrix_entries:
            ref = ref_by_id.get(row.paper_id) or ref_by_title.get(row.title.lower().strip())
            r_authors = getattr(row, "authors", None) or (ref.authors if ref else [])
            r_year = getattr(row, "year", None) or (getattr(ref, "year", None) if ref else None)
            r_doi = getattr(row, "doi", None) or (getattr(ref, "doi", None) if ref else None)
            r_url = getattr(row, "url", None) or (getattr(ref, "url", None) if ref else None)
            r_ft = getattr(row, "is_full_text", None)
            if r_ft is None:
                r_ft = getattr(ref, "is_full_text", True) if ref else True

            matrix_rows.append({
                "id": str(row.id),
                "paper_id": row.paper_id,
                "title": row.title,
                "authors": r_authors if isinstance(r_authors, list) else [],
                "year": r_year,
                "doi": r_doi,
                "url": r_url,
                "methodology": row.methodology_type or "Theoretical/Empirical",
                "methodology_type": row.methodology_type or "Theoretical/Empirical",
                "dataset": row.benchmark_dataset or "General Domain Benchmark",
                "benchmark_dataset": row.benchmark_dataset or "General Domain Benchmark",
                "primary_metric": row.primary_metric or "Accuracy/Precision",
                "limitations": [row.primary_limitation] if row.primary_limitation else [],
                "primary_limitation": row.primary_limitation or "",
                "key_findings": [],
                "confidence_score": 0.85,
                "has_full_text": bool(r_ft),
                "is_full_text": bool(r_ft),
            })

        gaps = [
            {
                "id": str(gap.id),
                "gap_id": gap.gap_id,
                "title": "",
                "description": gap.description,
                "importance": gap.importance,
                "priority": gap.importance.lower() if gap.importance else "high",
                "recommended_methodology": gap.recommended_methodology,
                "actionable_recommendations": [gap.recommended_methodology] if gap.recommended_methodology else [],
                "grounding_paper_ids": gap.grounding_paper_ids or [],
                "grounding_papers": gap.grounding_paper_ids or [],
            }
            for gap in gap_entries
        ]

        # Build accurate bibliography with authors and origin URLs
        bib_list = []
        if project.paper_references:
            for p in project.paper_references:
                auths = p.authors or []
                bib_list.append({
                    "paper_id": p.id,
                    "title": p.title or "Untitled",
                    "authors": auths if isinstance(auths, list) else [],
                    "year": getattr(p, "year", None),
                    "venue": getattr(p, "venue", None),
                    "doi": getattr(p, "doi", None),
                    "url": getattr(p, "url", None),
                    "pdf_url": getattr(p, "url", None),
                    "citation_count": getattr(p, "citation_count", 0),
                    "is_full_text_analyzed": getattr(p, "is_full_text", True),
                    "bibtex": f"@article{{{p.id},\n  title={{{p.title}}},\n  author={{{' and '.join(auths) if auths else 'Unknown Authors'}}}\n}}",
                })
        else:
            for m in matrix_rows:
                auths = m.get("authors") or []
                bib_list.append({
                    "paper_id": m.get("paper_id", ""),
                    "title": m.get("title", ""),
                    "authors": auths,
                    "year": m.get("year"),
                    "venue": None,
                    "doi": m.get("doi"),
                    "url": m.get("url"),
                    "pdf_url": m.get("url"),
                    "citation_count": 0,
                    "is_full_text_analyzed": m.get("is_full_text", True),
                    "bibtex": f"@article{{{m.get('paper_id')},\n  title={{{m.get('title')}}},\n  author={{{' and '.join(auths) if auths else 'Unknown Authors'}}}\n}}",
                })

        report_data = {
            "metadata": {
                "title": report_model.title,
                "research_question": project.research_question,
                "generated_at": report_model.generated_at.isoformat() if report_model.generated_at else datetime.utcnow().isoformat(),
                "target_academic_level": "graduate",
                "quality_score": report_model.quality_score,
                "total_papers_analyzed": len(matrix_rows) or project.total_papers_found,
                "synthesis_version": "3.2",
            },
            "title": report_model.title,
            "executive_summary": report_model.executive_summary,
            "methodology_overview": report_model.methodology_overview or {"distribution": {}, "dominant_approach": "", "trend_description": ""},
            "thematic_sections": report_model.thematic_sections or [],
            "sections": report_model.thematic_sections or [],
            "comparative_matrix": matrix_rows,
            "comparison_matrix": matrix_rows,
            "conflicting_debates": report_model.conflicts_and_debates or [],
            "conflicting_findings_and_debates": report_model.conflicts_and_debates or [],
            "debates": report_model.conflicts_and_debates or [],
            "actionable_gaps": gaps,
            "actionable_research_gaps": gaps,
            "research_gaps": gaps,
            "bibliography": bib_list,
            "quality_score": report_model.quality_score,
        }
        report_status = "complete"
    elif project.report:
        report_data = project.report

    return ReportResponse(
        project_id=project_id,
        report=report_data,
        report_status=report_status,
        message="Report retrieved successfully" if report_data else "Report is not yet available",
    )


@projects_router.get("/projects/{project_id}/matrix", response_model=MatrixResponse)
def get_project_matrix(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Get the comparative evidence matrix for a project."""
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    matrix_entries = (
        db.query(EvidenceMatrixEntry)
        .filter(EvidenceMatrixEntry.project_id == project_id)
        .order_by(EvidenceMatrixEntry.created_at.asc())
        .all()
    )

    ref_by_id = {p.id: p for p in (project.paper_references or [])}
    ref_by_title = {p.title.lower().strip(): p for p in (project.paper_references or []) if p.title}

    formatted_entries = []
    if matrix_entries:
        for entry in matrix_entries:
            ref = ref_by_id.get(entry.paper_id) or ref_by_title.get(entry.title.lower().strip())
            r_authors = getattr(entry, "authors", None) or (ref.authors if ref else [])
            r_year = getattr(entry, "year", None) or (getattr(ref, "year", None) if ref else None)
            r_doi = getattr(entry, "doi", None) or (getattr(ref, "doi", None) if ref else None)
            r_url = getattr(entry, "url", None) or (getattr(ref, "url", None) if ref else None)
            r_ft = getattr(entry, "is_full_text", None)
            if r_ft is None:
                r_ft = getattr(ref, "is_full_text", True) if ref else True

            formatted_entries.append({
                "id": str(entry.id),
                "paper_id": entry.paper_id,
                "title": entry.title,
                "authors": r_authors if isinstance(r_authors, list) else [],
                "year": r_year,
                "doi": r_doi,
                "url": r_url,
                "methodology": entry.methodology_type or "",
                "methodology_type": entry.methodology_type or "",
                "dataset": entry.benchmark_dataset or "",
                "benchmark_dataset": entry.benchmark_dataset or "",
                "primary_metric": entry.primary_metric or "",
                "limitations": [entry.primary_limitation] if entry.primary_limitation else [],
                "primary_limitation": entry.primary_limitation or "",
                "key_findings": [],
                "performance_metrics": {},
                "confidence_score": 0.85,
                "has_full_text": bool(r_ft),
                "is_full_text": bool(r_ft),
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            })
    elif project.report and isinstance(project.report, dict) and "comparative_matrix" in project.report:
        raw_matrix = project.report.get("comparative_matrix") or []
        for item in raw_matrix:
            if isinstance(item, dict):
                formatted_entries.append({
                    "id": item.get("id", str(uuid.uuid4())),
                    "paper_id": item.get("paper_id", ""),
                    "title": item.get("title", ""),
                    "authors": item.get("authors", []),
                    "year": item.get("year"),
                    "methodology": item.get("methodology", item.get("methodology_type", "")),
                    "methodology_type": item.get("methodology_type", item.get("methodology", "")),
                    "dataset": item.get("dataset", item.get("benchmark_dataset", "")),
                    "benchmark_dataset": item.get("benchmark_dataset", item.get("dataset", "")),
                    "primary_metric": item.get("primary_metric", ""),
                    "limitations": item.get("limitations", [item.get("primary_limitation")] if item.get("primary_limitation") else []),
                    "primary_limitation": item.get("primary_limitation", ""),
                    "key_findings": item.get("key_findings", []),
                    "performance_metrics": item.get("performance_metrics", {}),
                    "confidence_score": float(item.get("confidence_score", 0.85)),
                    "has_full_text": bool(item.get("has_full_text", True)),
                    "created_at": None,
                })

    return MatrixResponse(
        project_id=project_id,
        count=len(formatted_entries),
        total=len(formatted_entries),
        entries=formatted_entries,
        matrix=formatted_entries,
    )


@projects_router.get("/projects/{project_id}/gaps", response_model=GapsResponse)
def get_project_gaps(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Get the actionable research gaps for a project."""
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    gap_entries = (
        db.query(ResearchGapModel)
        .filter(ResearchGapModel.project_id == project_id)
        .order_by(ResearchGapModel.created_at.asc())
        .all()
    )

    formatted_gaps = []
    if gap_entries:
        for gap in gap_entries:
            formatted_gaps.append({
                "id": str(gap.id),
                "gap_id": gap.gap_id,
                "title": (gap.description[:80] + "...") if len(gap.description) > 80 else gap.description,
                "description": gap.description,
                "importance": gap.importance,
                "priority": gap.importance.lower() if gap.importance else "high",
                "recommended_methodology": gap.recommended_methodology,
                "actionable_recommendations": [gap.recommended_methodology] if gap.recommended_methodology else [],
                "grounding_paper_ids": gap.grounding_paper_ids or [],
                "grounding_papers": gap.grounding_paper_ids or [],
                "created_at": gap.created_at.isoformat() if gap.created_at else None,
            })
    elif project.report and isinstance(project.report, dict) and ("actionable_gaps" in project.report or "research_gaps" in project.report):
        raw_gaps = project.report.get("actionable_gaps") or project.report.get("research_gaps") or []
        for item in raw_gaps:
            if isinstance(item, dict):
                desc = item.get("description", item.get("title", ""))
                rec = item.get("recommended_methodology", "")
                recs = item.get("actionable_recommendations", [rec] if rec else [])
                formatted_gaps.append({
                    "id": item.get("id", str(uuid.uuid4())),
                    "gap_id": item.get("gap_id", item.get("id", str(uuid.uuid4()))),
                    "title": item.get("title", (desc[:80] + "...") if len(desc) > 80 else desc),
                    "description": desc,
                    "importance": item.get("importance", item.get("priority", "high")),
                    "priority": item.get("priority", item.get("importance", "high")).lower(),
                    "recommended_methodology": rec,
                    "actionable_recommendations": recs,
                    "grounding_paper_ids": item.get("grounding_paper_ids", item.get("grounding_papers", [])),
                    "grounding_papers": item.get("grounding_papers", item.get("grounding_paper_ids", [])),
                    "created_at": None,
                })

    return GapsResponse(
        project_id=project_id,
        count=len(formatted_gaps),
        total=len(formatted_gaps),
        gaps=formatted_gaps,
    )


# --- Papers Router ---
papers_router = APIRouter()


@papers_router.get("/papers/{paper_id}/sections", response_model=PaperSectionsResponse)
def get_paper_sections(
    paper_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Get parsed hierarchical sections, tables, and metadata for a paper from PaperCache."""
    # Search in PaperCache by DOI, arxiv_id, s2_id, or ID prefix
    cache_entry = (
        db.query(PaperCache)
        .filter(
            (PaperCache.doi == paper_id)
            | (PaperCache.arxiv_id == paper_id)
            | (PaperCache.s2_id == paper_id)
            | (PaperCache.doi == f"id:{paper_id}")
            | (PaperCache.doi.contains(paper_id))
        )
        .first()
    )

    if cache_entry:
        return PaperSectionsResponse(
            paper_id=paper_id,
            doi=cache_entry.doi,
            arxiv_id=cache_entry.arxiv_id,
            s2_id=cache_entry.s2_id,
            title=cache_entry.title,
            authors=cache_entry.authors or [],
            year=cache_entry.year,
            venue=cache_entry.venue,
            abstract=cache_entry.abstract,
            is_full_text=cache_entry.is_full_text,
            sections=cache_entry.sections_json or [],
            tables=cache_entry.tables_json or [],
            source_url=cache_entry.source_url,
        )

    # Fallback to PaperReference
    paper_ref = (
        db.query(PaperReference)
        .filter((PaperReference.id == paper_id) | (PaperReference.title.ilike(f"%{paper_id}%")))
        .first()
    )
    if paper_ref:
        return PaperSectionsResponse(
            paper_id=paper_ref.id,
            doi=None,
            arxiv_id=None,
            s2_id=None,
            title=paper_ref.title,
            authors=paper_ref.authors or [],
            year=None,
            venue=None,
            abstract=paper_ref.abstract,
            is_full_text=False,
            sections=[
                {
                    "heading": "Abstract",
                    "content": paper_ref.abstract or "No full text sections available for this reference.",
                    "section_index": 0,
                }
            ],
            tables=[],
            source_url=paper_ref.url,
        )

    raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found in cache")


# --- App Integration ---
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(projects_router, prefix="/api", tags=["Projects"])
app.include_router(papers_router, prefix="/api", tags=["Papers"])

# --- Users Router for Usage/Budget ---
users_router = APIRouter()


class UsageSummaryResponse(BaseModel):
    user_id: str
    tier: str
    month: str
    budget: dict
    tokens: dict
    activity: dict
    limits: dict


class BudgetCheckResponse(BaseModel):
    allowed: bool
    remaining_budget: float
    current_usage: float
    limit: float
    usage_percent: float
    warning: str | None = None
    error: str | None = None


@users_router.get("/users/me/usage", response_model=UsageSummaryResponse)
def get_user_usage(
    db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)
):
    """Get usage summary for the current user."""
    tracker = UsageTracker(db)
    summary = tracker.get_usage_summary(current_user)
    return summary


@users_router.get("/users/me/budget-check", response_model=BudgetCheckResponse)
def check_user_budget(
    estimated_cost: float = 0.0,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Check if user has remaining budget."""
    tracker = UsageTracker(db)
    result = tracker.check_budget(current_user, estimated_cost)
    return result


app.include_router(users_router, prefix="/api", tags=["Users"])

# --- Search Router ---
search_router = APIRouter()


class SearchRequest(BaseModel):
    text: str
    top_k: int = 10
    use_hybrid: bool = True


class SearchResultItem(BaseModel):
    chunk_id: str
    content: str
    paper_id: str | None = None
    paper_title: str | None = None
    chunk_type: str | None = None
    final_score: float


@search_router.post("/projects/{project_id}/search", response_model=list[SearchResultItem])
def semantic_search(
    project_id: str,
    request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Perform semantic search within a project's papers."""
    project = (
        db.query(ResearchProject)
        .filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if RAGService is not None:
        rag_service = RAGService()
        results = rag_service.search(
            query=request.text,
            project_id=project_id,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
        )
        return results
    else:
        return []


app.include_router(search_router, prefix="/api", tags=["Search"])


from fastapi import WebSocket, WebSocketDisconnect

from realtime.manager import get_connection_manager


@app.websocket("/ws/projects/{project_id}/stream")
async def websocket_project_stream(
    websocket: WebSocket,
    project_id: str,
    token: str | None = None,
):
    """
    WebSocket endpoint for real-time project updates.

    Streams agent progress, status changes, and completion events.
    Replaces polling for better UX and reduced server load.
    """
    manager = get_connection_manager()
    user_id = "anonymous"

    if token:
        try:
            from jose import jwt
            payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
            user_id = payload.get("sub", "anonymous")
        except Exception:
            user_id = "anonymous"

    connection_established = False

    try:
        success = await manager.connect(websocket, user_id, project_id)
        if not success:
            logger.warning(f"Failed to establish WebSocket connection for project {project_id}")
            try:
                await websocket.close(code=1008, reason="Connection failed")
            except Exception:
                pass
            return

        connection_established = True
        while True:
            try:
                data = await websocket.receive_text()

                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "project_id": project_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected for project {project_id}")
                break
            except RuntimeError as e:
                logger.warning(f"WebSocket runtime error for project {project_id}: {e}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for project {project_id}: {e}", exc_info=True)
                break

    except Exception as e:
        logger.error(f"WebSocket setup error for project {project_id}: {e}", exc_info=True)

    finally:
        if connection_established:
            try:
                await manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"Error during WebSocket cleanup: {e}")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ----Helper function for sending email--
from sib_api_v3_sdk import SendSmtpEmail
from sib_api_v3_sdk.api.transactional_emails_api import TransactionalEmailsApi
from sib_api_v3_sdk.api_client import ApiClient
from sib_api_v3_sdk.configuration import Configuration


def send_completion_email(
    user_email: str, user_name: str, project_title: str, synthesis_output: str
) -> bool:
    """
    Sends the final synthesized report to the user via the Brevo (Sendinblue) Transactional Email API
    using the official Python SDK (sib_api_v3_sdk).
    """
    api_key = os.environ.get("BREVO_API_KEY", "").strip().strip('"').strip("'")
    if not api_key or api_key in ["your_actual_api_key_here", "your-brevo-api-key", "your-brevo-api-key-here"]:
        logging.info("BREVO_API_KEY not set or placeholder used. Skipping email notification.")
        return False

    sender_email = os.environ.get("BREVO_SENDER_EMAIL", "sunilbishnoi7205@gmail.com").strip().strip('"').strip("'")
    sender_name = os.environ.get("BREVO_SENDER_NAME", "Scholar AI Agent").strip()

    configuration = Configuration()
    configuration.api_key["api-key"] = api_key
    api_client = ApiClient(configuration)
    email_api = TransactionalEmailsApi(api_client)

    formatted_output = markdown.markdown(synthesis_output)
    html_content = f"""
    <html>
    <head></head>
    <body style="font-family: sans-serif; line-height: 1.6;">
        <h2>Hello {user_name},</h2>
        <p>Your research project, <strong>{project_title}</strong>, has been successfully completed.</p>
        <p>Please find the synthesized literature review below.</p>
        <hr>
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
            {formatted_output}
        </div>
        <hr>
        <p>You can also view the full results, including the list of analyzed papers, by visiting the project page in your dashboard.</p>
        <p>Best regards,<br>The Scholar AI Agent Team</p>
    </body>
    </html>
    """

    subject = f"Research Complete: {project_title}"

    try:
        send_smtp_email = SendSmtpEmail(
            to=[{"email": user_email, "name": user_name}],
            sender={"name": sender_name, "email": sender_email},
            subject=subject,
            html_content=html_content,
        )

        response = email_api.send_transac_email(send_smtp_email)
        logging.info(f"Brevo email sent successfully to {user_email}. MessageId: {getattr(response, 'message_id', response)}")
        return True
    except ApiException as e:
        if getattr(e, "status", None) == 401:
            logging.error(
                f"Brevo API 401 Unauthorized when sending email: {e.body if hasattr(e, 'body') else e}. "
                "Ensure your Brevo account and transactional email platform are activated, "
                "and your API key is generated under SMTP & API in the Brevo dashboard."
            )
        elif getattr(e, "status", None) == 400:
            logging.error(
                f"Brevo API 400 Bad Request when sending email: {e.body if hasattr(e, 'body') else e}. "
                f"Ensure the sender email '{sender_email}' is a verified sender in your Brevo account settings."
            )
        else:
            logging.error(f"Brevo API exception when sending email: {e}")
        return False
    except Exception as e:
        logging.error(f"Failed to send email via Brevo: {e}")
        return False


def _run_literature_review_internal(project_id: str, max_papers: int):
    """Execute the full literature review pipeline synchronously/in worker thread."""
    task_engine = create_engine(
        os.environ.get("DATABASE_URL", "sqlite:///./test.db"),
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args=(
            {"check_same_thread": False, "timeout": 15}
            if "sqlite" in os.environ.get("DATABASE_URL", "sqlite:///./test.db")
            else {}
        ),
    )
    TaskSession = scoped_session(sessionmaker(bind=task_engine))
    db = TaskSession()

    try:
        project = (
            db.query(ResearchProject)
            .options(joinedload(ResearchProject.user))
            .filter(ResearchProject.id == project_id)
            .first()
        )
        if not project:
            return {"status": "error", "error": "Project not found"}

        # Extract necessary metadata
        user_id = project.user_id
        project_title = project.title
        research_question = project.research_question
        user_email = project.user.email if project.user else None
        user_name: str = str(project.user.name) if project.user and project.user.name else "User"

        # Update initial running status if not already stopped/cancelled
        if cancellation_manager.is_cancelled(project_id):
            project.status = "stopped"
            db.commit()
            logging.info(f"Project {project_id} cancelled before execution started.")
            return {"status": "stopped", "message": "Task cancelled by user."}

        project.status = "running"
        db.commit()

    except Exception as e:
        logging.error(f"Failed to read project {project_id}: {e}")
        try:
            db.close()
        except:
            pass
        TaskSession.remove()
        return {"status": "error", "error": str(e)}

    llm_client = get_llm_client()
    tracker = AgentProgressTracker(project_id)
    orchestrator = ScholarAgentOrchestrator(
        llm_client=llm_client,
        db_session=db,
        progress_callback=tracker.progress_callback_adapter,
    )

    try:
        if cancellation_manager.is_cancelled(project_id):
            logging.info(f"Project {project_id} cancelled before orchestrator run.")
            project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
            if project:
                project.status = "stopped"
                db.commit()
            return {"status": "stopped", "message": "Task cancelled by user."}

        final_state = orchestrator.run_sync(
            project_id=project_id,
            user_id=user_id,
            title=project_title,
            research_question=research_question,
            max_papers=max_papers,
            sync_to_db=True,
        )

        # Check if project was cancelled/stopped while running
        if cancellation_manager.is_cancelled(project_id) or final_state.get("status") == "stopped":
            logging.info(f"Project {project_id} stopped during multi-agent execution.")
            project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
            if project:
                project.status = "stopped"
                db.commit()
            return {"status": "stopped", "message": "Task stopped by user."}

        project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        if not project:
            logging.error(f"Project {project_id} disappeared during processing")
            return {"status": "error", "error": "Project disappeared"}

        # Determine terminal status
        raw_status = final_state.get("status", "completed")
        if raw_status in ["completed", "running", "auditing", "needs_refinement", None]:
            project.status = "completed"
        else:
            project.status = raw_status

        project.total_papers_found = final_state.get(
            "total_papers_found", len(final_state.get("candidate_papers", []))
        )

        # Persist PaperReference items if not already written
        analyzed_papers = final_state.get("analyzed_papers", []) or final_state.get("candidate_papers", [])
        if analyzed_papers:
            for paper in analyzed_papers:
                if hasattr(paper, "model_dump"):
                    paper_data = paper.model_dump()
                elif hasattr(paper, "__dict__"):
                    paper_data = paper.__dict__ if isinstance(paper, object) else paper
                else:
                    paper_data = paper

                title = paper_data.get("title", "Untitled")
                existing_ref = (
                    db.query(PaperReference)
                    .filter(PaperReference.project_id == project_id, PaperReference.title == title)
                    .first()
                )
                if not existing_ref:
                    raw_score = paper_data.get("relevance_score")
                    score_val = float(raw_score) if raw_score is not None else 0.0
                    paper_ref = PaperReference(
                        project_id=project_id,
                        title=title,
                        authors=paper_data.get("authors", []),
                        abstract=paper_data.get("abstract"),
                        url=paper_data.get("url") or paper_data.get("source_url"),
                        relevance_score=score_val,
                    )
                    db.add(paper_ref)

            logging.info(
                f"Saved {len(analyzed_papers)} paper references to database for project {project_id}"
            )

        if final_state.get("analyzer_output"):
            analyzer_output = final_state["analyzer_output"]
            analyses_summary = []
            if hasattr(analyzer_output, "paper_analyses"):
                analyses = analyzer_output.paper_analyses
            else:
                analyses = analyzer_output.get("paper_analyses", [])

            for analysis in analyses:
                if hasattr(analysis, "model_dump"):
                    analyses_summary.append(analysis.model_dump())
                else:
                    analyses_summary.append(
                        analysis if isinstance(analysis, dict) else vars(analysis)
                    )

            analyzer_plan = AgentPlan(
                id=str(uuid.uuid4()),
                project_id=project_id,
                agent_type="analyzer",
                plan_steps=[
                    {
                        "step": "analyze_papers",
                        "status": "success",
                        "output": {
                            "papers_analyzed": len(analyses_summary),
                            "analyses": analyses_summary,
                        },
                    }
                ],
                current_step=1,
                plan_metadata={
                    "model_used": "batched_llm",
                    "batches_processed": len(analyses_summary) // 5 + 1,
                },
            )
            db.add(analyzer_plan)
            logging.info(f"Created analyzer agent plan for project {project_id}")

        # Update Project report JSON container
        report_data = final_state.get("final_report") or final_state.get("report")
        if report_data:
            if hasattr(report_data, "model_dump"):
                report_dict = report_data.model_dump()
            elif hasattr(report_data, "dict"):
                report_dict = report_data.dict()
            elif isinstance(report_data, dict):
                report_dict = report_data
            else:
                report_dict = vars(report_data) if hasattr(report_data, "__dict__") else {}
            
            project.report = json.loads(json.dumps(report_dict, default=str))
            project.report_status = "complete" if project.status == "completed" else "partial"
        else:
            # Check if ResearchReportModel exists in DB
            rep_model = (
                db.query(ResearchReportModel)
                .filter(ResearchReportModel.project_id == project_id)
                .first()
            )
            if rep_model:
                project.report = {
                    "title": rep_model.title,
                    "executive_summary": rep_model.executive_summary,
                    "thematic_sections": rep_model.thematic_sections,
                    "quality_score": rep_model.quality_score,
                }
                project.report_status = "complete"

        # Determine synthesis string for email / plan
        if final_state.get("synthesis"):
            synthesis_output = final_state["synthesis"]
        elif project.report:
            title = project.report.get("title") or project_title
            summary = project.report.get("executive_summary", "")
            synthesis_output = f"# {title}\n\n{summary}"
        else:
            synthesis_output = ""

        synthesizer_plan = AgentPlan(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_type="synthesizer",
            plan_steps=[
                {
                    "step": "synthesize_report",
                    "status": "success",
                    "output": {
                        "response": synthesis_output,
                        "report_status": project.report_status if project.report else "empty",
                    },
                }
            ],
            current_step=1,
            plan_metadata={
                "sections_generated": (
                    len(project.report.get("thematic_sections", []))
                    if project.report and isinstance(project.report, dict)
                    else 0
                ),
                "word_count": len(synthesis_output.split()),
            },
        )
        db.add(synthesizer_plan)
        logging.info(f"Created synthesizer agent plan for project {project_id}")

        db.commit()

        tracker.complete(
            report=project.report,
            papers_analyzed=len(analyzed_papers),
            synthesis_words=len(synthesis_output.split()),
        )

        if user_email and synthesis_output:
            send_completion_email(
                user_email=user_email,
                user_name=user_name,
                project_title=project_title,
                synthesis_output=synthesis_output,
            )

        return {
            "status": project.status,
            "papers_analyzed": len(analyzed_papers),
        }
    except TaskCancelledException as tce:
        logging.info(f"Project {project_id} execution halted due to user cancellation: {tce}")
        try:
            project_to_update = (
                db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
            )
            if project_to_update:
                project_to_update.status = "stopped"
                db.commit()
        except Exception as update_err:
            logging.warning(f"Failed to update project status to stopped: {update_err}")
        return {"status": "stopped", "message": "Task stopped by user."}
    except Exception as e:
        # Check if active cancellation is requested
        if cancellation_manager.is_cancelled(project_id):
            logging.info(
                f"Project {project_id} aborted during exception handling due to active cancellation."
            )
            try:
                project_to_update = (
                    db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
                )
                if project_to_update:
                    project_to_update.status = "stopped"
                    db.commit()
            except Exception:
                pass
            return {"status": "stopped", "message": "Task stopped by user."}

        logging.error(
            f"An error occurred during literature review for project {project_id}: {e}",
            exc_info=True,
        )

        try:
            tracker.error(f"Error: {e!s}")
        except:
            pass
        try:
            project_to_update = (
                db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
            )
            analyzed_from_state = (
                final_state.get("analyzed_papers", []) if "final_state" in locals() else []
            )

            if project_to_update and analyzed_from_state:
                logging.info(
                    f"Attempting to complete project {project_id} with partial results "
                    f"({len(analyzed_from_state)} papers)"
                )
                project_to_update.status = "completed_partial"
                fallback_papers = [
                    p.get("title", f"Paper {i}") for i, p in enumerate(analyzed_from_state)
                ]
                subtopic = "Literature Review"
                if project_to_update.subtopics:
                    subtopic = project_to_update.subtopics[0]

                fallback_response = _create_fallback_synthesis(subtopic, fallback_papers)

                synthesizer_plan = AgentPlan(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    agent_type="synthesizer",
                    plan_steps=[
                        {
                            "step": "synthesize_section",
                            "status": "partial",
                            "output": {"response": fallback_response, "error": str(e)},
                        }
                    ],
                    current_step=1,
                    plan_metadata={"partial_completion": True, "error": str(e)},
                )
                db.add(synthesizer_plan)
                db.commit()
                return {
                    "status": "completed_partial",
                    "papers_analyzed": len(analyzed_from_state),
                    "error": str(e),
                }
            elif project_to_update:
                project_to_update.status = "error"
                db.commit()
        except Exception as recovery_error:
            logging.error(f"Recovery also failed: {recovery_error}")
            try:
                db.rollback()
            except:
                pass
        raise
    finally:
        cancellation_manager.unregister_task(project_id)
        try:
            db.close()
        except:
            pass
        TaskSession.remove()


@celery_app.task(name="run_literature_review", bind=True)
def run_literature_review(self, project_id: str, max_papers: int):
    """Execute the full literature review pipeline as a Celery background task."""
    return _run_literature_review_internal(project_id, max_papers)


def _create_fallback_synthesis(subtopic: str, paper_analyses: list[str]) -> str:
    """
    Create a basic synthesis when LLM-based synthesis fails.

    This ensures we NEVER fail to produce some output for the user.
    """
    synthesis_parts = [
        f"# Literature Review: {subtopic}\n\n",
        "## Overview\n\n",
        f"This literature review covers {len(paper_analyses)} papers on the topic of {subtopic}. ",
        "Due to processing constraints, this is a condensed summary of the research findings.\n\n",
        "## Key Papers Analyzed\n\n",
    ]

    for i, analysis in enumerate(paper_analyses[:10], 1):
        lines = analysis.split("\n")
        title = f"Paper {i}"
        for line in lines[:5]:
            if "title" in line.lower() or line.startswith("#"):
                title = line.replace("#", "").replace("Title:", "").strip()[:100]
                break

        synthesis_parts.append(f"### {title}\n\n")

        content_lines = [line for line in lines if line.strip() and not line.startswith("#")][:3]
        if content_lines:
            synthesis_parts.append(" ".join(content_lines)[:500] + "...\n\n")

    if len(paper_analyses) > 10:
        synthesis_parts.append(
            f"\n*Note: {len(paper_analyses) - 10} additional papers were analyzed "
            "but not included in this summary due to length constraints.*\n\n"
        )

    synthesis_parts.append(
        "\n## Conclusion\n\n"
        "This review provides an overview of the current research landscape. "
        "For a more detailed analysis, please review the individual paper analyses above."
    )

    return "".join(synthesis_parts)
