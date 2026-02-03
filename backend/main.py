import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta

import markdown
from celery import Celery
from dotenv import load_dotenv

# Initialize logger
logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sib_api_v3_sdk.rest import ApiException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, joinedload, scoped_session, sessionmaker

# Load environment variables first (must be before local imports)
load_dotenv()

import auth
from agents.llm import get_llm_client
from agents.planner import ResearchPlannerAgent
from db import engine, get_db
from models.database import AgentPlan, Base, LLMInteraction, PaperReference, ResearchProject, User
from paper_retriever import PaperRetriever
from services.usage_tracker import UsageTracker

try:
    from rag.service import RAGService
except ImportError:
    RAGService = None  # RAG service not available


def create_db_and_tables():
    """Initialize database tables and run schema migrations."""
    try:
        # First, create all tables defined in models
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created/verified successfully.")

        # Run schema migrations for missing columns on existing tables
        _run_schema_migrations()

        # Verify the schema is correct
        _verify_schema()

    except Exception as e:
        logging.error(f"Error during database initialization: {e}", exc_info=True)
        raise


async def start_redis_listener():
    """Start Redis pub/sub listener for WebSocket broadcasts."""
    try:
        from realtime.manager import get_connection_manager

        manager = get_connection_manager()
        await manager.start_redis_listener()
        logging.info("Redis listener started for WebSocket broadcasts")
    except Exception as e:
        logging.error(f"Failed to start Redis listener: {e}", exc_info=True)


def _is_postgresql() -> bool:
    """Check if we're connected to PostgreSQL."""
    return "postgresql" in str(engine.url)


def _get_existing_columns(conn, table_name: str) -> set:
    """Get existing columns for a table, handling both PostgreSQL and SQLite."""
    if _is_postgresql():
        # PostgreSQL: Query information_schema with proper schema filter
        result = conn.execute(
            text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = :table_name
        """),
            {"table_name": table_name},
        )
    else:
        # SQLite: Use pragma
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        # SQLite returns: (cid, name, type, notnull, dflt_value, pk)
        return {row[1] for row in result.fetchall()}

    return {row[0] for row in result.fetchall()}


def _add_column_if_not_exists(
    conn, table_name: str, column_name: str, column_def: str, existing_columns: set
) -> bool:
    """Add a column if it doesn't exist. Returns True if column was added or already exists."""
    if column_name in existing_columns:
        logging.info(f"Column '{column_name}' already exists in '{table_name}'.")
        return True

    try:
        if _is_postgresql():
            # PostgreSQL 9.6+ supports IF NOT EXISTS
            sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_def}"
        else:
            # SQLite doesn't support IF NOT EXISTS, but we already checked
            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"

        conn.execute(text(sql))
        logging.info(f"Successfully added column '{column_name}' to '{table_name}'.")
        return True
    except Exception as e:
        # Check if error is "column already exists" (can happen in race conditions)
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate column" in error_str:
            logging.info(f"Column '{column_name}' already exists (detected from error).")
            return True
        logging.error(f"Failed to add column '{column_name}' to '{table_name}': {e}")
        return False


def _run_schema_migrations():
    """Add missing columns to existing tables (lightweight migration).

    This handles the case where the ORM model has new columns but the
    database table was created with an older schema.
    """
    logging.info(
        f"Running schema migrations... (Database: {'PostgreSQL' if _is_postgresql() else 'SQLite'})"
    )

    try:
        # Use engine.begin() for automatic transaction commit/rollback (SQLAlchemy 2.0)
        with engine.begin() as conn:
            # Check if users table exists
            if _is_postgresql():
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'users'
                    )
                """))
                table_exists = result.scalar()
            else:
                result = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                )
                table_exists = result.fetchone() is not None

            if not table_exists:
                logging.info("Users table does not exist yet, skipping migrations.")
                return

            # Get existing columns
            existing_columns = _get_existing_columns(conn, "users")
            logging.info(f"Existing columns in 'users' table: {existing_columns}")

            if not existing_columns:
                logging.warning("No columns found in users table, this is unexpected.")
                return

            # Define migrations: (column_name, column_definition)
            migrations = [
                ("tier", "VARCHAR(50) DEFAULT 'free'"),
                (
                    "monthly_budget_usd",
                    "DOUBLE PRECISION DEFAULT 1.0",
                ),  # DOUBLE PRECISION is PostgreSQL's FLOAT
            ]

            # For SQLite, use different type names
            if not _is_postgresql():
                migrations = [
                    ("tier", "VARCHAR(50) DEFAULT 'free'"),
                    ("monthly_budget_usd", "REAL DEFAULT 1.0"),  # SQLite uses REAL for floats
                ]

            success_count = 0
            for column_name, column_def in migrations:
                if _add_column_if_not_exists(
                    conn, "users", column_name, column_def, existing_columns
                ):
                    success_count += 1

            logging.info(
                f"Schema migrations completed: {success_count}/{len(migrations)} columns processed."
            )

        # Transaction is auto-committed when exiting begin() context successfully

    except Exception as e:
        logging.error(f"Schema migration failed: {e}", exc_info=True)
        raise  # Re-raise to prevent app from starting with broken schema


def _verify_schema():
    """Verify that all required columns exist after migrations."""
    required_columns = {
        "id",
        "email",
        "name",
        "hashed_password",
        "tier",
        "monthly_budget_usd",
        "created_at",
    }

    try:
        with engine.connect() as conn:
            existing_columns = _get_existing_columns(conn, "users")

            missing = required_columns - existing_columns
            if missing:
                logging.error(
                    f"SCHEMA VERIFICATION FAILED! Missing columns in 'users' table: {missing}"
                )
                logging.error(f"Existing columns: {existing_columns}")
                raise RuntimeError(
                    f"Database schema verification failed. Missing columns: {missing}"
                )

            logging.info(
                f"Schema verification passed. All required columns present: {required_columns}"
            )

    except Exception as e:
        logging.error(f"Schema verification error: {e}", exc_info=True)
        raise


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = FastAPI(on_startup=[create_db_and_tables, start_redis_listener])
celery_app = Celery("literature_agent", broker=REDIS_URL)

origins = [
    "https://scholar-agent.vercel.app",  # production frontend
    "https://scholaragent.dpdns.org",
    "http://localhost:8000",
    "http://localhost:5174",
    "http://localhost:5173",
]


# Global exception handler to ensure CORS headers are always sent
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# CORS middleware must be added AFTER exception handlers
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# --- Root endpoint for health checks and API discovery ---
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


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# --- Pydantic Schemas for API responses ---
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
    total_papers_found: int  # <-- ADDED
    created_at: datetime
    agent_plans: list[AgentPlanSchema] = []
    paper_references: list[PaperReferenceSchema] = []

    class Config:
        from_attributes = True


class ProjectCreate(BaseModel):
    title: str
    research_question: str


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
    projects = db.query(ResearchProject).filter(ResearchProject.user_id == current_user.id).all()
    return projects


@projects_router.get("/projects/{project_id}", response_model=ResearchProjectSchema)
def get_project(
    project_id: str,
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
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project


@projects_router.post("/projects/{project_id}/start")
def start_literature_review(
    project_id: str,
    max_papers: int = 50,
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

    project.status = "searching"
    db.commit()

    job = celery_app.send_task("run_literature_review", args=[project_id, max_papers])
    return {"job_id": job.id, "status": "queued", "estimated_duration": f"PT{max_papers // 2}M"}


class DeleteProjectResponse(BaseModel):
    id: str
    deleted: bool
    message: str


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

    # Delete associated RAG data if the service is available
    if RAGService is not None:
        try:
            rag_service = RAGService()
            rag_service.delete_project_data(project_id)
            logging.info(f"Deleted RAG data for project {project_id}")
        except Exception as e:
            logging.warning(f"Failed to delete RAG data for project {project_id}: {e}")
            # Continue with deletion even if RAG cleanup fails

    # Delete associated LLM interactions (has FK to project)
    db.query(LLMInteraction).filter(LLMInteraction.project_id == project_id).delete()

    # Delete associated agent plans
    db.query(AgentPlan).filter(AgentPlan.project_id == project_id).delete()

    # Delete associated paper references
    db.query(PaperReference).filter(PaperReference.project_id == project_id).delete()

    # Delete the project itself
    db.delete(project)
    db.commit()

    logging.info(f"User {current_user.id} deleted project {project_id} ('{project_title}')")

    return DeleteProjectResponse(
        id=project_id,
        deleted=True,
        message=f"Project '{project_title}' and all associated data deleted successfully",
    )


# --- App Integration ---
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(projects_router, prefix="/api", tags=["Projects"])

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

    # Use RAG service if available
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
        # RAG service not available, return empty results
        return []


app.include_router(search_router, prefix="/api", tags=["Search"])


# --- WebSocket Router ---
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

    # For now, use a simple authentication (TODO: proper JWT validation)
    # If token is provided, validate it; otherwise use anonymous user
    user_id = "anonymous"  # TODO: Extract user_id from JWT token

    connection_established = False

    try:
        # Connect and subscribe to project
        success = await manager.connect(websocket, user_id, project_id)
        if not success:
            logger.warning(f"Failed to establish WebSocket connection for project {project_id}")
            try:
                await websocket.close(code=1008, reason="Connection failed")
            except:
                pass  # Connection may already be closed
            return

        connection_established = True

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages (ping/pong for keepalive)
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected for project {project_id}")
                break
            except RuntimeError as e:
                # Handle "WebSocket is not connected" errors gracefully
                logger.warning(f"WebSocket runtime error for project {project_id}: {e}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for project {project_id}: {e}", exc_info=True)
                break

    except Exception as e:
        logger.error(f"WebSocket setup error for project {project_id}: {e}", exc_info=True)

    finally:
        # Clean up connection only if it was established
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
):
    """
    Sends the final synthesized report to the user via the Brevo (Sendinblue) Transactional Email API
    using the official Python SDK (sib_api_v3_sdk).
    """
    # Load API key from environment
    api_key = os.environ.get("BREVO_API_KEY", "")
    if not api_key or api_key == "your_actual_api_key_here":
        logging.warning("BREVO_API_KEY not set or placeholder used. Skipping email notification.")
        return

    # Step 1: Configure client properly
    configuration = Configuration()
    configuration.api_key["api-key"] = api_key
    api_client = ApiClient(configuration)
    email_api = TransactionalEmailsApi(api_client)

    # Step 2: Prepare sender info
    sender_email = os.environ.get("BREVO_SENDER_EMAIL", "sunilbishnoi7205@gmail.com")
    sender_name = os.environ.get("BREVO_SENDER_NAME", "Scholar AI Agent")

    # Step 3: Format HTML content
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

    # Step 4: Send email
    try:
        send_smtp_email = SendSmtpEmail(
            to=[{"email": user_email, "name": user_name}],
            sender={"name": sender_name, "email": sender_email},
            subject=subject,
            html_content=html_content,
        )

        response = email_api.send_transac_email(send_smtp_email)
        logging.info(f"Brevo email sent successfully to {user_email}. Response: {response}")
    except ApiException as e:
        logging.error(f"Brevo API exception when sending email: {e}")
    except Exception as e:
        logging.error(f"Failed to send email via Brevo: {e}")


# ... ( Celery task run_literature_review) ...
@celery_app.task(name="run_literature_review", bind=True)
def run_literature_review(self, project_id: str, max_papers: int):
    """Execute the full literature review pipeline as a Celery background task."""
    from agents.analyzer import PaperAnalyzerAgent
    from agents.llm import get_llm_client
    from agents.synthesizer import SynthesisExecutorAgent
    from realtime.events import AgentProgressTracker

    # Create a fresh DB engine/session for this Celery worker process
    task_engine = create_engine(
        os.environ.get("DATABASE_URL", "sqlite:///./test.db"),
        connect_args=(
            {"check_same_thread": False, "timeout": 15}
            if "sqlite" in os.environ.get("DATABASE_URL", "sqlite:///./test.db")
            else {}
        ),
    )
    TaskSession = scoped_session(sessionmaker(bind=task_engine))
    db = TaskSession()
    llm_client = get_llm_client()
    analyzer = PaperAnalyzerAgent(llm_client)
    synthesizer = SynthesisExecutorAgent(llm_client)
    retriever = PaperRetriever()

    # Initialize progress tracker for real-time WebSocket updates
    tracker = AgentProgressTracker(project_id)

    try:
        project = (
            db.query(ResearchProject)
            .options(joinedload(ResearchProject.user))
            .filter(ResearchProject.id == project_id)
            .first()
        )
        if not project:
            return {"status": "error", "error": "Project not found"}

        # Start retriever agent
        tracker.start_agent("retriever", "Searching for relevant papers...")

        # 1. Retrieve papers
        papers_to_analyze = retriever.search_papers(
            search_terms=project.keywords, max_papers=max_papers
        )

        # Save the number of found papers
        project.total_papers_found = len(papers_to_analyze)
        db.commit()

        # Notify papers found
        tracker.log(f"Found {len(papers_to_analyze)} papers")
        tracker.update_progress(100)  # Retriever is done
        tracker.complete_agent("retriever", f"Found {len(papers_to_analyze)} papers")

        if not papers_to_analyze:
            project.status = "error_no_papers_found"
            db.commit()
            tracker.error("No papers found for the given search criteria")
            logging.warning(f"Could not retrieve any papers for project {project_id}.")
            return {
                "status": "completed_with_warning",
                "message": "No papers found for the given search criteria.",
            }

        # 2. Analyze each retrieved paper
        project.status = "analyzing"
        db.commit()

        tracker.start_agent("analyzer", "Analyzing papers...")

        paper_analyses = []

        for i, paper_data in enumerate(papers_to_analyze):
            content = paper_data.get("abstract", "")
            if not content:
                continue

            analyzer_response_str = analyzer.analyze_paper(
                paper_data["title"], paper_data["abstract"], content, project.research_question
            )

            relevance_score = 0.0
            analysis_json = {}
            try:
                from agents.tools import extract_json_from_response

                analysis_json = extract_json_from_response(analyzer_response_str, {})
                relevance_score = float(analysis_json.get("relevance_score", 0.0))
            except (json.JSONDecodeError, TypeError) as e:
                logging.error(
                    f"Failed to parse JSON from analyzer for paper '{paper_data['title']}': {e}"
                )
                paper_analyses.append(analyzer_response_str)
                relevance_score = 0.0

            analyzer_plan = AgentPlan(
                id=str(uuid.uuid4()),
                project_id=project_id,
                agent_type="analyzer",
                plan_steps=[
                    {
                        "step": "analyze_paper",
                        "status": "completed",
                        "output": {"response": analysis_json or analyzer_response_str},
                    }
                ],
                current_step=1,
                plan_metadata={"paper_title": paper_data["title"]},
            )
            db.add(analyzer_plan)

            paper = PaperReference(
                id=str(uuid.uuid4()),
                project_id=project_id,
                title=paper_data["title"],
                authors=paper_data.get("authors", []),
                abstract=paper_data.get("abstract"),
                url=paper_data.get("url"),
                relevance_score=relevance_score,
            )
            db.add(paper)
            db.commit()
            paper_analyses.append(analyzer_response_str)

            # Update progress and notify paper analyzed
            analyzed_count = i + 1
            total_papers = len(papers_to_analyze)
            progress_pct = (analyzed_count / total_papers) * 100
            tracker.update_progress(
                progress_pct, f"Analyzed {analyzed_count}/{total_papers} papers"
            )
            tracker.paper_analyzed(
                paper_data["title"],
                relevance_score,
                current=analyzed_count,
                total=total_papers,
            )

            time.sleep(1.5)

        tracker.complete_agent("analyzer", f"Analyzed {len(paper_analyses)} papers")

        # 3. Synthesizer logic
        project.status = "synthesizing"
        db.commit()

        tracker.start_agent("synthesizer", "Synthesizing final report...")

        subtopic = project.subtopics[0] if project.subtopics else "Comprehensive Literature Review"

        # Try synthesis with graceful degradation
        try:
            synthesizer_response = synthesizer.synthesize_section(
                subtopic=subtopic,
                paper_analyses="\n\n---\n\n".join(paper_analyses),
                academic_level="graduate",
                word_count=500,
            )
        except Exception as synth_error:
            # If synthesis fails, create a basic summary from what we have
            logging.error(f"Synthesis failed: {synth_error}. Creating basic summary.")
            synthesizer_response = _create_fallback_synthesis(subtopic, paper_analyses)

        synthesizer_plan = AgentPlan(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_type="synthesizer",
            plan_steps=[
                {
                    "step": "synthesize_section",
                    "status": "completed",
                    "output": {"response": synthesizer_response},
                }
            ],
            current_step=1,
            plan_metadata={},
        )
        db.add(synthesizer_plan)

        project.status = "completed"
        db.commit()

        tracker.complete_agent("synthesizer", "Synthesis complete")
        tracker.complete(papers_analyzed=len(paper_analyses))

        # ---send email after completion---

        if project.user:
            send_completion_email(
                user_email=project.user.email,
                user_name=project.user.name,
                project_title=project.title,
                synthesis_output=synthesizer_response,
            )
        else:
            logging.warning(
                f"Project {project_id} has no associated user. Cannot send completion email."
            )

        return {"status": "completed", "papers_analyzed": len(paper_analyses)}
    except Exception as e:
        logging.error(
            f"An error occurred during literature review for project {project_id}: {e}",
            exc_info=True,
        )

        # Notify error via tracker
        tracker.error(f"Error: {e!s}")

        # Try to complete with partial results instead of failing completely
        try:
            project_to_update = (
                db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
            )
            if project_to_update and paper_analyses:
                # We have some analyses - try to complete with what we have
                logging.info(
                    f"Attempting to complete project {project_id} with partial results "
                    f"({len(paper_analyses)} papers)"
                )
                project_to_update.status = "completed_partial"
                fallback_response = _create_fallback_synthesis(
                    (
                        project_to_update.subtopics[0]
                        if project_to_update.subtopics
                        else "Literature Review"
                    ),
                    paper_analyses,
                )
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
                    "papers_analyzed": len(paper_analyses),
                    "error": str(e),
                }
            elif project_to_update:
                project_to_update.status = "error"
                db.commit()
        except Exception as recovery_error:
            logging.error(f"Recovery also failed: {recovery_error}")
            db.rollback()
        raise
    finally:
        db.close()


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

    # Extract titles and key points from analyses
    for i, analysis in enumerate(paper_analyses[:10], 1):  # Limit to first 10
        # Try to extract title from analysis
        lines = analysis.split("\n")
        title = f"Paper {i}"
        for line in lines[:5]:  # Check first 5 lines
            if "title" in line.lower() or line.startswith("#"):
                title = line.replace("#", "").replace("Title:", "").strip()[:100]
                break

        synthesis_parts.append(f"### {title}\n\n")

        # Get first few meaningful lines as summary
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
