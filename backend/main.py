from fastapi import FastAPI, HTTPException, Depends, APIRouter, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, joinedload
from celery import Celery
import logging
import re
import os
import time
from typing import List, Optional
from datetime import datetime, timedelta
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import markdown

from models.database import Base, User, ResearchProject, AgentPlan, PaperReference
from paper_retriever import PaperRetriever
from agents.gemini_client import GeminiClient
from agents.planner import ResearchPlannerAgent
import auth 
from db import get_db

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 15} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

app = FastAPI()
celery_app = Celery('literature_agent', broker=REDIS_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    authors: Optional[List[str]] = []
    abstract: Optional[str] = None
    url: Optional[str] = None
    relevance_score: Optional[float] = None
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
    keywords: List[str]
    subtopics: List[str]
    status: str
    total_papers_found: int # <-- ADDED
    created_at: datetime
    agent_plans: List[AgentPlanSchema] = []
    paper_references: List[PaperReferenceSchema] = []
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
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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

@projects_router.get("/projects", response_model=List[ResearchProjectSchema])
def get_projects(db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    projects = db.query(ResearchProject).filter(ResearchProject.user_id == current_user.id).all()
    return projects

@projects_router.get("/projects/{project_id}", response_model=ResearchProjectSchema)
def get_project(project_id: str, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    project = db.query(ResearchProject).filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@projects_router.post("/projects", response_model=ResearchProjectSchema)
def create_project(project: ProjectCreate, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    gemini_client = GeminiClient()
    planner = ResearchPlannerAgent(gemini_client)
    
    initial_plan = planner.generate_initial_plan(project.research_question, project.title)
    generated_keywords = initial_plan.get("keywords", [])
    generated_subtopics = initial_plan.get("subtopics", [])

    new_project = ResearchProject(
        user_id=current_user.id,
        title=project.title,
        research_question=project.research_question,
        keywords=generated_keywords,
        subtopics=generated_subtopics,
        status="created"
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@projects_router.post("/projects/{project_id}/start")
def start_literature_review(project_id: str, max_papers: int = 50, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    project = db.query(ResearchProject).filter(ResearchProject.id == project_id, ResearchProject.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = "searching"
    db.commit()

    job = celery_app.send_task(
        'run_literature_review',
        args=[project_id, max_papers]
    )
    return {
        'job_id': job.id,
        'status': 'queued',
        'estimated_duration': f'PT{max_papers // 2}M'
    }

# --- App Integration ---
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(projects_router, prefix="/api", tags=["Projects"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

#----Helper function for sending email--
import os
import logging
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from sib_api_v3_sdk.configuration import Configuration
from sib_api_v3_sdk.api_client import ApiClient
from sib_api_v3_sdk.api.transactional_emails_api import TransactionalEmailsApi
from sib_api_v3_sdk import SendSmtpEmail

def send_completion_email(user_email: str, user_name: str, project_title: str, synthesis_output: str):
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
    configuration.api_key['api-key'] = api_key
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
            html_content=html_content
        )

        response = email_api.send_transac_email(send_smtp_email)
        logging.info(f"Brevo email sent successfully to {user_email}. Response: {response}")
    except ApiException as e:
        logging.error(f"Brevo API exception when sending email: {e}")
    except Exception as e:
        logging.error(f"Failed to send email via Brevo: {e}")



# ... ( Celery task run_literature_review) ...
@celery_app.task(name='run_literature_review', bind=True)
def run_literature_review(self, project_id: str, max_papers: int):
    # ... (imports inside the task remain the same) ...
    from sqlalchemy.orm import scoped_session
    from sqlalchemy import create_engine
    import uuid
    from agents.gemini_client import GeminiClient
    from agents.planner import ResearchPlannerAgent
    from agents.analyzer import PaperAnalyzerAgent
    from agents.synthesizer import SynthesisExecutorAgent

    # ... (DB setup inside the task) ...
    engine = create_engine(
        os.environ.get("DATABASE_URL", "sqlite:///./test.db"),
        connect_args={"check_same_thread": False, "timeout": 15} if "sqlite" in os.environ.get("DATABASE_URL",
                                                                                                "sqlite:///./test.db") else {}
    )
    Session = scoped_session(sessionmaker(bind=engine))
    db = Session()
    gemini = GeminiClient()
    analyzer = PaperAnalyzerAgent(gemini)
    synthesizer = SynthesisExecutorAgent(gemini)
    retriever = PaperRetriever()
    try:
        project = db.query(ResearchProject).options(joinedload(ResearchProject.user)).filter(ResearchProject.id == project_id).first()
        if not project:
            return {"status": "error", "error": "Project not found"}
        
        # 1. Retrieve papers
        papers_to_analyze = retriever.search_papers(
            search_terms=project.keywords,
            max_papers=max_papers
        )

        # --- MODIFIED BLOCK START ---
        # Save the number of found papers so the frontend can display progress correctly.
        project.total_papers_found = len(papers_to_analyze)
        db.commit()
        # --- MODIFIED BLOCK END ---

        if not papers_to_analyze:
            project.status = "error_no_papers_found"
            db.commit()
            logging.warning(f"Could not retrieve any papers for project {project_id}.")
            return {"status": "completed_with_warning", "message": "No papers found for the given search criteria."}

        # 2. Analyze each retrieved paper
        project.status = "analyzing"
        db.commit()
        paper_analyses = []
        import json

        for i, paper_data in enumerate(papers_to_analyze):
            content = paper_data.get("abstract", "")
            if not content:
                continue

            analyzer_response_str = analyzer.analyze_paper(
                paper_data["title"],
                paper_data["abstract"],
                content,
                project.research_question
            )

            relevance_score = 0.0
            analysis_json = {}
            try:
                clean_response = re.sub(r'```json\s*|\s*```', '', analyzer_response_str).strip()
                analysis_json = json.loads(clean_response)
                relevance_score = float(analysis_json.get("relevance_score", 0.0))
            except (json.JSONDecodeError, TypeError) as e:
                logging.error(f"Failed to parse JSON from analyzer for paper '{paper_data['title']}': {e}")
                paper_analyses.append(analyzer_response_str)
                relevance_score = 0.0

            analyzer_plan = AgentPlan(
                id=str(uuid.uuid4()), project_id=project_id, agent_type="analyzer",
                plan_steps=[
                    {"step": "analyze_paper", "status": "completed", "output": {"response": analysis_json or analyzer_response_str}}],
                current_step=1, plan_metadata={"paper_title": paper_data["title"]}
            )
            db.add(analyzer_plan)

            paper = PaperReference(
                id=str(uuid.uuid4()), project_id=project_id, title=paper_data["title"],
                authors=paper_data.get("authors", []), abstract=paper_data.get("abstract"),
                url=paper_data.get("url"),
                relevance_score=relevance_score
            )
            db.add(paper)
            db.commit()
            paper_analyses.append(analyzer_response_str)

            time.sleep(1.5)

        # 3. Synthesizer logic
        project.status = "synthesizing"
        db.commit()

        subtopic = project.subtopics[0] if project.subtopics else "Comprehensive Literature Review"
        
        synthesizer_response = synthesizer.synthesize_section(
            subtopic=subtopic,
            paper_analyses="\n\n---\n\n".join(paper_analyses),
            academic_level="graduate",
            word_count=500
        )
        synthesizer_plan = AgentPlan(
            id=str(uuid.uuid4()), project_id=project_id, agent_type="synthesizer",
            plan_steps=[{"step": "synthesize_section", "status": "completed", "output": {"response": synthesizer_response}}],
            current_step=1, plan_metadata={}
        )
        db.add(synthesizer_plan)
        
        project.status = "completed"
        db.commit()
        
        # ---send email after completion---

        if project.user:
            send_completion_email(
                user_email=project.user.email,
                user_name=project.user.name,
                project_title=project.title,
                synthesis_output=synthesizer_response
            )
        else:
            logging.warning(f"Project {project_id} has no associated user. Cannot send completion email.")

        return {"status": "completed", "papers_analyzed": len(paper_analyses)}
    except Exception as e:
        logging.error(f"An error occurred during literature review for project {project_id}: {e}", exc_info=True)
        project_to_update = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        if project_to_update:
            project_to_update.status = "error"
            db.commit()
        db.rollback()
        raise
    finally:
        db.close()
