from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from celery import Celery
import logging
import re
import os
from models.database import Base, User, ResearchProject, AgentPlan, PaperReference
from paper_retriever import PaperRetriever

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False, "timeout": 15} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI()
celery_app = Celery('literature_agent', broker='redis://localhost:6379')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Pydantic schema for project creation
class ProjectCreate(BaseModel):
    title: str
    research_question: str
    keywords: list[str]
    user_id: str = None  # Optional for now

# POST /api/projects
@app.post("/api/projects")
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    # For now, assign to first user or create a dummy user if none exists
    user = db.query(User).first()
    if not user:
        user = User(email="demo@demo.com", name="Demo User")
        db.add(user)
        db.commit()
        db.refresh(user)
    new_project = ResearchProject(
        user_id=user.id,
        title=project.title,
        research_question=project.research_question,
        keywords=project.keywords,
        status="planning"
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return {"id": new_project.id, "status": "created"}

# POST /api/projects/{id}/start
@app.post("/api/projects/{project_id}/start")
def start_literature_review(project_id: str, max_papers: int = 50):
    # Queue Celery task for agent pipeline
    job = celery_app.send_task(
        'run_literature_review',
        args=[project_id, max_papers]
    )
    return {
        'job_id': job.id,
        'status': 'queued',
        'estimated_duration': f'PT{max_papers//2}M'
    }



# Celery task for agent execution (agentic pipeline with Gemini LLM)
@celery_app.task(name='run_literature_review', bind=True)
def run_literature_review(self, project_id: str, max_papers: int):
    from sqlalchemy.orm import scoped_session
    from sqlalchemy import create_engine
    import uuid
    import datetime
    from agents.gemini_client import GeminiClient
    from agents.planner import ResearchPlannerAgent
    from agents.analyzer import PaperAnalyzerAgent
    from agents.synthesizer import SynthesisExecutorAgent

    # DB setup (Celery worker context)
    engine = create_engine(
    os.environ.get("DATABASE_URL", "sqlite:///./test.db"), 
    connect_args={"check_same_thread": False, "timeout": 15} if "sqlite" in os.environ.get("DATABASE_URL", "sqlite:///./test.db") else {}
    )
    Session = scoped_session(sessionmaker(bind=engine))
    db = Session()
    interactions = []
    gemini = GeminiClient()
    planner = ResearchPlannerAgent(gemini)
    analyzer = PaperAnalyzerAgent(gemini)
    synthesizer = SynthesisExecutorAgent(gemini)
    retriever = PaperRetriever()
    try:
        # 1. Planner: generate search strategy
        project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        if not project:
            return {"status": "error", "error": "Project not found"}

        # 1. Planner: generate search strategy
        project.status = "planning"
        db.commit()

        planner_response = planner.create_search_strategy(project.research_question, project.keywords, max_papers)
        planner_plan = AgentPlan( # ... (save planner plan as before)
            id=str(uuid.uuid4()), project_id=project_id, agent_type="planner",
            plan_steps=[{"step": "generate_search_terms", "status": "completed", "output": {"response": planner_response}}],
            current_step=1, plan_metadata={}
        )
        db.add(planner_plan)
        db.commit()

        # 2. Retrieve real papers using the new PaperRetriever
        project.status = "searching"
        db.commit()
        papers_to_analyze = retriever.search_papers(
            planner_response=planner_response, 
            max_papers=max_papers,
            fallback_keywords=project.keywords
        )
        
        if not papers_to_analyze:
            project.status = "error_no_papers_found"
            db.commit()
            logging.warning(f"Could not retrieve any papers for project {project_id}.")
            return {"status": "completed_with_warning", "message": "No papers found for the given search criteria."}

        # 3. Analyze each retrieved paper
        project.status = "analyzing"
        db.commit()
        paper_analyses = []

        import json

        for i, paper_data in enumerate(papers_to_analyze):
            # For now, we use the abstract as the "full content".
            content = paper_data.get("abstract", "")
            if not content:
                continue # Skip papers with no abstract

            analyzer_response_str = analyzer.analyze_paper( 
                paper_data["title"], 
                paper_data["abstract"], 
                content, 
                project.research_question
            )

            relevance_score = 0.0  # Default value
            analysis_json = {}
            try:
                # Clean up the response in case the LLM wraps it in markdown
                clean_response = re.sub(r'```json\s*|\s*```', '', analyzer_response_str).strip()
                analysis_json = json.loads(clean_response)
                # Ensure the score is a float
                relevance_score = float(analysis_json.get("relevance_score", 0.0))
            except (json.JSONDecodeError, TypeError) as e:
                logging.error(f"Failed to parse JSON from analyzer for paper '{paper_data['title']}': {e}")
                # We can still append the raw text for the final synthesis
                paper_analyses.append(analyzer_response_str)
                relevance_score = 0.0 # Assign a default score on failure

            analyzer_plan = AgentPlan(
                id=str(uuid.uuid4()), project_id=project_id, agent_type="analyzer",
                plan_steps=[{"step": "analyze_paper", "status": "completed", "output": {"response": analysis_json or analyzer_response_str}}],
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
            db.commit() # Commit after each paper to save progress
            paper_analyses.append(analyzer_response_str) # Append the original string for the synthesizer

        # 4. Synthesize a literature review section from the analyses
        project.status = "synthesizing"
        db.commit()

        # Dynamically get the main subtopic from the planner's output
        subtopic = "Literature Review" # Fallback title
        subtopics_match = re.search(r'5\.\s+Subtopics:([\s\S]*)', planner_response, re.IGNORECASE)
        if subtopics_match:
            first_subtopic = re.search(r'^\s*[\*\-\d]+\.?\s*(.*)', subtopics_match.group(1), re.MULTILINE)
            if first_subtopic:
                subtopic = first_subtopic.group(1).strip()
        
        synthesizer_response = synthesizer.synthesize_section(
            subtopic=subtopic,
            paper_analyses="\n\n---\n\n".join(paper_analyses), # Join analyses into a single context
            academic_level="graduate",
            word_count=500
        )
        synthesizer_plan = AgentPlan( # ... (save synthesizer plan as before)
            id=str(uuid.uuid4()), project_id=project_id, agent_type="synthesizer",
            plan_steps=[{"step": "synthesize_section", "status": "completed", "output": {"response": synthesizer_response}}],
            current_step=1, plan_metadata={}
        )
        db.add(synthesizer_plan)
        
        project.status = "completed"
        db.commit()

        return {"status": "completed", "papers_analyzed": len(paper_analyses)}
    except Exception as e:
        logging.error(f"An error occurred during literature review for project {project_id}: {e}", exc_info=True)
        # Safely query and update project status to error
        project_to_update = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        if project_to_update:
            project_to_update.status = "error"
            db.commit()
        db.rollback()
        # Re-raise the exception so Celery can mark the task as FAILED
        raise
    finally:
        db.close()