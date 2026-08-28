"""
End-to-End Real Multi-Agent Literature Review Test.
Performs live academic search, paper ingestion, matrix building, synthesis,
critic audit, and database persistence with real API keys and real services.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime

# Setup UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RealTestRunner")

# Ensure backend directory in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv()

from auth import get_password_hash
from db import SessionLocal, engine
from main import _run_literature_review_internal, create_db_and_tables
from models.database import (
    AgentPlan,
    Base,
    EvidenceMatrixEntry,
    PaperReference,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
    User,
)


def run_live_test():
    print("=" * 80)
    print("🚀 STARTING REAL MULTI-AGENT LITERATURE REVIEW (MAX 5 PAPERS)")
    print("=" * 80)

    # 1. Initialize tables & migrations
    print("\n[Step 1] Ensuring Database tables & migrations are ready...")
    create_db_and_tables()

    db = SessionLocal()
    try:
        # 2. Create or retrieve test user
        user_email = "scholar_tester@example.com"
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            user = User(
                id=str(uuid.uuid4()),
                email=user_email,
                name="Scholar Agent Researcher",
                hashed_password=get_password_hash("test_password_123!"),
                tier="pro",
                monthly_budget_usd=20.0,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✓ Created test user: {user.name} ({user.id})")
        else:
            print(f"✓ Found existing user: {user.name} ({user.id})")

        # 3. Create Research Project
        project_id = str(uuid.uuid4())
        topic = "Agentic AI and Tool Use in Autonomous Systems"
        research_question = (
            "How do autonomous multi-agent systems coordinate tool usage, reasoning, and planning to solve complex tasks?"
        )
        max_papers = 5

        project = ResearchProject(
            id=project_id,
            user_id=user.id,
            title=topic,
            research_question=research_question,
            keywords=["agentic ai", "tool use", "multi-agent collaboration", "reasoning", "autonomous agents"],
            subtopics=["Multi-Agent Architectures", "Tool Integration & Function Calling", "Autonomous Planning & Reasoning"],
            status="searching",
            max_papers=max_papers,
        )
        db.add(project)
        db.commit()
        print(f"✓ Created Research Project: '{topic}'")
        print(f"  Project ID: {project_id}")
        print(f"  Research Question: {research_question}")
        print(f"  Max Papers: {max_papers}")

        # 4. Execute Real Pipeline
        print("\n[Step 2] Executing Full Multi-Agent Pipeline with Real API Calls...")
        start_time = datetime.utcnow()
        result = _run_literature_review_internal(project_id, max_papers)
        duration = (datetime.utcnow() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"✅ PIPELINE EXECUTION COMPLETED IN {duration:.1f}s")
        print(f"   Status: {result.get('status')}")
        print(f"   Papers Analyzed: {result.get('papers_analyzed')}")
        print("=" * 80)

        # 5. Verify Database Records
        db.expire_all()
        updated_project = db.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        papers = db.query(PaperReference).filter(PaperReference.project_id == project_id).all()
        matrix_entries = db.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == project_id).all()
        plans = db.query(AgentPlan).filter(AgentPlan.project_id == project_id).all()
        reports = db.query(ResearchReportModel).filter(ResearchReportModel.project_id == project_id).all()
        gaps = db.query(ResearchGapModel).filter(ResearchGapModel.project_id == project_id).all()

        print("\n📊 DATABASE PERSISTENCE & RESULTS SUMMARY:")
        print(f"  • Final Project Status: {updated_project.status}")
        print(f"  • Total Papers Found / Referenced in DB: {len(papers)}")
        print(f"  • Evidence Matrix Entries in DB: {len(matrix_entries)}")
        print(f"  • Agent Plans Logged in DB: {len(plans)}")
        print(f"  • Research Gap Models in DB: {len(gaps)}")
        print(f"  • Synthesized Reports in DB: {len(reports)}")

        print("\n📚 PAPERS RETRIEVED & ANALYZED:")
        for idx, p in enumerate(papers, 1):
            authors = ", ".join(p.authors) if isinstance(p.authors, list) else str(p.authors)
            print(f"  [{idx}] {p.title}")
            print(f"      Authors: {authors[:60]}...")
            print(f"      Relevance Score: {p.relevance_score}")
            if p.url:
                print(f"      URL: {p.url}")

        if matrix_entries:
            print(f"\n🧬 EVIDENCE MATRIX SAMPLE (1 of {len(matrix_entries)}):")
            sample = matrix_entries[0]
            print(f"  • Paper: {sample.title}")
            print(f"  • Methodology: {sample.methodology_type}")
            print(f"  • Dataset / Metric: {sample.benchmark_dataset} | {sample.primary_metric}")
            print(f"  • Primary Limitation: {sample.primary_limitation}")

        if updated_project.report and isinstance(updated_project.report, dict):
            print("\n📝 SYNTHESIZED REPORT OVERVIEW:")
            rep = updated_project.report
            print(f"  • Title: {rep.get('title', 'N/A')}")
            summary = rep.get('executive_summary', '')
            print(f"  • Executive Summary Preview:\n    {summary[:300]}...")
            sections = rep.get('thematic_sections', [])
            print(f"  • Thematic Sections Generated: {len(sections)}")
            for s in sections:
                if isinstance(s, dict):
                    print(f"    - {s.get('heading', s.get('title', 'Section'))}")

        print("\n🎉 LIVE REAL API VERIFICATION FINISHED SUCCESSFULLY!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_live_test()
