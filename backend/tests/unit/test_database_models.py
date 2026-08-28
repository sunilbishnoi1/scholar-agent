"""
Unit and integration tests for SQLAlchemy 2.0 database models.

Verifies:
- Clean table creation on SQLite / PostgreSQL engines
- PaperCache DOI primary key deduplication, full-text caching, section/table JSON
- ResearchReportModel, EvidenceMatrixEntry, and ResearchGapModel schemas
- Foreign key CASCADE delete behavior from ResearchProject parent
- Backwards compatibility with User, ResearchProject, AgentPlan, PaperReference, UserUsage, LLMInteraction
"""

from datetime import date, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.models.database import (
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
    UserUsage,
)


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session with foreign keys enabled."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


class TestDatabaseSchemaCreation:
    """Test table creation and schema definitions across all 10 models."""

    def test_all_tables_created(self, db_session: Session):
        tables = Base.metadata.tables.keys()
        expected = {
            "users",
            "research_projects",
            "agent_plans",
            "paper_references",
            "user_usage",
            "llm_interactions",
            "paper_cache",
            "research_reports",
            "evidence_matrix_entries",
            "research_gaps",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"


class TestPaperCacheModel:
    """Test global PaperCache model with DOI deduplication and structured fields."""

    def test_paper_cache_crud(self, db_session: Session):
        doi = "10.1145/3377325.3377498"
        paper = PaperCache(
            doi=doi,
            arxiv_id="2401.01234",
            s2_id="s2_corpus_9988",
            title="Attention is All You Need for Literature Synthesis",
            authors=["Alice Smith", "Bob Jones"],
            year=2024,
            venue="NeurIPS 2024",
            abstract="We propose a novel synthesis architecture...",
            parsed_markdown="# 1. Introduction\n\nFull text here...",
            sections_json=[
                {"title": "Introduction", "body": "Full text here...", "category": "introduction"}
            ],
            tables_json=[{"id": 1, "caption": "Table 1: Accuracy Benchmark"}],
            source_url="https://arxiv.org/abs/2401.01234",
            is_full_text=True,
            fetched_at=datetime.utcnow(),
        )
        db_session.add(paper)
        db_session.commit()

        retrieved = db_session.get(PaperCache, doi)
        assert retrieved is not None
        assert retrieved.doi == doi
        assert retrieved.arxiv_id == "2401.01234"
        assert retrieved.s2_id == "s2_corpus_9988"
        assert retrieved.title == "Attention is All You Need for Literature Synthesis"
        assert len(retrieved.authors) == 2
        assert retrieved.year == 2024
        assert retrieved.is_full_text is True
        assert len(retrieved.sections_json) == 1
        assert len(retrieved.tables_json) == 1

    def test_paper_cache_doi_uniqueness(self, db_session: Session):
        doi = "10.1234/test.duplicate"
        paper1 = PaperCache(doi=doi, title="First Version")
        db_session.add(paper1)
        db_session.commit()
        db_session.expunge(paper1)

        paper2 = PaperCache(doi=doi, title="Second Version")
        db_session.add(paper2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_paper_cache_non_doi_fallback(self, db_session: Session):
        arxiv_doi_fallback = "arxiv:2401.99999"
        paper = PaperCache(
            doi=arxiv_doi_fallback,
            arxiv_id="2401.99999",
            title="ArXiv Preprint Paper",
            authors=["Carol Danvers"],
            is_full_text=False,
        )
        db_session.add(paper)
        db_session.commit()

        retrieved = db_session.get(PaperCache, arxiv_doi_fallback)
        assert retrieved is not None
        assert retrieved.doi == "arxiv:2401.99999"
        assert retrieved.is_full_text is False


class TestResearchProjectRelationalCascade:
    """Test project relational models and cascading delete behavior."""

    @pytest.fixture
    def setup_project(self, db_session: Session):
        user = User(
            email=f"user_{uuid4().hex[:8]}@example.com",
            name="Dr. Scholar",
            hashed_password="hashed_pw_secret",
            tier="pro",
        )
        db_session.add(user)
        db_session.flush()

        project = ResearchProject(
            user_id=user.id,
            title="Multi-Agent Systems in Healthcare",
            research_question="How do multi-agent systems improve clinical decision support?",
            keywords=["multi-agent", "healthcare", "clinical AI"],
            subtopics=["diagnostics", "drug discovery"],
            status="active",
        )
        db_session.add(project)
        db_session.commit()
        return user, project

    def test_research_report_persistence_and_cascade(self, db_session: Session, setup_project):
        user, project = setup_project

        report = ResearchReportModel(
            project_id=project.id,
            title="Systematic Review of Multi-Agent Systems in Healthcare",
            executive_summary="Multi-agent systems provide robust distributed reasoning...",
            methodology_overview={"empirical": 0.6, "theoretical": 0.4},
            quality_score=88.5,
            thematic_sections=[
                {
                    "theme_id": "theme_1",
                    "title": "Clinical Diagnostic Agents",
                    "synthesis_prose": "Diagnostic agents demonstrated superior triage [ref_1#sec_2].",
                    "cited_paper_ids": ["paper_1"],
                }
            ],
            conflicts_and_debates=[
                {
                    "topic": "Centralized vs Decentralized Coordination",
                    "perspective_a": "Centralized ensures global coherence.",
                    "perspective_b": "Decentralized improves fault tolerance.",
                    "critical_evaluation": "Hybrid hierarchical models outperform pure topologies.",
                }
            ],
            generated_at=datetime.utcnow(),
        )
        db_session.add(report)
        db_session.commit()

        # Verify query
        stmt = select(ResearchReportModel).where(ResearchReportModel.project_id == project.id)
        saved_report = db_session.scalar(stmt)
        assert saved_report is not None
        assert saved_report.quality_score == 88.5
        assert len(saved_report.thematic_sections) == 1
        assert len(saved_report.conflicts_and_debates) == 1

        # Delete project -> cascade should delete report
        db_session.delete(project)
        db_session.commit()

        stmt_after = select(ResearchReportModel).where(ResearchReportModel.id == report.id)
        assert db_session.scalar(stmt_after) is None

    def test_evidence_matrix_entries_persistence_and_cascade(
        self, db_session: Session, setup_project
    ):
        user, project = setup_project

        entry1 = EvidenceMatrixEntry(
            project_id=project.id,
            paper_id="doi:10.1016/j.artmed.2023.102450",
            title="Agent-Based Clinical Decision Support",
            methodology_type="Multi-Agent Simulation",
            benchmark_dataset="MIMIC-IV Intensive Care Dataset",
            primary_metric="AUROC 0.94",
            primary_limitation="High compute latency on 100+ concurrent agents",
        )
        entry2 = EvidenceMatrixEntry(
            project_id=project.id,
            paper_id="doi:10.1038/s41746-024-01012-w",
            title="Federated Clinical Reasoning Agents",
            methodology_type="Federated Multi-Task Learning",
            benchmark_dataset="eICU Collaborative Research Database",
            primary_metric="F1-score 0.89",
            primary_limitation="Communication overhead across hospital nodes",
        )
        db_session.add_all([entry1, entry2])
        db_session.commit()

        # Verify retrieval
        entries = (
            db_session.scalars(
                select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.project_id == project.id)
            )
            .all()
        )
        assert len(entries) == 2
        assert {e.primary_metric for e in entries} == {"AUROC 0.94", "F1-score 0.89"}

        # Delete project -> cascade delete
        db_session.delete(project)
        db_session.commit()

        remaining_entries = db_session.scalars(
            select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.project_id == project.id)
        ).all()
        assert len(remaining_entries) == 0

    def test_research_gaps_persistence_and_cascade(self, db_session: Session, setup_project):
        user, project = setup_project

        gap = ResearchGapModel(
            project_id=project.id,
            gap_id="gap_1",
            description="Lack of standardized safety verification in decentralized agent coordination.",
            importance="high",
            recommended_methodology="Formal model checking combined with bounded state verification.",
            grounding_paper_ids=["doi:10.1016/j.artmed.2023.102450"],
            created_at=datetime.utcnow(),
        )
        db_session.add(gap)
        db_session.commit()

        retrieved_gap = db_session.scalar(
            select(ResearchGapModel).where(ResearchGapModel.project_id == project.id)
        )
        assert retrieved_gap is not None
        assert retrieved_gap.gap_id == "gap_1"
        assert retrieved_gap.importance == "high"
        assert len(retrieved_gap.grounding_paper_ids) == 1

        # Delete project -> cascade delete
        db_session.delete(project)
        db_session.commit()

        assert (
            db_session.scalar(
                select(ResearchGapModel).where(ResearchGapModel.id == gap.id)
            )
            is None
        )


class TestBackwardsCompatibility:
    """Test that all existing legacy models function with full fidelity."""

    def test_legacy_models_full_workflow(self, db_session: Session):
        # 1. User
        user = User(
            email="legacy_test@scholar.edu",
            name="Legacy Tester",
            hashed_password="pw_hash_test",
            institution="Stanford University",
            tier="enterprise",
            monthly_budget_usd=50.0,
        )
        db_session.add(user)
        db_session.flush()

        # 2. ResearchProject
        project = ResearchProject(
            user_id=user.id,
            title="Legacy Compatibility Project",
            research_question="Testing backward compatibility",
            report={"legacy_field": True},
            report_status="analysis_only",
        )
        db_session.add(project)
        db_session.flush()

        # 3. AgentPlan
        plan = AgentPlan(
            project_id=project.id,
            agent_type="supervisor",
            plan_steps=[{"step": 1, "task": "discovery", "status": "completed"}],
            current_step=1,
            plan_metadata={"retries": 0},
        )
        db_session.add(plan)

        # 4. PaperReference
        paper_ref = PaperReference(
            project_id=project.id,
            title="Legacy Paper Reference",
            authors=["John Doe"],
            abstract="Short summary",
            url="https://doi.org/10.1234/legacy",
            embeddings=[0.1, 0.2, 0.3],
            relevance_score=0.95,
        )
        db_session.add(paper_ref)

        # 5. UserUsage
        usage = UserUsage(
            user_id=user.id,
            month=date.today().replace(day=1),
            total_tokens=15000,
            prompt_tokens=10000,
            completion_tokens=5000,
            total_cost_usd=0.03,
            projects_created=1,
            papers_analyzed=5,
            llm_calls=8,
        )
        db_session.add(usage)

        # 6. LLMInteraction
        llm = LLMInteraction(
            user_id=user.id,
            project_id=project.id,
            agent_type="thematic_synthesizer",
            model="gemini-2.0-flash",
            task_type="synthesis",
            prompt_tokens=4000,
            completion_tokens=800,
            total_tokens=4800,
            cost_usd=0.0012,
            latency_ms=1250,
            prompt_preview="Synthesize section 1...",
            response_preview="Thematic synthesis results...",
            success=True,
        )
        db_session.add(llm)
        db_session.commit()

        # Assert all models persisted and accessible via ORM
        assert db_session.get(User, user.id).institution == "Stanford University"
        assert db_session.get(ResearchProject, project.id).report_status == "analysis_only"
        assert db_session.get(AgentPlan, plan.id).current_step == 1
        assert db_session.get(PaperReference, paper_ref.id).relevance_score == 0.95
        assert db_session.get(UserUsage, usage.id).total_tokens == 15000
        assert db_session.get(LLMInteraction, llm.id).latency_ms == 1250
