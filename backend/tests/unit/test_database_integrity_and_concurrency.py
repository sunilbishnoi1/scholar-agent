"""
Database Integrity, Schema Constraints, Cascade Deletion, and Concurrency Stress Test Suite.

Performs empirical stress-testing and verification of:
1. Database Schema, Table Definitions, and Index Configurations on SQLite & PostgreSQL engines.
2. DOI uniqueness constraint, duplicate insertion handling, upsert/merge semantics, and rollback recovery.
3. Multi-field JSON serialization, heterogeneous nested types, large payloads, and roundtrip fidelity.
4. Foreign Key Cascade Deletions (ORM-level, raw SQL engine-level, transitive User->Project->Children).
5. Bidirectional ORM relationship synchronization and collection mutations.
6. Global PaperCache isolation and cross-project isolation.
7. Concurrency, multi-session transaction isolation, thread safety, and concurrent read-while-write under load.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import os
import tempfile
from typing import Any, Generator
from uuid import uuid4

import psycopg2
import pytest
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

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

POSTGRES_AVAILABLE = False
POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
)

try:
    _pg_conn = psycopg2.connect(
        dbname="postgres", user="postgres", password="postgres", host="localhost", port=5432
    )
    _pg_conn.close()
    POSTGRES_AVAILABLE = True
except Exception:
    POSTGRES_AVAILABLE = False


@pytest.fixture
def sqlite_memory_session() -> Generator[Session, None, None]:
    """Isolated SQLite in-memory session with foreign keys enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

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
        engine.dispose()


@pytest.fixture
def sqlite_file_engine() -> Generator[Any, None, None]:
    """Isolated SQLite file-based engine with WAL mode and foreign keys for concurrency testing."""
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test_concurrency.db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        poolclass=NullPool,
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        temp_dir.cleanup()


@pytest.fixture
def postgres_session() -> Generator[Session, None, None]:
    """Isolated PostgreSQL session creating tables and cleaning up after each test."""
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL service not accessible.")

    engine = create_engine(POSTGRES_URL, echo=False)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def postgres_engine():
    """PostgreSQL engine fixture for multi-session and concurrency testing."""
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL service not accessible.")

    engine = create_engine(POSTGRES_URL, pool_size=20, max_overflow=10, echo=False)
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# ============================================================================
# 1. Schema & Index Structure Verification
# ============================================================================


class TestSchemaAndIndexVerification:
    """Verifies all expected indices and composite keys exist across engines."""

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_composite_and_column_indices_created(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )
        inspector = inspect(session.bind)

        # Check evidence_matrix_entries index
        em_indexes = inspector.get_indexes("evidence_matrix_entries")
        em_index_names = {idx["name"] for idx in em_indexes}
        assert "ix_evidence_matrix_project_paper" in em_index_names

        # Check paper_cache indexes
        pc_indexes = inspector.get_indexes("paper_cache")
        pc_column_sets = [set(idx["column_names"]) for idx in pc_indexes]
        assert {"arxiv_id"} in pc_column_sets
        assert {"s2_id"} in pc_column_sets
        assert {"year"} in pc_column_sets

        # Check research_reports indexes
        rr_indexes = inspector.get_indexes("research_reports")
        rr_column_sets = [set(idx["column_names"]) for idx in rr_indexes]
        assert {"project_id"} in rr_column_sets
        assert {"generated_at"} in rr_column_sets


# ============================================================================
# 2. DOI Uniqueness & Primary Key Duplicate Handling
# ============================================================================


class TestDOIUniquenessAndDuplicateHandling:
    """Adversarial stress-testing of DOI uniqueness and error handling across engines."""

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_exact_duplicate_doi_raises_integrity_error(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        doi = f"10.1038/s41586-020-2649-2_{uuid4().hex[:6]}"
        p1 = PaperCache(
            doi=doi,
            title="Language Models are Few-Shot Learners",
            authors=["Tom B. Brown", "Benjamin Mann"],
            year=2020,
            is_full_text=True,
        )
        session.add(p1)
        session.commit()
        session.expunge(p1)

        # Attempt to insert identical DOI in separate object
        p2 = PaperCache(
            doi=doi,
            title="Duplicate Title",
            authors=["Imposter Author"],
            year=2021,
            is_full_text=False,
        )
        session.add(p2)
        with pytest.raises(IntegrityError):
            session.commit()

        # Verify rollback restores session health
        session.rollback()

        # Original paper should remain unaffected
        retrieved = session.get(PaperCache, doi)
        assert retrieved is not None
        assert retrieved.title == "Language Models are Few-Shot Learners"
        assert retrieved.authors == ["Tom B. Brown", "Benjamin Mann"]

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_doi_session_merge_upsert_behavior(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        doi = f"10.1145/3377325.3377498_{uuid4().hex[:6]}"
        p1 = PaperCache(
            doi=doi,
            title="Initial Title",
            authors=["Author A"],
            year=2023,
            is_full_text=False,
        )
        session.add(p1)
        session.commit()
        session.expunge(p1)

        # Update via session.merge
        p1_updated = PaperCache(
            doi=doi,
            title="Updated Title with Full Text",
            authors=["Author A", "Author B"],
            year=2023,
            parsed_markdown="# Heading 1\n\nResolved body.",
            is_full_text=True,
        )
        merged = session.merge(p1_updated)
        session.commit()

        retrieved = session.get(PaperCache, doi)
        assert retrieved is not None
        assert retrieved.title == "Updated Title with Full Text"
        assert retrieved.authors == ["Author A", "Author B"]
        assert retrieved.is_full_text is True
        assert retrieved.parsed_markdown == "# Heading 1\n\nResolved body."

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_non_doi_primary_key_variants(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        keys = [
            f"arxiv:2401.00001_{uuid4().hex[:6]}",
            f"s2:corpusId_987654321_{uuid4().hex[:6]}",
            f"pubmed:38291024_{uuid4().hex[:6]}",
            f"openalex:W4285719283_{uuid4().hex[:6]}",
        ]

        for k in keys:
            paper = PaperCache(
                doi=k,
                title=f"Paper with key {k}",
                authors=["Test Author"],
                is_full_text=False,
            )
            session.add(paper)
        session.commit()

        for k in keys:
            retrieved = session.get(PaperCache, k)
            assert retrieved is not None
            assert retrieved.title == f"Paper with key {k}"


# ============================================================================
# 3. Multi-Field JSON Serialization & Roundtrip Fidelity
# ============================================================================


class TestMultiFieldJSONSerialization:
    """Stress-testing JSON fields with complex nested structures, unicode, and large payloads."""

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_paper_cache_nested_json_roundtrip(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        complex_authors = ["François Chollet", "Yoshua Bengio (ベンジオ)", "李飞飞 (Fei-Fei Li)"]
        complex_sections = [
            {
                "anchor": "[ref_paper1#sec_1]",
                "category": "METHODOLOGY",
                "title": "3.1 Scaled Dot-Product Attention",
                "math_blocks": [r"$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$"],
                "subsections": [
                    {"sub_id": "3.1.1", "content": "Multi-head attention mechanisms."}
                ],
            },
            {
                "anchor": "[ref_paper1#sec_2]",
                "category": "RESULTS",
                "title": "4.2 Translation Benchmarks",
                "metrics": {"BLEU": 28.4, "ROUGE-L": 45.2, "latency_ms": 12.5},
            },
        ]
        complex_tables = [
            {
                "anchor": "[ref_paper1#tab_1]",
                "caption": "Table 1: Comparison of SOTA models on WMT 2014 English-to-German",
                "headers": ["Model", "BLEU (EN-DE)", "BLEU (EN-FR)", "Training Cost (FLOPs)"],
                "rows": [
                    ["Transformer (base model)", "27.3", "38.1", "3.3e18"],
                    ["Transformer (big model)", "28.4", "41.8", "2.3e19"],
                ],
                "numeric_meta": {"total_models_evaluated": 12, "significance_p": 0.001},
            }
        ]

        doi = f"10.5555/json.complex.{uuid4().hex[:8]}"
        paper = PaperCache(
            doi=doi,
            title="Complex JSON Test Paper",
            authors=complex_authors,
            year=2024,
            sections_json=complex_sections,
            tables_json=complex_tables,
            is_full_text=True,
        )
        session.add(paper)
        session.commit()
        session.expunge(paper)

        retrieved = session.get(PaperCache, doi)
        assert retrieved is not None
        assert retrieved.authors == complex_authors
        assert retrieved.sections_json == complex_sections
        assert retrieved.tables_json == complex_tables
        assert retrieved.sections_json[0]["math_blocks"][0].startswith("$$\\text{Attention}")
        assert retrieved.tables_json[0]["rows"][1][0] == "Transformer (big model)"

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_research_report_nested_json_roundtrip(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(
            email=f"user_{uuid4().hex[:6]}@domain.org",
            name="JSON Tester",
            hashed_password="pw",
        )
        session.add(user)
        session.flush()

        project = ResearchProject(
            user_id=user.id,
            title="Complex Report Project",
            research_question="Evaluating JSON fidelity in report synthesis",
        )
        session.add(project)
        session.flush()

        methodology_overview = {
            "empirical": 0.65,
            "theoretical": 0.20,
            "meta_analysis": 0.15,
            "sub_categories": {
                "neural_methods": 8,
                "symbolic_methods": 2,
                "hybrid": 5,
            },
        }
        thematic_sections = [
            {
                "theme_id": "theme_alpha",
                "title": "Deep Reinforcement Learning in Graph Reasoning",
                "synthesis_prose": "Graph neural agents achieve 95% accuracy [ref_10.1038#sec_1].",
                "key_takeaways": [
                    "Message passing converges in O(log N) iterations.",
                    "Reward shaping prevents policy collapse.",
                ],
                "cited_paper_ids": ["10.1038/nature14539", "arxiv:1706.03762"],
            }
        ]
        conflicts = [
            {
                "topic": "Discrete vs Continuous State Space",
                "perspective_a": "Continuous representations preserve fine-grained gradients.",
                "perspective_b": "Discrete tokens improve combinatorial interpretability.",
                "critical_evaluation": "Empirical evidence favors continuous embeddings with discrete bottleneck layers.",
            }
        ]

        report = ResearchReportModel(
            project_id=project.id,
            title="Comprehensive Graph Reasoning Review",
            executive_summary="Executive synthesis summary...",
            methodology_overview=methodology_overview,
            quality_score=94.2,
            thematic_sections=thematic_sections,
            conflicts_and_debates=conflicts,
        )
        session.add(report)
        session.commit()
        report_id = report.id
        session.expunge(report)

        retrieved = session.get(ResearchReportModel, report_id)
        assert retrieved is not None
        assert retrieved.quality_score == 94.2
        assert retrieved.methodology_overview == methodology_overview
        assert retrieved.thematic_sections == thematic_sections
        assert retrieved.conflicts_and_debates == conflicts
        assert retrieved.thematic_sections[0]["cited_paper_ids"] == [
            "10.1038/nature14539",
            "arxiv:1706.03762",
        ]

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_large_json_payload_stress(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Stress-tests storing and retrieving 250 parsed sections and 50 tables in PaperCache."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        large_sections = [
            {
                "anchor": f"[ref_large#sec_{i}]",
                "category": "RESULTS" if i % 2 == 0 else "METHODOLOGY",
                "title": f"Section {i} Detailed Empirical Analysis",
                "body": "Paragraph " * 50,
                "chunk_index": i,
            }
            for i in range(250)
        ]
        large_tables = [
            {
                "anchor": f"[ref_large#tab_{j}]",
                "caption": f"Table {j} Benchmark Grid",
                "matrix": [[k * 0.1 for k in range(10)] for _ in range(20)],
            }
            for j in range(50)
        ]

        doi = f"10.9999/large.payload.{uuid4().hex[:8]}"
        paper = PaperCache(
            doi=doi,
            title="Large Payload Stress Paper",
            authors=[f"Author {k}" for k in range(50)],
            sections_json=large_sections,
            tables_json=large_tables,
            is_full_text=True,
        )
        session.add(paper)
        session.commit()
        session.expunge(paper)

        retrieved = session.get(PaperCache, doi)
        assert retrieved is not None
        assert len(retrieved.sections_json) == 250
        assert len(retrieved.tables_json) == 50
        assert len(retrieved.authors) == 50
        assert retrieved.sections_json[249]["anchor"] == "[ref_large#sec_249]"

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_research_gap_json_grounding_papers(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@t.edu", name="U", hashed_password="h")
        session.add(user)
        session.flush()
        project = ResearchProject(user_id=user.id, title="P", research_question="Q")
        session.add(project)
        session.flush()

        gap = ResearchGapModel(
            project_id=project.id,
            gap_id="GAP-SEC-01",
            description="Lack of formal verification for multi-agent LLM consensus protocols.",
            importance="high",
            recommended_methodology="Apply TLA+ formal specifications combined with property-based testing.",
            grounding_paper_ids=[
                "10.1145/3377325.3377498",
                "arxiv:2401.99999",
                "doi:10.1016/j.artmed.2023.102450",
            ],
        )
        session.add(gap)
        session.commit()
        gap_id = gap.id
        session.expunge(gap)

        retrieved = session.get(ResearchGapModel, gap_id)
        assert retrieved is not None
        assert retrieved.importance == "high"
        assert len(retrieved.grounding_paper_ids) == 3
        assert "arxiv:2401.99999" in retrieved.grounding_paper_ids


# ============================================================================
# 4. Project Cascade Deletes & Referential Integrity
# ============================================================================


class TestCascadeDeletesAndReferentialIntegrity:
    """Verifies cascading deletions across all child models and isolation of PaperCache."""

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_project_orm_cascade_deletes_all_children(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@cascade.edu", name="C", hashed_password="h")
        session.add(user)
        session.flush()

        project = ResearchProject(
            user_id=user.id, title="Cascade Target Project", research_question="Cascade test"
        )
        session.add(project)
        session.flush()

        # Create child entities across all 5 project-dependent tables
        report = ResearchReportModel(
            project_id=project.id,
            title="Report 1",
            executive_summary="Exec",
            quality_score=85.0,
            thematic_sections=[{"theme": "1"}],
        )
        matrix1 = EvidenceMatrixEntry(
            project_id=project.id,
            paper_id="doi:10.1/paper1",
            title="Paper 1",
            primary_metric="Acc 92%",
        )
        matrix2 = EvidenceMatrixEntry(
            project_id=project.id,
            paper_id="doi:10.1/paper2",
            title="Paper 2",
            primary_metric="Acc 94%",
        )
        gap = ResearchGapModel(
            project_id=project.id,
            gap_id="gap_1",
            description="Gap desc",
            importance="medium",
            recommended_methodology="Method",
        )
        plan = AgentPlan(
            project_id=project.id,
            agent_type="supervisor",
            plan_steps=[{"step": 1}],
        )
        ref = PaperReference(
            project_id=project.id,
            title="Ref 1",
            url="http://test.com",
        )
        session.add_all([report, matrix1, matrix2, gap, plan, ref])

        # Also add global PaperCache row (must NOT be deleted)
        global_doi = f"10.1/paper1_{uuid4().hex[:6]}"
        global_paper = PaperCache(
            doi=global_doi,
            title="Global Cached Paper 1",
            authors=["Author 1"],
            is_full_text=True,
        )
        session.add(global_paper)
        session.commit()

        rep_id = report.id
        m1_id = matrix1.id
        m2_id = matrix2.id
        gap_id = gap.id
        plan_id = plan.id
        ref_id = ref.id
        pid = project.id

        # Confirm all exist
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_id)) is not None
        assert len(session.scalars(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.project_id == pid)).all()) == 2
        assert session.scalar(select(ResearchGapModel).where(ResearchGapModel.id == gap_id)) is not None
        assert session.scalar(select(AgentPlan).where(AgentPlan.id == plan_id)) is not None
        assert session.scalar(select(PaperReference).where(PaperReference.id == ref_id)) is not None
        assert session.get(PaperCache, global_doi) is not None

        # Delete project via ORM
        session.delete(project)
        session.commit()

        # Verify all children are deleted
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_id)) is None
        assert len(session.scalars(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.project_id == pid)).all()) == 0
        assert session.scalar(select(ResearchGapModel).where(ResearchGapModel.id == gap_id)) is None
        assert session.scalar(select(AgentPlan).where(AgentPlan.id == plan_id)) is None
        assert session.scalar(select(PaperReference).where(PaperReference.id == ref_id)) is None

        # Global PaperCache must persist
        cached = session.get(PaperCache, global_doi)
        assert cached is not None
        assert cached.title == "Global Cached Paper 1"

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_raw_sql_ddl_cascade_delete(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Verifies foreign keys ondelete='CASCADE' works directly via raw SQL execution."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@raw.edu", name="Raw", hashed_password="h")
        session.add(user)
        session.flush()

        project = ResearchProject(
            user_id=user.id, title="Raw SQL Project", research_question="Raw SQL cascade test"
        )
        session.add(project)
        session.flush()

        report = ResearchReportModel(
            project_id=project.id,
            title="Raw SQL Report",
            executive_summary="Exec",
        )
        matrix = EvidenceMatrixEntry(
            project_id=project.id,
            paper_id="doi:10.9/raw",
            title="Raw Matrix Entry",
        )
        gap = ResearchGapModel(
            project_id=project.id,
            gap_id="gap_raw",
            description="Raw Gap",
            recommended_methodology="Raw Method",
        )
        session.add_all([report, matrix, gap])
        session.commit()

        rep_id = report.id
        matrix_id = matrix.id
        gap_id = gap.id
        pid = project.id

        # Clear session to prevent holding deleted objects
        session.expunge_all()

        # Delete project via RAW SQL
        session.execute(
            text("DELETE FROM research_projects WHERE id = :pid"), {"pid": pid}
        )
        session.commit()

        # Verify child tables are completely cleaned up by engine DDL cascade
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_id)) is None
        assert session.scalar(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.id == matrix_id)) is None
        assert session.scalar(select(ResearchGapModel).where(ResearchGapModel.id == gap_id)) is None

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_user_cascade_deletes_projects_and_transitive_children(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Verifies User deletion cascades to projects and transitively to reports, matrix, and gaps."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@trans.edu", name="Trans", hashed_password="h")
        session.add(user)
        session.flush()

        project = ResearchProject(user_id=user.id, title="Trans Project", research_question="Q")
        session.add(project)
        session.flush()

        report = ResearchReportModel(project_id=project.id, title="R", executive_summary="E")
        matrix = EvidenceMatrixEntry(project_id=project.id, paper_id="p1", title="M")
        gap = ResearchGapModel(
            project_id=project.id, gap_id="g1", description="D", recommended_methodology="M"
        )
        session.add_all([report, matrix, gap])
        session.commit()

        user_id = user.id
        project_id = project.id
        rep_id = report.id
        mat_id = matrix.id
        gap_id = gap.id

        # Delete User
        session.delete(user)
        session.commit()

        # Check all are gone
        assert session.get(User, user_id) is None
        assert session.get(ResearchProject, project_id) is None
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_id)) is None
        assert session.scalar(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.id == mat_id)) is None
        assert session.scalar(select(ResearchGapModel).where(ResearchGapModel.id == gap_id)) is None

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_bidirectional_relationship_synchronization(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Verifies ORM relationship collections and back_populates synchronize properly."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@rel.edu", name="Rel", hashed_password="h")
        project = ResearchProject(user=user, title="Rel Project", research_question="Q")

        report = ResearchReportModel(title="R", executive_summary="E")
        project.research_reports.append(report)

        matrix = EvidenceMatrixEntry(paper_id="p1", title="M")
        project.evidence_matrix_entries.append(matrix)

        gap = ResearchGapModel(gap_id="g1", description="D", recommended_methodology="M")
        project.research_gaps.append(gap)

        session.add(user)
        session.commit()

        # Check relationships populated in both directions
        assert report.project_id == project.id
        assert report.project is project
        assert matrix.project_id == project.id
        assert matrix.project is project
        assert gap.project_id == project.id
        assert gap.project is project

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_multi_project_isolation(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Verifies deleting Project A does not delete Project B's reports or matrix rows."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        user = User(email=f"u_{uuid4().hex[:6]}@iso.edu", name="Iso", hashed_password="h")
        session.add(user)
        session.flush()

        p_a = ResearchProject(user_id=user.id, title="Project A", research_question="QA")
        p_b = ResearchProject(user_id=user.id, title="Project B", research_question="QB")
        session.add_all([p_a, p_b])
        session.flush()

        rep_a = ResearchReportModel(project_id=p_a.id, title="Report A", executive_summary="EA")
        rep_b = ResearchReportModel(project_id=p_b.id, title="Report B", executive_summary="EB")

        mat_a = EvidenceMatrixEntry(project_id=p_a.id, paper_id="p1", title="M A")
        mat_b = EvidenceMatrixEntry(project_id=p_b.id, paper_id="p2", title="M B")

        session.add_all([rep_a, rep_b, mat_a, mat_b])
        session.commit()

        rep_a_id = rep_a.id
        rep_b_id = rep_b.id
        mat_a_id = mat_a.id
        mat_b_id = mat_b.id

        # Delete Project A
        session.delete(p_a)
        session.commit()

        # Project A's records gone
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_a_id)) is None
        assert session.scalar(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.id == mat_a_id)) is None

        # Project B's records intact
        assert session.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_b_id)) is not None
        assert session.scalar(select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.id == mat_b_id)) is not None

    @pytest.mark.parametrize("engine_type", ["sqlite", "postgres"])
    def test_foreign_key_invalid_project_id_rejected(
        self, engine_type: str, sqlite_memory_session: Session, request
    ):
        """Verifies inserting child model with invalid project_id raises IntegrityError."""
        session = (
            sqlite_memory_session
            if engine_type == "sqlite"
            else request.getfixturevalue("postgres_session")
        )

        invalid_pid = "non-existent-project-uuid"
        report = ResearchReportModel(
            project_id=invalid_pid,
            title="Orphan Report",
            executive_summary="Orphan Exec",
        )
        session.add(report)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


# ============================================================================
# 5. Concurrency & Multi-Session Isolation
# ============================================================================


class TestConcurrencyAndMultiSessionIsolation:
    """Stress-testing multi-session transaction isolation, rollback isolation, and concurrent writes."""

    @pytest.mark.parametrize("engine_type", ["sqlite_file", "postgres"])
    def test_uncommitted_write_isolation_between_sessions(
        self, engine_type: str, sqlite_file_engine, request
    ):
        """Verifies distinct sessions do not experience dirty reads of uncommitted transactions."""
        engine = (
            sqlite_file_engine
            if engine_type == "sqlite_file"
            else request.getfixturevalue("postgres_engine")
        )
        SessionFactory = sessionmaker(bind=engine)

        s1 = SessionFactory()
        s2 = SessionFactory()

        try:
            # S1 creates user and project and commits
            user = User(email=f"u_{uuid4().hex[:6]}@iso.org", name="U", hashed_password="h")
            s1.add(user)
            s1.flush()
            project = ResearchProject(user_id=user.id, title="Iso Project", research_question="Q")
            s1.add(project)
            s1.commit()
            pid = project.id

            # S1 inserts report but DOES NOT COMMIT
            rep = ResearchReportModel(
                project_id=pid,
                title="Uncommitted Report",
                executive_summary="Draft in progress...",
            )
            s1.add(rep)
            s1.flush()
            rep_id = rep.id

            # S2 reads from research_reports -> should NOT see uncommitted report
            s2_result = s2.scalar(select(ResearchReportModel).where(ResearchReportModel.id == rep_id))
            assert s2_result is None, "Session 2 experienced dirty read of uncommitted data!"

            # S1 commits
            s1.commit()

            # Now S2 can see the committed report (after refreshing/new query)
            s2.expire_all()
            s2_committed_result = s2.scalar(
                select(ResearchReportModel).where(ResearchReportModel.id == rep_id)
            )
            assert s2_committed_result is not None
            assert s2_committed_result.title == "Uncommitted Report"

        finally:
            s1.close()
            s2.close()

    @pytest.mark.parametrize("engine_type", ["sqlite_file", "postgres"])
    def test_concurrent_paper_cache_inserts_race_condition(
        self, engine_type: str, sqlite_file_engine, request
    ):
        """Simulates 10 concurrent threads racing to cache the exact same DOI."""
        engine = (
            sqlite_file_engine
            if engine_type == "sqlite_file"
            else request.getfixturevalue("postgres_engine")
        )
        SessionFactory = sessionmaker(bind=engine)

        target_doi = f"10.1038/nature.race.{uuid4().hex[:6]}"

        def attempt_insert(thread_idx: int) -> str:
            session = SessionFactory()
            try:
                paper = PaperCache(
                    doi=target_doi,
                    title=f"Race Paper from Thread {thread_idx}",
                    authors=[f"Thread {thread_idx}"],
                    is_full_text=True,
                )
                session.add(paper)
                session.commit()
                return "SUCCESS"
            except (IntegrityError, OperationalError):
                session.rollback()
                return "INTEGRITY_OR_LOCK_ERROR"
            except Exception as ex:
                session.rollback()
                return f"OTHER_ERROR: {type(ex).__name__}"
            finally:
                session.close()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_insert, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        success_count = results.count("SUCCESS")
        conflict_count = results.count("INTEGRITY_OR_LOCK_ERROR")

        # Exactly 1 thread must succeed; 9 must fail cleanly with constraint or lock error
        assert success_count == 1, f"Expected exactly 1 success, got {success_count} ({results})"
        assert conflict_count == 9, f"Expected 9 conflicts, got {conflict_count} ({results})"

        # Verify only 1 row exists in database
        verify_session = SessionFactory()
        try:
            cached_papers = verify_session.scalars(
                select(PaperCache).where(PaperCache.doi == target_doi)
            ).all()
            assert len(cached_papers) == 1
        finally:
            verify_session.close()

    @pytest.mark.parametrize("engine_type", ["sqlite_file", "postgres"])
    def test_concurrent_matrix_and_gap_writes_same_project(
        self, engine_type: str, sqlite_file_engine, request
    ):
        """Simulates 10 concurrent threads writing matrix entries and gaps to the same project."""
        engine = (
            sqlite_file_engine
            if engine_type == "sqlite_file"
            else request.getfixturevalue("postgres_engine")
        )
        SessionFactory = sessionmaker(bind=engine)

        init_session = SessionFactory()
        user = User(email=f"u_{uuid4().hex[:6]}@conc.edu", name="Conc", hashed_password="h")
        init_session.add(user)
        init_session.flush()
        project = ResearchProject(user_id=user.id, title="Concurrent Project", research_question="Q")
        init_session.add(project)
        init_session.commit()
        project_id = project.id
        init_session.close()

        def worker_write_entry(idx: int) -> bool:
            session = SessionFactory()
            try:
                matrix_entry = EvidenceMatrixEntry(
                    project_id=project_id,
                    paper_id=f"paper_id_{idx}",
                    title=f"Title {idx}",
                    primary_metric=f"F1: {0.80 + idx * 0.01:.2f}",
                )
                gap_entry = ResearchGapModel(
                    project_id=project_id,
                    gap_id=f"gap_{idx}",
                    description=f"Research Gap {idx}",
                    importance="high" if idx % 2 == 0 else "medium",
                    recommended_methodology=f"Methodology {idx}",
                )
                session.add_all([matrix_entry, gap_entry])
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Worker {idx} failed: {e}")
                return False
            finally:
                session.close()

        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_write_entry, i) for i in range(num_workers)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results), f"Not all concurrent writes completed successfully: {results}"

        # Verify all matrix entries and gap entries exist
        verify_session = SessionFactory()
        try:
            matrix_count = len(
                verify_session.scalars(
                    select(EvidenceMatrixEntry).where(EvidenceMatrixEntry.project_id == project_id)
                ).all()
            )
            gap_count = len(
                verify_session.scalars(
                    select(ResearchGapModel).where(ResearchGapModel.project_id == project_id)
                ).all()
            )
            assert matrix_count == num_workers, f"Expected {num_workers} matrix entries, found {matrix_count}"
            assert gap_count == num_workers, f"Expected {num_workers} gaps, found {gap_count}"
        finally:
            verify_session.close()

    @pytest.mark.parametrize("engine_type", ["sqlite_file", "postgres"])
    def test_concurrent_read_while_write_stability(
        self, engine_type: str, sqlite_file_engine, request
    ):
        """Simulates simultaneous reads and updates on ResearchReportModel."""
        engine = (
            sqlite_file_engine
            if engine_type == "sqlite_file"
            else request.getfixturevalue("postgres_engine")
        )
        SessionFactory = sessionmaker(bind=engine)

        init_session = SessionFactory()
        user = User(email=f"u_{uuid4().hex[:6]}@rww.edu", name="RWW", hashed_password="h")
        init_session.add(user)
        init_session.flush()
        project = ResearchProject(user_id=user.id, title="RWW Project", research_question="Q")
        init_session.add(project)
        init_session.flush()
        report = ResearchReportModel(
            project_id=project.id,
            title="Initial Report Title",
            executive_summary="Initial Exec",
            quality_score=70.0,
        )
        init_session.add(report)
        init_session.commit()
        report_id = report.id
        init_session.close()

        def writer_task(iteration: int):
            session = SessionFactory()
            try:
                rep = session.get(ResearchReportModel, report_id)
                if rep:
                    rep.quality_score = 70.0 + iteration
                    rep.executive_summary = f"Updated summary iteration {iteration}"
                    session.commit()
                return True
            except Exception:
                session.rollback()
                return False
            finally:
                session.close()

        def reader_task():
            session = SessionFactory()
            try:
                rep = session.get(ResearchReportModel, report_id)
                if rep:
                    assert rep.quality_score >= 70.0
                return True
            except Exception:
                return False
            finally:
                session.close()

        with ThreadPoolExecutor(max_workers=8) as executor:
            writer_futures = [executor.submit(writer_task, i) for i in range(5)]
            reader_futures = [executor.submit(reader_task) for _ in range(10)]
            all_results = [f.result() for f in as_completed(writer_futures + reader_futures)]

        assert all(all_results), "Concurrent read/write encountered unhandled exceptions"
