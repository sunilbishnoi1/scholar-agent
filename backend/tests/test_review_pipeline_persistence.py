"""
Literature Review Pipeline Persistence and REST API Verification Test Suite.
Focus:
- Database persistence in backend/main.py and Celery run_literature_review task logic
- Verification of ResearchReportModel, EvidenceMatrixEntry, ResearchGapModel, PaperReference, and PaperCache queryability via REST API
- Authorization & data isolation across users
- Error handling, fallback mechanisms, cascade deletion, and concurrency
- WebSocket real-time streaming endpoint verification
"""

import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

TEST_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./test_api.db")

from backend.agents.blackboard import WorkingMemoryBlackboard
from backend.agents.schemas import (
    BibliographyItem,
    CitationAuditReport,
    ConflictingDebate,
    CriticEvaluation,
    EvidenceMatrixRow,
    MethodologyDistribution,
    ReportMetadata,
    ReportStatus,
    ResearchGapItem,
    ResearchReport,
    ThematicSection,
)
from backend.auth import create_access_token, get_password_hash
from backend.db import get_db
from backend.main import app, celery_app, run_literature_review
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
)


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test_challenger2.db"):
        try:
            os.remove("test_challenger2.db")
        except Exception:
            pass


@pytest.fixture
def db_session(engine):
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def test_client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def users_fixture(db_session):
    uid1 = uuid.uuid4().hex[:8]
    uid2 = uuid.uuid4().hex[:8]
    u1 = User(
        id=f"user-1-{uid1}",
        email=f"u1_{uid1}@example.com",
        name="User One",
        hashed_password=get_password_hash("pass123"),
        tier="pro",
        monthly_budget_usd=10.0,
    )
    u2 = User(
        id=f"user-2-{uid2}",
        email=f"u2_{uid2}@example.com",
        name="User Two",
        hashed_password=get_password_hash("pass456"),
        tier="free",
        monthly_budget_usd=1.0,
    )
    db_session.add_all([u1, u2])
    db_session.commit()

    token1 = create_access_token(data={"sub": u1.id, "email": u1.email})
    token2 = create_access_token(data={"sub": u2.id, "email": u2.email})

    return {
        "user1": u1,
        "token1": token1,
        "headers1": {"Authorization": f"Bearer {token1}"},
        "user2": u2,
        "token2": token2,
        "headers2": {"Authorization": f"Bearer {token2}"},
    }


class TestCeleryExecutionAndDBPersistence:
    """Empirically test Celery task run_literature_review execution logic & persistence."""

    def test_celery_task_successful_execution_and_persistence(self, db_session, users_fixture):
        """Verify that run_literature_review writes all models and updates project state."""
        user = users_fixture["user1"]
        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Transformer Interpretability & Attention",
            research_question="How do attention heads encode syntactic dependencies?",
            keywords=["transformers", "attention", "interpretability"],
            subtopics=["Syntactic Probing", "Induction Heads"],
            status="created",
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        mock_final_state = {
            "status": "completed",
            "total_papers_found": 2,
            "candidate_papers": [
                {
                    "paper_id": "paper-1",
                    "title": "What Does BERT Look At?",
                    "authors": ["Kevin Clark", "Urvashi Khandelwal"],
                    "abstract": "We analyze attention maps in BERT.",
                    "url": "https://arxiv.org/abs/1906.04341",
                    "relevance_score": 0.95,
                    "doi": "10.18653/v1/W19-4828",
                    "arxiv_id": "1906.04341",
                    "is_full_text": True,
                    "sections": [{"heading": "Introduction", "content": "BERT analysis.", "section_index": 1}],
                    "tables": [],
                },
                {
                    "paper_id": "paper-2",
                    "title": "In-context Learning and Induction Heads",
                    "authors": ["Catherine Olsson", "Nelson Elhage"],
                    "abstract": "Induction heads explain in-context learning.",
                    "url": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
                    "relevance_score": 0.91,
                    "doi": "10.48550/arXiv.2209.11895",
                    "arxiv_id": "2209.11895",
                    "is_full_text": True,
                    "sections": [{"heading": "Induction Heads", "content": "Circuit mechanism.", "section_index": 1}],
                    "tables": [],
                },
            ],
            "analyzed_papers": [
                {
                    "paper_id": "paper-1",
                    "title": "What Does BERT Look At?",
                    "authors": ["Kevin Clark", "Urvashi Khandelwal"],
                    "abstract": "We analyze attention maps in BERT.",
                    "url": "https://arxiv.org/abs/1906.04341",
                    "relevance_score": 0.95,
                },
                {
                    "paper_id": "paper-2",
                    "title": "In-context Learning and Induction Heads",
                    "authors": ["Catherine Olsson", "Nelson Elhage"],
                    "abstract": "Induction heads explain in-context learning.",
                    "url": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
                    "relevance_score": 0.91,
                },
            ],
            "evidence_matrix": [
                EvidenceMatrixRow(
                    paper_id="paper-1",
                    title="What Does BERT Look At?",
                    methodology="Attention Weight Probing",
                    benchmark_dataset="Penn Treebank",
                    primary_metric="Dependency Accuracy 86.8%",
                    primary_limitation="Correlational not causal",
                ),
                EvidenceMatrixRow(
                    paper_id="paper-2",
                    title="In-context Learning and Induction Heads",
                    methodology="Causal Activation Patching",
                    benchmark_dataset="Synthetic Induction Sequences",
                    primary_metric="Prefix Matching Loss Reduction 42%",
                    primary_limitation="Focused primarily on small toy models",
                ),
            ],
            "thematic_sections": [
                ThematicSection(
                    theme_id="theme_syntax",
                    title="Syntactic Attention Patterns",
                    synthesis_prose="Attention heads specifically track direct object relations [ref_1#sec2].",
                    cited_paper_ids=["paper-1"],
                    key_takeaways=["Attention maps correlate with dependency parse trees."],
                )
            ],
            "conflicting_debates": [
                ConflictingDebate(
                    topic="Attention as Explanation",
                    perspective_a="Attention weights faithfully reflect internal reasoning.",
                    perspective_b="Attention is not explanation; adversarial weights achieve identical outputs.",
                    critical_evaluation="Causal interventions are necessary beyond pure attention inspection.",
                )
            ],
            "research_gaps": [
                ResearchGapItem(
                    gap_id="GAP-SYN-01",
                    description="Lack of causal mechanistic proofs on 70B+ scale LLMs.",
                    importance="high",
                    recommended_methodology="Sparse Autoencoder attribution caching on Llama 3 70B.",
                    grounding_paper_ids=["paper-1", "paper-2"],
                )
            ],
            "methodology_overview": MethodologyDistribution(
                distribution={"Probing": 1, "Causal Interventions": 1},
                dominant_approach="Mechanistic Interpretability",
                trend_description="Shift from passive weight inspection to active causal patching.",
            ),
            "final_report": {
                "title": "Mechanistic Interpretability of Transformer Attention",
                "executive_summary": "Comprehensive synthesis of transformer attention mechanisms and circuit discovery.",
                "thematic_sections": [
                    {
                        "theme_id": "theme_syntax",
                        "title": "Syntactic Attention Patterns",
                        "synthesis_prose": "Attention heads specifically track direct object relations.",
                        "cited_paper_ids": ["paper-1"],
                    }
                ],
                "conflicts_and_debates": [
                    {
                        "topic": "Attention as Explanation",
                        "critical_evaluation": "Causal interventions are necessary.",
                    }
                ],
            },
            "synthesis": "# Mechanistic Interpretability\n\nExecutive summary of attention circuits.",
        }

        with patch("agents.orchestrator.ScholarAgentOrchestrator") as MockOrchestrator, \
             patch("backend.main.ScholarAgentOrchestrator", new=MockOrchestrator), \
             patch("main.send_completion_email") as mock_email:
            orchestrator_instance = MagicMock()
            
            def mock_init(llm_client=None, db_session=None, progress_callback=None):
                orchestrator_instance.db_session = db_session
                return orchestrator_instance

            MockOrchestrator.side_effect = mock_init

            def mock_run_sync(*args, **kwargs):
                db_session_passed = orchestrator_instance.db_session
                bb = WorkingMemoryBlackboard(
                    project_id=project_id,
                    user_id=user.id,
                    title=project.title,
                    research_question=project.research_question,
                )
                for p in mock_final_state["candidate_papers"]:
                    bb.add_parsed_paper(p)
                bb.set_evidence_matrix(mock_final_state["evidence_matrix"])
                bb.set_thematic_synthesis(
                    executive_summary=mock_final_state["final_report"]["executive_summary"],
                    sections=mock_final_state["thematic_sections"],
                    debates=mock_final_state["conflicting_debates"],
                    gaps=mock_final_state["research_gaps"],
                    methodology_overview=mock_final_state["methodology_overview"],
                )
                bb.add_critic_evaluation({"overall_score": 91.5, "passed": True, "weaknesses": []})
                bb.sync_to_database(db_session_passed)
                return mock_final_state

            orchestrator_instance.run_sync.side_effect = mock_run_sync

            result = run_literature_review(project_id, max_papers=10)

            assert result["status"] == "completed"
            assert result["papers_analyzed"] == 2

        # Verify DB Persistence directly in a fresh session
        verify_session = db_session
        verify_session.rollback()
        updated_project = verify_session.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        assert updated_project is not None
        assert updated_project.status == "completed"
        assert updated_project.report_status == "complete"
        assert updated_project.total_papers_found == 2
        assert updated_project.report is not None

        # Verify ResearchReportModel
        report_model = verify_session.query(ResearchReportModel).filter(ResearchReportModel.project_id == project_id).first()
        assert report_model is not None
        assert report_model.quality_score == 91.5
        assert len(report_model.thematic_sections) == 1
        assert len(report_model.conflicts_and_debates) == 1

        # Verify EvidenceMatrixEntry
        matrix_rows = db_session.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == project_id).all()
        assert len(matrix_rows) == 2
        paper_ids = {r.paper_id for r in matrix_rows}
        assert "paper-1" in paper_ids
        assert "paper-2" in paper_ids

        # Verify ResearchGapModel
        gap_rows = db_session.query(ResearchGapModel).filter(ResearchGapModel.project_id == project_id).all()
        assert len(gap_rows) == 1
        assert gap_rows[0].gap_id == "GAP-SYN-01"
        assert gap_rows[0].importance == "high"

        # Verify PaperReference
        refs = db_session.query(PaperReference).filter(PaperReference.project_id == project_id).all()
        assert len(refs) == 2
        titles = {r.title for r in refs}
        assert "What Does BERT Look At?" in titles

        # Verify PaperCache
        cache_entry1 = db_session.query(PaperCache).filter(PaperCache.doi == "10.18653/v1/W19-4828").first()
        assert cache_entry1 is not None
        assert cache_entry1.title == "What Does BERT Look At?"
        assert cache_entry1.is_full_text is True

        # Verify AgentPlans created
        plans = db_session.query(AgentPlan).filter(AgentPlan.project_id == project_id).all()
        agent_types = {p.agent_type for p in plans}
        assert "synthesizer" in agent_types

    def test_celery_task_project_not_found(self, db_session):
        """Verify Celery task gracefully returns error when project does not exist."""
        result = run_literature_review("nonexistent-project-id", max_papers=10)
        assert result["status"] == "error"
        assert "Project not found" in result["error"]

    def test_celery_task_error_recovery(self, db_session, users_fixture):
        """Verify Celery task handles unexpected exception during orchestrator execution."""
        user = users_fixture["user1"]
        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Fault Tolerant AI",
            research_question="How to handle pipeline failures?",
            status="created",
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        with patch("agents.orchestrator.ScholarAgentOrchestrator") as MockOrchestrator, \
             patch("backend.main.ScholarAgentOrchestrator", new=MockOrchestrator):
            orchestrator_instance = MagicMock()
            orchestrator_instance.run_sync.side_effect = RuntimeError("Simulated LLM rate limit explosion")
            MockOrchestrator.return_value = orchestrator_instance

            with pytest.raises(RuntimeError, match="Simulated LLM rate limit explosion"):
                run_literature_review(project_id, max_papers=5)

        db_session.expire_all()
        failed_project = db_session.query(ResearchProject).filter(ResearchProject.id == project_id).first()
        assert failed_project is not None
        assert failed_project.status == "error"


class TestRESTEndpointsEmpiricalVerification:
    """Empirically verify that all models are queryable via REST API."""

    def test_query_report_matrix_gaps_sections_endpoints(self, test_client, db_session, users_fixture):
        """Full end-to-end REST queryability test for all report and evidence models."""
        user1 = users_fixture["user1"]
        headers1 = users_fixture["headers1"]

        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user1.id,
            title="Neural Information Retrieval 2026",
            research_question="Are dense embeddings superior to BM25 in cross-domain transfer?",
            keywords=["dense retrieval", "BM25", "domain transfer"],
            status="completed",
            report_status="complete",
            total_papers_found=5,
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        # 1. Add ResearchReportModel
        report_model = ResearchReportModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            title="Neural Information Retrieval 2026",
            executive_summary="Dense retrieval achieves high in-domain MRR but degrades under out-of-domain shift without hybrid sparse indexing.",
            methodology_overview={"distribution": {"Dense Bi-encoders": 3, "Sparse Splade": 2}, "dominant_approach": "Hybrid Retrieval"},
            quality_score=94.0,
            thematic_sections=[
                {
                    "theme_id": "theme_generalization",
                    "title": "Cross-Domain Generalization",
                    "synthesis_prose": "SPLADE demonstrates 15% higher nDCG@10 on BEIR compared to ColBERT without fine-tuning.",
                    "cited_paper_ids": ["paper-beir-01"],
                }
            ],
            conflicts_and_debates=[
                {
                    "topic": "Dense vs Sparse Efficiency",
                    "perspective_a": "Dense indices require HNSW graph storage and higher RAM footprint.",
                    "perspective_b": "Sparse inverted indices allow inverted list pruning at lower RAM cost.",
                    "critical_evaluation": "Quantized dense indices narrow the memory gap.",
                }
            ],
            generated_at=datetime.now(timezone.utc),
        )
        db_session.add(report_model)

        # 2. Add EvidenceMatrixEntry records
        matrix_1 = EvidenceMatrixEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            paper_id="paper-beir-01",
            title="BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation",
            methodology_type="Empirical Benchmarking",
            benchmark_dataset="BEIR (18 datasets)",
            primary_metric="nDCG@10 0.428",
            primary_limitation="Excludes multilingual transfer",
            created_at=datetime.now(timezone.utc),
        )
        matrix_2 = EvidenceMatrixEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            paper_id="paper-splade-02",
            title="SPLADE v2: Sparse Lexical and Expansion Model",
            methodology_type="Sparse Neural Expansion",
            benchmark_dataset="MS MARCO Passage",
            primary_metric="MRR@10 0.383",
            primary_limitation="High latency during query expansion",
            created_at=datetime.now(timezone.utc),
        )
        db_session.add_all([matrix_1, matrix_2])

        # 3. Add ResearchGapModel records
        gap_1 = ResearchGapModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            gap_id="GAP-IR-01",
            description="Lack of low-latency multilingual cross-encoder rerankers on edge hardware.",
            importance="high",
            recommended_methodology="ONNX quantization + DistilBERT multilingual distillation.",
            grounding_paper_ids=["paper-beir-01", "paper-splade-02"],
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(gap_1)

        # 4. Add PaperCache records
        cache_paper = PaperCache(
            doi="10.1007/s10791-021-09401-4",
            arxiv_id="2104.08663",
            s2_id="s2-beir-999",
            title="BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation",
            authors=["Nandan Thakur", "Nils Reimers"],
            year=2021,
            venue="NeurIPS",
            abstract="BEIR is a robust evaluation benchmark.",
            is_full_text=True,
            sections_json=[
                {"heading": "Abstract", "content": "BEIR overview...", "section_index": 0},
                {"heading": "1. Introduction", "content": "Zero-shot IR challenge...", "section_index": 1},
                {"heading": "2. Benchmark Datasets", "content": "Details on 18 datasets...", "section_index": 2},
            ],
            tables_json=[{"id": "tab1", "caption": "Evaluation Results on BEIR"}],
            source_url="https://arxiv.org/abs/2104.08663",
            fetched_at=datetime.now(timezone.utc),
        )
        db_session.add(cache_paper)

        # 5. Add PaperReferences
        ref = PaperReference(
            id="ref-legacy-01",
            project_id=project_id,
            title="BM25 Baseline in Modern IR",
            authors=["Stephen Robertson"],
            abstract="Classic probabilistic information retrieval.",
            url="https://doi.org/10.1561/1500000019",
            relevance_score=0.88,
        )
        ref2 = PaperReference(
            id="ref-beir-02",
            project_id=project_id,
            title="BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation",
            authors=["Nandan Thakur", "Nils Reimers"],
            abstract="BEIR is a robust evaluation benchmark.",
            url="https://arxiv.org/abs/2104.08663",
            relevance_score=0.95,
        )
        db_session.add_all([ref, ref2])

        db_session.commit()

        # Test GET /api/projects/{project_id}/report
        res_report = test_client.get(f"/api/projects/{project_id}/report", headers=headers1)
        assert res_report.status_code == 200
        report_json = res_report.json()
        assert report_json["project_id"] == project_id
        assert report_json["report_status"] == "complete"
        rep_content = report_json["report"]
        assert rep_content["title"] == "Neural Information Retrieval 2026"
        assert rep_content["metadata"]["quality_score"] == 94.0
        assert len(rep_content["comparative_matrix"]) == 2
        assert len(rep_content["actionable_gaps"]) == 1
        assert len(rep_content["thematic_sections"]) == 1
        assert len(rep_content["conflicting_debates"]) == 1
        assert len(rep_content["bibliography"]) == 2

        # Test GET /api/projects/{project_id}/matrix
        res_matrix = test_client.get(f"/api/projects/{project_id}/matrix", headers=headers1)
        assert res_matrix.status_code == 200
        matrix_json = res_matrix.json()
        assert matrix_json["project_id"] == project_id
        assert matrix_json["count"] == 2
        assert matrix_json["total"] == 2
        assert matrix_json["entries"][0]["paper_id"] == "paper-beir-01"
        assert matrix_json["entries"][0]["benchmark_dataset"] == "BEIR (18 datasets)"
        assert matrix_json["entries"][1]["paper_id"] == "paper-splade-02"
        assert matrix_json["entries"][1]["primary_metric"] == "MRR@10 0.383"

        # Test GET /api/projects/{project_id}/gaps
        res_gaps = test_client.get(f"/api/projects/{project_id}/gaps", headers=headers1)
        assert res_gaps.status_code == 200
        gaps_json = res_gaps.json()
        assert gaps_json["project_id"] == project_id
        assert gaps_json["count"] == 1
        assert gaps_json["gaps"][0]["gap_id"] == "GAP-IR-01"
        assert gaps_json["gaps"][0]["priority"] == "high"
        assert "paper-beir-01" in gaps_json["gaps"][0]["grounding_papers"]

        # Test GET /api/papers/{paper_id}/sections by arXiv ID
        res_sec_arxiv = test_client.get("/api/papers/2104.08663/sections", headers=headers1)
        assert res_sec_arxiv.status_code == 200
        sec_arxiv_json = res_sec_arxiv.json()
        assert sec_arxiv_json["title"] == "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation"
        assert sec_arxiv_json["is_full_text"] is True
        assert len(sec_arxiv_json["sections"]) == 3
        assert len(sec_arxiv_json["tables"]) == 1
        assert sec_arxiv_json["doi"] == "10.1007/s10791-021-09401-4"

        # Test GET /api/papers/{paper_id}/sections by S2 ID
        res_sec_s2 = test_client.get("/api/papers/s2-beir-999/sections", headers=headers1)
        assert res_sec_s2.status_code == 200
        assert res_sec_s2.json()["title"] == "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation"

        # Test GET /api/papers/{paper_id}/sections fallback to PaperReference
        res_sec_ref = test_client.get("/api/papers/ref-legacy-01/sections", headers=headers1)
        assert res_sec_ref.status_code == 200
        sec_ref_json = res_sec_ref.json()
        assert sec_ref_json["title"] == "BM25 Baseline in Modern IR"
        assert sec_ref_json["is_full_text"] is False
        assert sec_ref_json["sections"][0]["heading"] == "Abstract"

    def test_user_data_isolation_and_authorization(self, test_client, db_session, users_fixture):
        """Verify that User 2 CANNOT access User 1's project report, matrix, or gaps."""
        user1 = users_fixture["user1"]
        headers2 = users_fixture["headers2"]

        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user1.id,
            title="Private Research Project",
            research_question="Confidential research topic",
            status="completed",
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        # User 2 tries to access User 1's endpoints -> must return 404
        res_rep = test_client.get(f"/api/projects/{project_id}/report", headers=headers2)
        assert res_rep.status_code == 404
        assert "Project not found" in res_rep.json()["detail"]

        res_mat = test_client.get(f"/api/projects/{project_id}/matrix", headers=headers2)
        assert res_mat.status_code == 404

        res_gap = test_client.get(f"/api/projects/{project_id}/gaps", headers=headers2)
        assert res_gap.status_code == 404

    def test_unauthenticated_requests_fail(self, test_client, db_session, users_fixture):
        """Verify endpoints reject unauthenticated requests with 401."""
        user1 = users_fixture["user1"]
        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user1.id,
            title="Unauth Check Project",
            research_question="Question?",
            status="completed",
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        # No auth headers
        assert test_client.get(f"/api/projects/{project_id}/report").status_code == 401
        assert test_client.get(f"/api/projects/{project_id}/matrix").status_code == 401
        assert test_client.get(f"/api/projects/{project_id}/gaps").status_code == 401
        assert test_client.get("/api/papers/2301.0001/sections").status_code == 401

    def test_fallback_when_only_project_report_json_exists(self, test_client, db_session, users_fixture):
        """Verify that /report, /matrix, and /gaps correctly fall back to project.report JSON if relational tables empty."""
        user1 = users_fixture["user1"]
        headers1 = users_fixture["headers1"]

        raw_report = {
            "title": "Fallback Literature Synthesis",
            "comparative_matrix": [
                {
                    "paper_id": "fallback-p1",
                    "title": "Fallback Paper Title",
                    "methodology": "Survey",
                    "dataset": "Benchmark-1",
                    "primary_metric": "Accuracy 95%",
                    "primary_limitation": "Small sample size",
                }
            ],
            "actionable_gaps": [
                {
                    "gap_id": "GAP-FB-01",
                    "title": "Missing large scale evaluation",
                    "description": "Need evaluation across 100+ datasets",
                    "importance": "high",
                    "recommended_methodology": "Distributed test harness",
                    "grounding_paper_ids": ["fallback-p1"],
                }
            ],
        }

        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user1.id,
            title="Fallback Test Project",
            research_question="Fallback test question?",
            status="completed",
            report_status="complete",
            report=raw_report,
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        # Test /report fallback
        res_rep = test_client.get(f"/api/projects/{project_id}/report", headers=headers1)
        assert res_rep.status_code == 200
        assert res_rep.json()["report"]["title"] == "Fallback Literature Synthesis"

        # Test /matrix fallback
        res_mat = test_client.get(f"/api/projects/{project_id}/matrix", headers=headers1)
        assert res_mat.status_code == 200
        mat_data = res_mat.json()
        assert mat_data["count"] == 1
        assert mat_data["entries"][0]["paper_id"] == "fallback-p1"
        assert mat_data["entries"][0]["methodology"] == "Survey"

        # Test /gaps fallback
        res_gap = test_client.get(f"/api/projects/{project_id}/gaps", headers=headers1)
        assert res_gap.status_code == 200
        gap_data = res_gap.json()
        assert gap_data["count"] == 1
        assert gap_data["gaps"][0]["gap_id"] == "GAP-FB-01"

    def test_cascade_deletion_of_all_related_models(self, test_client, db_session, users_fixture):
        """Verify that DELETE /api/projects/{project_id} deletes all related models."""
        user1 = users_fixture["user1"]
        headers1 = users_fixture["headers1"]

        project = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user1.id,
            title="Delete Me Project",
            research_question="Will I be deleted cleanly?",
            status="completed",
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        report = ResearchReportModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            title="Delete Me Report",
            executive_summary="Summary",
        )
        matrix = EvidenceMatrixEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            paper_id="p-del",
            title="Paper to Delete",
        )
        gap = ResearchGapModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            gap_id="GAP-DEL",
            description="Gap to Delete",
            recommended_methodology="Method",
        )
        plan = AgentPlan(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_type="synthesizer",
            current_step=1,
        )
        ref = PaperReference(
            id=str(uuid.uuid4()),
            project_id=project_id,
            title="Ref to Delete",
        )
        db_session.add_all([report, matrix, gap, plan, ref])
        db_session.commit()

        # Delete project via API
        res_del = test_client.delete(f"/api/projects/{project_id}", headers=headers1)
        assert res_del.status_code == 200
        assert res_del.json()["deleted"] is True

        # Verify DB cascade cleanup
        db_session.expire_all()
        assert db_session.query(ResearchProject).filter(ResearchProject.id == project_id).first() is None
        assert db_session.query(ResearchReportModel).filter(ResearchReportModel.project_id == project_id).first() is None
        assert db_session.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == project_id).first() is None
        assert db_session.query(ResearchGapModel).filter(ResearchGapModel.project_id == project_id).first() is None
        assert db_session.query(AgentPlan).filter(AgentPlan.project_id == project_id).first() is None
        assert db_session.query(PaperReference).filter(PaperReference.project_id == project_id).first() is None


class TestWebSocketStreamingEmpirical:
    """Empirically test WebSocket streaming endpoints and real-time event distribution."""

    def test_websocket_stream_lifecycle_and_heartbeat(self, test_client, users_fixture):
        """Verify WebSocket stream connects, authenticates token, responds to ping/pong, and cleans up."""
        token = users_fixture["token1"]
        project_id = "ws-test-proj-123"

        with test_client.websocket_connect(f"/ws/projects/{project_id}/stream?token={token}") as ws:
            # 1. First message should be connected confirmation
            initial_msg = ws.receive_json()
            assert initial_msg["type"] == "connected"
            assert initial_msg["project_id"] == project_id

            # 2. Test ping / pong heartbeat
            ws.send_text("ping")
            pong_msg = ws.receive_json()
            assert pong_msg["type"] == "pong"
            assert pong_msg["project_id"] == project_id
            assert "timestamp" in pong_msg

    @pytest.mark.asyncio
    async def test_websocket_stream_receives_broadcast_events(self, users_fixture):
        """Verify that broadcasting to project distributes event to connected WebSocket clients."""
        from backend.realtime.events import EventType, create_discovery_started_event, create_matrix_row_added_event
        from backend.realtime.manager import ConnectionManager

        manager = ConnectionManager()
        project_id = "ws-broadcast-proj-456"

        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        # Connect client
        await manager.connect(mock_ws, "user-1", project_id)

        # Broadcast discovery started event
        disc_event = create_discovery_started_event(project_id=project_id, queries=["deep learning", "nlp"])
        await manager.broadcast_to_project(project_id, disc_event.to_dict())

        # Broadcast matrix row added event
        row_event = create_matrix_row_added_event(
            project_id=project_id,
            row={
                "paper_id": "p-1",
                "title": "BERT Architecture",
                "methodology": "Masked Language Modeling",
                "benchmark_dataset": "GLUE Benchmark",
                "primary_metric": "Score 80.5",
                "primary_limitation": "Pre-training cost",
            },
        )
        await manager.broadcast_to_project(project_id, row_event.to_dict())

        # Verify calls: 1 initial connected + 2 broadcasts = 3 calls
        assert mock_ws.send_json.call_count == 3
        sent_payloads = [call.args[0] for call in mock_ws.send_json.call_args_list]

        assert sent_payloads[0]["type"] == "connected"
        assert sent_payloads[1]["type"] == EventType.DISCOVERY_STARTED
        assert sent_payloads[1]["data"]["queries"] == ["deep learning", "nlp"]
        assert sent_payloads[2]["type"] == EventType.MATRIX_ROW_ADDED
        assert sent_payloads[2]["data"]["row"]["paper_id"] == "p-1"

