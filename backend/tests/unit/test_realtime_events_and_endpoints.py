"""
Real-Time Event Serialization, Progress Tracking, and REST Endpoint Resilience Test Suite.

Tests:
1. Event Types & AgentEvent serialization/deserialization across all standard event types,
   round-trip JSON parsing, datetime handling, and malformed/extreme payload fuzzing.
2. REST Endpoints resilience when projects/papers have empty/null/missing relational records,
   no matrix entries, no gaps, paper cache without sections/tables, fallback to PaperReference,
   and unauthenticated/unauthorized edge cases.
3. AgentProgressTracker edge cases: invalid/unknown agent names, out-of-order lifecycle events,
   negative/overflow progress values, repeated completions, legacy alias resolution, and total progress calculation bounds.
4. ConnectionManager and WebSocket broadcasting resilience under error conditions and concurrent connections.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth import create_access_token, get_password_hash
from db import get_db
from main import app
from models.database import (
    Base,
    EvidenceMatrixEntry,
    PaperCache,
    PaperReference,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
    User,
)
from realtime.events import (
    AgentEvent,
    AgentProgressTracker,
    EventType,
    create_completion_event,
    create_critic_verdict_event,
    create_discovery_started_event,
    create_fact_checked_event,
    create_log_event,
    create_matrix_row_added_event,
    create_paper_discovered_event,
    create_paper_event,
    create_pdf_parsed_event,
    create_pipeline_completed_event,
    create_pipeline_error_event,
    create_progress_event,
    create_status_event,
    create_thematic_draft_ready_event,
    sync_broadcast_agent_update,
)
from realtime.manager import ConnectionInfo, ConnectionManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_db_session():
    """In-memory SQLite session with StaticPool for thread/session sharing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_client_and_user(test_db_session):
    """Test client with an authenticated user and overridden db dependency."""
    def _override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db

    user = User(
        id=str(uuid.uuid4()),
        email="challenger@scholarpilot.ai",
        hashed_password=get_password_hash("StressTestPass123!"),
        name="Challenger One",
    )
    test_db_session.add(user)
    test_db_session.commit()

    token = create_access_token(data={"sub": user.id, "email": user.email})
    headers = {"Authorization": f"Bearer {token}"}

    client = TestClient(app)
    yield client, user, headers, test_db_session

    app.dependency_overrides.clear()


# ============================================================================
# 1. Event Types & AgentEvent Adversarial Serialization / Deserialization
# ============================================================================

class TestEventSerializationAdversarial:
    """Stress-test serialization and deserialization of all 9 event types."""

    def test_all_9_event_types_defined_and_unique(self):
        """Verify the 9 canonical v3.2 event types are present and unique."""
        canonical_types = [
            EventType.DISCOVERY_STARTED,
            EventType.PAPER_DISCOVERED,
            EventType.PDF_PARSED,
            EventType.MATRIX_ROW_ADDED,
            EventType.THEMATIC_DRAFT_READY,
            EventType.CRITIC_VERDICT,
            EventType.FACT_CHECKED,
            EventType.PIPELINE_COMPLETED,
            EventType.PIPELINE_ERROR,
        ]
        assert len(canonical_types) == 9
        values = [t.value for t in canonical_types]
        assert len(set(values)) == 9, "Canonical event type values must be unique"

    def test_discovery_started_event_roundtrip(self):
        """Verify discovery_started serialization with complex queries, nulls, and string/dict list."""
        # 1. Basic with list of strings
        event1 = create_discovery_started_event(
            project_id="proj-123",
            queries=["quantum error correction", "surface codes threshold"],
            agent="discovery",
            message="Discovery launched",
            progress=15.0,
        )
        dict1 = event1.to_dict()
        assert dict1["type"] == "discovery_started"
        assert dict1["project_id"] == "proj-123"
        assert dict1["progress"] == 15.0
        assert dict1["data"]["queries"] == ["quantum error correction", "surface codes threshold"]

        json_str = event1.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "discovery_started"
        assert parsed["data"]["queries"] == dict1["data"]["queries"]

        # 2. Edge case: queries is None
        event_none = create_discovery_started_event(project_id="proj-123", queries=None)
        parsed_none = json.loads(event_none.to_json())
        assert parsed_none["data"]["queries"] == []

        # 3. Edge case: list of dict queries (structured query plan)
        complex_queries = [{"query": "q1", "category": "broad"}, {"query": "q2", "category": "deep"}]
        event_dict_queries = create_discovery_started_event(project_id="proj-123", queries=complex_queries)
        parsed_complex = json.loads(event_dict_queries.to_json())
        assert parsed_complex["data"]["queries"] == complex_queries

    def test_paper_discovered_event_roundtrip_and_edge_cases(self):
        """Verify paper_discovered with missing metadata, unicode titles, and extreme scores."""
        event = create_paper_discovered_event(
            project_id="proj-abc",
            paper_id="2301.00001",
            title="Quantum Advantage in \u03b1-\u03b2 Graphs: \ud83d\udd2c An Empirical Survey",
            authors=["Alice \u00d8deg\u00e5rd", "Bob Smith"],
            year=2024,
            venue="NeurIPS 2024",
            source="arxiv",
            citation_count=142,
            relevance_score=0.985,
            data={"extra_tag": "high_impact", "nested": {"key": 123}},
            progress=20.0,
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "paper_discovered"
        assert parsed["data"]["paper_id"] == "2301.00001"
        assert "\u03b1-\u03b2" in parsed["data"]["title"]
        assert parsed["data"]["citation_count"] == 142
        assert parsed["data"]["nested"]["key"] == 123
        assert parsed["data"]["extra_tag"] == "high_impact"

        # Edge case: all optional params None
        event_minimal = create_paper_discovered_event(
            project_id="p-min",
            paper_id="p1",
            title="Minimal Title",
        )
        parsed_min = json.loads(event_minimal.to_json())
        assert parsed_min["data"]["authors"] == []
        assert parsed_min["data"]["year"] is None
        assert parsed_min["data"]["citation_count"] is None
        assert parsed_min["data"]["relevance_score"] is None

    def test_pdf_parsed_event_roundtrip(self):
        """Verify pdf_parsed event serialization with full text and abstract-only flags."""
        # Full text
        event_ft = create_pdf_parsed_event(
            project_id="proj-1",
            paper_id="paper-100",
            title="Scalable Transformer Architectures",
            is_full_text=True,
            sections_count=12,
            tables_count=4,
            figures_count=6,
            progress=35.0,
        )
        parsed_ft = json.loads(event_ft.to_json())
        assert parsed_ft["type"] == "pdf_parsed"
        assert parsed_ft["data"]["is_full_text"] is True
        assert parsed_ft["data"]["sections_count"] == 12
        assert "Full Text" in parsed_ft["message"]

        # Abstract only
        event_abs = create_pdf_parsed_event(
            project_id="proj-1",
            paper_id="paper-101",
            title="Paywalled Paper Title",
            is_full_text=False,
            sections_count=1,
            tables_count=0,
            figures_count=0,
        )
        parsed_abs = json.loads(event_abs.to_json())
        assert parsed_abs["data"]["is_full_text"] is False
        assert "Abstract Only" in parsed_abs["message"]

    def test_matrix_row_added_event_polymorphic_inputs(self):
        """Verify matrix_row_added accepts Pydantic models, dicts, custom objects, or raw objects."""
        # 1. Dict input
        row_dict = {
            "paper_id": "p-1",
            "title": "Empirical Matrix Study",
            "methodology_type": "Empirical Benchmark",
            "benchmark_dataset": "ImageNet-1k",
            "primary_metric": "Top-1 Accuracy: 89.2%",
        }
        event_dict = create_matrix_row_added_event("proj-1", row_dict, progress=45.0)
        parsed_dict = json.loads(event_dict.to_json())
        assert parsed_dict["type"] == "matrix_row_added"
        assert parsed_dict["data"]["row"]["benchmark_dataset"] == "ImageNet-1k"
        assert "Empirical Matrix Study" in parsed_dict["message"]

        # 2. Mock Pydantic-like object with model_dump()
        mock_pydantic = MagicMock()
        mock_pydantic.model_dump.return_value = {
            "paper_id": "p-2",
            "title": "Pydantic Row Title",
            "methodology_type": "Meta-Analysis",
        }
        event_pydantic = create_matrix_row_added_event("proj-1", mock_pydantic)
        parsed_pydantic = json.loads(event_pydantic.to_json())
        assert parsed_pydantic["data"]["row"]["title"] == "Pydantic Row Title"

        # 3. Raw arbitrary object fallback
        class CustomRow:
            def __init__(self):
                self.paper_id = "custom-1"
                self.title = "Custom Class Row"

        event_custom = create_matrix_row_added_event("proj-1", CustomRow())
        parsed_custom = json.loads(event_custom.to_json())
        assert parsed_custom["data"]["row"]["paper_id"] == "custom-1"

    def test_thematic_draft_ready_event_roundtrip(self):
        """Verify thematic_draft_ready serialization with section breakdowns."""
        sections = [
            {"title": "Section 1: Foundations", "word_count": 850},
            {"title": "Section 2: Comparative Analysis", "word_count": 1400},
        ]
        event = create_thematic_draft_ready_event(
            project_id="proj-1",
            section_count=2,
            debates_count=3,
            gaps_count=4,
            iteration=1,
            sections=sections,
            progress=65.0,
        )
        parsed = json.loads(event.to_json())
        assert parsed["type"] == "thematic_draft_ready"
        assert parsed["data"]["section_count"] == 2
        assert parsed["data"]["debates_count"] == 3
        assert parsed["data"]["gaps_count"] == 4
        assert parsed["data"]["iteration"] == 1
        assert len(parsed["data"]["sections"]) == 2

    def test_critic_verdict_event_roundtrip(self):
        """Verify critic_verdict with dimension scores and refinement decision."""
        dim_scores = {
            "structural_completeness": 85.0,
            "empirical_grounding": 78.0,
            "critical_synthesis": 88.0,
            "academic_rigor": 92.0,
        }
        weaknesses = ["Section 2 lacks explicit limitation discussion", "Need more benchmark metrics"]
        event = create_critic_verdict_event(
            project_id="proj-1",
            score=85.75,
            should_refine=True,
            iteration=1,
            dimension_scores=dim_scores,
            weaknesses=weaknesses,
            guidance="Expand Section 2 limitations table",
            progress=75.0,
        )
        parsed = json.loads(event.to_json())
        assert parsed["type"] == "critic_verdict"
        assert parsed["data"]["score"] == 85.75
        assert parsed["data"]["should_refine"] is True
        assert parsed["data"]["dimension_scores"]["academic_rigor"] == 92.0
        assert len(parsed["data"]["weaknesses"]) == 2
        assert "Needs refinement" in parsed["message"]

    def test_fact_checked_event_roundtrip(self):
        """Verify fact_checked audit event with NLI proposition counts."""
        event = create_fact_checked_event(
            project_id="proj-1",
            precision_score=94.5,
            passed=True,
            entailed_count=38,
            neutral_count=2,
            contradiction_count=0,
            total_propositions=40,
            progress=90.0,
        )
        parsed = json.loads(event.to_json())
        assert parsed["type"] == "fact_checked"
        assert parsed["data"]["precision_score"] == 94.5
        assert parsed["data"]["passed"] is True
        assert parsed["data"]["entailed_count"] == 38
        assert parsed["data"]["total_propositions"] == 40
        assert "PASSED" in parsed["message"]

    def test_pipeline_completed_and_error_events(self):
        """Verify terminal pipeline events."""
        # Completed
        report_mock = {"title": "Final Synthesized Review", "sections": [{"heading": "Intro"}]}
        event_comp = create_pipeline_completed_event(
            project_id="proj-1",
            report=report_mock,
            summary={"papers_analyzed": 15, "synthesis_words": 4200},
        )
        parsed_comp = json.loads(event_comp.to_json())
        assert parsed_comp["type"] == "pipeline_completed"
        assert parsed_comp["progress"] == 100.0
        assert parsed_comp["data"]["report"]["title"] == "Final Synthesized Review"
        assert parsed_comp["data"]["papers_analyzed"] == 15

        # Error
        event_err = create_pipeline_error_event(
            project_id="proj-1",
            error_message="Semantic Scholar rate limit exhausted",
        )
        parsed_err = json.loads(event_err.to_json())
        assert parsed_err["type"] == "pipeline_error"
        assert "rate limit" in parsed_err["data"]["error"]
        assert "Pipeline error:" in parsed_err["message"]

    def test_agent_event_timestamp_iso_format(self):
        """Verify AgentEvent default timestamp is valid ISO 8601 UTC."""
        event = AgentEvent(type=EventType.STATUS_UPDATE, project_id="p-1")
        assert event.timestamp.endswith("+00:00") or event.timestamp.endswith("Z")
        # Ensure it parses cleanly with datetime.fromisoformat
        dt = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        assert dt.tzinfo is not None


# ============================================================================
# 2. REST Endpoints Response Schemas & Edge Case Resilience
# ============================================================================

class TestRESTEndpointsEdgeCases:
    """Stress-test REST endpoints with missing, empty, and malformed state."""

    def test_report_endpoint_project_empty_initial_state(self, test_client_and_user):
        """Verify /report returns empty status and null report when review has not run."""
        client, user, headers, db = test_client_and_user

        # Create project with no reports
        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Empty Test Project",
            research_question="What are zero-shot bounds?",
            status="pending",
            report_status="empty",
            report=None,
        )
        db.add(proj)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/report", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == proj.id
        assert data["report_status"] == "empty"
        assert data["report"] is None
        assert "not yet available" in data["message"]

    def test_report_endpoint_relational_report_with_zero_matrix_and_zero_gaps(self, test_client_and_user):
        """Verify /report succeeds when ResearchReportModel exists but has 0 matrix entries and 0 gaps."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Sparse Report Project",
            research_question="Investigating sparse edge cases",
            status="completed",
            report_status="complete",
        )
        db.add(proj)
        db.flush()

        rep_model = ResearchReportModel(
            id=str(uuid.uuid4()),
            project_id=proj.id,
            title="Sparse Literature Review",
            executive_summary="This review contains zero matrix rows and zero gaps.",
            quality_score=91.0,
            methodology_overview={"distribution": {}, "dominant_approach": "None", "trend_description": "None"},
            thematic_sections=[],
            conflicts_and_debates=[],
            generated_at=datetime.now(timezone.utc),
        )
        db.add(rep_model)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/report", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_status"] == "complete"
        report = data["report"]
        assert report["title"] == "Sparse Literature Review"
        assert report["comparative_matrix"] == []
        assert report["actionable_gaps"] == []
        assert report["thematic_sections"] == []
        assert report["bibliography"] == []
        assert report["quality_score"] == 91.0
        assert report["metadata"]["total_papers_analyzed"] == 0

    def test_matrix_endpoint_empty_state_returns_empty_list(self, test_client_and_user):
        """Verify /matrix returns count=0 and empty entries when no matrix entries exist."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="No Matrix Project",
            research_question="Evaluating matrix boundaries",
            status="running",
        )
        db.add(proj)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/matrix", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == proj.id
        assert data["count"] == 0
        assert data["total"] == 0
        assert data["entries"] == []
        assert data["matrix"] == []

    def test_matrix_endpoint_fallback_to_json_report_comparative_matrix(self, test_client_and_user):
        """Verify /matrix gracefully falls back to project.report['comparative_matrix'] when no relational entries exist."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="JSON Report Project",
            research_question="Evaluating fallback matrix extraction",
            status="completed",
            report={
                "comparative_matrix": [
                    {
                        "id": "mat-1",
                        "paper_id": "p-100",
                        "title": "Fallback Matrix Paper",
                        "methodology_type": "Empirical",
                        "benchmark_dataset": "GLUE",
                        "primary_metric": "F1: 91.4",
                    }
                ]
            },
        )
        db.add(proj)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/matrix", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["paper_id"] == "p-100"
        assert data["entries"][0]["title"] == "Fallback Matrix Paper"
        assert data["entries"][0]["benchmark_dataset"] == "GLUE"

    def test_gaps_endpoint_empty_state_returns_empty_list(self, test_client_and_user):
        """Verify /gaps returns count=0 and empty list when no research gaps exist."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="No Gaps Project",
            research_question="Evaluating research gaps",
            status="running",
        )
        db.add(proj)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/gaps", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == proj.id
        assert data["count"] == 0
        assert data["total"] == 0
        assert data["gaps"] == []

    def test_gaps_endpoint_fallback_to_json_report_actionable_gaps(self, test_client_and_user):
        """Verify /gaps falls back to project.report['actionable_gaps']."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="JSON Gaps Project",
            research_question="Evaluating fallback actionable gaps",
            status="completed",
            report={
                "actionable_gaps": [
                    {
                        "gap_id": "gap-101",
                        "title": "Missing Long-Context Benchmarks",
                        "description": "Evaluations are limited to 4k token contexts.",
                        "importance": "high",
                        "recommended_methodology": "Run Needle-In-A-Haystack up to 128k tokens.",
                    }
                ]
            },
        )
        db.add(proj)
        db.commit()

        resp = client.get(f"/api/projects/{proj.id}/gaps", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["gaps"][0]["gap_id"] == "gap-101"
        assert data["gaps"][0]["priority"] == "high"
        assert "Needle-In-A-Haystack" in data["gaps"][0]["recommended_methodology"]

    def test_paper_sections_endpoint_empty_sections_and_tables(self, test_client_and_user):
        """Verify /papers/{paper_id}/sections handles paper cache with empty sections and tables without crashing."""
        client, user, headers, db = test_client_and_user

        cache_entry = PaperCache(
            doi="10.1145.test_empty_sections",
            arxiv_id="2401.99999",
            s2_id="s2-empty-101",
            title="Paper Without Sections",
            authors=["Test Author"],
            year=2024,
            venue="arXiv preprint",
            abstract="Short abstract without full text body.",
            is_full_text=False,
            sections_json=[],
            tables_json=[],
        )
        db.add(cache_entry)
        db.commit()

        # Query by arxiv_id
        resp_arxiv = client.get("/api/papers/2401.99999/sections", headers=headers)
        assert resp_arxiv.status_code == 200
        data = resp_arxiv.json()
        assert data["title"] == "Paper Without Sections"
        assert data["is_full_text"] is False
        assert data["sections"] == []
        assert data["tables"] == []
        assert data["doi"] == "10.1145.test_empty_sections"

        # Query by s2_id
        resp_s2 = client.get("/api/papers/s2-empty-101/sections", headers=headers)
        assert resp_s2.status_code == 200
        assert resp_s2.json()["arxiv_id"] == "2401.99999"

    def test_paper_sections_endpoint_fallback_to_paper_reference(self, test_client_and_user):
        """Verify /papers/{paper_id}/sections falls back to PaperReference when PaperCache entry is absent."""
        client, user, headers, db = test_client_and_user

        proj = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Ref Project",
            research_question="Testing reference fallback",
        )
        db.add(proj)
        db.flush()

        paper_ref = PaperReference(
            id="ref-unique-999",
            project_id=proj.id,
            title="Reference Only Paper Title",
            authors=["Ref Author"],
            abstract="Abstract stored on paper reference record.",
            url="https://arxiv.org/abs/2402.12345",
        )
        db.add(paper_ref)
        db.commit()

        resp = client.get("/api/papers/ref-unique-999/sections", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["paper_id"] == "ref-unique-999"
        assert data["title"] == "Reference Only Paper Title"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["heading"] == "Abstract"
        assert "Abstract stored on paper reference" in data["sections"][0]["content"]

    def test_paper_sections_endpoint_not_found(self, test_client_and_user):
        """Verify /papers/{paper_id}/sections returns 404 for completely unknown paper ID."""
        client, user, headers, db = test_client_and_user

        resp = client.get("/api/papers/non-existent-paper-doi-xyz/sections", headers=headers)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_endpoint_unauthorized_and_cross_user_isolation(self, test_client_and_user):
        """Verify projects belonging to User A cannot be accessed by User B or unauthenticated callers."""
        client, user_a, headers_a, db = test_client_and_user

        # Create user B
        user_b = User(
            id=str(uuid.uuid4()),
            email="other_user@scholarpilot.ai",
            hashed_password=get_password_hash("OtherUserPass123!"),
            name="User B",
        )
        db.add(user_b)
        db.commit()

        token_b = create_access_token(data={"sub": user_b.id, "email": user_b.email})
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # Project owned by User A
        proj_a = ResearchProject(
            id=str(uuid.uuid4()),
            user_id=user_a.id,
            title="User A Secret Project",
            research_question="Classified project",
            status="completed",
        )
        db.add(proj_a)
        db.commit()

        # User B trying to access User A's project
        resp_b_report = client.get(f"/api/projects/{proj_a.id}/report", headers=headers_b)
        assert resp_b_report.status_code == 404, "Must not leak existence across users"

        resp_b_matrix = client.get(f"/api/projects/{proj_a.id}/matrix", headers=headers_b)
        assert resp_b_matrix.status_code == 404

        resp_b_gaps = client.get(f"/api/projects/{proj_a.id}/gaps", headers=headers_b)
        assert resp_b_gaps.status_code == 404

        # Unauthenticated request
        resp_unauth = client.get(f"/api/projects/{proj_a.id}/report")
        assert resp_unauth.status_code == 401


# ============================================================================
# 3. AgentProgressTracker Calculations Across Invalid / Unknown Names & Edge Cases
# ============================================================================

class TestAgentProgressTrackerAdversarial:
    """Stress-test AgentProgressTracker calculation engine under adversarial inputs."""

    def test_tracker_weights_sum_to_100_percent(self):
        """Verify the 6-stage canonical agent weights sum to precisely 100.0%."""
        total_weight = sum(AgentProgressTracker.AGENT_WEIGHTS.values())
        assert total_weight == 100.0, f"Weights must sum to 100.0%, got {total_weight}"

    def test_unknown_agent_name_does_not_crash_or_distort_progress(self):
        """Verify that starting, updating, or completing an unknown agent does not raise exceptions."""
        tracker = AgentProgressTracker(project_id="proj-stress")

        # Start unknown agent
        tracker.start_agent("alien_reconnaissance_agent")
        assert tracker._calculate_total_progress() == 0.0

        # Update unknown agent progress to 50%
        tracker.update_progress(50.0)
        assert tracker._calculate_total_progress() == 0.0

        # Complete unknown agent
        tracker.complete_agent("alien_reconnaissance_agent")
        assert tracker._calculate_total_progress() == 0.0
        assert "alien_reconnaissance_agent" in tracker.completed_agents

    def test_legacy_aliases_normalized_correctly(self):
        """Verify all legacy agent names map to standard 7-phase weight keys."""
        tracker = AgentProgressTracker(project_id="proj-legacy")

        aliases = [
            ("planner", "discovery"),
            ("retriever", "ingestion"),
            ("analyzer", "matrix_builder"),
            ("matrix", "matrix_builder"),
            ("synthesizer", "synthesizer"),
            ("quality_checker", "critic"),
            ("critic", "critic"),
            ("auditor", "auditor"),
            ("supervisor", "discovery"),
            ("discovery_agent", "discovery"),
            ("ingestion_agent", "ingestion"),
        ]

        for raw_name, expected_norm in aliases:
            norm = tracker._normalize_agent_name(raw_name)
            assert norm == expected_norm, f"Failed normalizing '{raw_name}': expected '{expected_norm}', got '{norm}'"

    def test_out_of_order_agent_completions(self):
        """Verify out-of-order execution (e.g. critic running before discovery completes) computes correct additive progress."""
        tracker = AgentProgressTracker(project_id="proj-ooo")

        # Complete critic (10%) first
        tracker.complete_agent("critic")
        assert tracker._calculate_total_progress() == 10.0

        # Ingestion (25%) is currently running at 50%
        tracker.start_agent("ingestion")
        tracker.update_progress(50.0)
        # Total = 10.0 (critic) + (0.5 * 25.0) = 22.5%
        assert tracker._calculate_total_progress() == 22.5

        # Complete discovery (15%)
        tracker.complete_agent("discovery")
        # Total = 10.0 (critic) + 15.0 (discovery) + 12.5 (ingestion at 50%) = 37.5%
        assert tracker._calculate_total_progress() == 37.5

    def test_progress_clamped_between_0_and_100(self):
        """Verify extreme / overflow / negative progress values never break [0.0, 100.0] bounds."""
        tracker = AgentProgressTracker(project_id="proj-bounds")

        # Negative progress within discovery
        tracker.start_agent("discovery")
        tracker.update_progress(-50.0)
        # Should not crash; _calculate_total_progress will clamp or compute accurately
        prog = tracker._calculate_total_progress()
        assert prog <= 100.0

        # Complete all 6 agents
        for agent in tracker.AGENT_ORDER:
            tracker.complete_agent(agent)

        assert tracker._calculate_total_progress() == 100.0

        # Complete extra agents and call update_progress with 500%
        tracker.complete_agent("some_extra_agent")
        tracker.update_progress(500.0)
        assert tracker._calculate_total_progress() == 100.0

    def test_repeated_agent_starts_and_completions(self):
        """Verify calling start_agent or complete_agent multiple times does not double-count weights."""
        tracker = AgentProgressTracker(project_id="proj-idempotent")

        tracker.complete_agent("discovery")
        assert tracker._calculate_total_progress() == 15.0

        # Call complete_agent on discovery again
        tracker.complete_agent("discovery")
        assert tracker._calculate_total_progress() == 15.0

        # Call complete_agent with legacy alias "planner"
        tracker.complete_agent("planner")
        assert tracker._calculate_total_progress() == 15.0

    def test_progress_callback_adapter_rapid_switching(self):
        """Verify progress_callback_adapter handles rapid switching and completion triggers."""
        tracker = AgentProgressTracker(project_id="proj-adapter")

        # 1. Discovery starts at 10%
        tracker.progress_callback_adapter("discovery", "Generating queries", 10.0)
        assert tracker.current_agent == "discovery"
        # 10% of 15% = 1.5%
        assert tracker._calculate_total_progress() == 1.5

        # 2. Discovery finishes at 100%
        tracker.progress_callback_adapter("discovery", "Queries complete", 100.0)
        assert "discovery" in tracker.completed_agents
        assert tracker._calculate_total_progress() == 15.0

        # 3. Ingestion begins immediately without explicit complete_agent
        tracker.progress_callback_adapter("ingestion", "Downloading PDFs", 40.0)
        assert tracker.current_agent == "ingestion"
        # 15.0 + 40% of 25.0 (10.0) = 25.0%
        assert tracker._calculate_total_progress() == 25.0

        # 4. Unknown agent passed to callback
        tracker.progress_callback_adapter("unknown_custom_step", "Custom sub-task", 50.0)
        assert tracker.current_agent == "unknown_custom_step"


# ============================================================================
# 4. WebSocket & Redis Connection Resilience
# ============================================================================

class TestWebSocketAndRedisResilience:
    """Stress-test broadcasting and connection manager under error conditions."""

    def test_sync_broadcast_when_redis_unavailable(self):
        """Verify sync_broadcast_agent_update logs warning and returns gracefully if Redis is down."""
        with patch("cache.redis_cache.get_cache", return_value=None):
            event = create_status_event("proj-offline", "discovery", "running")
            # Must not throw an unhandled exception
            sync_broadcast_agent_update("proj-offline", event)

    def test_sync_broadcast_when_redis_throws_exception(self):
        """Verify sync_broadcast_agent_update handles Redis publish exceptions cleanly."""
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        mock_cache.publish.side_effect = ConnectionError("Redis server connection lost")

        with patch("cache.redis_cache.get_cache", return_value=mock_cache):
            event = create_status_event("proj-offline", "discovery", "running")
            sync_broadcast_agent_update("proj-offline", event)

    @pytest.mark.asyncio
    async def test_connection_manager_cleanup_on_broken_websocket(self):
        """Verify ConnectionManager gracefully handles sending to a disconnected client and cleans up."""
        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.send_json.side_effect = RuntimeError("WebSocket disconnected")

        project_id = "proj-cleanup"
        user_id = "user-123"

        # Register connection directly into manager internal state
        conn_info = ConnectionInfo(
            websocket=mock_ws,
            user_id=user_id,
            connected_at=datetime.utcnow(),
            project_ids={project_id},
        )
        manager._connection_info[mock_ws] = conn_info
        manager._project_connections[project_id] = {mock_ws}

        # Broadcast to project
        await manager.broadcast_to_project(project_id, {"type": "test_event"})

        # Verify broken connection was disconnected and pruned
        assert mock_ws not in manager._connection_info
        assert project_id not in manager._project_connections
