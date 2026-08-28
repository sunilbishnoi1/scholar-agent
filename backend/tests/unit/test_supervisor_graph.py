"""
Unit tests for Supervisor StateGraph DAG & ScholarAgentOrchestrator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.core.supervisor import (
    AutonomousSupervisorAgent,
    build_scholar_agent_graph,
    finalizer_node,
    should_refine_or_finalize,
)
from backend.agents.orchestrator import ScholarAgentOrchestrator
from backend.agents.schemas import AcademicPaperCandidate
from backend.agents.state import create_initial_agent_state
from backend.models.database import Base, ResearchReportModel


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_should_refine_or_finalize_conditions():
    # 1. Low score, iteration 0 -> Refine
    state_refine = create_initial_agent_state(project_id="p1", max_iterations=2)
    state_refine["current_critic_score"] = 60.0
    state_refine["should_refine"] = True
    state_refine["iteration_count"] = 0
    route = should_refine_or_finalize(state_refine)
    assert route == "synthesizer"
    assert state_refine["iteration_count"] == 1

    # 2. High score, iteration 0 -> Audit (pass)
    state_pass = create_initial_agent_state(project_id="p1", max_iterations=2)
    state_pass["current_critic_score"] = 82.0
    state_pass["should_refine"] = False
    state_pass["iteration_count"] = 0
    route_pass = should_refine_or_finalize(state_pass)
    assert route_pass == "auditor"

    # 3. Low score, iteration 2 (max reached) -> Audit (bounded termination)
    state_max = create_initial_agent_state(project_id="p1", max_iterations=2)
    state_max["current_critic_score"] = 65.0
    state_max["should_refine"] = True
    state_max["iteration_count"] = 2
    route_max = should_refine_or_finalize(state_max)
    assert route_max == "auditor"


@pytest.mark.asyncio
async def test_scholar_agent_orchestrator_end_to_end(in_memory_db):
    orchestrator = ScholarAgentOrchestrator(
        llm_client=None,
        db_session=in_memory_db,
    )

    # Mock search tool to return deterministic candidates
    orchestrator.discovery_agent.search_tool.search = MagicMock(
        return_value=[
            AcademicPaperCandidate(
                paper_id="cand_1",
                title="Deep Residual Learning for Image Recognition",
                authors=["He et al."],
                year=2016,
                doi="10.1109/CVPR.2016.90",
                abstract="Deep residual networks introduce skip connections to address vanishing gradients.",
                source="openalex",
            )
        ]
    )
    orchestrator.discovery_agent.citation_tool.traverse_1hop = MagicMock(return_value=[])

    # Mock OA resolver to prevent real network calls during unit test
    from backend.agents.tools.oa_resolver import OAResolutionResult
    orchestrator.ingestion_agent.oa_resolver.resolve_paper = MagicMock(
        return_value=OAResolutionResult(
            doi="10.1109/CVPR.2016.90",
            is_oa=False,
            pdf_url=None,
            pdf_bytes=None,
            abstract_fallback={"abstract": "Deep residual networks introduce skip connections to address vanishing gradients."},
            source="mock_oa",
        )
    )


    final_state = await orchestrator.run(
        project_id="proj_e2e_1",
        user_id="test_user",
        title="Deep ResNet Survey",
        research_question="How do residual connections facilitate deep neural network optimization?",
        max_papers=3,
        max_iterations=2,
        sync_to_db=True,
    )

    assert final_state["status"] == "completed"
    assert final_state["final_report"] is not None
    assert len(final_state["evidence_matrix"]) >= 1
    assert len(final_state["draft_thematic_sections"]) >= 1
    assert final_state["audit_report"] is not None

    # Verify DB persistence
    report_db = in_memory_db.query(ResearchReportModel).filter(ResearchReportModel.project_id == "proj_e2e_1").first()
    assert report_db is not None
    assert report_db.title == "Deep ResNet Survey"

