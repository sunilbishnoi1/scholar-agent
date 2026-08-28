"""
Unit tests for AutonomousLiteratureExplorer (Discovery Agent).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.core.discovery import AutonomousLiteratureExplorer
from backend.agents.llm import MockLLMClient
from backend.agents.schemas import AcademicPaperCandidate, SearchQueryPlan
from backend.agents.state import create_initial_agent_state


@pytest.mark.asyncio
async def test_discovery_agent_fallback_query_plan():
    agent = AutonomousLiteratureExplorer(llm_client=None)
    plan = agent.formulate_query_plan(
        research_question="How do diffusion models improve image generation?",
        title="Diffusion Review",
        keywords=["diffusion models", "image generation"],
    )
    assert isinstance(plan, SearchQueryPlan) or type(plan).__name__ == "SearchQueryPlan"
    assert len(plan.primary_queries) > 0
    assert "diffusion models" in plan.primary_queries[0]


@pytest.mark.asyncio
async def test_discovery_agent_search_and_snowball():
    mock_search = MagicMock()
    mock_search.search.return_value = [
        AcademicPaperCandidate(
            paper_id="cand_1",
            title="Denoising Diffusion Probabilistic Models",
            doi="10.5555/ddpm",
            citation_count=100,
            source="openalex",
        ),
        AcademicPaperCandidate(
            paper_id="cand_2",
            title="High-Resolution Image Synthesis with Latent Diffusion Models",
            doi="10.5555/ldm",
            citation_count=200,
            source="semanticscholar",
        ),
    ]

    mock_citation = MagicMock()
    mock_citation.traverse_1hop.return_value = [
        AcademicPaperCandidate(
            paper_id="cand_3",
            title="Classifier-Free Diffusion Guidance",
            doi="10.5555/cfg",
            citation_count=50,
            source="openalex",
        )
    ]

    agent = AutonomousLiteratureExplorer(
        llm_client=None,
        search_tool=mock_search,
        citation_tool=mock_citation,
    )

    state = create_initial_agent_state(
        project_id="proj_disc",
        research_question="Diffusion models in computer vision",
        max_papers=5,
    )

    new_state = await agent.run(state)

    assert new_state["total_candidates_found"] == 3
    assert len(new_state["candidate_papers"]) == 3
    assert new_state["candidate_papers"][0]["paper_id"] == "ref_1"
    assert new_state["candidate_papers"][0]["relevance_score"] is not None
    assert new_state["candidate_papers"][0]["relevance_score"] > 0.5
    assert new_state["candidate_papers"][1]["paper_id"] == "ref_2"
    assert new_state["candidate_papers"][2]["paper_id"] == "ref_3"
    assert len(new_state["papers"]) == 3
    assert new_state["papers"][0]["relevance_score"] > 0.5

