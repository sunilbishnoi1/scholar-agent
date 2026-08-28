"""
Unit tests for AdversarialCriticAgent (Critic Agent).
"""

from __future__ import annotations

import pytest

from backend.agents.core.critic import AdversarialCriticAgent
from backend.agents.schemas import CriticEvaluation
from backend.agents.state import create_initial_agent_state


@pytest.mark.asyncio
async def test_critic_agent_scoring_thresholds():
    agent = AdversarialCriticAgent(llm_client=None, passing_threshold=75.0)

    # Test initial iteration (iteration=0) -> typically triggers refinement in fallback
    state_iter0 = create_initial_agent_state(
        project_id="proj_crit",
        research_question="Question?",
    )
    state_iter0["thematic_sections"] = [
        {"title": "Section 1", "synthesis_prose": "Prose without citation anchors"}
    ]
    state_iter0["iteration_count"] = 0

    res_iter0 = await agent.run(state_iter0)
    assert res_iter0["current_critic_score"] < 75.0
    assert res_iter0["should_refine"] is True
    assert len(res_iter0["refinement_guidance"]) > 0

    # Test refined iteration (iteration=1) with anchors -> passes threshold
    state_iter1 = create_initial_agent_state(
        project_id="proj_crit",
        research_question="Question?",
    )
    state_iter1["thematic_sections"] = [
        {"title": "Section 1", "synthesis_prose": "Prose with anchor [ref_1#sec_1]."},
        {"title": "Section 2", "synthesis_prose": "Prose with anchor [ref_2#sec_2]."},
    ]
    state_iter1["iteration_count"] = 1

    res_iter1 = await agent.run(state_iter1)
    assert res_iter1["current_critic_score"] >= 75.0
    assert res_iter1["should_refine"] is False

