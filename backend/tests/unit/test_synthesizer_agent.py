"""
Unit tests for ThematicSynthesizerAgent (Synthesizer Agent) & SectionAwareContextPacker.
"""

from __future__ import annotations

import pytest

from backend.agents.core.synthesizer import SectionAwareContextPacker, ThematicSynthesizerAgent
from backend.agents.schemas import EvidenceMatrixRow, ThematicSynthesisDraft
from backend.agents.state import create_initial_agent_state


def test_section_aware_context_packer():
    papers = [
        {
            "paper_id": "ref_1",
            "title": "Scaling Law Paper",
            "year": 2020,
            "sections": [
                {"heading": "Introduction", "content": "Intro text", "section_type": "introduction"},
                {"heading": "Results", "content": "Loss scales as power law.", "section_type": "results"},
                {"heading": "Methodology", "content": "Train models from 1M to 10B parameters.", "section_type": "methodology"},
            ],
        }
    ]
    matrix = [
        EvidenceMatrixRow(
            paper_id="ref_1",
            title="Scaling Law Paper",
            methodology="Empirical scaling",
            benchmark_dataset="WebText",
            primary_metric="Loss",
            primary_limitation="Compute budget",
            is_full_text=True,
        )
    ]

    packed = SectionAwareContextPacker.pack_corpus(papers=papers, evidence_matrix=matrix, max_chars=5000)
    assert "EVIDENCE COMPARISON MATRIX OVERVIEW" in packed
    assert "TYPE=RESULTS" in packed
    assert "Loss scales as power law" in packed
    # Results should precede Introduction due to priority
    res_idx = packed.find("TYPE=RESULTS")
    intro_idx = packed.find("TYPE=INTRODUCTION")
    assert res_idx < intro_idx


@pytest.mark.asyncio
async def test_synthesizer_agent_run_fallback():
    agent = ThematicSynthesizerAgent(llm_client=None)

    state = create_initial_agent_state(
        project_id="proj_synth",
        research_question="What are modern neural scaling laws?",
        title="Scaling Laws Review",
    )
    state["parsed_papers"] = {
        "ref_1": {
            "paper_id": "ref_1",
            "title": "Scaling Laws for Neural Language Models",
            "year": 2020,
            "sections": [
                {"heading": "Results", "content": "Power-law scaling observed across cross-entropy loss.", "section_type": "results"}
            ],
            "is_full_text": True,
        }
    }
    state["evidence_matrix"] = [
        {
            "paper_id": "ref_1",
            "title": "Scaling Laws for Neural Language Models",
            "methodology": "Empirical parameter scaling",
            "benchmark_dataset": "WebText",
            "primary_metric": "Cross-Entropy Loss",
            "primary_limitation": "Fixed dataset distribution",
            "is_full_text": True,
        }
    ]

    new_state = await agent.run(state)

    assert "draft_thematic_sections" in new_state
    assert len(new_state["draft_thematic_sections"]) >= 1
    assert len(new_state["conflicting_debates"]) >= 1
    assert len(new_state["research_gaps"]) >= 1
    assert new_state["executive_summary"] != ""

