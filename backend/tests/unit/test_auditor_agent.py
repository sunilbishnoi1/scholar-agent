"""
Unit tests for FactCheckerEngine & DeterministicCitationAuditorAgent (Auditor Agent).
"""

from __future__ import annotations

import pytest

from backend.agents.core.auditor import DeterministicCitationAuditorAgent
from backend.agents.schemas import (
    CitationAuditReport,
    NLIVerdict,
    PropositionVerification,
    ThematicSection,
    ThematicSynthesisDraft,
)
from backend.agents.state import create_initial_agent_state
from backend.agents.tools.fact_checker import AtomicProposition, FactCheckerEngine


def test_fact_checker_anchor_parsing_and_extraction():
    engine = FactCheckerEngine(llm_client=None)

    # 1. Parse anchor tag
    p1, s1 = engine.parse_anchor_tag("ref_1#sec_3")
    assert p1 == "ref_1"
    assert s1 == "sec_3"

    p2, s2 = engine.parse_anchor_tag("[ref_2]")
    assert p2 == "ref_2"
    assert s2 is None

    # 2. Extract atomic propositions
    text = (
        "FlashAttention reduces memory accesses through tiling [ref_1#sec_methods]. "
        "It achieves a 3x speedup over standard attention [ref_1#sec_results]. "
        "Unrelated sentence without citations."
    )
    props = engine.extract_atomic_propositions(text, theme_id="theme_attn")
    assert len(props) == 2
    assert props[0].paper_id == "ref_1"
    assert props[0].section_anchor == "sec_methods"
    assert props[1].section_anchor == "sec_results"


@pytest.mark.asyncio
async def test_fact_checker_audit_and_prose_cleaning():
    engine = FactCheckerEngine(llm_client=None)

    draft = ThematicSynthesisDraft(
        executive_summary="Review summary [ref_1].",
        thematic_sections=[
            ThematicSection(
                theme_id="t1",
                title="Attention Mechanisms",
                synthesis_prose="FlashAttention uses IO-aware tiling [ref_1#sec_methods]. Made-up claim [ref_hallucinated#sec_1].",
                cited_paper_ids=["ref_1", "ref_hallucinated"],
            )
        ],
        conflicting_findings_and_debates=[],
        actionable_research_gaps=[],
        methodology_overview={
            "distribution": {"Empirical": 1},
            "dominant_approach": "Empirical",
            "trend_description": "Scaling",
        },
    )

    chunks_map = {
        "ref_1": [
            {
                "chunk_id": "chunk_1",
                "paper_id": "ref_1",
                "anchor_tag": "[ref_1#sec_methods]",
                "section_anchor": "sec_methods",
                "content": "FlashAttention uses IO-aware tiling to minimize HBM memory transfers.",
            }
        ]
    }

    report = await engine.audit_thematic_draft(
        draft=draft,
        paper_chunks_map=chunks_map,
        known_paper_ids={"ref_1"},
    )

    assert report.total_propositions == 2
    assert "ref_hallucinated#sec_1" in report.hallucinated_anchors

    # Test prose cleaning
    cleaned = engine.canonicalize_and_clean_prose(draft.thematic_sections[0].synthesis_prose, report)
    assert "[ref_hallucinated#sec_1]" not in cleaned
    assert "[ref_1#sec_methods]" in cleaned


@pytest.mark.asyncio
async def test_auditor_agent_run():
    agent = DeterministicCitationAuditorAgent(llm_client=None)

    state = create_initial_agent_state(
        project_id="proj_audit",
        research_question="Question?",
    )
    state["parsed_papers"] = {
        "ref_1": {
            "paper_id": "ref_1",
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention",
            "doi": "10.1234/fa",
            "sections": [
                {"heading": "Methods", "content": "IO-aware tiling algorithm.", "anchor_tag": "[ref_1#sec_methods]"}
            ],
            "is_full_text": True,
        }
    }
    state["draft_thematic_sections"] = [
        {
            "theme_id": "t1",
            "title": "Efficiency",
            "synthesis_prose": "IO-aware tiling algorithm [ref_1#sec_methods].",
            "key_takeaways": [],
            "cited_paper_ids": ["ref_1"],
        }
    ]

    new_state = await agent.run(state)

    assert new_state["audit_report"] is not None
    assert new_state["audit_precision_score"] >= 80.0
    assert len(new_state["bibliography"]) == 1
    assert new_state["bibliography"][0]["paper_id"] == "ref_1"

