"""
Unit tests for WorkingMemoryBlackboard.
Tests goal stack, artifact mutation, state synchronization, and DB persistence.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.blackboard import WorkingMemoryBlackboard
from backend.agents.schemas import (
    CitationAuditReport,
    ConflictingDebate,
    CriticDimensionScore,
    CriticEvaluation,
    EvidenceMatrixRow,
    MethodologyDistribution,
    NLIVerdict,
    PropositionVerification,
    ResearchGapItem,
    ResearchReport,
    ThematicSection,
)
from backend.agents.state import AgentType, GoalStatus
from backend.models.database import Base, EvidenceMatrixEntry, PaperCache, ResearchGapModel, ResearchReportModel


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_blackboard_goal_stack():
    bb = WorkingMemoryBlackboard(project_id="proj_1", title="Test Literature Review")
    
    # Push goal
    g1 = bb.push_goal(
        goal_id="g_disc",
        name="Discovery",
        description="Search papers",
        target_agent=AgentType.DISCOVERY,
        priority=1,
    )
    assert len(bb.goal_stack) == 1
    assert bb.goal_stack[0]["status"] == GoalStatus.PENDING

    # Update goal
    res = bb.update_goal_status("g_disc", GoalStatus.COMPLETED)
    assert res is True
    assert bb.goal_stack[0]["status"] == GoalStatus.COMPLETED
    assert bb.goal_stack[0]["completed_at"] is not None


def test_blackboard_artifact_mutations():
    bb = WorkingMemoryBlackboard(project_id="proj_1", title="Test Review", research_question="What is X?")
    
    # 1. Add paper
    bb.add_parsed_paper({
        "paper_id": "ref_1",
        "title": "Attention Is All You Need",
        "doi": "10.1234/test.doi",
        "is_full_text": True,
    })
    assert "ref_1" in bb.parsed_papers
    assert bb.parsed_papers["ref_1"]["is_full_text"] is True

    # 2. Evidence Matrix
    row = EvidenceMatrixRow(
        paper_id="ref_1",
        title="Attention Is All You Need",
        methodology="Transformer architecture",
        benchmark_dataset="WMT 2014 En-De",
        primary_metric="BLEU 28.4",
        primary_limitation="Quadratic context scaling",
        is_full_text=True,
    )
    bb.set_evidence_matrix([row])
    assert len(bb.evidence_matrix) == 1
    assert bb.evidence_matrix[0].paper_id == "ref_1"

    # 3. Thematic Synthesis
    sec = ThematicSection(
        theme_id="theme_1",
        title="Transformer Scaling",
        synthesis_prose="Self-attention mechanisms provide strong inductive biases [ref_1#sec_3].",
        key_takeaways=["Scaling improves generalization"],
        cited_paper_ids=["ref_1"],
    )
    deb = ConflictingDebate(
        topic="Dense vs Sparse",
        perspective_a="Dense attention scales better [ref_1].",
        perspective_b="Sparse attention saves memory.",
        critical_evaluation="Trade-off between compute and accuracy.",
    )
    gap = ResearchGapItem(
        gap_id="GAP-1",
        description="Quadratic complexity in long documents",
        importance="high",
        recommended_methodology="Evaluate linear attention approximations.",
        grounding_paper_ids=["ref_1"],
    )
    bb.set_thematic_synthesis(
        executive_summary="Summary of test",
        sections=[sec],
        debates=[deb],
        gaps=[gap],
    )
    assert len(bb.draft_thematic_sections) == 1
    assert len(bb.debates) == 1
    assert len(bb.research_gaps) == 1

    # 4. Critic Feedback
    eval_item = CriticEvaluation(
        overall_score=82.5,
        dimension_scores=[
            CriticDimensionScore(dimension="Statistical Rigor", score=85.0, justification="Good"),
            CriticDimensionScore(dimension="Generalizability", score=80.0, justification="Adequate"),
        ],

        strengths=["Good synthesis"],
        weaknesses=[],
        refinement_guidance=[],
        should_refine=False,
    )
    bb.add_critic_evaluation(eval_item)
    assert len(bb.critic_feedback) == 1

    # 5. Audit Report
    prop = PropositionVerification(
        proposition="Self-attention mechanisms provide strong inductive biases",
        citation_anchor="ref_1#sec_3",
        paper_id="ref_1",
        verdict=NLIVerdict.ENTAILMENT,
        confidence=0.95,
        reasoning="Directly supported by Section 3.",
    )
    audit = CitationAuditReport(
        total_propositions=1,
        entailed_count=1,
        neutral_count=0,
        contradiction_count=0,
        precision_score=100.0,
        verifications=[prop],
        hallucinated_anchors=[],
        audit_passed=True,
    )
    bb.set_audit_report(audit)
    assert bb.audit_report is not None
    assert bb.audit_report.precision_score == 100.0


def test_blackboard_state_roundtrip():
    bb = WorkingMemoryBlackboard(project_id="proj_1", title="Test Roundtrip", research_question="Question?")
    bb.add_parsed_paper({"paper_id": "ref_1", "title": "Paper 1", "is_full_text": True})
    bb.executive_summary = "Executive Summary"
    
    agent_state = bb.to_agent_state()
    assert agent_state["project_id"] == "proj_1"
    assert agent_state["total_candidates_found"] == 1
    assert "ref_1" in agent_state["parsed_papers"]

    bb2 = WorkingMemoryBlackboard(project_id="proj_1")
    bb2.update_from_agent_state(agent_state)
    assert "ref_1" in bb2.parsed_papers
    assert bb2.executive_summary == "Executive Summary"


def test_blackboard_db_sync_and_load(in_memory_db):
    bb = WorkingMemoryBlackboard(project_id="proj_db_1", title="DB Synced Review", research_question="DB Q")
    bb.add_parsed_paper({
        "paper_id": "ref_1",
        "doi": "10.1000/182",
        "title": "DB Synced Paper",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "abstract": "Abstract text",
        "full_text_markdown": "# Title\n\nContent",
        "is_full_text": True,
    })
    row = EvidenceMatrixRow(
        paper_id="ref_1",
        title="DB Synced Paper",
        methodology="Test Method",
        benchmark_dataset="Test Set",
        primary_metric="Acc 95%",
        primary_limitation="None",
        is_full_text=True,
    )
    bb.set_evidence_matrix([row])
    gap = ResearchGapItem(
        gap_id="GAP-01",
        description="DB Gap",
        importance="medium",
        recommended_methodology="Test Method",
        grounding_paper_ids=["ref_1"],
    )
    bb.set_thematic_synthesis(
        executive_summary="DB Executive Summary",
        sections=[ThematicSection(theme_id="t1", title="Theme 1", synthesis_prose="Text [ref_1]", cited_paper_ids=["ref_1"])],
        debates=[],
        gaps=[gap],
    )

    # Sync
    bb.sync_to_database(in_memory_db)

    # Verify rows in DB
    report_db = in_memory_db.query(ResearchReportModel).filter(ResearchReportModel.project_id == "proj_db_1").first()
    assert report_db is not None
    assert report_db.title == "DB Synced Review"
    assert report_db.executive_summary == "DB Executive Summary"

    matrix_db = in_memory_db.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == "proj_db_1").all()
    assert len(matrix_db) == 1
    assert matrix_db[0].paper_id == "ref_1"

    gaps_db = in_memory_db.query(ResearchGapModel).filter(ResearchGapModel.project_id == "proj_db_1").all()
    assert len(gaps_db) == 1
    assert gaps_db[0].gap_id == "GAP-01"

    cache_db = in_memory_db.query(PaperCache).filter(PaperCache.doi == "10.1000/182").first()
    assert cache_db is not None
    assert cache_db.title == "DB Synced Paper"

    # Load into new blackboard
    bb_loaded = WorkingMemoryBlackboard(project_id="proj_db_1")
    loaded_ok = bb_loaded.load_from_database(in_memory_db, project_id="proj_db_1")
    assert loaded_ok is True
    assert bb_loaded.title == "DB Synced Review"
    assert len(bb_loaded.evidence_matrix) == 1
    assert len(bb_loaded.research_gaps) == 1

